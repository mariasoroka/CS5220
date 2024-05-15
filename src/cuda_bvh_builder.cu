#include "cuda_bvh_builder.h"

// Return the number of p-sized blocks needed to cover N
template<typename T>
constexpr T NP2(T n, T p) {
    return ((n) + (p-1)) / p;
}

// If a node contains under this number of triangles, it should become a leaf node.
#define LEAF_THRESHOLD 4

// The SIMD width on the A100
#define WARP_WIDTH 32

// The optimal number of threads per threadblock on the A100
#define OPTIMAL_WIDTH 64

// The number of splits used in SAH evaluation.
// WARP_WIDTH should be a multiple of this
#define NUM_SPLITS   16


// Takes the min of the first 3 elements of A and B.
// Fourth element is unchanged.
__device__ inline float3 fmin3(float3 a, float3 b) {
    a.x = fminf(a.x, b.x);
    a.y = fminf(a.y, b.y);
    a.z = fminf(a.z, b.z);
    return a;
}

// Takes the max of the first 3 elements of A and B.
// Fourth element is unchanged.
__device__ inline float3 fmax3(float3 a, float3 b) {
    a.x = fmaxf(a.x, b.x);
    a.y = fmaxf(a.y, b.y);
    a.z = fmaxf(a.z, b.z);
    return a;
}

__device__ inline void atomic_fmin(float* dst, float src) {
    // Floats are monotonic wrt sign-magnitude. Suppose src >= 0. 
    //  - If dst >= 0, then signed atomicMin will work normally.
    //  - If dst < 0, then it will be interpreted as some negative int, and
    //    signed atomicMin will correctly choose it as the min.
    //
    // Suppose src < 0.
    //  - If dst >= 0, then unsigned atomicMax will select src, and thus
    //    actually select the min.
    //  - If dst < 0, then unsigned atomicMax will select the one with the
    //    greater magnitude, which is the min since both are negative.
    signbit(src) 
        ? atomicMax((unsigned int*)dst, __float_as_uint(src))
        : atomicMin((int*)dst, __float_as_int(src));
}

__device__ inline void atomic_fmax(float* dst, float src) {
    // See reasoning for atomic_fmax.
    signbit(src)
        ? atomicMin((unsigned int*)dst, __float_as_uint(src))
        : atomicMax((int*)dst, __float_as_int(src));
}

// Convenience wrappers for atomic elementwise operations on float3. Each
// component is updated atomically, but the components may not be updated
// simultaneously.
__device__ inline void atomic_fmin_ew3(float3* dst, float3 src) {
    atomic_fmin(&dst->x, src.x);
    atomic_fmin(&dst->y, src.y);
    atomic_fmin(&dst->z, src.z);
}

__device__ inline void atomic_fmax_ew3(float3* dst, float3 src) {
    atomic_fmax(&dst->x, src.x);
    atomic_fmax(&dst->y, src.y);
    atomic_fmax(&dst->z, src.z);
}


// A BVH node (either an inner node or a leaf node.)
//
// NOTE: The split axis can be computed implicitly by taking the maximum
// dimension from the triangle centroid bounds.
struct alignas(64) Node {
    // The index of the first triangle in this node.
    int triStart;
    // One past the index of the last triangle in this node.
    int triEnd;

    // Index of the left child. Negative if this is a leaf node.
    int left;
    // Index of the right child. Negative if this is a leaf node.
    int right;

    // Triangle geometry bounds
    float3 tMin, tMax;

    // Triangle centroid bounds. Centroid is of the AABB, not the tri.
    float3 cMin, cMax;

    // Constructs an empty node that's ready to store _triCount triangles
    Node(int _triStart, int _triEnd)
      : triStart(_triStart), triEnd(_triEnd),
        left(-1), right(-1),
        tMin { +INFINITY, +INFINITY, +INFINITY },
        tMax { -INFINITY, -INFINITY, -INFINITY },
        cMin { +INFINITY, +INFINITY, +INFINITY },
        cMax { -INFINITY, -INFINITY, -INFINITY } 
    {}
};

static_assert(sizeof(Node) == 64);

struct DeviceUniforms {
    int numTris;
    triangle* __restrict__ tris;

    // We use float4 rather than a float3 because you can load a float4 with one
    // instruction. If every thread in a warp loads a consecutive float4, all
    // 128 bytes can be loaded in a single memory transaction. So with
    // mins/indices in one buffer and maxs in another, we can load triangle data
    // across a warp with two total memory transactions.

    // [minX minY minZ (bitcast idx)]
    float4* __restrict__ triMinsIds;

    // [maxX maxY maxZ (undefined)]
    float4* __restrict__ triMaxs;

    // Auxiliary arrays used as scratch space during partitioning
    float4* __restrict__ triMinsIdsAux;
    float4* __restrict__ triMaxsAux;

    // Node array. Equal to 2N-1 where N is the number of triangles.
    Node* __restrict__ nodes;
};

class CudaBVH {
public:
    CudaBVH(DeviceUniforms u)
      : _u(u) {}

    ~CudaBVH() {
        cudaFree(_u.tris);
        cudaFree(_u.triMinsIds);
        cudaFree(_u.triMaxs);
    }
private:
    DeviceUniforms _u;
};

__global__ void setupTris(DeviceUniforms u) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= u.numTris) return;

    triangle& tri = u.tris[id];
    float3 p1 = {float(tri.p1.x), float(tri.p1.y), float(tri.p1.z)};
    float3 p2 = {float(tri.p2.x), float(tri.p2.y), float(tri.p2.z)};
    float3 p3 = {float(tri.p3.x), float(tri.p3.y), float(tri.p3.z)};

    float4 minId = {
        fminf(p1.x, fminf(p2.x, p3.x)),
        fminf(p1.y, fminf(p2.y, p3.y)),
        fminf(p1.z, fminf(p2.z, p3.z)),
        __int_as_float(id)
    };
    u.triMinsIds[id] = minId;

    float4 max = {
        fmaxf(p1.x, fmaxf(p2.x, p3.x)),
        fmaxf(p1.y, fmaxf(p2.y, p3.y)),
        fmaxf(p1.z, fmaxf(p2.z, p3.z)),
        0.0f
    };
    u.triMaxs[id] = max;

    float3 centroid = {
        0.5f * (minId.x + max.x),
        0.5f * (minId.y + max.y),
        0.5f * (minId.z + max.z)
    };

    // This is a lot of atomics. Hopefully this does not become a bottleneck.
    Node& node = u.nodes[0];
    atomic_fmin_ew3(&node.tMin, {minId.x, minId.y, minId.z});
    atomic_fmax_ew3(&node.tMax, {max.x, max.y, max.z});
    atomic_fmin_ew3(&node.cMin, centroid);
    atomic_fmax_ew3(&node.cMax, centroid);
}

// Returns the index of the longest axis in the AABB defined by min and max.
// Writes out the length of that axis to length.
__device__ inline int getLongestAxis(float3 min, float3 max, float& length, float& low) {
    int axis = 0;
    length = max.x - min.x;
    low = min.x;

    float lenY = max.y - min.y;
    if (lenY > length) {
        length = lenY;
        low = min.y;
        axis = 1;
    }

    float lenZ = max.z - min.z;
    if (lenZ > length) {
        length = lenZ;
        low = min.z;
        axis = 2;
    }

    return axis;
}

// Returns half the surface area of the AABB defined by min, max.
__device__ inline float halfSurfaceArea(float3 min, float3 max) {
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return (dx*dy) + (dy*dz) + (dz*dx);
}

// dynamic parallelism: why write our own load balancing code when we could just use the GPU's firmware?
template<const int N>
__device__ void partitionTris(const DeviceUniforms& u, Node& leftChild, Node& rightChild, int start, int end, int axis, float pivot) {
    constexpr int numWarps = N / WARP_WIDTH;

    int myLeft = 0;
    int myRight = 0;
    for (int id = start+threadIdx.x; id < end; id += N) {
        float4 minId4 = u.triMinsIds[id];
        float4 max4 = u.triMaxs[id];

        float center;
        if (axis == 2) {
            center = 0.5f * (minId4.z + max4.z);
        } else if (axis == 1) {
            center = 0.5f * (minId4.y + max4.y);
        } else {
            center = 0.5f * (minId4.x + max4.x);
        }

        if (center < pivot) {
            myLeft += 1;
        } else {
            myRight += 1;
        }
        u.triMinsIdsAux[id] = minId4;
        u.triMaxsAux[id] = max4;
    }

    __shared__ int leftCounts[N];
    __shared__ int rightCounts[N];
    leftCounts[threadIdx.x] = myLeft;
    rightCounts[threadIdx.x] = myRight;

    __syncthreads();
    if (threadIdx.x == 0) {
        int cur = myLeft;
        for (int ii = 1; ii < N; ++ii) {
            cur += leftCounts[ii];
            leftCounts[ii] = cur;
        }
    } else if (threadIdx.x == N/2) {
        int cur = end-start;
        for (int ii = 1; ii < N; ++ii) {
            cur -= rightCounts[ii];
            rightCounts[ii] = cur;
        }
    }

    __syncthreads();
    myLeft = leftCounts[threadIdx.x];
    myRight = rightCounts[threadIdx.x];

    float3 lcMin, lcMax, rcMin, rcMax;
    lcMin = rcMin = { +INFINITY, +INFINITY, +INFINITY };
    lcMax = rcMax = { -INFINITY, -INFINITY, -INFINITY };
    for (int id = start+threadIdx.x; id < end; id += N) {
        float4 minId4 = u.triMinsIdsAux[id];
        float4 max4 = u.triMaxsAux[id];
        float3 centroid = {
            0.5f * (minId4.x + max4.x),
            0.5f * (minId4.y + max4.y),
            0.5f * (minId4.z + max4.z)
        };
        float center;
        if (axis == 2) {
            center = 0.5f * (minId4.z + max4.z);
        } else if (axis == 1) {
            center = 0.5f * (minId4.y + max4.y);
        } else {
            center = 0.5f * (minId4.x + max4.x);
        }
        if (center < pivot) {
            lcMin = fmin3(lcMin, centroid);
            lcMax = fmax3(lcMax, centroid);
            u.triMinsIds[start+myLeft] = minId4;
            u.triMaxs[start+myLeft] = max4;
            myLeft += 1;
        } else {
            rcMin = fmin3(rcMin, centroid);
            rcMax = fmax3(rcMax, centroid);
            u.triMinsIds[start+myRight] = minId4;
            u.triMaxs[start+myRight] = max4;
            myRight -= 1;
        }
    }

    // Reduce AABBs across warp
#define REDUCE(f, x) x = f (x, __shfl_up_sync(~0, x, buddy, N))
    for (int buddy = 1; buddy != N; buddy *= 2) {
        REDUCE(fminf, lcMin.x); REDUCE(fminf, lcMin.y); REDUCE(fminf, lcMin.z);
        REDUCE(fminf, rcMin.x); REDUCE(fminf, rcMin.y); REDUCE(fminf, rcMin.z);
        REDUCE(fmaxf, lcMax.x); REDUCE(fmaxf, lcMax.y); REDUCE(fmaxf, lcMax.z);
        REDUCE(fmaxf, rcMax.x); REDUCE(fmaxf, rcMax.y); REDUCE(fmaxf, rcMax.z);
    }
#undef REDUCE

    // Reduce AABBs across threadgroup
    __shared__ float3 lcMins[numWarps];
    __shared__ float3 lcMaxs[numWarps];
    __shared__ float3 rcMins[numWarps];
    __shared__ float3 rcMaxs[numWarps];
    if (threadIdx.x % numWarps == 0) {
        lcMins[threadIdx.x / numWarps] = lcMin;
        lcMaxs[threadIdx.x / numWarps] = lcMax;
        rcMins[threadIdx.x / numWarps] = rcMin;
        rcMaxs[threadIdx.x / numWarps] = rcMax;
    }
    __syncthreads();
    if (threadIdx.x == N-1) {
        for (int ii = 1; ii < numWarps; ++ii) {
            lcMin = fmin3(lcMin, lcMins[ii]);
            lcMax = fmax3(lcMax, lcMaxs[ii]);
            rcMin = fmin3(rcMin, rcMins[ii]);
            rcMax = fmax3(rcMax, rcMaxs[ii]);
        }
        leftChild.triStart = start;
        leftChild.triEnd = start + myLeft; // this is now the num in the left
        leftChild.cMin = lcMin;
        leftChild.cMax = lcMax;
        
        rightChild.triStart = start + myLeft;
        rightChild.triEnd = end;
        rightChild.cMin = rcMin;
        rightChild.cMax = rcMax;
    }
    __syncthreads();
}

template <const int N>
__global__ __launch_bounds__(N)
void buildTree(DeviceUniforms u, int nodeId) {
    Node& node = u.nodes[nodeId];
    int triCount = node.triEnd - node.triStart;
    if (triCount <= LEAF_THRESHOLD) {
        node.left = -1;
        node.right = -1;
        return;
    }

    if (threadIdx.x == 0 && triCount >= 1 << 15) {
        printf("%*s* %i\n", (nodeId+1)/2, "", triCount);
    }

    // Identify binning axis
    float axisLength, axisStart;
    int axis = getLongestAxis(node.cMin, node.cMax, axisLength, axisStart);

    // Each thread is assigned to exactly one split, and multiple threads can be
    // assigned to the same split. We ensure that all threads for split #i lie
    // within the same warp. This lets us perform warp reductions over the bin
    // bounds at the end, and then let one representative thread can perform a
    // threadblock reduction to find the split with the best SAH.
    constexpr int threadsPerSplit = N / NUM_SPLITS;

    // Identify split position along the selected axis. This should be optimized
    // down to a shift & mask, not division, since the divisor is constexpr
    int splitId = threadIdx.x / threadsPerSplit;
    int innerId = threadIdx.x % threadsPerSplit;

    constexpr float rcp = 1.0f / (NUM_SPLITS + 1.0f);
    float splitPos = axisStart + axisLength*rcp*(splitId + 1.0f);

    // tMin and tMax are the bounds of the *triangles* in this thread's bin.
    // This is what Wald07 refers to as the "bin bounds" (bb).
    // We have two sets of bounds, one for the left bin and one for the right.
    // This comes out to 12 registers.
    float inLeft = 0;
    float3 ltMin, ltMax, rtMin, rtMax;
    ltMin = rtMin = { +INFINITY, +INFINITY, +INFINITY };
    ltMax = rtMax = { -INFINITY, -INFINITY, -INFINITY };
    
    // We pull the axis conditional outside of the loop to avoid rechecking it
    // every loop iteration. GPUs don't have a branch predictor, and repeated
    // predication is still expensive. To avoid code duplication we use a macro.
#define BUILD_OVER(cmp) \
    for (int id = node.triStart + innerId; id < node.triEnd; id += threadsPerSplit) { \
        float3 min; { \
            float4 minId4 = u.triMinsIds[id]; \
            min = {minId4.x, minId4.y, minId4.z}; \
        } \
        float3 max; { \
            float4 max4 = u.triMaxs[id]; \
            max = {max4.x, max4.y, max4.z}; \
        } \
        float center = 0.5f * (max. cmp + min. cmp); \
        bool left = (center < splitPos); \
        min = fmin3(min, left ? ltMin : rtMin); \
        max = fmax3(max, left ? ltMax : rtMax); \
        if (left) { \
            ltMin = min; ltMax = max; \
        } else { \
            rtMin = min; rtMax = max; \
        }\
        inLeft += left ? 1.0f : 0.0f; \
    }

    // Build bins over appropriate axis.
    if (axis == 2) {
        BUILD_OVER(z)
    } else if (axis == 1) {
        BUILD_OVER(y)
    } else {
        BUILD_OVER(x)
    }

    // Reduce over buddy threads. First, every thread grabs from its rightmost
    // buddy, then from its buddy two to the right, then from its buddy four to
    // the right, and so on.
    //
    // With 4 threads per split, the process looks like this:
    // A    B    C    D
    // AB   BC   CD   DA
    // ABCD ABCD ABCD ABCD
#define REDUCE(f, x) x = f (x, __shfl_up_sync(~0, x, buddy, threadsPerSplit))
    for (int buddy = 1; buddy != threadsPerSplit; buddy *= 2) {
        REDUCE(fminf, ltMin.x); REDUCE(fminf, ltMin.y); REDUCE(fminf, ltMin.z);
        REDUCE(fminf, rtMin.x); REDUCE(fminf, rtMin.y); REDUCE(fminf, rtMin.z);

        REDUCE(fmaxf, ltMax.x); REDUCE(fmaxf, ltMax.y); REDUCE(fmaxf, ltMax.z);
        REDUCE(fmaxf, rtMax.x); REDUCE(fmaxf, rtMax.y); REDUCE(fmaxf, rtMax.z);

        inLeft += __shfl_up_sync(~0, inLeft, buddy, threadsPerSplit);
    }
#undef REDUCE
    
    // Compute SAH and write out to smem
    __shared__ float sahs[NUM_SPLITS];
    if (innerId == 0) {
        float inRight = float(triCount) - inLeft;
        sahs[splitId] = inLeft*halfSurfaceArea(ltMin, ltMax) 
                      + inRight*halfSurfaceArea(rtMin, rtMax);
    }

    // Master thread scans for least-cost split
    __syncthreads();
    __shared__ int selectedSplit;
    if (threadIdx.x == 0) {
        float minCost = sahs[0];
        int minSplit = 0;
        for (int curSplit=1; curSplit < NUM_SPLITS; ++curSplit) {
            float curCost = sahs[curSplit];
            if (curCost < minCost) {
                minCost = curCost;
                minSplit = curSplit;
            }
        }
        selectedSplit = minSplit;
    }

    // Build left & right-handed nodes so we can finally forget about
    // the 12 registers used for holding the geometry bounds.
    __syncthreads();
    node.left = 2*nodeId+1;
    node.right = 2*nodeId+2;
    Node& leftChild = u.nodes[2*nodeId+1];
    Node& rightChild = u.nodes[2*nodeId+2];
    if (splitId == selectedSplit && innerId == 0) {
        leftChild.tMin = ltMin;
        leftChild.tMax = ltMax;
        rightChild.tMin = rtMin;
        rightChild.tMax = rtMax;
        // ltMin, ltMax, rtMin, rtMax... you are free
#if 1
        printf("Split: %i, axis %i, pos %f\n", 
            selectedSplit, axis, splitPos);
        printf("  L: [%f, %f, %f] to [%f, %f, %f]\n", 
            ltMin.x, ltMin.y, ltMin.z, ltMax.x, ltMax.y, ltMax.z);
        printf("  R: [%f, %f, %f] to [%f, %f, %f]\n", 
            rtMin.x, rtMin.y, rtMin.z, rtMax.x, rtMax.y, rtMax.z);
#endif
    }

    // Unify split pos based on selected split
    splitPos = axisStart + axisLength*rcp*(selectedSplit + 1.0f);

    // TODO: CRASHES AFTER THIS
    return;
    
    partitionTris<N>(u, leftChild, rightChild, node.triStart, node.triEnd, axis, splitPos);

    // Spawn children using dynamic parallelism
    if (threadIdx.x == 0) {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        buildTree<N><<<1, N, 0, stream>>>(u, node.left);
        buildTree<N><<<1, N, 0, stream>>>(u, node.right);
        cudaStreamDestroy(stream);
    }
}


std::shared_ptr<CudaBVH> build_cuda_bvh(
    triangle* tris, 
    int numTris
) {
    DeviceUniforms u;

    u.numTris = numTris;
    cudaMalloc((triangle**)&u.tris, numTris*sizeof(triangle));
    cudaMalloc((float4**)&u.triMinsIds, numTris*sizeof(float4));
    cudaMalloc((float4**)&u.triMaxs, numTris*sizeof(float4));
    cudaMalloc((float4**)&u.triMinsIdsAux, numTris*sizeof(float4));
    cudaMalloc((float4**)&u.triMaxsAux, numTris*sizeof(float4));
    cudaMalloc((Node**)&u.nodes, (2*numTris-1)*sizeof(Node));

    // Copy triangles into device memory
    cudaMemcpy(
        u.tris,
        tris, 
        numTris*sizeof(triangle), 
        cudaMemcpyHostToDevice);

    // Copy root node into device memory
    Node root(0, numTris);
    cudaMemcpy(
        u.nodes,
        &root,
        sizeof(Node),
        cudaMemcpyHostToDevice);
    
    // Compute triangle AABBs
    setupTris<<<NP2(numTris, OPTIMAL_WIDTH), OPTIMAL_WIDTH>>>(u);

    // Build tree (vertical parallelism only for now)
    buildTree<OPTIMAL_WIDTH><<<1, OPTIMAL_WIDTH>>>(u, 0);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy device root node back onto host for debugging purposes
    cudaMemcpy(
        &root,
        u.nodes,
        sizeof(Node),
        cudaMemcpyDeviceToHost);
    
    // Debug info
    printf("Geometry: [%f, %f, %f] to [%f, %f, %f]\n", 
        root.tMin.x, root.tMin.y, root.tMin.z,
        root.tMax.x, root.tMax.y, root.tMax.z);
    printf("Centroid: [%f, %f, %f] to [%f, %f, %f]\n",
        root.cMin.x, root.cMin.y, root.cMin.z,
        root.cMax.x, root.cMax.y, root.cMax.z);

    return std::make_shared<CudaBVH>(u);
}