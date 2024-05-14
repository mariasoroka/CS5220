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
    // The number of triangles included in this node.
    int triCount;

    // Index of the left child. Negative if this is a leaf node.
    int left;
    // Index of the right child. Negative if this is a leaf node.
    int right;

    // Triangle geometry bounds
    float3 tMin, tMax;

    // Triangle centroid bounds. Centroid is of the AABB, not the tri.
    float3 cMin, cMax;

    // Constructs an empty node that's ready to store _triCount triangles
    Node(int _triStart, int _triCount)
      : triStart(_triStart), triCount(_triCount),
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

template <const int N>
__global__ __launch_bounds__(N)
void buildTree(DeviceUniforms u, int nodeId) {
    Node& node = u.nodes[nodeId];
    int triCount = node.triCount;
    if (triCount <= LEAF_THRESHOLD) {
        return;
    }

    // Identify binning axis
    float axisLength, splitPos;
    int axis = getLongestAxis(node.cMin, node.cMax, axisLength, splitPos);

    // Each thread is assigned to exactly one split, and multiple threads can be
    // assigned to the same split. We ensure that all threads for split #i lie
    // within the same warp. This lets us perform warp reductions over the bin
    // bounds at the end, and then let one representative thread can perform a
    // threadblock reduction to find the split with the best SAH.
    constexpr int threadsPerSplit = N / NUM_SPLITS;

    // Identify split position along the selected axis.
    int threadId = N*blockIdx.x + threadIdx.x;

    // This should be optimized down to a shift & mask, not integer division,
    // since the divisor is constexpr
    int splitId = threadId / threadsPerSplit;
    int innerId = threadId % threadsPerSplit;

    constexpr float rcp = 1.0f / (NUM_SPLITS + 1.0f);
    splitPos += axisLength*rcp*(splitId + 1.0f);

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
    for (int id = innerId; id < triCount; id += threadsPerSplit) { \
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
    if (threadId == 0) {
        float minCost = sahs[0];
        int minSplit = 0;
        for (int curSplit=1; curSplit<NUM_SPLITS; ++curSplit) {
            float curCost = sahs[curSplit];
            if (curCost < minCost) {
                minCost = curCost;
                minSplit = curSplit;
            }
        }
        selectedSplit = minSplit;
    }

    // All threads use least-cost split to partition the triangles
    __syncthreads();
#if 0
    if (splitId == selectedSplit && innerId == 0) {
        printf("Split: %i, axis %i, pos %f\n", 
            selectedSplit, axis, splitPos);
        printf("  L: [%f, %f, %f] to [%f, %f, %f]\n", 
            ltMin.x, ltMin.y, ltMin.z, ltMax.x, ltMax.y, ltMax.z);
        printf("  R: [%f, %f, %f] to [%f, %f, %f]\n", 
            rtMin.x, rtMin.y, rtMin.z, rtMax.x, rtMax.y, rtMax.z);
    }
#endif

    // TODO
    return;

    // TODO when setting up child nodes, use 2i+1 and 2i+2

    // Spawn children using dynamic parallelism
    if (threadIdx.x == 0) {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        buildTree<N><<<1, WARP_WIDTH, 0, stream>>>(u, -1);
        buildTree<N><<<1, WARP_WIDTH, 0, stream>>>(u, -1);
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