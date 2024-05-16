#include "cuda_bvh_builder.h"

#include <cfloat>

// Return the number of p-sized blocks needed to cover N
template<typename T>
constexpr T NP2(T n, T p) {
    return ((n) + (p-1)) / p;
}

// The number of bins used in SAH evaluation.
constexpr int NUM_BINS = 16;
// The number of threads in a warp on the NVIDIA A100.
constexpr int WARP_THREADS = 32;
// The number of streaming multiprocessors on the NVIDIA A100.
constexpr int NUM_SMS = 108;
// The max number of threadgroups you can schedule on one SM on the NVIDIA A100.
constexpr int MAX_BLOCKS_PER_SM = 32;
// The max number of warps that can be resident on one SM on the NVIDIA A100.
constexpr int MAX_WARPS_PER_SM = 64;
// So to maximize utilization, we should launch 32 threadgroups per SM, and each
// one of those threadgroups should contain 2 warps (64 threads)
constexpr int OPT_BLOCKS = NUM_SMS * MAX_BLOCKS_PER_SM;
constexpr int OPT_THREADS = WARP_THREADS * (MAX_WARPS_PER_SM / MAX_BLOCKS_PER_SM);

// The number of items each horizontal queue needs
// This is simply the next power of two greater than OPT_BLOCKS
constexpr int QUEUE_SIZE = 1 << (32 - __builtin_clz(OPT_BLOCKS - 1));

// The number of BVH levels which are built in a horizontal format
constexpr int HORIZONTAL_LEVELS = (32 - __builtin_clz(OPT_BLOCKS+1)) - 1;


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

__device__ inline void atomic_fmin_block(float* dst, float src) {
    signbit(src) 
        ? atomicMax_block((unsigned int*)dst, __float_as_uint(src))
        : atomicMin_block((int*)dst, __float_as_int(src));
}

__device__ inline void atomic_fmax_block(float* dst, float src) {
    signbit(src)
        ? atomicMin_block((unsigned int*)dst, __float_as_uint(src))
        : atomicMax_block((int*)dst, __float_as_int(src));
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

__device__ inline void atomic_fmin_ew3_block(float3* dst, float3 src) {
    atomic_fmin_block(&dst->x, src.x);
    atomic_fmin_block(&dst->y, src.y);
    atomic_fmin_block(&dst->z, src.z);
}

__device__ inline void atomic_fmax_ew3_block(float3* dst, float3 src) {
    atomic_fmax_block(&dst->x, src.x);
    atomic_fmax_block(&dst->y, src.y);
    atomic_fmax_block(&dst->z, src.z);
}

// Returns half the surface area of the AABB defined by min, max.
__device__ inline float halfSurfaceArea(float3 min, float3 max) {
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return (dx*dy) + (dy*dz) + (dz*dx);
}

// A BVH node (either an inner node or a leaf node.)
// Child nodes can be computed implicitly via 2i+1 and 2i+2.
//
// NOTE: The split axis can be computed implicitly by taking the maximum
// dimension from the triangle centroid bounds.
struct Node {
    // The index of the first triangle in this node.
    uint triStart;
    // One past the index of the last triangle in this node.
    uint triEnd;

    // Used during partitioning. Poorly named
    uint allocTriLeft;
    uint allocTriRight;

    // Split ID (packed into the high 4 bits)
    // left counter (lower 28 bits)
    uint splitId;

    // Centroid and triangle AABB bounds.
    float3 cMin, cMax;
    float3 tMin, tMax;

    // Constructs an empty node that stores from _triStart to _triEnd triangles
    constexpr Node(int _triStart, int _triEnd)
      : triStart(_triStart), triEnd(_triEnd),
        allocTriLeft(_triStart),
        allocTriRight(_triEnd),
        splitId(~0),
        cMin { +INFINITY, +INFINITY, +INFINITY },
        cMax { -INFINITY, -INFINITY, -INFINITY },
        tMin { +INFINITY, +INFINITY, +INFINITY },
        tMax { -INFINITY, -INFINITY, -INFINITY }
    {}
    
    __device__ inline uint count() const {
        return triEnd - triStart;
    }

    __device__ inline bool isLeaf() const {
        constexpr int LEAF_THRESHOLD = 4;
        return count() <= LEAF_THRESHOLD;
    }
};


struct TriBin {
    uint count;
    float3 min, max;

    __device__ constexpr TriBin()
      : count(0),
        min { +INFINITY, +INFINITY, +INFINITY },
        max { -INFINITY, -INFINITY, -INFINITY }
    {}
};

struct TriSplitStats {
    uint inLeft, inRight;
    float3 ltMin, ltMax;
    float3 rtMin, rtMax;

    __device__ constexpr TriSplitStats()
      : inLeft(0), inRight(0),
        ltMin { +INFINITY, +INFINITY, +INFINITY },
        ltMax { -INFINITY, -INFINITY, -INFINITY },
        rtMin { +INFINITY, +INFINITY, +INFINITY },
        rtMax { -INFINITY, -INFINITY, -INFINITY }
    {}

    __device__ float heuristic() const {
        return inLeft*halfSurfaceArea(ltMin, ltMax)
            + inRight*halfSurfaceArea(rtMin, rtMax);
    }
};

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
    float4* __restrict__ triMaxsBinsAux;

    // Node array. Equal to 2N-1 where N is the number of triangles.
    Node* __restrict__ nodes;

    // Global bins for the horizontal binning phase.
    // There are QUEUE_SIZE * NUM_BINS of these.
    // Indexed by queue ID, then by bin index.
    TriBin* __restrict__ bins;
};

class CudaBVH {
public:
    CudaBVH(DeviceUniforms u)
      : _u(u) {}

    ~CudaBVH() {
        cudaFree(_u.tris);
        cudaFree(_u.triMinsIds);
        cudaFree(_u.triMaxs);
        cudaFree(_u.triMinsIdsAux);
        cudaFree(_u.triMaxsBinsAux);
        cudaFree(_u.nodes);
        cudaFree(_u.bins);
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

// Each threadblock clears the bins for one queue entry.
__global__ void horizontalClearBins(DeviceUniforms u) {
    uint queueId = blockIdx.x;
    TriBin* bins = &u.bins[queueId*NUM_BINS];
    if (threadIdx.x < NUM_BINS) {
        TriBin& bin = bins[threadIdx.x];
        bin.count = 0;
        bin.min = {+INFINITY, +INFINITY, +INFINITY};
        bin.max = {-INFINITY, -INFINITY, -INFINITY};
    }
}

__global__ void horizontalPrebin(DeviceUniforms u, uint level) {
    // Level 0: All threadblocks process node 0.
    // Level 1: The first half of threadblocks process node 1+0,
    //          the other half process node 1+1.
    // Level 2: 3+0, then 3+1, then 3+2, then 3+3
    // Level 3: 7+0, etc
    uint mask = (1u << level) - 1u;
    uint queueId = mask & blockIdx.x;
    uint nodeId = mask + queueId;
    Node& node = u.nodes[nodeId];
    if (node.isLeaf()) {
        return;
    }

    // The index of this thread within all threads assigned to this node.
    uint localBlockId = blockIdx.x >> level;
    uint localBlocks = gridDim.x >> level;
    uint localThreadId = localBlockId*blockDim.x + threadIdx.x;
    uint localThreads = localBlocks*blockDim.x;

    // Debug stats
    if (localThreadId == 0 && level <= 2) {
        printf("%*s↪ node %u: %i tris, [%.2f,%2.f,%.2f][%.2f,%2.f,%2.f] centroid, [%.2f,%.2f,%.2f][%.2f,%.2f,%.2f] geometry\n", 
            level, "", nodeId, node.count(),
            node.cMin.x, node.cMin.y, node.cMin.z, 
            node.cMax.x, node.cMax.y, node.cMax.z,
            node.tMin.x, node.tMin.y, node.tMin.z, 
            node.tMax.x, node.tMax.y, node.tMax.z);
    }

    // Identify binning axis
    float axisLength, axisStart;
    int axis = getLongestAxis(node.cMin, node.cMax, axisLength, axisStart);

    // Precompute k1 constant from Wald07
    constexpr float almostOne = 1.0f - 10.0f*FLT_EPSILON;
    float k1 = (NUM_BINS*almostOne)/axisLength;

    // Cache bins in shared memory. We'll atomically update the global bins afterwards.
    // Each bin is 64 bytes, so with 16 bins we're using 1024 bytes of shared memory.
    // If we can keep register usage < 32 then we get 100% occupancy. :)
    __shared__ TriBin bins[NUM_BINS];
    if (threadIdx.x < NUM_BINS) {
        bins[threadIdx.x] = TriBin();
    }
    __syncthreads();

    // We pull the axis conditional outside of the loop to avoid rechecking it
    // every loop iteration. GPUs don't have a branch predictor, and repeated
    // predication is still expensive. To avoid code duplication we use a macro.
    // This also copies triangles into the aux space. Originally I tried to do
    // this inside the partitioning kernel, but I needed to split the
    // partitioning kernel into two and introduce a global sync, so it's
    // preferable to do it here.
#define BUILD_OVER(cmp) \
    for (uint triId = node.triStart + localThreadId; \
              triId < node.triEnd; \
              triId += localThreads) { \
        float4 minId4 = u.triMinsIds[triId]; \
        float4 max4 = u.triMaxs[triId]; \
        float center = 0.5f * (minId4. cmp + max4. cmp); \
        uint binId = uint(k1 * (center-axisStart)); \
        max4.w = __uint_as_float(binId); \
        u.triMinsIdsAux[triId] = minId4; \
        u.triMaxsBinsAux[triId] = max4; \
        TriBin& bin = bins[binId]; \
        atomicAdd_block(&bin.count, 1); \
        atomic_fmin_ew3_block(&bin.min, {minId4.x, minId4.y, minId4.z}); \
        atomic_fmax_ew3_block(&bin.max, {max4.x, max4.y, max4.z}); \
    }

    if (axis == 0) {
        BUILD_OVER(x);
    } else if (axis == 1) {
        BUILD_OVER(y);
    } else {
        BUILD_OVER(z);
    }
    __syncthreads();
#undef BUILD_OVER

    // Shared memory now contains an accurate view of all the bins processed by
    // this threadblock. Let's merge them into the global view.
    if (threadIdx.x < NUM_BINS) {
        TriBin& sBin = bins[threadIdx.x];
        TriBin& gBin = u.bins[queueId*NUM_BINS + threadIdx.x];
        atomicAdd(&gBin.count, sBin.count);
        atomic_fmin_ew3(&gBin.min, sBin.min);
        atomic_fmax_ew3(&gBin.max, sBin.max);
    }
}

// This kernel expects one threadblock for each filled queue entry.
// There are 2^level queue entries at that level.
__global__ __launch_bounds__(WARP_THREADS) 
void horizontalScan(DeviceUniforms u, uint level) {
    uint queueId = blockIdx.x;
    uint mask = (1u << level) - 1u;
    uint nodeId = mask + queueId;
    if (u.nodes[nodeId].isLeaf()) {
        // This kernel is in charge of setting up the child nodes, so we'll need
        // to ensure the child nodes are treated as leaves and skipped.
        if (threadIdx.x == 0) {
            Node& leftChild = u.nodes[2*nodeId+1];
            leftChild.triStart = 0;
            leftChild.triEnd = 0;
            Node& rightChild = u.nodes[2*nodeId+2];
            rightChild.triStart = 0;
            rightChild.triEnd = 0;
        }
        return;
    }

    const TriBin* bins = &u.bins[queueId*NUM_BINS];

    // Each thread evaluates a different split. A lot of threads are unused
    // since we only have NUM_BINS-1 splits to evaluate.
    TriSplitStats stats = {};
    uint splitId = 1 + threadIdx.x;

    for (uint binId = 0; binId < NUM_BINS; ++binId) {
        const TriBin& bin = bins[binId];
        if (binId < splitId) {
            stats.inLeft += bin.count;
            stats.ltMin = fmin3(stats.ltMin, bin.min);
            stats.ltMax = fmax3(stats.ltMax, bin.max);
        } else {
            stats.inRight += bin.count;
            stats.rtMin = fmin3(stats.rtMin, bin.min);
            stats.rtMax = fmax3(stats.rtMax, bin.max);
        }
    }

    // The fminf converts NAN to infinite cost
    float cost = fminf(stats.heuristic(), INFINITY);
    float minCost = cost;

    // Identify minimum cost across threads
    // (We assume there are only 32 threads in the threadblock)
    for (uint buddy = 1; buddy < WARP_THREADS; buddy *= 2) {
        float buddyMinCost = __shfl_sync(
            ~0, minCost, threadIdx.x + buddy, WARP_THREADS);
        minCost = fminf(minCost, buddyMinCost);
    }

    // This ballot trickery guarantees that exactly one thread writes to
    // u.splits, even if multiple threads have an identical min cost.
    uint hasMinCost = __ballot_sync(~0, cost == minCost);
    if (threadIdx.x == __ffs(hasMinCost)) {
        Node& node = u.nodes[nodeId];
        node.splitId = splitId;

        // Mostly-initialize child nodes. The partitioning kernel still needs to
        // compute centroid bounds, and the prebinning kernel in the next
        // iteration needs to compute split IDs.
        Node& leftChild = u.nodes[2*nodeId+1];
        leftChild.triStart = node.triStart;
        leftChild.triEnd = node.triStart + stats.inLeft;
        leftChild.allocTriLeft = leftChild.triStart;
        leftChild.allocTriRight = leftChild.triEnd;
        leftChild.splitId = ~0;
        leftChild.cMin = {+INFINITY, +INFINITY, +INFINITY};
        leftChild.cMax = {-INFINITY, -INFINITY, -INFINITY};
        leftChild.tMin = stats.ltMin;
        leftChild.tMax = stats.ltMax;

        Node& rightChild = u.nodes[2*nodeId+2];
        rightChild.triStart = leftChild.triEnd;
        rightChild.triEnd = node.triEnd;
        rightChild.allocTriLeft = rightChild.triStart;
        rightChild.allocTriRight = rightChild.triEnd;
        rightChild.splitId = ~0;
        rightChild.cMin = {+INFINITY, +INFINITY, +INFINITY};
        rightChild.cMax = {-INFINITY, -INFINITY, -INFINITY};
        rightChild.tMin = stats.rtMin;
        rightChild.tMax = stats.rtMax;
    }
}

__global__ void horizontalPartition(DeviceUniforms u, uint level) {
    uint mask = (1u << level) - 1u;
    uint queueId = mask & blockIdx.x;
    uint nodeId = mask + queueId;
    Node& node = u.nodes[nodeId];
    if (node.isLeaf()) {
        return;
    }

    // The index of this thread within all threads assigned to this node.
    uint localBlockId = blockIdx.x >> level;
    uint localBlocks = gridDim.x >> level;
    uint localThreadId = localBlockId*blockDim.x + threadIdx.x;
    uint localThreads = localBlocks*blockDim.x;

    // Identify binning axis.
    float axisLength, axisStart;
    int axis = getLongestAxis(node.cMin, node.cMax, axisLength, axisStart);

    // We build child centroid bounds while iterating, too
    float3 lcMin, lcMax, rcMin, rcMax;
    lcMin = rcMin = { +INFINITY, +INFINITY, +INFINITY };
    lcMax = rcMax = { -INFINITY, -INFINITY, -INFINITY };

    // Counts how many triangles processed by THIS THREAD
    // fall into the left or right children.
    uint myLeft = 0;
    uint myRight = 0;

#define BUILD_OVER(cmp) \
    for (uint triId = node.triStart + localThreadId; \
              triId < node.triEnd; \
              triId += localThreads) { \
        float4 minId4 = u.triMinsIdsAux[triId]; \
        float4 maxBin4 = u.triMaxsBinsAux[triId]; \
        float3 centroid = { \
            0.5f * (minId4.x + maxBin4.x), \
            0.5f * (minId4.y + maxBin4.y), \
            0.5f * (minId4.z + maxBin4.z) \
        }; \
        if (__float_as_uint(maxBin4.w) < node.splitId) { \
            lcMin = fmin3(lcMin, centroid); \
            lcMax = fmax3(lcMax, centroid); \
            myLeft += 1; \
        } else { \
            rcMin = fmin3(rcMin, centroid); \
            rcMax = fmax3(rcMax, centroid); \
            myRight += 1; \
        } \
    }

    if (axis == 0) {
        BUILD_OVER(x);
    } else if (axis == 1) {
        BUILD_OVER(y);
    } else {
        BUILD_OVER(z);
    }
#undef BUILD_OVER

    // Reduce values over threadblock
    __shared__ uint tbLeft;
    __shared__ uint tbRight;
    __shared__ float3 tbLcMin; 
    __shared__ float3 tbLcMax;
    __shared__ float3 tbRcMin;
    __shared__ float3 tbRcMax;
    if (threadIdx.x == 0) {
       tbLeft = tbRight = 0; 
       tbLcMin = tbRcMin = {+INFINITY, +INFINITY, +INFINITY};
       tbLcMax = tbRcMax = {-INFINITY, -INFINITY, -INFINITY};
    }
    __syncthreads();

    atomicAdd_block(&tbLeft, myLeft);
    atomicAdd_block(&tbRight, myRight);
    atomic_fmin_ew3_block(&tbLcMin, lcMin);
    atomic_fmax_ew3_block(&tbLcMax, lcMax);
    atomic_fmin_ew3_block(&tbRcMin, rcMin);
    atomic_fmax_ew3_block(&tbRcMax, rcMax);
    __syncthreads();

    // Reduce centroid AABBs for child nodes,
    // and globally allocate space for this threadblock's tris
    __shared__ uint tbLeftAlloc;
    __shared__ uint tbRightAlloc;
    if (threadIdx.x == 0) {
        tbLeftAlloc = atomicAdd(&node.allocTriLeft, tbLeft);
        tbRightAlloc = atomicSub(&node.allocTriRight, tbRight) - tbRight;

        Node& leftChild = u.nodes[2*nodeId+1];
        atomic_fmin_ew3(&leftChild.cMin, tbLcMin);
        atomic_fmax_ew3(&leftChild.cMax, tbLcMax);

        Node& rightChild = u.nodes[2*nodeId+2];
        atomic_fmin_ew3(&rightChild.cMin, tbRcMin);
        atomic_fmax_ew3(&rightChild.cMax, tbRcMax);
    }
    __syncthreads();

    // Now each thread allocates enough space for its own left and
    // right triangles.
    uint leftAlloc = atomicAdd_block(&tbLeftAlloc, myLeft);
    uint rightAlloc = atomicAdd_block(&tbRightAlloc, myRight);

    // Finally, we march through the aux triangles again and write them
    // out to the original array in partitioned order.
    for (uint triId = node.triStart + localThreadId;
              triId < node.triEnd;
              triId += localThreads) {
        float4 minId4 = u.triMinsIdsAux[triId];
        float4 maxBin4 = u.triMaxsBinsAux[triId];
        uint outId;
        if (__float_as_uint(maxBin4.w) < node.splitId) {
            outId = leftAlloc;
            leftAlloc += 1;
        } else {
            outId = rightAlloc;
            rightAlloc += 1;
        }
        u.triMinsIds[outId] = minId4;
        u.triMaxs[outId] = maxBin4;
    }
}

__global__ void horizontalValidate(DeviceUniforms u, uint level) {
    bool error = false;
    uint mask = (1u << level) - 1u;
    uint nodeId = mask + blockIdx.x;
    Node& node = u.nodes[nodeId];
    if (node.allocTriLeft != node.allocTriRight) {
        printf("[%u] Alloc L/R mismatch: %u vs %u\n", nodeId, node.allocTriLeft, node.allocTriRight);
        error = true;
    } 
    
    uint leftEnd = u.nodes[2*nodeId+1].triEnd;
    if (node.allocTriLeft != leftEnd) {
        printf("[%u] Left mismatch: %u vs %u\n", nodeId, node.allocTriLeft, leftEnd);
        error = true;
    }

    uint rightStart = u.nodes[2*nodeId+1].triEnd;
    if (node.allocTriRight != rightStart) {
        printf("[%u] Right mismatch: %u vs %u\n", nodeId, node.allocTriRight, rightStart);
        error = true;
    }

    // the big one...
    uint leftWrong = 0;
    for (uint triId = node.triStart; triId < leftEnd; ++triId) {
        float4 max4 = u.triMaxs[triId];
        if (__float_as_uint(max4.w) >= node.splitId && leftWrong < 5) {
            printf("[%u] Left not partitioned @ tri %u. SplitID: %u, binID: %u\n", 
                nodeId, triId, node.splitId, __float_as_uint(max4.w));
            leftWrong += 1;
            error = true;
        }
    }

    uint rightWrong = 0;
    for (uint triId = rightStart; triId < node.triEnd; ++triId) {
        float4 max4 = u.triMaxs[triId];
        if (__float_as_uint(max4.w) < node.splitId && rightWrong < 5) {
            printf("[%u] Right not partitioned @ tri %u. SplitID: %u, binID: %u\n", 
                nodeId, triId, node.splitId, __float_as_uint(max4.w));
            rightWrong += 1;
            error = true;
        }
    }

    if (!error) {
        printf("[%u] validated!\n", nodeId);
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
    cudaMalloc((float4**)&u.triMaxsBinsAux, numTris*sizeof(float4));
    cudaMalloc((Node**)&u.nodes, (2*numTris-1)*sizeof(Node));
    cudaMalloc((TriBin**)&u.bins, QUEUE_SIZE*NUM_BINS*sizeof(TriBin));

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
    setupTris<<<NP2(numTris, OPT_THREADS), OPT_THREADS>>>(u);

    // Create profiling events
    cudaEvent_t horizStart, horizEnd;
    cudaEventCreate(&horizStart);
    cudaEventCreate(&horizEnd);

    // Horizontal binning
    cudaEventRecord(horizStart);
    for (int hLevel = 0; hLevel < HORIZONTAL_LEVELS; ++hLevel) {
        horizontalClearBins<<<(1 << hLevel), WARP_THREADS>>>(u);
        horizontalPrebin<<<OPT_BLOCKS, OPT_THREADS>>>(u, hLevel);
        horizontalScan<<<(1 << hLevel), WARP_THREADS>>>(u, hLevel);
        horizontalPartition<<<OPT_BLOCKS, OPT_THREADS>>>(u, hLevel);
        horizontalValidate<<<(1 << hLevel), 1>>>(u, hLevel);
        break;
    }
    cudaEventRecord(horizEnd);

    // Synchronize
    cudaDeviceSynchronize();

    // Print stats
    float horizTimeMs = 0.0f;
    cudaEventElapsedTime(&horizTimeMs, horizStart, horizEnd);
    printf("GPU elapsed: %f ms\n", horizTimeMs);

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

    return std::make_shared<CudaBVH>(u);
}