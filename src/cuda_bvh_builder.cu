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

// Takes the min of the first 3 elements of A and B.
// Fourth element is unchanged.
__device__ inline float4 min3(float4 a, float4 b) {
    a.x = fminf(a.x, b.x);
    a.y = fminf(a.y, b.y);
    a.z = fminf(a.z, b.z);
    return a;
}

// Takes the max of the first 3 elements of A and B.
// Fourth element is unchanged.
__device__ inline float4 max3(float4 a, float4 b) {
    a.x = fmaxf(a.x, b.x);
    a.y = fmaxf(a.y, b.y);
    a.z = fmaxf(a.z, b.z);
    return a;
}

__device__ inline void atomic_min3(float4* dst, float4 src) {
    // This is legal since IEEE754 floats are bitwise monotonic
    atomicMin((int*)&dst->x, src.x);
    atomicMin((int*)&dst->y, src.y);
    atomicMin((int*)&dst->z, src.z);
}

__device__ inline void atomic_max3(float4* dst, float4 src) {
    // This is legal since IEEE754 floats are bitwise monotonic
    atomicMax((int*)&dst->x, src.x);
    atomicMax((int*)&dst->y, src.y);
    atomicMax((int*)&dst->z, src.z);
}


// A BVH node (either an inner node or a leaf node.)
//
// NOTE: The split axis can be computed implicitly by taking the maximum
// dimension from the triangle centroid bounds.
struct Node {
    // The index of the first triangle in this node.
    int triStart;
    // The number of triangles included in this node.
    int triCount;

    // Index of the left child. Negative if this is a leaf node.
    int left;
    // Index of the right child. Negative if this is a leaf node.
    int right;

    // Triangle geometry bounds
    float4 tMin, tMax;

    // Triangle centroid bounds. Centroid is of the AABB, not the tri.
    float4 cMin, cMax;

    // Constructs an empty node that's ready to store _triCount triangles
    Node(int _triStart, int _triCount)
      : triStart(_triStart), triCount(_triCount),
        left(-1), right(-1),
        tMin { +INFINITY, +INFINITY, +INFINITY, 0.0f },
        tMax { -INFINITY, -INFINITY, -INFINITY, 0.0f },
        cMin { +INFINITY, +INFINITY, +INFINITY, 0.0f },
        cMax { -INFINITY, -INFINITY, -INFINITY, 0.0f } 
    {}
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

    float4 min = {
        fminf(p1.x, fminf(p2.x, p3.x)),
        fminf(p1.y, fminf(p2.y, p3.y)),
        fminf(p1.z, fminf(p2.z, p3.z)),
        __int_as_float(id)
    };
    u.triMinsIds[id] = min;

    float4 max = {
        fmaxf(p1.x, fmaxf(p2.x, p3.x)),
        fmaxf(p1.y, fmaxf(p2.y, p3.y)),
        fmaxf(p1.z, fmaxf(p2.z, p3.z)),
    };
    u.triMaxs[id] = max;

    float4 centroid = {
        0.5f * (min.x + max.x),
        0.5f * (min.y + max.y),
        0.5f * (min.z + max.z),
        0.0f
    };

    // This is WAYYYYY too many atomics
    Node& node = u.nodes[0];
    atomic_min3(&node.tMin, min);
    atomic_max3(&node.tMax, max);
    atomic_min3(&node.cMin, centroid);
    atomic_max3(&node.cMax, centroid);
}


// dynamic parallelism: why write our own load balancing code when we could just use the GPU's firmware?

__global__ __launch_bounds__(WARP_WIDTH)
void buildTree(DeviceUniforms u, int nodeId) {
    Node& node = u.nodes[nodeId];
    int triCount = node.triCount;

    if (triCount <= LEAF_THRESHOLD) {
        return;
    }

    // Initialize this thread's local bounds to a box with negative volume
    float4 tMin = make_float4(+INFINITY, +INFINITY, +INFINITY, 0.0f);
    float4 tMax = make_float4(-INFINITY, -INFINITY, -INFINITY, 0.0f);

    // One difference from Wald's 2007 paper is that we assign triangles to
    // threads in a round-robin fashion. This helps with memory coalescing.
    for (int id = threadIdx.x; id < triCount; id += WARP_WIDTH) {
        float4 min = u.triMinsIds[id];
        float4 max = u.triMaxs[id];
        // TODO
    }

    // TODO
    return;

    // Spawn children using dynamic parallelism
    /*if (threadIdx.x == 0) {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        buildTree<<<1, WARP_WIDTH, 0, stream>>>(u, -1);
        buildTree<<<1, WARP_WIDTH, 0, stream>>>(u, -1);
        cudaStreamDestroy(stream);
    }*/
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
    setupTris<<<NP2(numTris, 64), 64>>>(u);

    // Build tree (vertical parallelism only for now)
    buildTree<<<1, 32>>>(u, 0);

    // Synchronize
    cudaDeviceSynchronize();

    return std::make_shared<CudaBVH>(u);
}