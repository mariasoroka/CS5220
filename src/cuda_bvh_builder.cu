#include "cuda_bvh_builder.h"

// Return the number of p-sized blocks needed to cover N
template<typename T>
constexpr T NP2(T n, T p) {
    return ((n) + (p-1)) / p;
}


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

    float4 max {
        fmaxf(p1.x, fmaxf(p2.x, p3.x)),
        fmaxf(p1.y, fmaxf(p2.y, p3.y)),
        fmaxf(p1.z, fmaxf(p2.z, p3.z)),
        0.0f
    };
    u.triMaxs[id] = max;
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

    // Copy triangles into device memory
    cudaMemcpy(
        u.tris,
        tris, 
        numTris*sizeof(triangle), 
        cudaMemcpyHostToDevice);

    // Compute centroids, mins, and maxes for triangles
    setupTris<<<NP2(numTris, 64), 64>>>(u);

    // Synchronize
    cudaDeviceSynchronize();

    return std::make_shared<CudaBVH>(u);
}