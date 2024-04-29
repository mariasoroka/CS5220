#include "cuda_bvh_builder.h"

// Return the number of p-sized blocks needed to cover N
template<typename T>
constexpr T NP2(T n, T p) {
    return ((n) + (p-1)) / p;
}

struct DeviceUniforms {
    int numTris;
    triangle* __restrict__ tris;
    float4* __restrict__ triCentroids;
    float4* __restrict__ triMins;
    float4* __restrict__ triMaxs;
};

class CudaBVH {
public:
    CudaBVH(DeviceUniforms u)
      : _u(u) {}

    ~CudaBVH() {
        cudaFree(_u.tris);
        cudaFree(_u.triCentroids);
        cudaFree(_u.triMins);
        cudaFree(_u.triMaxs);
    }
private:
    DeviceUniforms _u;
};

__global__ void setupTris(DeviceUniforms u) {
    const float oneThird = 1.0f / 3.0f;

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= u.numTris) return;

    triangle& tri = u.tris[id];
    float3 p1 = {tri.p1.x, tri.p1.y, tri.p1.z};
    float3 p2 = {tri.p2.x, tri.p2.y, tri.p2.z};
    float3 p3 = {tri.p3.x, tri.p3.y, tri.p3.z};

    float4 centroid = {
        oneThird * (p1.x + p2.x + p3.x),
        oneThird * (p1.y + p2.y + p3.y),
        oneThird * (p1.z + p2.z + p3.z),
        0.0f
    };
    u.triCentroids[id] = centroid;

    float4 min = {
        fminf(p1.x, fminf(p2.x, p3.x)),
        fminf(p1.y, fminf(p2.y, p3.y)),
        fminf(p1.z, fminf(p2.z, p3.z)),
        0.0f
    };
    u.triMins[id] = min;

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
    cudaMalloc((float4**)&u.triCentroids, numTris*sizeof(float4));
    cudaMalloc((float4**)&u.triMins, numTris*sizeof(float4));
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