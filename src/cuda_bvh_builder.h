#pragma once

#include "triangle.h"

#include <memory>

class CudaBVH;

// Initializes CUDA early.
// We don't want CUDA startup time to be included in our timings.
void touch_cuda();

std::shared_ptr<CudaBVH> build_cuda_bvh(
    triangle* tris, 
    int numTris
);

void print_cuda_bvh(
    const CudaBVH& bvh, 
    std::ostream &stream
);