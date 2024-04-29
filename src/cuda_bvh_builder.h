#pragma once

#include "triangle.h"

#include <memory>

class CudaBVH;

std::shared_ptr<CudaBVH> build_cuda_bvh(
    triangle* tris, 
    int numTris
);

void print_cuda_bvh(
    const CudaBVH& bvh, 
    std::ostream &stream
);