# pragma once

#include "builder.h"
#include "vector.h"
#include "triangle.h"
#include "aabb.h"

class BVHNode {
public:
    AABB aabb;
    BVHNode *children[2];
    bool is_leaf;
    int *triangle_indices = nullptr;
    int num_triangles;

    ~BVHNode() {
        if (is_leaf && triangle_indices != nullptr) {
            delete[] triangle_indices;
        }
    }
};

class BVH{
public:
    BVHNode* nodes;
    BVHNode* leaves;
    ~BVH() {
        delete[] nodes;
        delete[] leaves;
    }
};

BVH build_bvh(triangle* triangles, int num_triangles, int max_triangles, int n_bins);