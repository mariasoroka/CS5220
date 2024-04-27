# pragma once

#include "builder.h"
#include "vector.h"
#include "triangle.h"
#include "aabb.h"

/*BVH node class. Each node stroes its bounding box, pointers to two decendants if it is not a leaf node,
indices of triangles covered by the node if it is a leaf node.*/
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

/*Class for BVH. It stores an array of non-leaf nodes and an array with leaf nodes. 
The root of the tree is in nodes[0].*/
class BVH{
public:
    BVHNode* nodes;
    BVHNode* leaves;
    ~BVH() {
        delete[] nodes;
        delete[] leaves;
    }
};

/*Function building a bvh tree from an array of triangles.
max_triangles specifies how many triangles can a leaf node store.
n_bins specifies how many bins should be used to compute the best splitting.*/
BVH build_bvh(triangle* triangles, int num_triangles, int max_triangles, int n_bins);