#include <algorithm>
#include <numeric>

#include "bvh_builder.h"



struct StackNode {
    int node_idx;
    AABB aabb;
    int i0;
    int i1;
};
typedef struct StackNode StackNode;

struct SplitNode {
    StackNode child0;
    StackNode child1;
    StackNode operator[](int i) const {
        return i == 0 ? child0 : child1;
    }
};
typedef struct SplitNode SplitNode;


SplitNode get_split_node(const StackNode &node, double split_loc, int axis,
                         const AABB *triangle_bounds, int *triangle_idxs, const Vector3 *triangle_centers) {

    auto it = std::partition(triangle_idxs + node.i0, triangle_idxs + node.i1,
                                    [triangle_idxs, triangle_centers, &split_loc, &axis](int i) {
                                        return triangle_centers[triangle_idxs[i]][axis] < split_loc;
                                    });



    AABB aabb0 = std::transform_reduce(triangle_idxs + node.i0, it, AABB(), merge,
                                      [triangle_idxs, triangle_bounds](int i) {
                                            return triangle_bounds[triangle_idxs[i]];
                                      });

    AABB aabb1 = std::transform_reduce(it, triangle_idxs + node.i1, AABB(), merge,
                                        [triangle_idxs, triangle_bounds](int i) {
                                                return triangle_bounds[triangle_idxs[i]];
                                        });


    int split_idx = it - triangle_idxs;


    return SplitNode{StackNode{0, aabb0, node.i0, split_idx}, StackNode{0, aabb1, split_idx, node.i1}};
}

void get_costs(const StackNode &node, double *costs, const AABB *triangle_bounds,
                int *triangle_idxs, int n_bins, int axis, const Vector3 *triangle_centers){

    Vector3 diag = node.aabb.pmax - node.aabb.pmin;
    for(int i = 0; i < n_bins; i++){
        double split_loc = node.aabb.pmin[axis] + diag[axis] * (i + 1) / (n_bins + 1);
        SplitNode split_node = get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers);

        if (split_node.child0.i1 == node.i0 || split_node.child0.i1 == node.i1) {
            costs[i] = infinity();
            continue;
        }
        costs[i] = get_area(split_node.child0.aabb) * (split_node.child0.i1 - split_node.child0.i0)
                     + get_area(split_node.child1.aabb) * (split_node.child1.i1 - split_node.child1.i0);
    }

}

SplitNode split(const StackNode &node, const AABB *triangle_bounds,
                int *triangle_idxs, int n_bins, const Vector3 *triangle_centers) {

    Vector3 diag = node.aabb.pmax - node.aabb.pmin;
    int axis = max_component(diag);

    double *costs = new double[n_bins];

    get_costs(node, costs, triangle_bounds, triangle_idxs, n_bins, axis, triangle_centers);

    double min_cost = infinity();
    int split_bin = -1;

    for (int i = 0; i < n_bins; i++) {
        if (costs[i] < min_cost) {
            min_cost = costs[i];
            split_bin = i;
        }
    }

    double split_loc = node.aabb.pmin[axis] + diag[axis] * (split_bin + 1) / (n_bins + 1);


    delete[] costs;

    return get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers);

}

BVH build_bvh(triangle* triangles, int num_triangles, int max_triangles, int n_bins) {

    BVH bvh;
    bvh.nodes = new BVHNode[num_triangles];
    bvh.leaves = new BVHNode[num_triangles];

    AABB *triangle_bounds = new AABB[num_triangles];
    Vector3 *triangle_centers = new Vector3[num_triangles];
    int * triangle_idxs = new int[num_triangles];

    std::transform(triangles, triangles + num_triangles, triangle_bounds, triangle_aabb);
    std::transform(triangles, triangles + num_triangles, triangle_centers, get_center);


    std::iota(triangle_idxs, triangle_idxs + num_triangles, 0);

    AABB scene_bounds = std::reduce(triangle_bounds, triangle_bounds + num_triangles, AABB(), merge);

    StackNode root = StackNode{0, scene_bounds, 0, num_triangles};
    bvh.nodes[0].aabb = scene_bounds;
    bvh.nodes[0].is_leaf = false;


    int node_idx = 1;
    int leaf_idx = 0;

    StackNode stack[64];
    int stack_idx = 0;
    stack[stack_idx] = root;
    stack_idx++;

    while (stack_idx > 0) {
        stack_idx--;
        StackNode node = stack[stack_idx];
        SplitNode split_node = split(node, triangle_bounds, triangle_idxs, n_bins, triangle_centers);
        for(int i = 0; i < 2; i++) {
            StackNode child = split_node[i];
            if (child.i1 - child.i0 <= max_triangles) {
                bvh.leaves[leaf_idx].aabb = child.aabb;
                bvh.leaves[leaf_idx].is_leaf = true;
                bvh.leaves[leaf_idx].num_triangles = child.i1 - child.i0;
                bvh.leaves[leaf_idx].triangle_indices = new int[bvh.leaves[leaf_idx].num_triangles];
                std::copy(triangle_idxs + child.i0, triangle_idxs + child.i1, bvh.leaves[leaf_idx].triangle_indices);
                bvh.nodes[node.node_idx].children[i] = &bvh.leaves[leaf_idx];
                leaf_idx++;

            } else {
                bvh.nodes[node_idx].aabb = child.aabb;
                bvh.nodes[node_idx].is_leaf = false;
                stack[stack_idx] = child;
                bvh.nodes[node.node_idx].children[i] = &bvh.nodes[node_idx];
                stack_idx++;
                node_idx++;
            }
        }
    }

    delete[] triangle_bounds;
    delete[] triangle_centers;
    delete[] triangle_idxs;

    return bvh;

}