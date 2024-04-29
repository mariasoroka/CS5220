#include <algorithm>

#include <numeric>

#include "bvh_builder.h"

#include <omp.h>

#include <deque>

#include <cassert>

#define NUM_THREADS 16

/*A datastructure used during the tree construction. Each instance stores

the index of the corresponding node in the bvh.nodes array, a bounding box of this node,

index i0 and index i1 that specify which range of triangle indices in triangle_idxs array

belongs to this node. I.e. bvh.nodes[node_idx] covers all the triangles with indices

in triangle_idxs[i0:i1]*/

struct StackNode

{

    int node_idx;

    AABB aabb;

    int i0;

    int i1;
};

typedef struct StackNode StackNode;

/*A datastructure to store two descendants of a node. One could use std::pair instead.*/

struct SplitNode

{

    StackNode child0;

    StackNode child1;

    StackNode operator[](int i) const

    {

        return i == 0 ? child0 : child1;
    }
};

typedef struct SplitNode SplitNode;

/*This function computes two descendants of the node when provided with axis along which

the bounding box of the node should be split and location split_loc where the cut should be made.

The function also uses bounding boxes of all the triangles (triangle_bounds),

positions of triangle centers (triangle_centers) and indices of triangles array (triangle_idxs).*/

SplitNode get_split_node(const StackNode &node, double split_loc, int axis,

                         const AABB *triangle_bounds, int *triangle_idxs, const Vector3 *triangle_centers)

{

    // partition triangles in triangle_idxs[node.i0, node.i1] into two groups based on the position of their centers relative to split_loc

    auto it = std::partition(triangle_idxs + node.i0, triangle_idxs + node.i1,

                             [triangle_idxs, triangle_centers, &split_loc, &axis](int i)

                             {
                                 return (triangle_centers[i][axis] < split_loc);
                             });

    // if the split is degenerate, put half of the triangles in each group

    if (it == triangle_idxs + node.i0 || it == triangle_idxs + node.i1)

    {

        it = triangle_idxs + (node.i0 + node.i1) / 2;
    }

    // compute bounding boxes of the two groups of triangles

    AABB aabb0 = std::transform_reduce(triangle_idxs + node.i0, it, AABB(), merge,

                                       [triangle_idxs, triangle_bounds](int i)

                                       {
                                           return triangle_bounds[i];
                                       });

    AABB aabb1 = std::transform_reduce(it, triangle_idxs + node.i1, AABB(), merge,

                                       [triangle_idxs, triangle_bounds](int i)

                                       {
                                           return triangle_bounds[i];
                                       });

    int split_idx = it - triangle_idxs;

    return SplitNode{StackNode{0, aabb0, node.i0, split_idx}, StackNode{0, aabb1, split_idx, node.i1}};
}

/*This function computes the cost of splitting the node along the given axis.

The node will be split in n_bins places along axis axis. For each split, the cost (surface area heuristic)

will be computed and will be stored in costs*/

void get_costs(const StackNode &node, double *costs, const AABB *triangle_bounds,

               int *triangle_idxs, int n_bins, int axis, const Vector3 *triangle_centers)

{

    Vector3 diag = node.aabb.pmax - node.aabb.pmin;

    for (int i = 0; i < n_bins; i++)

    {

        // compute the location of the split

        double split_loc = node.aabb.pmin[axis] + diag[axis] * (i + 1) / (n_bins + 1);

        // get the two children of the node if it is split at split_loc

        SplitNode split_node = get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers);

        // if the split is degenerate, set the cost to infinity

        if (split_node.child0.i1 == node.i0 || split_node.child0.i1 == node.i1)

        {

            costs[i] = infinity();

            continue;
        }

        // compute SAH (surface area heuristic) cost of the split

        costs[i] = get_area(split_node.child0.aabb) * (split_node.child0.i1 - split_node.child0.i0) + get_area(split_node.child1.aabb) * (split_node.child1.i1 - split_node.child1.i0);
    }
}

/*This function splits the node in the best way according to SAH.*/

SplitNode split(const StackNode &node, const AABB *triangle_bounds,

                int *triangle_idxs, int n_bins, const Vector3 *triangle_centers)

{

    // compute the diagonal of the bounding box of the node and choose the longest axis

    Vector3 diag = node.aabb.pmax - node.aabb.pmin;

    int axis = max_component(diag);

    // allocate memory for the costs and compute them

    double *costs = new double[n_bins];

    get_costs(node, costs, triangle_bounds, triangle_idxs, n_bins, axis, triangle_centers);

    // choose the split with the smallest cost

    double min_cost = infinity();

    int split_bin = 0;

    for (int i = 0; i < n_bins; i++)

    {

        if (costs[i] < min_cost)

        {

            min_cost = costs[i];

            split_bin = i;
        }
    }

    delete[] costs;

    // compute the split

    double split_loc = node.aabb.pmin[axis] + diag[axis] * (split_bin + 1) / (n_bins + 1);

    return get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers);
}

/*This function builds a BVH for the given set of triangles.*/

BVH build_bvh(triangle *triangles, int num_triangles, int max_triangles, int n_bins)

{
    omp_set_num_threads(NUM_THREADS);
    // allocate memory for the BVH arrays

    BVH bvh;

    bvh.nodes = new BVHNode[num_triangles];

    bvh.leaves = new BVHNode[num_triangles];

    // allocate memory for the triangle bounds, centers and indices

    AABB *triangle_bounds = new AABB[num_triangles];

    Vector3 *triangle_centers = new Vector3[num_triangles];

    int *triangle_idxs = new int[num_triangles];

    // compute the bounds, centers and fill the buffer triangle_idxs with indices

    std::transform(triangles, triangles + num_triangles, triangle_bounds, triangle_aabb);

    std::transform(triangles, triangles + num_triangles, triangle_centers, get_center);

    std::iota(triangle_idxs, triangle_idxs + num_triangles, 0);

    // compute the bounding box of the scene

    AABB scene_bounds = std::reduce(triangle_bounds, triangle_bounds + num_triangles, AABB(), merge);

    // counters for number of nodes and leaves

    int node_idx = 0;

    int leaf_idx = 0;

    // initialize the root of the tree and add it to the bvh.nodes array

    StackNode root = StackNode{0, scene_bounds, 0, num_triangles};

    bvh.nodes[0].aabb = scene_bounds;

    bvh.nodes[0].is_leaf = false;

    node_idx++;

    // set up the stack for the tree construction

    StackNode stack[64];

    std::deque<StackNode> queue;

    queue.push_back(root);

    // int stack_idx = 0;

    // stack[stack_idx] = root;

    // stack_idx++;

    // basically keep track of stack_idx

    // while there are nodes to split

    while (queue.size() > 0 && queue.size() < NUM_THREADS)

    {

        // pop the node from the stack

        // stack_idx--;

        // StackNode node = stack[stack_idx];

        StackNode node = queue.front();

        queue.pop_front();

        // split the node

        SplitNode split_node = split(node, triangle_bounds, triangle_idxs, n_bins, triangle_centers);

        // for each child of the node

        for (int i = 0; i < 2; i++)

        {

            StackNode child = split_node[i];

            // if the child is a leaf, add it to the bvh.leaves array

            if (child.i1 - child.i0 <= max_triangles)

            {

                bvh.leaves[leaf_idx].aabb = child.aabb;

                bvh.leaves[leaf_idx].is_leaf = true;

                bvh.leaves[leaf_idx].num_triangles = child.i1 - child.i0;

                bvh.leaves[leaf_idx].triangle_indices = new int[bvh.leaves[leaf_idx].num_triangles];

                std::copy(triangle_idxs + child.i0, triangle_idxs + child.i1, bvh.leaves[leaf_idx].triangle_indices);

                bvh.nodes[node.node_idx].children[i] = &bvh.leaves[leaf_idx];

                leaf_idx++;
            }

            // if the child is not a leaf, add it to the bvh.nodes array and push it to the stack

            else

            {

                bvh.nodes[node_idx].aabb = child.aabb;

                bvh.nodes[node_idx].is_leaf = false;

                queue.push_back({node_idx, child.aabb, child.i0, child.i1});

                // stack[stack_idx] = {node_idx, child.aabb, child.i0, child.i1};

                bvh.nodes[node.node_idx].children[i] = &bvh.nodes[node_idx];

                // stack_idx++;

                node_idx++;
            }
        }
    }

    assert(queue.size() == NUM_THREADS);

#pragma omp parallel shared(bvh) shared(node_idx) shared(leaf_idx)

    {
        int curr_node_idx;
        int curr_leaf_idx;
        int thread_idx = omp_get_thread_num();
        // int thread_idx = 1;

        StackNode root = queue[thread_idx];
        // node_idx++;
        StackNode stack[64];
        int stack_idx = 0;
        stack[stack_idx] = root;
        stack_idx++;

        while (stack_idx > 0)

        {

            // pop the node from the stack

            stack_idx--;

            StackNode node = stack[stack_idx];

            // split the node

            SplitNode split_node = split(node, triangle_bounds, triangle_idxs, n_bins, triangle_centers);

            // for each child of the node

            for (int i = 0; i < 2; i++)

            {

                StackNode child = split_node[i];

                // if the child is a leaf, add it to the bvh.leaves array

                if (child.i1 - child.i0 <= max_triangles)

                {

                    // atomic

#pragma omp atomic capture
                    curr_leaf_idx = ++leaf_idx;

                    bvh.leaves[curr_leaf_idx].aabb = child.aabb;

                    bvh.leaves[curr_leaf_idx].is_leaf = true;

                    bvh.leaves[curr_leaf_idx].num_triangles = child.i1 - child.i0;

                    bvh.leaves[curr_leaf_idx].triangle_indices = new int[bvh.leaves[curr_leaf_idx].num_triangles];

                    std::copy(triangle_idxs + child.i0, triangle_idxs + child.i1, bvh.leaves[curr_leaf_idx].triangle_indices);

                    bvh.nodes[node.node_idx].children[i] = &bvh.leaves[curr_leaf_idx];
                }

                // if the child is not a leaf, add it to the bvh.nodes array and push it to the stack

                else

                {

#pragma omp atomic capture
                    curr_node_idx = ++node_idx;

                    bvh.nodes[curr_node_idx].aabb = child.aabb;

                    bvh.nodes[curr_node_idx].is_leaf = false;

                    stack[stack_idx] = {curr_node_idx, child.aabb, child.i0, child.i1};

                    bvh.nodes[node.node_idx].children[i] = &bvh.nodes[curr_node_idx];

                    stack_idx++;
                }
            }
        }
    }

    // something to keep track of depth more or less. probably change to bfs? so depth can be tracked?

    delete[] triangle_bounds;

    delete[] triangle_centers;

    delete[] triangle_idxs;

    return bvh;
}

struct PrintBvhNode

{

    int level;

    int node_idx;

    bool is_leaf;
};

typedef struct PrintBvhNode PrintBvhNode;

void print_bvh(std::ostream &stream, const BVH &bvh, triangle *triangles)

{

    PrintBvhNode stack[64];

    int stack_idx = 0;

    stack[stack_idx] = PrintBvhNode{0, 0, false};

    stack_idx++;

    while (stack_idx > 0)

    {

        stack_idx--;

        PrintBvhNode node = stack[stack_idx];

        if (node.is_leaf)

        {

            stream << "Level " << node.level << std::endl;

            stream << "Leaf node with " << bvh.leaves[node.node_idx].num_triangles << " triangles" << std::endl;

            stream << "Bounding box: " << bvh.leaves[node.node_idx].aabb.pmin << " " << bvh.leaves[node.node_idx].aabb.pmax << std::endl;

            for (int i = 0; i < bvh.leaves[node.node_idx].num_triangles; i++)

            {

                stream << "Triangle " << bvh.leaves[node.node_idx].triangle_indices[i] << ": " << triangles[bvh.leaves[node.node_idx].triangle_indices[i]] << std::endl;
            }
        }

        else

        {

            stream << "Level " << node.level << std::endl;

            stream << "Node with two descendants" << std::endl;

            stream << "Bounding box: " << bvh.nodes[node.node_idx].aabb.pmin << " " << bvh.nodes[node.node_idx].aabb.pmax << std::endl;

            for (int i = 0; i < 2; i++)

            {

                int idx = bvh.nodes[node.node_idx].children[i]->is_leaf ? bvh.nodes[node.node_idx].children[i] - bvh.leaves : bvh.nodes[node.node_idx].children[i] - bvh.nodes;

                stack[stack_idx] = PrintBvhNode{node.level + 1, idx, bvh.nodes[node.node_idx].children[i]->is_leaf};

                stack_idx++;
            }
        }
    }
}