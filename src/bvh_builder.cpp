#include <algorithm>

#include <numeric>
#include <chrono>

#include "bvh_builder.h"

#include <omp.h>

#include <deque>

#include <cassert>

#define NUM_THREADS 16

#define min(a, b) ((a) < (b) ? (a) : (b))

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
    int level;
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

SplitNode get_split_node(const StackNode &node, float split_loc, int axis,

                         const AABB *triangle_bounds, int *triangle_idxs, const Vector3 *triangle_centers, bool remove_degenerate)

{

    // partition triangles in triangle_idxs[node.i0, node.i1] into two groups based on the position of their centers relative to split_loc

    auto it = std::partition(triangle_idxs + node.i0, triangle_idxs + node.i1,

                             [triangle_idxs, triangle_centers, &split_loc, &axis](int i)

                             {
                                 return (triangle_centers[i][axis] < split_loc);
                             });

    // if the split is degenerate, put half of the triangles in each group
    if (remove_degenerate)
    {
        if (it == triangle_idxs + node.i0 || it == triangle_idxs + node.i1)

        {

            it = triangle_idxs + (node.i0 + node.i1) / 2;
        }
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

    return SplitNode{
        StackNode{0, aabb0, node.i0, split_idx, node.level+1}, 
        StackNode{0, aabb1, split_idx, node.i1, node.level+1}};
}

SplitNode get_split_node(int start, int end, float split_loc, int axis,

                         const AABB *triangle_bounds, int *triangle_idxs, const Vector3 *triangle_centers)

{

    // partition triangles in triangle_idxs[node.i0, node.i1] into two groups based on the position of their centers relative to split_loc

    auto it = std::partition(triangle_idxs + start, triangle_idxs + end,

                             [&triangle_centers, &split_loc, &axis](int i)

                             {
                                 return (triangle_centers[i][axis] < split_loc);
                             });



    // compute bounding boxes of the two groups of triangles
    AABB aabb0 = AABB();
    AABB aabb1 = AABB();
    if (it != triangle_idxs + start) {
        aabb0 = std::transform_reduce(triangle_idxs + start, it, AABB(), merge,

                                        [&triangle_bounds](int i)

                                        {
                                            return triangle_bounds[i];
                                        });
    }

    if (it != triangle_idxs + end) {

        aabb1 = std::transform_reduce(it, triangle_idxs + end, AABB(), merge,

                                        [&triangle_bounds](int i)

                                        {
                                            return triangle_bounds[i];
                                        });
    }

    int split_idx = it - triangle_idxs;

    return SplitNode{StackNode{0, aabb0, start, split_idx}, StackNode{0, aabb1, split_idx, end}};
}

/*This function computes the cost of splitting the node along the given axis.

The node will be split in n_bins places along axis axis. For each split, the cost (surface area heuristic)

will be computed and will be stored in costs*/

void get_costs(const StackNode &node, float *costs, const AABB *triangle_bounds,

               int *triangle_idxs, int n_bins, int axis, const Vector3 *triangle_centers)

{

    Vector3 diag = node.aabb.pmax - node.aabb.pmin;

    for (int i = 0; i < n_bins; i++)

    {

        // compute the location of the split

        float split_loc = node.aabb.pmin[axis] + diag[axis] * (i + 1) / (n_bins + 1);

        // get the two children of the node if it is split at split_loc

        SplitNode split_node = get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers, false);

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

    float *costs = new float[n_bins];

    get_costs(node, costs, triangle_bounds, triangle_idxs, n_bins, axis, triangle_centers);

    // choose the split with the smallest cost

    float min_cost = infinity();

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

    float split_loc = node.aabb.pmin[axis] + diag[axis] * (split_bin + 1) / (n_bins + 1);

    return get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers, true);
}

SplitNode split(const StackNode &node, const AABB *triangle_bounds,

                int *triangle_idxs, int n_bins, const Vector3 *triangle_centers, float *costs)

{
    // compute the diagonal of the bounding box of the node and choose the longest axis

    Vector3 diag = node.aabb.pmax - node.aabb.pmin;

    int axis = max_component(diag);

    // choose the split with the smallest cost

    float min_cost = infinity();

    int split_bin = 0;

    for (int i = 0; i < n_bins; i++)

    {

        if (costs[i] < min_cost)

        {

            min_cost = costs[i];

            split_bin = i;
        }
    }

    // compute the split

    float split_loc = node.aabb.pmin[axis] + diag[axis] * (split_bin + 1) / (n_bins + 1);

    return get_split_node(node, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers, true);
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
#pragma omp parallel for 
    for (int i = 0; i < num_triangles; i++) {

        triangle_bounds[i] = triangle_aabb(triangles[i]);

        triangle_centers[i] = get_center(triangles[i]);

        triangle_idxs[i] = i;
    }



// #pragma omp declare reduction(                             \
//                             mergeBbox :                     \
//                             AABB :   \
//                             operator+=(omp_out, omp_in)      \
//                             )                             \

    // compute the bounding box of the scene
    // AABB scene_bounds;
    // #pragma omp parallel for reduction(mergeBbox:scene_bounds)
    //     for (int i = 0; i < num_triangles; i++)
    //     {
    //         scene_bounds += triangle_bounds[i];
    //     }


    AABB scene_bounds = std::reduce(triangle_bounds, triangle_bounds + num_triangles, AABB(), merge);
    // variables necessary for horizontal parallellization
    // SplitNode **partial_splits = new SplitNode *[n_bins]; // can probably declare this earlier
    // int **num_n0 = new int *[n_bins];
    // int **num_n1 = new int *[n_bins];
    // AABB **AABB_n0 = new AABB *[n_bins];
    // AABB **AABB_n1 = new AABB *[n_bins];

    SplitNode **partial_splits = new SplitNode *[NUM_THREADS]; // can probably declare this earlier
    int **num_n0 = new int *[NUM_THREADS];
    int **num_n1 = new int *[NUM_THREADS];
    AABB **AABB_n0 = new AABB *[NUM_THREADS];
    AABB **AABB_n1 = new AABB *[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        // partial_splits[i] = new SplitNode[NUM_THREADS];
        // num_n0[i] = new int[NUM_THREADS];
        // num_n1[i] = new int[NUM_THREADS];
        // AABB_n0[i] = new AABB[NUM_THREADS];
        // AABB_n1[i] = new AABB[NUM_THREADS];

        partial_splits[i] = new SplitNode[n_bins];
        num_n0[i] = new int[n_bins];
        num_n1[i] = new int[n_bins];
        AABB_n0[i] = new AABB[n_bins];
        AABB_n1[i] = new AABB[n_bins];
    }
    float *costs = new float[n_bins];

    // counters for number of nodes and leaves

    int node_idx = 0;

    int leaf_idx = 0;

    // initialize the root of the tree and add it to the bvh.nodes array
    StackNode root = StackNode{0, scene_bounds, 0, num_triangles, 0};
    bvh.nodes[0].aabb = scene_bounds;
    bvh.nodes[0].is_leaf = false;
    node_idx++;
    std::deque<StackNode> queue;
    queue.push_back(root);
    // while there are nodes to split horizontal parallelization step

    while (queue.size() > 0 && queue.size() < NUM_THREADS)
    {
        // std::cout << "queue size is  " << queue.size() << " increasing \n" << std::endl;
        // pop the node from the stack
        auto start = std::chrono::high_resolution_clock::now();

        StackNode node = queue.front();
        queue.pop_front();

        // get necessary parameters for split
        Vector3 diag = node.aabb.pmax - node.aabb.pmin;
        int axis = max_component(diag);
#pragma omp parallel shared(node) shared(triangle_idxs) shared(num_n0) shared(num_n1) shared(AABB_n0) shared(AABB_n1) shared(triangle_bounds)
        {
            // each thread works on a subsection of triangles
            int thread_idx = omp_get_thread_num();
            int curr_num_triangles = node.i1 - node.i0;
            int n_per_thread = (curr_num_triangles + NUM_THREADS) / NUM_THREADS;
            int start = node.i0 + min(thread_idx * n_per_thread, curr_num_triangles);
            int end = node.i0 + min((thread_idx + 1) * n_per_thread, curr_num_triangles);
            // int end = node.i0 + (thread_idx == NUM_THREADS - 1 ? curr_num_triangles : (thread_idx + 1) * (curr_num_triangles / NUM_THREADS));
            if (start != end) {
                for (int i = 0; i < n_bins; i++)
                {
                    float split_loc = node.aabb.pmin[axis] + diag[axis] * (i + 1) / (n_bins + 1);
                    SplitNode sub_split_node = get_split_node(start, end, split_loc, axis, triangle_bounds, triangle_idxs, triangle_centers);
                    num_n0[thread_idx][i] = sub_split_node.child0.i1 - sub_split_node.child0.i0;
                    num_n1[thread_idx][i] = sub_split_node.child1.i1 - sub_split_node.child1.i0;
                    AABB_n0[thread_idx][i] = sub_split_node.child0.aabb;
                    AABB_n1[thread_idx][i] = sub_split_node.child1.aabb;
                }
            }
            else {
                for (int i = 0; i < n_bins; i++)
                {
                    num_n0[thread_idx][i] = 0;
                    num_n1[thread_idx][i] = 0;
                    AABB_n0[thread_idx][i] = AABB();
                    AABB_n1[thread_idx][i] = AABB();
                }
            }
        }

        for (int i = 0; i < n_bins; i++)
        {   
            AABB total_AABB_n0 = AABB();
            AABB total_AABB_n1 = AABB();
            int total_num_n0 = 0;
            int total_num_n1 = 0;
            for(int j = 0; j < NUM_THREADS; ++j) {
                total_num_n0 += num_n0[j][i];
                total_num_n1 += num_n1[j][i];
                total_AABB_n0 = merge(total_AABB_n0, AABB_n0[j][i]);
                total_AABB_n1 = merge(total_AABB_n1, AABB_n1[j][i]);
                // std::cout << get_area(total_AABB_n0) << " " << get_area(total_AABB_n1) << std::endl;
            }
            // std::cout << "total num n0 is " << total_num_n0 << " total num n1 is " << total_num_n1 << std::endl;
            // std::cout << "area 0 " << get_area(total_AABB_n0) << " area 1 " << get_area(total_AABB_n1) << std::endl;
            if (total_num_n0 == 0 || total_num_n1 == 0) {
                costs[i] = infinity();
            }
            else {
                costs[i] = get_area(total_AABB_n0) * (total_num_n0) + get_area(total_AABB_n1) * (total_num_n1);
            }
            // assert (total_num_n0 + total_num_n1 == node.i1 - node.i0);
            // assert (merge(total_AABB_n0, total_AABB_n1) == node.aabb);

        }
        SplitNode split_node = split(node, triangle_bounds, triangle_idxs, n_bins, triangle_centers, costs);

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
                queue.push_back({node_idx, child.aabb, child.i0, child.i1, node.level + 1});
                bvh.nodes[node.node_idx].children[i] = &bvh.nodes[node_idx];
                node_idx++;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        #pragma omp master
        {
            // Instrument this split
            if (node.level >= bvh.levelInfos.size()) {
                bvh.levelInfos.push_back({});
            }
            BVH::LevelInfo& info = bvh.levelInfos[node.level];
            info.splits += 1;
            info.time += std::chrono::duration<float>(end-start).count();
        }
    }

    // std::cout << "queue size is  " << queue.size() << " hopefully \n" << std::endl;
    //  assert(queue.size() == NUM_THREADS);
    if (queue.size() == NUM_THREADS)
    {   
        std::vector<BVH::LevelInfo> threadLevelInfos[NUM_THREADS];

// vertical parallelization
#pragma omp parallel shared(bvh) shared(node_idx) shared(leaf_idx) shared(threadLevelInfos)
        {
            int curr_node_idx;
            int curr_leaf_idx;
            int thread_idx = omp_get_thread_num();
            // std::cout << "thread  " << thread_idx << " is working" << std::endl;
            //  int thread_idx = 1;
            threadLevelInfos[thread_idx] = bvh.levelInfos;

            StackNode root = queue[thread_idx];
            // node_idx++;
            StackNode stack[64];
            int stack_idx = 0;
            stack[stack_idx] = root;
            stack_idx++;

            auto& locallevelInfos = threadLevelInfos[thread_idx];

            while (stack_idx > 0)

            {
                auto start = std::chrono::high_resolution_clock::now();
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

                        stack[stack_idx] = {curr_node_idx, child.aabb, child.i0, child.i1, child.level};

                        bvh.nodes[node.node_idx].children[i] = &bvh.nodes[curr_node_idx];

                        stack_idx++;
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();

                // Instrument this split
                if (node.level >= locallevelInfos.size()) {
                    locallevelInfos.push_back({0, 0.0f});
                }
                BVH::LevelInfo& info = locallevelInfos[node.level];
                info.splits += 1;
                info.time += std::chrono::duration<float>(end-start).count();
            }
        }

        // reduce times using max
        int startLevel = bvh.levelInfos.size();
        #pragma omp master
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            for (int level = startLevel; level < threadLevelInfos[tid].size(); ++level) {
                const auto& tInfo = threadLevelInfos[tid][level];
                if (level >= bvh.levelInfos.size()) {
                    bvh.levelInfos.push_back(tInfo);
                } else {
                    auto& mInfo = bvh.levelInfos[level];
                    mInfo.splits += tInfo.splits;
                    mInfo.time = std::max(mInfo.time, tInfo.time);
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
