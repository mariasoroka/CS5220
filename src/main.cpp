#include "builder.h"
#include "obj_loader.h"
#include "vector.h"
#include "triangle.h"
#include "bvh_builder.h"
#include "cuda_bvh_builder.h"

#include <cstring>
#include <fstream>
#include <chrono>

int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

int main(int argc, char** argv) {
    using namespace std::literals;

    char* savename = find_string_option(argc, argv, "-f", nullptr);
    char* output = find_string_option(argc, argv, "-o", nullptr);
    std::string filename(savename);
    triangle* triangles;
    int num_triangles;
    load_obj(filename, &triangles, num_triangles);

    auto start = std::chrono::high_resolution_clock::now();

#if 0
    BVH bvh = build_bvh(triangles, num_triangles, 4, 10);
    std::cout << "Built BVH with " << num_triangles << " triangles" << std::endl;

    if(output != nullptr){
        std::ofstream fsave(output);
        print_bvh(fsave, bvh, triangles);
        fsave.close();
    }
#else
    
    auto bvh = build_cuda_bvh(triangles, num_triangles);
    std::cout << "Built CUDA BVH with " << num_triangles << " triangles\n";

#endif

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "BVH build took " << (end - start) / 1ms << " ms\n";

    delete[] triangles;

}