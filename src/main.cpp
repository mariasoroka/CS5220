#include "builder.h"
#include "obj_loader.h"
#include "vector.h"
#include "triangle.h"
#include "bvh_builder.h"

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

    char* savename = find_string_option(argc, argv, "-f", nullptr);
    char* output = find_string_option(argc, argv, "-o", nullptr);
    std::string filename(savename);
    triangle* triangles;
    int num_triangles;
    load_obj(filename, &triangles, num_triangles);

    auto start = std::chrono::high_resolution_clock::now();
    BVH bvh = build_bvh(triangles, num_triangles, 4, 10);
    auto end = std::chrono::high_resolution_clock::now();

    float buildTime = std::chrono::duration<float>(end-start).count();
    printf("Built BVH with %i triangles in %f ms\n", num_triangles, 1000.0f * buildTime);
    printf("Levels:\n");
    for (int level = 0; level < bvh.levelInfos.size(); ++level) {
        const auto& info = bvh.levelInfos[level];
        printf("  %i: %i splits, %f ms\n", level, info.splits, 1000.0f * info.time);
    }

    if(output != nullptr){
        std::ofstream fsave(output);
        print_bvh(fsave, bvh, triangles);
        fsave.close();
    }
    
    delete[] triangles;

}