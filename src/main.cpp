#include "builder.h"
#include "obj_loader.h"
#include "vector.h"
#include "triangle.h"
#include "bvh_builder.h"

#include <cstring>
#include <cstring>
#include <string>

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
    std::string filename(savename);
    triangle* triangles;
    int num_triangles;
    load_obj(filename, &triangles, num_triangles);

    BVH bvh = build_bvh(triangles, num_triangles, 4, 10);
    std::cout << "Built BVH with " << num_triangles << " triangles" << std::endl;

    delete[] triangles;

}