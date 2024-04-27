#include "builder.h"
#include "obj_loader.h"
#include "vector.h"
#include "triangle.h"
#include "bvh_builder.h"

#include <string>

int main() {
    std::string filename = "/global/homes/m/msoroka/HW1/CS5220_2024SP/project/meshes/sphere1.obj";
    triangle* triangles;
    int num_triangles;
    load_obj(filename, &triangles, num_triangles);

    BVH bvh = build_bvh(triangles, num_triangles, 4, 10);
    std::cout << "Built BVH with " << num_triangles << " triangles" << std::endl;

    delete[] triangles;

}