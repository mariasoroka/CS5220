#include "obj_loader.h"
#include "vector.h"
#include <vector>

#include <igl/readOBJ.h>

void load_obj(std::string filename, triangle **triangles, int &num_triangles) {
    std::vector<std::vector<double>> V;
    std::vector<std::vector<int>> F;
    igl::readOBJ(filename, V, F);

    num_triangles = F.size();
    *triangles = new triangle[num_triangles];
    for (int i = 0; i < num_triangles; i++) {
        (*triangles + i)->p1 = Vector3((float) V[F[i][0]][0], (float) V[F[i][0]][1], (float) V[F[i][0]][2]);
        (*triangles + i)->p2 = Vector3((float) V[F[i][1]][0], (float) V[F[i][1]][1], (float) V[F[i][1]][2]);
        (*triangles + i)->p3 = Vector3((float) V[F[i][2]][0], (float) V[F[i][2]][1], (float) V[F[i][2]][2]);
    }

    std::cout << "Loaded " << num_triangles << " triangles from " << filename << std::endl;
}