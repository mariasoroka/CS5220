#pragma once 
#include "builder.h"
#include "vector.h"
#include "triangle.h"
#include <string>

/*Function to load an obj file and store the triangles in the array.*/
void load_obj(std::string filename, triangle **triangles, int &num_triangles);