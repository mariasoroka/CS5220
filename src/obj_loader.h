#pragma once 
#include "builder.h"
#include "vector.h"
#include "triangle.h"
#include <string>

void load_obj(std::string filename, triangle **triangles, int &num_triangles);