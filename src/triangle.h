#pragma once

#include "builder.h"
#include "vector.h"

class triangle{
public:
    triangle(const Vector3 &p1_, const Vector3 &p2_, const Vector3 &p3_) : p1(p1_), p2(p2_), p3(p3_) {}

    triangle(const triangle &t) : p1(t.p1), p2(t.p2), p3(t.p3) {}

    triangle() : p1(Vector3(0, 0, 0)), p2(Vector3(0, 0, 0)), p3(Vector3(0, 0, 0)) {}

    Vector3 p1;
    Vector3 p2;
    Vector3 p3;
};

Vector3 get_center(const triangle &t);

