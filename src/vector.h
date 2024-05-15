#pragma once
#include <iostream>

class Vector3{ 
public:
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vector3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}
    Vector3() : x(0), y(0), z(0) {}
    float operator[](int i) const { return (&x)[i]; }

    float x;
    float y;
    float z;
};

int max_component(const Vector3 &v);

Vector3 operator+(const Vector3 &v1, const Vector3 &v2);
Vector3 operator-(const Vector3 &v1, const Vector3 &v2);
Vector3 operator*(const Vector3 &v1, const Vector3 &v2);
Vector3 operator/(const Vector3 &v1, const Vector3 &v2);
Vector3 operator/(const Vector3 &v1, float s);
std::ostream &operator<<(std::ostream &os, const Vector3 &v);



