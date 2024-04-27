#pragma once
#include <iostream>

class Vector3{ 
public:
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vector3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}
    Vector3() : x(0), y(0), z(0) {}
    double operator[](int i) const { return (&x)[i]; }

    double x;
    double y;
    double z;
};

int max_component(const Vector3 &v);

Vector3 operator+(const Vector3 &v1, const Vector3 &v2);
Vector3 operator-(const Vector3 &v1, const Vector3 &v2);
Vector3 operator*(const Vector3 &v1, const Vector3 &v2);
Vector3 operator/(const Vector3 &v1, const Vector3 &v2);
Vector3 operator/(const Vector3 &v1, double s);
std::ostream &operator<<(std::ostream &os, const Vector3 &v);



