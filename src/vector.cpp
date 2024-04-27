#include "vector.h"

int max_component(const Vector3 &v) {
    if (v.x > v.y && v.x > v.z) {
        return 0;
    } else if (v.y > v.z) {
        return 1;
    } else {
        return 2;
    }
}

Vector3 operator+(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

Vector3 operator-(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

Vector3 operator*(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

Vector3 operator/(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

Vector3 operator/(const Vector3 &v1, double s) {
    return Vector3(v1.x / s, v1.y / s, v1.z / s);
}
std::ostream &operator<<(std::ostream &os, const Vector3 &v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}