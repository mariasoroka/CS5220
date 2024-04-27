#include "triangle.h"

Vector3 get_center(const triangle &t) {
    return Vector3((t.p1.x + t.p2.x + t.p3.x) / 3, (t.p1.y + t.p2.y + t.p3.y) / 3, (t.p1.z + t.p2.z + t.p3.z) / 3);
}