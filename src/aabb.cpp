#include "aabb.h"
#include "builder.h"
#include <math.h>      

bool is_degenerate(const AABB &aabb) {
    return aabb.pmin.x >= aabb.pmax.x || aabb.pmin.y >= aabb.pmax.y || aabb.pmin.z >= aabb.pmax.z;
}

float get_area(const AABB &aabb) {
    Vector3 diff = aabb.pmax - aabb.pmin;
    if (is_degenerate(aabb)) {
        return 0;
    }
    return 2 * (diff.x * diff.y + diff.y * diff.z + diff.z * diff.x);
}

AABB merge(const AABB &aabb1, const AABB &aabb2) {
    Vector3 pmin = Vector3(std::min(aabb1.pmin.x, aabb2.pmin.x), std::min(aabb1.pmin.y, aabb2.pmin.y), std::min(aabb1.pmin.z, aabb2.pmin.z));
    Vector3 pmax = Vector3(std::max(aabb1.pmax.x, aabb2.pmax.x), std::max(aabb1.pmax.y, aabb2.pmax.y), std::max(aabb1.pmax.z, aabb2.pmax.z));
    return AABB(pmin, pmax);
}

AABB triangle_aabb(const triangle &t) {
    Vector3 pmin = Vector3(std::min(std::min(t.p1.x, t.p2.x), t.p3.x), std::min(std::min(t.p1.y, t.p2.y), t.p3.y), std::min(std::min(t.p1.z, t.p2.z), t.p3.z));
    Vector3 pmax = Vector3(std::max(std::max(t.p1.x, t.p2.x), t.p3.x), std::max(std::max(t.p1.y, t.p2.y), t.p3.y), std::max(std::max(t.p1.z, t.p2.z), t.p3.z));
    return AABB(pmin, pmax);
}