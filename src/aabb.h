#pragma once

#include <algorithm>

#include "builder.h"
#include "vector.h"
#include "triangle.h"


class AABB{
public:
    AABB(const Vector3 &pmin_, const Vector3 &pmax_) {
        pmin = Vector3(std::min(pmin_.x, pmax_.x), std::min(pmin_.y, pmax_.y), std::min(pmin_.z, pmax_.z));
        pmax = Vector3(std::max(pmin_.x, pmax_.x), std::max(pmin_.y, pmax_.y), std::max(pmin_.z, pmax_.z));
    }
    AABB(const AABB &aabb) : pmin(aabb.pmin), pmax(aabb.pmax) {}
    AABB() : pmin(Vector3(infinity(), infinity(), infinity())), pmax(Vector3(-infinity(), -infinity(), -infinity())) {}


    Vector3 pmin;
    Vector3 pmax;
};

double get_area(const AABB &aabb);

AABB merge(const AABB &aabb1, const AABB &aabb2);

AABB triangle_aabb(const triangle &t);