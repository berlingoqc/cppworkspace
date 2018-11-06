#ifndef _2D_HELP_H_
#define _2D_HELP_H_

#include <vector>

namespace PathFinding {
class Point {
    float   x;
    float   y;

public:
    Point();
    Point(float x,float y);
};

Point pointMoveByVector(const Point& p, const Point& v, float n);

bool areIntersecting(const Point& l1p1, const Point& l1p2, const Point& l2p1, const Point& l2p2);
Point getIntersectionPoint(const Point& l1p1, const Point& l1p2, const Point& l2p1, const Point& l2p2);
float distanceBetweenPoints(const Point& p1, const Point& p2);

}

#endif
