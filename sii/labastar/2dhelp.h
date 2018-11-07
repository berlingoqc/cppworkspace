#ifndef _2D_HELP_H_
#define _2D_HELP_H_

#include <vector>

class Ptn {
    float   x;
    float   y;

public:
    Ptn();
    Ptn(float x,float y);
};

Ptn PtnMoveByVector(const Ptn& p, const Ptn& v, float n);

bool areIntersecting(const Ptn& l1p1, const Ptn& l1p2, const Ptn& l2p1, const Ptn& l2p2);
Ptn getIntersectionPtn(const Ptn& l1p1, const Ptn& l1p2, const Ptn& l2p1, const Ptn& l2p2);
float distanceBetweenPtns(const Ptn& p1, const Ptn& p2);


#endif
