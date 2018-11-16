#include "2dhelp.h"

Ptn::Ptn() {
	x = y = 0;
}
Ptn::Ptn(float x, float y) {
	this->x = x;
	this->y = y;
}

Ptn PtnMoveByVector(const Ptn* p, const Ptn* v, float n) {

	return Ptn();
}

bool areIntersecting(const Ptn* l1p1, const Ptn* l1p2, const Ptn* l2p1, const Ptn* l2p2) {

	return false;
}
Ptn getIntersectionPtn(const Ptn* l1p1, const Ptn* l1p2, const Ptn* l2p1, const Ptn* l2p2) {

	return Ptn();
}

float distanceBetweenPtns(const Ptn* p1, const Ptn* p2) {
	float deltaX = (p1->x - p2->x);
	float deltaY = (p1->y - p2->y);
	return sqrt(deltaX*deltaX + deltaY * deltaY);
}