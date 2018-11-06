#include "object.h"


 Object::Object(std::vector<Point*> pts) {
     pts = cleanupPointLists(pts);
     if(pts.size() <= 2) {
         for(int i = 0; i < pts.size(); i++) {
             nodeList.push_back(new Node(pts[i], new Point(0.0f,0.0f)));
         }
         return;
     }
     for(int i = 0; i< pts.size(); i++) {
         Point* temp = getAngleVectorFromPoints(pts[i], pts[(i+1) % pts.size()]);
     }
 }

 Object::~Object() {

 }

        
std::vector<Point*> Object::cleanupPointLists(std::vector<Point*> pnodelist) {

}

std::vector<Point*> Object::removeNodes(std::vector<Point*> pnodeList, Point* newPoint, int startIndex, int endIndex) {

}

void Object::deletePolygon() {
    
}

void Object::deleteNode() {

}

void Object::deletePoint() {

}

Point* Object::getAngleVectorFromPoints(Point* previous, Point* current, Point* next) {

}