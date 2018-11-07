#include "object.h"


 Object::Object(std::vector<Ptn*> pts) {
     pts = cleanupPtnLists(pts);
     if(pts.size() <= 2) {
         for(int i = 0; i < pts.size(); i++) {
         }
         return;
     }
 }

 Object::~Object() {

 }

        
std::vector<Ptn*> Object::cleanupPtnLists(std::vector<Ptn*> pnodelist) {

}

std::vector<Ptn*> Object::removeNodes(std::vector<Ptn*> pnodeList, Ptn* newPtn, int startIndex, int endIndex) {

}

void Object::deletePolygon() {
    
}

void Object::deleteNode() {

}

void Object::deletePtn() {

}

Ptn* Object::getAngleVectorFromPtns(Ptn* previous, Ptn* current, Ptn* next) {

}