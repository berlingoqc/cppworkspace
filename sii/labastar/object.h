#ifndef _OBJECT_H_
#define _OBJECT_H_

#include <stdarg.h>
#include <vector>
#include "node.h"

using namespace PathFinding;

class Object {
    private:
        std::vector<Node*> nodeList;
    public:
        Object(std::vector<Point*> pts);
        ~Object();

        

        static std::vector<Point*> cleanupPointLists(std::vector<Point*> pnodelist);
        static std::vector<Point*> removeNodes(std::vector<Point*> pnodeList, Point* newPoint, int startIndex, int endIndex);

        void deletePolygon();
        void deleteNode();
        void deletePoint();
    
    private:
        Point* getAngleVectorFromPoints(Point* previous, Point* current, Point* next);
};

#endif
