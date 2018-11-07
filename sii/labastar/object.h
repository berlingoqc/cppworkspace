#ifndef _OBJECT_H_
#define _OBJECT_H_

#include <stdarg.h>
#include <vector>
#include "node.h"
#include "2dhelp.h"

class Object {
    private:
        std::vector<Node*> nodeList;
    public:
        Object(std::vector<Ptn*> pts);
        ~Object();

        

        static std::vector<Ptn*> cleanupPtnLists(std::vector<Ptn*> pnodelist);
        static std::vector<Ptn*> removeNodes(std::vector<Ptn*> pnodeList, Ptn* newPtn, int startIndex, int endIndex);

        void deletePolygon();
        void deleteNode();
        void deletePtn();
    
    private:
        Ptn* getAngleVectorFromPtns(Ptn* previous, Ptn* current, Ptn* next);
};

#endif
