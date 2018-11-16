#ifndef _NODE_H_
#define _NODE_H_

#include "2dhelp.h"

class NodeMap;

class Node {
    public:
        struct AccessibleNode {
            AccessibleNode(Node* n,float distance) {
                accessibleNode = n;
                distanceFromSouce = distance;
            }
            Node* accessibleNode;
            float distanceFromSouce;
        };

        Node();
        Node(Ptn* centralPoint,Ptn* size);
        ~Node();

        void deleteNode();
        void deletePoint();

        Ptn*    position;
        Ptn*    size;

        std::vector<AccessibleNode*> accessibleNodeVector;

        bool    isVisited();
        void    setVisited(bool v);

        float getHCost();
        float getFCost();
        float getGCose();
        Node* getParent();

        float calculateCost(Node* end, Node::AccessibleNode* potentialParent = nullptr);

        void createAccessibleNodeVect(NodeMap* map, float gapwidth);
        void updateAccessibleNodeVect(NodeMap* map, float gapwidth);

    private:
        float   fcost, hcost, gcost;
        bool    visited;
        Node*   parent;
};

class NodeMap {
	std::vector<Object> items;

	Ptn					robot_size;
	float				width;
	float				height;

    public:
        NodeMap(std::vector<Object> items,Ptn robot_size,float width,float height);

        std::vector<Node*> getComputedNodeList(Ptn robotsize,Ptn* startpoint,Ptn* endpoint);
};

#endif