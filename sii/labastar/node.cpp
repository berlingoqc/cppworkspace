#include "node.h"

Node::Node() {
    position = new Ptn();
    size = new Ptn();
}

Node::Node(Ptn* centralPoint,Ptn* size) {
    position = centralPoint;
    this->size = size;

}
Node::~Node() {

}

void Node::deleteNode(){

}
void Node::deletePoint() {

}


bool Node::isVisited() {
    return visited;
}

void Node::setVisited(bool v) {
    visited = v;
}

float Node::getHCost() {
    return hcost;
}

float Node::getFCost() {
    return fcost;
}

float Node::getGCose() {
    return gcost;
}

Node* Node::getParent() {


    return nullptr;
}

float Node::calculateCost(Node* end, Node::AccessibleNode* potentialParent) {
	float g, h, f;
	h = distanceBetweenPtns(position, end->position);
	if (potentialParent == nullptr) {
		g = 0;
	}
	else {
		g = potentialParent->accessibleNode->getGCose() + distanceBetweenPtns(potentialParent->accessibleNode->position, position);
	}
	f = g + h;
	if (parent == nullptr || (f < fcost)) {
		if (potentialParent != nullptr) {
			parent = potentialParent->accessibleNode;
		}
		hcost = h;
		fcost = f;
		gcost = g;
	}
    return fcost;
}

void Node::createAccessibleNodeVect(NodeMap* map, float gapwidth){
	if (accessibleNodeVector.size() == 0) {
		updateAccessibleNodeVect(map,gapwidth);
	}
}
void Node::updateAccessibleNodeVect(NodeMap* map, float gapwidth) {
	for (int i = 0; i < accessibleNodeVector.size(); i++) {
		delete accessibleNodeVector[i];
	}
	accessibleNodeVector.clear();
	float diffWidth = gapwidth * 0.99f; // pour pas que ca pogne les gap entre

}


NodeMap::NodeMap(std::vector<Object> items, Ptn robot_size, float width, float height) {
	this->items = items;
	this->robot_size = robot_size;
	this->width = width;
	this->height = height;
}

std::vector<Node*> getComputerNodeList(Ptn robotsize, Ptn* startpoint, Ptn* endpoint) {
	// Démarre a notre point de départ , on crée les node
	Node* startNode = new Node(startpoint,&robotsize);
}