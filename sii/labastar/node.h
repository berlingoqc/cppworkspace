#ifndef _NODE_H_
#define _NODE_H_

#include <opencv2/core/utility.hpp>

int distanceBetweenPoint(cv::Point p1, cv::Point p2) {
	int deltaX = p1.x - p2.x;
	int deltaY = p1.y - p2.y;
	return sqrt((deltaX*deltaX + deltaY * deltaY));
}
class Node {
	cv::Point	index;
	cv::Point	central_point;
	cv::Point	size;

	int	fcost, hcost, gcost;

	bool visited;

	bool have_obstacle;

	bool is_open_list;
	bool is_close_list;

	Node*		parent;

public:
	Node()
	{
		this->central_point = {0,0};
		this->size = {0,0};
		parent = nullptr;
		visited = false;
		fcost = 0;
		hcost = 0;
		gcost = 0;
		have_obstacle = false;
		is_open_list = false;
		is_close_list = false;
	}
	Node(cv::Point central_point, cv::Point size, Node* parent = nullptr) {
		this->central_point = central_point;
		this->size = size;
		this->parent = nullptr;
		visited = false;
		fcost = 0;
		hcost = 0;
		gcost = 0;
		have_obstacle = false;
		is_open_list = false;
		is_close_list = false;
	}

	void setConfig(cv::Point central_point, cv::Point size, cv::Point index)
	{
		this->index = index;
		this->central_point = central_point;
		this->size = size;
		parent = nullptr;
		visited = false;
		fcost = 0;
		hcost = 0;
		gcost = 0;
		have_obstacle = false;
		is_open_list = false;
		is_close_list = false;
	}

	cv::Point getTopLeft()
	{
		return { this->central_point.x - this->size.x / 2, this->central_point.y - this->size.y / 2 };
	}

	cv::Point getCentralPoint() {
		return central_point;
	}

	int calculateCost(const Node* end, Node* potentiel_parent) {
		float g, h, f;
		h = distanceBetweenPoint(central_point, end->central_point);
		if (parent == nullptr) {
			g = 0;
		}
		else {
			g = potentiel_parent->getGCost() + distanceBetweenPoint(potentiel_parent->central_point, central_point);
		}

		f = g + h;
		if (parent == nullptr || (f < fcost)) {
			parent = potentiel_parent;
			hcost = h;
			fcost = f;
			gcost = g;
		}
		hcost = h;
		fcost = f;
		gcost = g;
		return fcost;
	}

	cv::Point getSize() const
	{
		return size;
	}

	Node* getParent() {
		return parent;
	}

	void setParent(Node* node)
	{
		parent = node;
	}

	int getHCost() {
		return hcost;
	}

	int getFCost() {
		return fcost;
	}

	int getGCost() {
		return gcost;
	}

	bool getVisited() {
		return visited;
	}

	void setVisisted(bool v) {
		visited = v;
	}
	bool getIsOpenList() {
		return is_open_list;
	}

	void setIsOpenList(bool v) {
		is_open_list = v;
	}	
	
	bool getIsCloseList() {
		return is_close_list;
	}

	void setIsCloseList(bool v) {
		is_close_list = v;
	}

	bool getHaveObstacle()
	{
		return have_obstacle;
	}

	void setHaveObstacle(bool v)
	{
		have_obstacle = v;
	}

	int getIndexX() { return index.x; }
	int getIndexY() { return index.y; }
};

#endif