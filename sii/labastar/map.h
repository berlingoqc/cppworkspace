#ifndef MAP_H
#define MAP_H


typedef std::vector<std::vector<cv::Point>> Contours;
typedef std::vector<cv::Vec4i> Hierarchy;

enum map_iterator_state {
	CAN_NEXT,
	CANT_NEXT,
	FOUND_END
};

enum color_point
{
	RED,
	GREEN,
	BLACK
};

enum state_node
{
	EXPLORE,
	UNKNOW,
	OBJECT
};

cv::Scalar	colors_point[3]{ {0,0,255}, {0,255,0}, {0,0,0} };

class Map {
	int					width_node_size;
	int					height_node_size;

	int cols;
	int rows;

	std::vector<Node*>	liste_ouverte;
	std::vector<Node*>  liste_fermer;

	Node*				start;
	Node*				end;

	Node**				node_map;

	Contours			contours;
	Hierarchy			hierarchy;

	cv::Mat				original_image;


public:
	Map(cv::Size robot_size, const cv::Mat& capture) {
		original_image = capture;

		start = nullptr;
		end = nullptr;

		rows = capture.rows / robot_size.height;
		cols = capture.cols / robot_size.width;

		width_node_size = original_image.cols / cols;
		height_node_size = original_image.rows / rows;

		node_map = new Node*[rows];
		for (int y = rows-1; y >= 0; y--)
		{
			node_map[y] = new Node[cols];
			for (int x = cols-1; x >= 0; x-- )
			{
				Node* n = (&node_map[y][x]);
				n->setConfig({ x*width_node_size + width_node_size / 2, y*height_node_size + height_node_size / 2 }, { width_node_size,height_node_size }, { x,y });
			}
		}

	}

	void setObstacle(Contours contours, Hierarchy hierarchy) {
		this->contours = contours;
		this->hierarchy = hierarchy;
	}

	cv::Mat get_image_info() {
		cv::Mat ret = original_image.clone();
		for (int i = width_node_size; i < original_image.cols; i += width_node_size) {
			cv::line(ret, cv::Point(i, 0), cv::Point(i, original_image.rows), (0, 0, 0));
		}

		for (int i = height_node_size; i < original_image.rows; i += height_node_size) {
			cv::line(ret, cv::Point(0, i), cv::Point(original_image.cols, i), (0, 0, 0));
		}
		if(start != nullptr)
		{
			cv::circle(ret, start->getCentralPoint(), 5, colors_point[BLACK],-1);
		}
		// si on n'a la fin on le dessine
		if (end != nullptr) {
			cv::circle(ret, end->getCentralPoint(), 5, colors_point[BLACK],-1);
		}

		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				Node* n = &node_map[y][x];
				if (n->getIsCloseList())
					draw_node(ret, n, RED);
				else if (n->getIsOpenList())
					draw_node(ret, n, GREEN);
				else if (n->getHaveObstacle())
					cv::circle(ret, n->getCentralPoint(), 6, colors_point[BLACK], -1);
			}
		}
		return ret;
	}

	void draw_node(cv::Mat& m, Node* node, color_point color) const {
		char text[50];
		cv::Point p;
		if (node->getParent() == nullptr) {
			p = cv::Point(-1, -1);
		}
		else {
			p = node->getParent()->getCentralPoint();
		}
		cv::Point tl = node->getTopLeft();
		// Get le coin superieur gauche du node
		sprintf_s(text, "%d,%d -> %d,%d", node->getCentralPoint().x, node->getCentralPoint().y, p.x, p.y);
		cv::putText(m, text, cv::Point(tl.x + 5, tl.y + 15), cv::FONT_HERSHEY_COMPLEX, 0.5, { 0,0,255 }, 1);
		sprintf_s(text, "F -> %d", node->getFCost());
		cv::putText(m, text, cv::Point(tl.x + 5, tl.y + 30), cv::FONT_HERSHEY_COMPLEX, 0.5, { 0,0,255 }, 1);
		sprintf_s(text, "H -> %d", node->getHCost());
		cv::putText(m, text, cv::Point(tl.x + 5, tl.y + 45), cv::FONT_HERSHEY_COMPLEX, 0.5, { 0,0,255 }, 1);
		sprintf_s(text, "G -> %d", node->getGCost());
		cv::putText(m, text, cv::Point(tl.x + 5, tl.y + 60), cv::FONT_HERSHEY_COMPLEX, 0.5, { 0,0,255 }, 1);

		cv::circle(m, node->getCentralPoint(), 5, colors_point[color], -1);

	}

	void setStart(Node* start)
	{
		this->start = start;
		this->start->setVisisted(true);
	}

	void setTarget(Node* end) {
		this->start->setIsOpenList(true);
		liste_ouverte.push_back(this->start);
		this->end = end;
	}

	Node* getSmallestCostOpenList() {
		Node* n = nullptr;
		for (auto i : liste_ouverte) {
			if (n == nullptr) {
				n = i;
				continue;
			}
			if (n->getFCost() > i->getFCost()) {
				// regarde s'il s'agit d'un mouvement en diagonal
				n = i;
			}
		}
		return n;
	}

	Node* getNodeFromPoint(cv::Point p) const {
		if(p.x < 0|| p.x > original_image.cols  || p.y < 0 || p.y > original_image.rows )
		{
			return nullptr;
		}
		int iX = p.x / width_node_size;
		int iY = p.y / height_node_size;
		return &node_map[iY][iX];
	}

	bool isThereObstacleInNode(const Node* node) {
		return false;
	}

	map_iterator_state iterate_next() {
		Node* n = getSmallestCostOpenList();
		if(n == end)
		{
			n->setIsOpenList(false);
			n->setIsCloseList(true);
			return FOUND_END;
		}
		liste_ouverte.clear();
		n->setIsOpenList(false);
		n->setIsCloseList(true);

		int start_y = n->getCentralPoint().y - n->getSize().y;
		int start_x = n->getCentralPoint().x - n->getSize().x;


		Node* e;
		for(int y = 0; y < 3;y++)
		{
			for (int x = 0; x < 3;x++)
			{
				e = getNodeFromPoint({ start_x + x * n->getSize().x, start_y + y * n->getSize().y });
				if(e != nullptr && !e->getVisited() && !e->getIsCloseList() && !e->getHaveObstacle())
				{
					e->setIsOpenList(true);
					e->setParent(n);
					e->calculateCost(end, n);
					liste_ouverte.push_back(e);
				}
				
			}
		}


		return CAN_NEXT;
	}



};



#endif 