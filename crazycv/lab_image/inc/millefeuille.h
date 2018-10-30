#ifndef MILLEFEUILLE_H
#define MILLEFEUILLE_H

#include "cvheaders.h"

enum MilleFeuilleView { TOP_VIEW, SIDE_VIEW };

enum MilleFeuilleState { NOT_CRUSTAD_STATE, NOT_TOP_STATE, NOT_CREME_STATE };

struct millefeuille_image {
    cv::Mat 	img;
    std::string filename;
    int 		view;
    int			state;
};

typedef std::vector<millefeuille_image> Millefeuille_Img_list;


#endif // MILLEFEUILLE_H
