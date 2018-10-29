#ifndef MILLEFEUILLE_H
#define MILLEFEUILLE_H

#include "cvheaders.h"

enum MilleFeuilleView { TOP_VIEW, SIDE_VIEW };

const char* toString(const MilleFeuilleView v) {
    switch (v) {
        case TOP_VIEW : return "TOP_VIEW";
        case SIDE_VIEW : return "SIDE_VIEW";
    }
}

enum MilleFeuilleState { NOT_CRUSTAD_STATE, NOT_TOP_STATE, NOT_CREME_STATE };


const char* toString(const MilleFeuilleState v) {
    switch (v) {
        case NOT_CRUSTAD_STATE: return "NO CRUSTAD";
        case NOT_TOP_STATE : return "NO TOP";
        case NOT_CREME_STATE: return "NO CREAM";
    }
}

struct millefeuille_image {
    cv::Mat 	img;
    std::string filename;
    int 		view;
    int			state;
};

typedef std::vector<millefeuille_image> Millefeuille_Img_list;


#endif // MILLEFEUILLE_H
