#ifndef MILLEFEUILLE_H
#define MILLEFEUILLE_H

#include "cvheaders.h"

enum MilleFeuilleView { TOP_VIEW, SIDE_VIEW };

enum MilleFeuilleError {  // Enum pour identifier les erreurs trouver lors du traitement
    NO_ERROR,
    TOO_MUTCH_CROUTE_ERROR, NO_CROUTE_FOUND_ERROR, INVALID_DISTANCE_CROUTE_ERROR, TOO_MUTCH_CROUTE_FOUND, // Erreur de la croute
    TOO_MUTCH_INTERFERENCE_IMG_ERROR // Erreur générique lié a l'image
};



enum MilleFeuilleState {  // Enum pour identifier le résultat du traitement du mille-feuille
    NOT_CRUSTAD_STATE, NOT_TOP_STATE, NOT_CREME_STATE, OK_STATE, TOO_SMALL_STATE, TOO_BIG_STATE 
};

struct millefeuille_image {
    std::string filename;
    int 		view;
    int			state;
};


struct filtre_image {
    thresh_range    range; // Range qu'on applique pour filtrer l'image


    bool            applyBlur;
    bool            applyMorphClose;
};

struct valid_croute {
    int             maxBruit;

    double          minArea;
    double          maxArea;

    float           minYDistanceMassCenter;
    float           maxYDistanceMassCenter;
};

typedef std::vector<millefeuille_image> Millefeuille_Img_list;


#endif // MILLEFEUILLE_H
