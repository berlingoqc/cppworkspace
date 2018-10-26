/*
 * File:          youbot.c
 * Date:          24th May 2011
 * Description:   Starts with a predefined behaviors and then
 *                read the user keyboard inputs to actuate the
 *                robot
 * Author:        fabien.rohrer@cyberbotics.com
 * Modifications: 
 */

#include "YoubotWebotsAllInclude.h"


// Valeur qui correspond a quand le lazer ne touche rien
#define ENDVALUELAZER 5.600000f

// Valeur de la résolution angulaire du lazer
#define RES_ANG 0.3515625f

// Définit le nombre d'item maximum detecter par le lazer
#define LAZER_ITEMS_DETECTION 1

// Definit de la distance a laquelle le robot va commencer la face d'approche
#define DISTANCE_DOCKING 1000.0f

// Constance pour pi qui est deja definit dans math.h mais faut inclure les defines pis ca me tente pas
#define M_PI 3.14159265358979323846f

// Le nombre d'index que je saute dans le lazer au début et a la fin pour me crée un zone de 180 d au lieu de 240
#define OFFSETLAZER 86
// Le nombre de valeur que je skip dans le lazer quand je regarde s'il y a des objets
#define OFFSETVALIDATION 3


// Définit des mes enum avec un instruction de preproccesseur pour generer l'équivalent en string pour afficher le debug

#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRINGS(STRING) #STRING,


#define FOREACH_DIRECTION(DIRECTION) \
		DIRECTION(INC) \
		DIRECTION(DEC) \
		DIRECTION(STABLE) \
		DIRECTION(UNKNOW) \

#define FOREACH_ROBOTSTATE(ROBOTSTATE) \
		ROBOTSTATE(SEARCHING_OBJECT) \
		ROBOTSTATE(REACHING_OBJECT) \
		ROBOTSTATE(ALLIGNEMENT_OBJECT) \
		ROBOTSTATE(TASK_OBJECT) \
		ROBOTSTATE(END_SEQUENCE) \
		ROBOTSTATE(MANUAL_SEQUENCE) \

#define FOREACH_CADRAN(CADRAN) \
		CADRAN(LEFT_CADRAN) \
		CADRAN(RIGHT_CADRAN) \

enum CADRAN_ENUM {
	FOREACH_CADRAN(GENERATE_ENUM)
};

static const char* CADRAN_STRING[] = {
	FOREACH_CADRAN(GENERATE_STRINGS)
};

enum DIRECTION_ENUM {
	FOREACH_DIRECTION(GENERATE_ENUM)
};

static const char *DIRECTION_STRING[] = {
	FOREACH_DIRECTION(GENERATE_STRINGS)
};

enum ROBOTSTATE_ENUM {
	FOREACH_ROBOTSTATE(GENERATE_ENUM)
};

static const char *ROBOTSTATE_STRING[] = {
	FOREACH_ROBOTSTATE(GENERATE_STRINGS)
};



typedef struct {
	float 	x;
	float 	y;
} vec2f;

struct line{
	vec2f p1;
	vec2f p2;
};

struct lscan {
	float 	distanceStart;
	int 	indexStart;

	float	distanceEnd;
	int		indexEnd;
};


static const vec2f robotvector = {0,1};


// Retourne le vecteur qui sépare deux points
vec2f getVectorBetweenPoints(const vec2f* p1, const vec2f* p2) {
	return (vec2f){ p2->x - p1->x, p2->y - p1->y};
}

float getAngleBetweenVector(const vec2f* p1, const vec2f* p2) {
	return atan2(p2->x, p2->y) - atan2(p1->x, p1->y);
}

// retourne l'angle du point dans le demi cercle du lazer a partir de l'index de la valeur
float getAngleFromIndex(int index) {
	return (index * RES_ANG) - 30.0f;
}

// convertit un angle en degree vers radian
float toRad(float angle) {
	return angle * (M_PI / 180);
}

// effectue le sin d'un angle en degree
float sinD(float angle) {
	return sin(toRad(angle));	
}

// effectue le cos d'un angle en degree
float cosD(float angle) {
	return cos(toRad(angle));
}


// retourne le vecteur qui nous sépare du point
vec2f getVectorToPoint(float distance, float angle) {
	vec2f v;
	v.x = cosD(angle) * distance;
	v.y = sinD(angle) * distance;
	return v;
}



// Get le vecteur qui nous sépare avec un item du scan
vec2f getVectorToPointFromScan(float distance, int index) {
	float angle = getAngleFromIndex(index);
	int cadran = LEFT_CADRAN;
	if(90.0f < angle && angle < 180.0f) {
		cadran = RIGHT_CADRAN;
		angle = 180.0f - angle;
	}

	printf("Index %d Distance %f équivaut à l'angle %f dans le cadran %d \n",index,distance,angle,cadran);

	// Calcul les informations du vecteur qui représente la distance avec ce point
	vec2f v = getVectorToPoint(distance,angle);
	if(cadran == LEFT_CADRAN) {
		v.x = -1.0f * v.x; // si le point est dans le cadran droit met le x négatif
	}
	return v;
}

// Retourne la ligne formé par les deux points anssi que le vecteur qu'elle représente
struct line getLineFromScan(struct lscan itemScan) {
	struct line l;
	l.p1 = getVectorToPointFromScan(itemScan.distanceStart,itemScan.indexStart);
	l.p2 = getVectorToPointFromScan(itemScan.distanceEnd,itemScan.indexEnd);
	return l;
}


// scanLaserForObject recoit le pointeur pour la structure lscan qui contient l'index et la distance du début et de la premiere face d'un object trouver
int scanLaserForObject(struct lscan* items) {
	float lazerscan[LASERSIZE];
	if(!GetLaserData(lazerscan,0.0f,0.0f)) {
		printf("Echec de lecture sur les données du lazer\n");
	}

	// les valeurs pour le début et la fin de l'object trouvé avec un nombre de valeur null de 2
	struct lscan scan = { 0.0f, 0, 0.0f, 0 };
	int direction = UNKNOW;
	int i;
	// Cherche pour un segment de point en ligne qui formerait une ligne
	for(i = OFFSETLAZER; i < (LASERSIZE - OFFSETLAZER);i += 3) {
		if(lazerscan[i] >=0.0f && lazerscan[i] < ENDVALUELAZER) {
			if(scan.distanceStart == 0.0) { // si on na pas trouver le depart on le met comme étant
				scan.distanceStart = lazerscan[i];
				scan.indexStart = i;
				continue;
			} else if (scan.distanceEnd == 0.0) {
				// premiere assignation de la variable de fin
				// on regarde le sens entre la variable de debut et de fin
				if(lazerscan[i] < scan.distanceStart) {
					// la valeur diminue a chaque itération
					direction = DEC;
				} else if (scan.distanceStart < lazerscan[i]) {
					direction = INC;
				} else {
					// Les valeurs sont égales donc on n'attend pour définir la direction
					direction = STABLE;
				}
				scan.distanceEnd = lazerscan[i]; // met la fin a cette distance
				scan.indexEnd = i;
				continue;
			}
			if(direction == DEC && scan.distanceEnd < lazerscan[i]) {
				// supposé descendre mais sa augmente donc on quiite la boucle
				printf("Value sont supposé descendre mais %f > %f \n",lazerscan[i],scan.distanceEnd);
				break;
			} else if (direction == INC && lazerscan[i] < scan.distanceEnd) {
				// supposé augmenter mais sa descend donc on quitte la boucle
				printf("Value sont supposé descendre mais %f < %f \n",lazerscan[i],scan.distanceEnd);
				break;
			}
		
			scan.distanceEnd = lazerscan[i]; // met la fin a cette distance
			scan.indexEnd = i;
		} else {
			// si on n'a deja trouver un point et que l'a on trouve plus rien on n'arrête la boucle
			if(scan.distanceStart != 0.0) {
				printf("Fin de la séquence de point a %d fin de la boucle lazer\n",i);
				break;
			}	
		}
	}

	if(scan.distanceStart == 0.0f || scan.distanceEnd == 0.0f) {
		printf("Aucun item de detecter avec le lazer\n");
		return 0;
	}

	printf("Object detecter entre %d %f et %d %f\n",scan.indexStart,scan.distanceStart,scan.indexEnd,scan.distanceEnd);

	items[0] = scan;
	return 1;
}

bool init_reelYB(YouBotBase* MyYouBotBase, YouBotManipulator* MyYouBotManipulator) {
	// Arm, base and gripper initialization
  	reelYB_Init(MyYouBotBase);
  	MyYouBotBase=reelYB_BaseInit();
  	if(MyYouBotBase==0)
    	return false;

  	MyYouBotManipulator=reelYB_ArmInit();
  	if(MyYouBotManipulator==0)
    	return false;

  
  	reelYB_GripperInit(MyYouBotManipulator);
	
	if(!OpenLaser()) {
		printf("Echer a l'ouverture du lazer\n");
		return false;
	}

	return true;
}

void end_reelYB(YouBotBase* MyYouBotBase, YouBotManipulator* MyYouBotManipulator) {
	CloseLaser();
  	reelYB_ExitBase(MyYouBotBase);
  	reelYB_ExitArm(MyYouBotManipulator);
}




int robotLoop() {
	// La boucle de recherche principal
	YouBotBase* base=0;
	YouBotManipulator* manipulator=0;
	if(!init_reelYB(base,manipulator)) {
		printf("Echer dans la séquence d'initialisation du reelYB\n");
		return -1;
	}

	printf("Démarrage de la boucle principal en mode REACHING_OBJECT\n");

	bool make_scan = true;
	int nbrItemScan;	
	int CurrentState = SEARCHING_OBJECT; // Initialize la boucle a la premiere sequence du program
	struct lscan scan[LAZER_ITEMS_DETECTION]; // Initialize un array pour les données de scan

	int nbrTryFindBox = 0;

	while(CurrentState != END_SEQUENCE) { // Boucle jusqu'a t'en qu'on soit rendu a la sequence

		if(make_scan) {
			nbrItemScan = scanLaserForObject(scan);
		}

		switch (CurrentState) {

			case SEARCHING_OBJECT: // Séquence de recherche de l'emplacement de l'item dans l'environnement
				// Regarde s'il a trouver un object en avant de lui
				if(0 < nbrItemScan) {
					CurrentState = REACHING_OBJECT;
					printf("Séquence de recherche terminer, début séquence REACHING_OBJECT\n");
					continue;
				}
				if(nbrTryFindBox == 3) {
					// Apres 3 rotation de 90 degree si on n'a encore rien trouvé on passe en mode manuel
					CurrentState = MANUAL_SEQUENCE;
					printf("3 rotation et aucune boite de trouvé, passage en mode manuel\n");
					continue;
				}
				
				// Effectue une rotation de 90 degree et regarde si on trouve dequoi
				reelYB_MoveBaseAngular(base,90,2.0f,true,-1);
				nbrTryFindBox = nbrTryFindBox + 1;
	
			break;
			case REACHING_OBJECT: // Séquence d'approche sommaire de l'object pour distance supérieur a un metre
			{
				
				// regarde si la distance du point est plus d'un metre
				if(scan[0].distanceStart <= 1.5f) {
					// Object a courte distance on passe en mode allignement de l'object
					CurrentState = ALLIGNEMENT_OBJECT;
					printf("L'object est a moins d'un mètre changement en mode d'allignement Distance : %f \n", scan[0].distanceStart);
					continue;
				}

				printf("Item a plus d'un mètre %f \n",scan[0].distanceStart);
				// si le point est loin on va juste s'avancer dans l'axe le plus lointain des deux (x ou y)
				vec2f vectorToPoint = getVectorToPointFromScan(scan[0].distanceStart,scan[0].indexStart);
				// Regarde si l'object est plus loin sur l'axe des x que sur l'axe des y
				printf("Le vecteur de distance avec le point est X : %f Y : %f \n",vectorToPoint.x,vectorToPoint.y);
				float f = 1.0f;
				if(vectorToPoint.x < 0) {
					f = -1.0f;
				}
				if( 1.0f < vectorToPoint.x * f) {
					// Effectue un mouvement transversal dans la direction
					float distance = (vectorToPoint.x*f - 0.8f) * 100.0f;	
					printf("Mouvement transversal pour s'approcher sur l'axe des x\n");

					reelYB_MoveBaseTransversal(base,distance*f,2.0f,true,-1);
				} else {
					// calcul la distance en cm a avancer pour ce retrouver a un metre envirron
					float distance = (scan[0].distanceStart - 0.8f) * 100.0f;
					
					// Avance vers le point de 20 cm
					reelYB_MoveBaseLongitudinal(base,20,2.0f,true,-1);
				}
			}
			break;
			case ALLIGNEMENT_OBJECT: // Séquence d'allignement avec le centre de l'object
				printf("Entrer sequence allignement\n");
				// Va chercher la ligne formé par nos deux points
				struct line l = getLineFromScan(scan[0]);
				printf("J'ai ma ligne je veut le vecteur\n");
				// Va chercher le vectuer qui est former par cette ligne entre ses deux points
				vec2f vLine = getVectorBetweenPoints(&l.p1,&l.p2);
				printf("J'ai mon vecteur %f %f\n",vLine.x,vLine.y);
				
				// Calcul l'angle entre cette ligne et mon vecteur de deplacment de (0,1)
				float angleBetween = getAngleBetweenVector(&robotvector,&vLine);
				printf("Je mon angle %f\n",angleBetween);

				//printf("Phase d'approche ligne entre les points X1 : %f Y1 : %f X2 : %f Y2 : %f. Vecteur de X : %f Y : %f Angle entre %f",
				//	l.p1.x,l.p1.y,l.p2.x,l.p2.y,vLine.x,vLine.y,angleBetween);
				sleep(2000);
			break;
			case TASK_OBJECT: // Séquence de la tache a effectuer une fois rendu a l'object (mettre item sur la boite)

			break;
			case MANUAL_SEQUENCE: // Séquence quand le robot passe en controle manuel pour le déplacer et enssuite recommencer la séquence
				reelYB_MoveArmAndBaseByKeyboard(base);
				CurrentState = SEARCHING_OBJECT;
			break;
		}
		printf("CurrentState : %s",ROBOTSTATE_STRING[CurrentState]);

	}

	printf("Fin de la boucle de controle\n");
	end_reelYB(base,manipulator);
	return 0;
}


void PickAndPlace(YouBotManipulator *MyYouBotManipulator, int NumBlock)
{

	float Distance = -10.0;
	float Speed = 2.0;

	switch (NumBlock)
	{
		case 1:
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick1, true);
			reelYB_GripperOpen(MyYouBotManipulator, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAnglePick1, true);
			reelYB_GripperClose(MyYouBotManipulator,true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick1, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproDrop1, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleDrop1, true);
			reelYB_GripperOpen(MyYouBotManipulator, true);
			break;
		case 2:
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick2, true);
			reelYB_GripperOpen(MyYouBotManipulator, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAnglePick2, true);
			reelYB_GripperClose(MyYouBotManipulator, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick2, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproDrop2, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleDrop2, true);
			reelYB_GripperOpen(MyYouBotManipulator, true);
			break;
		case 3:
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick3, true);
			reelYB_GripperOpen(MyYouBotManipulator,true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAnglePick3, true);
			reelYB_GripperClose(MyYouBotManipulator, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick3, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproDrop3, true);
			reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleDrop3, true);
			reelYB_GripperOpen(MyYouBotManipulator, true);
			break;
	}



	reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);

	
}




int main(int argc, char **argv)
{
  	return robotLoop();
}