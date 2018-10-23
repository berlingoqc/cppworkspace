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

void TstGripper()
{
	bool YouBotHasArm = false;
	bool GripperIsOpenState = false;

	YouBotManipulator *MyYouBotManipulator = 0;
	MyYouBotManipulator = reelYB_ArmInit();
	reelYB_GripperInit(MyYouBotManipulator);
	char Key = getchar();
	while (Key != 'Q' && Key != 'q')
	{
	
		if (!GripperIsOpenState)
		{
			reelYB_GripperOpen(MyYouBotManipulator, true);
		}
		else
		{
			reelYB_GripperClose(MyYouBotManipulator, true);
		}
		GripperIsOpenState = !GripperIsOpenState;
		Key = getchar(); // Read the enter 
		Key = getchar();
	}
	reelYB_GripperClose(MyYouBotManipulator, true);
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

// Valeur qui correspond a quand le lazer ne touche rien
#define ENDVALUELAZER 5.4f
// Valeur de la résolution angulaire du lazer
#define RES_ANG 0.3515625f

// Le nombre d'index que je saute dans le lazer au début et a la fin pour me crée un zone de 180 d au lieu de 240
int OFFSETLAZER = 86;
// Le nombre de valeur que je skip dans le lazer quand je regarde s'il y a des objets
int OFFSETVALIDATION = 3;

// Les différents états que le robot peut avoir 
enum ROBOT_STATE { SEARCHING_OBJECT , REACHING_OBJECT, ALLIGNEMENT_OBJECT, TASK_OBJECT, END_SEQUENCE };
enum CADRAN { RIGHT_CADRAN, LEFT_CADRAN };

struct Vec {
	float 	x;
	float 	y;

	float 	angle;
};

struct Line {
	float p1;
	float p2;
};


int CurrentState;

void scanLaserForObject() {
	float lazerscan[LASERSIZE];
	if(!GetLaserData(lazerscan,0.0f,0.0f)) {
		printf("Echec de lecture sur les données du lazer\n");
	}

	// les valeurs pour le début et la fin de l'object trouvé avec un nombre de valeur null de 2
	float start = 0.0f;
	float startIndex = 0;
	float end = 0.0f;;
	float endIndex = 0;

	for(int i = OFFSETLAZER; i < (LASERSIZE - OFFSETLAZER);i += 3) {
		if(lazerscan[i] >=0 && lazerscan[0] != ENDVALUELAZER) {
			if(start == 0.0) { // si on na pas trouver le depart on le met comme étant
				start = lazerscan[i];
				startIndex = i;
			}
			end = lazerscan[i]; // met la fin a cette distance
			endIndex = i;
		}
	}


	// si la distance sur l'axe des x du point initial est supperieur a un mettre, on va se déplacer l'attéralement




}




int main(int argc, char **argv)
{

  	// Arm, base and gripper initialization
  	YouBotBase *MyYouBotBase=0;
  	reelYB_Init(MyYouBotBase);
  	MyYouBotBase=reelYB_BaseInit();
  	if(MyYouBotBase==0)
    	return -1;
     
  	YouBotManipulator *MyYouBotManipulator=0;
  	MyYouBotManipulator=reelYB_ArmInit();
  	if(MyYouBotManipulator==0)
      	return -1;
  
  	reelYB_GripperInit(MyYouBotManipulator);

	OpenLaser();


	// La phase de recherche principal
	CurrentState = SEARCHING_OBJECT;
	while(CurrentState != END_SEQUENCE) {


	}

	// Execute la séquence finale


  	// Exit the robot
	reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
	reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleFoetalPosition, true);
 	reelYB_ExitBase(MyYouBotBase);
  	reelYB_ExitArm(MyYouBotManipulator);
	CloseLaser();


  	return 0;
}
