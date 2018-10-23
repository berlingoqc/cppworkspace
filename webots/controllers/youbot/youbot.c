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

	reelYB_MoveArmAndBaseByKeyboard(MyYouBotManipulator);

  /*
  // Move arm and robot       
  passive_wait(2.0);
  float Distance =-10.0;
  float Speed =2.0;
 */
  // passive_wait(5.0);
/*  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, false);
  reelYB_MoveBaseTransversal(MyYouBotBase,  Distance, Speed, true,-1);
  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick1, false);
  reelYB_MoveBaseAngular(MyYouBotBase,  -90, Speed*5, true,-1);
  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAnglePick1, false);
  reelYB_MoveBaseLongitudinal(MyYouBotBase,  Distance, Speed, true,-1);
  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleAproPick1, false);
  reelYB_MoveBaseLongitudinal(MyYouBotBase, -Distance, Speed, true, -1);
  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, false);
  reelYB_MoveBaseAngular(MyYouBotBase, 90, Speed * 5, true, -1);
  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleFoetalPosition, false);
  reelYB_MoveBaseTransversal(MyYouBotBase,-Distance, Speed, true, -1);
*/
 
 // reelYB_ArmSetPosition(MyYouBotManipulator,ArmPositionbloc1[1],Speed,true,-1);
 /*int i=0;
 wb_robot_keyboard_enable(TIME_STEP);
 for(i=0;i<8;i++)
 { 
  reelYB_ArmSetPosition(MyYouBotManipulator,ArmPositionbloc1[i],Speed,true,-1); 
  wb_robot_keyboard_get_key();
  }
  */
  //PickAndPlace(MyYouBotManipulator, 1);
//  PickAndPlace(MyYouBotManipulator, 2);
//  PickAndPlace(MyYouBotManipulator, 3);


  // Exit the robot
//  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleReadyPosition, true);
//  reelYB_ArmSetAngle(MyYouBotManipulator, ArmAngleFoetalPosition, true);
  reelYB_ExitBase(MyYouBotBase);
  reelYB_ExitArm(MyYouBotManipulator);
  


  return 0;
}
