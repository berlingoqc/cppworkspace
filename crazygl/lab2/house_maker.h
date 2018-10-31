#ifndef _HOUSE_MAKER_H_
#define _HOUSR_MAKER_H_


#include <header.h>

namespace House {


	class House_Maker {
		public:
			House_Maker(void);
			~House_Maker(void);

			void createBase(float largeur, float hauteur, float profondeur, glm::vec3 couleur);
			void createCeiling(float largeur, float hauteur, float profondeur, glm::vec3 couleur);
			void createFloor(float largeur, float profondeur, glm::vec3 couleur);

		private:

			void creeSommetsBase(float largeur, float hauteur, float profondeur);
			void creeCouleursBase(glm::vec3 couleur);

			void creeSommetsToit(float largeur, float hauteur, float profondeur);
			void creeCouleursToit(glm::vec3 couleur);

			void creeSommetsSol(float largeur, float profondeur);
			void creeCouleursSol(glm::vec3 couleur);

		private:
			float m_hauteurBase;
			glm::vec3 m_sommetsBase[12];
			glm::vec3 m_couleursBase[12];
			const unsigned int m_indicesBase[42] = {
				0,1,2,
				1,2,3,
				4,5,6,
				5,6,7,
				2,6,7,
				2,3,7,
				0,4,5,
				0,1,5,
				1,5,7,
				1,3,7,
				0,2,4,
				2,4,6,
				8,9,10,
				8,10,11
			};

			glm::vec3 m_sommetsToit[6];
			glm::vec3 m_couleursToit[6];
			const unsigned int m_indicesToit[18] = {
				0,1,2,
				3,4,5,
				0,5,2,
				0,3,5,
				0,4,1,
				0,3,4
			};

			glm::vec3 m_sommetsSol[4];
			glm::vec3 m_couleursSol[4];
	};
};

#endif 