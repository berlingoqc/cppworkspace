#include "house_maker.h"
#include "engine.h"


using namespace House;
using namespace ENGINE;


House_Maker::House_Maker(void) {

}

House_Maker::~House_Maker(void) {

}


// FONCTION PUBLIC

// Crée la base de la maison 
void House_Maker::createBase(float largeur, float hauteur, float profondeur, glm::vec3 couleur) {
	m_hauteurBase = hauteur / 2.0f;
	creeSommetsBase(largeur / 2.0f, hauteur / 2.0f, profondeur / 2.0f);
	creeCouleursBase(couleur);

	uint IBOBase;
	genIBOBuffer(&IBOBase, 1, sizeof(m_indicesBase), m_indicesBase);
}

void House_Maker::createCeiling(float largeur, float hauteur, float profondeur, glm::vec3 couleur) {
	creeSommetsToit(largeur / 2.0f, hauteur / 2.0f, profondeur / 2.0f);
	creeCouleursToit(couleur);
	
	uint IBOToit;
	genIBOBuffer(&IBOToit, 1, sizeof(m_indicesToit), m_indicesToit);
}

void House_Maker::createFloor(float largeur, float profondeur, glm::vec3 couleur) {
	creeSommetsSol(largeur / 2.0f, profondeur / 2.0f);
	creeCouleursSol(couleur);
}


// FONCTION PRIVÉE


void House_Maker::creeSommetsBase(float largeur, float hauteur, float profondeur) {

}

void House_Maker::creeCouleursBase(glm::vec3 couleur) {

}

void House_Maker::creeSommetsToit(float largeur, float hauteur, float profondeur) {

}

void House_Maker::creeCouleursToit(glm::vec3 couleur) {

}

void House_Maker::creeSommetsSol(float largeur, float profondeur) {

}

void House_Maker::creeCouleursSol(glm::vec3 couleur) {

}