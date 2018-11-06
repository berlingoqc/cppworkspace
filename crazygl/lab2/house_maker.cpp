#include "house_maker.h"
#include "engine.h"


using namespace House;
using namespace ENGINE;
House::House_Maker::House_Maker(void)
{
}

House::House_Maker::~House_Maker(void)
{
}

void House_Maker::creeBase(GLfloat largeur, GLfloat hauteur, GLfloat profondeur, glm::vec3 couleur)
{
	m_hauteurBase = hauteur / 2;
	creeSommetsBase(largeur / 2, hauteur / 2, profondeur / 2);
	creeCouleursBase(couleur);

	GLuint IBOBase;
	glGenBuffers(1, &IBOBase);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBOBase);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indicesBase), m_indicesBase, GL_STATIC_DRAW);

}

void House::House_Maker::creeToit(GLfloat largeur, GLfloat hauteur, GLfloat profondeur, glm::vec3 couleur)
{
	creeSommetsToit(largeur / 2, hauteur / 2, profondeur / 2);
	creeCouleursToit(couleur);

	GLuint IBOToit;
	glGenBuffers(1, &IBOToit);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBOToit);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indicesToit), m_indicesToit, GL_STATIC_DRAW);
}

void House::House_Maker::creeSol(GLfloat largeur, GLfloat profondeur, glm::vec3 couleur)
{
	creeSommetsSol(largeur / 2, profondeur / 2);
	creeCouleursSol(couleur);
}


void House::House_Maker::creeSommetsBase(GLfloat largeur, GLfloat hauteur, GLfloat profondeur)
{
	m_sommetsBase[0] = glm::vec3(-largeur, hauteur, -profondeur);
	m_sommetsBase[1] = glm::vec3(largeur, hauteur, -profondeur);
	m_sommetsBase[2] = glm::vec3(-largeur, -hauteur, -profondeur);
	m_sommetsBase[3] = glm::vec3(largeur, -hauteur, -profondeur);
	m_sommetsBase[4] = glm::vec3(-largeur, hauteur, profondeur);
	m_sommetsBase[5] = glm::vec3(largeur, hauteur, profondeur);
	m_sommetsBase[6] = glm::vec3(-largeur, -hauteur, profondeur);
	m_sommetsBase[7] = glm::vec3(largeur, -hauteur, profondeur);
	m_sommetsBase[8] = glm::vec3(largeur / 3, -hauteur, -profondeur - 0.001); //porte
	m_sommetsBase[9] = glm::vec3(-largeur / 3, -hauteur, -profondeur - 0.001);
	m_sommetsBase[10] = glm::vec3(-largeur / 3, hauteur / 2, -profondeur - 0.001);
	m_sommetsBase[11] = glm::vec3(largeur / 3, hauteur / 2, -profondeur - 0.001);

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_sommetsBase), m_sommetsBase, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
}

void House::House_Maker::creeCouleursBase(glm::vec3 couleur)
{
	for (int i = 0; i < 4; i++)
	{
		m_couleursBase[i] = glm::vec3(1.0, 0.5, 0.0);
	}
	for (int i = 4; i < 8; i++)
	{
		m_couleursBase[i] = couleur;
	}
	for (int i = 8; i < 12; i++)
	{
		m_couleursBase[i] = glm::vec3(0.0, 0.5, 1.0);;
	}

	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_couleursBase), m_couleursBase, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

}

void House::House_Maker::creeSommetsToit(GLfloat largeur, GLfloat hauteur, GLfloat profondeur)
{
	m_sommetsToit[0] = glm::vec3(0.0, m_hauteurBase + hauteur, -profondeur);
	m_sommetsToit[1] = glm::vec3(-largeur, m_hauteurBase, -profondeur);
	m_sommetsToit[2] = glm::vec3(largeur, m_hauteurBase, -profondeur);;
	m_sommetsToit[3] = glm::vec3(0.0, m_hauteurBase + hauteur, profondeur);;
	m_sommetsToit[4] = glm::vec3(-largeur, m_hauteurBase, profondeur);;
	m_sommetsToit[5] = glm::vec3(largeur, m_hauteurBase, profondeur);;

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_sommetsToit), m_sommetsToit, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

}

void House::House_Maker::creeCouleursToit(glm::vec3 couleur)
{
	float r, g, b;
	glm::vec3 coul;


	for (int i = 0; i < 6; i++)
	{
		coul = couleur;
		r = rand() % 100 / 100.0;
		g = rand() % 100 / 100.0;
		b = rand() % 100 / 100.0;

		if (coul.r == 1.0)
			coul.r -= r;
		else
			coul.r += r;
		if (coul.g == 1.0)
			coul.g -= g;
		else
			coul.g += g;
		if (coul.b == 1.0)
			coul.b -= b;
		else
			coul.b += b;

		m_couleursToit[i] = coul;
	}

	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_couleursToit), m_couleursToit, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

}

void House::House_Maker::creeSommetsSol(GLfloat largeur, GLfloat profondeur)
{
	m_sommetsSol[0] = glm::vec3(-largeur, -m_hauteurBase - 0.02, -profondeur);
	m_sommetsSol[1] = glm::vec3(largeur, -m_hauteurBase - 0.02, -profondeur);
	m_sommetsSol[2] = glm::vec3(-largeur, -m_hauteurBase - 0.02, profondeur);
	m_sommetsSol[3] = glm::vec3(largeur, -m_hauteurBase - 0.02, profondeur);


	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_sommetsSol), m_sommetsSol, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

}

void House::House_Maker::creeCouleursSol(glm::vec3 couleur)
{
	for (int i = 0; i < 4; i++)
	{
		m_couleursSol[i] = couleur;
	}

	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_couleursSol), m_couleursSol, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

}
