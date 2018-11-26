#ifndef CITY_H
#define CITY_H

#include "engine.h"
#include "shaders.h"
#include "textures.h"
#include "mesh.h"
#include <glm/vec3.hpp>
#include <filesystem>

#include <thread>
#include <future>

#include "camera.h"

using namespace std;
using namespace ENGINE;
namespace fs = std::filesystem;

inline glm::vec3 generate_random_vec3(float minx, float maxx, float miny, float maxy, float minz, float maxz)
{
	return glm::vec3(((2.0f - 0.25f) * ((float)rand() / RAND_MAX)) + 0.25f, ((5.0f - 0.25f) * ((float)rand() / RAND_MAX)) + 0.25f,
		((2.0f - 0.75f) * ((float)rand() / RAND_MAX)) + 0.75f);
	//return glm::vec3(generateFloatInRange(minx, maxx), generateFloatInRange(miny,maxy), generateFloatInRange(minz,maxz));
}

inline uint get_random_item_vector(std::vector<uint>& vector)
{
	return vector[generateUintInRange(0, vector.size() - 1)];
}


class BaseGenerator
{
protected:
	MyTexture	texture_loader;

	std::atomic<bool>	is_ready = false;

	float m_hauteurBase;
	glm::vec3 m_sommetsBase[8];
	glm::vec3 m_couleursBase[8];
	glm::vec2 m_textureBase[8];
	const unsigned int m_indicesBase[36] = {
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
		2,4,6
	};

	glm::vec3 m_sommetsToit[6];
	glm::vec3 m_couleursToit[6];
	glm::vec2 m_textureToit[6];
	const unsigned int m_indicesToit[18] = {
		0,1,2,
		3,4,5,
		0,5,2,
		0,3,5,
		0,4,1,
		0,3,4
	};

	glm::vec3 m_sommetsSurface[4];
	glm::vec3 m_couleursSurface[4];
	glm::vec2 m_textureSurface[4];

	BaseGenerator(uint wraps, uint wrapt, uint minfilter, uint magfilter, uint imgformat) : texture_loader(wraps,wrapt,minfilter,magfilter,imgformat)
	{
		
	}

	void loadTexturesFolder(std::vector<uint>& textures, fs::path folder_tex)
	{
		std::cout << "Chargement des textures depuis " << folder_tex.string() << std::endl;
		for (auto& p : fs::directory_iterator(folder_tex))
		{
			uint tex = texture_loader.GetTexture(p.path().string().c_str());
			if (tex == ERROR_TEXTURE)
			{
				std::cout << "Erreur chargement de la texture " << p.path().string() << std::endl;
				continue;
			}
			textures.push_back(tex);
		}
		std::cout << "Fin de chargement des textures de " << folder_tex.string() << " " << textures.size() << " textures charger";
	}

	void create_base(float width, float height, float depth, glm::vec3 color)
	{
		m_hauteurBase = height / 2;
		create_vertex_base(width / 2, height / 2, depth / 2);
		create_color_base(color);
		create_texture_base();

		GLuint IBOBase;
		glGenBuffers(1, &IBOBase);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBOBase);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indicesBase), m_indicesBase, GL_STATIC_DRAW);

	}

	void create_roof(float width, float height, float depth, glm::vec3 color)
	{
		create_vertex_roof(width / 2, height / 2, depth / 2);
		create_color_roof(color);
		create_texture_roof();

		GLuint IBOToit;
		glGenBuffers(1, &IBOToit);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBOToit);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indicesToit), m_indicesToit, GL_STATIC_DRAW);
	}

	void create_wall(float width, float height, float depth, float iteration, bool horizontal, bool side)
	{
		create_vertex_wall(width / 2, height, depth / 2, horizontal, side);
		create_texture_wall(iteration);
	}
public:

	bool getIsLoaded() {
		return is_ready;
	}

	void setIsLoaded(bool v) {
		is_ready = v;
	}

private:
	void create_vertex_base(GLfloat largeur, GLfloat hauteur, GLfloat profondeur)
	{
		m_sommetsBase[0] = glm::vec3(-largeur, hauteur, -profondeur);
		m_sommetsBase[1] = glm::vec3(largeur, hauteur, -profondeur);
		m_sommetsBase[2] = glm::vec3(-largeur, -hauteur, -profondeur);
		m_sommetsBase[3] = glm::vec3(largeur, -hauteur, -profondeur);
		m_sommetsBase[4] = glm::vec3(-largeur, hauteur, profondeur);
		m_sommetsBase[5] = glm::vec3(largeur, hauteur, profondeur);
		m_sommetsBase[6] = glm::vec3(-largeur, -hauteur, profondeur);
		m_sommetsBase[7] = glm::vec3(largeur, -hauteur, profondeur);
		m_sommetsBase[8] = glm::vec3(largeur / 3, -hauteur, -profondeur - 0.001);
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

	void create_color_base(glm::vec3 couleur)
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

	void create_texture_base()
	{
		m_textureBase[0] = glm::vec2(0.0f, 1.0f);
		m_textureBase[1] = glm::vec2(1.0f, 1.0f);
		m_textureBase[2] = glm::vec2(0.0f, 0.0f);
		m_textureBase[3] = glm::vec2(1.0f, 0.0f);

		m_textureBase[4] = glm::vec2(1.0f, 1.0f);
		m_textureBase[5] = glm::vec2(0.0f, 1.0f);
		m_textureBase[6] = glm::vec2(1.0f, 0.0f);
		m_textureBase[7] = glm::vec2(0.0f, 0.0f);

		GLuint texVBO;
		glGenBuffers(1, &texVBO);
		glBindBuffer(GL_ARRAY_BUFFER, texVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_textureBase), m_textureBase, GL_STATIC_DRAW);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);

	}

	void create_vertex_roof(GLfloat largeur, GLfloat hauteur, GLfloat profondeur)
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
	void create_color_roof(glm::vec3 couleur)
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

	void create_texture_roof()
	{
		m_textureToit[0] = glm::vec2(0.0f, 1.0f);
		m_textureToit[1] = glm::vec2(0.0f, 0.0f);
		m_textureToit[2] = glm::vec2(0.0f, 0.0f);
		m_textureToit[3] = glm::vec2(1.0f, 1.0f);
		m_textureToit[4] = glm::vec2(1.0f, 0.0f);
		m_textureToit[5] = glm::vec2(1.0f, 0.0f);

		GLuint texVBO;
		glGenBuffers(1, &texVBO);
		glBindBuffer(GL_ARRAY_BUFFER, texVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_textureToit), m_textureToit, GL_STATIC_DRAW);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);
	}

	void create_vertex_wall(GLfloat largeur, GLfloat hauteur, GLfloat profondeur, bool horizontal, bool cote)
	{
		if (horizontal)
		{
			m_sommetsSurface[0] = glm::vec3(-largeur, hauteur, -profondeur);
			m_sommetsSurface[1] = glm::vec3(largeur, hauteur, -profondeur);
			m_sommetsSurface[2] = glm::vec3(-largeur, hauteur, profondeur);
			m_sommetsSurface[3] = glm::vec3(largeur, hauteur, profondeur);
		}
		else
		{
			if (cote == 0)
			{
				m_sommetsSurface[0] = glm::vec3(-largeur, -m_hauteurBase, profondeur);
				m_sommetsSurface[1] = glm::vec3(largeur, -m_hauteurBase, profondeur);
				m_sommetsSurface[2] = glm::vec3(-largeur, hauteur, profondeur);
				m_sommetsSurface[3] = glm::vec3(largeur, hauteur, profondeur);
			}
			else
			{
				m_sommetsSurface[0] = glm::vec3(-largeur, -m_hauteurBase, -profondeur);
				m_sommetsSurface[1] = glm::vec3(-largeur, -m_hauteurBase, profondeur);
				m_sommetsSurface[2] = glm::vec3(-largeur, hauteur, -profondeur);
				m_sommetsSurface[3] = glm::vec3(-largeur, hauteur, profondeur);


			}
		}

		GLuint vertexbuffer;
		glGenBuffers(1, &vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_sommetsSurface), m_sommetsSurface, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);
	}
	void create_texture_wall(GLfloat iterations)
	{
		m_textureSurface[0] = glm::vec2(0.0f, 0.0f);
		m_textureSurface[1] = glm::vec2(iterations, 0.0f);
		m_textureSurface[2] = glm::vec2(0.0f, iterations);
		m_textureSurface[3] = glm::vec2(iterations, iterations);


		GLuint texVBO;
		glGenBuffers(1, &texVBO);
		glBindBuffer(GL_ARRAY_BUFFER, texVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_textureSurface), m_textureSurface, GL_STATIC_DRAW);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);

	}

	void create_vertex_skybox()
	{
		m_sommetsBase[0] = glm::vec3(-1.0, 1.0, -1.0);
		m_sommetsBase[1] = glm::vec3(1.0, 1.0, -1.0);
		m_sommetsBase[2] = glm::vec3(-1.0, -1.0, -1.0);
		m_sommetsBase[3] = glm::vec3(1.0, -1.0, -1.0);
		m_sommetsBase[4] = glm::vec3(-1.0, 1.0, 1.0);
		m_sommetsBase[5] = glm::vec3(1.0, 1.0, 1.0);
		m_sommetsBase[6] = glm::vec3(-1.0, -1.0, 1.0);
		m_sommetsBase[7] = glm::vec3(1.0, -1.0, 1.0);

		GLuint vertexbuffer;
		glGenBuffers(1, &vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_sommetsBase), m_sommetsBase, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);
	}
};

class SkyGenerator : public BaseGenerator
{
	std::vector<uint>			texture_cloud;
	std::vector<uint>			texture_horizon;

	uint						texture_skybox;

	uint						vao_cloud[2];
	uint						vao_horizon[4];
	uint						vao_skybox;

	std::atomic<bool>			is_loaded;

public:
	SkyGenerator();
	bool LoadSkyTextures(fs::path sky_folder);

	void generateBase();

	void drawBox(uint shader);

};



class ProceduralCity
{


private:
	Shrapper	shader_texture;
	Shrapper	shader_skybox;
	Shrapper	shader_obj;

	uint	u_projection;
	uint	u_view;
	uint	u_model;

	Model3D		model_obj;

	SkyGenerator			sky_generator;

	FPSCamera				camera;

	std::future<bool>		loader;

	fs::path				root_folder;


	int		w_height = 900;
	int		w_width = 1600;



public:
	ProceduralCity();


	bool configure(fs::path root_folder);
	void load();

	void render();

	FPSCamera&	getCamera()
	{
		return camera;
	}

	std::thread load_thread() {
		return std::thread{ [this] {this->load(); } };
	}
	
};

#endif
