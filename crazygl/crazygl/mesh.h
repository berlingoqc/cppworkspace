#ifndef _MESH_H_
#define _MESH_H_

#include "header.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"


struct Vertex {
	// position
	glm::vec3 Position;
	// normal
	glm::vec3 Normal;
	// textCoords
	glm::vec3 TexCoords;
	// tangent
	glm::vec3 Tangent;
	// bitangent
	glm::vec3 Bitangent;
};

struct Texture {
	unsigned int ID;
	const char* Type;
	const char& Path;
};

class Mesh {
public:
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	std::vector<Texture> textures;
private:
	unsigned int VAO, VBO, EBO;

public:
	Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures);
	void Draw(unsigned int shader);
	void setupMesh();
};

class Model3D {
public:
	Model3D(const char* path);
	void Draw(unsigned int shaderID);
private:
	std::vector<Mesh> meshes;
	const char* directory;

	void loadModel(std::string path);
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName);
};

#endif