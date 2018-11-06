#include "mesh.h"

using namespace Assimp;

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures) {
	this->vertices = vertices;
	this->indices = indices;
	this->textures = textures;

	setupMesh();
}

void Mesh::Draw(unsigned int shader) {
	// bind les textures approprier
	unsigned int diffuseNr = 1;
	unsigned int specularNr = 1;
	unsigned int normalNr = 1;
	unsigned int heightNr = 1;
	for (unsigned int i = 0; i < textures.size(); i++) {
		glActiveTexture(GL_TEXTURE0 + 1); // Active l'uniter de texture approprié avant de bind
		std::string number;
		std::string name = textures[i].Type;
		if (name == "texture_diffuse") {
			number = std::to_string(diffuseNr++);
		}
		else if (name == "texture_normal") {
			number = std::to_string(normalNr++);
		}
		else if (name == "texture_specular") {
			number = std::to_string(specularNr++);
		}
		else if (name == "texture_height") {
			number = std::to_string(heightNr++);
		}

		glUniform1i(glGetUniformLocation(shader, (name + number).c_str()), i);
		glBindTexture(GL_TEXTURE_2D, textures[i].ID);
	}
}

void Mesh::setupMesh() {
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		// vertex position
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
		// vertex normals
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
		// vertex texture coords
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
		// vertex tangeant
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
		// vertex bitangeant
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

		glBindVertexArray(0);
}



Model3D::Model3D(const char* path) {
	loadModel(path);
}

void Model3D::Draw(unsigned int shaderID) {
	// Loop over mes merchs pis on les draw
	for (unsigned int i = 0; i < meshes.size(); i++) {
		meshes[i].Draw(shaderID);
	}
}

void Model3D::loadModel(std::string path) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		return;
	}
	directory = path.substr(0, path.find_last_of('/')).c_str();

	processNode(scene->mRootNode, scene);
}

void Model3D::processNode(aiNode *node, const aiScene *scene) {

}

Mesh Model3D::processMesh(aiMesh *mesh, const aiScene *scene) {

}

std::vector<Texture> Model3D::loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName) {

}