#include "city.h"

// ################################
// Implémentation de SkyGenerator
// ################################


SkyGenerator::SkyGenerator() : BaseGenerator(GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_RGB)
{
	texture_skybox = 0;
}

bool SkyGenerator::LoadSkyTextures(fs::path sky_folder)
{
	std::vector<std::string> faces{
	"textures/outside/right.png",
	"textures/outside/left.png",
	"textures/outside/top.png",
	"textures/outside/bot.png",
	"textures/outside/front.png",
	"textures/outside/back.png"
	};
	texture_skybox = texture_loader.GetTextureSky(faces);
	return true;
}

void SkyGenerator::generateBase()
{
	glGenVertexArrays(1, &vao_skybox);

	glBindVertexArray(vao_skybox);
	create_base(1.0, 1.0, 1.0, glm::vec3(1.0, 0.0, 0.0));
	glBindVertexArray(0);

}

void SkyGenerator::drawBox(uint shader)
{
	glActiveTexture(GL_TEXTURE0);
	glUniform1i(glGetUniformLocation(shader, "Skybox"), 0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_skybox);
	glBindVertexArray(vao_skybox);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}



// ################################
// Implémentation de ProceduralCity
// ################################

ProceduralCity::ProceduralCity()
{

}

bool ProceduralCity::configure(fs::path root_folder)
{
	this->root_folder = root_folder;
	// Load les shaders 
	MyShader shader;
	if(!shader.OpenMyShader("shaders/vert_shader.glsl", "shaders/frag_shader.glsl"))
	{
		std::cout << "Echer lors de l'ouverture des shaders par default" << std::endl;
		return false;
	}
	this->shader_texture = shader.GetShaderID();
	if (!shader.OpenMyShader("shaders/vert_skybox.glsl", "shaders/frag_skybox.glsl"))
	{
		std::cout << "Echer lors de l'ouverture des shaders de la skybox" << std::endl;
		return false;
	}
	this->shader_skybox = shader.GetShaderID();

	if (!shader.OpenMyShader("shaders/vert_obj.glsl", "shaders/frag_obj.glsl"))
	{
		std::cout << "Echer lors de l'ouverture des shaders des objects" << std::endl;
		return false;
	}
	this->shader_obj= shader.GetShaderID();


	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);



	return true;

}

void ProceduralCity::load() {
	model_obj = Model3D("model/TIE-fighter.obj");


	if (!sky_generator.LoadSkyTextures(root_folder / "outside"))
	{
		std::cout << "Echer au chargement des textures du ciels" << std::endl;
		return;
	}

	sky_generator.generateBase();
	sky_generator.setIsLoaded(true);



}

const float multipleCouleur[4] = { 1.0,1.0,1.0,1.0 };

float x = 0.0f;
float y = 0.0f;

float angle = 0.0f;

float rotX = 0.0f;
float rotY = 0.0f;

float rayon = 20.0f;

void ProceduralCity::render()
{


	angle += 0.5f;
	if (angle >= 360)
		angle = 0.0f;


	glm::mat4	view;
	glm::mat4	projection;
	glm::mat4	modele;
	// Update la position de la camera selon les inputs recu entre les appels de render
	camera.update();
	projection = glm::perspective(glm::radians(camera.getFOV()),w_width / w_height * 1.0f , 0.1f, 800.0f);

	shader_skybox.Use();
	view = camera.getView();
	shader_skybox.setMat4("gProjection", projection);
	shader_skybox.setMat4("gVue", view);

	sky_generator.drawBox(shader_skybox.getID());
	
	view = camera.getLookAt();

	shader_obj.Use();
	shader_obj.setMat4("gProjection", projection);
	shader_obj.setMat4("gVue", view);
	modele = glm::mat4(1.0f);

	for (int i = 0; i < 10; i++) {
		modele = glm::translate(modele, glm::vec3(-rayon*glm::cos(glm::radians(angle)), i*10.0f, -rayon*glm::radians(angle))); // translate it down so it's at the center of the scene
		modele = glm::rotate(modele, glm::radians(-90.0f+i*10.0f), glm::vec3(0, 1, 0));
		modele = glm::scale(modele, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
		shader_obj.setMat4("gModele", modele);
		model_obj.Draw(shader_obj.getID());
	}

}

