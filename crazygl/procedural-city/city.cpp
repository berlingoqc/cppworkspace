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
	loadTexturesFolder(texture_cloud, sky_folder / "sky");
	loadTexturesFolder(texture_horizon, sky_folder / "horizon");
	std::vector<const char*> faces;
	for(int i = 0;i < 6;i++)
	{
		faces.push_back((sky_folder / "box" / (std::to_string(i) + ".jpg")).string().c_str());
	}
	texture_skybox = texture_loader.GetTextureSky(faces);
	return true;
}

void SkyGenerator::generateBase()
{
	glGenVertexArrays(2, vao_cloud);
	glGenVertexArrays(4, vao_horizon);
	glGenVertexArrays(1, &vao_skybox);

	glBindVertexArray(vao_cloud[0]);
	create_wall(600.0, 350.0, 700.0, 1.0, true, false);
	glBindVertexArray(0);

	glBindVertexArray(vao_cloud[1]);
	create_wall(600.0, 150.0, 700.0, 2.0, true, false);
	glBindVertexArray(0);

	glBindVertexArray(vao_horizon[0]);
	create_wall(500.0, 500.0, -500.0, 3.0, false, false);
	glBindVertexArray(0);

	glBindVertexArray(vao_horizon[1]);
	create_wall(500.0, 500.0, 500.0, 3.0, false, false);
	glBindVertexArray(0);

	glBindVertexArray(vao_horizon[2]);
	create_wall(500.0, 500.0, 500.0, 3.0, false, true);
	glBindVertexArray(0);

	glBindVertexArray(vao_horizon[3]);
	create_wall(-500.0, 500.0, 500.0, 3.0, false, true);
	glBindVertexArray(0);

	glBindVertexArray(vao_skybox);
	create_base(1.0, 1.0, 1.0, glm::vec3(1.0, 0.0, 0.0));
	glBindVertexArray(0);

}

void SkyGenerator::drawBox(uint* shader)
{
	glActiveTexture(GL_TEXTURE0);
	glUniform1i(glGetUniformLocation(*shader, "Skybox"), 0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_skybox);
	glBindVertexArray(vao_skybox);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void SkyGenerator::drawCloud(uint* shader)
{
	for (int i = 0; i < 2; i++) {
		glBindVertexArray(vao_cloud[i]);
		glBindTexture(GL_TEXTURE_2D, texture_cloud[0]);
		glUniform1i(glGetUniformLocation(*shader, "ourTexture1"), 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}

void SkyGenerator::drawHorizon(uint* shader)
{
	for (int i = 0; i < 4; i++) {
		glBindVertexArray(vao_horizon[i]);
		glBindTexture(GL_TEXTURE_2D, texture_horizon[0]);
		glUniform1i(glGetUniformLocation(*shader, "ourTexture1"), 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}

// #################################
// Implémentation de GroundGenerator
// #################################


GroundGenerator::GroundGenerator() : BaseGenerator(GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR, GL_RGB)
{
	texture_grass = 0;
	texture_street = 0;
	texture_building = 0;
}

bool GroundGenerator::LoadGroundTextures(fs::path ground_folder)
{
	fs::path street = ground_folder / "paver.jpg";
	texture_street = texture_loader.GetTexture(street.string().c_str());
	return true;
}

void GroundGenerator::generateBase()
{

	glGenVertexArrays(1, &vao_base);
	glGenVertexArrays(1, &vao_toit);
	glGenVertexArrays(1, &vao_sol);

	glBindVertexArray(vao_base);
	create_base(10, 50, 5.0f, glm::vec3(rand() % 2, rand() % 2, rand() % 2));
	glBindVertexArray(0);

	glBindVertexArray(vao_toit);
	create_roof(10.0f, 90.0f, 5.0f, glm::vec3(0.0f, 0.0f, 1.0f));
	glBindVertexArray(0);

	glBindVertexArray(vao_sol);
	create_wall(500.0, -1.01, 500.0, 100.0, true, false);
	glBindVertexArray(0);

}

void GroundGenerator::drawGround(uint* shader)
{
	glBindVertexArray(vao_sol);
	glBindTexture(GL_TEXTURE_2D, texture_street);
	glUniform1i(glGetUniformLocation(*shader, "ourTexture1"), 0);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

// ###################################
// Implémentation de BuildingGenerator
// ###################################


BuildingGenerator::BuildingGenerator() : BaseGenerator(GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_RGB)
{

}

bool BuildingGenerator::LoadBuildingTextures(fs::path building_folder)
{
	loadTexturesFolder(textures_side, building_folder / "side");
	texture_loader.setMinFilter(GL_LINEAR);
	loadTexturesFolder(textures_roof, building_folder / "roof");
	if(textures_roof.empty() || textures_side.empty())
	{
		std::cout << "Erreur chargement des textures des buildlings une des listes est vide" << std::endl;
		return false;
	}
	return true;
}

void BuildingGenerator::init_base()
{
}


// ################################
// Implémentation de ProceduralCity
// ################################

ProceduralCity::ProceduralCity()
{

}

bool ProceduralCity::configure(fs::path root_folder)
{
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

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);



	// Load les textures
	if(!building_generator.LoadBuildingTextures(root_folder / "building"))
	{
		std::cout << "Echer au chargement des textures des batiments" << std::endl;
		return false;
	}

	if(!sky_generator.LoadSkyTextures(root_folder / "outside"))
	{
		std::cout << "Echer au chargement des textures du ciels" << std::endl;
		return false;
	
	}

	if(!ground_generator.LoadGroundTextures(root_folder / "ground"))
	{
		std::cout << "Echer au chargement des textures du sol" << std::endl;
		return false;

	}

	ground_generator.generateBase();
	sky_generator.generateBase();


	return true;

}

const float multipleCouleur[4] = { 1.0,1.0,1.0,1.0 };

void ProceduralCity::render()
{
	glm::mat4	view;
	glm::mat4	projection;
	glm::mat4	modele;
	// Update la position de la camera selon les inputs recu entre les appels de render
	camera.update();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	projection = glm::perspective(glm::radians(camera.getFOV()), glutGet(GLUT_WINDOW_WIDTH) / glutGet(GLUT_WINDOW_HEIGHT)*1.0f, 0.1f, 800.0f);
	loadUniforms(&shader_skybox);
	view = camera.getView();
	glUniformMatrix4fv(u_projection, 1, GL_FALSE, &projection[0][0]);
	glUniformMatrix4fv(u_view, 1, GL_FALSE, &view[0][0]);

	sky_generator.drawBox(&shader_skybox);

	view = camera.getLookAt();

	u_model = glGetUniformLocation(shader_texture, "gModele");
	assert(u_model != 0xFFFFFFFF);

	loadUniforms(&shader_texture);
	glUniformMatrix4fv(u_projection, 1, GL_FALSE, &projection[0][0]);
	glUniformMatrix4fv(u_view, 1, GL_FALSE, &view[0][0]);
	glUniform4fv(glGetUniformLocation(shader_texture, "color"), 1, multipleCouleur);

	glActiveTexture(GL_TEXTURE0);

	// Dessine les immeubles


	modele = glm::mat4(1.0);
	glUniformMatrix4fv(u_model, 1, GL_FALSE, &modele[0][0]);


	ground_generator.drawGround(&shader_texture);

	sky_generator.drawCloud(&shader_texture);
	sky_generator.drawHorizon(&shader_texture);

	glutSwapBuffers();
}

void ProceduralCity::loadUniforms(uint* shader_id)
{
	glUseProgram(*shader_id);
	u_projection = glGetUniformLocation(*shader_id, "gProjection");
	assert(u_projection != 0xFFFFFFFF);

	u_view = glGetUniformLocation(*shader_id, "gVue");
	assert(u_projection != 0xFFFFFFFF);
}
