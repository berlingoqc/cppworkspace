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
	std::vector<std::string> faces;
	for(int i = 0;i < 6;i++)
	{
		fs::path p = sky_folder / "box";
		p = p.append(std::to_string(i) + ".jpg");
		faces.push_back(p.string());
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

void SkyGenerator::drawBox(uint shader)
{
	glActiveTexture(GL_TEXTURE0);
	glUniform1i(glGetUniformLocation(shader, "Skybox"), 0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_skybox);
	glBindVertexArray(vao_skybox);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void SkyGenerator::drawCloud(uint shader)
{
	for (int i = 0; i < 2; i++) {
		glBindVertexArray(vao_cloud[i]);
		glBindTexture(GL_TEXTURE_2D, texture_cloud[0]);
		glUniform1i(glGetUniformLocation(shader, "ourTexture1"), 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}

void SkyGenerator::drawHorizon(uint shader)
{
	for (int i = 0; i < 4; i++) {
		glBindVertexArray(vao_horizon[i]);
		glBindTexture(GL_TEXTURE_2D, texture_horizon[0]);
		glUniform1i(glGetUniformLocation(shader, "ourTexture1"), 0);
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

	glGenVertexArrays(1, &vao_sol);
	glBindVertexArray(vao_sol);
	create_wall(500.0, -1.01, 500.0, 100.0, true, false);
	glBindVertexArray(0);

}

void GroundGenerator::drawGround(uint shader)
{
	glBindVertexArray(vao_sol);
	glBindTexture(GL_TEXTURE_2D, texture_street);
	glUniform1i(glGetUniformLocation(shader, "ourTexture1"), 0);
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

void BuildingGenerator::generateBase()
{
	glGenVertexArrays(1, &vao_base);
	glGenVertexArrays(1, &vao_toit);

	glBindVertexArray(vao_base);
	create_base(10, 50, 5.0f, glm::vec3(rand() % 2, rand() % 2, rand() % 2));
	glBindVertexArray(0);

	glBindVertexArray(vao_toit);
	create_roof(10.0f, 90.0f, 5.0f, glm::vec3(0.0f, 0.0f, 1.0f));
	glBindVertexArray(0);
}

void BuildingGenerator::Reset()
{
	nbr_building = generateUintInRange(min_building, max_building);
	int count = 0;
	uint v = nbr_building / 10;
	float toggle = 1.0f;
	for (uint z = 1; z < v; z++)
	{
		for (uint x = 1; x < nbr_building / v; x++)
		{
			BuildingValue bv;
			bv.size = generate_random_vec3(0.5f, 2.0f, 0.5f, 5.0f, .75f, 2.00f);
			bv.translate = glm::vec3(13.0f*x, 0.0f, -15.0f*z*toggle);
			bv.texture_roof = get_random_item_vector(textures_roof);
			bv.texture_side = get_random_item_vector(textures_side);
			building_values.push_back(bv);
			toggle *= -1.0f;
			count++;
		}
	}
	nbr_building = count;
}

void BuildingGenerator::Render(uint shader)
{
	glm::mat4 modele = glm::mat4(1.0);

	for (auto bv : building_values) {
		modele = glm::mat4(1.0);
		modele = glm::translate(modele, bv.translate);
		modele = glm::scale(modele, bv.size);
		glUniformMatrix4fv(glGetUniformLocation(shader, "gModele"), 1, GL_FALSE, &modele[0][0]);
		glBindVertexArray(vao_base);
		glBindTexture(GL_TEXTURE_2D, bv.texture_side);
		glUniform1i(glGetUniformLocation(shader, "ourTexture1"), 0);
		glDrawElements(GL_TRIANGLES, 12 * 3, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		glBindVertexArray(vao_toit);
		glBindTexture(GL_TEXTURE_2D, bv.texture_roof);
		glUniform1i(glGetUniformLocation(shader, "ourTexture1"), 0);
		glDrawElements(GL_TRIANGLES, 6 * 3, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

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
	//model_obj = Model3D("obj/nanosuit/nanosuit.obj");



	// Load les textures
	if (!building_generator.LoadBuildingTextures(root_folder / "building"))
	{
		std::cout << "Echer au chargement des textures des batiments" << std::endl;
		return;
	}

	building_generator.generateBase();
	building_generator.Reset();
	building_generator.setIsLoaded(true);

	if (!sky_generator.LoadSkyTextures(root_folder / "outside"))
	{
		std::cout << "Echer au chargement des textures du ciels" << std::endl;
		return;
	}

	sky_generator.generateBase();
	sky_generator.setIsLoaded(true);



	if (!sky_generator.LoadSkyTextures(root_folder / "outside"))
	{
		std::cout << "Echer au chargement des textures du ciels" << std::endl;
		return;

	}

	if (!ground_generator.LoadGroundTextures(root_folder / "ground"))
	{
		std::cout << "Echer au chargement des textures du sol" << std::endl;
		return;

	}

	ground_generator.generateBase();
	ground_generator.setIsLoaded(true);
}

const float multipleCouleur[4] = { 1.0,1.0,1.0,1.0 };

void ProceduralCity::render()
{
	glm::mat4	view;
	glm::mat4	projection;
	glm::mat4	modele;
	// Update la position de la camera selon les inputs recu entre les appels de render
	camera.update();

	projection = glm::perspective(glm::radians(camera.getFOV()), glutGet(GLUT_WINDOW_WIDTH) / glutGet(GLUT_WINDOW_HEIGHT)*1.0f, 0.1f, 800.0f);

	if (sky_generator.getIsLoaded()) {
		shader_skybox.Use();
		view = camera.getView();
		shader_skybox.setMat4("gProjection", projection);
		shader_skybox.setMat4("gVue", view);

		sky_generator.drawBox(shader_skybox.getID());
	}
	view = camera.getLookAt();

	/*
	shader_obj.Use();
	shader_obj.setMat4("gProjection", projection);
	shader_obj.setMat4("gVue", view);
	modele = glm::mat4(1.0f);
	modele = glm::translate(modele, glm::vec3(10.0f, 0.0f, 12.0f)); // translate it down so it's at the center of the scene
	modele = glm::rotate(modele,glm::radians(-90.0f),glm::vec3(0,1,0));
	modele = glm::scale(modele, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
	shader_obj.setMat4("gModele", modele);
	model_obj.Draw(shader_obj.getID());
	*/
	shader_texture.Use();
	shader_texture.setMat4("gProjection", projection);
	shader_texture.setMat4("gVue", view);

	glActiveTexture(GL_TEXTURE0);

	if (building_generator.getIsLoaded()) {
		building_generator.Render(shader_texture.getID());
	}

	modele = glm::mat4(1.0);
	shader_texture.setMat4("gModele", modele);
	if (ground_generator.getIsLoaded()) {
		ground_generator.drawGround(shader_texture.getID());
	}
	if (sky_generator.getIsLoaded()) {
		sky_generator.drawCloud(shader_texture.getID());
		sky_generator.drawHorizon(shader_texture.getID());
	}

}

