#version 460 core
layout(location=0) in vec4 Position;
layout(location=1) in vec4 Couleur;

out vec4 coul;

void main()
{
	gl_Position = Position;
	//coul = vec4(0.0,0.0,0.0,1.0);
	coul = Couleur;
}