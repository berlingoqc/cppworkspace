#version 320 core

layout(location=0) in vec4 Position;
layout(location=1) in vec4 Couleur;

out vec4 coul;

void main()
{
	gl_Position = Position;

	coul = Couleur;
}