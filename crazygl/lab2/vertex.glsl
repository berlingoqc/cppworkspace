#version 430 core
layout(location=0) in vec3 Position;
layout(location=1) in vec3 Couleur;

uniform mat4 gRot;
uniform mat4 gScale;

out vec4 coul;
out vec2 positionXY;

void main()
{
	gl_Position = gRot * gScale * vec4(Position,1.0);
	positionXY = gl_Position.xy;
	coul = vec4(Couleur,1.0);
	//coul = Couleur;
}