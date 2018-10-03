#version 330 core
layout(location=0) in vec4 Position;

out vec4 coul;

uniform float Scale;

void main()
{
	
	vec4 a = Position;
	a.x = a.x * Scale;
	a.y = a.y * Scale;
	gl_Position = a;

	// J'ai essayé ça pour l'intensiter de la couleur avant la transformation
	// et ca marche pas dutout
	/*
    float red = 0.0f;
	float green = 0.0f;
	float blue = 0.0f;
	
	if(Position.x > 0.0f) {
		red = 255.0f * Position.x;
	} else {
		blue = 255.0f * Position.x;
	}
	if(Position.y > 0.0f) {
		green = 255.0f * Position.y;
	} else {
		blue = 255.0f * Position.y;
	}
	coul = vec4(red,green,blue,1.0f);
	*/
	coul = vec4(1.0f,1.0f,1.0f,1.0f);
}