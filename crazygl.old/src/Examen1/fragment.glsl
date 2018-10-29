#version 330 core
in vec4 coul;

out vec4 color;

void main()
{
	float red = gl_FragCoord.x/1000.0;
	float green = gl_FragCoord.y/1000.0;

	color = vec4(red,green,0.0f,1.0);
	/*
    color = coul;
	*/
}