#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 gModele;
uniform mat4 gVue;
uniform mat4 gProjection;

void main()
{
	TexCoords = aTexCoords;
	gl_Position = gProjection * gVue * gModele * vec4(aPos, 1.0);
}