#version 430 core
in vec4 Color;
in vec2 positXY;

out vec4 fragColor;

void main(void)
{
	vec3 couleurBrique = Color.xyz;
	vec3 couleurMortier = vec3(0.5,0.5,0.5);
	vec2 tailleMortier = vec2(0.1,0.1);
	vec2 tailleBrique = vec2(0.9,0.9);
	vec2 positionActu, utiliseBrique;
	vec3 couleur;

	positionActu = positXY / tailleMortier;

	if (fract(positionActu.y * 0.5) > 0.5)
		positionActu.x += 0.5;

	positionActu = fract(positionActu);

	utiliseBrique = step(positionActu, tailleBrique);

	couleur = mix(couleurMortier, couleurBrique, utiliseBrique.x * utiliseBrique.y);
	
	fragColor = vec4(couleur, 1.0);
}