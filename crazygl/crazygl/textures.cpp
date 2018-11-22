#include "textures.h"


namespace ENGINE
{
	uchar* MyTexture::GetContent(std::string filename)
	{
		uchar*	img = SOIL_load_image(filename.c_str(), &w, &h, &channel, SOIL_LOAD_AUTO);
		return img;
	}

	MyTexture::MyTexture(uint wraps, uint wrapt, uint minfilter, uint magfilter, uint imgformat)
	{
		w = 0;
		h = 0;
		channel = 0;
		wrap_s = wraps;
		wrap_t = wrapt;
		min_filter = minfilter;
		mag_filter = magfilter;
		img_format = imgformat;
	}



	uint MyTexture::GetTexture(std::string filename)
	{
		uint text;

		glGenTextures(1, &text);
		glBindTexture(GL_TEXTURE_2D, text);
		//Texture wrapping				
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
		// texture filtering
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);

		uchar* img = GetContent(filename);
		if (img == nullptr)
		{
			return ERROR_TEXTURE;
		}

		glTexImage2D(GL_TEXTURE_2D, 0, channel, w, h, 0, img_format, GL_UNSIGNED_BYTE, img);
		glGenerateMipmap(GL_TEXTURE_2D);
		SOIL_free_image_data(img);
		glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture 
		return text;
	}

	uint MyTexture::GetTextureSky(std::vector<std::string> faces)
	{
		GLuint textureID;
		unsigned char* image;

		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

		//wrapping
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		//filtering
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);


		for (GLuint i = 0; i < faces.size(); i++)
		{
			image = GetContent(faces[i]);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, channel, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
			SOIL_free_image_data(image);
		}

		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

		return textureID;
	}
}
