#ifndef TEXTURE_H
#define TEXTURE_H

#include <SOIL/SOIL.h>
#include <vector>

#include "engine.h"

namespace ENGINE
{

	#define ERROR_TEXTURE 999999

	typedef unsigned char uchar;
	
	class MyTexture
	{
	private:
		int w, h, channel;

		uint	wrap_s;
		uint	wrap_t;
		uint	min_filter;
		uint	mag_filter;
		uint	img_format;

		uchar*		GetContent(std::string filepath);

	public:
		MyTexture(uint wraps, uint wrapt, uint minfilter, uint magfilter, uint img_format);

		uint		GetTexture(std::string filename);
		uint		GetTextureSky(std::vector<std::string> faces);

		void setMinFilter(uint minfilter)
		{
			min_filter = minfilter;
		}
		
	};
}


#endif // TEXTURE_H