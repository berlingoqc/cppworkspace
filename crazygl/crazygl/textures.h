#ifndef TEXTURE_H
#define TEXTURE_H

#include <SOIL.h>
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

		uchar*		GetContent(const char* filename);

	public:
		MyTexture(uint wraps, uint wrapt, uint minfilter, uint magfilter, uint img_format);

		uint		GetTexture(const char* filename);
		uint		GetTextureSky(std::vector<const char*> faces);

		void setMinFilter(uint minfilter)
		{
			min_filter = minfilter;
		}
		
	};
}


#endif // TEXTURE_H