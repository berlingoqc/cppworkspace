#pragma once

// Import des header de opengl
#include <GL/glew.h>

#include <GL/freeglut.h>

// import les trucs de math
#include <glm/glm.hpp>
#include <glm/trigonometric.hpp>    // sin , cos , radians ...
#include <glm/gtc/matrix_transform.hpp>
#include <glm/exponential.hpp>      // pow , log, exp2, sqrt, ...
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>


// Import des autres headers 
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h> 
#include <vector>

#include <random>

#include <time.h>

typedef unsigned int uint;

#ifdef WITH_STB_IMAGE
	#define STB_IMAGE_IMPLEMENTATION
	#include <stb_image.h>
	#define STB_IMAGE_WRITE_IMPLEMENTATION
	#include <stb_image_write.h>
#endif // WITH_STB_IMAGE



