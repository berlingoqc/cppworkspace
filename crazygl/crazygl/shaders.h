#ifndef SHADER_H
#define SHADER_H

#include "header.h"

#include <fstream>

namespace ENGINE {

	class MyShader {

	private:
		unsigned int ID;
		char ErrorMessage[1024];
	public:
		MyShader() {}

		bool OpenMyShader(const char* vertexPath, const char* fragmentPath);
		unsigned int GetShaderID();
		void PrintErrorStack();
	protected:
		// ReadMyShaderCode lit le code du MyShader depuis son fichier et le retourne
		std::string ReadMyShaderCode(const char * filePath);
		// CompileMyShader compile le MyShader
		unsigned int CompileMyShader(std::string code, GLenum type, std::string nameType);
		bool CheckCompileErrors(GLuint MyShader, std::string nameType);
	};

}

#endif