#ifdef _WIN32
    #include <io.h>
    #define access      _access_s;
#else
    #include <unistd.h>
#endif
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

std::string JoinString(std::string Strings[],int nbr, char sep);

bool FileExists(std::string &Filename);
bool IsValidExtensions(std::string &Filename,const std::string Extensions[],int nbr);
void ValidFile(const std::string &Filename,const std::string Extensions[],int nbr);



std::string JoinString(std::string Strings[],int nbr,char sep) {
    std::stringstream ss;
    for (int i=0;i<nbr;i++) {
        ss << Strings[i] << sep;
    }
    return ss.str();
}


bool FileExists(std::string &Filename) {
    return access(Filename.c_str(),0) == 0;
}

std::string GetFileExtensions(std::string &Filename) {
    return Filename.substr(Filename.find_last_of(".")+1);
}

bool IsValidExtensions(std::string &Filename,const  std::string Extensions[],int nbr) {
    string ext = GetFileExtensions(Filename);
    for(int i=0;i<nbr;i++)
        if(Extensions[0].compare(ext))
            return true;
    return false;
}


void ValidFile(std::string &FileName,const std::string Extensions[], int nbr) {
    if(!FileExists(FileName)) {
        printf("File %s does not exists\n",FileName);
        std::exit(1);
    }
    if(!IsValidExtensions(FileName,Extensions,nbr)) {
        printf("File %s does not have an appropriate extensions %s");
        std::exit(1);
    }
}