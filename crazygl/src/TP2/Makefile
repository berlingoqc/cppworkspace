CPPFLAGS=-O0 -g2 -Wall -c -fmessage-length=0
STD_VERSION=c++11

ifeq ($(OS),Windows_NT)
	LDLIBS=-lopengl32 -lglu32 -lfreeglut -lglew32
else
	LDLIBS=-lGL -lglut -lGLU -lGLEW -lpthread
endif

EXEC_NAME=tp2.exe

all : $(EXEC_NAME)

$(EXEC_NAME): main.cpp
	g++ -std=${STD_VERSION} -g  -o ${EXEC_NAME} main.cpp $(LDLIBS)

clean :
	rm -f *.exe 
