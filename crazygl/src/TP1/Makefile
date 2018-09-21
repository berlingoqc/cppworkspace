CPPFLAGS=-O0 -g2 -Wall -c -fmessage-length=0 -I/usr/include

ifeq ($(OS),Windows_NT)
	LDLIBS=-lopengl32 -lglu32 -lfreeglut -lglew32
else
	LDLIBS=-lGL -lglut -lGLU -lGLEW
endif

EXEC_NAME=tp1.exe

all : $(EXEC_NAME)

$(EXEC_NAME): main.cpp
	g++ -std=c++11 -g  -o ${EXEC_NAME} main.cpp $(LDLIBS)

clean :
	rm -f *.exe 
