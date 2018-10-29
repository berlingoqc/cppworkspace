CPPFLAGS=-O0 -g2 -Wall -c -fmessage-length=0 -I/usr/include
ifeq ($(OS),Windows_NT)
	LDLIBS=-lopengl32 -lglu32 -lfreeglut -lglew32
else
	LDLIBS=-lGL -lglut -lGLU -lGLEW -lpthread
endif

EXEC_NAME=test.exe

all : $(EXEC_NAME)

$(EXEC_NAME): test.cpp
	g++ -std=c++11 -g  -o test.exe test.cpp $(LDLIBS)

clean :
	rm -f *.exe 
