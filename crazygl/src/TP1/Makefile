CPPFLAGS=-O0 -g2 -Wall -c -fmessage-length=0 -I/usr/include
LDLIBS=-lGL -lglut -lGLU -lGLEW
EXEC_NAME=tp1.exe

all : $(EXEC_NAME)

$(EXEC_NAME): main.cpp
	g++ -g  -o ${EXEC_NAME} main.cpp $(LDLIBS)

clean :
	rm -f *.exe 
