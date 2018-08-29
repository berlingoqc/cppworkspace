CPPFLAGS=-O0 -g2 -Wall -c -fmessage-length=0 -I/usr/include
LDLIBS=-lGL -lglut -lGLU
EXEC_NAME=test.exe

all : $(EXEC_NAME)

$(EXEC_NAME): test.cpp
	gcc  -o test.exe test.cpp $(LDLIBS)

clean :
	rm *.exe *.o
