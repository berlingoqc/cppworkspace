CPPFLAGS=-O0 -g2 -Wall -c -fmessage-length=0 -I/usr/include
LDLIBS=-lGL -lglut -lGLU
EXEC_NAME=test

all : $(EXEC_NAME)

test.o: test.cpp
	gcc $(CPPFLAGS)  -o test.o test.cpp

test: test.o
	gcc -o test.exe test.o $(LDLIBS)

clean :
	rm *.exe *.o
