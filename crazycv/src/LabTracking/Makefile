LDLIBS=-LC:/msys64/mingw64/bin -lopencv_core342 -lopencv_videoio342 -lopencv_imgproc342 -lopencv_highgui342 -lopencv_imgcodecs342
#LDLIBS=-lopencv_world
EXEC_NAME=tracking.exe

all : $(EXEC_NAME)

$(EXEC_NAME): main.cpp
	g++ -std=c++11 -g -o $(EXEC_NAME) main.cpp $(LDLIBS)

clean :
	rm -f *.exe
