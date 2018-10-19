LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
EXEC_NAME=tracking.exe

all : $(EXEC_NAME)

$(EXEC_NAME): main.cpp
	g++ -std=c++11 -g -o $(EXEC_NAME) main.cpp $(LDLIBS)

clean :
	rm -f *.exe
