LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
EXEC_NAME=sharpencv.exe

all : $(EXEC_NAME)

$(EXEC_NAME): sharpencv.cpp
	g++ -o $(EXEC_NAME) sharpencv.cpp $(LDLIBS)

clean :
	rm *.exe
