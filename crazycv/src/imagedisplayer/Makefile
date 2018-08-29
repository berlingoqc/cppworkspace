LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
EXEC_NAME=imgdisplayercv.exe

all : $(EXEC_NAME)

$(EXEC_NAME): imgdisplayercv.cpp
	g++ -o $(EXEC_NAME) imgdisplayercv.cpp $(LDLIBS)

clean :
	rm *.exe
