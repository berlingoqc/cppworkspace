LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
EXEC_NAME=colorswitcher.exe

all : $(EXEC_NAME)

$(EXEC_NAME): colorswitcher.cpp
	g++ -o $(EXEC_NAME) colorswitcher.cpp $(LDLIBS)

clean :
	rm *.exe
