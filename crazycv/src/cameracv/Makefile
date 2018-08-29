LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
EXEC_NAME=cameracv.exe

all : $(EXEC_NAME)

$(EXEC_NAME): cameracv.cpp
	g++ -o $(EXEC_NAME) cameracv.cpp $(LDLIBS)

clean :
	rm *.exe
