LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lcuda -lcudart
EXEC_NAME=lab1.exe

all : $(EXEC_NAME)

$(EXEC_NAME): hsvtorgbkernel.o
	g++ -std=c++11 -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o $(EXEC_NAME) lab1.cpp hsvtorgbkernel.o $(LDLIBS)

hsvtorgbkernel.o:
	nvcc -g -G -arch sm_30 -c hsvtorgbkernel.cu

clean :
	rm -f *.exe *.o
