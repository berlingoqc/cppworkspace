LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lcuda -lcudart
EXEC_NAME=test.exe

all : $(EXEC_NAME)

$(EXEC_NAME): kernel.o hsvtorgbkernel.o
	g++ -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o $(EXEC_NAME) testcuda.cpp kernel.o hsvtorgbkernel.o $(LDLIBS)

hsvtorgbkernel.o:
	nvcc -g -G -arch sm_30 -c hsvtorgbkernel.cu

kernel.o:
	nvcc -g -G -arch sm_30 -c kernel.cu

clean :
	rm -f *.exe *.o
