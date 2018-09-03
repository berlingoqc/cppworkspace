LDLIBS=-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lcuda -lcudart
EXEC_NAME=testcuda.exe

all : $(EXEC_NAME)

$(EXEC_NAME): kernel.o
	g++ -g -I/opt/cuda/include -L/opt/cuda/lib64 -o $(EXEC_NAME) testcuda.cpp kernel.o $(LDLIBS)

kernel.o:
	nvcc -g -G -arch sm_30 -c kernel.cu

clean :
	rm *.exe *.o
