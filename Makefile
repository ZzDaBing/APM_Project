default: all

all: cuda

cuda:
	nvcc -I/home/canastier/softs/FreeImage/include -L/home/canastier/softs/FreeImage/lib/ -lfreeimage -o modif_img modif_img.cu

cpp:
	g++ -I${HOME}/softs/FreeImage/include modif_img.cpp -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o modif_img

clean:
	rm -f *.o modif_img
