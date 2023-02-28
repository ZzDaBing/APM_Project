default: all

all:
	nvcc filter.cu -I${HOME}/softs/FreeImage/include modif_img.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o modif_img.exe

clean:
	rm -f *.o modif_img.exe
