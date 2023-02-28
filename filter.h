#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void saturation(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void symetry(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void blur(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void grayscale(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void sobel(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void negative(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void only_blue(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void rotate45(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
__global__ void resize(unsigned int* d_img, unsigned int* d_tmp, int width, int height, int new_width, int new_height);
__global__ void photomaton(unsigned int* d_img, unsigned int* d_tmp, int width, int height);
void popart(dim3 nbBlocks, dim3 nbThreadsPerBlock, unsigned int *img, unsigned int *d_img, unsigned int *d_tmp, int width, int height);
