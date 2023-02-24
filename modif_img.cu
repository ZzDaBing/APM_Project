#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "FreeImage.h"

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

#define BLOCK_WIDTH 32

using namespace std;

// Saturation Filter
__global__ void saturation(unsigned int* d_img, unsigned int* d_tmp, int width, int height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  
  if(idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;
    d_img[ida + 0] = 255;  //No red on img
    d_img[ida + 1] = d_tmp[ida + 1];
    d_img[ida + 2] = d_tmp[ida + 2];
  }
}

// Symetry Filter
__global__ void symetry(unsigned int* d_img, unsigned int* d_tmp, int width, int height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;
    int idinvers = ((width * height) - ((idy * width) + idx)) * 3;
    d_img[ida + 0] = d_tmp[idinvers];
    d_img[ida + 1] = d_tmp[idinvers + 1];
    d_img[ida + 2] = d_tmp[idinvers + 2];
  }
}

// Grayscale Filter
__global__ void grayscale(unsigned int* d_img, unsigned int* d_tmp, int width, int height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;
    double val = (0.299*d_tmp[ida + 0]) + (0.587*d_tmp[ida + 1]) + (0.114*d_tmp[ida + 2]);
    d_img[ida + 0] = (int)val;
    d_img[ida + 1] = (int)val;
    d_img[ida + 2] = (int)val;
  }
}

int main (int argc , char** argv)
{
  if(argc < 2)
    return printf("USAGE: %s <FILTER 1> [<FILTER 2> ...]\n FILTERS = satR, satG, satB, sym, grey\n", argv[0]), 1;

  FreeImage_Initialise();
  const char *PathName = "img.jpg";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);
  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);
  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  unsigned int *img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);

  //Allocates arrays on GPU
  unsigned int *d_img, *d_tmp;
  cudaMalloc(&d_img, sizeof(unsigned int) * 3 * width * height);
  cudaMalloc(&d_tmp, sizeof(unsigned int) * 3 * width * height);

  //Init
  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  //Init blockdim
  dim3 nbThreadsPerBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);

  int dimx = 0, dimy = 0;
  if((width) % BLOCK_WIDTH) dimx++;
  if((height) % BLOCK_WIDTH) dimy++;
  dim3 nbBlocks( (width / nbThreadsPerBlock.x)+dimx, (height / nbThreadsPerBlock.y)+dimy, 1);
  cudaError_t cudaerr;

  //##############################

  for (int i = 1; i < argc; ++i)
  {
    if(strcmp(argv[i], "satR") == 0)
    {
      cudaMemcpy(d_img, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
      saturation<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);

      cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

      cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      printf("Red Saturation Done !\n");
    }
    else if(strcmp(argv[i], "sym") == 0)
    {
      cudaMemcpy(d_img, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
      symetry<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);

      cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

      cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      printf("Symetry Done !\n");
    }
    else if (strcmp(argv[i], "grey") == 0)
    {
      cudaMemcpy(d_img, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
      grayscale<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);

      cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));

      cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      printf("Grayscale Done !\n");
    }
  }

  //##############################

  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
    // next line
    bits += pitch;
  }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !

  cudaFree(d_img);
  cudaFree(d_tmp);
  free(img);
}
