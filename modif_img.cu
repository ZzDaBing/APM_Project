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

// Kernel definition
__global__ void blur(unsigned int* d_img, unsigned int* d_tmp, int width, int height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){

    //
    int ida = ((idy * width) + idx) * 3;
    int avg_red = d_tmp[ida + 0];
    int avg_green = d_tmp[ida + 1];
    int avg_blue = d_tmp[ida + 2];

    //TOP BORDER
    if(idx < width && idy == 0){
      //Top-left corner
      if(ida == 0){ 
        avg_red += d_tmp[3] + d_tmp[(width * 3)];
        avg_green += d_tmp[4] + d_tmp[(width * 3) + 1];
        avg_blue += d_tmp[5] + d_tmp[(width * 3) + 2];

        avg_red /= 3;
        avg_green /= 3;
        avg_blue /= 3;
      }
      else{
        //Top-right corner
        if(ida == width - 1){ 
          avg_red += d_tmp[ida - 3] + d_tmp[ida + (width * 3)];
          avg_green += d_tmp[ida - 2] + d_tmp[ida + (width * 3) + 1];
          avg_blue += d_tmp[ida - 1] + d_tmp[ida + (width * 3) + 2];

          avg_red /= 3;
          avg_green /= 3;
          avg_blue /= 3;
        }
        else{
          avg_red += d_tmp[ida - 3] + d_tmp[ida + 3] + d_tmp[ida + (width * 3)];
          avg_green += d_tmp[ida - 2] + d_tmp[ida + 4] + d_tmp[ida + (width * 3) + 1];
          avg_blue += d_tmp[ida - 1] + d_tmp[ida + 5] + d_tmp[ida + (width * 3) + 2];

          avg_red /= 4;
          avg_green /= 4;
          avg_blue /= 4;
        }
      }
    }

    //BOTTOM BORDER
    if(idy == (height - 1)){
      //Bottom-left corner
      if(idx == 0){
        avg_red += d_tmp[ida + 3] + d_tmp[(ida - width * 3)];
        avg_green += d_tmp[ida + 4] + d_tmp[(ida - width * 3) + 1];
        avg_blue += d_tmp[ida + 5] + d_tmp[(ida - width * 3) + 2];

        avg_red /= 3;
        avg_green /= 3;
        avg_blue /= 3;
      }
      else{
        //Bottom-right corner
        if(idx == (width - 1)){
          avg_red += d_tmp[ida - 3] + d_tmp[ida - (width * 3)];
          avg_green += d_tmp[ida - 2] + d_tmp[ida - (width * 3) + 1];
          avg_blue += d_tmp[ida - 1] + d_tmp[ida - (width * 3) + 2];

          avg_red /= 3;
          avg_green /= 3;
          avg_blue /= 3;
        }
        else{
          avg_red += d_tmp[ida - 3] + d_tmp[ida + 3] + d_tmp[ida - (width * 3)];
          avg_green += d_tmp[ida - 2] + d_tmp[ida + 4] + d_tmp[ida - (width * 3) + 1];
          avg_blue += d_tmp[ida - 1] + d_tmp[ida + 5] + d_tmp[ida - (width * 3) + 2];

          avg_red /= 4;
          avg_green /= 4;
          avg_blue /= 4;
        }
      }
    }

    //LEFT BORDER (without corners)
    if( idx == 0 && idy != 0 && idy != height - 1 ){
      avg_red += d_tmp[(ida - width * 3)] + d_tmp[ida + 3] + d_tmp[(ida + width * 3)];
      avg_green += d_tmp[(ida - width * 3) + 1] + d_tmp[ida + 4] + d_tmp[(ida + width * 3) + 1];
      avg_blue += d_tmp[(ida - width * 3) + 2] + d_tmp[ida + 5] + d_tmp[(ida + width * 3) + 2];

      avg_red /= 4;
      avg_green /= 4;
      avg_blue /= 4;
    }

    //RIGHT BORDER (without corners)
    if( idx == width - 1 && idy != 0 && idy != height - 1 ){
      avg_red += d_tmp[(ida - width * 3)] + d_tmp[ida - 3] + d_tmp[(ida + width * 3)];
      avg_green += d_tmp[(ida - width * 3) + 1] + d_tmp[ida - 2] + d_tmp[(ida + width * 3) + 1];
      avg_blue += d_tmp[(ida - width * 3) + 2] + d_tmp[ida - 1] + d_tmp[(ida + width * 3) + 2];

      avg_red /= 4;
      avg_green /= 4;
      avg_blue /= 4;
    }

    //
    if( (idx > 0) && (idx < (width - 1)) && (idy > 0) && (idy < (height - 1)) ){
      avg_red += d_tmp[(ida - width * 3)] + d_tmp[ida - 3]  + d_tmp[ida + 3] + d_tmp[(ida + width * 3)];
      avg_green += d_tmp[(ida - width * 3) + 1] + d_tmp[ida - 2]  + d_tmp[ida + 4] + d_tmp[(ida + width * 3) + 1];
      avg_blue += d_tmp[(ida - width * 3) + 2] + d_tmp[ida - 1]  + d_tmp[ida + 5] + d_tmp[(ida + width * 3) + 2];

      avg_red /= 5;
      avg_green /= 5;
      avg_blue /= 5;
    }

    //Update pixel color
    d_img[ida + 0] = avg_red;
    d_img[ida + 1] = avg_green;
    d_img[ida + 2] = avg_blue;

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
    return printf("USAGE: %s <FILTER 1> [<FILTER 2> ...]\n FILTERS = satR, satG, satB, sym, grey, blur\n", argv[0]), 1;

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
    cudaMemcpy(d_img, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);

    if(strcmp(argv[i], "satR") == 0)
      saturation<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
    else if(strcmp(argv[i], "sym") == 0)
      symetry<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
    else if (strcmp(argv[i], "grey") == 0)
      grayscale<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
    else if (strcmp(argv[i], "blur") == 0){
      int blur_lvl = 100; //Default blur
      for (int i = 0; i < blur_lvl; ++i){
        cudaMemcpy(d_img, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
        blur<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
        cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      }
    }

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
