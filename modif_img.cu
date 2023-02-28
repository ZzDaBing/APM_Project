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
    d_img[ida + 0] = 255;
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

// Blur definition
__global__ void blur(unsigned int* d_img, unsigned int* d_tmp, int width, int height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
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

// Sobel Filter
__global__ void sobel(unsigned int* d_img, unsigned int* d_tmp, int width, int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
    int ida_1 = (((idy-1) * width) + idx) * 3;
    int ida_2 = ((idy     * width) + idx+1) * 3;
    int ida_3 = (((idy+1) * width) + idx+1) * 3;
    int ida_4 = (((idy-1) * width) + idx)   * 3;
    int ida_5 = ((idy     * width) + idx)   * 3;
    int ida_6 = (((idy+1) * width) + idx)   * 3;
    int ida_7 = (((idy-1) * width) + idx-1) * 3;
    int ida_8 = ((idy     * width) + idx-1) * 3;
    int ida_9 = (((idy+1) * width) + idx-1) * 3;

    int Gx = 0, Gy = 0;

    if (idy < height-1 && idy > 0 && idx < width-1 && idx > 0){
        Gx = -1 * d_tmp[ida_7] + d_tmp[ida_1]
            - 2 * d_tmp[ida_8] + 2 * d_tmp[ida_2]
            - d_tmp[ida_9] + d_tmp[ida_3];
        Gy = -1 * d_tmp[ida_7] - 2 * d_tmp[ida_4]
            - d_tmp[ida_1] + d_tmp[ida_9]
            + 2 * d_tmp[ida_6] + d_tmp[ida_3];
    }

    int sum = Gx * Gx + Gy * Gy;
    int res = sqrt((float)sum);
    d_img[ida_5] = res;
    d_img[ida_5 + 1] = res;
    d_img[ida_5 + 2] = res;
  }
}

// Negative Filter
__global__ void negative(unsigned int* d_img, unsigned int* d_tmp, int width, int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;
    d_img[ida + 0] = 255 - d_tmp[ida];
    d_img[ida + 1] = 255 - d_tmp[ida + 1];
    d_img[ida + 2] = 255 - d_tmp[ida + 2];
  }
}

// Only-one-color Filter
__global__ void only_blue(unsigned int* d_img, unsigned int* d_tmp, int width, int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;
    d_img[ida + 0] = 0;
    d_img[ida + 1] = 0;
    d_img[ida + 2] = d_tmp[ida + 2];
  }
}

// Rotate 45
__global__ void rotate45(unsigned int* d_img, unsigned int* d_tmp, int width, int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;
    int ida_2 = ((idx * width) + (height-idy-1)) * 3;
    d_img[ida_2] = d_tmp[ida];
    d_img[ida_2 + 1] = d_tmp[ida + 1];
    d_img[ida_2 + 2] = d_tmp[ida + 2];
  }
}

// Resize Filter
__global__ void resize(unsigned int* d_img, unsigned int* d_tmp, int width, int height, int new_width, int new_height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  int ida = (idy * width + idx) * 3;

  if(idy < height && idx < width) {
    d_img[ida] = 0;
    d_img[ida + 1] = 0;
    d_img[ida + 2] = 0;
  }

  if (idx < new_width && idy < new_height) {
    double scale = (double)new_width / (double)width;
    int idx_2 = (int)((double)idx / scale);
    int idy_2 = (int)((double)idy / scale);

    int ida_2 = (idy_2 * width + idx_2) * 3;

    d_img[ida] = d_tmp[ida_2];
    d_img[ida + 1] = d_tmp[ida_2 + 1];
    d_img[ida + 2] = d_tmp[ida_2 + 2];
  }
}

// Popart filter
__global__ void popart(unsigned int* d_img, unsigned int* d_tmp, int width, int height)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idy < height && idx < width){
    int ida = ((idy * width) + idx) * 3;

    int x,y;
    int w_d = (int)width / 2;
    int h_d = (int)height / 2;

    x = idx;
    y = idy;

    if(idx % 2 == 0) x = (int)(idx / 2);
    else{ x = (int)( ((idx - 1) / 2) + w_d );}
    if(idy % 2 == 0) y = (int)(idy / 2);
    else{ y = (int)( ((idy - 1) / 2) + h_d );}

    int idb = ((y * width) + x) * 3;
    
    d_img[idb + 0] = d_tmp[ida + 0];
    d_img[idb + 1] = d_tmp[ida + 1];
    d_img[idb + 2] = d_tmp[ida + 2];
  }
}

int main (int argc , char** argv)
{
  if(argc < 2)
    return printf("USAGE: %s <FILTER 1> [<FILTER 2> ...]\n FILTERS = satR, sym, grey, blur, sobel, negative, blue, rotate, resize, popart\n", argv[0]), 1;

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
      blur<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
      cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      for (int i = 1; i < blur_lvl; ++i){
        cudaMemcpy(d_img, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
        blur<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
        cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      }
    }
    else if (strcmp(argv[i], "sobel") == 0)
      sobel<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
    else if (strcmp(argv[i], "negative") == 0)
      negative<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
    else if (strcmp(argv[i], "blue") == 0)
      only_blue<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
    else if (strcmp(argv[i], "rotate") == 0){
      rotate45<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
      // Not working with sizes exchange
      // even if it would be logic
      //unsigned tmp = width;
      //width = height;
      //height = tmp;
    }
    else if (strcmp(argv[i], "resize") == 0)
      resize<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height, width/2, height/2);
    else if (strcmp(argv[i], "popart") == 0){
      //
      popart<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);
      cudaMemcpy(img, d_img, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

      //Update threads 
      nbBlocks.x /= 2;
      nbBlocks.y /= 2;
      nbBlocks.x++;
      nbBlocks.y++;

      //Streams
      cudaStream_t streams[4];
      for (int i = 0; i < 4; ++i)
        cudaStreamCreate(&streams[i]);

      //Small images
      unsigned int *topl, *topr, *botl, *botr, *d_topl, *d_topr, *d_botl, *d_botr, *d_tmptl, *d_tmptr, *d_tmpbl, *d_tmpbr;
      topl = (unsigned int*) malloc(sizeof(unsigned int) * 3 * ((width * height) / 2));
      topr = (unsigned int*) malloc(sizeof(unsigned int) * 3 * ((width * height) / 2));
      botl = (unsigned int*) malloc(sizeof(unsigned int) * 3 * ((width * height) / 2));
      botr = (unsigned int*) malloc(sizeof(unsigned int) * 3 * ((width * height) / 2));
      cudaMalloc(&d_topl, sizeof(unsigned int) * 3 * ((width * height) / 2));
      cudaMalloc(&d_topr, sizeof(unsigned int) * 3 * ((width * height) / 2)); 
      cudaMalloc(&d_botl, sizeof(unsigned int) * 3 * ((width * height) / 2)); 
      cudaMalloc(&d_botr, sizeof(unsigned int) * 3 * ((width * height) / 2));
      cudaMalloc(&d_tmptl, sizeof(unsigned int) * 3 * ((width * height) / 2));
      cudaMalloc(&d_tmptr, sizeof(unsigned int) * 3 * ((width * height) / 2));
      cudaMalloc(&d_tmpbl, sizeof(unsigned int) * 3 * ((width * height) / 2));
      cudaMalloc(&d_tmpbr, sizeof(unsigned int) * 3 * ((width * height) / 2));

      //Splits
      for (int i = 0; i < height / 2; ++i){
        for(int j = 0; j < width / 2; ++j){
          botl[(i * (width / 2) + j) * 3 + 0] = img[(i * width + j) * 3 + 0];
          botl[(i * (width / 2) + j) * 3 + 1] = img[(i * width + j) * 3 + 1];
          botl[(i * (width / 2) + j) * 3 + 2] = img[(i * width + j) * 3 + 2];

          botr[(i * (width / 2) + j) * 3 + 0] = img[(i * width + j) * 3 + 0];
          botr[(i * (width / 2) + j) * 3 + 1] = img[(i * width + j) * 3 + 1];
          botr[(i * (width / 2) + j) * 3 + 2] = img[(i * width + j) * 3 + 2];

          topl[(i * (width / 2) + j) * 3 + 0] = img[(i * width + j) * 3 + 0];
          topl[(i * (width / 2) + j) * 3 + 1] = img[(i * width + j) * 3 + 1];
          topl[(i * (width / 2) + j) * 3 + 2] = img[(i * width + j) * 3 + 2];

          topr[(i * (width / 2) + j) * 3 + 0] = img[(i * width + j) * 3 + 0];
          topr[(i * (width / 2) + j) * 3 + 1] = img[(i * width + j) * 3 + 1];
          topr[(i * (width / 2) + j) * 3 + 2] = img[(i * width + j) * 3 + 2];
        }
      }

      //Copy 
      cudaMemcpyAsync(d_topl, topl, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]);
      cudaMemcpyAsync(d_topr, topr, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[1]);
      cudaMemcpyAsync(d_botl, botl, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[2]);
      cudaMemcpyAsync(d_botr, botr, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[3]);
      cudaMemcpyAsync(d_tmptl, d_topl, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToDevice, streams[0]);
      cudaMemcpyAsync(d_tmptr, d_topr, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToDevice, streams[1]);
      cudaMemcpyAsync(d_tmpbl, d_botl, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToDevice, streams[2]);
      cudaMemcpyAsync(d_tmpbr, d_botr, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToDevice, streams[3]);
      cudaDeviceSynchronize();

      //Default filters applied
      sobel<<<nbBlocks, nbThreadsPerBlock, 0, streams[0]>>>(d_topl, d_tmptl, height / 2, width / 2); //TOP LEFT
      negative<<<nbBlocks, nbThreadsPerBlock, 0, streams[1]>>>(d_topr, d_tmptr, height / 2, width / 2); //TOP RIGHT
      saturation<<<nbBlocks, nbThreadsPerBlock, 0, streams[2]>>>(d_botl, d_tmpbl, height / 2, width / 2);  //BOT LEFT
      symetry<<<nbBlocks, nbThreadsPerBlock, 0, streams[3]>>>(d_botr, d_tmpbr, height / 2, width / 2);  //BOT RIGHT

      cudaMemcpyAsync(topl, d_topl, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[0]);
      cudaMemcpyAsync(topr, d_topr, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[1]);
      cudaMemcpyAsync(botl, d_botl, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[2]);
      cudaMemcpyAsync(botr, d_botr, 3 * ((width * height) / 2) * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[3]);

      //Regroups
      for (int i = 0; i < height / 2; ++i){
        for(int j = 0; j < width / 2; ++j){
          img[(i * width + j) * 3 + 0] = botl[(i * (width / 2) + j) * 3 + 0];
          img[(i * width + j) * 3 + 1] = botl[(i * (width / 2) + j) * 3 + 1];
          img[(i * width + j) * 3 + 2] = botl[(i * (width / 2) + j) * 3 + 2];

          img[(i * width + (width / 2) + j) * 3 + 0] = botr[(i * (width / 2) + j) * 3 + 0];
          img[(i * width + (width / 2) + j) * 3 + 1] = botr[(i * (width / 2) + j) * 3 + 1];
          img[(i * width + (width / 2) + j) * 3 + 2] = botr[(i * (width / 2) + j) * 3 + 2];

          img[((i + (height / 2)) * width + j) * 3 + 0] = topl[(i * (width / 2) + j) * 3 + 0];
          img[((i + (height / 2)) * width + j) * 3 + 1] = topl[(i * (width / 2) + j) * 3 + 1];
          img[((i + (height / 2)) * width + j) * 3 + 2] = topl[(i * (width / 2) + j) * 3 + 2];

          img[((i + (height / 2)) * width + (width / 2) + j) * 3 + 0] = topr[(i * (width / 2) + j) * 3 + 0];
          img[((i + (height / 2)) * width + (width / 2) + j) * 3 + 1] = topr[(i * (width / 2) + j) * 3 + 1];
          img[((i + (height / 2)) * width + (width / 2) + j) * 3 + 2] = topr[(i * (width / 2) + j) * 3 + 2];
        }
      }

      //Free memory

      for (int i = 0; i < 4; ++i)
        cudaStreamDestroy(streams[i]);

      cudaFree(d_topl);
      cudaFree(d_topr);
      cudaFree(d_botl);
      cudaFree(d_botr);
      free(topl);
      free(topr);
      free(botl);
      free(botr);
    }
    else
      printf("Unreconized filter : %s\n", argv[i]);

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    if (strcmp(argv[i], "popart") != 0)
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
