#include "filter.h"
#include "FreeImage.h"

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

#define BLOCK_WIDTH 32

using namespace std;

int main (int argc , char** argv)
{
  if(argc < 2)
    return printf("USAGE: %s <FLAGS/FILTER 1> [<FILTER 2> ...]\n FLAGS = --help, FILTERS = satR, sym, grey, blur, sobel, negative, blue, rotate, resize, photomaton, popart\n", argv[0]), 1;

  if(strcmp(argv[1], "--help") == 0){
      printf("\nUSAGE: %s <FLAGS/FILTER 1> [<FILTER 2> ...]\n FLAGS = --help, FILTERS = satR, sym, grey, blur, sobel, negative, blue, rotate, resize, photomaton, popart\n", argv[0]);
      printf("\n# FLAGS\n- --help : Prints this message\n");
      printf("\n# FILTERS\n \
        - satR : Sets red color at maximum value\n \
        - sym : Horizontal symetry of the image\n \
        - grey : Grays out the image\n \
        - blur : Blurs the image at a blur level (Default=100)\n \
        - sobel : Applies the sobel filter\n \
        - negative : Inverses light degrees and colors\n \
        - blue : Sets red and green colors at 0\n \
        - rotate : rotates the image at 90 degree\n \
        - resize : resizes the image (Default is width/2 height/2)\n \
        - photomaton : splits the image in 4 small ones\n \
        - popart : Splits the image in 4 small ones and applies filters on each (Default filters are satR, blue, symetry and negative)\n");
      return 0;
    }

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
    else if (strcmp(argv[i], "photomaton") == 0)
      photomaton<<<nbBlocks, nbThreadsPerBlock>>>(d_img, d_tmp, width, height);

    else if (strcmp(argv[i], "popart") == 0){
      //
      popart(nbBlocks, nbThreadsPerBlock, img, d_img, d_tmp, width, height);
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
