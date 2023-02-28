# CHPS - APM - Mini Project
Guillaume Bigand & Candice Astier
February 28, 2023

## Prerequisites

Library :
- FreeImage

Compilers :
- cuda_11.7


## Run
```bash
make
./modif_img <FLAGS/FILTER 1> [<FILTER 2> ...]
```

Input image example :
![image](https://user-images.githubusercontent.com/57614894/221928938-1536377f-6076-4851-a7ef-76ca9fce8216.png)

Available flags: ``--help``

--help : prints the usage and the flags/filters available

Available filters: ``satR``, ``sym``, ``grey``, ``blur``, ``sobel``, ``negative``, ``blue``, ``rotate``, ``resize``, ``photomaton``, ``popart``.

- satR : Sets red color at maximum value
![image](https://user-images.githubusercontent.com/57614894/221921291-8198cfe1-be66-45bc-a19a-0e2c1e91b010.png)
- sym : Horizontal symetry of the image
![image](https://user-images.githubusercontent.com/57614894/221922505-855301b7-8f1a-4b6e-b7af-ec6b3f3e951e.png)
- grey : Grays out the image
![image](https://user-images.githubusercontent.com/57614894/221922621-e881ccbe-991d-4db3-8105-d7a27d38eb29.png)
- blur : Blurs the image at a blur level (Default=100)
![image](https://user-images.githubusercontent.com/57614894/221922791-3972dddf-f646-4b0e-bf50-aeb7c4af827b.png)
- sobel : Applies the sobel filter
![image](https://user-images.githubusercontent.com/57614894/221922877-4d84876e-20f9-4fa4-b0f9-170a6102a9f7.png)
- negative : Inverses light degrees and colors
![image](https://user-images.githubusercontent.com/57614894/221923000-7c5b6c31-b872-47f1-8cc8-5d613b897fb2.png)
- blue : Sets red and green colors at 0
![image](https://user-images.githubusercontent.com/57614894/221923074-0a0ae5ea-f909-408d-bcec-fa08f16721e9.png)
- rotate : rotates the image at 90 degree
![image](https://user-images.githubusercontent.com/57614894/221999171-df743e94-7806-47a8-962b-f531759cc523.png)
- resize : resizes the image (Default is width/2 height/2)
![image](https://user-images.githubusercontent.com/57614894/221923218-c9ab1171-9d59-4c5c-9565-ef924634b147.png)
- photomaton : splits the image in 4 small ones
![image](https://user-images.githubusercontent.com/57614894/221923348-511caf16-7347-493a-a8bc-eabe26334d4e.png)
- popart : Splits the image in 4 small ones and applies filters on each (Default filters are satR, blue, symetry and negative)
![image](https://user-images.githubusercontent.com/57614894/221923446-f189a0d8-0d1f-4fa7-b5a5-7fb7aa8c713d.png)
