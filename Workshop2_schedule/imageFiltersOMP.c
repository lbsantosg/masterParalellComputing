
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>

// librerias para proc de imagenes

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int mymax(int x, int y) { return x > y ? x : y; }
int mymin(int x, int y) { return x < y ? x : y; }

// variables of input image
int width, height, channels;
unsigned char *input_img;
unsigned char *output_img;

char name_file_in[255], name_file_out[255];
float K[5][5] = {{-1,-1,-1}, {-1, 8, -1}, {-1,-1,-1}}; // Outline
//float K[3][3] = {{0,-1,0}, {-1, 5, -1}, {0,-1,0}}; // Sharpen
//float K[3][3] = {{0.11111111111,0.11111111111,0.11111111111}, {0.11111111111, 0.11111111111, 0.11111111111}, {0.11111111111,0.11111111111,0.11111111111}}; // blur
// 
int kernel_sz = 5, threads;
int K_sz = 5;
// order file_in_name file_out_name threads kernel_sz

void loadImage(unsigned char **img, char *filename, int *width, int *height, int *channels)
{

    *img = stbi_load(filename, width, height, channels, 0);
    if (*img == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }
    //printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n",
    //*width, *height, *channels);
}


int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        fprintf(stderr, "Usaste solo %d argumento(s), ingrese el nombre de la"
                        "imagen de entrada, el nombre de la imagen de salida, el número de"
                        " hílos y el tamaño del kernel\n",
                argc);

        exit(-1);
    }

    strcpy(name_file_in, argv[1]);
    strcpy(name_file_out, argv[2]);
    threads = atoi(argv[3]);

    loadImage(&input_img, name_file_in, &width, &height, &channels);
    size_t img_size = width * height * channels;
    output_img = malloc(img_size);

    // get init time
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for
        for (int y = 0; y < width; y++)
        {
         //   printf("%d %d\n", y, omp_get_thread_num());
            for (int x = 0; x < height; x++)
            {
                for (int ch = 0; ch < channels; ch++) {

                // el pixel al que se le esta realizando la convolucion
                    unsigned char *current_pix = output_img + (x*width+y)*channels + ch;
                    
                    float f_nval = 0;
                    // se itera por una matriz del tamano del kernel
                    for (int nx = x - (K_sz)/2, xk =0 ; nx < x+(K_sz)/2 + (K_sz&1); nx++, xk++) {
                        for (int ny = y - (K_sz)/2, yk = 0; ny < y+(K_sz)/2 + (K_sz&1); ny++, yk++) {
                            int xi = mymax(0, nx);
                            xi = mymin(height-1, xi);
                            int yi = mymax(0, ny);
                            yi = mymin(width-1, yi);

                            unsigned char* npix = input_img + (xi*width+yi)*channels+ch;

                            f_nval += *npix * K[xk][yk] ; 
                        }

                    }
                    int nval = f_nval;
                    nval = mymax(0, nval);
                    nval = mymin(nval, 255);
                    *current_pix = (uint8_t) nval;
                }

            }
        }
    }
    

    // get end time
    gettimeofday(&tv2, NULL);
    printf("%f ", (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec));

    // write output image file
    stbi_write_jpg(name_file_out, width, height, channels, output_img, 100);

    // stbi_write_png(name_file_out, width, height, channels, output_img, width*channels );

    stbi_image_free(input_img);
    free(output_img);
    return 0;
}