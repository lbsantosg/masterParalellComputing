#include<stdio.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include<math.h>
// librerias para proc de imagenes

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"


#define mymax( x,  y) (x > y ? x : y)
#define mymin( x,  y) (x < y ? x : y)

// variables of input image
unsigned char *input_img;
unsigned char *output_img;

char name_file_in[255], name_file_out[255],write_file[255];
int kernel_sz;


void loadImage(unsigned char **img, char *filename, int *width, int *height, int *channels)
{

    *img = stbi_load(filename, width, height, channels, 0);
    if (*img == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }
    //printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", *width, *height, *channels);
}

int main(int argc, char *argv[])
{   

    int kernel_sz;
	  FILE * params_file; 
		params_file = fopen("params_file.txt","r");
		fscanf(params_file, "%d %s %s %s",&kernel_sz,name_file_in,name_file_out,write_file);
		fclose(params_file); 

    struct timeval  tv1, tv2; 
    gettimeofday(&tv1, NULL); 

    int width, height, channels;
    unsigned char *input_img;
    unsigned char *output_img;


    // Load image data and allocate host memory
    loadImage(&input_img, name_file_in, &width, &height, &channels);
    
    size_t img_size = width * height * channels;
		output_img = (unsigned char*) malloc(img_size);
    
    if (output_img == NULL) {
        fprintf(stderr, "Failed to allocate host array of output image!\n");
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv1, NULL);
    // calculate image 

    int i, tag=0, tasks, iam, root=0, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam); 
	 	if (iam == 0)    
        for ( i = 0 ; i < img_size ; i ++){
    		output_img[i] = input_img[i];	
    	}
    int nIters = height / tasks;
    int initIteration = nIters * iam;
    int endIteration = initIteration + nIters -1 ; 

		unsigned char *buff2send = (unsigned char*) malloc(sizeof(unsigned char) * width * nIters*channels);
		for (int x = initIteration, xr = 0 ; x <= endIteration; x++ , xr++ ) {
        for (int y = 0; y< width; y++) {
            // para cada uno de los canales se debe realizar la convolucion 
            for (int ch = 0; ch < channels; ch++) {

                // el pixel al que se le esta realizando la convolucion
                unsigned char *current_pix =buff2send + (xr*width+y)*channels + ch;
                
                int nval = 0;
                // se itera por una matriz del tamano del kernel
                for (int nx = x - (kernel_sz)/2; nx < x+(kernel_sz)/2 + (kernel_sz&1); nx++) {
                    for (int ny = y - (kernel_sz)/2; ny < y+(kernel_sz)/2 + (kernel_sz&1); ny++) {
                        int xi = mymax(0, nx);
                        xi = mymin(height-1, xi);
                        int yi = mymax(0, ny);
                        yi = mymin(width-1, yi);

                        unsigned char* npix = input_img + (xi*width+yi)*channels+ch;

                        nval += *npix;
                    }
                }
                nval = nval/( kernel_sz*kernel_sz);
             		*current_pix = (unsigned char) nval;
						}
        }
    }  
    MPI_Gather((void *)buff2send, width * nIters*channels, MPI_UNSIGNED_CHAR, output_img, width * nIters*channels, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
   	free(buff2send);
    MPI_Finalize();   
    if (iam == 0 ) {
    	stbi_write_jpg(name_file_out, width, height, channels, output_img, 100);

    // stbi_write_png(name_file_out, width, height, channels, output_img, width*channels );
		}
		stbi_image_free(input_img);
		free(output_img);
		


    gettimeofday(&tv2, NULL);  
    if (iam == 0) {
		   FILE * time_file;
	  	 time_file = fopen(write_file,"a");	
			 fprintf (time_file, "%f", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +(double) (tv2.tv_sec - tv1.tv_sec));
       fclose(time_file); 
		}
		
    
   // puts("Program run succesfully");
    return 0;
}
