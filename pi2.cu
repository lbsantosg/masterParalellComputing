

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>

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

char name_file_in[255], name_file_out[255];
int kernel_sz, threads;

void loadImage(unsigned char **img, char *filename, int *width, int *height, int *channels)
{

    *img = stbi_load(filename, width, height, channels, 0);
    if (*img == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", *width, *height, *channels);
}




//const long long int  ITERATIONS = (long long ) 16e09;

/*****************************************************************************
/*kernel
*****************************************************************************/

__global__ void img_blr(unsigned char *input_img , unsigned char *output_img, int width, int height, int channels, int kernel_sz, int threads ){
    

    int threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
    int initIteration, endIteration;
    initIteration = (width / threads) * threadId; 
    endIteration =(threadId == threads-1? width : initIteration + (width/threads));

    for (int x = 0; x < height; x++ ) {
        for (int y = initIteration; y< endIteration; y++) {
            // para cada uno de los canales se debe realizar la convolucion 
            for (int ch = 0; ch < channels; ch++) {

                // el pixel al que se le esta realizando la convolucion
                unsigned char *current_pix = output_img + (x*width+y)*channels + ch;
                
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
    return;

}


/******************************************************************************/


int main(int argc, char *argv[])
{   
   if (argc != 6) {
      fprintf(stderr, "Usaste solo %d argumento(s), ingrese el nombre de la"
      "imagen de entrada, el nombre de la imagen de salida, el número de bloques, "
      " el de hílos por bloque y el tamaño del kernel\n", argc);

      exit(-1);
   }

    strcpy(name_file_in, argv[1]);
    strcpy(name_file_out, argv[2]);
    int kernel_sz = atoi(argv[5]);
    int threads_per_block = atoi(argv[4]);
    int num_blocks = atoi(argv[3]);

    printf("Num blocks %d , Num hilos por bloque %d , tamano kernel %d ", num_blocks,threads_per_block,kernel_sz);
    
    struct timeval  tv1, tv2; 
    gettimeofday(&tv1, NULL); 
    
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);


    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
   
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }


/*
    deviceProp.multiProcessorCount -> # MP
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) -> # cores per mp  
    deviceProp.sharedMemPerBlock -> shared memory
    deviceProp.regsPerBlock -> registers per block 

*/
    // Set device 0
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Number of multiprocessors %d \n", deviceProp.multiProcessorCount);
    printf("Number of cores per multiprocessor %d \n",_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("Shared memory per block: %zu \n", deviceProp.sharedMemPerBlock);

    long long cnt_shared_memory = deviceProp.sharedMemPerBlock;

    size_t size_pixel = sizeof(unsigned char) * 3;
  
    long long cnt_pixels_in_sm = cnt_shared_memory/size_pixel;
    long long sz_matrix_sm = sqrt(cnt_pixels_in_sm);
    printf("You can store %ld pixels in the shared memory, what will you do with that?\n", cnt_pixels_in_sm);
    printf("So you can store a matrix of %ld by %ld\n", sz_matrix_sm, sz_matrix_sm);
    printf("Registers per block: %d \n", deviceProp.regsPerBlock);

    // Variables for host
    // variables of input image of host
    int width, height, channels;
    unsigned char *h_input_img;
    unsigned char *h_output_img;


    //Variables for device
    unsigned char *d_input_img;
    unsigned char *d_output_img;

    // Load image data and allocate host memory

    loadImage(&h_input_img, name_file_in, &width, &height, &channels);


    
    size_t img_size = width * height * channels;
    h_output_img = (unsigned char*) malloc(img_size);

    int col_super_pixels =  width / sz_matrix_sm + ((width % sz_matrix_sm) != 0 ) ;
    int row_super_pixels =  height / sz_matrix_sm + ((height % sz_matrix_sm) != 0 );
    printf("cnt of super pixels in image %d x %d = %d\n", col_super_pixels, row_super_pixels, col_super_pixels*row_super_pixels);
    
    if (h_output_img == NULL) {
        fprintf(stderr, "Failed to allocate host array of output image!\n");
        exit(EXIT_FAILURE);
    }


    
    // Allocate device memory 
    cudaError_t err = cudaSuccess;


    err = cudaMalloc((void**)&d_input_img, img_size);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (input img) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_output_img, img_size);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (output img) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy image data from host to device 

    err = cudaMemcpy(d_input_img, h_input_img, img_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //int nBlocks = deviceProp.
    int total_threads = num_blocks * threads_per_block;
    img_blr<<<3, 192>>>(d_input_img, d_output_img, width, height, channels, kernel_sz, total_threads);



    // Copy data to host memory
    err = cudaMemcpy(h_output_img, d_output_img, img_size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // free device memory
    err = cudaFree(d_input_img);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output_img);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    // write output image file

    
    stbi_write_jpg(name_file_out, width, height, channels, h_output_img, 100);

    // stbi_write_png(name_file_out, width, height, channels, output_img, width*channels );

    stbi_image_free(h_input_img);
    free(h_output_img);

  

    gettimeofday(&tv2, NULL);  
    printf ("%fs time\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +(double) (tv2.tv_sec - tv1.tv_sec));
   
    puts("Program run succesfully");
    return 0;
}