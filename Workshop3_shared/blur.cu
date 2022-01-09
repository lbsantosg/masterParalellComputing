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
#define PrintGpuDesc false


//float K[3][3] = {{-1,-1,-1}, {-1, 8, -1}, {-1,-1,-1}}; // Outline
//float K[3][3] = {{0,-1,0}, {-1, 5, -1}, {0,-1,0}}; // Sharpen
//float K[3][3] = {{0.11111111111,0.11111111111,0.11111111111}, {0.11111111111, 0.11111111111, 0.11111111111}, {0.11111111111,0.11111111111,0.11111111111}}; // blur
// 

void loadImage(unsigned char **img, char *filename, int *width, int *height, int *channels)
{

    *img = stbi_load(filename, width, height, channels, 0);
    if (*img == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }
   // printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", *width, *height, *channels);
}




//const long long int  ITERATIONS = (long long ) 16e09;

/*****************************************************************************
/*kernel
*****************************************************************************/

__global__ void img_blr(unsigned char *input_img , unsigned char *output_img, int width, int height, int channels, int kernel_sz, float *K,
    int n_sp, int sz_sp, int blocks, int threads ){
    
    // float K...

    int init_iter_sp = (n_sp / blocks) *  blockIdx.x;
    int end_iter_sp = (blockIdx.x == blocks-1 ? n_sp : init_iter_sp + (n_sp/blocks)); 
    

    /* very very very important 128 = (sz_sp+kernel_sz-1) equivalent to the maz size of matrix of pixels(3char) you can 
    store in shared memory, i couldnt set as variable to use as matrix size in the  definition of img_portion, thats
    why i put that [128][128][3]*/


    __shared__ unsigned char img_portion[128][128][3];

    int total_pixels_to_copy_per_step = 128 * 128;

    int threadId = threadIdx.x;


    int initIteration, endIteration;
    initIteration = (total_pixels_to_copy_per_step / threads) * threadId; 
    endIteration =(threadId == threads-1? total_pixels_to_copy_per_step : initIteration + (total_pixels_to_copy_per_step/threads));
    
    int col_super_pixels =  width / sz_sp +  ((width % sz_sp) != 0 ) ;

    int total_pixels_to_process_per_step = sz_sp*sz_sp;
    int initIterationProc = (total_pixels_to_process_per_step / threads) * threadId; 
    int endIterationProc = (threadId == threads-1? total_pixels_to_process_per_step : initIterationProc + (total_pixels_to_process_per_step/threads));
    

    for (int st = init_iter_sp; st < end_iter_sp; st++) {
        int x = (st / col_super_pixels)*sz_sp - kernel_sz/2;
        int y = (st % col_super_pixels)*sz_sp - kernel_sz/2;


        /// for each step we copy the portion of the image in shared memory
        for (int i = initIteration; i < endIteration; i++) {
            int xr = i / (sz_sp + kernel_sz - 1);
            int yr = i % (sz_sp + kernel_sz - 1);
            int xg =  x + xr;
            xg = mymax(0, xg);
            xg = mymin(height-1, xg);
            int yg = y + yr;
            yg  = mymax(0, yg);
            yg = mymin(width-1, yg);

            for (int ch = 0; ch < channels; ch++) {
                unsigned char* npix = input_img + (xg*width+yg)*channels+ch;
                img_portion[xr][yr][ch] = *npix;
            }            
        }
        __syncthreads();

        x = (st / col_super_pixels)*sz_sp;
        y = (st % col_super_pixels)*sz_sp;

        /// for each step we process the portion of the image in shared memory
        for (int i = initIterationProc; i < endIterationProc; i++) {
            int xr = (i / (sz_sp))+(kernel_sz/2);
            int yr = (i % (sz_sp))+(kernel_sz/2);
            int xg = x + xr - kernel_sz/2, yg = y + yr - kernel_sz/2;
            if( xg >= height or yg >= width) continue;
            for (int ch = 0; ch < channels; ch++) {
                unsigned char *current_pix_out = output_img + (xg*width+yg)*channels + ch;
                float f_nval = 0;
                for (int nx = xr - (kernel_sz)/2, xk = 0; nx < xr+(kernel_sz)/2 + (kernel_sz&1); nx++, xk++) {
                    for (int ny = yr - (kernel_sz)/2, yk =0; ny < yr+(kernel_sz)/2 + (kernel_sz&1); ny++, yk++) {
                        f_nval += img_portion[nx][ny][ch] * K[xk*kernel_sz + yk];
                    }                                
                }
                int nval = f_nval;
                nval = mymax(0, nval);
                nval = mymin(nval, 255);
                *current_pix_out = (unsigned char) nval;
            }
            
          
            
        }
        __syncthreads();
    }
  
    return;

}


/******************************************************************************/

char name_file_in[255], name_file_out[255];

int main(int argc, char *argv[])
{   
   
    if (argc != 6) {
        fprintf(stderr, "Usaste solo %d argumento(s), ingrese el nombre de la"
        "imagen de entrada, el nombre de la imagen de salida, el número de bloques, "
        " el de hílos por bloque y nombre del archivo del kernel\n", argc);
  
        exit(-1);
    }

    strcpy(name_file_in, argv[1]);
    strcpy(name_file_out, argv[2]);
    int kernel_sz = 3;
    int threads_per_block = atoi(argv[4]);
    int num_blocks = atoi(argv[3]);

    //Read kernel info from file;
    FILE *kernel_file = fopen(argv[5], "r");
    fscanf(kernel_file, "%d", &kernel_sz);
    size_t mat_sz = kernel_sz * kernel_sz * sizeof(float);
    float *K = (float*) malloc(mat_sz);  // Outline
    if (kernel_sz > 50) {
        fprintf(stderr,
        "Tamano de kernel superior a 50");
        exit(-1);
    }
    for (int i=0; i<kernel_sz * kernel_sz; i++) fscanf(kernel_file, "%f", &K[i]);

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
        exit(-1);
    } else {
        if (PrintGpuDesc)
            printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }



    // Set device 0
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (PrintGpuDesc){
        printf("Number of multiprocessors %d \n", deviceProp.multiProcessorCount);
        printf("Number of cores per multiprocessor %d \n",_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        printf("Shared memory per block: %zu \n", deviceProp.sharedMemPerBlock);
    }
    long long cnt_shared_memory = deviceProp.sharedMemPerBlock;

    size_t size_pixel = sizeof(unsigned char) * 3;
  
    long long cnt_pixels_in_sm = cnt_shared_memory/size_pixel;
    long long sz_matrix_sm = sqrt(cnt_pixels_in_sm);
    int sz_super_pixel = sz_matrix_sm - kernel_sz + 1;
    
    if (PrintGpuDesc){
        printf("You can store %ld pixels in the shared memory, what will you do with that?\n", cnt_pixels_in_sm);
        printf("So you can store a matrix of %ld by %ld\n", sz_matrix_sm, sz_matrix_sm);
        printf("Registers per block: %d \n", deviceProp.regsPerBlock);
        printf("kernel sz %d sp sz %d\n", kernel_sz, sz_super_pixel);
    }

    // Variables for host
    // variables of input image of host
    int width, height, channels;
    unsigned char *h_input_img;
    unsigned char *h_output_img;


    //Variables for device
    unsigned char *d_input_img;
    unsigned char *d_output_img;
    float *d_K;
    // Load image data and allocate host memory
    loadImage(&h_input_img, name_file_in, &width, &height, &channels);



    
    size_t img_size = width * height * channels;
    h_output_img = (unsigned char*) malloc(img_size);

    int col_super_pixels =  width / sz_super_pixel + ((width % sz_super_pixel) != 0 ) ;
    int row_super_pixels =  height / sz_super_pixel + ((height % sz_super_pixel) != 0 );
    int total_super_pixels = col_super_pixels*row_super_pixels;

    if (h_output_img == NULL) {
        fprintf(stderr, "Failed to allocate host array of output image!\n");
        exit(EXIT_FAILURE);
    }


    
    // Allocate device memory 
    cudaError_t err = cudaSuccess;


    err = cudaMalloc((void**)&d_K, mat_sz);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (input img) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


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


    err = cudaMemcpy(d_K, K, mat_sz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   
  

    img_blr<<<num_blocks, threads_per_block>>>(d_input_img, d_output_img, width, height, channels, kernel_sz, d_K, total_super_pixels, sz_super_pixel, num_blocks,  threads_per_block);



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



    gettimeofday(&tv2, NULL);  
    printf ("%f", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +(double) (tv2.tv_sec - tv1.tv_sec));
     // write output image file

    
    stbi_write_jpg(name_file_out, width, height, channels, h_output_img, 100);

    // stbi_write_png(name_file_out, width, height, channels, output_img, width*channels );

    stbi_image_free(h_input_img);
    free(h_output_img);
    free(K);
  
   // puts("Program run succesfully");
    return 0;
}