#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int main(void) {
    int width, height, channels;
    unsigned char *img = stbi_load("mario.png", &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

// convert the input image to grey
    size_t img_size = width * height * channels;
    int gray_channels = channels == 4 ? 2 : 1;
    size_t gray_img_size = width * height * gray_channels;

    unsigned char *gray_img = malloc(gray_img_size);
    if (gray_img == NULL) {
         puts("Unable to allocate memory for the gray image.\n");
         exit(1);
    }

    for (unsigned char *p = img, *pg = gray_img; p != img + img_size; p+=channels, pg += gray_channels ) {
        *pg = (uint8_t)((*p+*(p+1)+*(p+3))/3.0);
        if ( channels == 4) {
             *(pg + 1) = *(p+3);
        }
    }

    stbi_write_png("test.png", width, height, gray_channels, gray_img, width*gray_channels);

    stbi_image_free(img);
    free(gray_img);
}