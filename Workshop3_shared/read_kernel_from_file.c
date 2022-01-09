// C Program to demonstrat fscanf
#include <stdio.h>
  
// Driver Code
int main()
{
    FILE* ptr = fopen("outline.txt", "r");
    if (ptr == NULL) {
        printf("no such file.");
        return 0;
    }
  
    /* Assuming that abc.txt has content in below
       format
       NAME    AGE   CITY
       abc     12    hyderbad
       bef     25    delhi
       cce     65    bangalore */
    float buf;
    while (fscanf(ptr, "%f ", &buf) == 1)
        printf("%f\n", buf);
  
    return 0;
}