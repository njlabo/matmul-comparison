
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <Accelerate/Accelerate.h>

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} LinearConfig;

static void linear_org(float *xout, float *x, float *p, LinearConfig lc, bool relu)
{
	// linear layer with ReLU activation: w(out,in) @ x (in,) + b(out,) -> xout (out,)
    int in = lc.in;
    int out = lc.out;

	int i;
	float *w = p + lc.offset;
	float *b = w + in * out;
	#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		for (int j = 0; j < in; j++) {
			val += w[i * in + j] * x[j];
		}
        xout[i] = (relu) ? fmax(val + b[i], 0.0f) : val + b[i];
	}
}

static void linear_opt(float *xout, float *x, float *p, LinearConfig lc, bool relu) {
    int in = lc.in;
    int out = lc.out;

    float *w = p + lc.offset;
    float *b = w + in * out;

    vDSP_mmul(w, 1, x, 1, xout, 1, out, 1, in);
    vDSP_vadd(b, 1, xout, 1, xout, 1, out);

    if (relu) {
        #pragma omp parallel for
        for (int i = 0; i < out; i++) {
            xout[i] = fmax(xout[i], 0.0f);
        }
    }
}

int main(int argc, char** argv) {
    // use non-optimized functions by default. If true, use functions from BLAS library
    bool optimize = false; 
    if (argc == 2 && strcmp(argv[1], "blas") == 0) {
        optimize = true;
    }
    // read parameters
    char *path = "model.pt";
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    size_t file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    int fd = open(path, O_RDONLY); // open in read only mode
    if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    float *params = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (params == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    // initialize linear layer
    int in = 4096;
    int out = 4096;
    LinearConfig lc = { in, out, 0 };

    // allocate memory for buffers and read input data
    float *x = malloc(in * sizeof(float));
    float *x2 = malloc(out * sizeof(float));
    file = fopen("data.txt", "rb");
    size_t rd = fread(x, sizeof(float), in, file);
    if (rd != in) { fprintf(stderr, "read failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    // forward pass
    if (optimize) {
        for (int i = 0; i < 500; i++) {
            linear_opt(x2, x, params, lc, false);
            linear_opt(x, x2, params, lc, false);
        }
    } else {
        for (int i = 0; i < 500; i++) {
            linear_org(x2, x, params, lc, false);
            linear_org(x, x2, params, lc, false);
        }
    }

    // print outputs
    for (int i = 0; i < out; i++) {
        printf("%f\n", x[i]);
    }

    free(x);
    free(x2);
    munmap(params, file_size);
    close(fd);

    return 0;
}
