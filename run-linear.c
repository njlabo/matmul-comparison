
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

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
	int bias;		// if layer has bias the value equals to 1 else 0
} LinearConfig;

static void linear_org(float *xout, float *x, float *p, LinearConfig lc,
		       bool relu)
{
	// linear layer with ReLU activation: w(out,in) @ x (in,) + b(out,) -> xout (out,)
	int in = lc.in;
	int out = lc.out;

	float *w = p + lc.offset;
	float *b = w + in * out;
	#pragma omp parallel for
	for (int i = 0; i < out; i++) {
		float val = 0.0f;
		for (int j = 0; j < in; j++) {
			val += w[i * in + j] * x[j];
		}
		float bias_val = lc.bias ? b[i] : 0.0f;
		xout[i] = (relu) ? fmax(val + bias_val, 0.0f) : val + bias_val;
	}
}

static void linear_opt(float *xout, float *x, float *p, LinearConfig lc,
		       bool relu)
{
	int in = lc.in;
	int out = lc.out;

	float *w = p + lc.offset;
	float *b = w + in * out;

	cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in, 1.0f, w, in, x, 1,
		    0.0f, xout, 1);
	if (lc.bias)
		cblas_saxpy(out, 1.0f, b, 1, xout, 1);

	if (relu) {
	#pragma omp parallel for
		for (int i = 0; i < out; i++) {
			xout[i] = xout[i] > 0.0f ? x[i] : 0.0f;
		}
	}
}

int main(int argc, char **argv)
{
	// use non-optimized functions by default. If true, use functions from BLAS library
	bool optimize = false;
	if (argc == 2 && strcmp(argv[1], "blas") == 0) {
		optimize = true;
	}
	// read parameters
	char *path = "model.pt";
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);	// move file pointer to end of file
	size_t file_size = ftell(file);	// get the file size, in bytes
	fclose(file);
	int fd = open(path, O_RDONLY);	// open in read only mode
	if (fd == -1) {
		fprintf(stderr, "open failed!\n");
		exit(EXIT_FAILURE);
	}
	float *params = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (params == MAP_FAILED) {
		fprintf(stderr, "mmap failed!\n");
		exit(EXIT_FAILURE);
	}
	// initialize linear layer
	int dim = 4096;
	LinearConfig lc = { dim, dim, 0, 1 };

	// allocate memory for buffers and read input data
	float *x = malloc(dim * sizeof(float));
	float *x2 = malloc(dim * sizeof(float));
	file = fopen("data.txt", "rb");
	size_t rd = fread(x, sizeof(float), dim, file);
	if (rd != dim) {
		fprintf(stderr, "read failed!\n");
		exit(EXIT_FAILURE);
	}
	fclose(file);

	// forward pass
	if (optimize) {
		for (int i = 0; i < 50; i++) {
			linear_opt(x2, x, params, lc, false);
			linear_opt(x, x2, params, lc, false);
		}
	} else {
		for (int i = 0; i < 50; i++) {
			linear_org(x2, x, params, lc, false);
			linear_org(x, x2, params, lc, false);
		}
	}

	// print outputs
	for (int i = 0; i < dim; i++) {
		printf("%f\n", x[i]);
	}

	free(x);
	free(x2);
	munmap(params, file_size);
	close(fd);

	return 0;
}
