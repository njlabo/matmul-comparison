
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
#include <dirent.h>
#include <sys/stat.h>
#include <stdbool.h>

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
	int bias;		// if layer has bias the value equals to 1 else 0
} ConvConfig;

static float im2col_get_pixel(float *im, int height, int width, int row,
			      int col, int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

// TODO: BLAS may be used here
static void im2col_cpu(float *col, float *im, int *height, int *width,
		       ConvConfig cc)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
	int nchannels = cc.ic;
	int ksize = cc.ksize;
	int stride = cc.stride;
	int pad = cc.pad;

	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	#pragma omp parallel for
	for (int c = 0; c < col_size; c++) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int channel = c / ksize / ksize;
		for (int h = 0; h < out_height; h++) {
			for (int w = 0; w < out_width; w++) {
				int input_row = h_offset + h * stride;
				int input_col = w_offset + w * stride;
				int col_index =
				    (c * out_height + h) * out_width + w;
				col[col_index] =
				    im2col_get_pixel(im, *height, *width,
						     input_row, input_col,
						     channel, pad);
			}
		}
	}
	// update current height and width
	*height = out_height;
	*width = out_width;
}

static void matmul_conv_org(float *xout, float *x, float *p, ConvConfig cc,
			    int out, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(nchannels,) -> xout (nchannels,out)
	int nchannels = cc.oc;
	int in = cc.ic * cc.ksize * cc.ksize;

	float *w = p + cc.offset;
	float *b = w + nchannels * in;
	#pragma omp parallel for
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
			float bias_val = (cc.bias) ? b[c] : 0.0f;
			xout[c * out + i] =
			    relu ? fmax(val + bias_val,
					0.0f) : (val + bias_val);
		}
	}
}

static void matmul_conv_opt(float *xout, float *x, float *p, ConvConfig cc,
			    int out, bool relu)
{
	int nchannels = cc.oc;
	int in = cc.ic * cc.ksize * cc.ksize;

	float *w = p + cc.offset;
	float *b = w + nchannels * in;

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    nchannels, out, in, 1.0f, w, in, x, out, 0.0f, xout, out);
	// TODO: current bias addition vs. reshaping bias to (nchannels, out) and adding elementwise without need of for loop
	if (cc.bias) {
	#pragma omp parallel for
		for (int c = 0; c < nchannels; c++) {
			cblas_saxpy(out, 1.0f, &b[c], 0, &xout[c * out], 1);
		}
	}

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
	// initialize convolutional layers
	int h = 28, w = 28;
	int nch_in = 3, nch = 64;
	int ks = 3, st = 1, pad = 1;

	ConvConfig cc_in = { ks, st, pad, nch_in, nch, 0, 1 };
	ConvConfig cc =
	    { ks, st, pad, nch, nch, nch_in * nch * ks * ks + nch, 1 };

	// allocate memory and read input data
	float *x = malloc(nch * h * w * sizeof(float));
	float *x2 = malloc(nch * h * w * ks * ks * sizeof(float));
	file = fopen("data.txt", "rb");
	size_t rd = fread(x, sizeof(float), nch_in * h * w, file);
	if (rd != nch_in * h * w) {
		fprintf(stderr, "read failed!\n");
		exit(EXIT_FAILURE);
	}
	fclose(file);

	// forward pass
	if (optimize) {
		im2col_cpu(x2, x, &h, &w, cc_in);
		matmul_conv_opt(x, x2, params, cc_in, h * w, false);
		for (int i = 0; i < 100; i++) {
			im2col_cpu(x2, x, &h, &w, cc);
			matmul_conv_opt(x, x2, params, cc, h * w, false);
		}
	} else {
		im2col_cpu(x2, x, &h, &w, cc_in);
		matmul_conv_org(x, x2, params, cc_in, h * w, false);
		for (int i = 0; i < 100; i++) {
			im2col_cpu(x2, x, &h, &w, cc);
			matmul_conv_org(x, x2, params, cc, h * w, false);
		}
	}

	// print outputs
	for (int i = 0; i < nch * h * w; i++) {
		printf("%f\n", x[i]);
	}

	free(x);
	free(x2);
	munmap(params, file_size);
	close(fd);

	return 0;
}
