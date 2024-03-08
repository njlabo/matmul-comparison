
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
#include <Accelerate/Accelerate.h>

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} ConvConfig;

static float im2col_get_pixel(float *im, int height, int width, int row, int col,
		       int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

// TODO: BLAS may be used here
static void im2col_cpu(float *col, float *im, int *height, int *width, ConvConfig cc)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
    int nchannels = cc.ic;
    int ksize = cc.ksize;
    int stride = cc.stride;
    int pad = cc.pad;

	int c, h, w;
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
    #pragma omp parallel for private(c, h, w)
	for (c = 0; c < col_size; c++) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int channel = c / ksize / ksize;
		for (h = 0; h < out_height; h++) {
			for (w = 0; w < out_width; w++) {
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

static void matmul_conv_org(float *xout, float *x, float *p, ConvConfig cc, int out, bool bias, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(nchannels,) -> xout (nchannels,out)
    int nchannels = cc.oc;
    int in = cc.ic * cc.ksize * cc.ksize;

	int c;
	float *w = p + cc.offset;
    float *b = w + nchannels * in;
	#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
            float bias_val = (bias) ? b[c] : 0.0f;
			xout[c * out + i] = relu ? fmax(val + bias_val, 0.0f) : (val + bias_val);
		}
	}
}

static void matmul_conv_opt(float *xout, float *x, float *p, ConvConfig cc, int out, bool bias, bool relu) {
    int nchannels = cc.oc;
    int in = cc.ic * cc.ksize * cc.ksize;

    float *w = p + cc.offset;
    float *b = w + nchannels * in;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nchannels, out, in,
                1.0f, w, in, x, out, 0.0f, xout, out);

    if (bias) {
        #pragma omp parallel for
        for (int c = 0; c < nchannels; c++) {
            cblas_saxpy(out, 1.0f, &b[c], 0, &xout[c * out], 1);
        }
    }

    if (relu) {
        #pragma omp parallel for
        for (int i = 0; i < nchannels * out; i++) {
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

    // initialize convolutional layers
    ConvConfig cc_in = { 3, 1, 1, 3, 64, 0};
    ConvConfig cc = { 3, 1, 1, 64, 64, 3 * 64 * 3 * 3 + 64};

    // allocate memory and read input data
    float *x = malloc(64*28*28 * sizeof(float));
    float *x2 = malloc(64*9*28*28 * sizeof(float));
    file = fopen("data.txt", "rb");
    size_t rd = fread(x, sizeof(float), 3 * 28 * 28, file);
    if (rd != 3 * 28 * 28) { fprintf(stderr, "read failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    int h = 28;
    int w = 28;

    // forward pass
    if (optimize) {
        im2col_cpu(x2, x, &h, &w, cc_in);
        matmul_conv_opt(x, x2, params, cc_in, h * w, true, false);
        for (int i = 0; i < 100; i++) {
            im2col_cpu(x2, x, &h, &w, cc);
            matmul_conv_opt(x, x2, params, cc, h * w, true, false);
        }
    } else {
        im2col_cpu(x2, x, &h, &w, cc_in);
        matmul_conv_org(x, x2, params, cc_in, h * w, true, false);
        for (int i = 0; i < 100; i++) {
            im2col_cpu(x2, x, &h, &w, cc);
            matmul_conv_org(x, x2, params, cc, h * w, true, false);
        }
    }

    // print outputs
    for (int i = 0; i < 64 * 28 * 28; i++) {
        printf("%f\n", x[i]);
    }

    free(x);
    free(x2);
    munmap(params, file_size);
    close(fd);

    return 0;
}
