/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
//#define N_IMAGES 10000
#define N_IMAGES 1

#define NUM_THREADS 256
#define NUM_BLKS (IMG_HEIGHT*IMG_WIDTH)/NUM_THREADS


typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ uchar arr_min(uchar arr[], int arr_size) {
    return 0; //TODO
}

__device__ void prefix_sum(int arr[], int arr_size, int histogram[]) {
 
// TODO: can it be MORE parallel ?
    for (int i = 1; i < (arr_size-2)/2; i++) {

		arr[i] = arr[i-1] + histogram[i];
		arr[255 - i] = arr[(arr_size-1) - (i-1) ] - histogram[(arr_size-1) - (i-1)];
    }


    return; //TODO
}

__global__ void process_image_kernel(uchar *in, uchar *out, int temp_histogram[]) {

    int histogram[256] = { 0 };
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	atomicAdd(&temp_histogram[ in[tid + bid*NUM_THREADS]] , 1); // TODO: remove later
	atomicAdd(&histogram[in[tid + bid*NUM_THREADS]], 1);

	__syncthreads();

    int cdf[256] = { 0 };

//	cudaMemset(cdf+1, 0, 254);
	cdf[0] = histogram[0];
	cdf[255] = IMG_HEIGHT*IMG_WIDTH;

	__syncthreads();

//	prefix_sum <<< 254/2, 2 >>> (cdf, 256, histogram);

	if (tid < 2)
		prefix_sum(cdf, 256, histogram);

	__syncthreads();

    //int cdf_min = 0;
    //for (int i = 0; i < 256; i++) {
    //    if (cdf[i] != 0) {
    //        cdf_min = cdf[i];
    //        break;
    //    }
    //}

    //uchar map[256] = { 0 };
    //for (int i = 0; i < 256; i++) {
    //    int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
    //    map[i] = (uchar)map_value;
    //}

    //for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    //    img_out[i] = map[img_in[i]];
    //}

    return ; //TODO
}

int main() {
///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    /* instead of loading real images, we'll load the arrays with random data */
    srand(0);
    for (long long int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        images_in[i] = rand() % 256;
    }

    double t_start, t_finish;

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
//        process_image(img_in, img_out);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;
///////////////////////////////////////////////////////////////////////////////////////////////////////////
	uchar *image_in;
	uchar *image_out;

	int *temp_histogram;
	int cpu_histogram[256] = { 0 };
	int total_sum = 0;
    CUDA_CHECK( cudaMalloc((void **)&temp_histogram, 256 * sizeof(*temp_histogram)) );

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change

    //TODO: allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc((void **)&image_in, IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void **)&image_out, IMG_HEIGHT * IMG_WIDTH) );

    t_start = get_time_msec(); //Do not change

    //TODO: in a for loop:
    for (int i=0; i < N_IMAGES; i++) {
		// Copying src image from the input images
		cudaMemcpy(image_in, &images_in[i * IMG_WIDTH*IMG_HEIGHT], IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault);

		//printf("\n Image pixels are:\n");
		//for (int i = 0; i < 256; i++) {
		//	for(int j = 0; j < 256; j++)
		//		printf("%d ", image_in[i*256 + j]);
		//	printf("\n");
		//}
		
		cudaMemset(temp_histogram, 0, 256 * sizeof(*temp_histogram));
		process_image_kernel <<< NUM_BLKS, NUM_THREADS >>> (image_in, image_out, temp_histogram);   

		cudaMemcpy(cpu_histogram, temp_histogram, 256 * sizeof(*temp_histogram), cudaMemcpyDefault);
		printf("\n\nsize of int: %lu", sizeof(int));

		printf ("\n\nHistogram array is as followed:\n");
		for (int i=0; i< 4; i++) {
			for (int j = 0; j < 64; j++) {
				printf("h[%d] = %d  ",i*64 + j , cpu_histogram[i*64 + j]);
//				total_sum += cpu_histogram[i*64 + j];
				
			}

		printf("\n\n");
		}

		for(int i = 0; i < 256; i++)
			total_sum += cpu_histogram[i];

	printf("Total sum is: %d\n", total_sum);
	printf("\n\n");
	printf("\n\n");

    }
    //   1. copy the relevant image from images_in to the GPU memory you allocated CHECKED
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not change

    // GPU bulk
    printf("\n=== GPU Bulk ===\n"); //Do not change
    //TODO: allocate GPU memory for a all input images and all output images
    t_start = get_time_msec(); //Do not change
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out_gpu_bulk
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not chhange

    return 0;
}
