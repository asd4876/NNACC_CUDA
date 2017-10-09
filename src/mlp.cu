#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include "mlp.h"

using namespace std;

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

mlp::mlp() {
	weight_list = NULL;
	dev_weight = NULL;
	dev_input = NULL;
	dev_weight_list = NULL;
	dev_nodes_list = NULL;
	dev_pitch = 0;
	layers_num = 0;
	max_nodes_num = 0;
	kernel_time = 0;
	load_time = 0;
	cout << "class mlp is created" << endl;
}

void mlp::loadModFromFile(const char* file_path) {
	ifstream fin(file_path);
	if (fin) {
		fin >> layers_num;

		int *nodes_num = new int[layers_num];

		for (int i = 0; i < layers_num; i++) {
			fin >> nodes_num[i];
			if (nodes_num[i] > max_nodes_num)
				max_nodes_num = nodes_num[i];
		}

		weight_list = new Matrix[layers_num - 1];

		for (int i = 0; i < layers_num - 1; i++) {

			weight_list[i].width = nodes_num[i + 1];
			weight_list[i].height = nodes_num[i] + 1;
			weight_list[i].pitch = 0;

			int num = weight_list[i].width * weight_list[i].height;
			weight_list[i].elements = new double[num];

			for (int j = 0; j < num; j++) {
				fin >> weight_list[i].elements[j];
			}

		}
		//cout << "layers_num : " << layers_num << endl;

		//added for small net;
		if (max_nodes_num < BLOCK_SIZE) {
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_nodes_list,
							sizeof(int) * layers_num));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_nodes_list, nodes_num,
							sizeof(int) * layers_num, cudaMemcpyHostToDevice));
			double *tmp_weight = new double[(layers_num - 1) * max_nodes_num
					* (max_nodes_num + 1)];
			int height = 0;
			for (int i = 0; i < layers_num - 1; i++) {
				int tmp = 0;
				for (int j = 0; j < weight_list[i].height; j++) {
					for (int k = 0; k < weight_list[i].width; k++) {
						tmp_weight[max_nodes_num * height + k] =
								weight_list[i].elements[tmp];
						tmp++;
					}
					height++;
				}
			}
			//cout << "height:" << height << endl;
			CUDA_CHECK_RETURN(
					cudaMallocPitch((void ** ) &dev_weight_list, &dev_pitch,
							sizeof(double) * max_nodes_num, height));

			CUDA_CHECK_RETURN(
					cudaMemcpy2D(dev_weight_list, dev_pitch, tmp_weight,
							sizeof(double) * max_nodes_num,
							sizeof(double) * max_nodes_num, height,
							cudaMemcpyHostToDevice));
		}
		delete[] nodes_num;

	} else {
		cout << "can't open mod file" << endl;

	}

}

void mlp::load_mod(const char* file_path) {

	loadModFromFile(file_path);

	dev_weight = new Matrix[layers_num - 1];

	for (int i = 0; i < layers_num - 1; i++) {

		dev_weight[i].width = weight_list[i].width;

		dev_weight[i].height = weight_list[i].height;

		CUDA_CHECK_RETURN(
				cudaMallocPitch((void ** ) &dev_weight[i].elements,
						&dev_weight[i].pitch,
						sizeof(double) * dev_weight[i].width,
						dev_weight[i].height));

		CUDA_CHECK_RETURN(
				cudaMemcpy2D(dev_weight[i].elements, dev_weight[i].pitch,
						weight_list[i].elements,
						sizeof(double) * dev_weight[i].width,
						sizeof(double) * dev_weight[i].width,
						dev_weight[i].height, cudaMemcpyHostToDevice));
	}

	CUDA_CHECK_RETURN(
			cudaMalloc((void ** )&dev_input, sizeof(double) * max_nodes_num));

}

__device__ double sigmoid(double x) {
	return 1.0 / (1 + exp(0 - x));
}

__global__ void MatMulSmallKernel(double *weight, size_t pitch, int *nodes, int layers, double *input){

	int tid = threadIdx.x;
	__shared__ double share_input[256];
	if(tid < nodes[0]){
		share_input[tid] = input[tid];
	}
	__syncthreads();
	int height = 0;
	for (int i = 0; i < layers - 1; i++) {
		int input_size = nodes[i];
		int output_size = nodes[i + 1];
		if (tid < output_size) {
			double sum = 0;
			for (int j = 0; j < input_size + 1; j++) {
				double* row = (double*) ((char*) weight + (height+j) * pitch);
				if (j != input_size) {
					sum += share_input[j] * row[tid];
				} else {
					sum += row[tid];
				}
			}
			share_input[tid] =  sigmoid(sum);
		}
		height += input_size+1;
		__syncthreads();

	}
	if(tid< nodes[layers-1]){
		input[tid] = share_input[tid];
	}

}

__global__ void MatMulKernel(Matrix weight, double *input) {
	int input_size = weight.height;
	int output_size = weight.width;

	int tid = threadIdx.x;
	// load input into shared memory.
	__shared__ double share_input[1024];
	for (int i = tid; i < input_size - 1; i += blockDim.x) {
		share_input[i] = input[i];
	}
	__syncthreads();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < output_size) {
		double sum = 0;
		for (int i = 0; i < input_size; i++) {
			double* row = (double*) ((char*) weight.elements + i * weight.pitch);
			if (i != input_size - 1) {
				sum += share_input[i] * row[idx];
			} else {
				sum += row[idx];
			}
		}
		input[idx] = sigmoid(sum);
	}
}


void mlp::run_gpu(const double* input, double *output) {
	int output_length = dev_weight[layers_num - 2].width;
	int input_length = dev_weight[0].height - 1;

	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_input, input, sizeof(double) * input_length,
					cudaMemcpyHostToDevice));
	if (max_nodes_num < BLOCK_SIZE) {

		for (int i = 0; i < layers_num - 1; i++) {

			const int blockCount = (dev_weight[i].width + BLOCK_SIZE - 1)
					/ BLOCK_SIZE;
			MatMulKernel<<<blockCount, BLOCK_SIZE>>>(dev_weight[i], dev_input);

		}

	} else {
		MatMulSmallKernel<<<1, BLOCK_SIZE>>>(dev_weight_list, dev_pitch,
				dev_nodes_list, layers_num, dev_input);
	}

	CUDA_CHECK_RETURN(
			cudaMemcpy(output, dev_input, sizeof(double) * output_length,
					cudaMemcpyDeviceToHost));

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&kernel_time, start1, stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
}

float mlp::get_kernel_time() {
	return kernel_time;
}

void mlp::kernel_free() {
	CUDA_CHECK_RETURN(cudaFree(dev_input));
	for (int i = 0; i < layers_num - 1; i++) {
		delete[] weight_list[i].elements;
		CUDA_CHECK_RETURN(cudaFree(dev_weight[i].elements));
	}
	delete[] weight_list;
	delete[] dev_weight;

}
mlp::~mlp() {
	cout << "class mlp is deleted" << endl;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}
