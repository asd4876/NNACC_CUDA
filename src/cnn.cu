#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"

using namespace std;

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__device__ float  non_linear(int type, float input_num)
{
	float result = 0.0;
	if (type == 0)
	{
		result = 1.0 / (1.0 + exp(0.0 - input_num));
	}
	return result;
}

__global__ void fc(
		float* inputs,
		float* outputs,
		float* ws,
		float* bs,
		int input_w,
		int output_w,
		int lmethod)
{
	int batch_id = blockIdx.x;
	int output_id = blockIdx.y;
	float* cur_input = inputs + batch_id * input_w;
	float* cur_output = outputs + batch_id * output_w;
	int  idx = output_id * blockDim.x + threadIdx.x;
	float cur_bs = bs[idx];

	extern __shared__ float sm[];

	for (int i = threadIdx.x; i < input_w; i += blockDim.x) {
		sm[i] = cur_input[i];
	}

	 __syncthreads();

	 if (idx < output_w) {
	 		float sum = 0;
	 		for (int i = 0; i < input_w; i++) {
	 			sum += sm[i] * ws[i*output_w+idx];
	 		}
	 		cur_output[idx] = non_linear(lmethod,sum + cur_bs);
	 	}
}

__global__ void pool_average(
		float* inputs,
		float* outputs,
		int input_num,
        int input_h,
        int input_w,
        int batch,
        int kernel_h,
        int kernel_w,
        int trans_flag)
{
    int batch_id = blockIdx.x;
    int output_id = blockIdx.y;

    int output_h = input_h/kernel_h;
    int output_w = input_w/kernel_w;
    float* cur_input = inputs + batch_id * input_num * input_h * input_w + output_id * input_h * input_w ;
    float* cur_output = outputs + batch_id * input_num * output_h * output_w + output_id *  output_h * output_w;

	for (int i = 0; i < output_h * output_w; i += blockDim.x) {
		int ti = i + threadIdx.x;
		if (ti < output_h * output_w) {
			int tiy = ti / output_w;
			int tix = ti % output_w;
			float val = 0.0;
			for (int h = 0; h < kernel_h; h++) {
				int tmp_hid = (h + tiy * kernel_h) * input_w + tix * kernel_w;
				for (int w = 0; w < kernel_w; w++) {
					val += cur_input[tmp_hid + w];
				}
			}
			int trans_tid = (trans_flag == 1) ? (tix * output_h + tiy) : ti;
			cur_output[trans_tid] = val / (kernel_h * kernel_w);
		}
	}
}
__global__ void conv_shared(
        float*  inputs,
        float*  outputs,
        float*  ws,
        float*  bs,
        int*  k_index,
		int*  k_offset,
		int input_num,
        int input_h,
        int input_w,
        int output_num,
        int output_h,
        int output_w,
        int batch,
        int kernel_h,
        int kernel_w,
        int lmethod,
        int stride)
{
    int batch_id = blockIdx.x;
    int output_id = blockIdx.y;

    extern __shared__ float sm[];

    int input_length = input_h * input_w;

    float* sm_w = sm + input_length;
    float* cur_input = inputs + batch_id * input_num * input_length;
    float* cur_output = outputs + batch_id * output_num * output_h * output_w + output_id *  output_h * output_w;
    float* cur_ws = ws + k_offset[output_id] * kernel_h * kernel_w;
    float cur_bs = bs[output_id];
    int* cur_index = k_index + k_offset[output_id];

	//load weights to shared memory
    int ws_num = k_offset[output_id + 1] - k_offset[output_id];
	int ws_length = ws_num * kernel_h * kernel_w;
	for (int i = 0; i < ws_length; i += blockDim.x) {
		int ti = i + threadIdx.x;
		if (ti < ws_length) {
			sm_w[ti] = cur_ws[ti];
		}
	}
	//initial shared memory of input data
	for (int i = 0; i < input_length; i += blockDim.x) {
		int ti = i + threadIdx.x;
		if (ti < input_length) {
			sm[ti] = 0;
		}
	}
    __syncthreads();

    //convolution
	for (int i = 0; i < output_h * output_w; i += blockDim.x) {
		int ti = i + threadIdx.x;

		if (ti < output_h * output_w) {
			int tiy = ti/output_w;
			int tix = ti%output_w;
			float val = 0.0;

			for(int j=0;j<ws_num;j++){
				int input_id = cur_index[j];
				int tmp_wid = j*kernel_h*kernel_w;
				// load input data to shared memory
				for(int k=0;k<input_length;k += blockDim.x){
					int tk = k+ threadIdx.x;
					if(tk<input_length){
						sm[tk] = cur_input[input_id*input_length+tk];
					}
				}
				__syncthreads();
				for(int h=0;h<kernel_h;h++){
					int tmp_wid_h = tmp_wid + h*kernel_w;
					int tmp_pid_h = (tiy*stride+h)*input_w + tix*stride;
					for(int w=0;w<kernel_w;w++){
						val += sm[tmp_pid_h+w] * sm_w[tmp_wid_h+w];
					}
				}
			}

			cur_output[ti] = non_linear(lmethod,val + cur_bs);
		}

	}
}


cnn::cnn(){
	dev_data = NULL;
	dev_weights = NULL;
	layer_num = 0;
	batch = 0;
	input_h = INPUT_HEIGHT;
	input_w = INPUT_WIDTH;
	max_shared_memory_size = get_shared_memory();
	kernel_time = 0;
	cout<<"class cnn is created"<<endl;
}

size_t cnn::get_shared_memory() {
	size_t sm_size = 0;
	int dev_num = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&dev_num));
	if (dev_num > 0) {
		cudaSetDevice(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		sm_size = deviceProp.sharedMemPerBlock;
	}
	return sm_size;
}

cnn::~cnn(){
	cout<<"class cnn is deleted"<<endl;
}

void cnn::load_mod(const char* file_path){
	FILE *net_config;
	net_config = fopen(file_path, "r");
	fscanf(net_config, "%d", &layer_num);
	dev_data = new nn_data[layer_num+1];
	dev_weights = new nn_weights[layer_num];
	dev_data[0].height = input_h;
	dev_data[0].width = input_w;
	dev_data[0].feature_num = 1;
	for(int i=0;i<layer_num;i++){
		int l_type = 0;
		fscanf(net_config, "%d", &l_type);
		dev_weights[i].layer_type = l_type;
		if(l_type == LAYER_CONV){
			//cout<<"read conv:"<<endl;
			int flt_size, front_feature_size, flt_w, flt_h, lstride;
			fscanf(net_config, "%d", &flt_size);
			fscanf(net_config, "%d", &front_feature_size);
			fscanf(net_config, "%d", &flt_w);
			fscanf(net_config, "%d", &flt_h);
			fscanf(net_config, "%d", &lstride);
			dev_weights[i].kernel_w = flt_w;
			dev_weights[i].kernel_h = flt_h;
			dev_weights[i].stride = lstride;
			dev_data[i+1].feature_num = flt_size;
			dev_data[i+1].width = (dev_data[i].width - flt_w + 1)/lstride;
			dev_data[i+1].height = (dev_data[i].height - flt_h + 1)/lstride;
			dev_weights[i].method_type = LINEAR_SIGMOID;
			float* tmp_w = (float*)malloc(sizeof(float) * flt_size*front_feature_size*flt_w*flt_h);
			float* tmp_b = (float*)malloc(sizeof(float) * flt_size);
			int* tmp_index = (int*)malloc(sizeof(int) * flt_size*front_feature_size);
			int* tmp_offset = (int*)malloc(sizeof(int) * (flt_size+1));
			int sum_index=0;
			for(int j=0;j<flt_size;j++){
				int flt_num = 0;
				fscanf(net_config, "%d", &flt_num);
				sum_index += flt_num;
				if(j==0) tmp_offset[j] = 0;
				tmp_offset[j+1] = tmp_offset[j] + flt_num;
				for (int k = tmp_offset[j]; k < tmp_offset[j]+flt_num; k++){
					fscanf(net_config, "%d", &tmp_index[k] );
					for(int m = k*flt_w*flt_h; m< (k+1)*flt_w*flt_h;m++){
						fscanf(net_config, "%f", &tmp_w[m] );
					}
				}
				fscanf(net_config, "%f", &tmp_b[j]);
			}
			int total_rows = tmp_offset[flt_size];
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_weights[i].weights,
							sizeof(float) * total_rows * flt_w * flt_h));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_weights[i].weights, tmp_w,
							sizeof(float) * total_rows * flt_w * flt_h,
							cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_weights[i].bias,
							sizeof(float) * flt_size));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_weights[i].bias, tmp_b,
							sizeof(float) * flt_size, cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_weights[i].kernel_index,
							sizeof(int) * total_rows));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_weights[i].kernel_index, tmp_index,
							sizeof(int) * total_rows, cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_weights[i].kernel_offset,
							sizeof(int) * (flt_size + 1)));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_weights[i].kernel_offset, tmp_offset,
							sizeof(int) * (flt_size + 1),
							cudaMemcpyHostToDevice));
			free(tmp_w);
			free(tmp_b);
			free(tmp_index);
			free(tmp_offset);
		}
		else if(l_type == LAYER_FC){
			//cout<<"read fc:"<<endl;
			int input_num, output_num;
			fscanf(net_config, "%d", &input_num);
			fscanf(net_config, "%d", &output_num);
			dev_data[i+1].feature_num = 1;
			dev_data[i+1].width = output_num;
			dev_data[i+1].height = 1;
			dev_weights[i].kernel_h = input_num;
			dev_weights[i].kernel_w = output_num;
			dev_weights[i].method_type = LINEAR_SIGMOID;
			dev_weights[i].stride = 1;
			dev_weights[i].kernel_index = NULL;
			dev_weights[i].kernel_offset = NULL;
			float* tmp_w = (float*)malloc(sizeof(float) * input_num*output_num);
			float* tmp_b = (float*)malloc(sizeof(float) * output_num);
			for(int j=0;j<input_num*output_num;j++){
				fscanf(net_config, "%f", &tmp_w[j]);
			}
			for(int j=0;j<output_num;j++){
				fscanf(net_config, "%f", &tmp_b[j]);
			}
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_weights[i].weights,
							sizeof(float) * input_num * output_num));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_weights[i].weights, tmp_w,
							sizeof(float) * input_num * output_num,
							cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(
					cudaMalloc((void ** )&dev_weights[i].bias,
							sizeof(float) * output_num));
			CUDA_CHECK_RETURN(
					cudaMemcpy(dev_weights[i].bias, tmp_b,
							sizeof(float) * output_num,
							cudaMemcpyHostToDevice));
			free(tmp_w);
			free(tmp_b);
		}
		else if(l_type == LAYER_POOL){
			//cout<<"read pool:"<<endl;
			int lmethod = POOL_MAX;
			int front_feature_size, width, height;
			fscanf(net_config, "%d", &lmethod);
			fscanf(net_config, "%d", &front_feature_size);
			fscanf(net_config, "%d", &width);
			fscanf(net_config, "%d", &height);
			dev_data[i+1].feature_num = dev_data[i].feature_num;
			dev_data[i+1].width = dev_data[i].width/width;
			dev_data[i+1].height = dev_data[i].height/height;
			dev_weights[i].kernel_h = width;
			dev_weights[i].kernel_w = height;
			dev_weights[i].method_type = lmethod;
			dev_weights[i].stride = 0;
			dev_weights[i].kernel_index = NULL;
			dev_weights[i].kernel_offset = NULL;
			dev_weights[i].weights = NULL;
			dev_weights[i].bias = NULL;
		}
		else{
			printf("Error: unknown layer type.");
		}

	}
	fclose(net_config);


}

void cnn::load_input(const char* file_path){
	FILE *input_data;
	input_data = fopen(file_path, "r");
	int input_width, input_height;
	fscanf(input_data, "%d", &batch);
	fscanf(input_data, "%d", &input_width);
	fscanf(input_data, "%d", &input_height);
	float *tmp = (float*)malloc(sizeof(float)*batch*input_width*input_height);
	for(int i=0;i<batch*input_width*input_height;i++){
		fscanf(input_data,"%f",&tmp[i]);
	}
	for(int i=0;i<layer_num+1;i++){
		CUDA_CHECK_RETURN(
				cudaMalloc((void ** )&dev_data[i].data,
						sizeof(float) * batch * dev_data[i].feature_num
								* dev_data[i].height * dev_data[i].width));
	}
	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_data[0].data, tmp,
					sizeof(float) * batch * dev_data[0].feature_num
							* dev_data[0].height * dev_data[0].width,
					cudaMemcpyHostToDevice));
	free(tmp);
	fclose(input_data);
}

void cnn::kernel_free(){
	for(int i=0;i<layer_num;i++){
		int t = dev_weights[i].layer_type;
		if(t == LAYER_CONV){
			CUDA_CHECK_RETURN(cudaFree(dev_weights[i].bias));
			CUDA_CHECK_RETURN(cudaFree(dev_weights[i].weights));
			CUDA_CHECK_RETURN(cudaFree(dev_weights[i].kernel_offset));
			CUDA_CHECK_RETURN(cudaFree(dev_weights[i].kernel_index));
		}
		else if(t == LAYER_FC){
			CUDA_CHECK_RETURN(cudaFree(dev_weights[i].bias));
			CUDA_CHECK_RETURN(cudaFree(dev_weights[i].weights));
		}
	}
	for(int i=0;i<layer_num+1;i++){
		CUDA_CHECK_RETURN(cudaFree(dev_data[i].data));
	}
	delete[] dev_weights;
	delete[] dev_data;
}

void cnn::run(float *result){
	//
//	float kernel_time;
//	 cudaEvent_t start1;
//	 cudaEventCreate(&start1);
//	 cudaEvent_t stop1;
//	 cudaEventCreate(&stop1);
//	 cudaEventRecord(start1, NULL);
	for(int i=0;i<layer_num;i++){
		int l_type = dev_weights[i].layer_type;
		if(l_type == LAYER_CONV){
			//cout<<"execution conv"<<endl;
			int sm_size = sizeof(float) * (dev_data[i].height * dev_data[i].width
							+ dev_data[i].feature_num * dev_weights[i].kernel_h * dev_weights[i].kernel_w);
			//int trans_flag = (i < layer_num - 1 && dev_weights[i + 1].layer_type == LAYER_FC) ? 1 : 0;
			if (sm_size < max_shared_memory_size) {
				dim3 block = dim3(batch, dev_data[i + 1].feature_num);
				int thread_num = ((dev_data[i + 1].height * dev_data[i + 1].width + MIN_THREADS_UNIT - 1) / MIN_THREADS_UNIT)
						* MIN_THREADS_UNIT;
				dim3 thread = dim3(thread_num < MAX_THREADS_PER_BLOCK ? thread_num : MAX_THREADS_PER_BLOCK);
				conv_shared<<<block, thread, sm_size>>>(
						dev_data[i].data,
						dev_data[i + 1].data,
						dev_weights[i].weights,
						dev_weights[i].bias,
						dev_weights[i].kernel_index,
						dev_weights[i].kernel_offset,
						dev_data[i].feature_num,
						dev_data[i].height,
						dev_data[i].width,
						dev_data[i + 1].feature_num,
						dev_data[i + 1].height,
						dev_data[i + 1].width,
						batch,
						dev_weights[i].kernel_h,
						dev_weights[i].kernel_w,
						dev_weights[i].method_type,
						dev_weights[i].stride);
			}
			else {
				cout<<"Error: don't support too large input data"<<endl;
				return;
			}
		}
		else if(l_type == LAYER_POOL){
			//cout<<"execution pool"<<endl;
			int trans_flag = (i < layer_num - 1 && dev_weights[i + 1].layer_type == LAYER_FC) ? 1 : 0;
			dim3 block = dim3(batch, dev_data[i + 1].feature_num);
			int thread_num = ((dev_data[i + 1].height * dev_data[i + 1].width + MIN_THREADS_UNIT - 1) / MIN_THREADS_UNIT)
					* MIN_THREADS_UNIT;
			dim3 thread = dim3(thread_num < MAX_THREADS_PER_BLOCK ? thread_num : MAX_THREADS_PER_BLOCK);

			if(dev_weights[i].method_type == POOL_AVERAGE){
				pool_average<<<block,thread>>>(
						dev_data[i].data,
						dev_data[i + 1].data,
						dev_data[i].feature_num,
						dev_data[i].height,
						dev_data[i].width,
						batch,
						dev_weights[i].kernel_h,
						dev_weights[i].kernel_w,
						trans_flag);
			}

			if(trans_flag == 1){
				dev_data[i+1].width = dev_data[i + 1].feature_num * dev_data[i + 1].height * dev_data[i + 1].width;
				dev_data[i+1].feature_num = 1;
				dev_data[i+1].height = 1;
				//cout<<"after trans:"<< dev_data[i+1].width<<endl;
			}

		}
		else if(l_type == LAYER_FC){
			//cout << "execution fc" << endl;
			int sm_size = sizeof(float) * dev_data[i].width;
			if (sm_size < max_shared_memory_size) {
				int thread_num = ((dev_data[i + 1].width + MIN_THREADS_UNIT - 1)
						/ MIN_THREADS_UNIT) * MIN_THREADS_UNIT;
				dim3 thread = dim3(
						thread_num < MAX_THREADS_PER_BLOCK ?
								thread_num : MAX_THREADS_PER_BLOCK);
				dim3 block = dim3(batch,
						thread_num < MAX_THREADS_PER_BLOCK ?
								1 : (thread_num + MAX_THREADS_PER_BLOCK - 1)
										/ MAX_THREADS_PER_BLOCK);

				fc<<<block, thread, sm_size>>>(
						dev_data[i].data,
						dev_data[i + 1].data,
						dev_weights[i].weights,
						dev_weights[i].bias,
						dev_data[i].width,
						dev_data[i + 1].width,
						dev_weights[i].method_type);
			}
		}
		else{
				printf("Error: unknown layer type.");
		}
		//used for debug:
//		float *tmp_out = new float[batch * dev_data[i + 1].feature_num
//				* dev_data[i + 1].height * dev_data[i + 1].width];
//		CUDA_CHECK_RETURN(
//				cudaMemcpy(tmp_out, dev_data[i + 1].data,
//						sizeof(float) * batch * dev_data[i + 1].feature_num
//								* dev_data[i + 1].height
//								* dev_data[i + 1].width,
//						cudaMemcpyDeviceToHost));
//		int zz = 0;
//		for (int t = 0; t < dev_data[i + 1].feature_num * batch; t++) {
//			cout << "feature_map" << t << ":" << endl;
//			for (int x = 0; x < dev_data[i + 1].height; x++) {
//				for (int y = 0; y < dev_data[i + 1].width; y++) {
//					cout << tmp_out[zz] << " ";
//					zz++;
//				}
//				cout << endl;
//			}
//		}
//		free(tmp_out);
		//debug end;
	}
//	 cudaEventRecord(stop1, NULL);
//	 cudaEventSynchronize(stop1);
//	 cudaEventElapsedTime(&kernel_time, start1, stop1);
//	 cudaEventDestroy(start1);
//	 cudaEventDestroy(stop1);
//	 cout<<kernel_time<<endl;
	CUDA_CHECK_RETURN(
			cudaMemcpy(result, dev_data[layer_num].data,
					sizeof(float) * batch * dev_data[layer_num].feature_num
							* dev_data[layer_num].height * dev_data[layer_num].width,
					cudaMemcpyDeviceToHost));
	//cout<<batch<<endl;


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
