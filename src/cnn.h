#ifndef CNN_H
#define CNN_H
#include "BaseNet.h"
#include "time.h"
#define LAYER_CONV      0
#define LAYER_FC            1
#define LAYER_POOL       2

#define POOL_MAX            0
#define POOL_AVERAGE    1

#define LINEAR_SIGMOID     0

#define INPUT_WIDTH  28
#define INPUT_HEIGHT 28

//inner layers data.
typedef struct {
	float *data;
	int feature_num;
	int height;
	int width;
} nn_data;

typedef struct{
	float *weights;
	float *bias;//output_dim;
	int kernel_h;
	int kernel_w;
	int *kernel_index;//length:<=input_dim*output_dim;
	int *kernel_offset;//length: output_dim + 1;
	int layer_type;
	int method_type;
	int stride;
}nn_weights;

class cnn:public BaseNet{

    public:
	static const int MAX_THREADS_PER_BLOCK = 512;
	static const int MIN_THREADS_UNIT             = 32;

	cnn();
	~cnn();
	void load_mod(const char* file_path);
	void load_input(const char* file_path);
	void kernel_free();
	void run(float *result);
    void run_gpu(const char* file_path, float* output){
    	load_input(file_path);
    	clock_t start =clock();
    	run(output);
    	kernel_time = (float) (((clock() - start) * 1000.0) / CLOCKS_PER_SEC);
    }
    void run_gpu(const double* input, double* output){};
	float get_kernel_time(){
		return kernel_time;
	}

    private:
	size_t get_shared_memory();
	int layer_num;
	int batch;
	int input_w;
	int input_h;
	size_t max_shared_memory_size;
    nn_weights* dev_weights;
    nn_data* dev_data;
    float kernel_time;
};

#endif
