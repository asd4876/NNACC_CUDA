#ifndef MLP_H
#define MLP_H
#include "BaseNet.h"

typedef struct {
	int width;
	int height;
	size_t pitch;
	double* elements;
} Matrix;

class mlp:public BaseNet{

    public:

	static const int BLOCK_SIZE = 256;

	static const int MAX_SHARED_MEMORY_SIZE = 1024;

    mlp();

    ~mlp();

    void load_mod(const char* file_path);

    void run_gpu(const char* file_path, float* output){};

    void run_gpu(const double* input, double* output);

    void kernel_free();

    float get_kernel_time();

    private:

   // read mod file.
    void loadModFromFile(const char* file_path);

   // weight_list in host memory.
    Matrix *weight_list;
   //  weight_list in gpu memory.
    Matrix *dev_weight;
   // input in gpu memory.
    double *dev_input;
   // layers number.
    int layers_num;
   // max nodes per layer.
    int max_nodes_num;
    // kernel run time.
    float kernel_time;
    // load time.
    float load_time;

    // added for small net: max nodes num < 256.
    // weight_list for small net.
    double *dev_weight_list;
    // nodes_num_list for small net.
    int *dev_nodes_list;
    size_t dev_pitch;

};

#endif
