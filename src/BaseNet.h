#ifndef BASENET_H
#define BASENET_H

class BaseNet{
public:
	BaseNet(){

	}
	virtual ~BaseNet(){

	}
	virtual void load_mod(const char* file_path) = 0;
	virtual void run_gpu(const double* input, double* output)=0;
	virtual void run_gpu(const char* file_path, float* output)=0;
	virtual void kernel_free() = 0;
	virtual float get_kernel_time() = 0;
};


#endif
