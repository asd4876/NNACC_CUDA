#include<iostream>
#include "NetFactory.h"
#include "BaseNet.h"
using namespace std;

int main(){
	NetFactory& factory = NetFactory::getInstance();
	BaseNet *net = NULL;

	net = factory.getNet("src/123.mod");

	double input[6] = { 1, 2, 3, 4, 5, 6};
	double *result = new double[1];

	net->run_gpu(input, result);

	cout << result[0] << endl;
	cout << net->get_kernel_time() << endl;

/*

	net = NULL;
	net = factory.getNet("src/123.mod");
	net->run_gpu(input, result);

	cout << result[0] << endl;
	cout << net->get_kernel_time() << endl;
*/
	delete[] result;
	/*
	net = NULL;
	net = factory.getNet("src/test.cnet");
	float *result2 = new float[10];

	net ->run_gpu("src/test.cdat", result2);

	cout << result2[0] << endl;
	cout << net->get_kernel_time() << endl;

	delete[] result2;

	net = NULL;
	net = factory.getNet("src/test2.cnet");
	float *result3 = new float[10];

	net->run_gpu("src/test.cdat", result3);

	cout << result3[0] << endl;
	cout << net->get_kernel_time() << endl;

	delete[] result3;
*/
	factory.netFree();


//	BaseNet *net = new mlp();
//	net->load_mod("/home/devil/cuda-workspace/Test/src/123.mod");
//
//	double input[6] = { 1, 2, 3, 4, 5, 6 };
//	double *result = new double[1];
//	net->run_gpu(input, result);
//
//	cout << result[0] << endl;
//	cout << net->get_kernel_time() << endl;
//
//	net->kernel_free();
//	delete[] result;
//	delete net;
//
//	BaseNet *net2 = new cnn();
//	net2->load_mod("src/test.cnet");
//
//	float *result2 = new float[10];
//	net2->run_gpu("src/test.cdat", result2);
//
//	cout << result2[0] << endl;
//	cout << net2->get_kernel_time() << endl;
//
//	net2->kernel_free();
//	delete[] result2;
//	delete net2;

	return 0;
}
