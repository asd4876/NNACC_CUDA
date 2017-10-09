#include <iostream>
#include "NetFactory.h"
#include "cnn.h"
#include "mlp.h"
using namespace std;

NetFactory& NetFactory::getInstance(){
	static NetFactory instance;
	return instance;
}

BaseNet* NetFactory::getNet(const char* filepath){
	string key(filepath);
	BaseNet* net = NULL;
	if(net_map.find(key) != net_map.end()){
		net = net_map[key];
	}
	else{
		if(key.find(".cnet") != string::npos){
			net = new cnn();
			net->load_mod(filepath);
			net_map[key] = net;
		}
		else if(key.find(".mod") != string::npos){
			net = new mlp();
			net->load_mod(filepath);
			net_map[key] = net;
		}
		else{
			cout<<"Error: unknown mod type"<<endl;
		}
	}
	return net;
}

void NetFactory::netFree(){
	for (map<string, BaseNet*>::iterator it = net_map.begin(); it != net_map.end(); it++) {
		BaseNet* net = it->second;
		net->kernel_free();
		delete net;
	}
	net_map.clear();
}

NetFactory::NetFactory(){
	cout<<"net factory is created"<<endl;
}

NetFactory::~NetFactory(){
	cout<<"net factory is deleted"<<endl;
}


