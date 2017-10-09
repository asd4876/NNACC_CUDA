#ifndef NETFACTORY_H
#define NETFACTORY_H
#include <iostream>
#include <map>
#include <string>
#include "BaseNet.h"
using namespace std;

class NetFactory{
public:

	static NetFactory& getInstance();

	BaseNet* getNet(const char* filepath);

	void netFree();

private:

	map<string, BaseNet*> net_map;

	NetFactory();

	~NetFactory();

	NetFactory(NetFactory const&);

	void operator=(NetFactory const&);

};

#endif
