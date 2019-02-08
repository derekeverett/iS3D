#include <stdlib.h>
#include <fstream>
#include <libconfig.h>

#include "../include/properties.hpp"

using namespace std;

void getIntegerProperty(config_t * cfg, const char * propName, int * propValue)
{
	  if(config_lookup_int(cfg, propName, propValue))
	  {
	  	printf("%s = %d\n", propName, *propValue);
	  	return;
	  }
	  else
	  {
	  	printf("Couldn't set value, please check configuration file: exiting..\n");
	  	exit(-1);
	  }
}

void getDoubleProperty(config_t * cfg, const char * propName, double * propValue)
{
	  if(config_lookup_float(cfg, propName, propValue))
	  {
	  	printf("%s = %f\n", propName, *propValue);
	  	return;
	  }
	  else
	  {
	  	printf("Couldn't set value, please check configuration file: exiting..\n");
	  	exit(-1);
	  }
}





















