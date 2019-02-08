#include <stdlib.h>
#include <libconfig.h>

#ifndef PROPERTIES_H

#define PROPERTIES_H

void getIntegerProperty(config_t * cfg, const char * propName, int * propValue);

void getDoubleProperty(config_t * cfg, const char * propName, double * propValue);

#endif