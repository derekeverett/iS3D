// Version 1.6.2
// Zhi Qiu

#ifndef arsenal_h
#define arsenal_h

#include "stdlib.h"
#include <vector>
#include <string>

using namespace std;

double sixPoint2dInterp(double x, double y,
    double v00, double v01, double v02, double v10, double v11, double v20);

double interpCubicDirect(vector<double>* x, vector<double>* y, double xx);
double interpCubicMono(vector<double>* x, vector<double>* y, double xx);
double interpLinearDirect(vector<double>* x, vector<double>* y, double xx);
double interpLinearMono(vector<double>* x, vector<double>* y, double xx);
double interpNearestDirect(vector<double>* x, vector<double>* y, double xx);
double interpNearestMono(vector<double>* x, vector<double>* y, double xx);

double invertFunc(double (*func)(double), double y, double xL, double xR, double dx, double x0, double relative_accuracy=1e-10);

double invertTableDirect_hook(double xx);
double invertTableDirect(vector<double>* x, vector<double>* y, double y0, double x0, double relative_accuracy=1e-10);

double stringToDouble(string);
vector<double> stringToDoubles(string);
string toLower(string str);
string trim(string str);

vector< vector<double>* >* readBlockData(istream &stream_in);
void releaseBlockData(vector< vector<double>* >* data);

double adaptiveSimpsonsAux(double (*f)(double), double a, double b, double epsilon, double S, double fa, double fb, double fc, int bottom);
double adaptiveSimpsons(double (*f)(double), double a, double b,  double epsilon=1e-15, int maxRecursionDepth=50);

double qiu_simpsons(double (*f)(double), double a, double b, double epsilon=1e-15, int maxRecursionDepth=50);

long binarySearch(vector<double>* A, double value, bool skip_out_of_range=false);

void formatedPrint(ostream&, int count, ...);
double gamma_function(double x);

void print_progressbar(double percentage, int length=50, string symbol="#");
void display_logo(int which=1);

//**********************************************************************
inline long irand(long LB, long RB)
// Get an interger between LB and RB (including) with uniform distribution.
{
  return LB + (long)((RB-LB+1)*(drand48()-1e-25));
}


//**********************************************************************
inline double drand(double LB, double RB)
// Get random number with uniform distribution between LB and RB with
// boundary-protection.
{
  double width = RB-LB;
  double dw = width*1e-30;
  return LB+dw+(width-2*dw)*drand48();
}

void GaussLegendre_getWeight(int npts,double* x,double* w, double A, double B, int opt=1);

void get_bin_average_and_count(istream& is, ostream& os, vector<double>* bins, long col_to_bin=0, void (*func)(vector<double>*)=NULL, long wanted_data_columns=-1, bool silence=false); // Note that col_to_bin starts with 1, and bins is assumed to be monotonically increasing



// anisotropic functions for vah_pl_matching surface reader
double aL_fit(double pl_peq_ratio);
double R200(double aL);

#endif





