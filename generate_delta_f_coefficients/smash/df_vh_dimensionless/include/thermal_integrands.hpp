
#include <stdlib.h>

#ifndef THERMAL_INTEGRANDS_H

#define THERMAL_INTEGRANDS_H

// for 14 moment approximation
double J20_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double J21_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double J40_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double J41_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double J42_int(double pbar, double mbar, double T, double muB, double b, double Theta);

double N10_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double N30_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double N31_int(double pbar, double mbar, double T, double muB, double b, double Theta);

double M20_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double M21_int(double pbar, double mbar, double T, double muB, double b, double Theta);





// for chapman-enskog expansion
double nB_int(double pbar, double mbar, double T, double muB, double b, double Theta); // b*I10
double e_int(double pbar, double mbar, double T, double muB, double b, double Theta);  // I20
double p_int(double pbar, double mbar, double T, double muB, double b, double Theta);  // I21

double J30_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double J32_int(double pbar, double mbar, double T, double muB, double b, double Theta);

double N20_int(double pbar, double mbar, double T, double muB, double b, double Theta);

double M10_int(double pbar, double mbar, double T, double muB, double b, double Theta);
double M11_int(double pbar, double mbar, double T, double muB, double b, double Theta);


#endif