
#include <stdlib.h>
#include "../include/gauss_integration.hpp"

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                        1D GAUSS INTEGRATION                     ::
//                                                                 ::
//     Compute 1D thermal integrals over radial momentum bar       ::
//     using Gauss Laguerre quadrature.                            ::
//                                                                 ::
//                			    Gauss1D                            ::
//                                                                 ::
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



// for thermodynamic integrals (1D integral over pbar coordinate from 0 to infinity)
double Gauss1D(double thermal_1D_integrand(double pbar, double mbar, double T, double muB, double b, double Theta), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double T, double muB, double b, double Theta)
{
	double sum = 0.0;
	for(int k = 0; k < pbar_pts; k++) sum += pbar_weight[k] * thermal_1D_integrand(pbar_root[k], mbar, T, muB, b, Theta);
	return sum;
}


