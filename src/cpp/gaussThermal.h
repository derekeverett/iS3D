
#ifndef GAUSSTHERMAL_H
#define GAUSSTHERMAL_H


double GaussThermal(double thermal_integrand(double pbar, double mbar, double alphaB, double baryon, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double alphaB, double baryon, double sign);

// equilibrium particle density
double neq_int(double pbar, double mbar, double alphaB, double baryon, double sign);

// for linearized particle density
double J10_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J11_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J20_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J30_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J31_int(double pbar, double mbar, double alphaB, double baryon, double sign);


// for jonah coefficient calculation
double Gauss1D_mod(double modified_1D_integrand(double pbar, double mbar, double lambda, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double lambda, double sign);

double E_mod_int(double pbar, double mbar, double lambda, double sign);
double P_mod_int(double pbar, double mbar, double lambda, double sign);


#endif
