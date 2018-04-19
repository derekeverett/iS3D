
#ifndef GAUSSTHERMAL_H
#define GAUSSTHERMAL_H


double GaussThermal(double thermal_integrand(double pbar, double mbar, double alphaB, double baryon, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double alphaB, double baryon, double sign);

// for equilibrium particle density 
double neq_int(double pbar, double mbar, double alphaB, double baryon, double sign);

// for linearized particle density (w/o bulkPi factor)
double N10_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J20_int(double pbar, double mbar, double alphaB, double baryon, double sign);
//double J21_int(double pbar, double mbar, double alphaB, double baryon, double sign);


// the modified particle density could be done differently?


#endif
