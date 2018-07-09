
#ifndef GAUSSTHERMAL_H
#define GAUSSTHERMAL_H


double GaussThermal(double thermal_integrand(double pbar, double mbar, double alphaB, double baryon, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double alphaB, double baryon, double sign);

// equilibrium particle density
double neq_int(double pbar, double mbar, double alphaB, double baryon, double sign);
//double Eeq_int(double pbar, double mbar, double alphaB, double baryon, double sign);
//double Peq_int(double pbar, double mbar, double alphaB, double baryon, double sign);

// for linearized particle density
double J10_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J11_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J20_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J30_int(double pbar, double mbar, double alphaB, double baryon, double sign);
double J31_int(double pbar, double mbar, double alphaB, double baryon, double sign);

// the modified particle density could be done differently?


#endif
