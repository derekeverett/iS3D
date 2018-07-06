
#include <stdlib.h>
#include <math.h>
#include "gaussThermal.h"

// gauss integration routine 
double GaussThermal(double thermal_integrand(double pbar, double mbar, double alphaB, double baryon, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double alphaB, double baryon, double sign)
{
	double sum = 0.0;
	for(int k = 0; k < pbar_pts; k++) sum += pbar_weight[k] * thermal_integrand(pbar_root[k], mbar, alphaB, baryon, sign);
	return sum;
}

// for equilibrium particle density 
double neq_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);

	// gauss laguerre (a = 1)
	return pbar * exp(pbar) / (exp(Ebar - baryon*alphaB) + sign);
}

// for linearized particle density correction (w/o bulkPi factor)
double N10_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar - baryon*alphaB) + sign;

	// gauss laguerre (a = 1)
	return baryon * pbar * exp(pbar + Ebar - baryon*alphaB) / (qstat*qstat);
}
double J20_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar - baryon*alphaB) + sign;
	
	// gauss laguerre (a = 2)
	return Ebar * exp(pbar + Ebar - baryon*alphaB) / (qstat*qstat);
}
//double J21_int(double pbar, double mbar, double alphaB, double baryon, double sign);


// the modified particle density could be done differently?


