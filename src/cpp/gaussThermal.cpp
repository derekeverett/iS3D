
#include <stdlib.h>
#include <cmath>
#include "gaussThermal.h"

// gauss integration routine
double GaussThermal(double thermal_integrand(double pbar, double mbar, double alphaB, double baryon, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double alphaB, double baryon, double sign)
{
	double integral = 0.0;
	for(int k = 0; k < pbar_pts; k++)
	{
		integral += pbar_weight[k] * thermal_integrand(pbar_root[k], mbar, alphaB, baryon, sign);
	}
	return integral;
}


// equilibrium particle density
double neq_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);

	// gauss laguerre (a = 1)
	return pbar * exp(pbar) / (exp(Ebar - baryon*alphaB) + sign);
}

// // equilibrium energy density
// double E_int(double pbar, double mbar, double alphaB, double baryon, double sign)
// {
// 	double Ebar = sqrt(pbar * pbar + mbar * mbar);
// 	// gauss laguerre (a = 2)
// 	return Ebar * exp(pbar) / (exp(Ebar - baryon * alphaB) + sign);
// }

// // equilibrium pressure
// double P_int(double pbar, double mbar, double alphaB, double baryon, double sign)
// {
// 	double Ebar = sqrt(pbar * pbar + mbar * mbar);
// 	// gauss laguerre (a = 2)
// 	return pbar * pbar / Ebar * exp(pbar) / (exp(Ebar - baryon * alphaB) + sign);
// }


// for linearized particle density corrections
double J10_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar - baryon*alphaB) + sign;

	// gauss laguerre (a = 1)
	return pbar * exp(pbar + Ebar - baryon*alphaB) / (qstat*qstat);
}

double J11_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar * pbar + mbar * mbar);
	double qstat = exp(Ebar - baryon * alphaB) + sign;
	// gauss laguerre (a = 1)
	return pbar * pbar * pbar / (Ebar * Ebar) * exp(pbar + Ebar - baryon * alphaB) / (qstat * qstat);
}

double J20_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar - baryon*alphaB) + sign;

	// gauss laguerre (a = 2)
	return Ebar * exp(pbar + Ebar - baryon*alphaB) / (qstat*qstat);
}

double J30_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar * pbar + mbar * mbar);
	double qstat = exp(Ebar - baryon * alphaB) + sign;
	// gauss laguerre (a = 3)
	return Ebar * Ebar / pbar * exp(pbar + Ebar - baryon * alphaB) / (qstat * qstat);
}

double J31_int(double pbar, double mbar, double alphaB, double baryon, double sign)
{
	double Ebar = sqrt(pbar * pbar + mbar * mbar);
	double qstat = exp(Ebar - baryon * alphaB) + sign;
	// gauss laguerre (a = 3)
	return pbar * exp(pbar + Ebar - baryon * alphaB) / (qstat * qstat);
}





// for Jonah's coefficient calculation

double Gauss1D_mod(double modified_1D_integrand(double pbar, double mbar, double lambda, double sign), double * pbar_root, double * pbar_weight, int pbar_pts, double mbar, double lambda, double sign)
{
	double sum = 0.0;
	for(int k = 0; k < pbar_pts; k++) sum += pbar_weight[k] * modified_1D_integrand(pbar_root[k], mbar, lambda, sign);
	return sum;
}

double E_mod_int(double pbar, double mbar, double lambda, double sign)
{
	double scale2 = (1.0 + lambda) * (1.0 + lambda);

	double Ebar = sqrt(pbar*pbar + mbar*mbar);

	return sqrt(pbar*pbar*scale2 + mbar*mbar) * exp(pbar) / (exp(Ebar) + sign);
}

double P_mod_int(double pbar, double mbar, double lambda, double sign)
{
	double scale2 = (1.0 + lambda) * (1.0 + lambda);

	double Ebar = sqrt(pbar*pbar + mbar*mbar);

	return pbar*pbar*scale2 / sqrt(pbar*pbar*scale2 + mbar*mbar) * exp(pbar) / (exp(Ebar) + sign);
}














