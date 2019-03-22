
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "../include/thermal_integrands.hpp"
using namespace std;


// integrands for 14 moment approximation coefficients
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


double J20_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	if(qstat < 0.0)
	{
		// there were no problems for my (T,muB) range 
		cout << "f_eq is negative for T = " << T << ", muB = " << muB << endl;
		exit(-1);
	}
	// (a = 2)
	return Ebar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double J21_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	 // (a = 2)
	return pbar*pbar/Ebar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double J40_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	 // (a = 4)
	return Ebar*Ebar*Ebar/pbar/pbar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double J41_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	 // (a = 4)
	return Ebar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double J42_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	 // (a = 4)
	return pbar*pbar/Ebar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double N10_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	// (a = 1)
	return b * pbar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double N30_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	// (a = 3)
	return b * Ebar*Ebar/pbar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double N31_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	// (a = 3)
	return b * pbar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double M20_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;

	// (a = 2)
	return b*b * Ebar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


double M21_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T) + Theta;
	// (a = 2)
	return b*b * pbar*pbar/Ebar * exp(pbar+Ebar-b*muB/T) / (qstat*qstat);
}


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::













// integrands for chapman-enskog coefficients (optional)
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

// net baryon density
double nB_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	// gauss laguerre (a = 1)
	return b * pbar * exp(pbar) / (exp(Ebar-b*muB/T)+Theta);
}


// energy density
double e_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);

	// gauss laguerre (a = 2)
	return Ebar * exp(pbar) / (exp(Ebar-b*muB/T)+Theta);
}


// equilibrium pressure
double p_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);

	// gauss laguerre (a = 2)
	return pbar*pbar/Ebar * exp(pbar) / (exp(Ebar-b*muB/T)+Theta);
}


double J30_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T)+Theta;
	// gauss laguerre (a = 3)
	return Ebar*Ebar / pbar * exp(pbar+Ebar-b*muB/T)/(qstat*qstat);
}


double J32_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T)+Theta;
	// gauss laguerre (a = 3)
	return pbar*pbar*pbar / (Ebar*Ebar) * exp(pbar+Ebar-b*muB/T)/(qstat*qstat);
}


double N20_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T)+Theta;
	// gauss laguerre (a = 2)
	return b * Ebar * exp(pbar+Ebar-b*muB/T)/(qstat*qstat);
}


double M10_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T)+Theta;
	// gauss laguerre (a = 1)
	return b*b * pbar * exp(pbar+Ebar-b*muB/T)/(qstat*qstat);
}


double M11_int(double pbar, double mbar, double T, double muB, double b, double Theta)
{
	double Ebar = sqrt(pbar*pbar + mbar*mbar);
	double qstat = exp(Ebar-b*muB/T)+Theta;
	// gauss laguerre (a = 1)
	return b * b * pbar*pbar*pbar / (Ebar*Ebar) * exp(pbar+Ebar-b*muB/T)/(qstat*qstat);
}


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::







