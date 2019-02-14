#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <string.h>
#include <iostream>
#include <iomanip>
using namespace std;
#include <sstream>
#include <fstream>
#include <libconfig.h>
#include "../include/gauss_integration.hpp"
#include "../include/thermal_integrands.hpp"
#include "../include/freearray.hpp"
#include "../include/readindata.hpp"

const double hbarC = 0.197327053;  // GeV*fm
const double two_pi2_hbarC3 = 2.0 * pow(M_PI, 2) * pow(hbarC, 3);

int main()
{
  	printf("\nLoading particle info from SMASH pdg.dat\n");

  	particle_info * particle = new particle_info[Maxparticle];		// particle info struct



  	// read resonances in pdg smash file
  	//**************************************************/|
	int N_resonances = read_resonances_list(particle);	//
	//**************************************************/|



  	// table parameters:
  	//::::::::::::::::::::::::::::::::::::::

	double T_min = 0.1;			// min and max baryon chemical potential (GeV)
	double T_max = 0.2;

	double muB_min = 0.0;		// min and max baryon chemical potential (GeV)
	double muB_max = 0.8;

	int Tpts = 101;				// number of grid points
	int muBpts = 81;

	int gla_pts = 64;			// gauss laguerre points

	//::::::::::::::::::::::::::::::::::::::



	// load temperature and chemical potential arrays
	double T_array[Tpts];
	double muB_array[muBpts];

	T_array[0] = T_min;
	muB_array[0] = muB_min;

	double dT = (T_max - T_min)/(double)(Tpts - 1);
	double dmuB = (muB_max - muB_min)/(double)(muBpts - 1);

	for(int i = 1; i < Tpts; i++)
	{
		T_array[i] = T_min + (double)i*dT;
	}
	for(int i = 1; i < muBpts; i++)
	{
		muB_array[i] = muB_min + (double)i*dmuB;
	}


	//printf("Loading gauss laguerre roots and weights..\n");

  	char laguerre_file[255] = "";
  	sprintf(laguerre_file, "gauss_laguerre/gla_roots_weights_%d_points.txt", gla_pts);

  	Gauss_Laguerre laguerre;

  	laguerre.load_roots_and_weights(laguerre_file);

	// gauss laguerre (alpha specific) roots and weights
	double * pbar_root1 = laguerre.root[1];
	double * pbar_root2 = laguerre.root[2];
	double * pbar_root3 = laguerre.root[3];
	double * pbar_root4 = laguerre.root[4];

	double * pbar_weight1 = laguerre.weight[1];
	double * pbar_weight2 = laguerre.weight[2];
	double * pbar_weight3 = laguerre.weight[3];
	double * pbar_weight4 = laguerre.weight[4];

	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//																					        ::
	// 					   Compute coefficients of 14 moment approximation 		     	        ::
	//																					        ::
	// df ~ (cT.m^2 + b.c1.(u.p) + cE.(u.p)^2).Pi + (b.c3 + c4(u.p))pmu.Vmu + c5.pmu.pnu.pimunu ::
	//																					        ::
	//      			cT = c0 - c2		cE = 4c2 - c0 		c5 = 1 / (2(E+P)T^2)			::
	//																						    ::
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	size_t SF = 8;	// significant digits


	printf("Computing 14 moment df coefficients...\n");

	ofstream c0_table_df14;
	ofstream c1_table_df14;
	ofstream c2_table_df14;
	ofstream c3_table_df14;
	ofstream c4_table_df14;

	c0_table_df14.open("smash/c0.dat", ios::out);
	c1_table_df14.open("smash/c1.dat", ios::out);
	c2_table_df14.open("smash/c2.dat", ios::out);
	c3_table_df14.open("smash/c3.dat", ios::out);
	c4_table_df14.open("smash/c4.dat", ios::out);

	// put number of T, muB points before header
	c0_table_df14 << Tpts << "\n" << muBpts << endl;
	c1_table_df14 << Tpts << "\n" << muBpts << endl;
	c2_table_df14 << Tpts << "\n" << muBpts << endl;
	c3_table_df14 << Tpts << "\n" << muBpts << endl;
	c4_table_df14 << Tpts << "\n" << muBpts << endl;

	// header labels
	c0_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c0_T4 [fm^3/GeV^3 * GeV^4]" << endl;
	c1_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c1_T3 [fm^3/GeV^2 * GeV^3]" << endl;
	c2_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c2_T4 [fm^3/GeV^3 * GeV^4]" << endl;
	c3_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c3_T4 [fm^3/GeV * GeV^4]" << endl;
	c4_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c4_T5 [fm^3/GeV^2 * GeV^5]" << endl;


	// main calculation (14 moment approximation)
	for(int i = 0; i < muBpts; i++)					// baryon chemical potential column
	{
		for(int j = 0; j < Tpts; j++)				// temperature column
		{
			double muB = muB_array[i];
			double T = T_array[j];

			// prefactors
			double J20_fact = pow(T,4) / (two_pi2_hbarC3);
			double J21_fact = pow(T,4) / (3.0 * two_pi2_hbarC3);
			double J40_fact = pow(T,6) / (two_pi2_hbarC3);
			double J41_fact = pow(T,6) / (3.0 * two_pi2_hbarC3);
			double N10_fact = pow(T,3) / (two_pi2_hbarC3);
			double N30_fact = pow(T,5) / (two_pi2_hbarC3);
			double N31_fact = pow(T,5) / (3.0 * two_pi2_hbarC3);
			double M20_fact = J20_fact;
			double M21_fact = J21_fact;
			double A20_fact = J20_fact;
			double A21_fact = J21_fact;
			double B10_fact = N10_fact;

			// thermodynamic integrals
			double J20 = 0.0;
			double J21 = 0.0;
			double J40 = 0.0;
			double J41 = 0.0;
			double N10 = 0.0;
			double N30 = 0.0;
			double N31 = 0.0;
			double M20 = 0.0;
			double M21 = 0.0;

			double A20 = 0.0;
			double A21 = 0.0;
			double B10 = 0.0;

			// sum over resonance contributions to thermodynamic integrals
			for(int k = 0; k < N_resonances; k++)
			{
				if(particle[k].mass == 0.0) continue;	// skip the photon contribution

				// spin degeneracy , m/T, baryon number, quantum statistics sign for particle k
				double dof = (double)particle[k].gspin;
				double mbar = particle[k].mass/T;
				double b = (double)particle[k].baryon;
				double Theta = (double)particle[k].sign;

				double mass2 = particle[k].mass * particle[k].mass;

				// evaluate gauss integrals
				A20 += mass2 * dof * A20_fact * Gauss1D(J20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				A21 += mass2 * dof * A21_fact * Gauss1D(J21_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);

				J20 += dof * J20_fact * Gauss1D(J20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				J21 += dof * J21_fact * Gauss1D(J21_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				J40 += dof * J40_fact * Gauss1D(J40_int, pbar_root4, pbar_weight4, gla_pts, mbar, T, muB, b, Theta);
				J41 += dof * J41_fact * Gauss1D(J41_int, pbar_root4, pbar_weight4, gla_pts, mbar, T, muB, b, Theta);

				// skip iterations for baryon = 0
				if(particle[k].baryon != 0)
				{
				 	B10 += mass2 * dof * B10_fact * Gauss1D(N10_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
				 	N10 += dof * N10_fact * Gauss1D(N10_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
					N30 += dof * N30_fact * Gauss1D(N30_int, pbar_root3, pbar_weight3, gla_pts, mbar, T, muB, b, Theta);
					N31 += dof * N31_fact * Gauss1D(N31_int, pbar_root3, pbar_weight3, gla_pts, mbar, T, muB, b, Theta);
					M20 += dof * M20_fact * Gauss1D(M20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
					M21 += dof * M21_fact * Gauss1D(M21_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				}

			} // k

			// evaluate coefficients
			//bulk0 = N30*N30 - M20*J40;
			//bulk1 = N30*J20 - N10*J40;
			//bulk2 = M20*J20 - N10*N30;
			//denom = J21*bulk0 - N31*bulk1 + J41*bulk2;

			// update 3/25
			double bulk0 = (4.0*N30-B10)*N30 - M20*(4.0*J40-A20);
			double bulk1 = (B10-N30)*(4.0*J40-A20) - (4.0*N30-B10)*(A20-J40);
			double bulk2 = M20*(A20-J40) - (B10-N30)*N30;
			double denom = (A21-J41)*bulk0 + N31*bulk1 + (4.0*J41-A21)*bulk2;

			double c0_df14 = bulk0 / denom;
			double c1_df14 = bulk1 / denom;
			double c2_df14 = bulk2 / denom;

			double c3_df14 = J41 / (N31*N31 - M21*J41);
			double c4_df14 = - N31 / (N31*N31 - M21*J41);   // (accounted for the factor of 2)

			// check for singularities
			if(J21*bulk0 - N31*bulk1 + J41*bulk2 == 0.0)
			{
				cout << "Bulk denominator is zero" << endl;
				exit(-1);
			}
			if(N31*N31 - M21*J41 == 0.0)
			{
				cout << "Diffusion denominator is zero" << endl;
				exit(-1);
			}

			// output to file
			c0_table_df14  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << c0_df14 * pow(T,4) << endl;
			c1_table_df14  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << c1_df14 * pow(T,3) << endl;
			c2_table_df14  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << c2_df14 * pow(T,4) << endl;
			c3_table_df14  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << c3_df14 * pow(T,4) << endl;
			c4_table_df14  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << c4_df14 * pow(T,5) << endl;


		} // j
	} // i

	c0_table_df14.close();
	c1_table_df14.close();
	c2_table_df14.close();
	c3_table_df14.close();
	c4_table_df14.close();



	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//																					         ::
	// 					  Compute coefficients of chapman-enskog expansion 		     	         ::
	//																					         ::
	// df ~ (b.G + (u.p).F/T^2 - Deltamunu.pmu.pnu/(3(u.p)T)).Pi/betaPi 						 ::
	//																							 ::
	//		+ (nB/(e+p) - b/(u.p))pmu.Vmu/beta  +  pmu.pnu.pimunu/(2.betapi.T(u.p))              ::
	//																							 ::
	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	printf("Computing Chapman Enskog df coefficients...\n");

	ofstream G_table_dfce;
	ofstream F_table_dfce;
	ofstream betabulk_table_dfce;
	ofstream betaV_table_dfce;
	ofstream betapi_table_dfce;

	G_table_dfce.open("smash/G.dat", ios::out);
	F_table_dfce.open("smash/F.dat", ios::out);
	betabulk_table_dfce.open("smash/betabulk.dat", ios::out);
	betaV_table_dfce.open("smash/betaV.dat", ios::out);
	betapi_table_dfce.open("smash/betapi.dat", ios::out);

	G_table_dfce << Tpts << "\n" << muBpts << endl;
	F_table_dfce << Tpts << "\n" << muBpts << endl;
	betabulk_table_dfce << Tpts << "\n" << muBpts << endl;
	betaV_table_dfce << Tpts << "\n" << muBpts << endl;
	betapi_table_dfce << Tpts << "\n" << muBpts << endl;

	G_table_dfce << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "G [1]" << endl;
	F_table_dfce << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "F_over_T [fm^-1 / GeV]" << endl;
	betabulk_table_dfce << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "betabulk_over_T4 [fm^-4 / GeV^4]" << endl;
	betaV_table_dfce << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "betaV_over_T3 [fm^-3 / GeV^3]" << endl;
	betapi_table_dfce << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "betapi_over_T4 [fm^-4 / GeV^4]" << endl;


	// main calculation (Chapman-Enskog expansion)
	for(int i = 0; i < muBpts; i++)					// baryon chemical potential column
	{
		for(int j = 0; j < Tpts; j++)				// temperature column
		{
			// set T and muB
			double muB = muB_array[i];
			double T = T_array[j];

			// evaluate prefactors
			double nB_fact = pow(T,3) / (two_pi2_hbarC3);
			double e_fact = pow(T,4) / (two_pi2_hbarC3);
			double p_fact = pow(T,4) / (3.0 * two_pi2_hbarC3);
			double J30_fact = pow(T,5) / (two_pi2_hbarC3);
			double J32_fact = pow(T,5) / (15.0 * two_pi2_hbarC3);
			double N20_fact = pow(T,4) / (two_pi2_hbarC3);
			double M10_fact = pow(T,3) / (two_pi2_hbarC3);
			double M11_fact = pow(T,3) / (3.0 * two_pi2_hbarC3);

			// reset thermodynamic integrals to zero
			double nB = 0.0;
			double e = 0.0;
			double p = 0.0;
			double J30 = 0.0;
			double J32 = 0.0;
			double N20 = 0.0;
			double M10 = 0.0;
			double M11 = 0.0;

			// sum over resonance contributions to thermodynamic integrals
			for(int k = 0; k < N_resonances; k++)
			{
				if(particle[k].mass == 0.0) continue;	// skip the photon contribution

				// spin degeneracy , m/T, baryon number, quantum statistics sign for particle k
				double dof = (double)particle[k].gspin;
				double mbar = particle[k].mass/T;
				double b = (double)particle[k].baryon;
				double Theta = (double)particle[k].sign;

				// evaluate gauss integrals
				e += dof * e_fact * Gauss1D(e_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				p += dof * p_fact * Gauss1D(p_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				J30 += dof * J30_fact * Gauss1D(J30_int, pbar_root3, pbar_weight3, gla_pts, mbar, T, muB, b, Theta);
				J32 += dof * J32_fact * Gauss1D(J32_int, pbar_root3, pbar_weight3, gla_pts, mbar, T, muB, b, Theta);


				// skip iterations for baryon = 0
				if(particle[k].baryon != 0)
				{
					nB += dof * nB_fact * Gauss1D(nB_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
					N20 += dof * N20_fact * Gauss1D(N20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				 	M10 += dof * M10_fact * Gauss1D(M10_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
					M11 += dof * M11_fact * Gauss1D(M11_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
				}

			} // k

			// evaluate Chapman-Enskog coefficients (alphaB form)   // changed 3/29
			double G = ((e+p)*N20 - J30*nB) / (J30*M10 - N20*N20);
			double F = T * T * (N20*nB - (e+p)*M10) / (J30*M10 - N20*N20);
			double betabulk = G*nB*T + F*(e+p)/T + 5.0*J32/(3.0*T);


			// evaluate Chapman-Enskog coefficients (muB form)
			//G = T * ((e+p+muB*nB)*N20 - J30*nB - muB*(e+p)*M10) / (J30*M10 - N20*N20);
			//F = T*T * (N20*nB - (e+p)*M10) / (J30*M10 - N20*N20);
			//betabulk = G*nB + F*(e+p-muB*nB)/T + 5.0*J32/(3.0*T);     // why is this different than the formula in my paper?
																	    // this was done in muB, my paper done in alphaB

			double betaV = M11 - nB*nB*T/(e+p);
			double betapi = J32/T;


			// check for singularities
			if(betapi == 0.0)
			{
				cout << "Shear denominator is zero" << endl;
				exit(-1);
			}
			if(betabulk == 0.0)
			{
				cout << "Bulk denominator is zero" << endl;
				exit(-1);
			}
			if(betaV == 0.0)
			{
				cout << "Diffusion denominator is zero" << endl;
				exit(-1);
			}

			// output to file
			G_table_dfce  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << G << endl;
			F_table_dfce  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << F / T<< endl;
			betabulk_table_dfce  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << betabulk / pow(T,4) << endl;
			betaV_table_dfce  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << betaV / pow(T,3) << endl;
			betapi_table_dfce  << fixed << setw(SF) << T << "\t\t" << muB << "\t\t" << betapi / pow(T,4) << endl;


		} // j
	} // i

	G_table_dfce.close();
	F_table_dfce.close();
	betabulk_table_dfce.close();
	betaV_table_dfce.close();
	betapi_table_dfce.close();



	printf("Freeing memory...");

	free(pbar_root1);
	free(pbar_root2);
	free(pbar_root3);
	free(pbar_root4);

	free(pbar_weight1);
	free(pbar_weight2);
	free(pbar_weight3);
	free(pbar_weight4);

	delete[] particle;


	printf("done\n\n");

	printf("Finished!\n\n");

	return 0;
}
