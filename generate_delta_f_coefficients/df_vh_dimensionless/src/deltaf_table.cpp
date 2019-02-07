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
#include "../include/gauss_integration.hpp"
#include "../include/thermal_integrands.hpp"
#include "../include/freearray.hpp"
//#include <gsl/gsl_sf.h>

#define GEV_TO_INVERSE_FM 5.067731

const double hbarC = 0.197327053;  // GeV*fm
const double two_pi2_hbarC3 = 2.0 * pow(M_PI, 2) * pow(hbarC, 3);

// temporary
const int a = 21;
const int gla_pts = 64;
double root_gla[a][gla_pts];
double weight_gla[a][gla_pts];

int load_gauss_laguerre_data()
{
  FILE *fp;

  stringstream laguerre_roots_weights;
  laguerre_roots_weights << "gauss_laguerre/gla_roots_weights_" << gla_pts << "_points.txt";

  if((fp = fopen(laguerre_roots_weights.str().c_str(), "r")) == NULL)
  {
     return 1;
  }
  for(int i = 0; i < a; i++)
  {
   for(int j = 0; j < gla_pts; j++)
   {
      if(fscanf(fp, "%i %lf %lf", &i, &root_gla[i][j], &weight_gla[i][j])!= 3)
      	{
        	printf("error loading roots/weights of Gauss-Laguerre Quadradture at %d %d\n", i, j);
    		return 1;
    	}
   }
  }
  fclose(fp);
  return 0;
}




int main()
{

  printf("Start loading pdg.dat resonance info...");
	FILE *HRG;													// load hadron resonance data from pdg.dat
	stringstream resonances;
	resonances << "pdg.dat";
	HRG = fopen(resonances.str().c_str(),"r");					// open pdg.dat


	// pdg.dat contains (anti)mesons and baryons but
	// not antibaryons, so add antibaryons manually

	int N_mesons;												// number of mesons
	int N_baryons;												// number of baryons
	int N_antibaryons;											// number of antibaryons

	fscanf(HRG, "%d", &N_mesons);								// read first line: number of mesons
	fscanf(HRG, "%d", &N_baryons);								// read second line: number of baryons

	N_antibaryons = N_baryons;

	int N_resonances = N_mesons + N_baryons + N_antibaryons;    // total number of resonances

	// any extraneous data type is not an array
	int particle_id;											// particle #id
	char name[20];												// particle name
	double mass[N_resonances]; 									// (***) mass (GeV)
	double width;												// resonance width (GeV)
	int degeneracy[N_resonances];								// (***) degeneracy factor ~ 2*spin+1
	int baryon[N_resonances]; 									// (***) baryon number
	int strange;												// strangeness
	int charm;													// charmness
	int bottom;													// bottomness
	int isospin;												// isospin
	double charge;												// electric charge
	int decays;													// decay modes

	int anti_counter = 0; 									    // antibaryon counter (see for_loop below)

	// load data of mesons+baryons
	for(int k = 0; k < N_mesons + N_baryons; k++)
	{
		fscanf(HRG, "%d %s %lf %lf %d %d %d %d %d %d %lf %d", &particle_id, name, &mass[k], &width, &degeneracy[k], &baryon[k], &strange, &charm, &bottom, &isospin, &charge, &decays);

		if(baryon[k] == 1)
		{
			// fill in antibaryon data at end of array
			mass[anti_counter+N_mesons+N_baryons] = mass[k];
			degeneracy[anti_counter+N_mesons+N_baryons] = degeneracy[k];
			baryon[anti_counter+N_mesons+N_baryons] = -1;
			anti_counter++;
			// antibaryons correctly ammended to list (checked)
		}
	}

	int sign[N_resonances];										// sign array for bose/fermi distributions

	for(int k = 0; k < N_resonances; k++)
	{
		// degeneracy = (2*spin + 1)    			            // (isospin degeneracy split as individual particles)
		if(degeneracy[k] % 2 == 0) sign[k] = 1; 				// fermions
		else if(degeneracy[k] % 2 == 1) sign[k] = -1;			// bosons
	}

	fclose(HRG);												// close pdg.dat

  printf("done\n\n");


	// set up momentum-bar (pbar) roots and weights for thermodynamic integrals
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	// associated Laguerre polynomial type a
	const int a1 = 1; // a = 1 for X1q, where X = I, J, N, or M
	const int a2 = 2; // a = 2 for X2q
	const int a3 = 3; // a = 3 for X3q
	const int a4 = 4; // a = 4 for X4q


	// allocate roots
	double * pbar_root1 = (double *)malloc(gla_pts * sizeof(double));
	double * pbar_root2 = (double *)malloc(gla_pts * sizeof(double));
	double * pbar_root3 = (double *)malloc(gla_pts * sizeof(double));
	double * pbar_root4 = (double *)malloc(gla_pts * sizeof(double));


	// allocate weights
	double * pbar_weight1 = (double *)malloc(gla_pts * sizeof(double));
	double * pbar_weight2 = (double *)malloc(gla_pts * sizeof(double));
	double * pbar_weight3 = (double *)malloc(gla_pts * sizeof(double));
	double * pbar_weight4 = (double *)malloc(gla_pts * sizeof(double));


	// Load gauss laguerre roots-weights
	printf("Start loading gauss data...");
	int num_error;
	if((num_error = load_gauss_laguerre_data()) != 0)
	{
		fprintf(stderr, "Error loading gauss data (%d)!\n", num_error);
		return 1;
	}
	printf("done\n\n");


	// // set root-weight values
	for(int i = 0; i < gla_pts; i++)
	{
		pbar_root1[i] = root_gla[a1][i];
		pbar_root2[i] = root_gla[a2][i];
		pbar_root3[i] = root_gla[a3][i];
		pbar_root4[i] = root_gla[a4][i];

		pbar_weight1[i] = weight_gla[a1][i];
		pbar_weight2[i] = weight_gla[a2][i];
		pbar_weight3[i] = weight_gla[a3][i];
		pbar_weight4[i] = weight_gla[a4][i];
	}

	cout << "Starting..." << endl;


	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//																					   ::
	// 					   Compute coefficients of 14 moment approximation 		     	   ::
	//																					   ::
	// df ~ (c0 + b*c1*(u.p) + c2*(u.p)^2)*Pi + (b*c3 + c4(u.p))p_u*V^u + c5*p_u*p_v*pi^uv ::
	//																					   ::
	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



	// set up (T,muB) ranges
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	// Temperature / chemical potential table based on chemical freezeout data (Andronic 2017)

																	// (for main calculation loop below)
	double T;				  					            		// temperature (fm^-1)
	double muB;				   						        		// baryon chemical potential (fm^-1)


	int Tpts = 101;													// temperature grid points
	int muBpts = 81;												// chemical potential grid points

	double T_min = 0.1;       				               			// min / max temperature (GeV)
	double T_max = 0.2;
	double dT = (T_max - T_min)/(double)(Tpts-1);				    // temperature resolution


	double muB_min = 0.0;	   						    			// min / max chemical potential
	double muB_max = 0.8;
	double dmuB = (muB_max - muB_min)/(double)(muBpts-1);		    // chemical potential resolution

	cout << "dT =  " << dT << "\t" << "dmuB = " << dmuB << endl;
	//exit(-1);

	double * T_array = (double *)malloc(Tpts * sizeof(double));		// allocate temperature array
	double * muB_array = (double *)malloc(muBpts * sizeof(double));	// allocate chemical potential array

	// load temperature array
	for(int i = 0; i < Tpts; i++) T_array[i] = T_min + (double)i*dT;

	// load chemical potential array
	for(int i = 0; i < muBpts; i++)	muB_array[i] = muB_min + (double)i*dmuB;


	// declare thermodynamic integrals and prefactors
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	double A20, A20_fact;
	double A21, A21_fact;			// update bulk formula
	double J20, J20_fact;			// bulk
	double J21, J21_fact;			// bulk
	double J40, J40_fact;			// bulk
	double J41, J41_fact;			// bulk and diffusion
	//double J42, J42_fact; 			// shear

	double B10, B10_fact;
	double N10, N10_fact;			// bulk
	double N30, N30_fact;			// bulk
	double N31, N31_fact;			// bulk and difussion

	double M20, M20_fact;			// bulk
	double M21, M21_fact;			// diffusion


	// set up coefficient placeholders
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	size_t SF = 8;	        // significant digits (better way?)

	double c0_df14;				// scalar bulk
	double c1_df14;				// baryon bulk
	double c2_df14;				// energy bulk
	double c3_df14;				// baryon diffusion
	double c4_df14;				// energy diffusion
	double c5_df14;				// shear (not required)

	double bulk0;			// to facilitate organization of the bulk coefficients
	double bulk1;
	double bulk2;
	double denom;


	// data tables for the (T,muB) dependent coefficients
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	ofstream c0_table_df14;							// output file stream coefficient tables
	ofstream c1_table_df14;
	ofstream c2_table_df14;
	ofstream c3_table_df14;
	ofstream c4_table_df14;

	c0_table_df14.open("vh/c0_df14_vh.dat", ios::out);	// open .dat files for output
	c1_table_df14.open("vh/c1_df14_vh.dat", ios::out);
	c2_table_df14.open("vh/c2_df14_vh.dat", ios::out);
	c3_table_df14.open("vh/c3_df14_vh.dat", ios::out);
	c4_table_df14.open("vh/c4_df14_vh.dat", ios::out);

	// put number of T, muB points at header
	c0_table_df14 << Tpts << "\n" << muBpts << endl;
	c1_table_df14 << Tpts << "\n" << muBpts << endl;
	c2_table_df14 << Tpts << "\n" << muBpts << endl;
	c3_table_df14 << Tpts << "\n" << muBpts << endl;
	c4_table_df14 << Tpts << "\n" << muBpts << endl;

	// header labels
	c0_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c0_T4 [fm^3/GeV^3 * GeV^4]" << endl; // output stream first lines
	c1_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c1_T3 [fm^3/GeV^2 * GeV^3]" << endl; // (fill in units later)
	c2_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c2_T4 [fm^3/GeV^3 * GeV^4]" << endl;
	c3_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c3_T4 [fm^3/GeV * GeV^4]" << endl;
	c4_table_df14 << "T [GeV]" << "\t\t" << "muB [GeV]" << "\t\t" << "c4_T5 [fm^3/GeV^2 * GeV^5]" << endl;


	// stat mech quantities for resonance sum
	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	double dof;		// degrees of freedom
	double Theta;	// quantum statistics sign
	double mbar; 	// mass/T
	double b; 		// baryon number


		// main calculation (14 moment approximation)
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	for(int i = 0; i < muBpts; i++)					// baryon chemical potential column
	{
		for(int j = 0; j < Tpts; j++)				// temperature column
		{
			// set T and muB
			muB = muB_array[i];
			T = T_array[j];

			// evaluate prefactors
			J20_fact = pow(T,4) / (two_pi2_hbarC3);
			J21_fact = pow(T,4) / (3.0 * two_pi2_hbarC3);
			J40_fact = pow(T,6) / (two_pi2_hbarC3);
			J41_fact = pow(T,6) / (3.0 * two_pi2_hbarC3);
			//J42_fact = pow(T,6) / (15.0 * two_pi2_hbarC3);
			N10_fact = pow(T,3) / (two_pi2_hbarC3);
			N30_fact = pow(T,5) / (two_pi2_hbarC3);
			N31_fact = pow(T,5) / (3.0 * two_pi2_hbarC3);
			M20_fact = J20_fact;
			M21_fact = J21_fact;
			A20_fact = J20_fact;
			A21_fact = J21_fact;
			B10_fact = N10_fact;

			// reset thermodynamic integrals to zero
			J20 = 0.0;
			J21 = 0.0;
			J40 = 0.0;
			J41 = 0.0;
			//J42 = 0.0;
			N10 = 0.0;
			N30 = 0.0;
			N31 = 0.0;
			M20 = 0.0;
			M21 = 0.0;

			A20 = 0.0;
			A21 = 0.0;
			B10 = 0.0;

			// sum over resonance contributions to thermodynamic integrals
			for(int k = 0; k < N_resonances; k++)
			{
				// degrees of freedom, etc for particle k
				dof = (double)degeneracy[k];
				mbar = mass[k]/T;
				b = (double)baryon[k];
				Theta = (double)sign[k];

				double mass2 = mass[k]*mass[k];

				// evaluate gauss integrals
				A20 += mass2 * dof * A20_fact * Gauss1D(J20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				A21 += mass2 * dof * A21_fact * Gauss1D(J21_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);

				J20 += dof * J20_fact * Gauss1D(J20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				J21 += dof * J21_fact * Gauss1D(J21_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				J40 += dof * J40_fact * Gauss1D(J40_int, pbar_root4, pbar_weight4, gla_pts, mbar, T, muB, b, Theta);
				J41 += dof * J41_fact * Gauss1D(J41_int, pbar_root4, pbar_weight4, gla_pts, mbar, T, muB, b, Theta);
				//J42 += dof * J42_fact * Gauss1D(J42_int, pbar_root4, pbar_weight4, gla_pts, mbar, T, muB, b, Theta);

				// skip iterations for baryon = 0
				if(baryon[k] != 0)
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
			bulk0 = (4.0*N30-B10)*N30 - M20*(4.0*J40-A20);
			bulk1 = (B10-N30)*(4.0*J40-A20) - (4.0*N30-B10)*(A20-J40);
			bulk2 = M20*(A20-J40) - (B10-N30)*N30;
			denom = (A21-J41)*bulk0 + N31*bulk1 + (4.0*J41-A21)*bulk2;


			c0_df14 = bulk0 / denom;
			c1_df14 = bulk1 / denom;
			c2_df14 = bulk2 / denom;

			c3_df14 = J41 / (N31*N31 - M21*J41);
			c4_df14 = - N31 / (N31*N31 - M21*J41);   // (accounted for the factor of 2)

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

	c0_table_df14.close();	// close table files (14 moment approximation)
	c1_table_df14.close();
	c2_table_df14.close();
	c3_table_df14.close();
	c4_table_df14.close();

	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::




	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//																					         ::
	// 					  Compute coefficients of chapman-enskog expansion 		     	         ::
	//																					         ::
	// df ~ (b*G + (u.p)*F/T^2 - Delta_munu*p^mu*p^nu/(3(u.p)T))*Pi/betaPi 						 ::
	//																							 ::
	//		+ (nB/(e+p) - b/(u.p))p_u*V^u/betaV                                                  ::
	//																							 ::
	//		+ p_u*p_v*pi^uv/(2*betapi*T(u.p)) 		                                             ::
	//																					         ::
	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


	// thermodynamic integrals and prefactors
	double nB, nB_fact;		// bulk and diffusion
	double e, e_fact;		// bulk and diffusion
	double p, p_fact;		// bulk and diffusion
	double J30, J30_fact;	// bulk
	double J32, J32_fact;	// shear
	double N20, N20_fact;	// bulk
	double M10, M10_fact;	// bulk
	double M11, M11_fact;	// diffusion


	// coefficient placeholders
	double G;
	double F;
	double betabulk;
	double betaV;
	double betapi;


	// data tables for the (T,muB) dependent coefficients
	ofstream G_table_dfce;
	ofstream F_table_dfce;
	ofstream betabulk_table_dfce;
	ofstream betaV_table_dfce;
	ofstream betapi_table_dfce;

	G_table_dfce.open("vh/G_dfce_vh.dat", ios::out);
	F_table_dfce.open("vh/F_dfce_vh.dat", ios::out);
	betabulk_table_dfce.open("vh/betabulk_dfce_vh.dat", ios::out);
	betaV_table_dfce.open("vh/betaV_dfce_vh.dat", ios::out);
	betapi_table_dfce.open("vh/betapi_dfce_vh.dat", ios::out);

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
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	for(int i = 0; i < muBpts; i++)					// baryon chemical potential column
	{
		for(int j = 0; j < Tpts; j++)				// temperature column
		{
			// set T and muB
			muB = muB_array[i];
			T = T_array[j];

			// evaluate prefactors
			nB_fact = pow(T,3) / (two_pi2_hbarC3);
			e_fact = pow(T,4) / (two_pi2_hbarC3);
			p_fact = pow(T,4) / (3.0 * two_pi2_hbarC3);
			J30_fact = pow(T,5) / (two_pi2_hbarC3);
			J32_fact = pow(T,5) / (15.0 * two_pi2_hbarC3);
			N20_fact = pow(T,4) / (two_pi2_hbarC3);
			M10_fact = pow(T,3) / (two_pi2_hbarC3);
			M11_fact = pow(T,3) / (3.0 * two_pi2_hbarC3);

			// reset thermodynamic integrals to zero
			nB = 0.0;
			e = 0.0;
			p = 0.0;
			J30 = 0.0;
			J32 = 0.0;
			N20 = 0.0;
			M10 = 0.0;
			M11 = 0.0;

			// sum over resonance contributions to thermodynamic integrals
			for(int k = 0; k < N_resonances; k++)
			{
				// degrees of freedom, etc for particle k
				dof = (double)degeneracy[k];
				mbar = mass[k]/T;
				b = (double)baryon[k];
				Theta = (double)sign[k];


				// evaluate gauss integrals
				e += dof * e_fact * Gauss1D(e_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				p += dof * p_fact * Gauss1D(p_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				J30 += dof * J30_fact * Gauss1D(J30_int, pbar_root3, pbar_weight3, gla_pts, mbar, T, muB, b, Theta);
				J32 += dof * J32_fact * Gauss1D(J32_int, pbar_root3, pbar_weight3, gla_pts, mbar, T, muB, b, Theta);


				// skip iterations for baryon = 0
				if(baryon[k] != 0)
				{
					nB += dof * nB_fact * Gauss1D(nB_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
					N20 += dof * N20_fact * Gauss1D(N20_int, pbar_root2, pbar_weight2, gla_pts, mbar, T, muB, b, Theta);
				 	M10 += dof * M10_fact * Gauss1D(M10_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
					M11 += dof * M11_fact * Gauss1D(M11_int, pbar_root1, pbar_weight1, gla_pts, mbar, T, muB, b, Theta);
				}

			} // k

			// cout << "e = " << e << endl;
			// cout << "p = " << p << endl;
			// cout << "nB = " << nB << endl;

			// evaluate Chapman-Enskog coefficients (alphaB form)   // changed 3/29
			G = ((e+p)*N20 - J30*nB) / (J30*M10 - N20*N20);
			F = T * T * (N20*nB - (e+p)*M10) / (J30*M10 - N20*N20);
			betabulk = G*nB*T + F*(e+p)/T + 5.0*J32/(3.0*T);


			// evaluate Chapman-Enskog coefficients (muB form)
			//G = T * ((e+p+muB*nB)*N20 - J30*nB - muB*(e+p)*M10) / (J30*M10 - N20*N20);
			//F = T*T * (N20*nB - (e+p)*M10) / (J30*M10 - N20*N20);
			//betabulk = G*nB + F*(e+p-muB*nB)/T + 5.0*J32/(3.0*T);     // why is this different than the formula in my paper?
																	    // this was done in muB, my paper done in alphaB

			betaV = M11 - nB*nB*T/(e+p);
			betapi = J32/T;


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

	G_table_dfce.close();	// close table files (Chapman-Enskog expansion)
	F_table_dfce.close();
	betabulk_table_dfce.close();
	betaV_table_dfce.close();
	betapi_table_dfce.close();

	//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


	printf("Freeing memory...");

	free(pbar_root1);
	free(pbar_root2);
	free(pbar_root3);
	free(pbar_root4);

	free(pbar_weight1);
	free(pbar_weight2);
	free(pbar_weight3);
	free(pbar_weight4);


	free(T_array);
	free(muB_array);


	//free_2D(A,n);

	printf("done\n\n");

	printf("Finished!\n\n");

	return 0;
}
