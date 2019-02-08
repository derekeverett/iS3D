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

const double hbarC = 0.197327053;  // GeV*fm
const double two_pi2_hbarC3 = 2.0 * pow(M_PI, 2) * pow(hbarC, 3);

const int Maxparticle = 600; //size of array for storage of the particles
const int Maxdecaychannel = 50;
const int Maxdecaypart = 5;

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



typedef struct
{
  long int mc_id; 	// Monte Carlo number according PDG
  string name;		// name
  double mass;		// mass (GeV)
  double width;		// resonance width (GeV)
  int gspin; 		// spin degeneracy
  int baryon;		// baryon number
  int strange;		// strangeness
  int charm;		// charmness
  int bottom;		// bottomness
  int gisospin; 	// isospin degeneracy
  int charge;		// electric charge
  int decays; 		// amount of decays listed for this resonance
  int stable; 		// defines whether this particle is considered as stable under strong interactions

  int decays_Npart[Maxdecaychannel];
  double decays_branchratio[Maxdecaychannel];
  int decays_part[Maxdecaychannel][Maxdecaypart];

  int sign; 		// Bose-Einstein or Dirac-Fermi statistics

} particle_info;


int main()
{
  	printf("Loading particle info from pdg.dat...(make sure only 1 blank line at end of file\n");

  	particle_info particle[Maxparticle];		// particle info struct

  	int N_resonances = 0;						// count number of resonances in pdg file

	double eps = 1e-15;
	
	ifstream pdg("pdg.dat");
	int local_i = 0;
	int dummy_int;

	while(!pdg.eof())
	{
		pdg >> particle[local_i].mc_id;			// monte carlo id
		pdg >> particle[local_i].name;			// name (now it's in strange characters is that okay?)
		pdg >> particle[local_i].mass;			// mass (GeV)
		pdg >> particle[local_i].width;			// resonance width (GeV)
		pdg >> particle[local_i].gspin;	      	// spin degeneracy
		pdg >> particle[local_i].baryon;		// baryon number
		pdg >> particle[local_i].strange;	   	// strangeness
		pdg >> particle[local_i].charm;			// charmness
		pdg >> particle[local_i].bottom;	   	// bottomness
		pdg >> particle[local_i].gisospin;     	// isospin degeneracy
		pdg >> particle[local_i].charge;		// electric charge
		pdg >> particle[local_i].decays;		// decay channels

	   for (int j = 0; j < particle[local_i].decays; j++)
	    {
	      pdg >> dummy_int;
	      pdg >> particle[local_i].decays_Npart[j];
	      pdg >> particle[local_i].decays_branchratio[j];
	      pdg >> particle[local_i].decays_part[j][0];
	      pdg >> particle[local_i].decays_part[j][1];
	      pdg >> particle[local_i].decays_part[j][2];
	      pdg >> particle[local_i].decays_part[j][3];
	      pdg >> particle[local_i].decays_part[j][4];
	    }

	    //decide whether particle is stable under strong interactions
	    if (particle[local_i].decays_Npart[0] == 1) particle[local_i].stable = 1;
	    else particle[local_i].stable = 0;

	    // add anti-particle entry
	    if (particle[local_i].baryon == 1)
	    {
	      local_i++;
	      particle[local_i].mc_id = -particle[local_i-1].mc_id;
	      ostringstream antiname;
	      antiname << "Anti-baryon-" << particle[local_i-1].name;
	      particle[local_i].name = antiname.str();
	      particle[local_i].mass = particle[local_i-1].mass;
	      particle[local_i].width = particle[local_i-1].width;
	      particle[local_i].gspin = particle[local_i-1].gspin;
	      particle[local_i].baryon = -particle[local_i-1].baryon;
	      particle[local_i].strange = -particle[local_i-1].strange;
	      particle[local_i].charm = -particle[local_i-1].charm;
	      particle[local_i].bottom = -particle[local_i-1].bottom;
	      particle[local_i].gisospin = particle[local_i-1].gisospin;
	      particle[local_i].charge = -particle[local_i-1].charge;
	      particle[local_i].decays = particle[local_i-1].decays;
	      particle[local_i].stable = particle[local_i-1].stable;

	      for (int j = 0; j < particle[local_i].decays; j++)
	      {
	        particle[local_i].decays_Npart[j]=particle[local_i-1].decays_Npart[j];
	        particle[local_i].decays_branchratio[j]=particle[local_i-1].decays_branchratio[j];

	        for (int k=0; k< Maxdecaypart; k++)
	        {
	          if(particle[local_i-1].decays_part[j][k] == 0) particle[local_i].decays_part[j][k] = (particle[local_i-1].decays_part[j][k]);
	          else
	          {
	            int idx;
	            // find the index for decay particle
	            for(idx = 0; idx < local_i; idx++)
	            if (particle[idx].mc_id == particle[local_i-1].decays_part[j][k]) break;
	            if(idx == local_i && particle[local_i-1].stable == 0 && particle[local_i-1].decays_branchratio[j] > eps)
	            {
	              cout << "Error: can not find decay particle index for anti-baryon!" << endl;
	              cout << "particle mc_id : " << particle[local_i-1].decays_part[j][k] << endl;
	              exit(1);
	            }
	            if (particle[idx].baryon == 0 && particle[idx].charge == 0 && particle[idx].strange == 0) particle[local_i].decays_part[j][k] = (particle[local_i-1].decays_part[j][k]);
	            else particle[local_i].decays_part[j][k] = (- particle[local_i-1].decays_part[j][k]);
	          }
	        }
	      }
	    }
	    local_i++;	// add one to the counting variable "i" for the meson/baryon
	}

	pdg.close();

	N_resonances = local_i; 				    // take account the final fake one
	//N_resonances = local_i - 1; 				// take account the final fake one (why were there two blanks?)

	for(int i = 0; i < N_resonances; i++)
	{
		if(particle[i].baryon == 0) 			// set the quantum statistics sign
		{
			particle[i].sign = -1;
		}
		else particle[i].sign = 1;
	}


	// pdg.dat contains (anti)mesons and baryons but
	// not antibaryons, so add antibaryons manually
	/*
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
   */


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

	int Tpts = 101;													// temperature grid points
	int muBpts = 81;												// chemical potential grid points

	double T_min = 0.1;       				               			// min / max temperature (GeV)
	double T_max = 0.2;
	double dT = (T_max - T_min)/(double)(Tpts-1);				    // temperature resolution


	double muB_min = 0.0;	   						    			// min / max chemical potential
	double muB_max = 0.8;
	double dmuB = (muB_max - muB_min)/(double)(muBpts-1);		    // chemical potential resolution

	printf("\nTable properties:\n\n");
	printf("T_min = %lf GeV\n", T_min);
	printf("T_max = %lf GeV\n", T_max);
	printf("T_pts = %d\n", Tpts);
	printf("dT = %lf GeV\n", dT);
	printf("\n");
	printf("muB_min = %lf GeV\n", muB_min);
	printf("muB_max = %lf GeV\n", muB_max);
	printf("muB_pts = %d\n", muBpts);
	printf("dmuB = %lf GeV\n", dmuB);
	printf("\n");

	double * T_array = (double *)malloc(Tpts * sizeof(double));		// allocate temperature array
	double * muB_array = (double *)malloc(muBpts * sizeof(double));	// allocate chemical potential array

	// load temperature array
	for(int i = 0; i < Tpts; i++) T_array[i] = T_min + (double)i*dT;

	// load chemical potential array
	for(int i = 0; i < muBpts; i++)	muB_array[i] = muB_min + (double)i*dmuB;


	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	size_t SF = 8;	        	// significant digits (better way?)

	// data tables for the (T,muB) dependent coefficients
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	ofstream c0_table_df14;								// output file stream coefficient tables
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




	// main calculation (14 moment approximation)
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	for(int i = 0; i < muBpts; i++)					// baryon chemical potential column
	{
		for(int j = 0; j < Tpts; j++)				// temperature column
		{
			// set T and muB
			double muB = muB_array[i];
			double T = T_array[j];

			// evaluate prefactors
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

			// reset thermodynamic integrals to zero
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
				//J42 += dof * J42_fact * Gauss1D(J42_int, pbar_root4, pbar_weight4, gla_pts, mbar, T, muB, b, Theta);

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


	printf("done\n\n");

	printf("Finished!\n\n");

	return 0;
}
