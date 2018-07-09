#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <array>
#ifdef _OMP
#include <omp.h>
#endif
#include "main.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "Stopwatch.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include "gaussThermal.h"
#include "particle.h"
#ifdef _OPENACC
#include <accelmath.h>
#endif
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

//#define FORCE_F0

using namespace std;

void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2)
  {
    printf("Sampling particles from VH \n");

    //double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
    int npart = number_of_chosen_particles;
    int particle_index = 0; //runs over all particles in final sampled particle list

    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts];    // eta_weight * delta_eta
    double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
    }

    //get viscous correction coefficients
    double F        = df_coeff[0];
    double G        = df_coeff[1];
    double betabulk = df_coeff[2];
    double betaV    = df_coeff[3];
    double betapi   = df_coeff[4];

    //loop over all freezeout cells
    for (int icell = 0; icell < FO_length; icell++)
    {
      // set freezeout info to local varibles to reduce(?) memory access outside cache :
      //get fo cell coordinates
      double tau = tau_fo[icell];         // longitudinal proper time
      double x = x_fo[icell];
      double y = y_fo[icell];
      double tau2 = tau * tau;

      if(DIMENSION == 3)
      {
        etaValues[0] = eta_fo[icell];     // spacetime rapidity from surface file
      }

      //FIX THIS - NEED A LOOP OVER ETA FOR CASE OF 2D SURFACE
      double eta = etaValues[0];
      //FIX THIS

      double dat = dat_fo[icell];         // covariant normal surface vector
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];

      double ux = ux_fo[icell];           // contravariant fluid velocity
      double uy = uy_fo[icell];           // reinforce normalization
      double un = un_fo[icell];
      double ut = sqrt(fabs(1.0 + ux*ux + uy*uy + tau2*un*un));

      double T = T_fo[icell];             // temperature
      double E = E_fo[icell];             // energy density
      double P = P_fo[icell];             // pressure

      double pitt = pitt_fo[icell];       // pi^munu
      double pitx = pitx_fo[icell];
      double pity = pity_fo[icell];
      double pitn = pitn_fo[icell];
      double pixx = pixx_fo[icell];
      double pixy = pixy_fo[icell];
      double pixn = pixn_fo[icell];
      double piyy = piyy_fo[icell];
      double piyn = piyn_fo[icell];
      double pinn = pinn_fo[icell];

      double bulkPi = bulkPi_fo[icell];   // bulk pressure

      double muB = 0.0;
      double alphaB = 0.0;                       // baryon chemical potential
      double nB = 0.0;                        // net baryon density
      double Vt = 0.0;                        // baryon diffusion
      double Vx = 0.0;
      double Vy = 0.0;
      double Vn = 0.0;

      if(INCLUDE_BARYON)
      {
        muB = muB_fo[icell];
        alphaB = muB / T;

        if(INCLUDE_BARYONDIFF_DELTAF)
        {
          nB = nB_fo[icell];
          Vt = Vt_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
        }
      }

      //common prefactor
      double udotdsigma = dat*ut + dax*ux + day*uy + dan*un;
      double num_factor = 1.0 / 2.0 / M_PI / M_PI / hbarC / hbarC / hbarC;
      double prefactor = num_factor * udotdsigma * T;

      //the total mean number of particles (all species) for the FO cell
      double dN_tot = 0.0;

      //a list with species dependent densities
      std::vector<double> density_list;
      density_list.resize(npart);

      //loop over all species
      for (int ipart = 0; ipart < npart; ipart++)
      {
        // set particle properties
        double mass = Mass[ipart];    // (GeV)
        double mass2 = mass * mass;
        double mbar = mass / T;
        double sign = Sign[ipart];
        double degeneracy = Degeneracy[ipart];
        double baryon = 0.0;
        double chem = 0.0;           // chemical potential term in feq
        if(INCLUDE_BARYON)
        {
          baryon = Baryon[ipart];
          chem = baryon * muB;
        }

        //first calculate thermal equil. number of particles
        double sum = 0.0;
        double dN_thermal = 0.0;

        //for light particles need more than leading order term in BE or FD expansion
        int sum_max = 2;
        if ( mass / T < 2.0 ) sum_max = 10;

        for (int kk = 1; kk < sum_max; kk++) //truncate expansion of BE or FD distribution
        {
          double k = (double)kk;
          sum += pow(-sign, kk+1) * exp(k * chem / T) * gsl_sf_bessel_Kn(2, k*mbar) / k;
        }
        dN_thermal = prefactor * sum * degeneracy * mass2;

        //correction from bulk pressure
        double dN_bulk = 0.0;
        if (INCLUDE_BULK_DELTAF)
        {
          double N10_fact = pow(T,3) / (2.0 * M_PI * M_PI) / hbarC / hbarC / hbarC;
          double J20_fact = pow(T,4) / (2.0 * M_PI * M_PI) / hbarC / hbarC / hbarC;
          double N10 = degeneracy * N10_fact * GaussThermal(N10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
          double J20 = degeneracy * J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
          dN_bulk = (bulkPi / betabulk) * ( dN_thermal + (N10 * G) + (J20 * F / pow(T,2)) );
        }

        //correction from diffusion current
        double dN_diff = 0.0;
        if (INCLUDE_BARYONDIFF_DELTAF)
        {

        }

        //add them up
        //this is the mean number of particles of species ipart
        double dN = dN_thermal + dN_bulk + dN_diff;

        //save to a list to later sample inidividual species
        density_list[ipart] = dN;

        //add this to the total mean number of hadrons for the FO cell
        dN_tot += dN;

    }

    //now sample the number of particles (all species)
    //non-deterministic random number generator
    std::random_device gen1;
    std::random_device gen2;

    //deterministic random number generator
    //std::mt19937 gen(1701);

    std::poisson_distribution<> poisson_distr(dN_tot);

    //sample total number of hadrons in FO cell
    int N_hadrons = poisson_distr(gen1);

    //make a discrete distribution with weights according to particle densities
    std::discrete_distribution<int> particle_numbers ( density_list.begin(), density_list.end() );

    for (int n = 0; n < N_hadrons; n++)
    {
      //sample discrete distribution
      int idx_sampled = particle_numbers(gen2); //this gives the index of the sampled particle

      //get the mass of the sampled particle
      double mass = Mass[idx_sampled];  // (GeV)

      //get the MC ID of the sampled particle
      int mcid = MCID[idx_sampled];
      particle_list[particle_index].mcID = mcid;

      //sample the momentum from distribution using Scott Pratt Trick
      //stuff

      //set coordinates of production to FO cell coords
      particle_list[particle_index].tau = tau;
      particle_list[particle_index].x = x;
      particle_list[particle_index].y = y;
      particle_list[particle_index].eta = eta;

      //FIX MOMENTA TO NONTRIVIAL VALUES
      //set particle momentum to sampled values
      particle_list[particle_index].E = 0.0;
      particle_list[particle_index].px = 0.0;
      particle_list[particle_index].py = 0.0;
      particle_list[particle_index].pz = 0.0;
      // FIX MOMENTA TO NONTRIVIAL VALUES

      particle_index += 1;

    } // for (int n = 0; n < N_hadrons; n++)
  } // for (int icell = 0; icell < FO_length; icell++)
}

void EmissionFunctionArray::sample_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
  double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
  double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
{
  printf("Sampling particles from VAH \n");

}
