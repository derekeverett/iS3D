#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <chrono>
#include <limits>
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
//#ifdef _OPENACC
//#include <accelmath.h>
//#endif
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

using namespace std;

lrf_momentum Sample_Momentum(double mass, double T, double alphaB)
{
  lrf_momentum pLRF;

  // only need to seed once right?
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);

  //TO DO add routine that works for pions!
  // if (mass / T < 2.0)
  //stuff

  //light hadrons, heavier than pions
  if(mass / T < 0.6)
  {
    bool rejected = true;
    while(rejected)
    {
      // do you need different generators for (r1,r2,r3,propose)?
      // I'm guessing no...
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p = - T * (l1 + l2 + l3);

      if(::isnan(p)) printf("found p nan!\n");

      double E = sqrt(fabs(p * p + mass * mass));
      double weight = exp((p - E) / T);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // check acceptance
      if(propose < weight)
      {
        rejected = false;

        // calculate angles and pLRF components
        double costheta = (l1 - l2) / (l1 + l2);
        if(::isnan(costheta)) printf("found costheta nan!\n");
        double phi = 2.0 * M_PI * pow(l1 + l2, 2) / pow(l1 + l2 + l3, 2);
        if(::isnan(phi)) printf("found phi nan!\n");
        double sintheta = sqrt(1.0 - costheta * costheta);
        if(::isnan(sintheta)) printf("found sintheta nan!\n");

        pLRF.x = p * costheta;
        pLRF.y = p * sintheta * cos(phi);
        pLRF.z = p * sintheta * sin(phi);
      } // acceptance
    } // while loop
  } // mass / T < 0.6

  //heavy hadrons
  //use variable transformation described in LongGang's Sampler Notes
  else
  {
    //determine which part of integrand dominates
    double I1 = mass * mass * T;
    double I2 = 2 * mass * T * T;
    double I3 = 2 * pow(T,3);

    double Itot = I1 + I2 + I3;

    bool rejected = true;
    while(rejected)
    {
      // do you need different generators for (r1,r2,r3,propose)?
      // I'm guessing no...
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      double k = 0.0; //variable transform k = E - m
      if (propose < I1 / Itot)
      {
        k = -T * l1;
      }
      else if (propose < (I1 + I2) / Itot)
      {
        k = -T * (l1 + l2);
      }
      else
      {
        k = -T * (l1 + l2 + l3);
      }
      double E = k + mass;
      double p = sqrt( E * E - mass * mass);
      propose = generate_canonical<double, numeric_limits<double>::digits>(generator);
      double weight = p / E;

      // check acceptance
      if(propose < weight)
      {
        rejected = false;

        // calculate angles and pLRF components
        double costheta = (l1 - l2) / (l1 + l2);
        if(::isnan(costheta)) printf("found costheta nan!\n");
        double phi = 2.0 * M_PI * pow(l1 + l2, 2) / pow(l1 + l2 + l3, 2);
        if(::isnan(phi)) printf("found phi nan!\n");
        double sintheta = sqrt(1.0 - costheta * costheta);
        if(::isnan(sintheta)) printf("found sintheta nan!\n");

        pLRF.x = p * costheta;
        pLRF.y = p * sintheta * cos(phi);
        pLRF.z = p * sintheta * sin(phi);
      } // acceptance
    } // while loop
  } //heavy hadron

  return pLRF;
}





void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2, double *pbar_root3, double *pbar_weight3)
  {
    printf("sampling particles from vhydro with df...\n");

    //double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
    int npart = number_of_chosen_particles;
    //int particle_index = 0; //runs over all particles in final sampled particle list

    double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);

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

    // set df coefficients (viscous corrections)
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;
    double F = 0.0;
    double G = 0.0;
    double betabulk = 0.0;
    double betaV = 0.0;
    double betapi = 0.0;

    switch(DF_MODE)
    {
      case 1: // 14-moment
      {
        // bulk coefficients
        c0 = df_coeff[0];
        c1 = df_coeff[1];
        c2 = df_coeff[2];
        // diffusion coefficients
        c3 = df_coeff[3];
        c4 = df_coeff[4];
        // shear coefficient evaluated later
        break;
      }
      case 2: // Chapman-Enskog
      {
        // bulk coefficients
        F = df_coeff[0];
        G = df_coeff[1];
        betabulk = df_coeff[2];
        // diffusion coefficient
        betaV = df_coeff[3];
        // shear coefficient
        betapi = df_coeff[4];
        break;
      }
      default:
      {
        cout << "Please choose df_mode = 1 or 2 in parameters.dat" << endl;
        exit(-1);
      }
    }

    //loop over all freezeout cells
    #pragma omp parallel for
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
      double un = un_fo[icell];           // u^\eta
      double ut = sqrt(fabs(1.0 + ux*ux + uy*uy + tau2*un*un)); //u^\tau

      //get the cartesian components of fluid velocity for the boost LRF -> Lab
      double u0 = ut * cosh(un);
      double u3 = ut * sinh(un);

      //beta factors for boost from LRF to Lab
      double betax = ux/u0;
      double betay = uy/u0;
      double betaz = u3/u0;
      double beta2 = betax * betax + betay * betay + betaz * betaz;

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
      double Vdotdsigma = 0.0;
      double baryon_enthalpy_ratio = 0.0;

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
          baryon_enthalpy_ratio = nB / (E + P);
          // V^\mu d^3sigma_\mu
          Vdotdsigma = dat*Vt + dax*Vx + day*Vy + dan*Vn;
        }
      }

      //common prefactor
      double udotdsigma = dat*ut + dax*ux + day*uy + dan*un;
      //double num_factor = 1.0 / 2.0 / M_PI / M_PI / hbarC / hbarC / hbarC;
      //double prefactor = num_factor * udotdsigma * T;

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

        // first calculate thermal equilibrium particles
          double neq = 0.0;
          double dN_thermal = 0.0;

          // for light particles need more than leading order term in BE or FD expansion
          int jmax = 2;
          if(mass / T < 2.0) jmax = 10;

          double sign_factor = -sign;

          for(int j = 1; j < jmax; j++) // truncate expansion of BE or FD distribution
          {
            double k = (double)j;
            sign_factor *= (-sign);
            //neq += (pow(-sign, k+1) * exp(n * chem / T) * gsl_sf_bessel_Kn(2, n * mbar) / n);
            //cout << "Here" << endl;
            neq += (sign_factor * exp(k * alphaB) * gsl_sf_bessel_Kn(2, k * mbar) / k);
          }
          neq *= (degeneracy * mass2 * T / two_pi2_hbarC3);

          dN_thermal = udotdsigma * neq;

          // bulk pressure: density correction
            double dN_bulk = 0.0;
            if(INCLUDE_BULK_DELTAF)
            {
              switch(DF_MODE)
              {
                case 1: // 14 moment (not yet finished, work out the code stupid :P)
                {
                  double J10_fact = pow(T,3) / two_pi2_hbarC3;
                  double J20_fact = pow(T,4) / two_pi2_hbarC3;
                  double J30_fact = pow(T,5) / two_pi2_hbarC3;
                  double J10 = degeneracy * J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
                  double J20 = degeneracy * J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
                  double J30 = degeneracy * J30_fact * GaussThermal(J30_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);

                  dN_bulk = udotdsigma * bulkPi * ((c0 - c2) * mass2 * J10 + (c1 * baryon * J20) + (4.0 * c2 - c0) * J30);
                  break;
                }
                case 2: // Chapman-Enskog
                {
                  double J10_fact = pow(T,3) / two_pi2_hbarC3;
                  double J20_fact = pow(T,4) / two_pi2_hbarC3;
                  double J10 = degeneracy * J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
                  double J20 = degeneracy * J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
                  dN_bulk = udotdsigma * (bulkPi / betabulk) * (neq + (baryon * J10 * G) + (J20 * F / T / T));

                  break;
                }
                default:
                {
                  cout << "Please choose df_mode = 1 or 2 in parameters.dat" << endl;
                  exit(-1);
                }
              } //switch(DF_MODE)
            } //if(INCLUDE_BULK_DELTAF)

            // baryon diffusion: density correction
              double dN_diff = 0.0;
              if(INCLUDE_BARYONDIFF_DELTAF)
              {
                switch(DF_MODE)
                {
                  case 1: // 14 moment
                  {
                    // it's probably faster to use J31
                    double J31_fact = pow(T,5) / two_pi2_hbarC3 / 3.0;
                    double J31 = degeneracy * J31_fact * GaussThermal(J31_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);
                    // these coefficients need to be loaded.
                    // c3 ~ cV / V
                    // c4 ~ 2cW / V
                    dN_diff = - (Vdotdsigma / betaV) * ((baryon * c3 * neq * T) + (c4 * J31));
                    break;
                  }
                  case 2: // Chapman Enskog
                  {
                    double J11_fact = pow(T,3) / two_pi2_hbarC3 / 3.0;
                    double J11 = degeneracy * J11_fact * GaussThermal(J11_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
                    dN_diff = - (Vdotdsigma / betaV) * (neq * T * baryon_enthalpy_ratio - baryon * J11);
                    break;
                  }
                  default:
                  {
                    cout << "Please choose df_mode = 1 or 2 in parameters.dat" << endl;
                    exit(-1);
                  }
                } //switch(DF_MODE)
              } //if(INCLUDE_BARYONDIFF_DELTAF)

        //add them up
        //this is the mean number of particles of species ipart
        double dN = dN_thermal + dN_bulk + dN_diff;

        //save to a list to later sample inidividual species
        density_list[ipart] = dN;

        //add this to the total mean number of hadrons for the FO cell
        dN_tot += dN;

    }

    //now sample the number of particles (all species)

    //FIX
    //move these declarations to beginning of function !
    //OR does each thread need its own ???

    //non-deterministic random number generator
    std::random_device gen1;
    std::random_device gen2;

    //FIX
    std::poisson_distribution<> poisson_distr(dN_tot);

    //sample total number of hadrons in FO cell
    int N_hadrons = poisson_distr(gen1);

    //make a discrete distribution with weights according to particle densities
    std::discrete_distribution<int> particle_numbers ( density_list.begin(), density_list.end() );

    for (int n = 0; n < N_hadrons; n++)
    {
      //a new particle
      Sampled_Particle new_particle;

      //sample discrete distribution
      int idx_sampled = particle_numbers(gen2); //this gives the index of the sampled particle

      //get the mass of the sampled particle
      double mass = Mass[idx_sampled];  // (GeV)
      double mass2 = mass * mass;

      //get the MC ID of the sampled particle
      int mcid = MCID[idx_sampled];
      //particle_list[particle_index].mcID = mcid;
      new_particle.mcID = mcid;

      //sample LRF Momentum with Scott Pratt's Trick - See LongGang's Sampler Notes
      lrf_momentum p_LRF = Sample_Momentum(mass, T, alphaB);
      double E_LRF = sqrt(mass2 + p_LRF.x * p_LRF.x + p_LRF.y * p_LRF.y + p_LRF.z * p_LRF.z);
      double px_LRF = p_LRF.x;
      double py_LRF = p_LRF.y;
      double pz_LRF = p_LRF.z;

      //boost the momenta from LRF to lab frame
      //lorentz boost formula from Jackson (11.19) from LRF to Lab
      double betadotp = betax * px_LRF + betay * py_LRF + betaz * pz_LRF;
      if ( ::isnan(betadotp) ) printf("found betadotp nan!\n");

      //momentum in Lab frame
      double px = px_LRF + (u0 - 1.0) / beta2 * betadotp * betax - u0 * betax * E_LRF;
      double py = py_LRF + (u0 - 1.0) / beta2 * betadotp * betay - u0 * betay * E_LRF;
      double pz = pz_LRF + (u0 - 1.0) / beta2 * betadotp * betaz - u0 * betaz * E_LRF;

      //set energy using on-shell condition
      double E = sqrt(mass2 + px * px + py * py + pz * pz);

      //set coordinates of production to FO cell coords
      //particle_list[particle_index].tau = tau;
      //particle_list[particle_index].x = x;
      //particle_list[particle_index].y = y;
      //particle_list[particle_index].eta = eta;

      new_particle.tau = tau;
      new_particle.x = x;
      new_particle.y = y;
      new_particle.eta = eta;

      //set particle momentum to sampled values
      //particle_list[particle_index].E = E;
      //particle_list[particle_index].px = px;
      //particle_list[particle_index].py = py;
      //particle_list[particle_index].pz = pz;

      new_particle.E = E;
      new_particle.px = px;
      new_particle.py = py;
      new_particle.pz = pz;

      //particle_index += 1;

      //add to particle list
      particle_list.push_back(new_particle);

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
  printf("NOTHING HERE YET \n");
}
