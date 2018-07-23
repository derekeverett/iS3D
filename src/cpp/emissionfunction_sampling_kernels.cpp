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

  // Derek: can we fix the seed temporarily we can compare results?

  // only need to seed once right?
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);

  // light hadrons

  //TO DO
  //add routine that works for pions!
  // if (mass / T < 2.0)
  //stuff

  // the ARS thing should be worked on after this 
  // where does in the SPREW code is it employed? 


  // I should I put in the r_ideal or r_visc accept/reject conditions
  // after I drawed all the momentum variables (p,phi,costheta) (build it up tonight; at least for ideal hydro)
  // note: don't assume that r_ideal/visc is independent of p like LongGong does 


  // light hadrons, but heavier than pions
  if(T / mass >= 0.6) // fixed bug on 7/12
  {
    // I should wrap the r_ideal/visc while loop around this stuff here: {}
    bool rejected = true;
    while(rejected)
    {
      // do you need different generators for (r1,r2,r3,propose)? Guess not..
      // draw (p, phi, costheta) from distribution p^2 * exp[-p/T] by
      // sampling three variables (r1,r2,r3) uniformly from [0,1)
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

      // check p acceptance
      if(propose < weight)
      {
        rejected = false; // stop while loop

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
      } // p acceptance
    } // while loop
  } // mass / T < 0.6

  //heavy hadrons
  //use variable transformation described in LongGang's Sampler Notes
  else
  {
    // determine which part of integrand dominates
    double I1 = mass * mass;
    double I2 = 2.0 * mass * T;
    double I3 = 2.0 * T * T;

    double Itot = I1 + I2 + I3;

    double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

    // accept-reject distributions to sample momentum from based on integrated weights
    if(propose_distribution < I1 / Itot)
    {
      // draw k from distribution exp(-k/T) by sampling
      // one variable r1 uniformly from [0,1):
      //    k = -T * log(r1)

      bool rejected = true;
      while(rejected)
      {
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double k = - T * log(r1);
        if(::isnan(k)) printf("found k nan!\n");

        double E = k + mass;
        double p = sqrt(fabs(E * E - mass * mass));
        double weight = p / E;
        double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

        // check k acceptance
        if(propose < weight)
        {
          rejected = false; // stop while loop

          // sample LRF angles costheta = [-1,1] and phi = [0,2pi) uniformly
          uniform_real_distribution<double> phi_distribution(0.0 , 2.0 * M_PI);
          uniform_real_distribution<double> costheta_distribution(-1.0 , nextafter(1.0, numeric_limits<double>::max()));
          double phi = phi_distribution(generator);
          double costheta = costheta_distribution(generator);
          double sintheta = sqrt(1.0 - costheta * costheta);
          if(::isnan(sintheta)) printf("found sintheta nan!\n");

          // evaluate LRF momentum components
          pLRF.x = p * costheta;
          pLRF.y = p * sintheta * cos(phi);
          pLRF.z = p * sintheta * sin(phi);
        } // k acceptance
      } // while loop
    } // distribution 1
    else if(propose_distribution < (I1 + I2) / Itot)
    {
      // draw (k,phi) from distribution k * exp(-k/T) by
      // sampling two variables (r1,r2) uniformly from [0,1):
      //    k = -T * log(r1)
      //    phi = 2 * \pi * log(r1) / (log(r1) + log(r2)); (double check this formula!!)

      // I could have also done costheta instead of phi (I guess it doesn't matter)

      bool rejected = true;
      while(rejected)
      {
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        double k = - T * (l1 + l2);
        if(::isnan(k)) printf("found k nan!\n");

        double E = k + mass;
        double p = sqrt(fabs(E * E - mass * mass));
        double weight = p / E;
        double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

        // check k acceptance
        if(propose < weight)
        {
          rejected = false; // stop while loop

          // need to verify this formula
          double phi = 2.0 * M_PI * l1 / (l1 + l2);
          if(::isnan(phi)) printf("found phi nan!\n");

          // sample LRF angle costheta = [-1,1] uniformly
          uniform_real_distribution<double> costheta_distribution(-1.0 , nextafter(1.0, numeric_limits<double>::max()));
          double costheta = costheta_distribution(generator);
          double sintheta = sqrt(1.0 - costheta * costheta);
          if(::isnan(sintheta)) printf("found sintheta nan!\n");

          // evaluate momentum components
          pLRF.x = p * costheta;
          pLRF.y = p * sintheta * cos(phi);
          pLRF.z = p * sintheta * sin(phi);
        } // k acceptance
      } // while loop
    } // distribution 2
    else
    {
      // draw (k,phi,costheta) from k^2 * exp(-k/T) distribution by
      // sampling three variables (r1,r2,r3) uniformly from [0,1):
      //    k = - T * (log(r1) + log(r2) + log(r3))
      //    phi = 2 * \pi * (log(r1) + log(r2))^2 / (log(r1) + log(r2) + log(r3))^2
      //    costheta = (log(r1) - log(r2)) / (log(r1) + log(r2))

      bool rejected = true;
      while(rejected)
      {
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        double k = - T * (l1 + l2 + l3);
        if(::isnan(k)) printf("found k nan!\n");

        double E = k + mass;
        double p = sqrt(fabs(E * E - mass * mass));
        double weight = p / E;
        double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

        if(propose < weight)
        {
          rejected = false;

          // calculate angles
          double costheta = (l1 - l2) / (l1 + l2);
          if(::isnan(costheta)) printf("found costheta nan!\n");
          double phi = 2.0 * M_PI * pow(l1 + l2, 2) / pow(l1 + l2 + l3, 2);
          if(::isnan(phi)) printf("found phi nan!\n");
          double sintheta = sqrt(1.0 - costheta * costheta);
          if(::isnan(sintheta)) printf("found sintheta nan!\n");

          // evaluate momentum components
          pLRF.x = p * costheta;
          pLRF.y = p * sintheta * cos(phi);
          pLRF.z = p * sintheta * sin(phi);
        } // k acceptance
      }
    } // distribution 3
  } // heavy hadron

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
      //double eta = etaValues[0];

      //double coshn = cosh(eta);
      //double sinhn = sinh(eta);
      //FIX THIS

      double dat = dat_fo[icell];         // covariant normal surface vector
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];

      double ux = ux_fo[icell];           // contravariant fluid velocity
      double uy = uy_fo[icell];           // reinforce normalization
      double un = un_fo[icell];           // u^\eta
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux2 + uy2);
      double ut = sqrt(fabs(1.0 + uperp * uperp + tau2*un*un)); //u^\tau

      double ut2 = ut * ut;
      double utperp = sqrt(1.0 + uperp * uperp);

      //get the cartesian components of fluid velocity for the boost LRF -> Lab
      //double u0 = ut * cosh(un);
      //double u3 = ut * sinh(un);

      //beta factors for boost from LRF to Lab
      //double betax = ux/u0;
      //double betay = uy/u0;
      //double betaz = u3/u0;
      //double beta2 = betax * betax + betay * betay + betaz * betaz;

      double T = T_fo[icell];             // temperature
      double E = E_fo[icell];             // energy density
      double P = P_fo[icell];             // pressure

      //double pitt = pitt_fo[icell];     // pi^munu
      //double pitx = pitx_fo[icell];     // enforce orthogonality and tracelessness
      //double pity = pity_fo[icell];
      //double pitn = pitn_fo[icell];
      double pixx = pixx_fo[icell];
      double pixy = pixy_fo[icell];
      double pixn = pixn_fo[icell];
      double piyy = piyy_fo[icell];
      double piyn = piyn_fo[icell];
      //double pinn = pinn_fo[icell];
      double pinn = (pixx*(ux2 - ut2) + piyy*(uy2 - ut2) + 2.0*(pixy*ux*uy + tau2*un*(pixn*ux + piyn*uy))) / (tau2 * utperp * utperp);
      double pitn = (pixn*ux + piyn*uy + tau2*pinn*un) / ut;
      double pity = (pixy*ux + piyy*uy + tau2*piyn*un) / ut;
      double pitx = (pixx*ux + pixy*uy + tau2*pixn*un) / ut;
      double pitt = (pitx*ux + pity*uy + tau2*pitn*un) / ut;

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
          // Vt = Vt_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
          Vt = (Vx*ux + Vy*uy + tau2*Vn*un) / ut;
          baryon_enthalpy_ratio = nB / (E + P);
        }
      }

      // set milne basis vectors:
      double Xt, Xx, Xy, Xn;  // X^mu
      double Yx, Yy;          // Y^mu
      double Zt, Zn;          // Z^mu

      double sinhL = tau * un / utperp;
      double coshL = ut / utperp;

      Xt = uperp * coshL;
      Xn = uperp * sinhL / tau;
      Zt = sinhL;
      Zn = coshL / tau;

      // stops (ux=0)/(uperp=0) nans
      if(uperp < 1.e-5)
      {
        Xx = 1.0; Xy = 0.0;
        Yx = 0.0; Yy = 1.0;
      }
      else
      {
        Xx = utperp * ux / uperp;
        Xy = utperp * uy / uperp;
        Yx = - uy / uperp;
        Yy = ux / uperp;
      }


      // loop over eta points [table for 2+1d (eta_trapezoid_table_21pt.dat), no table for 3+1d (eta_pts = 1)]
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta = etaValues[ieta];
        double delta_eta_weight = etaDeltaWeights[ieta];

        double cosheta = cosh(eta);
        double sinheta = sinh(eta);

        double udotdsigma = delta_eta_weight * (dat*ut + dax*ux + day*uy + dan*un);

        double Vdotdsigma = 0.0;

        // V^\mu d^3sigma_\mu
        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          Vdotdsigma = delta_eta_weight * (dat*Vt + dax*Vx + day*Vy + dan*Vn);
        }
        // evaluate particle densities:

        //the total mean number of particles (all species) for the FO cell
        double dN_tot = 0.0;

        //a list with species dependent densities
        std::vector<double> density_list;
        density_list.resize(npart);

        //loop over all species
        for(int ipart = 0; ipart < npart; ipart++)
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
            chem = baryon * alphaB;
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
            neq += (sign_factor * exp(k * chem) * gsl_sf_bessel_Kn(2, k * mbar) / k); // fixed chem term 7/15
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

          // add them up
          // this is the mean number of particles of species ipart
          double dN = dN_thermal + dN_bulk + dN_diff;

          //save to a list to later sample inidividual species
          density_list[ipart] = dN;

          //add this to the total mean number of hadrons for the FO cell
          dN_tot += dN;

        } // for(int ipart = 0; ipart < npart; ipart++))


        // now sample the number of particles (all species)

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
        std::discrete_distribution<int> particle_numbers (density_list.begin(), density_list.end());

        for(int n = 0; n < N_hadrons; n++)
        {
          // sample discrete distribution
          int idx_sampled = particle_numbers(gen2); //this gives the index of the sampled particle

          // get the mass of the sampled particle
          double mass = Mass[idx_sampled];  // (GeV)
          double mass2 = mass * mass;

          // get the MC ID of the sampled particle
          int mcid = MCID[idx_sampled];
          
          // sample LRF Momentum with Scott Pratt's Trick - See LongGang's Sampler Notes
          // perhap divide this into sample_momentum_from_distribution_x()
          lrf_momentum p_LRF = Sample_Momentum(mass, T, alphaB);

          double px_LRF = p_LRF.x;
          double py_LRF = p_LRF.y;
          double pz_LRF = p_LRF.z;
          double E_LRF = sqrt(mass2 + px_LRF * px_LRF + py_LRF * py_LRF + pz_LRF * pz_LRF);

          // lab frame milne p^mu
          double ptau = E_LRF * ut + px_LRF * Xt + pz_LRF * Zt;
          double px = E_LRF * ux + px_LRF * Xx + py_LRF * Yx;
          double py = E_LRF * uy + px_LRF * Xy + py_LRF * Yy;
          double pn = E_LRF * un + px_LRF * Xn + pz_LRF * Zn;

          // lab frame cartesian p^mu
          // ptau = mT * cosh(y-eta) = E * coshn - pz * sinhn
          // tau * pn = mT * sinh(y-eta) = pz * coshn - E * sinhn;
          double E = (ptau * cosheta) + (tau * pn * sinheta);
          double pz = (tau * pn * cosheta) + (ptau * sinheta);

          //boost the momenta from LRF to lab frame
          //lorentz boost formula from Jackson (11.19) from LRF to Lab
          //double betadotp = betax * px_LRF + betay * py_LRF + betaz * pz_LRF;
          //if ( ::isnan(betadotp) ) printf("found betadotp nan!\n");

          //momentum in Lab frame
          //double px = px_LRF + (u0 - 1.0) / beta2 * betadotp * betax - u0 * betax * E_LRF;
          //double py = py_LRF + (u0 - 1.0) / beta2 * betadotp * betay - u0 * betay * E_LRF;
          //double pz = pz_LRF + (u0 - 1.0) / beta2 * betadotp * betaz - u0 * betaz * E_LRF;

          //set energy using on-shell condition
          //double E = sqrt(mass2 + px * px + py * py + pz * pz);

          //set coordinates of production to FO cell coords
          //particle_list[particle_index].tau = tau;
          //particle_list[particle_index].x = x;
          //particle_list[particle_index].y = y;
          //particle_list[particle_index].eta = eta;


          // keep p.dsigma > 0 and throw out p.dsigma < 0 sampled particles

          // milne coordinate formula
          // for case of boost invariance: delta_eta_weight can be factored out 
          // double pdotdsigma = delta_eta_weight * (ptau * dat + px * dax + py * pday + pn * dan);
          double pdotdsigma = ptau * dat + px * dax + py * day + pn * dan;

          // add sampled particle to particle_list 
          if(pdotdsigma >= 0.0)
          {
            // a new particle
            Sampled_Particle new_particle;

            //particle_list[particle_index].mcID = mcid;
            new_particle.mcID = mcid;

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
          } // if(pdotsigma >= 0)
        } // for (int n = 0; n < N_hadrons; n++)
      } // for(int ieta = 0; ieta < eta_ptsl ieta++)
  } // for (int icell = 0; icell < FO_length; icell++)
}





void EmissionFunctionArray::sample_dN_pTdpTdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2)
  {
    printf("sampling particles from vhydro with df...\n");

    if((MODE != 1) || (DF_MODE != 3))
    {
      cout << "Stopping: please set mode = 1 (viscous hydro) and df_mode = 3 (modified) in parameters.dat" << endl;
      exit(-1);
    }

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

    // set df coefficients:
    double F = df_coeff[0];
    double G = df_coeff[1];
    double betabulk = df_coeff[2];
    double betaV = df_coeff[3];
    double betapi = df_coeff[4];

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

      double dat = dat_fo[icell];         // covariant normal surface vector
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];

      double ux = ux_fo[icell];           // contravariant fluid velocity
      double uy = uy_fo[icell];           // reinforce normalization
      double un = un_fo[icell];           // u^\eta
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux2 + uy2);
      double ut = sqrt(fabs(1.0 + uperp * uperp + tau2*un*un)); //u^\tau

      double ut2 = ut * ut;
      double utperp = sqrt(1.0 + uperp * uperp);

      double T = T_fo[icell];             // temperature
      double T_mod = T;                   // modified temperature (default)
      double E = E_fo[icell];             // energy density
      double P = P_fo[icell];             // pressure

      //double pitt = pitt_fo[icell];     // pi^munu
      //double pitx = pitx_fo[icell];     // enforce orthogonality and tracelessness
      //double pity = pity_fo[icell];
      //double pitn = pitn_fo[icell];
      double pixx = pixx_fo[icell];
      double pixy = pixy_fo[icell];
      double pixn = pixn_fo[icell];
      double piyy = piyy_fo[icell];
      double piyn = piyn_fo[icell];
      //double pinn = pinn_fo[icell];
      double pinn = (pixx*(ux2 - ut2) + piyy*(uy2 - ut2) + 2.0*(pixy*ux*uy + tau2*un*(pixn*ux + piyn*uy))) / (tau2 * utperp * utperp);
      double pitn = (pixn*ux + piyn*uy + tau2*pinn*un) / ut;
      double pity = (pixy*ux + piyy*uy + tau2*piyn*un) / ut;
      double pitx = (pixx*ux + pixy*uy + tau2*pixn*un) / ut;
      double pitt = (pitx*ux + pity*uy + tau2*pitn*un) / ut;
      double bulkPi = bulkPi_fo[icell];   // bulk pressure

      double muB = 0.0;
      double alphaB = 0.0;                    // baryon chemical potential
      double alphaB_mod = alphaB;             // modified chemical potential (default)
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
          //Vt = Vt_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
          Vt = (Vx*ux + Vy*uy + tau2*Vn*un) / ut;
          baryon_enthalpy_ratio = nB / (E + P);
        }
      }

      // set modified coefficients, temperature, chemical potential
      double shear_coeff = 0.5 / betapi;
      double bulk_coeff = bulkPi / betabulk / 3.0;
      double diff_coeff = 0.0;
      if(INCLUDE_BULK_DELTAF)
      {
        double dT = F * bulkPi / betabulk;
        double dalphaB = G * bulkPi / betabulk;
        T_mod += dT;
        alphaB_mod += dalphaB;
      }
      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        diff_coeff = T / betaV;
      }

      // set milne basis vectors:
      double Xt, Xx, Xy, Xn;  // X^mu
      double Yx, Yy;          // Y^mu
      double Zt, Zn;          // Z^mu

      double sinhL = tau * un / utperp;
      double coshL = ut / utperp;

      Xt = uperp * coshL;
      Xn = uperp * sinhL / tau;
      Zt = sinhL;
      Zn = coshL / tau;

      // stops (ux=0)/(uperp=0) nans
      if(uperp < 1.e-5)
      {
        Xx = 1.0; Xy = 0.0;
        Yx = 0.0; Yy = 1.0;
      }
      else
      {
        Xx = utperp * ux / uperp;
        Xy = utperp * uy / uperp;
        Yx = - uy / uperp;
        Yy = ux / uperp;
      }

      // set LRF shear stress and diffusion components
      double pixx_LRF = 0.0;
      double pixy_LRF = 0.0;
      double pixz_LRF = 0.0;
      double piyy_LRF = 0.0;
      double piyz_LRF = 0.0;
      double pizz_LRF = 0.0;

      if(INCLUDE_SHEAR_DELTAF)
      {
          // pimunu in the LRF: piij = Xi.pi.Xj
          //double tau4 = tau2 * tau2;
          pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
                  + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));
          pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xy + pixy*Xx + piyy*Xy + tau2*piyn*Xn);
          pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xn - piyn*Xy - tau2*pinn*Xn);
          piyy_LRF = pixx*Yx*Yx + piyy*Yy*Yy + 2.0*pixy*Yx*Yy;
          piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);
          pizz_LRF = - (pixx_LRF + piyy_LRF);
      }

      double Vx_LRF = 0.0;
      double Vy_LRF = 0.0;
      double Vz_LRF = 0.0;

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        // Vmu in the LRF: Vi = - Xi.V
        Vx_LRF = - Vt*Xt + Vx*Xx + Vy*Xy + tau2*Vn*Xn;
        Vy_LRF = Vx*Yx + Vy*Yy;
        Vz_LRF = - Vt*Zt + tau2*Vn*Zn;
      }

      // loop over eta points [table for 2+1d (eta_trapezoid_table_21pt.dat), no table for 3+1d (eta_pts = 1)]
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta = etaValues[ieta];
        double delta_eta_weight = etaDeltaWeights[ieta];

        double cosheta = cosh(eta);
        double sinheta = sinh(eta);

        double udotdsigma = delta_eta_weight * (dat*ut + dax*ux + day*uy + dan*un);

        double Vdotdsigma = 0.0;

        // V^\mu d^3sigma_\mu
        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          Vdotdsigma = delta_eta_weight * (dat*Vt + dax*Vx + day*Vy + dan*Vn);
        }
        // evaluate particle densities:

        //the total mean number of particles (all species) for the FO cell
        double dN_tot = 0.0;

        //a list with species dependent densities
        std::vector<double> density_list;
        density_list.resize(npart);

        //loop over all species
        for(int ipart = 0; ipart < npart; ipart++)
        {
          // set particle properties
          double mass = Mass[ipart];   // (GeV)
          double mass2 = mass * mass;
          double mbar = mass / T;
          double sign = Sign[ipart];
          double degeneracy = Degeneracy[ipart];
          double baryon = 0.0;
          double chem = 0.0;           // chemical potential term in feq
          if(INCLUDE_BARYON)
          {
            baryon = Baryon[ipart];
            chem = baryon * alphaB;
            // how does result change exactly if the sampling has finite chemical potential?
            // include more terms in the expansion?
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
            neq += (sign_factor * exp(k * chem) * gsl_sf_bessel_Kn(2, k * mbar) / k); // fixed on 7/15
          }
          neq *= (degeneracy * mass2 * T / two_pi2_hbarC3);

          dN_thermal = udotdsigma * neq;

          // bulk pressure: density correction
          // (modified bulk correction exactly same as linear one because of renormalization factor)
          double dN_bulk = 0.0;
          if(INCLUDE_BULK_DELTAF)
          {
            double J10_fact = pow(T,3) / two_pi2_hbarC3;
            double J20_fact = pow(T,4) / two_pi2_hbarC3;
            double J10 = degeneracy * J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
            double J20 = degeneracy * J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
            dN_bulk = udotdsigma * (bulkPi / betabulk) * (neq + (baryon * J10 * G) + (J20 * F / T / T));
          }

          // baryon diffusion: density correction
          // (~ assumes modified diffusion current of each species is approximately same as linear one)
          double dN_diff = 0.0;
          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            double J11_fact = pow(T,3) / two_pi2_hbarC3 / 3.0;
            double J11 = degeneracy * J11_fact * GaussThermal(J11_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
            dN_diff = - (Vdotdsigma / betaV) * (neq * T * baryon_enthalpy_ratio - baryon * J11);
          }

          // add them up
          // this is the mean number of particles of species ipart
          double dN = dN_thermal + dN_bulk + dN_diff;

          //save to a list to later sample inidividual species
          density_list[ipart] = dN;

          //add this to the total mean number of hadrons for the FO cell
          dN_tot += dN;

        } // for(int ipart = 0; ipart < npart; ipart++))


        // now sample the number of particles (all species)

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
        std::discrete_distribution<int> particle_numbers (density_list.begin(), density_list.end());

        for(int n = 0; n < N_hadrons; n++)
        {
          //sample discrete distribution
          int idx_sampled = particle_numbers(gen2); // this gives the index of the sampled particle

          //get the mass of the sampled particle
          double mass = Mass[idx_sampled]; // (GeV)
          double mass2 = mass * mass;

          //get the MC ID of the sampled particle
          int mcid = MCID[idx_sampled];
          //particle_list[particle_index].mcID = mcid;

          // sample prime LRF Momentum p' from an "equilibrium" distribution
          // with Scott Pratt's Trick - See LongGang's Sampler Notes

          // modified version:
          // should not invoke the linear df corrections (make Sample_Momentum a class member)
          lrf_momentum p_LRF_prime = Sample_Momentum(mass, T_mod, alphaB_mod);

          double px_LRF_prime = p_LRF_prime.x;
          double py_LRF_prime = p_LRF_prime.y;
          double pz_LRF_prime = p_LRF_prime.z;
          double E_LRF_prime = sqrt(mass2 + px_LRF_prime * px_LRF_prime + py_LRF_prime * py_LRF_prime + pz_LRF_prime * pz_LRF_prime);

          // default LRF momentum (ideal)
          double px_LRF = px_LRF_prime;
          double py_LRF = py_LRF_prime;
          double pz_LRF = pz_LRF_prime;

          // rescale the momentum with matrix Mij
          if(INCLUDE_SHEAR_DELTAF)
          {
            // add shear corrections
            px_LRF += (shear_coeff * (pixx_LRF * px_LRF_prime + pixy_LRF * py_LRF_prime + pixz_LRF * pz_LRF_prime));
            px_LRF += (shear_coeff * (pixy_LRF * px_LRF_prime + piyy_LRF * py_LRF_prime + piyz_LRF * pz_LRF_prime));
            pz_LRF += (shear_coeff * (pixz_LRF * px_LRF_prime + piyz_LRF * py_LRF_prime + pizz_LRF * pz_LRF_prime));
          }
          if(INCLUDE_BULK_DELTAF)
          {
            // add bulk corrections
            px_LRF += (bulk_coeff * px_LRF_prime);
            py_LRF += (bulk_coeff * py_LRF_prime);
            pz_LRF += (bulk_coeff * pz_LRF_prime);
          }
          if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
          {
            // add diffusion corrections
            double baryon = Baryon[idx_sampled];

            px_LRF = (diff_coeff * Vx_LRF * (E_LRF_prime * baryon_enthalpy_ratio + baryon));
            py_LRF = (diff_coeff * Vy_LRF * (E_LRF_prime * baryon_enthalpy_ratio + baryon));
            pz_LRF = (diff_coeff * Vz_LRF * (E_LRF_prime * baryon_enthalpy_ratio + baryon));
          }

          // final LRF energy
          double E_LRF = sqrt(mass2 + px_LRF * px_LRF + py_LRF * py_LRF + pz_LRF * pz_LRF);


          // lab frame milne p^mu
          double ptau = E_LRF * ut + px_LRF * Xt + pz_LRF * Zt;
          double px = E_LRF * ux + px_LRF * Xx + py_LRF * Yx;
          double py = E_LRF * uy + px_LRF * Xy + py_LRF * Yy;
          double pn = E_LRF * un + px_LRF * Xn + pz_LRF * Zn;

          // lab frame cartesian p^mu
          // ptau = mT * cosh(y-eta) = E * coshn - pz * sinhn
          // tau * pn = mT * sinh(y-eta) = pz * coshn - E * sinhn;
          double E = (ptau * cosheta) + (tau * pn * sinheta);
          double pz = (tau * pn * cosheta) + (ptau * sinheta);


          // keep p.dsigma > 0 sampled particles
          // throw out p.dsigma < 0 sampled particles

          // milne coordinate formula
          // for case of boost invariance: delta_eta_weight can be factored out 
          // double pdotdsigma = delta_eta_weight * (ptau * dat + px * dax + py * pday + pn * dan);
          double pdotdsigma = ptau * dat + px * dax + py * day + pn * dan;

          // add sampled particle to particle_list 
          if(pdotdsigma >= 0.0)
          {
            // a new particle
            Sampled_Particle new_particle;

            //particle_list[particle_index].mcID = mcid;
            new_particle.mcID = mcid;

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
          } // if(pdotsigma >= 0)

        } // for (int n = 0; n < N_hadrons; n++)
      } // for(int ieta = 0; ieta < eta_ptsl ieta++)
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
