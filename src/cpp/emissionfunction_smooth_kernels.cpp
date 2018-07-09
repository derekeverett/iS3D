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

void EmissionFunctionArray::calculate_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff)

  {
    double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
    int FO_chunk = 10000;

    double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
    for (int j = 0; j < phi_tab_length; j++)
    {
      double phi = phi_tab->get(1,j+1);
      trig_phi_table[j][0] = cos(phi);
      trig_phi_table[j][1] = sin(phi);
    }

    //fill arrays with values points in pT
    double pTValues[pT_tab_length];
    for (int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

    // fill arrays with values points in y and eta
    int y_pts = y_tab_length;    // defaults pts for 3+1d
    int eta_pts = 1;

    if(DIMENSION == 2)
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts];    // eta_weight * delta_eta

    double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
        //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
      for (int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);
    }


    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      printf("Progress : Finished chunk %d of %ld \n", n, FO_length / FO_chunk);
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
      #pragma omp parallel for
      #pragma acc kernels
      //#pragma acc loop independent
      for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell;     // global FO cell index

        // set freezeout info to local varibles to reduce(?) memory access outside cache :
        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double tau2 = tau * tau;

        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
        }
        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ux = ux_fo[icell_glb];           // contravariant fluid velocity
        double uy = uy_fo[icell_glb];           // reinforce normalization
        double un = un_fo[icell_glb];
        double ut = sqrt(fabs(1.0 + ux*ux + uy*uy + tau2*un*un));

        double T = T_fo[icell_glb];             // temperature
        double E = E_fo[icell_glb];             // energy density
        double P = P_fo[icell_glb];             // pressure

        double pitt = pitt_fo[icell_glb];       // pi^munu
        double pitx = pitx_fo[icell_glb];
        double pity = pity_fo[icell_glb];
        double pitn = pitn_fo[icell_glb];
        double pixx = pixx_fo[icell_glb];
        double pixy = pixy_fo[icell_glb];
        double pixn = pixn_fo[icell_glb];
        double piyy = piyy_fo[icell_glb];
        double piyn = piyn_fo[icell_glb];
        double pinn = pinn_fo[icell_glb];

        double bulkPi = bulkPi_fo[icell_glb];   // bulk pressure

        double muB = 0.0;                       // baryon chemical potential
        double nB = 0.0;                        // net baryon density
        double Vt = 0.0;                        // baryon diffusion
        double Vx = 0.0;
        double Vy = 0.0;
        double Vn = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell_glb];

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_fo[icell_glb];
            Vt = Vt_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
          }
        }

        double shear_coeff = 0.5 / (T * T * (E + P));  // (see df_shear)

        //now loop over all particle species and momenta
        for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          // set particle properties
          double mass = Mass[ipart];    // (GeV)
          double mass2 = mass * mass;
          double sign = Sign[ipart];
          double degeneracy = Degeneracy[ipart];
          double baryon = 0.0;
          double chem = 0.0;           // chemical potential term in feq
          if(INCLUDE_BARYON)
          {
            baryon = Baryon[ipart];
            chem = baryon * muB;
          }

          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            // set transverse radial momentum and transverse mass (GeV)
            double pT = pTValues[ipT];
            double mT = sqrt(mass2 + pT * pT);
            // useful expression
            double mT_over_tau = mT / tau;

            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double px = pT * trig_phi_table[iphip][0]; //contravariant
              double py = pT * trig_phi_table[iphip][1]; //contravariant

              for (int iy = 0; iy < y_pts; iy++)
              {
                //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                double y = yValues[iy];

                double pdotdsigma_f_eta_sum = 0.0;

                // sum over eta
                for (int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];
                  double delta_eta_weight = etaDeltaWeights[ieta];

                  double pt = mT * cosh(y - eta); // contravariant
                  double pn = mT_over_tau * sinh(y - eta); // contravariant
                  // useful expression
                  double tau2_pn = tau2 * pn;

                  //momentum vector is contravariant, surface normal vector is COVARIANT
                  double pdotdsigma = pt * dat + px * dax + py * day + pn * dan;

                  //thermal equilibrium distributions - for viscous hydro
                  double pdotu = pt * ut - px * ux - py * uy - tau2_pn * un;    // u.p = LRF energy
                  double feq = 1.0 / ( exp(( pdotu - chem ) / T) + sign);

                  //viscous corrections
                  double feqbar = 1.0 - sign*feq;

                  // shear correction:
                  double df_shear = 0.0;
                  if (INCLUDE_SHEAR_DELTAF)
                  {
                    // pi^munu * p_mu * p_nu
                    double pimunu_pmu_pnu = pitt * pt * pt + pixx * px * px + piyy * py * py + pinn * tau2_pn * tau2_pn
                    + 2.0 * (-(pitx * px + pity * py) * pt + pixy * px * py + tau2_pn * (pixn * px + piyn * py - pitn * pt));

                    df_shear = shear_coeff * pimunu_pmu_pnu;  // df / (feq*feqbar)
                  }

                  // bulk correction:
                  double df_bulk = 0.0;
                  if (INCLUDE_BULK_DELTAF)
                  {
                    double c0 = df_coeff[0];
                    double c2 = df_coeff[2];
                    if (INCLUDE_BARYON)
                    {
                      double c1 = df_coeff[1];
                      df_bulk = ((c0-c2)*mass2 + c1 * baryon * pdotu + (4.0*c2-c0)*pdotu*pdotu) * bulkPi;
                    }
                    else
                    {
                      df_bulk = ((c0-c2)*mass2 + (4.0*c2-c0)*pdotu*pdotu) * bulkPi;
                    }
                  }

                  // baryon diffusion correction:
                  double df_baryondiff = 0.0;
                  if (INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                  {
                    double c3 = df_coeff[3];
                    double c4 = df_coeff[4];
                    // V^mu * p_mu
                    double Vmu_pmu = Vt * pt - Vx * px - Vy * py - Vn * tau2_pn;
                    df_baryondiff = (baryon * c3 + c4 * pdotu) * Vmu_pmu;
                  }
                  double df = df_shear + df_bulk + df_baryondiff;


                  if (REGULATE_DELTAF)
                  {
                    double reg_df = max( -1.0, min( feqbar * df, 1.0 ) );
                    pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * feq * (1.0 + reg_df));
                  }
                  else pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * feq * (1.0 + df));

                } // ieta

                long long int iSpectra = icell + endFO * (ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy)));

                dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);

              } //iy
            } //iphip
          } //ipT
        } //ipart
      } //icell
      if(endFO != 0)
      {
        //now perform the reduction over cells
        #pragma omp parallel for collapse(4)
        #pragma acc kernels
        for (int ipart = 0; ipart < npart; ipart++)
        {
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              for (int iy = 0; iy < y_pts; iy++)
              {
                //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
                long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
                double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  long long int iSpectra = icell + endFO * iS3D;
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                }//icell
                dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; //sum over all chunks
              }//iy
            }//iphip
          }//ipT
        }//ipart species
      } //if (endFO != 0 )
    }//n FO chunk

    //free memory
    free(dN_pTdpTdphidy_all);
  }

  // haven't defined in the header yet...
  void EmissionFunctionArray::calculate_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
    double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
    double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
    double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
    {
      double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
      int FO_chunk = 10000;

      double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
      for (int j = 0; j < phi_tab_length; j++)
      {
        double phi = phi_tab->get(1,j+1);
        trig_phi_table[j][0] = cos(phi);
        trig_phi_table[j][1] = sin(phi);
      }

      //fill arrays with values points in pT and y
      double pTValues[pT_tab_length];
      for (int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

      // fill arrays with values points in y and eta
      int y_pts = y_tab_length;    // defaults pts for 3+1d
      int eta_pts = 1;

      if(DIMENSION == 2)
      {
        y_pts = 1;
        eta_pts = eta_tab_length;
      }

      double yValues[y_pts];
      double etaValues[eta_pts];
      double etaDeltaWeights[eta_pts];    // eta_weight * delta_eta

      double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

      if(DIMENSION == 2)
      {
        yValues[0] = 0.0;
        for(int ieta = 0; ieta < eta_pts; ieta++)
        {
          etaValues[ieta] = eta_tab->get(1, ieta + 1);
          etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
          //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
        }
      }
      else if(DIMENSION == 3)
      {
        etaValues[0] = 0.0;       // below, will load eta_fo
        etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
        for (int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);
      }


      //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
      //to hold the spectra for each surface cell in a chunk, for all particle species
      int npart = number_of_chosen_particles;
      double *dN_pTdpTdphidy_all;
      dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

      //loop over bite size chunks of FO surface
      for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
      {
        printf("Progress : Finished chunk %d of %ld \n", n, FO_length / FO_chunk);
        int endFO = FO_chunk;
        if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); // don't go out of array bounds
        #pragma omp parallel for
        #pragma acc kernels
        //#pragma acc loop independent
        for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
        {
          int icell_glb = n * FO_chunk + icell; // global FO cell index

          // set freezeout info to local varibles
          double tau = tau_fo[icell_glb];         // longitudinal proper time
          double tau2 = tau * tau;                // useful expression
          if(DIMENSION == 3)
          {
            etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
          }
          double dat = dat_fo[icell_glb];         // covariant normal surface vector
          double dax = dax_fo[icell_glb];
          double day = day_fo[icell_glb];
          double dan = dan_fo[icell_glb];

          double T = T_fo[icell_glb];

          double ux = ux_fo[icell_glb];           // contravariant fluid velocity
          double uy = uy_fo[icell_glb];           // enforce normalization: u.u = 1
          double un = un_fo[icell_glb];
          double ut = sqrt(1.0 + ux*ux + uy*uy + tau2*un*un);

          double u0 = sqrt(1.0 + ux*ux + uy*uy);
          double zt = tau * un / u0;
          double zn = ut / (u0 * tau);            // z basis vector

          double pitt = pitt_fo[icell_glb];       // piperp^munu
          double pitx = pitx_fo[icell_glb];
          double pity = pity_fo[icell_glb];
          double pitn = pitn_fo[icell_glb];
          double pixx = pixx_fo[icell_glb];
          double pixy = pixy_fo[icell_glb];
          double pixn = pixn_fo[icell_glb];
          double piyy = piyy_fo[icell_glb];
          double piyn = piyn_fo[icell_glb];
          double pinn = pinn_fo[icell_glb];

          double bulkPi = bulkPi_fo[icell_glb];   // residual bulk pressure

          double Wx = Wx_fo[icell_glb];           // W^mu
          double Wy = Wy_fo[icell_glb];           // reinforce orthogonality
          double Wt = (ux * Wx  +  uy * Wy) * ut / (u0 * u0);
          double Wn = Wt * un / ut;

          double Lambda = Lambda_fo[icell_glb];   // anisotropic variables
          double aL = aL_fo[icell_glb];

          double c0 = c0_fo[icell_glb];           // 14-moment coefficients
          double c1 = c1_fo[icell_glb];
          double c2 = c2_fo[icell_glb];
          double c3 = c3_fo[icell_glb];
          double c4 = c4_fo[icell_glb];


          //now loop over all particle species and momenta
          for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
          {
            // set particle properties
            double mass = Mass[ipart];    // (GeV)
            double mass2 = mass * mass;
            double sign = Sign[ipart];
            double degeneracy = Degeneracy[ipart];


            for (int ipT = 0; ipT < pT_tab_length; ipT++)
            {
              // set transverse radial momentum and transverse mass (GeV)
              double pT = pTValues[ipT];
              double mT = sqrt(mass2 + pT * pT);
              // useful expression
              double mT_over_tau = mT / tau;

              for (int iphip = 0; iphip < phi_tab_length; iphip++)
              {
                double px = pT * trig_phi_table[iphip][0]; //contravariant
                double py = pT * trig_phi_table[iphip][1]; //contravariant

                for (int iy = 0; iy < y_pts; iy++)
                {
                  // all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                  double y = yValues[iy];

                  double pdotdsigma_f_eta_sum = 0.0;

                  // sum over eta
                  for (int ieta = 0; ieta < eta_pts; ieta++)
                  {
                    double eta = etaValues[ieta];
                    double delta_eta_weight = etaDeltaWeights[ieta];

                    double pt = mT * cosh(y - eta);
                    double pn = mT_over_tau * sinh(y - eta);

                    // useful expression
                    double tau2_pn = tau2 * pn;

                    // momentum vector is contravariant, surface normal vector is COVARIANT
                    double pdotdsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;

                    //thermal equilibrium distributions - for viscous hydro
                    double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;    // u.p = LRF energy
                    double pdotz = pt * zt  -  tau2_pn * zn;                            // z.p = - LRF longitudinal momentum

                    double xiL = 1.0 / (aL * aL)  -  1.0;

                    #ifdef FORCE_F0
                    Lambda = T;
                    xiL = 0.0;
                    #endif

                    double Ea = sqrt(pdotu * pdotu  +  xiL * pdotz * pdotz);

                    double fa = 1.0 / ( exp(Ea / Lambda) + sign);

                    // residual viscous corrections
                    double fabar = 1.0 - sign*fa;

                    // residual shear correction:
                    double df_shear = 0.0;

                    if (INCLUDE_SHEAR_DELTAF)
                    {
                      // - (-z.p)p_{mu}.W^mu
                      double Wmu_pmu_pz = pdotz * (Wt * pt  -  Wx * px  -  Wy * py  -  Wn * tau2_pn);

                      // p_{mu.p_nu}.piperp^munu
                      double pimunu_pmu_pnu = pitt * pt * pt + pixx * px * px + piyy * py * py + pinn * tau2_pn * tau2_pn
                      + 2.0 * (-(pitx * px + pity * py) * pt + pixy * px * py + tau2_pn * (pixn * px + piyn * py - pitn * pt));

                      df_shear = c3 * Wmu_pmu_pz  +  c4 * pimunu_pmu_pnu;  // df / (feq*feqbar)
                    }

                    // residual bulk correction:
                    double df_bulk = 0.0;
                    if (INCLUDE_BULK_DELTAF) df_bulk = (c0 * mass2  +  c1 * pdotz * pdotz  +  c2 * pdotu * pdotu) * bulkPi;

                    double df = df_shear + df_bulk;

                    if (REGULATE_DELTAF)
                    {
                      double reg_df = max( -1.0, min( fabar * df, 1.0 ) );
                      pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * fa * (1.0 + reg_df));
                    }
                    else pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * fa * (1.0 + fabar * df));

                  } // ieta

                  long long int iSpectra = icell + endFO * (ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy)));

                  dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);
                } //iy
              } //iphip
            } //ipT
          } //ipart
        } //icell
        if(endFO != 0)
        {
          //now perform the reduction over cells
          #pragma omp parallel for collapse(4)
          #pragma acc kernels
          for (int ipart = 0; ipart < npart; ipart++)
          {
            for (int ipT = 0; ipT < pT_tab_length; ipT++)
            {
              for (int iphip = 0; iphip < phi_tab_length; iphip++)
              {
                for (int iy = 0; iy < y_pts; iy++)
                {
                  //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
                  long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
                  double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                  #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                  for (int icell = 0; icell < endFO; icell++)
                  {
                    //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                    long long int iSpectra = icell + endFO * iS3D;
                    dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                  }//icell
                  dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; //sum over all chunks
                }//iy
              }//iphip
            }//ipT
          }//ipart species
        } //if (endFO != 0 )
      }//n FO chunk

      //free memory
      free(dN_pTdpTdphidy_all);
    }
