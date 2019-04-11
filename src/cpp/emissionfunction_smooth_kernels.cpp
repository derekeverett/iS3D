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
#include "iS3D.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "Stopwatch.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include <gsl/gsl_linalg.h>
#include "gaussThermal.h"
#include "particle.h"

using namespace std;

void EmissionFunctionArray::calculate_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data)
  {
    printf("computing thermal spectra from vhydro with df...\n\n");

    double prefactor = pow(2.0 * M_PI * hbarC, -3);   // prefactor of CFF
    int FO_chunk = 10000;                             // size of chunk

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab -> get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);
    }

    // pT array
    double pTValues[pT_tab_length];

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab -> get(1, ipT + 1);
    }

    // y and eta arrays
    int y_pts = y_tab_length;    // default points for 3+1d
    int eta_pts = 1;

    if(DIMENSION == 2)
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaWeights[eta_pts];


    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
        //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_pts; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc((long long int)npart * (long long int)FO_chunk * (long long int)pT_tab_length * (long long int)phi_tab_length * (long long int)y_tab_length, sizeof(double));


    //loop over bite size chunks of FO surface
    for(int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      int endFO = FO_chunk;
      if(n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); // don't go out of array bounds
      #pragma omp parallel for
      for(int icell = 0; icell < endFO; icell++)  // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell;     // global FO cell index

        if(icell_glb < 0 || icell_glb > FO_length - 1)
        {
          printf("Error: icell_glb out of bounds! Probably want to fix integer type...\n");
          exit(-1);
        }

        // set freezeout info to local varibles to reduce memory access outside cache:
        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double tau2 = tau * tau;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
        }

        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];         // dan should be 0 for 2+1d

        double ux = ux_fo[icell_glb];           // contravariant fluid velocity
        double uy = uy_fo[icell_glb];           // enforce normalization
        double un = un_fo[icell_glb];
        double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un);

        double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;  // u.dsigma / eta_weight

        if(udsigma <= 0.0) continue;            // skip cells with u.dsigma < 0

        double ux2 = ux * ux;                   // useful expressions
        double uy2 = uy * uy;
        double ut2 = ut * ut;
        double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

        double T = T_fo[icell_glb];             // temperature (GeV)
        double P = P_fo[icell_glb];             // equilibrium pressure (GeV/fm^3)
        double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

        double pitt = 0.0;                      // contravariant shear stress tensor pi^munu (GeV/fm^3)
        double pitx = 0.0;                      // enforce orthogonality pi.u = 0
        double pity = 0.0;                      // and tracelessness Tr(pi) = 0
        double pitn = 0.0;
        double pixx = 0.0;
        double pixy = 0.0;
        double pixn = 0.0;
        double piyy = 0.0;
        double piyn = 0.0;
        double pinn = 0.0;

        if(INCLUDE_SHEAR_DELTAF)
        {
          pixx = pixx_fo[icell_glb];
          pixy = pixy_fo[icell_glb];
          pixn = pixn_fo[icell_glb];
          piyy = piyy_fo[icell_glb];
          piyn = piyn_fo[icell_glb];
          pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
          pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
          pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
          pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
          pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
        }

        double bulkPi = 0.0;                    // bulk pressure (GeV/fm^3)

        if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell_glb];

        double muB = 0.0;                       // baryon chemical potential (GeV)
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density (fm^-3)
        double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
        double Vx = 0.0;                        // enforce orthogonality V.u = 0
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          muB = muB_fo[icell_glb];
          nB = nB_fo[icell_glb];
          Vx = Vx_fo[icell_glb];
          Vy = Vy_fo[icell_glb];
          Vn = Vn_fo[icell_glb];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

          alphaB = muB / T;
          baryon_enthalpy_ratio = nB / (E + P);
        }

        // set df coefficients
        deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

        double c0 = df.c0;             // 14 moment coefficients
        double c1 = df.c1;
        double c2 = df.c2;
        double c3 = df.c3;
        double c4 = df.c4;

        double F = df.F;               // Chapman Enskog
        double G = df.G;
        double betabulk = df.betabulk;
        double betaV = df.betaV;
        double betapi = df.betapi;

        // evaluate shear and bulk coefficients
        double shear_coeff = 0.0;
        double bulk0_coeff = 0.0;
        double bulk1_coeff = 0.0;
        double bulk2_coeff = 0.0;

        switch(DF_MODE)
        {
          case 1: // 14 moment
          {
            shear_coeff = 0.5 / (T * T * (E + P));
            bulk0_coeff = c0 - c2;
            bulk1_coeff = c1;
            bulk2_coeff = 4.0 * c2  -  c0;
            break;
          }
          case 2: // Chapman enskog
          {
            shear_coeff = 0.5 / (betapi * T);
            bulk0_coeff = F / (T * T * betabulk);
            bulk1_coeff = G / betabulk;
            bulk2_coeff = 1.0 / (3.0 * T * betabulk);
            break;
          }
          default:
          {
            printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
          }
        }


        // now loop over all particle species and momenta
        for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          // set particle properties
          double mass = Mass[ipart];              // mass (GeV)
          double mass2 = mass * mass;
          double sign = Sign[ipart];              // quantum statistics sign
          double degeneracy = Degeneracy[ipart];  // spin degeneracy
          double baryon = Baryon[ipart];          // baryon number
          double chem = baryon * alphaB;          // chemical potential term in feq

          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pTValues[ipT];              // p_T (GeV)
            double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
            double mT_over_tau = mT / tau;

            for(int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double px = pT * cosphiValues[iphip]; // p^x
              double py = pT * sinphiValues[iphip]; // p^y

              for(int iy = 0; iy < y_pts; iy++)
              {
                double y = yValues[iy];

                double pdotdsigma_f_eta_sum = 0.0;          // Cooper Frye integral over eta

                // sum over eta
                for(int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];

                  double pt = mT * cosh(y - eta);           // p^\tau (GeV)
                  double pn = mT_over_tau * sinh(y - eta);  // p^\eta (GeV^2)
                  double tau2_pn = tau2 * pn;

                  double pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                  double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;  // u.p

                  double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                  double feqbar = 1.0  -  sign * feq;

                  // pi^munu.p_mu.p_nu
                  double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                  + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                  // V^mu.p_mu
                  double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                  double df;  // delta f correction

                  switch(DF_MODE)
                  {
                    case 1: // 14 moment
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu;
                      double df_bulk = (bulk0_coeff * mass2  +  (bulk1_coeff * baryon  +  bulk2_coeff * pdotu) * pdotu) * bulkPi;
                      double df_diff = (c3 * baryon  +  c4 * pdotu) * Vmu_pmu;

                      df = feqbar * (df_shear + df_bulk + df_diff);

                      break;
                    }
                    case 2: // Chapman enskog
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                      double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;

                      df = feqbar * (df_shear + df_bulk + df_diff);
                      break;
                    }
                    default:
                    {
                      printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
                    }
                  } // DF_MODE

                  if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                  double f = feq * (1.0 + df);  // distribution function with linear df correction

                  pdotdsigma_f_eta_sum += (pdotdsigma * f);


                } // eta points (ieta)

                long long int iSpectra = (long long int)icell + (long long int)endFO * ((long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy)));

                dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

      } // freezeout cells in the chunk (icell)

      if(endFO != 0)
      {
        // now perform the reduction over cells
        #pragma omp parallel for collapse(4)
        for(int ipart = 0; ipart < npart; ipart++)
        {
          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            for(int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              for(int iy = 0; iy < y_pts; iy++)
              {
                long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

                double dN_pTdpTdphidy_tmp = 0.0; // reduction variable

                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for(int icell = 0; icell < endFO; icell++)
                {
                  long long int iSpectra = (long long int)icell + (long long int)endFO * iS3D;
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];

                } // sum over freezeout cells in the chunk (icell)

                dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; // sum over all chunks

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

      } // do reduction step if chunk size is nonzero if(endFO != 0)

      printf("Progress: finished chunk %d of %ld \n", n, FO_length / FO_chunk);

    } // FO surface chunks (n)

    // free memory
    free(dN_pTdpTdphidy_all);
  }


  void EmissionFunctionArray::calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data * df_data)
  {
    printf("computing thermal spectra from vhydro with feqmod...\n\n");

    double prefactor = pow(2.0 * M_PI * hbarC, -3);

    int FO_chunk = 10000;

    double detA_min = DETA_MIN;   // default value for minimum detA
    int breakdown = 0;            // # times feqmod breaks down

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    double phiWeights[phi_tab_length];

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab->get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);
      phiWeights[iphip] = phi_tab->get(2, iphip + 1);
    }

    // pT array
    double pTValues[pT_tab_length];

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab->get(1, ipT + 1);
    }

    // y and eta arrays
    int y_pts = y_tab_length;    // default points for 3+1d
    int eta_pts = 1;

    if(DIMENSION == 2)
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaWeights[eta_pts];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
        //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;           // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_pts; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    /// gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    //in this way, we are limited to small values of FO_chunk, because our array holds values for all species...
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc((long long int)npart * (long long int)FO_chunk * (long long int)pT_tab_length * (long long int)phi_tab_length * (long long int)y_pts, sizeof(double));

    double ** A_copy = (double**)calloc(3, sizeof(double*));
    for(int i = 0; i < 3; i++) A_copy[i] = (double*)calloc(3, sizeof(double));

    double ** A_inv = (double**)calloc(3, sizeof(double*));
    for(int i = 0; i < 3; i++) A_inv[i] = (double*)calloc(3, sizeof(double));

    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
      #pragma omp parallel for
      //#pragma acc kernels
      //#pragma acc loop independent
      for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell;     // global FO cell index

        if(icell_glb < 0 || icell_glb > FO_length - 1)
        {
          printf("Error: icell_glb out of bounds! Probably want to fix integer type...\n");
          exit(-1);
        }

        // set freezeout info to local variables to reduce memory access outside cache:
        double tau = tau_fo[icell_glb];     // longitudinal proper time
        double tau2 = tau * tau;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb]; // spacetime rapidity from freezeout cell
        }

        double dat = dat_fo[icell_glb];     // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ux = ux_fo[icell_glb];       // contravariant fluid velocity
        double uy = uy_fo[icell_glb];       // enforce normalization u.u = 1
        double un = un_fo[icell_glb];
        double ut = sqrt(1.0 +  ux * ux  +  uy * uy  +  tau2 * un * un);

        double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;   // udotdsigma / eta_weight

        if(udsigma <= 0.0) continue;        // skip cells with u.dsigma < 0

        double ut2 = ut * ut;               // useful expressions
        double ux2 = ux * ux;
        double uy2 = uy * uy;
        double uperp = sqrt(ux * ux  +  uy * uy);
        double utperp = sqrt(1.0  +   ux * ux  +  uy * uy);

        double T = T_fo[icell_glb];             // temperature (GeV)
        double P = P_fo[icell_glb];             // equilibrium pressure (GeV/fm^3)
        double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

        double pitt = 0.0;                  // contravariant shear stress tensor pi^munu
        double pitx = 0.0;                  // enforce orthogonality pi.u = 0
        double pity = 0.0;                  // and tracelessness Tr(pi) = 0
        double pitn = 0.0;
        double pixx = 0.0;
        double pixy = 0.0;
        double pixn = 0.0;
        double piyy = 0.0;
        double piyn = 0.0;
        double pinn = 0.0;

        if(INCLUDE_SHEAR_DELTAF)
        {
          pixx = pixx_fo[icell_glb];
          pixy = pixy_fo[icell_glb];
          pixn = pixn_fo[icell_glb];
          piyy = piyy_fo[icell_glb];
          piyn = piyn_fo[icell_glb];
          pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
          pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
          pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
          pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
          pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
        }

        double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

        if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell_glb];

        double muB = 0.0;                       // baryon chemical potential (GeV)
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density (fm^-3)
        double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
        double Vx = 0.0;                        // enforce orthogonality V.u = 0
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          muB = muB_fo[icell_glb];
          nB = nB_fo[icell_glb];
          Vx = Vx_fo[icell_glb];
          Vy = Vy_fo[icell_glb];
          Vn = Vn_fo[icell_glb];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

          alphaB = muB / T;
          baryon_enthalpy_ratio = nB / (E + P);
        }

        // regulate bulk pressure if goes out of bounds given 
        // by Jonah's feqmod to avoid gsl interpolation errors
        if(DF_MODE == 4)
        {
          double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

          if(bulkPi < - P) bulkPi = - (1.0 - 1.e-5) * P;
          else if(bulkPi / P > bulkPi_over_Peq_max) bulkPi = P * (bulkPi_over_Peq_max - 1.e-5);
        }

        // set df coefficients
        deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

        // modified coefficients (Mike / Jonah)
        double F = df.F;               
        double G = df.G;                
        double betabulk = df.betabulk;    
        double betaV = df.betaV;          
        double betapi = df.betapi;       
        double lambda = df.lambda;
        double z = df.z;
        double delta_lambda = df.delta_lambda;
        double delta_z = df.delta_z;

        // milne basis class
        Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
        basis_vectors.test_orthonormality(tau2);

        double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
        double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
        double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
        double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

        // shear stress class
        Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
        pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
        pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

        // baryon diffusion class
        Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
        Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
        Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);

      
        // modified temperature / chemical potential
        double T_mod = T;
        double alphaB_mod = alphaB;

        if(DF_MODE == 3)
        {
          T_mod = T  +  bulkPi * F / betabulk;
          alphaB_mod = alphaB  +  bulkPi * G / betabulk;
        }

        // linearized Chapman Enskog df coefficients (for Mike only)
        double shear_coeff = 0.5 / (betapi * T);      // Jonah linear df also shares shear coeff            
        double bulk0_coeff = F / (T * T * betabulk);
        double bulk1_coeff = G / betabulk;
        double bulk2_coeff = 1.0 / (3.0 * T * betabulk);

        // pimunu and Vmu LRF components
        double pixx_LRF = pimunu.pixx_LRF;
        double pixy_LRF = pimunu.pixy_LRF;
        double pixz_LRF = pimunu.pixz_LRF;
        double piyy_LRF = pimunu.piyy_LRF;
        double piyz_LRF = pimunu.piyz_LRF;
        double pizz_LRF = pimunu.pizz_LRF;

        double Vx_LRF = Vmu.Vx_LRF;
        double Vy_LRF = Vmu.Vy_LRF;
        double Vz_LRF = Vmu.Vz_LRF;

        // local momentum transformation matrix Mij = Aij
        // Aij = ideal + shear + bulk is symmetric
        // Mij is not symmetric if include baryon diffusion (leave for future work)

        // coefficients in Aij
        double shear_mod = 0.5 / betapi;                           
        double bulk_mod = bulkPi / (3.0 * betabulk);

        if(DF_MODE == 4) bulk_mod = lambda;
    
        double Axx = 1.0  +  pixx_LRF * shear_mod  +  bulk_mod;
        double Axy = pixy_LRF * shear_mod;
        double Axz = pixz_LRF * shear_mod;
        double Ayx = Axy;
        double Ayy = 1.0  +  piyy_LRF * shear_mod  +  bulk_mod;
        double Ayz = piyz_LRF * shear_mod;
        double Azx = Axz;
        double Azy = Ayz;
        double Azz = 1.0  +  pizz_LRF * shear_mod  +  bulk_mod;

        double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);

        // set Mij matrix
        double A[] = {Axx, Axy, Axz, 
                      Ayx, Ayy, Ayz, 
                      Azx, Azy, Azz};           // gsl matrix format

        A_copy[0][0] = Axx;  A_copy[0][1] = Axy;  A_copy[0][2] = Axz;
        A_copy[1][0] = Ayx;  A_copy[1][1] = Ayy;  A_copy[1][2] = Ayz;
        A_copy[2][0] = Azx;  A_copy[2][1] = Azy;  A_copy[2][2] = Azz;

        int s;
        gsl_matrix_view M = gsl_matrix_view_array(A, 3, 3);
        gsl_matrix_view LU = gsl_matrix_view_array(A, 3, 3);

        gsl_permutation * p = gsl_permutation_calloc(3);

        gsl_linalg_LU_decomp(&LU.matrix, p, &s);

        gsl_matrix * A_inverse = gsl_matrix_alloc(3,3);

        gsl_linalg_LU_invert(&LU.matrix, p, A_inverse);

        for(int i = 0; i < 3; i++)
        {
          for(int j = 0; j < 3; j++)
          {
            A_inv[i][j] = gsl_matrix_get(A_inverse, i, j);
          }
        }

         // prefactors for equilibrium, linear bulk correction and modified densities (Mike's feqmod)
        double neq_fact = T * T * T / two_pi2_hbarC3;
        double dn_fact = bulkPi / betabulk;
        double J20_fact = T * neq_fact;
        double N10_fact = neq_fact;
        double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

        // determine if feqmod breaks down
        bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, detA_min, z, laguerre, DF_MODE, 0, T);

        if(feqmod_breaks_down)
        {
          breakdown++;
          //cout << setw(5) << setprecision(4) << "feqmod breaks down at " << breakdown << " / " << FO_length << " cell at tau = " << tau << " fm/c:" << "\t detA = " << detA << "\t detA_min = " << detA_min << endl;
        }

        // uniformly rescale eta space by detA if modified momentum space elements are shrunk
        // this rescales the dsigma components orthogonal to the eta direction (only works for 2+1d, y = 0)
        // for integrating modified distribution with narrow (y-eta) distributions
        double eta_scale = 1.0;
        if(detA > detA_min && detA < 1.0 && DIMENSION == 2) eta_scale = detA;

        // loop over hadrons
        for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          // set particle properties
          double mass = Mass[ipart];              // mass (GeV)
          double mass2 = mass * mass;
          double sign = Sign[ipart];              // quantum statistics sign
          double degeneracy = Degeneracy[ipart];  // spin degeneracy
          double baryon = Baryon[ipart];          // baryon number

          double chem = baryon * alphaB;          // chemical potential term in feq
          double chem_mod = baryon * alphaB_mod;  // chemical potential term in feqmod

          // modified renormalization factor
          double renorm = 1.0;           

          if(INCLUDE_BULK_DELTAF)
          {
            if(DF_MODE == 3)
            {
              double mbar = mass / T;
              double mbar_mod = mass / T_mod;

              double neq = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

              double N10 = baryon * N10_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

              double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

              double n_linear = neq  +  dn_fact * (neq  +  N10 * G  +  J20 * F / T / T);

              double n_mod = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);

              renorm = n_linear / n_mod;
            }
            else if(DF_MODE == 4)
            {
              renorm = z;
            }
            
          }

          if((std::isnan(renorm) || std::isinf(renorm)))
          {
            // skip freezeout cell since jonah feqmod doesn't have a switching linearized df option
            cout << "Error: renormalization factor is " << renorm << endl;
            continue;  
          }

          if(DIMENSION == 3)
          {
            // avoid roundoff errors in 2+1d if detA small
            renorm /= detA;
          }

          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pTValues[ipT];              // p_T (GeV)
            double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
            double mT_over_tau = mT / tau;

            for(int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double px = pT * cosphiValues[iphip]; // p^x
              double py = pT * sinphiValues[iphip]; // p^y

              double phi_weight = phiWeights[iphip];

              for(int iy = 0; iy < y_pts; iy++)
              {
                double y = yValues[iy];

                double pdotdsigma_f_eta_sum = 0.0;  // Cooper Frye integral over eta

                // integrate over eta
                for(int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];  

                  bool feqmod_breaks_down_narrow = false;

                  if(DIMENSION == 3 && !feqmod_breaks_down)
                  {
                    if(detA < 0.01 && fabs(y - eta) < detA) 
                    {
                      feqmod_breaks_down_narrow = true;
                    }
                  }

                  double pdotdsigma;
                  double f;           // feqmod (if breakdown do feq(1+df))                               

                  // calculate feqmod
                  if(feqmod_breaks_down || feqmod_breaks_down_narrow)
                  {
                    double pt = mT * cosh(y - eta);          // p^\tau (GeV)
                    double pn = mT_over_tau * sinh(y - eta); // p^\eta (GeV^2)
                    double tau2_pn = tau2 * pn;

                    pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day)  +  pn * dan;

                    if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                    if(DF_MODE == 3)
                    {
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                      double feqbar = 1.0  -  sign * feq;

                       // pi^munu.p_mu.p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                      // V^mu.p_mu
                      double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                      double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu  -  mass2 / pdotu)) * bulkPi;
                      double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;
                    
                      double df = feqbar * (df_shear + df_bulk + df_diff);

                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                      f = feq * (1.0 + df);
                    }
                    else if(DF_MODE == 4)
                    {
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T) + sign);
                      double feqbar = 1.0  -  sign * feq;

                       // pi^munu.p_mu.p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                      double df_shear = feqbar * shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = delta_z  -  3.0 * delta_lambda  +  feqbar * delta_lambda * (pdotu  -  mass2 / pdotu) / T;
                    
                      double df = df_shear + df_bulk;

                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                      f = feq * (1.0 + df);
                    }
                  } // feqmod breaks down
                  else
                  {
                    double pt = mT * cosh(y - eta_scale * eta);          // p^\tau (GeV)
                    double pn = mT_over_tau * sinh(y - eta_scale * eta); // p^\eta (GeV^2)
                    double tau2_pn = tau2 * pn;

                    pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day)  +  pn * dan;

                    if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                    // LRF momentum components pi_LRF = - Xi.p
                    double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;
                    double py_LRF = Yx * px  +  Yy * py;
                    double pz_LRF = -Zt * pt  +  Zn * tau2_pn;

                    double pLRF[3] = {px_LRF, py_LRF, pz_LRF};
                    double pLRF_prev[3];
                    
                    double pLRF_mod_prev[3];
                    double pLRF_mod[3];

                    double dpLRF[3];
                    double dpLRF_mod[3];

                    matrix_multiplication(A_inv, pLRF, pLRF_mod, 3, 3);   // evaluate p_mod = A^-1.p at least once
                    
                    double dp;
                    double eps = 1.e-16;
                    
                    for(int i = 0; i < 5; i++)
                    {
                      vector_copy(pLRF_mod, pLRF_mod_prev, 3);                        // copy result for iteration
                      matrix_multiplication(A_copy, pLRF_mod_prev, pLRF_prev, 3, 3);  // compute pLRF error
                      vector_subtraction(pLRF, pLRF_prev, dpLRF, 3);

                      dp = sqrt(dpLRF[0] * dpLRF[0]  +  dpLRF[1] * dpLRF[1]  +  dpLRF[2] * dpLRF[2]);

                      if(dp <= eps) break;

                      matrix_multiplication(A_inv, dpLRF, dpLRF_mod, 3, 3);           // compute correction to pLRF_mod
                      vector_addition(pLRF_mod_prev, dpLRF_mod, pLRF_mod, 3);         // add correction to pLRF_mod
                    } 
                    
                    double px_LRF_mod = pLRF_mod[0];
                    double py_LRF_mod = pLRF_mod[1];
                    double pz_LRF_mod = pLRF_mod[2];

                    double E_mod = sqrt(mass2  +  px_LRF_mod * px_LRF_mod  +  py_LRF_mod * py_LRF_mod  +  pz_LRF_mod * pz_LRF_mod);

                    f = fabs(renorm) / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod 
                  }

                  pdotdsigma_f_eta_sum += (pdotdsigma * f); // add contribution to integral

                } // eta points (ieta)

                long long int iSpectra = (long long int)icell + (long long int)endFO * ((long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy)));

                dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

        gsl_matrix_free(A_inverse);
        gsl_permutation_free(p);

      } // freezeout cells in the chunk (icell)

      if(endFO != 0)
      {
        // now perform the reduction over cells
        #pragma omp parallel for collapse(4)
        //#pragma acc kernels
        for(int ipart = 0; ipart < npart; ipart++)
        {
          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            for(int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              for(int iy = 0; iy < y_pts; iy++)
              {
                long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

                double dN_pTdpTdphidy_tmp = 0.0; // reduction variable

                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for(int icell = 0; icell < endFO; icell++)
                {
                  long long int iSpectra = (long long int)icell + (long long int)endFO * iS3D;
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];

                } // sum over freezeout cells in the chunk (icell)

                dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; // sum over all chunks

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

      } // do reduction step if chunk size is nonzero if(endFO != 0)

      printf("Progress: finished chunk %d of %ld \n", n, FO_length / FO_chunk);

    } // FO surface chunks (n)

    cout << setw(5) << setprecision(4) << "\nfeqmod breaks down for " << breakdown << " cells\n" << endl;

    //free memory
    free(dN_pTdpTdphidy_all);
  }



  void EmissionFunctionArray::calculate_dN_dX(int *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data)
  {
    printf("computing thermal spacetime distribution from vhydro with df...\n\n");

    // dX = tau.dtau.deta, 2.pi.r.dr.deta or 2.pi.tau.r.dtau.dr.deta 
    // only have boost invariance in mind right now so
    // deta = dy and only need to integrate over (pT,phi)

    double prefactor = pow(2.0 * M_PI * hbarC, -3);   // prefactor of CFF
    int FO_chunk = 10000;                             // size of chunk

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    double phiWeights[phi_tab_length];

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab -> get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);

      phiWeights[iphip] = phi_tab -> get(2, iphip + 1);
    }

    // pT array
    double pTValues[pT_tab_length];
    double pTWeights[pT_tab_length];

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab -> get(1, ipT + 1);
      pTWeights[ipT] = pT_tab -> get(2, ipT + 1);
    }

    // y and eta arrays
    int y_pts = y_tab_length;    // default points for 3+1d
    int eta_pts = 1;

    if(DIMENSION == 2)
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaWeights[eta_pts];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_pts; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    // tau grid (bin midpoints)
    const int taubins = TAU_BINS;
    const double taubinwidth = (TAU_MAX - TAU_MIN) / (double)TAU_BINS;
    double tau_midpoint[taubins];
    for(int itau = 0; itau < taubins; itau++)
    {
      tau_midpoint[itau] = TAU_MIN + taubinwidth * ((double)itau + 0.5);    
    }

    // r grid (bin midpoints)
    const int rbins = R_BINS;
    const double rbinwidth = (R_MAX - R_MIN) / (double)R_BINS;
    double r_midpoint[rbins];
    for(int ir = 0; ir < rbins; ir++)
    {
      r_midpoint[ir] = R_MIN + rbinwidth * ((double)ir + 0.5);  
    }


    const int npart = number_of_chosen_particles;

    double dN_taudtaudy[taubins];                // dN_tau.dtau.dy distribution
    double dN_twopirdrdy[rbins];                 // dN_2.pi.r.dr.dy distribution
    double dN_twopitaurdtaudrdy[taubins][rbins]; // dN_2.pi.tau.r.tau.dr.dy distribution  

    double * dN_dy = (double*)calloc(npart, sizeof(double));

    double ** dN_dydeta = (double**)calloc(npart, sizeof(double*));
    for(int i = 0; i < npart; i++) dN_dydeta[i] = (double*)calloc(eta_pts, sizeof(double)); 


    // calculate the spacetime distributions for each particle species
    for(int ipart = 0; ipart < npart; ipart++)
    {
      // make the files
      int mcid = MCID[ipart];    

      printf("Starting spacetime distribution %d\n", mcid);       

      char file_time[255] = "";
      char file_radial[255] = "";
      char file_timeradial[255] = "";

      sprintf(file_time, "results/spacetime_distribution/dN_taudtaudy_%d.dat", mcid);
      sprintf(file_radial, "results/spacetime_distribution/dN_twopirdrdy_%d.dat", mcid);
      sprintf(file_timeradial, "results/spacetime_distribution/dN_twopitaurdtaudrdy_%d.dat", mcid);

      ofstream time_distribution(file_time, ios_base::out);
      ofstream radial_distribution(file_radial, ios_base::out);
      ofstream timeradial_distribution(file_timeradial, ios_base::out);

      // rapidity distribution
      char file_rapidity[255] = "";
      sprintf(file_rapidity, "results/spacetime_distribution/dN_dydeta_%d_%dpt.dat", mcid, eta_pts);
      ofstream rapidity_distribution(file_rapidity, ios_base::out);

      // set particle properties
      double mass = Mass[ipart];              // mass (GeV)
      double mass2 = mass * mass;
      double sign = Sign[ipart];              // quantum statistics sign
      double degeneracy = Degeneracy[ipart];  // spin degeneracy
      double baryon = Baryon[ipart];          // baryon number
    
      // reset spacetime distributions to zero
      for(int itau = 0; itau < taubins; itau++)
      {
        dN_taudtaudy[itau] = 0.0; 
        for(int ir = 0; ir < rbins; ir++) dN_twopitaurdtaudrdy[itau][ir] = 0.0; 
      }
      for(int ir = 0; ir < rbins; ir++) dN_twopirdrdy[ir] = 0.0;


      for(long icell = 0; icell < FO_length; icell++)  // cell index inside each chunk
      {
        // set freezeout info to local varibles to reduce memory access outside cache:
        double tau = tau_fo[icell];         // longitudinal proper time
        double x_pos = x_fo[icell];         // x position 
        double y_pos = y_fo[icell];         // y position

        double tau2 = tau * tau;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell];     // spacetime rapidity from surface file
        }

        double dat = dat_fo[icell];         // covariant normal surface vector
        double dax = dax_fo[icell];
        double day = day_fo[icell];
        double dan = dan_fo[icell];         // dan should be 0 for 2+1d

        double ux = ux_fo[icell];           // contravariant fluid velocity
        double uy = uy_fo[icell];           // enforce normalization
        double un = un_fo[icell];
        double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un);

        double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;  // u.dsigma / eta_weight

        if(udsigma <= 0.0) continue;            // skip cells with u.dsigma < 0

        double ux2 = ux * ux;                   // useful expressions
        double uy2 = uy * uy;
        double ut2 = ut * ut;
        double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

        double T = T_fo[icell];             // temperature (GeV)
        double P = P_fo[icell];             // equilibrium pressure (GeV/fm^3)
        double E = E_fo[icell];             // energy density (GeV/fm^3)

        double pitt = 0.0;                      // contravariant shear stress tensor pi^munu (GeV/fm^3)
        double pitx = 0.0;                      // enforce orthogonality pi.u = 0
        double pity = 0.0;                      // and tracelessness Tr(pi) = 0
        double pitn = 0.0;
        double pixx = 0.0;
        double pixy = 0.0;
        double pixn = 0.0;
        double piyy = 0.0;
        double piyn = 0.0;
        double pinn = 0.0;

        if(INCLUDE_SHEAR_DELTAF)
        {
          pixx = pixx_fo[icell];
          pixy = pixy_fo[icell];
          pixn = pixn_fo[icell];
          piyy = piyy_fo[icell];
          piyn = piyn_fo[icell];
          pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
          pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
          pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
          pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
          pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
        }

        double bulkPi = 0.0;                    // bulk pressure (GeV/fm^3)

        if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

        double muB = 0.0;                       // baryon chemical potential (GeV)
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density (fm^-3)
        double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
        double Vx = 0.0;                        // enforce orthogonality V.u = 0
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          muB = muB_fo[icell];
          nB = nB_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

          alphaB = muB / T;
          baryon_enthalpy_ratio = nB / (E + P);
        }

        double chem = baryon * alphaB;          // chemical potential term in feq

        // set df coefficients
        deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

        double c0 = df.c0;             // 14 moment coefficients
        double c1 = df.c1;
        double c2 = df.c2;
        double c3 = df.c3;
        double c4 = df.c4;

        double F = df.F;               // Chapman Enskog
        double G = df.G;
        double betabulk = df.betabulk;
        double betaV = df.betaV;
        double betapi = df.betapi;

        // evaluate shear and bulk coefficients
        double shear_coeff = 0.0;
        double bulk0_coeff = 0.0;
        double bulk1_coeff = 0.0;
        double bulk2_coeff = 0.0;

        switch(DF_MODE)
        {
          case 1: // 14 moment
          {
            shear_coeff = 0.5 / (T * T * (E + P));
            bulk0_coeff = c0 - c2;
            bulk1_coeff = c1;
            bulk2_coeff = 4.0 * c2  -  c0;
            break;
          }
          case 2: // Chapman enskog
          {
            shear_coeff = 0.5 / (betapi * T);
            bulk0_coeff = F / (T * T * betabulk);
            bulk1_coeff = G / betabulk;
            bulk2_coeff = 1.0 / (3.0 * T * betabulk);
            break;
          }
          default:
          {
            printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
          }
        }


        // compute the dNdy of each freezeout cell (integrating over pT,phi)
        double dN_dy_cell = 0.0;

        for(int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pTValues[ipT];              // p_T (GeV)
          double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
          double mT_over_tau = mT / tau;

          double pT_weight = pTWeights[ipT];

          for(int iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double px = pT * cosphiValues[iphip]; // p^x
            double py = pT * sinphiValues[iphip]; // p^y

            double phi_weight = phiWeights[iphip];

            for(int iy = 0; iy < y_pts; iy++)
            {
              double y = yValues[iy];

              double pdotdsigma_f_eta_sum = 0.0;

              // sum over eta
              for(int ieta = 0; ieta < eta_pts; ieta++)
              {
                double eta = etaValues[ieta];
                double eta_weight = etaWeights[ieta];

                double pt = mT * cosh(y - eta);           // p^\tau (GeV)
                double pn = mT_over_tau * sinh(y - eta);  // p^\eta (GeV^2)
                double tau2_pn = tau2 * pn;

                double pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;  // u.p

                double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                double feqbar = 1.0  -  sign * feq;

                // pi^munu.p_mu.p_nu
                double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                // V^mu.p_mu
                double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                double df;  // delta f correction

                switch(DF_MODE)
                {
                  case 1: // 14 moment
                  {
                    double df_shear = shear_coeff * pimunu_pmu_pnu;
                    double df_bulk = (bulk0_coeff * mass2  +  (bulk1_coeff * baryon  +  bulk2_coeff * pdotu) * pdotu) * bulkPi;
                    double df_diff = (c3 * baryon  +  c4 * pdotu) * Vmu_pmu;

                    df = feqbar * (df_shear + df_bulk + df_diff);

                    break;
                  }
                  case 2: // Chapman enskog
                  {
                    double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                    double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                    double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;

                    df = feqbar * (df_shear + df_bulk + df_diff);
                    break;
                  }
                  default:
                  {
                    printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
                  }
                } // DF_MODE

                if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                double f = feq * (1.0 + df);  // distribution function with linear df correction

                pdotdsigma_f_eta_sum += (pdotdsigma * f);

                dN_dydeta[ipart][ieta] += (pT_weight * phi_weight * prefactor * degeneracy * pdotdsigma * f / eta_weight);


              } // eta points (ieta)

              dN_dy_cell += (pT_weight * phi_weight * prefactor * degeneracy * pdotdsigma_f_eta_sum);

            } // rapidity points (iy)

          } // azimuthal angle points (iphip)

        } // transverse momentum points (ipT)

        dN_dy[ipart] += dN_dy_cell;

        // now determine which spacetime bin the freezeout cell lies

        double r = sqrt(x_pos * x_pos  +  y_pos * y_pos);

        int itau = (int)floor((tau - TAU_MIN) / taubinwidth);
        int ir = (int)floor((r - R_MIN) / rbinwidth);

        if(itau < 0) printf("Error: tau bin index is negative. Adjust the tau_min parameter\n");
        if(ir < 0) printf("Error: r bin index is negative. Adjust the r_min parameter\n");

        // add dNdy to the corresponding bin(s)
        if(itau >= 0 && itau < taubins)
        {
          dN_taudtaudy[itau] += dN_dy_cell;  

          if(ir >= 0 && ir < rbins) dN_twopitaurdtaudrdy[itau][ir] += dN_dy_cell;
        }

        if(ir >= 0 && ir < rbins) dN_twopirdrdy[ir] += dN_dy_cell;                          

      } // freezeout cells (icell)

      // write to file (normalize by the binwidths)
      for(int ir = 0; ir < rbins; ir++)
      {
        double r_mid = r_midpoint[ir];

        radial_distribution << setprecision(6) << scientific << r_mid << "\t" << dN_twopirdrdy[ir] / (2.0 * M_PI * r_mid * rbinwidth)  << "\n";

        for(int itau = 0; itau < taubins; itau++)
        {
          double tau_mid = tau_midpoint[itau];

          timeradial_distribution << setprecision(6) << scientific << tau_mid << "\t" << r_mid << "\t" << dN_twopitaurdtaudrdy[itau][ir] / (2.0 * M_PI * tau_mid * r_mid * taubinwidth * rbinwidth) << "\n";
        }
      }

      for(int itau = 0; itau < taubins; itau++)
      {
        double tau_mid = tau_midpoint[itau];

        time_distribution << setprecision(6) << scientific << tau_mid << "\t" << dN_taudtaudy[itau] / (tau_mid * taubinwidth) << "\n";
      }

      // close files
      time_distribution.close();
      radial_distribution.close();
      timeradial_distribution.close();

      // write rapidity distribution to file
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        rapidity_distribution << setprecision(6) << scientific << etaValues[ieta] << "\t" << dN_dydeta[ipart][ieta] << "\n";
      }
      rapidity_distribution.close();

    } // hadron species (ipart)

    for(int i = 0; i < npart; i++)
    {
      printf("dN_dy = %lf\n", dN_dy[i]);
    }

    free_2D(dN_dydeta, npart); 
    free(dN_dy);

  }


  void EmissionFunctionArray::calculate_dN_dX_feqmod(int *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data *df_data)
  {
    printf("computing thermal spacetime distribution from vhydro with feqmod...\n\n");

    // dX = tau.dtau.deta, 2.pi.r.dr.deta or 2.pi.tau.r.dtau.dr.deta 
    // only have boost invariance in mind right now so
    // deta = dy and only need to integrate over (pT,phi)

    // you compute the spatial distributions by setting up a spacetime
    // grid and binning the freezeout cell's mean particle number 

    double prefactor = pow(2.0 * M_PI * hbarC, -3);   // prefactor of CFF
    int FO_chunk = 10000;                             // size of chunk

    double detA_min = DETA_MIN;   // default value for minimum detA
    int breakdown = 0;            // # times feqmod breaks down

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    double phiWeights[phi_tab_length];

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab -> get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);

      phiWeights[iphip] = phi_tab -> get(2, iphip + 1);
    }

    // pT array
    double pTValues[pT_tab_length];
    double pTWeights[pT_tab_length];

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab -> get(1, ipT + 1);
      pTWeights[ipT] = pT_tab -> get(2, ipT + 1);
    }

    // y and eta arrays
    int y_pts = y_tab_length;    // default points for 3+1d
    int eta_pts = 1;

    if(DIMENSION == 2)
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaWeights[eta_pts];


    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_pts; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    // tau grid (bin midpoints)
    const int taubins = TAU_BINS;
    const double taubinwidth = (TAU_MAX - TAU_MIN) / (double)TAU_BINS;
    double tau_midpoint[taubins];
    for(int itau = 0; itau < taubins; itau++)
    {
      tau_midpoint[itau] = TAU_MIN + taubinwidth * ((double)itau + 0.5);    
    }

    // r grid (bin midpoints)
    const int rbins = R_BINS;
    const double rbinwidth = (R_MAX - R_MIN) / (double)R_BINS;
    double r_midpoint[rbins];
    for(int ir = 0; ir < rbins; ir++)
    {
      r_midpoint[ir] = R_MIN + rbinwidth * ((double)ir + 0.5);  
    }


    const int npart = number_of_chosen_particles;

    double dN_taudtaudy[taubins];                // dN_tau.dtau.dy distribution
    double dN_twopirdrdy[rbins];                 // dN_2.pi.r.dr.dy distribution
    double dN_twopitaurdtaudrdy[taubins][rbins]; // dN_2.pi.tau.r.tau.dr.dy distribution   


    /// gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    double ** A_copy = (double**)calloc(3, sizeof(double*));
    for(int i = 0; i < 3; i++) A_copy[i] = (double*)calloc(3, sizeof(double));

    double ** A_inv = (double**)calloc(3, sizeof(double*));
    for(int i = 0; i < 3; i++) A_inv[i] = (double*)calloc(3, sizeof(double));

    double * dN_dy = (double*)calloc(npart, sizeof(double));

    double ** dN_dydeta = (double**)calloc(npart, sizeof(double*));
    for(int i = 0; i < npart; i++) dN_dydeta[i] = (double*)calloc(eta_pts, sizeof(double));

    // calculate the spacetime distributions for each particle species
    for(int ipart = 0; ipart < npart; ipart++)
    {
      // make the files
      int mcid = MCID[ipart];    

      printf("Starting spacetime distribution %d\n", mcid);       

      char file_time[255] = "";
      char file_radial[255] = "";
      char file_timeradial[255] = "";

      sprintf(file_time, "results/spacetime_distribution/dN_taudtaudy_%d.dat", mcid);
      sprintf(file_radial, "results/spacetime_distribution/dN_twopirdrdy_%d.dat", mcid);
      sprintf(file_timeradial, "results/spacetime_distribution/dN_twopitaurdtaudrdy_%d.dat", mcid);

      ofstream time_distribution(file_time, ios_base::out);
      ofstream radial_distribution(file_radial, ios_base::out);
      ofstream timeradial_distribution(file_timeradial, ios_base::out);

      // rapidity distribution
      char file_rapidity[255] = "";
      sprintf(file_rapidity, "results/spacetime_distribution/dN_dydeta_%d_%dpt.dat", mcid, eta_pts);
      ofstream rapidity_distribution(file_rapidity, ios_base::out);

      // set particle properties
      double mass = Mass[ipart];              // mass (GeV)
      double mass2 = mass * mass;
      double sign = Sign[ipart];              // quantum statistics sign
      double degeneracy = Degeneracy[ipart];  // spin degeneracy
      double baryon = Baryon[ipart];          // baryon number
    
      // reset spacetime distributions to zero
      for(int itau = 0; itau < taubins; itau++)
      {
        dN_taudtaudy[itau] = 0.0; 
        for(int ir = 0; ir < rbins; ir++) dN_twopitaurdtaudrdy[itau][ir] = 0.0; 
      }
      for(int ir = 0; ir < rbins; ir++) dN_twopirdrdy[ir] = 0.0;

 
      // loop over bite size chunks of FO surface
      for(long icell = 0; icell < FO_length; icell++)  // cell index inside each chunk
      {
        // set freezeout info to local varibles to reduce memory access outside cache:
        double tau = tau_fo[icell];         // longitudinal proper time
        double x_pos = x_fo[icell];         // x position 
        double y_pos = y_fo[icell];         // y position

        double tau2 = tau * tau;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell];     // spacetime rapidity from surface file
        }

        double dat = dat_fo[icell];         // covariant normal surface vector
        double dax = dax_fo[icell];
        double day = day_fo[icell];
        double dan = dan_fo[icell];         // dan should be 0 for 2+1d

        double ux = ux_fo[icell];           // contravariant fluid velocity
        double uy = uy_fo[icell];           // enforce normalization
        double un = un_fo[icell];
        double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un);

        double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;  // u.dsigma / eta_weight

        if(udsigma <= 0.0) continue;            // skip cells with u.dsigma < 0

        double ux2 = ux * ux;                   // useful expressions
        double uy2 = uy * uy;
        double ut2 = ut * ut;
        double uperp = sqrt(ux * ux  +  uy * uy);
        double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

        double T = T_fo[icell];             // temperature (GeV)
        double P = P_fo[icell];             // equilibrium pressure (GeV/fm^3)
        double E = E_fo[icell];             // energy density (GeV/fm^3)

        double pitt = 0.0;                      // contravariant shear stress tensor pi^munu (GeV/fm^3)
        double pitx = 0.0;                      // enforce orthogonality pi.u = 0
        double pity = 0.0;                      // and tracelessness Tr(pi) = 0
        double pitn = 0.0;
        double pixx = 0.0;
        double pixy = 0.0;
        double pixn = 0.0;
        double piyy = 0.0;
        double piyn = 0.0;
        double pinn = 0.0;

        if(INCLUDE_SHEAR_DELTAF)
        {
          pixx = pixx_fo[icell];
          pixy = pixy_fo[icell];
          pixn = pixn_fo[icell];
          piyy = piyy_fo[icell];
          piyn = piyn_fo[icell];
          pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
          pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
          pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
          pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
          pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
        }

        double bulkPi = 0.0;                    // bulk pressure (GeV/fm^3)

        if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

        double muB = 0.0;                       // baryon chemical potential (GeV)
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density (fm^-3)
        double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
        double Vx = 0.0;                        // enforce orthogonality V.u = 0
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          muB = muB_fo[icell];
          nB = nB_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

          alphaB = muB / T;
          baryon_enthalpy_ratio = nB / (E + P);
        }

        // regulate bulk pressure if goes out of bounds given 
        // by Jonah's feqmod to avoid gsl interpolation errors
        if(DF_MODE == 4)
        {
          double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

          if(bulkPi <= - P) bulkPi = - (1.0 - 1.e-5) * P;
          else if(bulkPi / P >= bulkPi_over_Peq_max) bulkPi = P * (bulkPi_over_Peq_max - 1.e-5);
        }

        // set df coefficients
        deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

        // modified coefficients (Mike / Jonah)
        double F = df.F;               
        double G = df.G;                
        double betabulk = df.betabulk;    
        double betaV = df.betaV;          
        double betapi = df.betapi;       
        double lambda = df.lambda;
        double z = df.z;
        double delta_lambda = df.delta_lambda;
        double delta_z = df.delta_z;

        // milne basis class
        Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
        basis_vectors.test_orthonormality(tau2);

        double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
        double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
        double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
        double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

        // shear stress class
        Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
        pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
        pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

        // baryon diffusion class
        Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
        Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
        Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);

      
        // modified temperature / chemical potential
        double T_mod = T;
        double alphaB_mod = alphaB;

        if(DF_MODE == 3)
        {
          T_mod = T  +  bulkPi * F / betabulk;
          alphaB_mod = alphaB  +  bulkPi * G / betabulk;
        }

        double chem = baryon * alphaB;          // chemical potential term in feq
        double chem_mod = baryon * alphaB_mod;  // chemical potential term in feqmod

        // linearized Chapman Enskog df coefficients (for Mike only)
        double shear_coeff = 0.5 / (betapi * T);      // Jonah linear df also shares shear coeff            
        double bulk0_coeff = F / (T * T * betabulk);
        double bulk1_coeff = G / betabulk;
        double bulk2_coeff = 1.0 / (3.0 * T * betabulk);

        // pimunu and Vmu LRF components
        double pixx_LRF = pimunu.pixx_LRF;
        double pixy_LRF = pimunu.pixy_LRF;
        double pixz_LRF = pimunu.pixz_LRF;
        double piyy_LRF = pimunu.piyy_LRF;
        double piyz_LRF = pimunu.piyz_LRF;
        double pizz_LRF = pimunu.pizz_LRF;

        double Vx_LRF = Vmu.Vx_LRF;
        double Vy_LRF = Vmu.Vy_LRF;
        double Vz_LRF = Vmu.Vz_LRF;

        // local momentum transformation matrix Mij = Aij
        // Aij = ideal + shear + bulk is symmetric
        // Mij is not symmetric if include baryon diffusion (leave for future work)

        // modified coefficients in Aij
        double shear_mod = 0.5 / betapi;                           
        double bulk_mod = bulkPi / (3.0 * betabulk);

        if(DF_MODE == 4) bulk_mod = lambda;
    
        // Aij elements
        double Axx = 1.0  +  pixx_LRF * shear_mod  +  bulk_mod;
        double Axy = pixy_LRF * shear_mod;
        double Axz = pixz_LRF * shear_mod;
        double Ayy = 1.0  +  piyy_LRF * shear_mod  +  bulk_mod;
        double Ayz = piyz_LRF * shear_mod;
        double Azz = 1.0  +  pizz_LRF * shear_mod  +  bulk_mod;

        double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);

        // determine if feqmod breaks down
        bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, detA_min, z, laguerre, DF_MODE, 0, T);

        // set Aij matrix
        double A[] = {Axx, Axy, Axz, 
                      Axy, Ayy, Ayz, 
                      Axz, Ayz, Azz};           // gsl matrix format

        A_copy[0][0] = Axx;  A_copy[0][1] = Axy;  A_copy[0][2] = Axz;
        A_copy[1][0] = Axy;  A_copy[1][1] = Ayy;  A_copy[1][2] = Ayz;
        A_copy[2][0] = Axz;  A_copy[2][1] = Ayz;  A_copy[2][2] = Azz;

        // compute Aij^-1 using LUP decomposition
        int s;
        gsl_matrix_view LU = gsl_matrix_view_array(A, 3, 3);
        gsl_permutation * p = gsl_permutation_calloc(3);
        gsl_linalg_LU_decomp(&LU.matrix, p, &s);

        gsl_matrix * A_inverse = gsl_matrix_alloc(3, 3);
        gsl_linalg_LU_invert(&LU.matrix, p, A_inverse);

        for(int i = 0; i < 3; i++)
        {
          for(int j = 0; j < 3; j++)
          {
            A_inv[i][j] = gsl_matrix_get(A_inverse, i, j);
          }
        }


        // prefactors for equilibrium, linear bulk correction and modified densities (Mike's feqmod)
        double neq_fact = T * T * T / two_pi2_hbarC3;
        double dn_fact = bulkPi / betabulk;
        double J20_fact = T * neq_fact;
        double N10_fact = neq_fact;
        double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

        if(feqmod_breaks_down)
        {
          breakdown++;
          cout << setw(5) << setprecision(4) << "feqmod breaks down for " << breakdown << " / " << FO_length << " cells  (cell = " << icell << ", tau = " << tau << " fm/c, " << ", detA = " << detA << ", detA_min = " << detA_min << ")" << endl;
        }

        // rescale eta by detA if modified momentum space elements are shrunk
        // for integrating modified distribution with narrow (y-eta) distributions
        // note: this only works for boost invariant surfaces, where dsigma is always orthogonal to eta direction (dan = 0)
        double eta_scale = 1.0;
        if(detA > detA_min && DIMENSION == 2)
        {
          eta_scale = detA;
        }

        // compute the modified renormalization factor
        double renorm = 1.0;     

        if(INCLUDE_BULK_DELTAF)
        {
          if(DF_MODE == 3)
          {
            double mbar = mass / T;
            double mbar_mod = mass / T_mod;

            double neq = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

            double N10 = baryon * N10_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

            double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

            double n_linear = neq  +  dn_fact * (neq  +  N10 * G  +  J20 * F / T / T);

            double n_mod = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);

            renorm = n_linear / n_mod;
          }
          else if(DF_MODE == 4)
          {
            renorm = z;
          }
        }


        if((std::isnan(renorm / detA) || std::isinf(renorm / detA)))
        {
          cout << "Error: renormalization factor is " << renorm << endl;
          continue;  
        }

        if(DIMENSION == 3)
        {
          // avoid roundoff errors in 2+1d if detA small
          renorm /= detA;
        }


        // compute the dNdy of each freezeout cell (integrating over pT,phi)
        double dN_dy_cell = 0.0;

        for(int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pTValues[ipT];              // p_T (GeV)
          double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
          double mT_over_tau = mT / tau;

          double pT_weight = pTWeights[ipT];

          for(int iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double px = pT * cosphiValues[iphip]; // p^x
            double py = pT * sinphiValues[iphip]; // p^y

            double phi_weight = phiWeights[iphip];

            for(int iy = 0; iy < y_pts; iy++)
            {
              double y = yValues[iy];

              double pdotdsigma_f_eta_sum = 0.0;

              // sum over eta
              for(int ieta = 0; ieta < eta_pts; ieta++)
              {
                double eta = etaValues[ieta];
                double eta_weight = etaWeights[ieta];

                bool feqmod_breaks_down_narrow = false;

                /*
                if(DIMENSION == 3 && !feqmod_breaks_down)
                {
                  if(detA < 0.01 && fabs(y - eta) < detA) 
                  {
                    feqmod_breaks_down_narrow = true;
                  }
                }
                */

                double f;                                   // feqmod (if breakdown do feq(1+df))
                double pdotdsigma;

                // calculate feqmod
                if(feqmod_breaks_down || feqmod_breaks_down_narrow)
                {
                  double pt = mT * cosh(y - eta);           // p^\tau (GeV)
                  double pn = mT_over_tau * sinh(y - eta);  // p^\eta (GeV^2)
                  double tau2_pn = tau2 * pn;

                  pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                  if(DF_MODE == 3)
                  {
                    double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                    double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                    double feqbar = 1.0  -  sign * feq;

                     // pi^munu.p_mu.p_nu
                    double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                     + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                    // V^mu.p_mu
                    double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                    double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                    double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu  -  mass2 / pdotu)) * bulkPi;
                    double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;
                  
                    double df = feqbar * (df_shear + df_bulk + df_diff);

                    if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                    f = feq * (1.0 + df);
                  }
                  else if(DF_MODE == 4)
                  {
                    double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                    double feq = 1.0 / (exp(pdotu / T) + sign);
                    double feqbar = 1.0  -  sign * feq;

                     // pi^munu.p_mu.p_nu
                    double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                     + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                    double df_shear = feqbar * shear_coeff * pimunu_pmu_pnu / pdotu;
                    double df_bulk = delta_z  -  3.0 * delta_lambda  +  feqbar * delta_lambda * (pdotu  -  mass2 / pdotu) / T;
                  
                    double df = df_shear + df_bulk;

                    if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                    f = feq * (1.0 + df);
                  }
                } // feqmod breaks down
                else
                {
                  double pt = mT * cosh(y - eta_scale * eta);           // p^\tau (GeV)
                  double pn = mT_over_tau * sinh(y - eta_scale * eta);  // p^\eta (GeV^2)
                  double tau2_pn = tau2 * pn;

                  pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow
                  // LRF momentum components pi_LRF = - Xi.p
                  double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;
                  double py_LRF = Yx * px  +  Yy * py;
                  double pz_LRF = -Zt * pt  +  Zn * tau2_pn;

                  double pLRF[3] = {px_LRF, py_LRF, pz_LRF};
                  double pLRF_mod[3];

                  // LUP method with iterations
            
                  double pLRF_prev[3];
                  double pLRF_mod_prev[3];

                  double dpLRF[3];
                  double dpLRF_mod[3];

                  matrix_multiplication(A_inv, pLRF, pLRF_mod, 3, 3);   // evaluate pLRF_mod = A^-1.pLRF at least once
                
                  double dp;                                            // |pLRF| error (i.e. dp = pLRF - A.pLRF_mod)
                  double epsilon = 1.e-16;

                  // iterate the solution p_LRF_mod (sometimes it gets no improvement)
                  for(int i = 0; i < 5; i++)
                  {
                    vector_copy(pLRF_mod, pLRF_mod_prev, 3);                        // copy solution for iteration
                    matrix_multiplication(A_copy, pLRF_mod_prev, pLRF_prev, 3, 3);  // check |pLRF| error
                    vector_subtraction(pLRF, pLRF_prev, dpLRF, 3);

                    dp = sqrt(dpLRF[0] * dpLRF[0]  +  dpLRF[1] * dpLRF[1]  +  dpLRF[2] * dpLRF[2]);

                    if(dp <= epsilon) break;

                    matrix_multiplication(A_inv, dpLRF, dpLRF_mod, 3, 3);           // compute correction to pLRF_mod
                    vector_addition(pLRF_mod_prev, dpLRF_mod, pLRF_mod, 3);         // add correction to pLRF_mod
                  }
                  
                  double E_mod = sqrt(mass2  +  pLRF_mod[0] * pLRF_mod[0]  +  pLRF_mod[1] * pLRF_mod[1]  +  pLRF_mod[2] * pLRF_mod[2]);
                  
                  f = fabs(renorm) / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod
                  

                }

                pdotdsigma_f_eta_sum += (pdotdsigma * f);

                dN_dydeta[ipart][ieta] += (pT_weight * phi_weight * prefactor * degeneracy * pdotdsigma * f / eta_weight);

              } // eta points (ieta)

              dN_dy_cell += (pT_weight * phi_weight * prefactor * degeneracy * pdotdsigma_f_eta_sum);

            } // rapidity points (iy)

          } // azimuthal angle points (iphip)

        } // transverse momentum points (ipT)

        dN_dy[ipart] += dN_dy_cell;

        gsl_matrix_free(A_inverse);
        gsl_permutation_free(p);


        // now determine which spacetime bin the freezeout cell lies

        double r = sqrt(x_pos * x_pos  +  y_pos * y_pos);

        int itau = (int)floor((tau - TAU_MIN) / taubinwidth);
        int ir = (int)floor((r - R_MIN) / rbinwidth);

        if(itau < 0) printf("Error: tau bin index is negative. Adjust the tau_min parameter\n");
        if(ir < 0) printf("Error: r bin index is negative. Adjust the r_min parameter\n");

        // add dNdy to the corresponding bin(s)
        if(itau >= 0 && itau < taubins)
        {
          dN_taudtaudy[itau] += dN_dy_cell;  

          if(ir >= 0 && ir < rbins) dN_twopitaurdtaudrdy[itau][ir] += dN_dy_cell;
        }

        if(ir >= 0 && ir < rbins) dN_twopirdrdy[ir] += dN_dy_cell;                          

      } // freezeout cells in the chunk (icell)

      // write to file (normalize by the binwidths)
      for(int ir = 0; ir < rbins; ir++)
      {
        double r_mid = r_midpoint[ir];

        radial_distribution << setprecision(6) << scientific << r_mid << "\t" << dN_twopirdrdy[ir] / (2.0 * M_PI * r_mid * rbinwidth)  << "\n";

        for(int itau = 0; itau < taubins; itau++)
        {
          double tau_mid = tau_midpoint[itau];

          timeradial_distribution << setprecision(6) << scientific << tau_mid << "\t" << r_mid << "\t" << dN_twopitaurdtaudrdy[itau][ir] / (2.0 * M_PI * tau_mid * r_mid * taubinwidth * rbinwidth) << "\n";
        }
      }

      for(int itau = 0; itau < taubins; itau++)
      {
        double tau_mid = tau_midpoint[itau];

        time_distribution << setprecision(6) << scientific << tau_mid << "\t" << dN_taudtaudy[itau] / (tau_mid * taubinwidth) << "\n";
      }

      // close files
      time_distribution.close();
      radial_distribution.close();
      timeradial_distribution.close();


      // write rapidity distribution to file
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        rapidity_distribution << setprecision(6) << scientific << etaValues[ieta] << "\t" << dN_dydeta[ipart][ieta] << "\n";
      }
      rapidity_distribution.close();

    } // hadron species (ipart)

    // free memory
    free_2D(A_copy, 3);
    free_2D(A_inv, 3);

    for(int i = 0; i < npart; i++)
    {
      printf("dN_dy = %lf\n", dN_dy[i]);
    }

    free_2D(dN_dydeta, npart); 
    free(dN_dy);
  }




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
        double etaWeights[eta_pts];

        double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

        if(DIMENSION == 2)
        {
          yValues[0] = 0.0;
          for(int ieta = 0; ieta < eta_pts; ieta++)
          {
            etaValues[ieta] = eta_tab->get(1, ieta + 1);
            etaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
            //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
          }
        }
        else if(DIMENSION == 3)
        {
          etaValues[0] = 0.0;       // below, will load eta_fo
          etaWeights[0] = 1.0; // 1.0 for 3+1d
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
                      double eta_weight = etaWeights[ieta];

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
                        pdotdsigma_f_eta_sum += (eta_weight * pdotdsigma * fa * (1.0 + reg_df));
                      }
                      else pdotdsigma_f_eta_sum += (eta_weight * pdotdsigma * fa * (1.0 + fabar * df));

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
