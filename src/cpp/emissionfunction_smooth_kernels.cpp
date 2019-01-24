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
      for(int icell = 0; icell < endFO; icell++) // cell index inside each chunk
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


  void EmissionFunctionArray::calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * gla, Deltaf_Data * df_data)
  {
    printf("computing thermal spectra from vhydro with feqmod...\n\n");

    double prefactor = pow(2.0 * M_PI * hbarC, -3);

    int FO_chunk = 10000;

    double detA_min = DETA_MIN;   // default value for minimum detA
    int breakdown = 0;            // # times feqmod breaks down

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
        etaValues[ieta] = eta_tab -> get(1, ieta + 1);
        etaWeights[ieta] = eta_tab -> get(2, ieta + 1);
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
    const int pbar_pts = gla->points;

    double * pbar_root1 = gla->root[1];
    double * pbar_root2 = gla->root[2];

    double * pbar_weight1 = gla->weight[1];
    double * pbar_weight2 = gla->weight[2];

    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    //in this way, we are limited to small values of FO_chunk, because our array holds values for all species...
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc((long long int)npart * (long long int)FO_chunk * (long long int)pT_tab_length * (long long int)phi_tab_length * (long long int)y_tab_length, sizeof(double));

    // momentum rescaling matrix (so far, w/ shear and bulk corrections only)
    double ** M = (double**)calloc(3, sizeof(double*));
    for(int i = 0; i < 3; i++) M[i] = (double*)calloc(3, sizeof(double));

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

        double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;
        double bulkPi_over_Peq = bulkPi / P;

        // this can potentially happen... 
        if(bulkPi_over_Peq > bulkPi_over_Peq_max || bulkPi_over_Peq < -1.0)
        {
          printf("Error: bulk pressure is out of bounds given by Jonah's feqmod\n");
          exit(-1);
        }

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

        // linearized Chapman Enskog df coefficients (for Mike)
        double shear_coeff = 0.5 / (betapi * T);                    // Mike and Jonah share shear_coeff
        double bulk0_coeff = F / (T * T * betabulk);
        double bulk1_coeff = G / betabulk;
        double bulk2_coeff = 1.0 / (3.0 * T * betabulk);

        // linearized Jonah df coefficients (for Jonah)
        // FILL IN LATER

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

        // I'm considering axing this
        double detA_bulk = pow(1.0 + bulk_mod, 3);

        // set Mij matrix
        M[0][0] = Axx;  M[0][1] = Axy;  M[0][2] = Axz;
        M[1][0] = Ayx;  M[1][1] = Ayy;  M[1][2] = Ayz;
        M[2][0] = Azx;  M[2][1] = Azy;  M[2][2] = Azz;

        int permutation[3];                   // permutation vector for LUP decomposition
        LUP_decomposition(M, 3, permutation); // LUP decompose M once

         // prefactors for equilibrium, linear bulk correction and modified densities (Mike's feqmod)
        double neq_fact = T * T * T / two_pi2_hbarC3;
        double dn_fact = bulkPi / betabulk;
        double J20_fact = T * neq_fact;
        double N10_fact = neq_fact;
        double nmod_fact = detA * T_mod * T_mod * T_mod / two_pi2_hbarC3;

        bool feqmod_breaks_down = false;

        if(DF_MODE == 3)
        {
          // calculate linearized pion0 density
          double mbar_pion0 = MASS_PION0 / T;
          double neq_pion0 = neq_fact * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_pion0, 0.0, 0.0, -1.0);
          double J20_pion0 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar_pion0, 0.0, 0.0, -1.0);

          bool pion_density_negative = is_linear_pion0_density_negative(T, neq_pion0, J20_pion0, bulkPi, F, betabulk);

          if(pion_density_negative) detA_min = max(detA_min, detA_bulk); // update min value if pion0 density negative

          if(detA <= detA_min || pion_density_negative) feqmod_breaks_down = true;
        }
        else if(DF_MODE == 4)
        {
          if(z < 0.0)
          {
            printf("Error: z should be positive");
            detA_min = max(detA_min, detA_bulk);  
          }

          if(detA <= detA_min) feqmod_breaks_down = true;
        }

        if(feqmod_breaks_down)
        {
          breakdown++;
          cout << setw(5) << setprecision(4) << "feqmod breaks down at " << breakdown << " / " << FO_length << " cell at tau = " << tau << " fm/c:" << "\t detA = " << detA << "\t detA_min = " << detA_min << endl;
        }

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
          double renorm = 1.0 / detA;             // default (shear only)

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
              renorm = z / detA;
            }
            
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

              for(int iy = 0; iy < y_pts; iy++)
              {
                double y = yValues[iy];

                double pdotdsigma_f_eta_sum = 0.0;  // Cooper Frye integral over eta

                // integrate over eta
                for(int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];  

                  double pt = mT * cosh(y - eta);          // p^\tau (GeV)
                  double pn = mT_over_tau * sinh(y - eta); // p^\eta (GeV^2)
                  double tau2_pn = tau2 * pn;

                  double pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan);

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                  double f;                                   // feqmod (if breakdown do feq(1+df))

                  // calculate feqmod
                  if(!feqmod_breaks_down)
                  //if(detA > 0.001 && !pion_density_negative)
                  //if(detA >= DETA_MIND && !negative_pions)
                  {
                    // LRF momentum components pi_LRF = - Xi.p
                    double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;
                    double py_LRF = Yx * px  +  Yy * py;
                    double pz_LRF = -Zt * pt  +  Zn * tau2_pn;

                    double pLRF_mod[3] = {px_LRF, py_LRF, pz_LRF};

                    LUP_solve(M, 3, permutation, pLRF_mod); // solve the matrix equation pi_LRF = Mij.pj_LRF

                    double px_LRF_mod2 = pLRF_mod[0] * pLRF_mod[0];
                    double py_LRF_mod2 = pLRF_mod[1] * pLRF_mod[1];
                    double pz_LRF_mod2 = pLRF_mod[2] * pLRF_mod[2];

                    double E_mod = sqrt(mass2 + px_LRF_mod2 + py_LRF_mod2 + pz_LRF_mod2);

                    f = renorm / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod

                    // test accuracy of the matrix solver (it works pretty well)
                    double px_LRF_test = Axx * pLRF_mod[0]  +  Axy * pLRF_mod[1]  +  Axz * pLRF_mod[2];
                    double py_LRF_test = Ayx * pLRF_mod[0]  +  Ayy * pLRF_mod[1]  +  Ayz * pLRF_mod[2];
                    double pz_LRF_test = Azx * pLRF_mod[0]  +  Azy * pLRF_mod[1]  +  Azz * pLRF_mod[2];

                    double dpX = px_LRF - px_LRF_test;
                    double dpY = py_LRF - py_LRF_test;
                    double dpZ = pz_LRF - pz_LRF_test;

                    double dp = sqrt(dpX * dpX  +  dpY * dpY  +  dpZ * dpZ);
                    double p = sqrt(px_LRF * px_LRF  +  py_LRF * py_LRF  +  pz_LRF * pz_LRF);

                    // is it not accurate enough?...
                    if(dp / p > 1.e-10)
                    {
                      cout << "Error: dp / p = " << setprecision(16) << dp / p << " not accurate enough at tau = " << tau << " fm/c:"<< endl;
                    }

                  }
                  // if feqmod breaks down switch to chapman enskog instead
                  else
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

                    double df_bulk = 0.0;
                    double df_diff = 0.0;

                    if(DF_MODE == 3)
                    {
                      df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu  -  mass2 / pdotu)) * bulkPi;
                      df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;
                    }
                    else if(DF_MODE == 4)
                    {
                      df_bulk = 0.0;  // fill bulk correction later
                      df_diff = 0.0;
                    }
                    
                    double df = feqbar * (df_shear + df_bulk + df_diff);

                    if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                    f = feq * (1.0 + df);
                  }

                  pdotdsigma_f_eta_sum += (pdotdsigma * f); // add contribution to integral

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
        #pragma omp parallel for collapse(3)
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

    free_2D(M,3);

    cout << setw(5) << setprecision(4) << "\nfeqmod breaks down for " << breakdown << " cells\n" << endl;

    //free memory
    free(dN_pTdpTdphidy_all);
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
