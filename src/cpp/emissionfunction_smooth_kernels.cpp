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
//#ifdef _OPENACC
//#include <accelmath.h>
//#endif
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

using namespace std;

void EmissionFunctionArray::calculate_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff)
  {
    printf("computing thermal spectra from vhydro with df...\n\n");

    double prefactor = pow(2.0 * M_PI * hbarC, -3);
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

    // set rapidity and spacetime rapidity arrays
    // depending on the hydro dimension
    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
      //for (int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);

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

    // Set df coefficients from df_coeff (temperature dependent part below)
    // 14 moment
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;
    // Chapman Enskog
    double F = 0.0;
    double G = 0.0;
    double betabulk = 1.0;
    double betaV = 1.0;
    double betapi = 1.0;

    switch(DF_MODE)
    {
      case 1: // 14 moment
      {
        // bulk coefficients
        c0 = df_coeff[0];
        c1 = df_coeff[1];
        c2 = df_coeff[2];
        // diffusion coefficients
        c3 = df_coeff[3];
        c4 = df_coeff[4];
        break;
      }
      case 2: // Chapman enskog
      {
        // bulk coefficients
        F = df_coeff[0];
        G = df_coeff[1];
        betabulk = df_coeff[2];
        // diffusion coeffficients
        betaV = df_coeff[3];
        // shear coefficient
        betapi = df_coeff[4];
        break;
      }
      default:
      {
        cout << "Please specify df_mode = (1,2) in parameters.dat..." << endl;
        exit(-1);
      }
    } // DF_MODE

    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc((long long int)npart * (long long int)FO_chunk * (long long int)pT_tab_length * (long long int)phi_tab_length * (long long int)y_tab_length, sizeof(double));

    double error = 0.0;

    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      printf("Progress: finished chunk %d of %ld \n", n, FO_length / FO_chunk);
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
      #pragma omp parallel for
      for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell;     // global FO cell index

        if(icell_glb < 0 || icell_glb > FO_length - 1)
        {
          printf("Error: icell_glb out of bounds! Probably want to fix integer type...\n");
          exit(-1);
        }

        // set freezeout info to local varibles to reduce(?) memory access outside cache:
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
        double ux2 = ux * ux;
        double uy2 = uy * uy;
        double ut = sqrt(fabs(1.0 + ux2 + uy2 + tau2*un*un));
        double ut2 = ut * ut;
        double u0 = sqrt(fabs(1.0 + ux2 + uy2));

        double udotdsigma = ut * dat + ux * dax + uy * day + un * dan;

        double T = T_fo[icell_glb];             // temperature
        double E = E_fo[icell_glb];             // energy density
        double P = P_fo[icell_glb];             // pressure

        //double pitt = pitt_fo[icell_glb];     // pi^munu
        //double pitx = pitx_fo[icell_glb];     // enforce orthogonality and tracelessness
        //double pity = pity_fo[icell_glb];
        //double pitn = pitn_fo[icell_glb];
        double pixx = pixx_fo[icell_glb];
        double pixy = pixy_fo[icell_glb];
        double pixn = pixn_fo[icell_glb];
        double piyy = piyy_fo[icell_glb];
        double piyn = piyn_fo[icell_glb];
        //double pinn = pinn_fo[icell_glb];
        double pinn = (pixx*(ux2 - ut2) + piyy*(uy2 - ut2) + 2.0*(pixy*ux*uy + tau2*un*(pixn*ux + piyn*uy))) / (tau2 * u0 * u0);
        double pitn = (pixn*ux + piyn*uy + tau2*pinn*un) / ut;
        double pity = (pixy*ux + piyy*uy + tau2*piyn*un) / ut;
        double pitx = (pixx*ux + pixy*uy + tau2*pixn*un) / ut;
        double pitt = (pitx*ux + pity*uy + tau2*pitn*un) / ut;

        double bulkPi = bulkPi_fo[icell_glb];   // bulk pressure

        double muB = 0.0;                       // baryon chemical potential
        double alphaB = 0.0;
        double nB = 0.0;                        // net baryon density
        double Vt = 0.0;                        // baryon diffusion
        double Vx = 0.0;                        // enforce orthogonality
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell_glb];
          alphaB = muB / T;

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
            //Vt = Vt_fo[icell_glb];
            Vt = (Vx*ux + Vy*uy + tau2*Vn*un) / ut;
            if(DF_MODE == 2) baryon_enthalpy_ratio = nB / (E + P);
          }
        }

        double uperp =  sqrt(ux2 + uy2);
        double utperp = sqrt(1.0 + uperp * uperp);

        Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);

        double Xt = basis_vectors.Xt;
        double Xx = basis_vectors.Xx;
        double Xy = basis_vectors.Xy;
        double Xn = basis_vectors.Xn;
        double Yx = basis_vectors.Yx;
        double Yy = basis_vectors.Yy;
        double Zt = basis_vectors.Zt;
        double Zn = basis_vectors.Zn;

        Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
        pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);
        //pimunu.compute_pimunu_max();

        //cout << setprecision(15) << pimunu.pitx_LRF << endl;


        // evaluate shear and bulk coefficients (temperature dependent)
        double shear_coeff = 0.0;
        double bulk0_coeff = 0.0;
        double bulk1_coeff = 0.0;
        double bulk2_coeff = 0.0;

        switch(DF_MODE)
        {
          case 1: // 14 moment
          {
            if(INCLUDE_SHEAR_DELTAF) shear_coeff = 0.5 / (T * T * (E + P));
            if(INCLUDE_BULK_DELTAF)
            {
              bulk0_coeff = c0 - c2;
              bulk1_coeff = c1;
              bulk2_coeff = 4.0*c2 - c0;
            }
            break;
          }
          case 2: // Chapman enskog
          {
            if(INCLUDE_SHEAR_DELTAF) shear_coeff = 0.5 / (betapi * T);
            if(INCLUDE_BULK_DELTAF)
            {
              bulk0_coeff = F / (T * T * betabulk);
              bulk1_coeff = G / betabulk;
              bulk2_coeff = 0.3333333333333333 / (T * betabulk);
            }
            break;
          }
          default:
          {
            cout << "Please specify df_mode = (1,2) in parameters.dat..." << endl;
            exit(-1);
          }
        }

        //now loop over all particle species and momenta
        for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          if(udotdsigma <= 0.0)
          {
            break;
          }
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
                  double pdotdsigma = delta_eta_weight * (pt * dat + px * dax + py * day + pn * dan);

                  // u.p LRF energy
                  double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                  // thermal distribution
                  double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);

                  double f;

                  switch(DF_MODE)
                  {
                    case 1: // 14 moment
                    {
                      // shear correction
                      double df_shear = 0.0;
                      if(INCLUDE_SHEAR_DELTAF)
                      {
                        // pi^munu.p_mu.p_nu
                        double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                        + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));
                        // df_shear / feqbar
                        df_shear = shear_coeff * pimunu_pmu_pnu;
                      }
                      // bulk correction
                      double df_bulk = 0.0;
                      if(INCLUDE_BULK_DELTAF)
                      {
                        if(!INCLUDE_BARYON) df_bulk = (bulk0_coeff * mass2  +  bulk2_coeff * pdotu * pdotu) * bulkPi;
                        else df_bulk = (bulk0_coeff * mass2  +  (baryon * bulk1_coeff + bulk2_coeff * pdotu) * pdotu) * bulkPi;
                      }
                      // baryon diffusion correction
                      double df_baryondiff = 0.0;
                      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                      {
                        // V^mu * p_mu
                        double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;
                        df_baryondiff = (baryon * c3  +  c4 * pdotu) * Vmu_pmu;
                      }
                      // viscous corrections
                      double feqbar = 1.0  -  sign * feq;
                      double df = feqbar * (df_shear + df_bulk + df_baryondiff);
                      // regulate df
                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

                      f = feq * (1.0 + df);
                      break;
                    } // case 1
                    case 2: // Chapman enskog
                    {
                      double df_shear = 0.0;
                      if(INCLUDE_SHEAR_DELTAF)
                      {
                        double E = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                        double pX = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;  // -x.p
                        double pY = Yx * px  +  Yy * py;                                // -y.p
                        double pZ = -Zt * pt  +  Zn * tau2_pn;                          // -z.p

                        double pitt_LRF = pimunu.pitt_LRF;
                        double pitx_LRF = pimunu.pitx_LRF;
                        double pity_LRF = pimunu.pity_LRF;
                        double pitz_LRF = pimunu.pitz_LRF;
                        double pixx_LRF = pimunu.pixx_LRF;
                        double pixy_LRF = pimunu.pixy_LRF;
                        double pixz_LRF = pimunu.pixz_LRF;
                        double piyy_LRF = pimunu.piyy_LRF;
                        double piyz_LRF = pimunu.piyz_LRF;
                        double pizz_LRF = pimunu.pizz_LRF;

                        // double pimunu_pmu_pnu =  pitt_LRF * E * E  +  pixx_LRF * pX * pX  +  piyy_LRF * pY * pY  +  pizz_LRF * pZ * pZ
                        // + 2.0 * (-(pitx_LRF * pX  +  pity_LRF * pY) * E  +  pixy_LRF * pX * pY  +  pZ * (pixz_LRF * pX  +  piyz_LRF * pY  -  pitz_LRF * E));

                        double pimunu_pmu_pnu =  pixx_LRF * pX * pX  +  piyy_LRF * pY * pY  +  pizz_LRF * pZ * pZ
                        + 2.0 * (pixy_LRF * pX * pY  +  pZ * (pixz_LRF * pX  +  piyz_LRF * pY));


                        double pimunu_pmu_pnu2 = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                        + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                        if(fabs(pimunu_pmu_pnu - pimunu_pmu_pnu2) > error)
                        {
                          error = fabs(pimunu_pmu_pnu - pimunu_pmu_pnu2);
                          cout << setprecision(15) << error << endl;
                        }


                        //double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                        //+ 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));
                        df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      }
                      double df_bulk = 0.0;
                      if(INCLUDE_BULK_DELTAF)
                      {
                        if(!INCLUDE_BARYON) df_bulk = (bulk0_coeff * pdotu  +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                        else df_bulk = (bulk0_coeff * pdotu  +  baryon * bulk1_coeff  +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                      }
                      double df_baryondiff = 0.0;
                      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                      {
                        double Vmu_pmu = Vt * pt   -   Vx * px   -   Vy * py   -   Vn * tau2_pn;
                        df_baryondiff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;
                      }
                      double feqbar = 1.0  -  sign * feq;
                      double df = feqbar * (df_shear + df_bulk + df_baryondiff);
                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

                      f = feq * (1.0 + df);
                      break;
                    } // case 2
                    default:
                    {
                      cout << "Please set df_mode = (1,2) in parameters.dat..." << endl;
                      exit(-1);
                    } // default error
                  } // df_mode
                  if(pdotdsigma > 0.0)
                  {
                    pdotdsigma_f_eta_sum += (pdotdsigma * f);
                  }

                } // ieta

                long long int iSpectra = (long long int)icell + (long long int)endFO * ((long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy)));

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
                long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
                double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  long long int iSpectra = (long long int)icell + (long long int)endFO * iS3D;
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


  void EmissionFunctionArray::calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
    double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
    double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
    double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, const int pbar_pts, double * pbar_root1, double * pbar_root2, double * pbar_weight1, double * pbar_weight2)
  {
    printf("computing thermal spectra from vhydro with feqmod...\n\n");

    double prefactor = pow(2.0 * M_PI * hbarC, -3);
    double two_pi2_hbarC3 =  2.0 * pow(M_PI,2) * pow(hbarC,3);

    int FO_chunk = 10000;

    int breakdown = 0; // counts how many times feqmod breaks down

    // Set momentum, spacetime rapidity (2+1d) points from tables

    // cos(phi), sin(phi) arrays
    double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
    for (int j = 0; j < phi_tab_length; j++)
    {
      double phi = phi_tab->get(1,j+1);
      trig_phi_table[j][0] = cos(phi);
      trig_phi_table[j][1] = sin(phi);
    }

    // pT array
    double pTValues[pT_tab_length];
    for(int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1,ipT+1);

    // y and eta arrays (default 3+1d)
    int y_pts = y_tab_length;
    int eta_pts = 1;
    if(DIMENSION == 2)  // pts for 2+1d
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts]; // eta_weight * delta_eta

    if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // will load eta_fo later
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1,iy+1);
    }
    else if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1,ieta+1);
        // assume uniform eta interval 
        double delta_eta = fabs(eta_tab->get(1,2) - eta_tab->get(1,1));
        double eta_weight = eta_tab->get(2,ieta+1);

        etaDeltaWeights[ieta] = eta_weight * delta_eta;
      }
    }

    if(chosen_pion0.size() == 0 || chosen_pion0[0] != 0)
    {
      // since pion-0 is most susceptible to negative particle densities
      // (one of the feqmod breakdown criteria) at large negative bulk pressure
      printf("Error: please put pion-0 (mc_id = 111) at the top of PDG/chosen_particles.dat before calculating...\n");
      exit(-1);
    }

    int chosen_index_pion0 = chosen_pion0[0];

    // set the df coefficients
    double F = df_coeff[0];           // bulk coefficients
    double G = df_coeff[1];
    double betabulk = df_coeff[2];
    double betaV = df_coeff[3];       // diffusion coefficient
    double betapi = df_coeff[4];      // shear coefficient

    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    //in this way, we are limited to small values of FO_chunk, because our array holds values for all species...
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc((long long int)npart * (long long int)FO_chunk * (long long int)pT_tab_length * (long long int)phi_tab_length * (long long int)y_tab_length, sizeof(double));


    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      printf("Progress: Finished chunk %d of %ld \n", n, FO_length / FO_chunk);
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
      #pragma omp parallel for
      #pragma acc kernels
      //#pragma acc loop independent
      for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell;     // global FO cell index

        if(icell_glb < 0 || icell_glb > FO_length - 1)
        {
          printf("Error: icell_glb out of bounds! Probably want to fix integer type...\n");
          exit(-1);
        }

        // set freezeout info to local varibles to reduce memory access outside cache:
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
        
        double ut2 = ut * ut;               // useful expressions
        double ux2 = ux * ux;
        double uy2 = uy * uy;
        double uperp = sqrt(ux2 + uy2);
        double utperp = sqrt(1.0  +  uperp * uperp);

        double T = T_fo[icell_glb];         // temperature
        double P = P_fo[icell_glb];         // pressure
        double E = E_fo[icell_glb];         // energy density

        double pitt = 0.0;                  // contravariant shear stress tensor pi^munu (GeV/fm^3)
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

        double muB = 0.0;                       // baryon chemical potential
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density
        double Vt = 0.0;                        // baryon diffusion
        double Vx = 0.0;
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell_glb];
          alphaB = muB / T;

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
            Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
            baryon_enthalpy_ratio = nB / (E + P);
          }
        }

        // udotdsigma / delta_eta_weight
        double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;

        // set milne basis vectors
        Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);

        double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
        double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
        double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
        double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

        // shear stress class 
        Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
        pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

        // baryon diffusion class
        Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
        Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);

        // modified temperature and chemical potential 
        double T_mod = T  +  F * bulkPi / betabulk;
        double alphaB_mod = alphaB  +  G * bulkPi / betabulk;

        // linearized Chapman Enskog shear and bulk coefficients
        double shear_coeff = 0.5 / (betapi * T);
        double bulk0_coeff = F / (T * T * betabulk);
        double bulk1_coeff = G / betabulk;
        double bulk2_coeff = 1.0 / (3.0 * T * betabulk); 
        
        double pixx_LRF = pimunu.pixx_LRF;    // pimunu LRF components
        double pixy_LRF = pimunu.pixy_LRF;
        double pixz_LRF = pimunu.pixz_LRF;
        double piyy_LRF = pimunu.piyy_LRF;
        double piyz_LRF = pimunu.piyz_LRF;
        double pizz_LRF = pimunu.pizz_LRF;

        double shear_fact = 0.5 / betapi;
        double bulk_term = bulkPi / (3.0 * betabulk);

        // momentum rescaling matrix Mij = Aij for now
        // - Aij is symmetric, Mij is not symmetric if include baryon diffusion

        double Axx = 1.0  +  pixx_LRF * shear_fact  +  bulk_term;
        double Axy = pixy_LRF * shear_fact;
        double Axz = pixz_LRF * shear_fact;
        double Ayx = Axy;
        double Ayy = 1.0  +  piyy_LRF * shear_fact  +  bulk_term;
        double Ayz = piyz_LRF * shear_fact;
        double Azx = Axz;
        double Azy = Ayz;
        double Azz = 1.0  +  pizz_LRF * shear_fact  +  bulk_term;
   
        // baryon diffusion correction is more complicated b/c it depends on p^prime

        // evaluate detA (only depends on ideal + shear + bulk)
        double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);

        // momentum rescaling matrix (so far, w/ shear and bulk corrections only)
        double ** M = (double**)calloc(3, sizeof(double*));
        for(int i = 0; i < 3; i++) M[i] = (double*)calloc(3, sizeof(double));

        int permutation[3]; // permutation vector for LUP decomposition

        // set M-matrix
        M[0][0] = Axx;  M[0][1] = Axy;  M[0][2] = Axz;
        M[1][0] = Ayx;  M[1][1] = Ayy;  M[1][2] = Ayz;
        M[2][0] = Azx;  M[2][1] = Azy;  M[2][2] = Azz;

        // LUP decompose M once
        LUP_decomposition(M, 3, permutation);

         // prefactors for equilibrium, linear bulk correction and modified densities
        double neq_fact = T * T * T / two_pi2_hbarC3;;
        double dn_fact = bulkPi / betabulk;
        double J20_fact = T * neq_fact;
        double N10_fact = neq_fact;
        double nmod_fact = detA * T_mod * T_mod * T_mod / two_pi2_hbarC3;

        bool negative_pions = false;

        // loop over hadrons
        for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          if(udsigma <= 0.0) break;               // skip cells with u.dsigma < 0 
          
          // set particle properties
          double mass = Mass[ipart];              // mass in GeV
          double mass2 = mass * mass;
          double sign = Sign[ipart];              // quantum statistics sign
          double degeneracy = Degeneracy[ipart];  // spin degeneracy
          double baryon = Baryon[ipart];          // baryon number 

          double chem = baryon * alphaB;          // chemical potential term in feq (for linear CE) 
          double chem_mod = baryon * alphaB_mod;  // chemical potential term in feqmod

          // modified renormalization factor
          double renorm = 1.0;
          double n_linear_over_neq = 1.0;

          if(INCLUDE_BULK_DELTAF)
          {
            double mbar = mass / T;
            double mbar_mod = mass / T_mod;

            double neq = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
            double N10 = baryon * N10_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
            double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

            double dn = dn_fact * (neq + (N10 * G) + (J20 * F / T / T));
            double n_linear = neq + dn;
            n_linear_over_neq = n_linear / neq;
            double n_mod = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);

            renorm = n_linear / n_mod;
          }
          else
          {
            renorm = 1.0 / detA;
          }

          if(ipart == chosen_index_pion0 && n_linear_over_neq < 0.0 && !negative_pions)
          {
            negative_pions = true;
          }

          //if((detA < DETA_MIN || negative_pions) && ipart == chosen_index_pion0)
          if((detA < DETA_MIN || negative_pions || T_mod <= 0.0) && ipart == chosen_index_pion0)
          {
            breakdown++;
            cout << setw(5) << setprecision(4) << "feqmod breaks down at " << breakdown << " / " << FO_length << " cells at tau = " << tau << " fm/c:" << "\t detA = " << detA << "\tnegative pions = " << negative_pions << endl;
          }


          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            // transverse radial momentum and transverse mass
            double pT = pTValues[ipT];
            double mT = sqrt(mass2 + pT * pT);
            double mT_over_tau = mT / tau;

            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double cosphi = trig_phi_table[iphip][0];
              double sinphi = trig_phi_table[iphip][1];
              double px = pT * cosphi;  // p^x
              double py = pT * sinphi;  // p^y

              for (int iy = 0; iy < y_pts; iy++)
              {
                double y = yValues[iy];             // y
                double pdotdsigma_f_eta_sum = 0.0;  // reset sum to zero

                // integrate over eta (flexible for 2+1 or 3+1 hydro)
                for(int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];                     // eta point
                  double delta_eta_weight = etaDeltaWeights[ieta];  // delta_eta * eta_weight (trapezoid rule)

                  // all vector components are CONTRAVARIANT EXCEPT the surface
                  // normal vector dat, dax, day, dan, which are COVARIANT
                  double pt = mT * cosh(y - eta);           // p^tau
                  double pn = mT_over_tau * sinh(y - eta);  // p^eta
                  double tau2_pn = tau2 * pn;
                  // momentum vector is contravariant, surface normal vector is COVARIANT
                  double pdotdsigma = delta_eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan);
                  // distribution feq(1+df) or feqmod
                  double f;

                  // calculate f:
                  // if feqmod breaks down, do chapman enskog instead
                  if(detA < DETA_MIN || negative_pions || T_mod <= 0.0)
                  {
                     double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                     double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                     double df_shear = 0.0;
                     if(INCLUDE_SHEAR_DELTAF)
                     {
                       double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));
                       df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                     }
                     double df_bulk = 0.0;
                     if(INCLUDE_BULK_DELTAF)
                     {
                       if(!INCLUDE_BARYON) df_bulk = (bulk0_coeff * pdotu  +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                       else df_bulk = (bulk0_coeff * pdotu  +  baryon * bulk1_coeff  +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                     }
                     double df_baryondiff = 0.0;
                     if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                     {
                       double Vmu_pmu = Vt * pt   -   Vx * px   -   Vy * py   -   Vn * tau2_pn;
                       df_baryondiff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;
                     }
                     double feqbar = 1.0  -  sign * feq;
                     double df = feqbar * (df_shear + df_bulk + df_baryondiff);

                     // regulate df option
                     if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

                     f = feq * (1.0 + df);
                  }
                  else
                  {
                    // local momentum
                    double pX = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;  // -x.p
                    double pY = Yx * px  +  Yy * py;                                // -y.p
                    double pZ = -Zt * pt  +  Zn * tau2_pn;                          // -z.p
                    double pmod[3] = {pX, pY, pZ};
                    // solve the matrix equation: pi = Mij.pmodj
                    LUP_solve(M, 3, permutation, pmod);
                    // pmod stored in p
                    double pXmod2 = pmod[0] * pmod[0];
                    double pYmod2 = pmod[1] * pmod[1];
                    double pZmod2 = pmod[2] * pmod[2];
                    // effective LRF energy
                    double Emod = sqrt(mass2 + pXmod2 + pYmod2 + pZmod2);

                    // modified equilibrium distribution
                    f = renorm / (exp(Emod / T_mod  -  chem_mod) + sign);

                    // test accuracy of the matrix solver
                    double pX_test = Axx * pmod[0] + Axy * pmod[1] + Axz * pmod[2];
                    double pY_test = Ayx * pmod[0] + Ayy * pmod[1] + Ayz * pmod[2];
                    double pZ_test = Azx * pmod[0] + Azy * pmod[1] + Azz * pmod[2];

                    double dpX = pX - pX_test;
                    double dpY = pY - pY_test;
                    double dpZ = pZ - pZ_test;

                    double dp = sqrt(dpX * dpX  +  dpY * dpY  +  dpZ * dpZ);
                    double p = sqrt(pX * pX  +  pY * pY  +  pZ * pZ);

                    if(dp / p > 1.e-6)
                    {
                      printf("Error: dp / p = %f not accurate enough; please adjust detA_max (increase)\n", dp / p);
                      exit(-1);
                    }

                  }
                  if(pdotdsigma > 0.0) pdotdsigma_f_eta_sum += (pdotdsigma * f);

                } // ieta

                long long int iSpectra = (long long int)icell + (long long int)endFO * ((long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy)));

                dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);

              } //iy
            } //iphip
          } //ipT
        } //ipart
        free_2D(M,3);
      } //icell
      if(endFO != 0)
      {
        //now perform the reduction over cells
        #pragma omp parallel for collapse(3)
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
                long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
                double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  long long int iSpectra = (long long int)icell + (long long int)endFO * iS3D;
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                }//icell
                dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; //sum over all chunks
              }//iy
            }//iphip
          }//ipT
        }//ipart species
      } //if (endFO != 0 )
    }//n FO chunk

    cout << setw(5) << setprecision(4) << "\nfeqmod breaks down for the first " << breakdown << " cells\n" << endl;

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
