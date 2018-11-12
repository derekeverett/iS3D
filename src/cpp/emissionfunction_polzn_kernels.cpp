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

#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

using namespace std;

void EmissionFunctionArray::calculate_spin_polzn(double *Mass, double *Sign, double *Degeneracy,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *wtx_fo, double *wty_fo, double *wtn_fo, double *wxy_fo, double *wxn_fo, double *wyn_fo)
  {
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
    //to hold the spin polarization vector for each surface cell in a chunk, for all particle species
    int npart = number_of_chosen_particles;

    //four components of spin polarization vector for each FO element
    double *St_all, *Sx_all, *Sy_all, *Sn_all;
    //normalization of the spin polzn vector
    double *Snorm_all;
    St_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));
    Sx_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));
    Sy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));
    Sn_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));
    Snorm_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

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
        double uy = uy_fo[icell_glb];           // enforce normalization
        double un = un_fo[icell_glb];
        double ux2 = ux * ux;
        double uy2 = uy * uy;
        double ut = sqrt(fabs(1.0 + ux2 + uy2 + tau2*un*un));
        double ut2 = ut * ut;
        double u0 = sqrt(fabs(1.0 + ux2 + uy2));

        double T = T_fo[icell_glb];             // temperature
        double E = E_fo[icell_glb];             // energy density
        double P = P_fo[icell_glb];             // pressure

        //thermal vorticity components
        double wtx = wtx_fo[icell];
        double wty = wty_fo[icell];
        double wtn = wtn_fo[icell];
        double wxy = wxy_fo[icell];
        double wxn = wxn_fo[icell];
        double wyn = wyn_fo[icell];

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
                double y = yValues[iy];

                double St_eta_sum = 0.0;
                double Sx_eta_sum = 0.0;
                double Sy_eta_sum = 0.0;
                double Sn_eta_sum = 0.0;
                double Snorm_eta_sum = 0.0;

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

                  // u.p LRF energy
                  double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                  // thermal distribution
                  double f0 = 1.0 / (exp(pdotu / T) + sign);

                  //the components of the covariant polarization vector S_\mu (x,p)
                  double prefactor = -(1.0 / 8.0 / mass ) * (1.0 - sign * f0);
                  double spin_t = prefactor * 2.0 * ( wxy * pn - wxn * py + wyn * px);
                  double spin_x = prefactor * 2.0 * ( wyn * pt - wtn * py + wty * pn);
                  double spin_y = prefactor * 2.0 * ( -wxn * pt + wtn * px - wtx * pn);
                  double spin_n = prefactor * 2.0 * ( wtx * py + wxy * pt - wty * px);

                  St_eta_sum += (delta_eta_weight * pdotdsigma * f0 * spin_t);
                  Sx_eta_sum += (delta_eta_weight * pdotdsigma * f0 * spin_x);
                  Sy_eta_sum += (delta_eta_weight * pdotdsigma * f0 * spin_y);
                  Sn_eta_sum += (delta_eta_weight * pdotdsigma * f0 * spin_n);
                  Snorm_eta_sum += (delta_eta_weight * pdotdsigma * f0);

                } // ieta

                long long int iSpectra = icell + endFO * (ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy)));
                St_all[iSpectra] = St_eta_sum;
                Sx_all[iSpectra] = Sx_eta_sum;
                Sy_all[iSpectra] = Sy_eta_sum;
                Sn_all[iSpectra] = Sn_eta_sum;
                Snorm_all[iSpectra] = Snorm_eta_sum;
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
                double St_tmp = 0.0; //reduction variable
                double Sx_tmp = 0.0; //reduction variable
                double Sy_tmp = 0.0; //reduction variable
                double Sn_tmp = 0.0; //reduction variable
                double Snorm_tmp = 0.0; //reduction variable
                //#pragma omp simd reduction(+:St_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  long long int iSpectra = icell + endFO * iS3D;
                  St_tmp += St_all[iSpectra];
                  Sx_tmp += Sx_all[iSpectra];
                  Sy_tmp += Sy_all[iSpectra];
                  Sn_tmp += Sn_all[iSpectra];
                  Snorm_tmp += Snorm_all[iSpectra];
                }//icell
                St[iS3D] += St_tmp; //sum over all chunks
                Sx[iS3D] += Sx_tmp; //sum over all chunks
                Sy[iS3D] += Sy_tmp; //sum over all chunks
                Sn[iS3D] += Sn_tmp; //sum over all chunks
                Snorm[iS3D] += Snorm_tmp; //sum over all chunks
              }//iy
            }//iphip
          }//ipT
        }//ipart species
      } //if (endFO != 0 )
    }//n FO chunk

    //free memory
    free(St_all);
    free(Sx_all);
    free(Sy_all);
    free(Sn_all);
    free(Snorm_all);
  }
