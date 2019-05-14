#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <complex>
#include <stdio.h>
#include "main.cuh"
#include "readindata.cuh"
#include "emissionfunction.cuh"
#include "Stopwatch.cuh"
#include "arsenal.cuh"
#include "ParameterReader.cuh"
#include "deltafReader.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#define AMOUNT_OF_OUTPUT 0    // smaller value means less outputs
//#define THREADS_PER_BLOCK 128  // try optimizing this, also we can define two different threads/block for the separate kernels
//#define FO_CHUNK 6400

using namespace std;

// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in, Deltaf_Data * df_data_in)
{
  // kernel parameters
  threadsPerBlock = paraRdr_in->getVal("threads_per_block");
  FO_chunk = paraRdr_in->getVal("chunk_size");

  // control parameters
  OPERATION = paraRdr_in->getVal("operation");
  if(OPERATION == 2) 
  {
    printf("Error: there is no particle sampler in cuda\n");
    exit(-1);
  }
  MODE = paraRdr_in->getVal("mode");
  DF_MODE = paraRdr_in->getVal("df_mode");
  DIMENSION = paraRdr_in->getVal("dimension");
  INCLUDE_BARYON = paraRdr_in->getVal("include_baryon");
  INCLUDE_SHEAR_DELTAF = paraRdr_in->getVal("include_shear_deltaf");
  INCLUDE_BULK_DELTAF = paraRdr_in->getVal("include_bulk_deltaf");
  INCLUDE_BARYONDIFF_DELTAF = paraRdr_in->getVal("include_baryondiff_deltaf");
  OUTFLOW = paraRdr_in->getVal("outflow");
  REGULATE_DELTAF = paraRdr_in->getVal("regulate_deltaf");


  // freezeout surface
  surf_ptr = surf_ptr_in;
  FO_length = FO_length_in;


  // momentum tables
  pT_tab = pT_tab_in;
  phi_tab = phi_tab_in;
  y_tab = y_tab_in;
  eta_tab = eta_tab_in;
  pT_tab_length = pT_tab->getNumberOfRows();
  phi_tab_length = phi_tab->getNumberOfRows();
  y_tab_length = y_tab->getNumberOfRows();
  eta_tab_length = 1;

  if(DIMENSION == 2)
  {
    y_tab_length = 1;
    eta_tab_length = eta_tab->getNumberOfRows();
  }
  y_minus_eta_tab_length = eta_tab->getNumberOfRows();


  // particle info
  particles = particles_in;
  Nparticles = Nparticles_in;
  npart = chosen_particles_in->getNumberOfRows();
  chosen_particles_table = (long*)calloc(npart, sizeof(long));

  for(long ipart = 0; ipart < npart; ipart++)
  {
    long mc_id = chosen_particles_in->get(1, ipart + 1);
    bool found_match = false;

    // search for match in PDG file
    for(long iPDG = 0; iPDG < Nparticles; iPDG++)
    {
      if(particles[iPDG].mc_id == mc_id)
      {
        chosen_particles_table[ipart] = iPDG;
        found_match = true;
        break;
      }
    }
    if(!found_match)
    {
      printf("Chosen particles error: %ld not found in pdg.dat\n", mc_id);
      exit(-1);
    }
  }


  // df coefficients
  df_data = df_data_in;


  // particle spectra of chosen particle species
  momentum_length = pT_tab_length * phi_tab_length * y_tab_length;
  spectra_length = npart * momentum_length;
  if(OPERATION == 1) dN_pTdpTdphidy = (double*)calloc(spectra_length, sizeof(double));


  // spacetime grid
  tau_min = paraRdr_in->getVal("tau_min");
  tau_max = paraRdr_in->getVal("tau_max");
  tau_bins = paraRdr_in->getVal("tau_bins");
  tau_width = (tau_max - tau_min) / (double)tau_bins;

  r_min = paraRdr_in->getVal("r_min");
  r_max = paraRdr_in->getVal("r_max");
  r_bins = paraRdr_in->getVal("r_bins");
  r_width = (r_max - r_min) / (double)r_bins;

  phi_bins = paraRdr_in->getVal("phip_bins");
  phi_width = two_pi / (double)phi_bins;

  double eta_cut = paraRdr_in->getVal("eta_cut");
  eta_min = - eta_cut;
  eta_max = eta_cut;
  eta_bins = paraRdr_in->getVal("eta_bins");
  eta_width = 2.0 * eta_cut / eta_bins;
  if(DIMENSION == 2) 
  {
    eta_bins = 1;
    eta_width = 1.0;
  }


  // spacetime distributions 
  spacetime_length = eta_bins * (tau_bins + r_bins + phi_bins);
  X_length = npart * spacetime_length;
  if(OPERATION == 0) dN_dX = (double*)calloc(X_length, sizeof(double));
}


EmissionFunctionArray::~EmissionFunctionArray()
{

}


__global__ void calculate_dN_pTdpTdphidy_threadReduction(double* dN_pTdpTdphidy_d_blocks, long endFO, long momentum_length, long pT_tab_length, long phi_tab_length, long y_tab_length, long eta_tab_length, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaWeights_d, double mass_squared, double sign, double prefactor, double baryon, double *T_d, double *tau_d, double *eta_d, double *ux_d, double *uy_d, double *un_d, double *dat_d, double *dax_d, double *day_d, double *dan_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *bulkPi_d, double *alphaB_d, double *Vx_d, double *Vy_d, double *Vn_d, deltaf_coefficients *df_coeff_d, int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION, int OUTFLOW, int DF_MODE)
{

  // contribution of dN_pTdpTdphidy from each thread
  //long threads = THREADS_PER_BLOCK;
  extern __shared__ double dN_pTdpTdphidy_thread[];

  long icell = (long)threadIdx.x +  (long)blockDim.x * (long)blockIdx.x;  // thread index in kernel
  int ithread = threadIdx.x;                                              // thread index in block

  dN_pTdpTdphidy_thread[ithread] = 0.0;                                   // initialize thread contribution to zero

  __syncthreads();                                                        // sync threads for each block

  if(icell < endFO)
  {
    double tau = tau_d[icell];        // longitudinal proper time
    double tau2 = tau * tau;

    if(DIMENSION == 3)
    {
      etaValues_d[0] = eta_d[icell];  // spacetime rapidity from surface
    }
    double dat = dat_d[icell];        // dsigma_mu
    double dax = dax_d[icell];
    double day = day_d[icell];
    double dan = dan_d[icell];

    double ux = ux_d[icell];          // u^mu
    double uy = uy_d[icell];          // enforce u.u = 1
    double un = un_d[icell];
    double ux2 = ux * ux;             // useful expressions
    double uy2 = uy * uy;
    double utperp = sqrt(1.0 + ux2 + uy2);
    double tau2_un = tau2 * un;
    double ut = sqrt(utperp * utperp  +  tau2_un * un);
    double ut2 = ut * ut;

    bool positive_time_volume = (ut * dat  +  ux * dax  +  uy * day  +  un * dan > 0.0);

    double T = T_d[icell];            // temperature

    double pitt = 0.0;                // pi^munu
    double pitx = 0.0;                // enforce pi.u = 0, Tr(pi) = 0
    double pity = 0.0;
    double pitn = 0.0;
    double pixx = 0.0;
    double pixy = 0.0;
    double pixn = 0.0;
    double piyy = 0.0;
    double piyn = 0.0;
    double pinn = 0.0;

    if(INCLUDE_SHEAR_DELTAF)
    {
      pixx = pixx_d[icell];
      pixy = pixy_d[icell];
      pixn = pixn_d[icell];
      piyy = piyy_d[icell];
      piyn = piyn_d[icell];
      pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2_un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
      pitn = (pixn * ux  +  piyn * uy  +  tau2_un * pinn) / ut;
      pity = (pixy * ux  +  piyy * uy  +  tau2_un * piyn) / ut;
      pitx = (pixx * ux  +  pixy * uy  +  tau2_un * pixn) / ut;
      pitt = (pitx * ux  +  pity * uy  +  tau2_un * pitn) / ut;
    }

    double bulkPi = 0.0;              // bulk pressure

    if(INCLUDE_BULK_DELTAF)
    {
      bulkPi = bulkPi_d[icell];
    }

    double alphaB = 0.0;              // muB / T
    double Vt = 0.0;                  // V^mu
    double Vx = 0.0;                  // enforce orthogonality V.u = 0
    double Vy = 0.0;
    double Vn = 0.0;
    double chem = 0.0;

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      alphaB = alphaB_d[icell];
      chem = baryon * alphaB;
      Vx = Vx_d[icell];
      Vy = Vy_d[icell];
      Vn = Vn_d[icell];
      Vt = (Vx * ux  +  Vy * uy  +  Vn * tau2_un) / ut;
    }

    double tau2_pitn = tau2 * pitn;   // useful expressions
    double tau2_pixn = tau2 * pixn;
    double tau2_piyn = tau2 * piyn;
    double tau4_pinn = tau2 * tau2 * pinn;
    double tau2_Vn = tau2 * Vn;

    // get df coefficients
    double c0 = df_coeff_d->c0;       // 14 moment coefficients
    double c1 = df_coeff_d->c1;
    double c2 = df_coeff_d->c2;
    double c3 = df_coeff_d->c3;
    double c4 = df_coeff_d->c4;
    double shear14_coeff = df_coeff_d->shear14_coeff;
    

    double F = df_coeff_d->F;         // Chapman Enskog
    double G = df_coeff_d->G;
    double betabulk = df_coeff_d->betabulk;
    double betaV = df_coeff_d->betaV;
    double betapi = df_coeff_d->betapi;
    double baryon_enthalpy_ratio = df_coeff_d->baryon_enthalpy_ratio;

    // shear and bulk coefficients
    double shear_coeff = 0.0;
    double bulk0_coeff = 0.0;
    double bulk1_coeff = 0.0;
    double bulk2_coeff = 0.0;
    double diff0_coeff = 0.0;
    double diff1_coeff = 0.0;

    switch(DF_MODE)
    {
      case 1: // 14 moment
      {
        shear_coeff = 1.0 / shear14_coeff;
        bulk0_coeff = (c0 - c2) * mass_squared * bulkPi;
        bulk1_coeff = c1 * baryon * bulkPi;
        bulk2_coeff = (4.*c2 - c0) * bulkPi;
        diff0_coeff = c3 * baryon;
        diff1_coeff = c4;
        break;
      }
      case 2: // Chapman enskog
      {
        shear_coeff = 0.5 / (betapi * T);
        bulk0_coeff = F / (T * T * betabulk) * bulkPi;
        bulk1_coeff = G / betabulk * baryon * bulkPi;
        bulk2_coeff = bulkPi / (3.0 * T * betabulk);
        diff0_coeff = baryon_enthalpy_ratio / betaV;
        diff1_coeff = baryon / betaV;
        break;
      }
      default:
      {
        printf("Error: set df_mode = (1,2) in parameters.dat\n");
      }
    }


    // loop over momentum
    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      long iP1D = phi_tab_length * ipT;

      double pT = pT_d[ipT];

      double mT = sqrt(mass_squared  +  pT * pT);
      double mT_over_tau = mT / tau;

      for(long iphip = 0; iphip < phi_tab_length; iphip++)
      {
        long iP2D = y_tab_length * (iphip + iP1D);

        double px = pT * trig_d[iphip];
        double py = pT * trig_d[iphip + phi_tab_length];

        double px_dax = px * dax;   // useful expressions
        double py_day = py * day;

        double px_ux = px * ux;
        double py_uy = py * uy;

        double pixx_px_px = pixx * px * px;
        double piyy_py_py = piyy * py * py;
        double pitx_px = pitx * px;
        double pity_py = pity * py;
        double pixy_px_py = pixy * px * py;
        double tau2_pixn_px = tau2_pixn * px;
        double tau2_piyn_py = tau2_piyn * py;

        double Vx_px = Vx * px;
        double Vy_py = Vy * py;

        for(long iy = 0; iy < y_tab_length; iy++)
        {
          // momentum_length index
          long iP3D = iy + iP2D;

          double y = y_d[iy];

          //long iP = ipT  +  pT_tab_length * (iphip  +  phi_tab_length * iy);

          if(positive_time_volume)
          {
            double eta_integral = 0.0;

            // sum over eta
            for(long ieta = 0; ieta < eta_tab_length; ieta++)
            {
              double eta = etaValues_d[ieta];
              double eta_weight = etaWeights_d[ieta];

              double sinhyeta = sinh(y - eta);
              double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

              double pt = mT * coshyeta;           // p^tau
              double pn = mT_over_tau * sinhyeta;  // p^eta

              double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

              if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

              double E = pt * ut  -  px_ux  -  py_uy  -  pn * tau2_un;  // u.p
              double feq = 1.0 / (exp(E/T  -  chem) + sign);

              double feqbar = 1.0  -  sign * feq;

              // pi^munu.p_mu.p_nu
              double pimunu_pmu_pnu = pitt * pt * pt  +  pixx_px_px  +  piyy_py_py  +  tau4_pinn * pn * pn
                  + 2.0 * (-(pitx_px + pity_py) * pt  +  pixy_px_py  +  pn * (tau2_pixn_px  +  tau2_piyn_py  -  tau2_pitn * pt));

              // V^mu.p_mu
              double Vmu_pmu = Vt * pt  -  Vx_px  -  Vy_py  -  tau2_Vn * pn;

              double df;

              switch(DF_MODE)
              {
                case 1: // 14 moment
                {
                  double df_shear = shear_coeff * pimunu_pmu_pnu;
                  double df_bulk = bulk0_coeff  +  (bulk1_coeff  +  bulk2_coeff * E) * E;
                  double df_diff = (diff0_coeff  +  diff1_coeff * E) * Vmu_pmu;

                  df = feqbar * (df_shear + df_bulk + df_diff);
                  break;
                }
                case 2: // Chapman enskog
                {
                  double df_shear = shear_coeff * pimunu_pmu_pnu / E;
                  double df_bulk = bulk0_coeff * E  +  bulk1_coeff  +  bulk2_coeff * (E  -  mass_squared / E);
                  double df_diff = (diff0_coeff  -  diff1_coeff / E) * Vmu_pmu;

                  df = feqbar * (df_shear + df_bulk + df_diff);
                  break;
                }
                default:
                {
                  printf("Error: set df_mode = (1,2) in parameters.dat\n");
                }
              } // DF_MODE

              if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

              double f = feq * (1.0 + df);

              eta_integral += eta_weight * pdotdsigma * f;

            } // ieta

            dN_pTdpTdphidy_thread[ithread] = prefactor * eta_integral;
          }
          else
          {
            dN_pTdpTdphidy_thread[ithread] = 0.0;
          }

          // perform reduction over threads in each block:
          int N = blockDim.x;  // number of threads in block (must be power of 2)
          __syncthreads();     // prepare threads for reduction

          while(N != 1)
          {
            N /= 2;

            if(ithread < N) // reduce thread pairs
            {
              dN_pTdpTdphidy_thread[ithread] += dN_pTdpTdphidy_thread[ithread + N];
            }
            __syncthreads();
          }

          // store block's contribution to the spectra
          if(ithread == 0)
          {
            long iP_block = iP3D  +  blockIdx.x * momentum_length;

            dN_pTdpTdphidy_d_blocks[iP_block] = dN_pTdpTdphidy_thread[0];
          }

        } // iy

      } // iphip

    } // ipT

  } // icell < endFO

} // end function


__global__ void calculate_dN_dX_threadReduction(double *dN_dX_d_blocks, long endFO, long tau_bins, long r_bins, long phi_bins, long eta_bins, double tau_min, double r_min, double eta_min, double tau_width, double r_width, double phi_width, double eta_width, long pT_tab_length, long phi_tab_length, long y_minus_eta_tab_length, double *pT_d, double *pT_weight_d, double *trig_d, double *phip_weight_d, double *y_minus_eta_d, double *y_minus_eta_weight_d, double mass_squared, double sign, double prefactor, double baryon, double *T_d, double *tau_d, double *x_d, double *y_d, double *eta_d, double *ux_d, double *uy_d, double *un_d, double *dat_d, double *dax_d, double *day_d, double *dan_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *bulkPi_d, double *alphaB_d, double *Vx_d, double *Vy_d, double *Vn_d, deltaf_coefficients *df_coeff_d, int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION, int OUTFLOW, int DF_MODE)
{
  // contribution of dN_pTdpTdphidy from each thread
  //__shared__ double dN_pTdpTdphidy_thread[THREADS_PER_BLOCK];
  long icell = (long)threadIdx.x +  (long)blockDim.x * (long)blockIdx.x;  // thread index in kernel
  long ithread = (long)threadIdx.x;                                       // thread index in block
  long N = blockDim.x;                                                    // number threads per block

  long spacetime_block = (long)blockIdx.x * eta_bins * (tau_bins + r_bins + phi_bins);

  /*
  extern __shared__ double dN_taudtaudeta[];  // worried about going over stack
  
  if(ithread == 0)
  {
    for(long i = 0; i < spacetime_length; i++)
    {
      dN_dX[i] = 0.0;
    }
  }
  */

  __syncthreads();                                             

  if(icell < endFO)
  {
    double tau = tau_d[icell];        // longitudinal proper time
    double tau2 = tau * tau;

    double x = x_d[icell];
    double y = y_d[icell];

    double r = sqrt(x*x + y*y);
    double phi = atan2(y, x);
    if(phi < 0.0) phi += two_pi;

    double eta = 0.0;
    if(DIMENSION == 3)
    {
      eta = eta_d[icell];             // spacetime rapidity from surface
    }

    double dat = dat_d[icell];        // dsigma_mu
    double dax = dax_d[icell];
    double day = day_d[icell];
    double dan = dan_d[icell];

    double ux = ux_d[icell];          // u^mu
    double uy = uy_d[icell];          // enforce u.u = 1
    double un = un_d[icell];
    double ux2 = ux * ux;             // useful expressions
    double uy2 = uy * uy;
    double utperp = sqrt(1.0 + ux2 + uy2);
    double tau2_un = tau2 * un;
    double ut = sqrt(utperp * utperp  +  tau2_un * un);
    double ut2 = ut * ut;

    bool positive_time_volume = (ut * dat  +  ux * dax  +  uy * day  +  un * dan) > 0.0;

    double T = T_d[icell];            // temperature

    double pitt = 0.0;                // pi^munu
    double pitx = 0.0;                // enforce pi.u = 0, Tr(pi) = 0
    double pity = 0.0;
    double pitn = 0.0;
    double pixx = 0.0;
    double pixy = 0.0;
    double pixn = 0.0;
    double piyy = 0.0;
    double piyn = 0.0;
    double pinn = 0.0;

    if(INCLUDE_SHEAR_DELTAF)
    {
      pixx = pixx_d[icell];
      pixy = pixy_d[icell];
      pixn = pixn_d[icell];
      piyy = piyy_d[icell];
      piyn = piyn_d[icell];
      pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2_un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
      pitn = (pixn * ux  +  piyn * uy  +  tau2_un * pinn) / ut;
      pity = (pixy * ux  +  piyy * uy  +  tau2_un * piyn) / ut;
      pitx = (pixx * ux  +  pixy * uy  +  tau2_un * pixn) / ut;
      pitt = (pitx * ux  +  pity * uy  +  tau2_un * pitn) / ut;
    }

    double bulkPi = 0.0;              // bulk pressure

    if(INCLUDE_BULK_DELTAF)
    {
      bulkPi = bulkPi_d[icell];
    }

    double alphaB = 0.0;              // muB / T
    double Vt = 0.0;                  // V^mu
    double Vx = 0.0;                  // enforce orthogonality V.u = 0
    double Vy = 0.0;
    double Vn = 0.0;
    double chem = 0.0;

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      alphaB = alphaB_d[icell];
      chem = baryon * alphaB;
      Vx = Vx_d[icell];
      Vy = Vy_d[icell];
      Vn = Vn_d[icell];
      Vt = (Vx * ux  +  Vy * uy  +  Vn * tau2_un) / ut;
    }

    double tau2_pitn = tau2 * pitn;   // useful expressions
    double tau2_pixn = tau2 * pixn;
    double tau2_piyn = tau2 * piyn;
    double tau4_pinn = tau2 * tau2 * pinn;
    double tau2_Vn = tau2 * Vn;

    // get df coefficients
    double c0 = df_coeff_d->c0;       // 14 moment coefficients
    double c1 = df_coeff_d->c1;
    double c2 = df_coeff_d->c2;
    double c3 = df_coeff_d->c3;
    double c4 = df_coeff_d->c4;
    double shear14_coeff = df_coeff_d->shear14_coeff;
    
    double F = df_coeff_d->F;         // Chapman Enskog
    double G = df_coeff_d->G;
    double betabulk = df_coeff_d->betabulk;
    double betaV = df_coeff_d->betaV;
    double betapi = df_coeff_d->betapi;
    double baryon_enthalpy_ratio = df_coeff_d->baryon_enthalpy_ratio;

    // shear and bulk coefficients
    double shear_coeff = 0.0;
    double bulk0_coeff = 0.0;
    double bulk1_coeff = 0.0;
    double bulk2_coeff = 0.0;
    double diff0_coeff = 0.0;
    double diff1_coeff = 0.0;

    switch(DF_MODE)
    {
      case 1: // 14 moment
      {
        shear_coeff = 1.0 / shear14_coeff;
        bulk0_coeff = (c0 - c2) * mass_squared * bulkPi;
        bulk1_coeff = c1 * baryon * bulkPi;
        bulk2_coeff = (4.*c2 - c0) * bulkPi;
        diff0_coeff = c3 * baryon;
        diff1_coeff = c4;
        break;
      }
      case 2: // Chapman enskog
      {
        shear_coeff = 0.5 / (betapi * T);
        bulk0_coeff = F / (T * T * betabulk) * bulkPi;
        bulk1_coeff = G / betabulk * baryon * bulkPi;
        bulk2_coeff = bulkPi / (3.0 * T * betabulk);
        diff0_coeff = baryon_enthalpy_ratio / betaV;
        diff1_coeff = baryon / betaV;
        break;
      }
      default:
      {
        printf("Error: set df_mode = (1,2) in parameters.dat\n");
      }
    }

    double dN_deta = 0.0;

    if(positive_time_volume)
    {
      // loop over momentum
      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        double pT = pT_d[ipT];
        double pT_weight = pT_weight_d[ipT];

        double mT = sqrt(mass_squared  +  pT * pT);
        double mT_over_tau = mT / tau;

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip_weight = phip_weight_d[iphip];

          double px = pT * trig_d[iphip];
          double py = pT * trig_d[iphip + phi_tab_length];

          double px_dax = px * dax;   // useful expressions
          double py_day = py * day;

          double px_ux = px * ux;
          double py_uy = py * uy;

          double pixx_px_px = pixx * px * px;
          double piyy_py_py = piyy * py * py;
          double pitx_px = pitx * px;
          double pity_py = pity * py;
          double pixy_px_py = pixy * px * py;
          double tau2_pixn_px = tau2_pixn * px;
          double tau2_piyn_py = tau2_piyn * py;

          double Vx_px = Vx * px;
          double Vy_py = Vy * py;

          // integral over y (centered around eta point) 
          for(long iyeta = 0; iyeta < y_minus_eta_tab_length; iyeta++)     
          {
            double y_minus_eta = y_minus_eta_d[iyeta];                 // this should be a seperate table for spacetime integration (borrow from eta gauss table) 
            double y_minus_eta_weight = y_minus_eta_weight_d[iyeta];
          
            double sinhyeta = sinh(y_minus_eta);
            double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

            double pt = mT * coshyeta;           // p^tau
            double pn = mT_over_tau * sinhyeta;  // p^eta

            double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

            if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

            double E = pt * ut  -  px_ux  -  py_uy  -  pn * tau2_un;  // u.p
            double feq = 1.0 / (exp(E/T  -  chem) + sign);

            double feqbar = 1.0  -  sign * feq;

            // pi^munu.p_mu.p_nu
            double pimunu_pmu_pnu = pitt * pt * pt  +  pixx_px_px  +  piyy_py_py  +  tau4_pinn * pn * pn
                + 2.0 * (-(pitx_px + pity_py) * pt  +  pixy_px_py  +  pn * (tau2_pixn_px  +  tau2_piyn_py  -  tau2_pitn * pt));

            // V^mu.p_mu
            double Vmu_pmu = Vt * pt  -  Vx_px  -  Vy_py  -  tau2_Vn * pn;

            double df;

            switch(DF_MODE)
            {
              case 1: // 14 moment
              {
                double df_shear = shear_coeff * pimunu_pmu_pnu;
                double df_bulk = bulk0_coeff  +  (bulk1_coeff  +  bulk2_coeff * E) * E;
                double df_diff = (diff0_coeff  +  diff1_coeff * E) * Vmu_pmu;

                df = feqbar * (df_shear + df_bulk + df_diff);
                break;
              }
              case 2: // Chapman enskog
              {
                double df_shear = shear_coeff * pimunu_pmu_pnu / E;
                double df_bulk = bulk0_coeff * E  +  bulk1_coeff  +  bulk2_coeff * (E  -  mass_squared / E);
                double df_diff = (diff0_coeff  -  diff1_coeff / E) * Vmu_pmu;

                df = feqbar * (df_shear + df_bulk + df_diff);
                break;
              }
              default:
              {
                printf("Error: set df_mode = (1,2) in parameters.dat\n");
              }
            } // DF_MODE

            if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

            double f = feq * (1.0 + df);

            dN_deta += prefactor * pT_weight * phip_weight * y_minus_eta_weight * pdotdsigma * f;

          } // iy

        } // iphip

      } // ipT

    } // if positive time volume


    // bin the spacetime distributions one thread at a time (try N = 32)
    __syncthreads();     

    for(long n = 0; n < N; n++)
    {
      if(ithread == n && icell < endFO)
      {
        long ieta = 0;
        if(DIMENSION == 3)
        {
          ieta = (long)floor((eta - eta_min) / eta_width); 
        }
        
        if(ieta >= 0 && ieta < eta_bins)
        {
          long itau = (long)floor((tau - tau_min) / tau_width); 
          long ir = (long)floor((r - r_min) / r_width);
          long iphi = (long)floor(phi / phi_width);

          if(itau >= 0 && itau < tau_bins) 
          {
            dN_dX_d_blocks[ieta  +  eta_bins * itau  +  spacetime_block] += dN_deta;
          }

          if(ir >= 0 && ir < r_bins) 
          {
            dN_dX_d_blocks[ieta  +  eta_bins * (ir + tau_bins)  +  spacetime_block] += dN_deta;
          }

          if(iphi >= 0 && iphi < phi_bins) 
          {
            dN_dX_d_blocks[ieta  +  eta_bins * (iphi + tau_bins + r_bins) +  spacetime_block] += dN_deta;
          }
        }
      }
      __syncthreads();
    }
   
  } // icell < endFO

} // end function



/*


__global__ void calculate_dN_pTdpTdphidy_feqmod_threadReduction(double* dN_pTdpTdphidy_d_blocks, long endFO, long momentum_length, long pT_tab_length, long phi_tab_length, long y_tab_length, long eta_tab_length, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaWeights_d, double mass_squared, double sign, double prefactor, double baryon, double *T_d, double *tau_d, double *eta_d, double *ux_d, double *uy_d, double *un_d, double *dat_d, double *dax_d, double *day_d, double *dan_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *bulkPi_d, double *alphaB_d, double *Vx_d, double *Vy_d, double *Vn_d, deltaf_coefficients *df_coeff_d, int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION, int OUTFLOW, int DF_MODE, double DETA_MIN, double MASS_PION0, Gauss_Laguerre * laguerre)
{

  // contribution of dN_pTdpTdphidy from each thread
  __shared__ double dN_pTdpTdphidy_thread[THREADS_PER_BLOCK];

  long icell = (long)threadIdx.x +  (long)blockDim.x * (long)blockIdx.x;  // thread index in kernel
  int ithread = threadIdx.x;                                              // thread index in block

  dN_pTdpTdphidy_thread[ithread] = 0.0;                                   // initialize thread contribution to zero

  __syncthreads();                                                        // sync threads for each block

  if(icell < endFO)
  {
    /// gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    double A[3][3];

    double tau = tau_d[icell];        // longitudinal proper time
    double tau2 = tau * tau;

    if(DIMENSION == 3)
    {
      etaValues_d[0] = eta_d[icell];  // spacetime rapidity from surface
    }
    double dat = dat_d[icell];        // dsigma_mu
    double dax = dax_d[icell];
    double day = day_d[icell];
    double dan = dan_d[icell];

    double ux = ux_d[icell];          // u^mu
    double uy = uy_d[icell];          // enforce u.u = 1
    double un = un_d[icell];
    double ux2 = ux * ux;             // useful expressions
    double uy2 = uy * uy;
    double uperp == sqrt(ux2 + uy2);
    double utperp = sqrt(1.0 + ux2 + uy2);
    double tau2_un = tau2 * un;
    double ut = sqrt(utperp * utperp  +  tau2_un * un);
    double ut2 = ut * ut;

    bool positive_time_volume = (ut * dat  +  ux * dax  +  uy * day  +  un * dan > 0.0);

    double T = T_d[icell];            // temperature

    double pitt = 0.0;                // pi^munu
    double pitx = 0.0;                // enforce pi.u = 0, Tr(pi) = 0
    double pity = 0.0;
    double pitn = 0.0;
    double pixx = 0.0;
    double pixy = 0.0;
    double pixn = 0.0;
    double piyy = 0.0;
    double piyn = 0.0;
    double pinn = 0.0;

    if(INCLUDE_SHEAR_DELTAF)
    {
      pixx = pixx_d[icell];
      pixy = pixy_d[icell];
      pixn = pixn_d[icell];
      piyy = piyy_d[icell];
      piyn = piyn_d[icell];
      pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2_un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
      pitn = (pixn * ux  +  piyn * uy  +  tau2_un * pinn) / ut;
      pity = (pixy * ux  +  piyy * uy  +  tau2_un * piyn) / ut;
      pitx = (pixx * ux  +  pixy * uy  +  tau2_un * pixn) / ut;
      pitt = (pitx * ux  +  pity * uy  +  tau2_un * pitn) / ut;
    }

    double bulkPi = 0.0;              // bulk pressure

    if(INCLUDE_BULK_DELTAF)
    {
      bulkPi = bulkPi_d[icell];
    }

    double alphaB = 0.0;              // muB / T
    double Vt = 0.0;                  // V^mu
    double Vx = 0.0;                  // enforce orthogonality V.u = 0
    double Vy = 0.0;
    double Vn = 0.0;
    double chem = 0.0;

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      alphaB = alphaB_d[icell];
      chem = baryon * alphaB;
      Vx = Vx_d[icell];
      Vy = Vy_d[icell];
      Vn = Vn_d[icell];
      Vt = (Vx * ux  +  Vy * uy  +  Vn * tau2_un) / ut;
    }

    // need basis vector components
    double sinhL = tau * un / utperp;
    double coshL = ut / utperp;

    double Xt = uperp * coshL;
    double Xn = uperp * sinhL / tau;
    double Zt = sinhL;
    double Zn = coshL / tau;

    double tau2_Xn = tau2 * Xn;
    double tau2_Zn = tau2 * Zn;

    double Xx, Xy;
    double Yx, Yy;

    if(uperp < 1.e-5)
    {
      // stops (ux=0)/(uperp=0) nans for freezeout cells with no transverse flow
      Xx = 1.0;   Yx = 0.0;
      Xy = 0.0;   Yy = 1.0;
    }
    else
    {
        Xx = utperp * ux / uperp;   Yx = - uy / uperp;
        Xy = utperp * uy / uperp;   Yy = ux / uperp;
    }

    // pimunu LRF components
    double pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
            + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));

    double pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xt + pixy*Xx + piyy*Xy + tau2*piyn*Xn);

    double pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xx - piyn*Xy - tau2*pinn*Xn);

    double piyy_LRF = pixx*Yx*Yx + 2.0*pixy*Yx*Yy + piyy*Yy*Yy;

    double piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);

    double pizz_LRF = - (pixx_LRF + piyy_LRF);

    // get df coefficients
    double F = df_coeff_d->F;
    double G = df_coeff_d->G;
    double betabulk = df_coeff_d->betabulk;
    double baryon_enthalpy_ratio = df_coeff_d->baryon_enthalpy_ratio;
    double betaV = df_coeff_d->betaV;
    double betapi = df_coeff_d->betapi;

    double z = df_coeff_d->z;
    double lambda = df_coeff_d->lambda;
    double delta_z = df_coeff_d->delta_z;
    double delta_lambda = df_coeff_d->delta_lambda;


    double shear_mod = 0.5 / betapi;
    double bulk_mod, T_mod, chem_mod;

    // linearized shear and bulk coefficients
    double shear_coeff = 0.5 / (betapi * T);
    double bulk0_coeff, bulk1_coeff, bulk2_coeff;

    switch(DF_MODE)
    {
      case 3: // Mike
      {
        bulk_mod = bulkPi / (3.0 * betabulk);
        T_mod = T  +  F * bulkPi / betabulk;
        chem_mod = baryon * (alphaB  + G * bulkPi / betabulk);

        bulk0_coeff = F / (T * T * betabulk) * bulkPi;
        bulk1_coeff = G / betabulk * baryon * bulkPi;
        bulk2_coeff = bulkPi / (3.0 * T * betabulk);
        break;
      }
      case 4: // Jonah
      {
        bulk_mod = lambda;
        T_mod = T;
        alphaB_mod = 0.0;

        bulk0_coeff = delta_z  -  3.0 * delta_lambda;
        bulk1_coeff = delta_lambda / T;
        break;
      }
      default:
      {
        printf("Error: set df_mode = (3,3) in parameters.dat\n");
      }
    }

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
    double detA_bulk_two_thirds = (1.0 + bulk_mod) * (1.0 + bulk_mod);

    double neq_fact = T * T * T / two_pi2_hbarC3;
    double dn_fact = bulkPi / betabulk;
    double J20_fact = T * neq_fact;
    double N10_fact = neq_fact;
    double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

    // determine if feqmod breaks down
    bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, DETA_MIN, z, laguerre, DF_MODE, 0, T, F, betabulk);


    // I need to use my old method (no matrix solver iterations)


    // uniformly rescale eta space by detA if modified momentum space elements are shrunk
    // this rescales the dsigma components orthogonal to the eta direction (only works for 2+1d, y = 0)
    // for integrating modified distribution with narrow (y-eta) distributions
    double eta_scale = 1.0;
    if(detA > DETA_MIN && DIMENSION == 2)
    {
      eta_scale = detA / detA_bulk_two_thirds;
    }

    // compute renormalization factor
    double renorm = 1.0;

    if(INCLUDE_BULK_DELTAF)
    {
      if(DF_MODE == 3)
      {
        double mbar = mass / T;
        double mbar_mod = mass / T_mod;

        // maybe compute these individually on the device
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

    // rescale normalization factor
    if(DIMENSION == 2)
    {
      renorm /= detA_bulk_two_thirds;
    }
    else if(DIMENSION == 3)
    {
      renorm /= detA;
    }



    // loop over momentum
    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      double pT = pT_d[ipT];

      double mT = sqrt(mass_squared  +  pT * pT);
      double mT_over_tau = mT / tau;

      for(long iphip = 0; iphip < phi_tab_length; iphip++)
      {
        double px = pT * trig_d[iphip];
        double py = pT * trig_d[iphip + phi_tab_length];

        double px_dax = px * dax;   // useful expressions
        double py_day = py * day;

        double px_ux = px * ux;
        double py_uy = py * uy;

        double pixx_px_px = pixx * px * px;
        double piyy_py_py = piyy * py * py;
        double pitx_px = pitx * px;
        double pity_py = pity * py;
        double pixy_px_py = pixy * px * py;
        double tau2_pixn_px = tau2_pixn * px;
        double tau2_piyn_py = tau2_piyn * py;

        for(long iy = 0; iy < y_tab_length; iy++)
        {
          double y = y_d[iy];

          // momentum_length index
          long iP = ipT  +  pT_tab_length * (iphip  +  phi_tab_length * iy);

          if(positive_time_volume)
          {
            double eta_integral = 0.0;

            // sum over eta
            for(long ieta = 0; ieta < eta_tab_length; ieta++)
            {
              double eta = etaValues_d[ieta];
              double eta_weight = etaWeights_d[ieta];

              bool feqmod_breaks_down_narrow = false;

              if(DIMENSION == 3 && !feqmod_breaks_down)
              {
                if(detA < 0.01 && fabs(y - eta) < detA)
                {
                  feqmod_breaks_down_narrow = true;
                }
              }

              double sinhyeta = sinh(y - eta);
              double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

              double pt = mT * coshyeta;           // p^tau
              double pn = mT_over_tau * sinhyeta;  // p^eta

              double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

              if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow



              double f;

              // calculate feqmod
              if(feqmod_breaks_down || feqmod_breaks_down_narrow)
              {
                double sinhyeta = sinh(y - eta);
                double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

                double pt = mT * coshyeta;           // p^tau
                double pn = mT_over_tau * sinhyeta;  // p^eta

                double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

                if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                double E = pt * ut  -  px_ux  -  py_uy  -  pn * tau2_un;  // u.p
                double feq = 1.0 / (exp(E/T  -  chem) + sign);
                double feqbar = 1.0  -  sign * feq;

                // pi^munu.p_mu.p_nu
                double pimunu_pmu_pnu = pitt * pt * pt  +  pixx_px_px  +  piyy_py_py  +  tau4_pinn * pn * pn
                    + 2.0 * (-(pitx_px + pity_py) * pt  +  pixy_px_py  +  pn * (tau2_pixn_px  +  tau2_piyn_py  -  tau2_pitn * pt));

                double df;

                if(DF_MODE == 3)
                {
                  double df_shear = shear_coeff * pimunu_pmu_pnu / E;
                  double df_bulk = (bulk0_coeff * E  +  bulk1_coeff  +  bulk2_coeff * (E  -  mass_squared / E));

                  df = feqbar * (df_shear + df_bulk);
                }
                else if(DF_MODE == 4)
                {
                  double df_shear = feqbar * shear_coeff * pimunu_pmu_pnu / E;
                  double df_bulk = bulk0_coeff  +  feqbar * bulk1_coeff * (E  -  mass_squared / E);

                  df = df_shear + df_bulk;
                }

                if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                f = feq * (1.0 + df);

              } // feqmod breaks down
              else
              {
                double sinhyeta_scale = sinh(y - eta_scale * eta);

                double pt = mT * sqrt(1.0  +  sinhyeta_scale * sinhyeta_scale);  // p^\tau (GeV)
                double pn = mT_over_tau *  sinhyeta_scale                        // p^\eta (GeV^2)
                double tau2_pn = tau2 * pn;

                // LRF momentum components pi_LRF = - Xi.p
                double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  tau2_Xn * pn;
                double py_LRF = Yx * px  +  Yy * py;
                double pz_LRF = -Zt * pt  +  tau2_Zn * pn;

                double pLRF[3] = {px_LRF, py_LRF, pz_LRF};

                // this is where I need the old method
                //matrix_multiplication(A_inv, pLRF, pLRF_mod, 3, 3);   // evaluate p_mod = A^-1.p at least once

                double px_LRF_mod = pLRF[0];
                double py_LRF_mod = pLRF[1];
                double pz_LRF_mod = pLRF[2];

                double E_mod = sqrt(mass_squared  +  px_LRF_mod * px_LRF_mod  +  py_LRF_mod * py_LRF_mod  +  pz_LRF_mod * pz_LRF_mod);

                f = fabs(renorm) / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod
              }

              eta_integral += eta_weight * pdotdsigma * f;

            } // ieta

            dN_pTdpTdphidy_thread[ithread] = prefactor * eta_integral;
          }
          else
          {
            dN_pTdpTdphidy_thread[ithread] = 0.0;
          }

          // perform reduction over threads in each block:
          int N = blockDim.x;  // number of threads in block (must be power of 2)
          __syncthreads();     // prepare threads for reduction

          while(N != 1)
          {
            N /= 2;

            if(ithread < N) // reduce thread pairs
            {
              dN_pTdpTdphidy_thread[ithread] += dN_pTdpTdphidy_thread[ithread + N];
            }
            __syncthreads();
          }

          // store block's contribution to the spectra
          if(ithread == 0)
          {
            long iP_block = iP  +  blockIdx.x * momentum_length;

            dN_pTdpTdphidy_d_blocks[iP_block] = dN_pTdpTdphidy_thread[0];
          }

        } // iy

      } // iphip

    } // ipT

  } // icell < endFO

} // end function

*/

/*

__global__ void calculate_dN_pTdpTdphidy_feqmod_threadReduction(double* dN_pTdpTdphidy_d_blocks, long endFO, long momentum_length, long pT_tab_length, long phi_tab_length, long y_tab_length, long eta_tab_length, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaWeights_d, double mass_squared, double sign, double prefactor, double baryon, double *T_d, double *tau_d, double *eta_d, double *ux_d, double *uy_d, double *un_d, double *dat_d, double *dax_d, double *day_d, double *dan_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *bulkPi_d, double *alphaB_d, double *Vx_d, double *Vy_d, double *Vn_d, deltaf_coefficients *df_coeff_d, int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION, int OUTFLOW, int DF_MODE, double DETA_MIN, double MASS_PION0, Gauss_Laguerre * laguerre)
{

  // contribution of dN_pTdpTdphidy from each thread
  __shared__ double dN_pTdpTdphidy_thread[THREADS_PER_BLOCK];

  long icell = (long)threadIdx.x +  (long)blockDim.x * (long)blockIdx.x;  // thread index in kernel
  int ithread = threadIdx.x;                                              // thread index in block

  dN_pTdpTdphidy_thread[ithread] = 0.0;                                   // initialize thread contribution to zero

  __syncthreads();                                                        // sync threads for each block

  if(icell < endFO)
  {
    /// gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    double A[3][3];

    double tau = tau_d[icell];        // longitudinal proper time
    double tau2 = tau * tau;

    if(DIMENSION == 3)
    {
      etaValues_d[0] = eta_d[icell];  // spacetime rapidity from surface
    }
    double dat = dat_d[icell];        // dsigma_mu
    double dax = dax_d[icell];
    double day = day_d[icell];
    double dan = dan_d[icell];

    double ux = ux_d[icell];          // u^mu
    double uy = uy_d[icell];          // enforce u.u = 1
    double un = un_d[icell];
    double ux2 = ux * ux;             // useful expressions
    double uy2 = uy * uy;
    double uperp == sqrt(ux2 + uy2);
    double utperp = sqrt(1.0 + ux2 + uy2);
    double tau2_un = tau2 * un;
    double ut = sqrt(utperp * utperp  +  tau2_un * un);
    double ut2 = ut * ut;

    bool positive_time_volume = (ut * dat  +  ux * dax  +  uy * day  +  un * dan > 0.0);

    double T = T_d[icell];            // temperature

    double pitt = 0.0;                // pi^munu
    double pitx = 0.0;                // enforce pi.u = 0, Tr(pi) = 0
    double pity = 0.0;
    double pitn = 0.0;
    double pixx = 0.0;
    double pixy = 0.0;
    double pixn = 0.0;
    double piyy = 0.0;
    double piyn = 0.0;
    double pinn = 0.0;

    if(INCLUDE_SHEAR_DELTAF)
    {
      pixx = pixx_d[icell];
      pixy = pixy_d[icell];
      pixn = pixn_d[icell];
      piyy = piyy_d[icell];
      piyn = piyn_d[icell];
      pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2_un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
      pitn = (pixn * ux  +  piyn * uy  +  tau2_un * pinn) / ut;
      pity = (pixy * ux  +  piyy * uy  +  tau2_un * piyn) / ut;
      pitx = (pixx * ux  +  pixy * uy  +  tau2_un * pixn) / ut;
      pitt = (pitx * ux  +  pity * uy  +  tau2_un * pitn) / ut;
    }

    double bulkPi = 0.0;              // bulk pressure

    if(INCLUDE_BULK_DELTAF)
    {
      bulkPi = bulkPi_d[icell];
    }

    double alphaB = 0.0;              // muB / T
    double Vt = 0.0;                  // V^mu
    double Vx = 0.0;                  // enforce orthogonality V.u = 0
    double Vy = 0.0;
    double Vn = 0.0;
    double chem = 0.0;

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      alphaB = alphaB_d[icell];
      chem = baryon * alphaB;
      Vx = Vx_d[icell];
      Vy = Vy_d[icell];
      Vn = Vn_d[icell];
      Vt = (Vx * ux  +  Vy * uy  +  Vn * tau2_un) / ut;
    }

    // need basis vector components
    double sinhL = tau * un / utperp;
    double coshL = ut / utperp;

    double Xt = uperp * coshL;
    double Xn = uperp * sinhL / tau;
    double Zt = sinhL;
    double Zn = coshL / tau;

    double tau2_Xn = tau2 * Xn;
    double tau2_Zn = tau2 * Zn;

    double Xx, Xy;
    double Yx, Yy;

    if(uperp < 1.e-5)
    {
      // stops (ux=0)/(uperp=0) nans for freezeout cells with no transverse flow
      Xx = 1.0;   Yx = 0.0;
      Xy = 0.0;   Yy = 1.0;
    }
    else
    {
        Xx = utperp * ux / uperp;   Yx = - uy / uperp;
        Xy = utperp * uy / uperp;   Yy = ux / uperp;
    }

    // pimunu LRF components
    double pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
            + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));

    double pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xt + pixy*Xx + piyy*Xy + tau2*piyn*Xn);

    double pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xx - piyn*Xy - tau2*pinn*Xn);

    double piyy_LRF = pixx*Yx*Yx + 2.0*pixy*Yx*Yy + piyy*Yy*Yy;

    double piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);

    double pizz_LRF = - (pixx_LRF + piyy_LRF);

    // get df coefficients
    double F = df_coeff_d->F;
    double G = df_coeff_d->G;
    double betabulk = df_coeff_d->betabulk;
    double baryon_enthalpy_ratio = df_coeff_d->baryon_enthalpy_ratio;
    double betaV = df_coeff_d->betaV;
    double betapi = df_coeff_d->betapi;

    double z = df_coeff_d->z;
    double lambda = df_coeff_d->lambda;
    double delta_z = df_coeff_d->delta_z;
    double delta_lambda = df_coeff_d->delta_lambda;


    double shear_mod = 0.5 / betapi;
    double bulk_mod, T_mod, chem_mod;

    // linearized shear and bulk coefficients
    double shear_coeff = 0.5 / (betapi * T);
    double bulk0_coeff, bulk1_coeff, bulk2_coeff;

    switch(DF_MODE)
    {
      case 3: // Mike
      {
        bulk_mod = bulkPi / (3.0 * betabulk);
        T_mod = T  +  F * bulkPi / betabulk;
        chem_mod = baryon * (alphaB  + G * bulkPi / betabulk);

        bulk0_coeff = F / (T * T * betabulk) * bulkPi;
        bulk1_coeff = G / betabulk * baryon * bulkPi;
        bulk2_coeff = bulkPi / (3.0 * T * betabulk);
        break;
      }
      case 4: // Jonah
      {
        bulk_mod = lambda;
        T_mod = T;
        alphaB_mod = 0.0;

        bulk0_coeff = delta_z  -  3.0 * delta_lambda;
        bulk1_coeff = delta_lambda / T;
        break;
      }
      default:
      {
        printf("Error: set df_mode = (3,3) in parameters.dat\n");
      }
    }

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
    double detA_bulk_two_thirds = (1.0 + bulk_mod) * (1.0 + bulk_mod);

    double neq_fact = T * T * T / two_pi2_hbarC3;
    double dn_fact = bulkPi / betabulk;
    double J20_fact = T * neq_fact;
    double N10_fact = neq_fact;
    double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

    // determine if feqmod breaks down
    bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, DETA_MIN, z, laguerre, DF_MODE, 0, T, F, betabulk);


    // I need to use my old method (no matrix solver iterations)


    // uniformly rescale eta space by detA if modified momentum space elements are shrunk
    // this rescales the dsigma components orthogonal to the eta direction (only works for 2+1d, y = 0)
    // for integrating modified distribution with narrow (y-eta) distributions
    double eta_scale = 1.0;
    if(detA > DETA_MIN && DIMENSION == 2)
    {
      eta_scale = detA / detA_bulk_two_thirds;
    }

    // compute renormalization factor
    double renorm = 1.0;

    if(INCLUDE_BULK_DELTAF)
    {
      if(DF_MODE == 3)
      {
        double mbar = mass / T;
        double mbar_mod = mass / T_mod;

        // maybe compute these individually on the device
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

    // rescale normalization factor
    if(DIMENSION == 2)
    {
      renorm /= detA_bulk_two_thirds;
    }
    else if(DIMENSION == 3)
    {
      renorm /= detA;
    }



    // loop over momentum
    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      double pT = pT_d[ipT];

      double mT = sqrt(mass_squared  +  pT * pT);
      double mT_over_tau = mT / tau;

      for(long iphip = 0; iphip < phi_tab_length; iphip++)
      {
        double px = pT * trig_d[iphip];
        double py = pT * trig_d[iphip + phi_tab_length];

        double px_dax = px * dax;   // useful expressions
        double py_day = py * day;

        double px_ux = px * ux;
        double py_uy = py * uy;

        double pixx_px_px = pixx * px * px;
        double piyy_py_py = piyy * py * py;
        double pitx_px = pitx * px;
        double pity_py = pity * py;
        double pixy_px_py = pixy * px * py;
        double tau2_pixn_px = tau2_pixn * px;
        double tau2_piyn_py = tau2_piyn * py;

        for(long iy = 0; iy < y_tab_length; iy++)
        {
          double y = y_d[iy];

          // momentum_length index
          long iP = ipT  +  pT_tab_length * (iphip  +  phi_tab_length * iy);

          if(positive_time_volume)
          {
            double eta_integral = 0.0;

            // sum over eta
            for(long ieta = 0; ieta < eta_tab_length; ieta++)
            {
              double eta = etaValues_d[ieta];
              double eta_weight = etaWeights_d[ieta];

              bool feqmod_breaks_down_narrow = false;

              if(DIMENSION == 3 && !feqmod_breaks_down)
              {
                if(detA < 0.01 && fabs(y - eta) < detA)
                {
                  feqmod_breaks_down_narrow = true;
                }
              }

              double sinhyeta = sinh(y - eta);
              double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

              double pt = mT * coshyeta;           // p^tau
              double pn = mT_over_tau * sinhyeta;  // p^eta

              double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

              if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow



              double f;

              // calculate feqmod
              if(feqmod_breaks_down || feqmod_breaks_down_narrow)
              {
                double sinhyeta = sinh(y - eta);
                double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

                double pt = mT * coshyeta;           // p^tau
                double pn = mT_over_tau * sinhyeta;  // p^eta

                double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

                if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                double E = pt * ut  -  px_ux  -  py_uy  -  pn * tau2_un;  // u.p
                double feq = 1.0 / (exp(E/T  -  chem) + sign);
                double feqbar = 1.0  -  sign * feq;

                // pi^munu.p_mu.p_nu
                double pimunu_pmu_pnu = pitt * pt * pt  +  pixx_px_px  +  piyy_py_py  +  tau4_pinn * pn * pn
                    + 2.0 * (-(pitx_px + pity_py) * pt  +  pixy_px_py  +  pn * (tau2_pixn_px  +  tau2_piyn_py  -  tau2_pitn * pt));

                double df;

                if(DF_MODE == 3)
                {
                  double df_shear = shear_coeff * pimunu_pmu_pnu / E;
                  double df_bulk = (bulk0_coeff * E  +  bulk1_coeff  +  bulk2_coeff * (E  -  mass_squared / E));

                  df = feqbar * (df_shear + df_bulk);
                }
                else if(DF_MODE == 4)
                {
                  double df_shear = feqbar * shear_coeff * pimunu_pmu_pnu / E;
                  double df_bulk = bulk0_coeff  +  feqbar * bulk1_coeff * (E  -  mass_squared / E);

                  df = df_shear + df_bulk;
                }

                if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                f = feq * (1.0 + df);

              } // feqmod breaks down
              else
              {
                double sinhyeta_scale = sinh(y - eta_scale * eta);

                double pt = mT * sqrt(1.0  +  sinhyeta_scale * sinhyeta_scale);  // p^\tau (GeV)
                double pn = mT_over_tau *  sinhyeta_scale                        // p^\eta (GeV^2)
                double tau2_pn = tau2 * pn;

                // LRF momentum components pi_LRF = - Xi.p
                double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  tau2_Xn * pn;
                double py_LRF = Yx * px  +  Yy * py;
                double pz_LRF = -Zt * pt  +  tau2_Zn * pn;

                double pLRF[3] = {px_LRF, py_LRF, pz_LRF};

                // this is where I need the old method
                //matrix_multiplication(A_inv, pLRF, pLRF_mod, 3, 3);   // evaluate p_mod = A^-1.p at least once

                double px_LRF_mod = pLRF[0];
                double py_LRF_mod = pLRF[1];
                double pz_LRF_mod = pLRF[2];

                double E_mod = sqrt(mass_squared  +  px_LRF_mod * px_LRF_mod  +  py_LRF_mod * py_LRF_mod  +  pz_LRF_mod * pz_LRF_mod);

                f = fabs(renorm) / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod
              }

              eta_integral += eta_weight * pdotdsigma * f;

            } // ieta

            dN_pTdpTdphidy_thread[ithread] = prefactor * eta_integral;
          }
          else
          {
            dN_pTdpTdphidy_thread[ithread] = 0.0;
          }

          // perform reduction over threads in each block:
          int N = blockDim.x;  // number of threads in block (must be power of 2)
          __syncthreads();     // prepare threads for reduction

          while(N != 1)
          {
            N /= 2;

            if(ithread < N) // reduce thread pairs
            {
              dN_pTdpTdphidy_thread[ithread] += dN_pTdpTdphidy_thread[ithread + N];
            }
            __syncthreads();
          }

          // store block's contribution to the spectra
          if(ithread == 0)
          {
            long iP_block = iP  +  blockIdx.x * momentum_length;

            dN_pTdpTdphidy_d_blocks[iP_block] = dN_pTdpTdphidy_thread[0];
          }

        } // iy

      } // iphip

    } // ipT

  } // icell < endFO

} // end function

*/


// does a block reduction, where the previous kernel did a thread reduction.
__global__ void calculate_dN_pTdpTdphidy_blockReduction(double *dN_pTdpTdphidy_d, double* dN_pTdpTdphidy_d_blocks, long momentum_length, long blocks_ker1, long ipart, long npart)
{
  long ithread = (long)threadIdx.x  +  (long)blockDim.x * (long)blockIdx.x;

  // each thread is assigned a momentum coordinate
  if(ithread < momentum_length)
  {
    long iS3D = ipart + npart * ithread;

    // sum spectra contributions of the blocks from first kernel
    for(long iblock_ker1 = 0; iblock_ker1 < blocks_ker1; iblock_ker1++)
    {
      dN_pTdpTdphidy_d[iS3D] += dN_pTdpTdphidy_d_blocks[ithread  +  iblock_ker1 * momentum_length];
    }
  }
}

// does a block reduction, where the previous kernel did a thread reduction.
__global__ void calculate_dN_dX_blockReduction(double *dN_dX_d, double* dN_dX_d_blocks, long spacetime_length, long blocks_ker1, long ipart, long npart)
{
  long ithread = (long)threadIdx.x  +  (long)blockDim.x * (long)blockIdx.x;

  // each thread is assigned a spacetime coordinate
  if(ithread < spacetime_length)
  {
    long iX = ipart  +  npart * ithread;

    // sum spacetime contributions of the blocks from first kernel
    for(long iblock_ker1 = 0; iblock_ker1 < blocks_ker1; iblock_ker1++)
    {
      dN_dX_d[iX] += dN_dX_d_blocks[ithread  +  iblock_ker1 * spacetime_length];
    }
  }
}



void EmissionFunctionArray::write_dN_2pipTdpTdy_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_2pipTdpTdy to file...\n");

  for(long ipart = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_2pipTdpTdy_%ld.dat", MCID[ipart]);
    ofstream spectra(filename, ios_base::out);

    for(long iy = 0; iy < y_tab_length; iy++)
    {
      double y = 0.0;
      if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        double pT =  pT_tab->get(1, ipT + 1);

        double dN_2pipTdpTdy = 0.0;

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip_weight = phi_tab->get(2, iphip + 1);

          long iS3D = ipart  +  npart * (iy  +  y_tab_length * (iphip  +  phi_tab_length * ipT));

          dN_2pipTdpTdy += phip_weight * dN_pTdpTdphidy[iS3D] / two_pi;

        }

        spectra << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT << "\t" << dN_2pipTdpTdy << "\n";
      }

      if(iy < y_tab_length - 1) spectra << "\n";
    }

    spectra.close();
  }
}

void EmissionFunctionArray::write_vn_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal vn to file...\n");

  const complex<double> I(0.0,1.0);   // imaginary i

  const int k_max = 7;                // v_n = {v_1, ..., v_7}

  for(long ipart = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/vn_%ld.dat", MCID[ipart]);
    ofstream vn_File(filename, ios_base::out);

    for(long iy = 0; iy < y_tab_length; iy++)
    {
      double y = 0.0;
      if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        double pT = pT_tab->get(1, ipT + 1);

        double Vn_real_numerator[k_max];
        double Vn_imag_numerator[k_max];

        for(int k = 0; k < k_max; k++)
        {
          Vn_real_numerator[k] = 0.0;
          Vn_imag_numerator[k] = 0.0;
        }

        double vn_denominator = 0.0;

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1, iphip + 1);
          double phip_weight = phi_tab->get(2, iphip + 1);

          long iS3D = ipart  +  npart * (iy  +  y_tab_length * (iphip  +  phi_tab_length * ipT));

          for(int k = 0; k < k_max; k++)
          {
            Vn_real_numerator[k] += cos(((double)k + 1.0) * phip) * phip_weight * dN_pTdpTdphidy[iS3D];
            Vn_imag_numerator[k] += sin(((double)k + 1.0) * phip) * phip_weight * dN_pTdpTdphidy[iS3D];
          }

          vn_denominator += phip_weight * dN_pTdpTdphidy[iS3D];
        } 

        vn_File << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT;

        for(long k = 0; k < k_max; k++)
        {
          double vn = abs(Vn_real_numerator[k]  +  I * Vn_imag_numerator[k]) / vn_denominator;

          if(vn_denominator < 1.e-15) vn = 0.0;

          vn_File << "\t" << vn;
        }

        vn_File << "\n";
      }

       if(iy < y_tab_length - 1) vn_File << "\n";
    } 

    vn_File.close();
  }
}


void EmissionFunctionArray::write_dN_dphidy_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_dphidy to file...\n");

  for(long ipart  = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_dphipdy_%ld.dat", MCID[ipart]);
    ofstream spectra(filename, ios_base::app);

    for(long iy = 0; iy < y_tab_length; iy++)
    {
      double y = 0.0;
      if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

      for(long iphip = 0; iphip < phi_tab_length; iphip++)
      {
        double phip = phi_tab->get(1,iphip + 1);

        double dN_dphipdy = 0.0;

        for(long ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT_weight = pT_tab->get(2, ipT + 1);

          long iS3D = ipart  +  npart * (iy  +  y_tab_length * (iphip  +  phi_tab_length * ipT));

          dN_dphipdy += pT_weight * dN_pTdpTdphidy[iS3D];
        }

        spectra << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << dN_dphipdy << "\n";
      }

      if(iy < y_tab_length - 1) spectra << "\n";
    }

    spectra.close();
  }
}



void EmissionFunctionArray::write_dN_dy_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_dy to file...\n");

  for(long ipart  = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_dy_%ld.dat", MCID[ipart]);
    ofstream spectra(filename, ios_base::out);

    for(long iy = 0; iy < y_tab_length; iy++)
    {
      double y = 0.0;
      if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

      double dN_dy = 0.0;

      for(long iphip = 0; iphip < phi_tab_length; iphip++)
      {
        double phip_weight = phi_tab->get(2, iphip + 1);

        for(long ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT_weight = pT_tab->get(2, ipT + 1);

          long iS3D = ipart  +  npart * (ipT  +  pT_tab_length * (iphip  +  phi_tab_length * iy));

          dN_dy += phip_weight * pT_weight * dN_pTdpTdphidy[iS3D];

        } 
      }

      spectra << setw(5) << setprecision(8) << y << "\t" << dN_dy << "\n";
    } 

    spectra.close();
  }
}

void EmissionFunctionArray::write_dN_taudtaudeta_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_taudtaudeta to file...\n");

  for(long ipart = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_taudtaudeta_%ld.dat", MCID[ipart]);
    ofstream dN_taudtaudeta(filename, ios_base::out);

    for(long ieta = 0; ieta < eta_bins; ieta++)
    {
      double eta_mid = 0.0;

      if(DIMENSION == 3) eta_mid = eta_min  +  eta_width * (ieta + 0.5);

      for(long itau = 0; itau < tau_bins; itau++)
      {
        double tau_mid = tau_min  +  tau_width * (itau + 0.5);

        long iX = ipart  +  npart * (ieta  +  eta_bins * itau);

        dN_taudtaudeta << setprecision(6) << scientific << eta_mid << "\t" << tau_mid << "\t" << dN_dX[iX] / (tau_mid * tau_width * eta_width) << "\n";
      }

      if(ieta < eta_bins - 1) dN_taudtaudeta << "\n";
    } 

    dN_taudtaudeta.close();
  }
}

void EmissionFunctionArray::write_dN_2pirdrdeta_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_2pirdrdeta to file...\n");

  for(long ipart = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_2pirdrdeta_%ld.dat", MCID[ipart]);
    ofstream dN_2pirdrdeta(filename, ios_base::out);

    for(long ieta = 0; ieta < eta_bins; ieta++)
    {
      double eta_mid = 0.0;

      if(DIMENSION == 3) eta_mid = eta_min  +  eta_width * (ieta + 0.5);

      for(long ir = 0; ir < r_bins; ir++)
      {
        double r_mid = r_min  +  r_width * (ir + 0.5);

        long iX = ipart  +  npart * (ieta  +  eta_bins * (ir + tau_bins));

        dN_2pirdrdeta << setprecision(6) << scientific << eta_mid << "\t" << r_mid << "\t" << dN_dX[iX] / (two_pi * r_mid * r_width * eta_width) << "\n";
      }

      if(ieta < eta_bins - 1) dN_2pirdrdeta << "\n";
    } 

    dN_2pirdrdeta.close();
  }
}

void EmissionFunctionArray::write_dN_dphideta_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_dphideta to file...\n");

  for(long ipart = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_dphideta_%ld.dat", MCID[ipart]);
    ofstream dN_dphideta(filename, ios_base::out);

    for(long ieta = 0; ieta < eta_bins; ieta++)
    {
      double eta_mid = 0.0;

      if(DIMENSION == 3) eta_mid = eta_min  +  eta_width * (ieta + 0.5);

      for(long iphi = 0; iphi < phi_bins; iphi++)
      {
        double phi_mid = phi_width * (iphi + 0.5);

        long iX = ipart  +  npart * (ieta  +  eta_bins * (iphi + tau_bins + r_bins));

        dN_dphideta << setprecision(6) << scientific << eta_mid << "\t" << phi_mid << "\t" << dN_dX[iX] / (phi_width * eta_width) << "\n";
      }

      if(ieta < eta_bins - 1) dN_dphideta << "\n";
    } 

    dN_dphideta.close();
  }
}




void EmissionFunctionArray::calculate_spectra()
{
  cout << "calculate_spectra() has started... " << endl;
  Stopwatch sw;
  sw.tic();

  cudaDeviceSynchronize();    // test synchronize device
  cudaError_t err;            // errors

  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error at very beginning: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  double h3 = pow(2.0 * M_PI * hbarC, 3);

  long chunks = (FO_length + FO_chunk - 1) / FO_chunk;                    // number of chunks
  long partial = FO_length % FO_chunk;                                    // remainder cells

  long blocks_ker1 = (FO_chunk + threadsPerBlock - 1) / threadsPerBlock;  // number of blocks in kernel 1 (thread reduction)
  long blocks_ker2;                                                       // number of blocks in kernel 2 (block reduction)
  long block_length;

  if(OPERATION == 0)  // spacetime distribution
  {
    blocks_ker2 = (spacetime_length + threadsPerBlock - 1) / threadsPerBlock; 
    block_length = blocks_ker1 * spacetime_length;
  } 
  else if(OPERATION == 1) // spectra
  {
    blocks_ker2 = (momentum_length + threadsPerBlock - 1) / threadsPerBlock;
    block_length = blocks_ker1 * momentum_length;
  }


  cout << "________________________|______________" << endl;
  cout << "chosen_particles        |\t" << npart << endl;
  cout << "pT_tab_length           |\t" << pT_tab_length << endl;
  cout << "phi_tab_length          |\t" << phi_tab_length << endl;
  cout << "y_tab_length            |\t" << y_tab_length << endl;
  cout << "eta_tab_length          |\t" << eta_tab_length << endl;
  cout << "momentum length         |\t" << momentum_length << endl;
  cout << "total spectra length    |\t" << spectra_length << endl;
  cout << "tau_bins                |\t" << tau_bins << endl;
  cout << "r_bins                  |\t" << r_bins << endl;
  cout << "phi_bins                |\t" << phi_bins << endl;
  cout << "eta_bins                |\t" << eta_bins << endl;
  cout << "spacetime length        |\t" << spacetime_length << endl;
  cout << "total spacetime length  |\t" << X_length << endl;  
  cout << "block length            |\t" << block_length << endl;
  cout << "freezeout cells         |\t" << FO_length << endl;
  cout << "chunk size              |\t" << FO_chunk << endl;
  cout << "number of chunks        |\t" << chunks << endl;
  cout << "remainder cells         |\t" << partial << endl;
  cout << "blocks in first kernel  |\t" << blocks_ker1 << endl;
  cout << "blocks in second kernel |\t" << blocks_ker2 << endl;
  cout << "threads per block       |\t" << threadsPerBlock << endl;


  cout << "Declaring and filling host arrays with particle and freezeout surface info" << endl;

  // particle info
  particle_info *particle;
  double *Mass = (double*)calloc(npart, sizeof(double));
  double *Sign = (double*)calloc(npart, sizeof(double));
  double *Degen = (double*)calloc(npart, sizeof(double));
  double *Baryon = (double*)calloc(npart, sizeof(double));
  long *MCID = (long*)calloc(npart, sizeof(long));

  for(long ipart = 0; ipart < npart; ipart++)
  {
    long pdg_index = chosen_particles_table[ipart];
    particle = &particles[pdg_index];

    Mass[ipart] = particle->mass;
    Sign[ipart] = particle->sign;
    Degen[ipart] = particle->gspin;
    Baryon[ipart] = particle->baryon;
    MCID[ipart] = particle->mc_id;
  }


  // freezeout surface info
  FO_surf *surf;
  double *T, *alphaB;                         // thermodynamic properties
  double *tau, *x, *y, *eta;                  // position
  double *ux, *uy, *un;                       // u^mu
  double *dat, *dax, *day, *dan;              // dsigma_mu
  double *pixx, *pixy, *pixn, *piyy, *piyn;   // pi^munu
  double *bulkPi;                             // bulk pressure
  double *Vx, *Vy, *Vn;                       // V^mu

  deltaf_coefficients *df_coeff;

  // callocate memory
  T = (double*)calloc(FO_chunk, sizeof(double));

  tau = (double*)calloc(FO_chunk, sizeof(double));
  x = (double*)calloc(FO_chunk, sizeof(double));
  y = (double*)calloc(FO_chunk, sizeof(double));
  if(DIMENSION == 3)
  {
    eta = (double*)calloc(FO_chunk, sizeof(double));
  }

  ux = (double*)calloc(FO_chunk, sizeof(double));
  uy = (double*)calloc(FO_chunk, sizeof(double));
  un = (double*)calloc(FO_chunk, sizeof(double));

  dat = (double*)calloc(FO_chunk, sizeof(double));
  dax = (double*)calloc(FO_chunk, sizeof(double));
  day = (double*)calloc(FO_chunk, sizeof(double));
  dan = (double*)calloc(FO_chunk, sizeof(double));

  if(INCLUDE_SHEAR_DELTAF)
  {
    pixx = (double*)calloc(FO_chunk, sizeof(double));
    pixy = (double*)calloc(FO_chunk, sizeof(double));
    pixn = (double*)calloc(FO_chunk, sizeof(double));
    piyy = (double*)calloc(FO_chunk, sizeof(double));
    piyn = (double*)calloc(FO_chunk, sizeof(double));
  }

  if(INCLUDE_BULK_DELTAF)
  {
    bulkPi = (double*)calloc(FO_chunk, sizeof(double));
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    alphaB = (double*)calloc(FO_chunk, sizeof(double));  // muB / T
    Vx = (double*)calloc(FO_chunk, sizeof(double));
    Vy = (double*)calloc(FO_chunk, sizeof(double));
    Vn = (double*)calloc(FO_chunk, sizeof(double));
  }

  df_coeff = (deltaf_coefficients*)calloc(FO_chunk, sizeof(deltaf_coefficients));

  // set up momentum tables to pass to GPU
  double *pT = (double*)calloc(pT_tab_length, sizeof(double));
  double *trig = (double*)calloc(2 * phi_tab_length, sizeof(double));  // {cos(phip), sin(phip)}
  double *yp = (double*)calloc(y_tab_length, sizeof(double));
  double *etaValues = (double*)calloc(eta_tab_length, sizeof(double));
  double *etaWeights = (double*)calloc(eta_tab_length, sizeof(double));
  double *y_minus_eta = (double*)calloc(y_minus_eta_tab_length, sizeof(double));

  double *pT_weight = (double*)calloc(pT_tab_length, sizeof(double));
  double *phip_weight = (double*)calloc(phi_tab_length, sizeof(double));
  double *y_minus_eta_weight = (double*)calloc(y_minus_eta_tab_length, sizeof(double));

  for(long ipT = 0; ipT < pT_tab_length; ipT++)
  {
    pT[ipT] = pT_tab->get(1, ipT + 1);

    if(OPERATION == 0) 
    {
      pT_weight[ipT] = pT_tab->get(2, ipT + 1);
    }
  }

  for(long iphip = 0; iphip < phi_tab_length; iphip++)
  {
    double phip = phi_tab->get(1, iphip + 1);

    trig[iphip] = cos(phip);
    trig[iphip + phi_tab_length] = sin(phip);

    if(OPERATION == 0)
    {
      phip_weight[iphip] = phi_tab->get(2, iphip + 1);
    }
  }

  if(DIMENSION == 2)
  {
    yp[0] = 0.0;
    for(long ieta = 0; ieta < eta_tab_length; ieta++)
    {
      etaValues[ieta] = eta_tab->get(1, ieta + 1);
      etaWeights[ieta] = eta_tab->get(2, ieta + 1);
    }
  }
  else if(DIMENSION == 3)
  {
    etaValues[0] = 0.0;
    etaWeights[0] = 1.0;
    for(long iy = 0; iy < y_tab_length; iy++)
    {
      yp[iy] = y_tab->get(1, iy + 1);
    }
  }

  for(long iyeta = 0; iyeta < y_minus_eta_tab_length; iyeta++)
  {
    y_minus_eta[iyeta] = eta_tab->get(1, iyeta + 1);
    y_minus_eta_weight[iyeta] = eta_tab->get(2, iyeta + 1);
  }


  cout << "Declaring and allocating device arrays " << endl;

  double *T_d, *alphaB_d;
  double *tau_d, *x_d, *y_d, *eta_d;
  double *ux_d, *uy_d, *un_d;
  double *dat_d, *dax_d, *day_d, *dan_d;
  double *pixx_d, *pixy_d, *pixn_d, *piyy_d, *piyn_d;
  double *bulkPi_d;
  double *Vx_d, *Vy_d, *Vn_d;

  deltaf_coefficients *df_coeff_d;

  double *pT_d, *trig_d, *yp_d, *etaValues_d, *etaWeights_d, *y_minus_eta_d;
  double *pT_weight_d, *phip_weight_d, *y_minus_eta_weight_d;

  double *dN_pTdpTdphidy_d, *dN_dX_d;
  double *dN_pTdpTdphidy_d_blocks, *dN_dX_d_blocks;



  // allocate memory on device
  cudaMalloc((void**) &T_d, FO_chunk * sizeof(double));

  cudaMalloc((void**) &tau_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &x_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &y_d, FO_chunk * sizeof(double));
  if(DIMENSION == 3)
  {
    cudaMalloc((void**) &eta_d, FO_chunk * sizeof(double));
  }

  cudaMalloc((void**) &ux_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &uy_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &un_d, FO_chunk * sizeof(double));

  cudaMalloc((void**) &dat_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &dax_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &day_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &dan_d, FO_chunk * sizeof(double));

  if(INCLUDE_SHEAR_DELTAF)
  {
    cudaMalloc((void**) &pixx_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &pixy_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &pixn_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &piyy_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &piyn_d, FO_chunk * sizeof(double));
  }

  if(INCLUDE_BULK_DELTAF)
  {
    cudaMalloc((void**) &bulkPi_d, FO_chunk * sizeof(double));
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    cudaMalloc((void**) &alphaB_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &Vx_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &Vy_d, FO_chunk * sizeof(double));
    cudaMalloc((void**) &Vn_d, FO_chunk * sizeof(double));
  }

  cudaMalloc((void**) &df_coeff_d, FO_chunk * sizeof(deltaf_coefficients));

  cudaMalloc((void**) &pT_d,  pT_tab_length * sizeof(double));
  cudaMalloc((void**) &trig_d, 2 * phi_tab_length * sizeof(double));
  cudaMalloc((void**) &yp_d, y_tab_length * sizeof(double));
  cudaMalloc((void**) &etaValues_d, eta_tab_length * sizeof(double));
  cudaMalloc((void**) &etaWeights_d, eta_tab_length * sizeof(double));
  cudaMalloc((void**) &y_minus_eta_d, y_minus_eta_tab_length * sizeof(double));

  cudaMalloc((void**) &pT_weight_d,  pT_tab_length * sizeof(double));
  cudaMalloc((void**) &phip_weight_d, phi_tab_length * sizeof(double));
  cudaMalloc((void**) &y_minus_eta_weight_d, y_minus_eta_tab_length * sizeof(double));

  if(OPERATION == 0)
  {
    cudaMalloc((void**) &dN_dX_d, X_length * sizeof(double));
    cudaMalloc((void**) &dN_dX_d_blocks, block_length * sizeof(double));
  }
  else if(OPERATION == 1)
  {
    cudaMalloc((void**) &dN_pTdpTdphidy_d, spectra_length * sizeof(double));
    cudaMalloc((void**) &dN_pTdpTdphidy_d_blocks, block_length * sizeof(double));
  }
  

  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error in device memory allocation: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  printf("Copying momentum tables from host to device...\n");
  cudaMemcpy(pT_d, pT, pT_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(trig_d, trig, 2 * phi_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(yp_d, yp, y_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(etaValues_d, etaValues, eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(etaWeights_d, etaWeights, eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_minus_eta_d, y_minus_eta, y_minus_eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(pT_weight_d, pT_weight, pT_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(phip_weight_d, phip_weight, phi_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_minus_eta_weight_d, y_minus_eta_weight, y_minus_eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error in memory copy from host to device: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  if(OPERATION == 0)
  {
    cudaMemset(dN_dX_d, 0.0, X_length * sizeof(double));
  }
  else
  {
    cudaMemset(dN_pTdpTdphidy_d, 0.0, spectra_length * sizeof(double));
  }

  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error in device memory set: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }


  printf("\n");
  

  for(long n = 0; n < chunks; n++)
  {
    long endFO = FO_chunk;

    if((n == chunks - 1) && partial > 0) endFO = partial;

    // get freezeout surface info for the chunk
    for(long icell = 0; icell < endFO; icell++)
    {
      long icell_glb = icell  +  n * FO_chunk;   // global cell index

      surf = &surf_ptr[icell_glb];

      T[icell] = surf->T;
   
      tau[icell] = surf->tau;
      x[icell] = surf->x;
      y[icell] = surf->y;
      if(DIMENSION == 3)
      {
        eta[icell] = surf->eta;
      }

      ux[icell] = surf->ux;
      uy[icell] = surf->uy;
      un[icell] = surf->un;

      dat[icell] = surf->dat;
      dax[icell] = surf->dax;
      day[icell] = surf->day;
      dan[icell] = surf->dan;

      if(INCLUDE_SHEAR_DELTAF)
      {
        pixx[icell] = surf->pixx;
        pixy[icell] = surf->pixy;
        pixn[icell] = surf->pixn;
        piyy[icell] = surf->piyy;
        piyn[icell] = surf->piyn;
      }

      if(INCLUDE_BULK_DELTAF)
      {
        bulkPi[icell] = surf->bulkPi;
      }

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        alphaB[icell] = (surf->muB) / (surf->T);
        Vx[icell] = surf->Vx;
        Vy[icell] = surf->Vy;
        Vn[icell] = surf->Vn;
      }

      // evaluate the df coefficients
      double T_FO = T[icell];
      double P = surf->P;
      double E = surf->E;
      double muB = 0.0;
      double nB = 0.0;
      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        muB = T_FO * alphaB[icell];
        nB = surf->nB;
      }
      double bulkPi_FO = 0.0;
      if(INCLUDE_BULK_DELTAF)
      {
        bulkPi_FO = bulkPi[icell];
      }

      df_coeff[icell] = df_data->evaluate_df_coefficients(T_FO, muB, E, P, nB, bulkPi_FO);

    } // icell


    // copy memory from host to device
    cudaMemcpy(T_d, T, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(tau_d, tau, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    if(DIMENSION == 3)
    {
      cudaMemcpy(eta_d, eta, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(ux_d, ux, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(uy_d, uy, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(un_d, un, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(dat_d, dat, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dax_d, dax, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(day_d, day, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dan_d, dan, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);

    if(INCLUDE_SHEAR_DELTAF)
    {
      cudaMemcpy(pixx_d, pixx, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(pixy_d, pixy, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(pixn_d, pixn, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(piyy_d, piyy, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(piyn_d, piyn, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    }

    if(INCLUDE_BULK_DELTAF)
    {
      cudaMemcpy(bulkPi_d, bulkPi, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    }

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      cudaMemcpy(alphaB_d, alphaB, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(Vx_d, Vx, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(Vy_d, Vy, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(Vn_d, Vn, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(df_coeff_d, df_coeff, FO_chunk * sizeof(deltaf_coefficients), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
      printf("Error in memory copy from host to device: %s\n", cudaGetErrorString(err));
      err = cudaSuccess;
    }

    // loop over particles
    for(int ipart = 0; ipart < npart; ipart++)
    {
      // reset block spectra to zero
      if(OPERATION == 0)
      {
        cudaMemset(dN_dX_d_blocks, 0.0, block_length * sizeof(double));
      }
      else if(OPERATION == 1)
      {
        cudaMemset(dN_pTdpTdphidy_d_blocks, 0.0, block_length * sizeof(double));
      }

      double mass = Mass[ipart];
      double sign = Sign[ipart];
      double degen = Degen[ipart];
      double baryon = Baryon[ipart];

      double mass_squared = mass * mass;
      double prefactor = degen / h3;

      switch(DF_MODE)
      {
        case 1:
        case 2:
        {
          // launch thread reduction kernel
          if(OPERATION == 0)
          {
            cudaDeviceSynchronize();
            calculate_dN_dX_threadReduction<<<blocks_ker1, threadsPerBlock>>>(dN_dX_d_blocks, endFO, tau_bins, r_bins, phi_bins, eta_bins, tau_min, r_min, eta_min, tau_width, r_width, phi_width, eta_width, pT_tab_length, phi_tab_length, y_minus_eta_tab_length, pT_d, pT_weight_d, trig_d, phip_weight_d, y_minus_eta_d, y_minus_eta_weight_d, mass_squared, sign, prefactor, baryon, T_d, tau_d, x_d, y_d, eta_d, ux_d, uy_d, un_d, dat_d, dax_d, day_d, dan_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, bulkPi_d, alphaB_d, Vx_d, Vy_d, Vn_d, df_coeff_d, INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DIMENSION, OUTFLOW, DF_MODE);
            cudaDeviceSynchronize();
          }
          else if(OPERATION == 1)
          {
            cudaDeviceSynchronize();
            calculate_dN_pTdpTdphidy_threadReduction<<<blocks_ker1, threadsPerBlock, sizeof(double)*threadsPerBlock>>>(dN_pTdpTdphidy_d_blocks, endFO, momentum_length, pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length, pT_d, trig_d, yp_d, etaValues_d, etaWeights_d, mass_squared, sign, prefactor, baryon, T_d, tau_d, eta_d, ux_d, uy_d, un_d, dat_d, dax_d, day_d, dan_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, bulkPi_d, alphaB_d, Vx_d, Vy_d, Vn_d, df_coeff_d, INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DIMENSION, OUTFLOW, DF_MODE);
            cudaDeviceSynchronize();
          }
          
          break;
        }
        case 3:
        case 4:
        {
          // launch thread reduction kernel
          //cudaDeviceSynchronize();
          //calculate_dN_pTdpTdphidy_feqmod_threadReduction<<<blocks_ker1, threadsPerBlock>>>(dN_pTdpTdphidy_d_blocks, endFO, momentum_length, pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length, pT_d, trig_d, yp_d, etaValues_d, etaWeights_d, mass_squared, sign, prefactor, baryon, T_d, P_d, E_d, tau_d, eta_d, ux_d, uy_d, un_d, dat_d, dax_d, day_d, dan_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, bulkPi_d, alphaB_d, nB_d, Vx_d, Vy_d, Vn_d, df_coeff_d, INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DIMENSION, OUTFLOW, DF_MODE);
          //cudaDeviceSynchronize();

          break;
        }
      }

      err = cudaGetLastError();
      if(err != cudaSuccess)
      {
        printf("Error in thread reduction kernel: %s\n", cudaGetErrorString(err));
        err = cudaSuccess;
      }

      // launch block reduction kernel
      if(OPERATION == 0)
      {
        cudaDeviceSynchronize();
        calculate_dN_dX_blockReduction<<<blocks_ker2, threadsPerBlock>>>(dN_dX_d, dN_dX_d_blocks, spacetime_length, blocks_ker1, ipart, npart);
        cudaDeviceSynchronize();
      }
      else if(OPERATION == 1)
      {
        cudaDeviceSynchronize();
        calculate_dN_pTdpTdphidy_blockReduction<<<blocks_ker2, threadsPerBlock>>>(dN_pTdpTdphidy_d, dN_pTdpTdphidy_d_blocks, momentum_length, blocks_ker1, ipart, npart);
        cudaDeviceSynchronize();
      }
    
      err = cudaGetLastError();
      if(err != cudaSuccess)
      {
        printf("Error in block reduction kernel: %s\n", cudaGetErrorString(err));
        err = cudaSuccess;
      }

    } // ipart

    printf("Progress: finished chunk %ld of %ld", n + 1, chunks);
    printf("\n");

  } // loop over chunks


  // copy results from device to host and write to file
  if(OPERATION == 0)
  {
    cudaMemcpy(dN_dX, dN_dX_d, X_length * sizeof(double), cudaMemcpyDeviceToHost);

    write_dN_taudtaudeta_toFile(MCID);
    write_dN_2pirdrdeta_toFile(MCID);
    write_dN_dphideta_toFile(MCID);

    free(dN_dX);
  }
  else if(OPERATION == 1)
  {
    cudaMemcpy(dN_pTdpTdphidy, dN_pTdpTdphidy_d, spectra_length * sizeof(double), cudaMemcpyDeviceToHost);

    write_dN_2pipTdpTdy_toFile(MCID);
    write_vn_toFile(MCID);
    //write_dN_dphidy_toFile(MCID);
    //write_dN_dy_toFile(MCID);

    free(dN_pTdpTdphidy);
  }
 
  cout << "Deallocating host and device memory" << endl;  
  free(chosen_particles_table);

  free(Mass);
  free(Sign);
  free(Degen);
  free(Baryon);
  free(MCID);

  free(T);

  free(tau);
  free(x);
  free(y);
  if(DIMENSION == 3) free(eta);

  free(ux);
  free(uy);
  free(un);

  free(dat);
  free(dax);
  free(day);
  free(dan);

  if(INCLUDE_SHEAR_DELTAF)
  {
    free(pixx);
    free(pixy);
    free(pixn);
    free(piyy);
    free(piyn);
  }

  if(INCLUDE_BULK_DELTAF)
  {
    free(bulkPi);
  }


  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    free(alphaB);
    free(Vx);
    free(Vy);
    free(Vn);
  }

  free(pT);
  free(trig);
  free(yp);
  free(etaValues);
  free(etaWeights);
  free(y_minus_eta);
  free(pT_weight);
  free(phip_weight);
  free(y_minus_eta_weight);

  free(df_coeff);

  // cudaFree = deallocate memory on device
  cudaFree(T_d);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(tau_d);
  if(DIMENSION == 3)
  {
    cudaFree(eta_d);
  }

  cudaFree(ux_d);
  cudaFree(uy_d);
  cudaFree(un_d);

  cudaFree(dat_d);
  cudaFree(dax_d);
  cudaFree(day_d);
  cudaFree(dan_d);

  if(INCLUDE_SHEAR_DELTAF)
  {
    cudaFree(pixx_d);
    cudaFree(pixy_d);
    cudaFree(pixn_d);
    cudaFree(piyy_d);
    cudaFree(piyn_d);
  }

  if(INCLUDE_BULK_DELTAF)
  {
    cudaFree(bulkPi_d);
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    cudaFree(alphaB_d);
    cudaFree(Vx_d);
    cudaFree(Vy_d);
    cudaFree(Vn_d);
  }

  cudaFree(df_coeff_d);

  cudaFree(pT_d);
  cudaFree(trig_d);
  cudaFree(yp_d);
  cudaFree(etaValues_d);
  cudaFree(etaWeights_d);

  if(OPERATION == 0)
  {
    cudaFree(dN_dX_d);
    cudaFree(dN_dX_d_blocks);
  }
  else if(OPERATION == 1)
  {
    cudaFree(dN_pTdpTdphidy_d);
    cudaFree(dN_pTdpTdphidy_d_blocks);
  }

  sw.toc();
  cout << "\nKernel launches took " << sw.takeTime() << " seconds.\n" << endl;
}


