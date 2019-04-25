#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
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

#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs
#define threadsPerBlock 512 //try optimizing this, also we can define two different threads/block for the separate kernels
#define FO_chunk 10000
#define debug 1	//1 for debugging, 0 otherwise

using namespace std;

// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in, deltaf_coefficients df_in)
{
  // control parameters
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


  // particle info
  particles = particles_in;
  Nparticles = Nparticles_in;
  npart = chosen_particles_in->getNumberOfRows();
  chosen_particles_table = new int[npart];

  for(int ipart = 0; ipart < npart; ipart++)
  {
    int mc_id = chosen_particles_in->get(1, ipart + 1);
    bool found_match = false;

    // search for match in PDG file
    for(int iPDG = 0; iPDG < Nparticles; iPDG++)
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
      printf("Chosen particles error: %d not found in pdg.dat\n", mc_id);
      exit(-1);
    }
  }


  // df coefficients
  df = df_in;


  // particle spectra of chosen particle species
  spectra_length = npart * pT_tab_length * phi_tab_length * y_tab_length;
  dN_pTdpTdphidy = new double[spectra_length];
  for(long iSpectra = 0; iSpectra < spectra_length; iSpectra++) dN_pTdpTdphidy[iSpectra] = 0.0;

}


EmissionFunctionArray::~EmissionFunctionArray()
{
  delete[] chosen_particles_table;
  delete[] dN_pTdpTdphidy;
}

__global__ void calculate_dN_pTdpTdphidy(long FO_length, long npart, long pT_tab_length, long phi_tab_length, long y_tab_length, long eta_tab_length, double* dN_pTdpTdphidy_d_blocks, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaWeights_d, double *Mass_d, double *Sign_d, double *Degen_d, double *Baryon_d, double *T_d, double *P_d, double *E_d, double *tau_d, double *eta_d, double *ux_d, double *uy_d, double *un_d, double *dat_d, double *dax_d, double *day_d, double *dan_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *bulkPi_d, double *alphaB_d, double *nB_d, double *Vx_d, double *Vy_d, double *Vn_d, double prefactor, double *df_coeff_d, int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION, int OUTFLOW, int DF_MODE)
  {
    long number_of_chunks = (FO_length / FO_chunk) + 1;
    long spectra_length = npart * pT_tab_length * phi_tab_length * y_tab_length;

    const int Nthreads = blockDim.x;                      // number of threads in block (must be power of 2)

    // contribution of dN_pTdpTdphidy from each thread
    __shared__ double dN_pTdpTdphidy_thread[threadsPerBlock];

    int icell = threadIdx.x  +  Nthreads * blockIdx.x;    // thread index in kernel
    int idx_local = threadIdx.x;                          // thread index in block

    __syncthreads();                                      // sync threads for each block

    // loop over spectra
    for(long ipart = 0; ipart < npart; ipart++)
    {
      double mass = Mass_d[ipart];
      double sign = Sign_d[ipart];
      double degeneracy = Degen_d[ipart];
      double baryon = Baryon_d[ipart];
      double mass2 = mass * mass;
      double constant = prefactor * degeneracy;

      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        double pT = pT_d[ipT];
        double mT = sqrt(mass2  +  pT * pT);

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double px = pT * trig_d[iphip + phi_tab_length];
          double py = pT * trig_d[iphip];

          for(long iy = 0; iy < y_tab_length; iy++)
          {
            double y = y_d[iy];

            // spectra linear column index
            long iS3D = ipart  +  npart * (ipT  +  pT_tab_length * (iphip  +  phi_tab_length * iy));

            dN_pTdpTdphidy_thread[idx_local] = 0.0;     // reset thread contribution

            // loop over chunks
            for(long n = 0; n < number_of_chunks; n++)
            {
              // set the chunk size
              long endFO = FO_chunk;
              if(n == number_of_chunks - 1) endFO = FO_length  -  n * endFO;


              double dN_pTdpTdphidy_cell = 0.0;         // contribution of an individual cell


              // only do calculation for chunk cells (some threads are extraneous)
              if(icell < endFO)
              {
                // global freezeout cell index
                long icell_glb = (long) icell + n * FO_chunk;

                double tau = tau_d[icell_glb];          // longitudinal proper time
                double tau2 = tau * tau;
                double mT_over_tau = mT / tau;

                if(DIMENSION == 3)
                {
                  etaValues_d[0] = eta_d[icell_glb];    // spacetime rapidity from surface
                }
                double dat = dat_d[icell_glb];          // dsigma_mu
                double dax = dax_d[icell_glb];
                double day = day_d[icell_glb];
                double dan = dan_d[icell_glb];

                double ux = ux_d[icell_glb];            // u^mu
                double uy = uy_d[icell_glb];            // enforce normalization
                double un = un_d[icell_glb];
                double ut = sqrt(1.0 +  ux * ux  +  uy * uy  +  tau2 * un * un);

                double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;

                if(udsigma <= 0.0) break;               // skip cells with u.dsigma < 0

                double ux2 = ux * ux;                   // useful expressions
                double uy2 = uy * uy;
                double ut2 = ut * ut;
                double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

                double T = T_d[icell_glb];              // temperature
                double E = E_d[icell_glb];              // energy density
                double P = P_d[icell_glb];              // pressure

                double pitt = 0.0;                      // pi^munu
                double pitx = 0.0;                      // enforce orthogonality, tracelessness
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
                  pixx = pixx_d[icell_glb];
                  pixy = pixy_d[icell_glb];
                  pixn = pixn_d[icell_glb];
                  piyy = piyy_d[icell_glb];
                  piyn = piyn_d[icell_glb];
                  pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
                  pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
                  pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
                  pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
                  pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
                }

                double bulkPi = 0.0;                    // bulk pressure

                if(INCLUDE_BULK_DELTAF)
                {
                  bulkPi = bulkPi_d[icell_glb];
                }

                double muB = 0.0;                       // baryon chemical potential
                double nB = 0.0;                        // net baryon density
                double Vt = 0.0;                        // V^mu
                double Vx = 0.0;                        // enforce orthogonality V.u = 0
                double Vy = 0.0;
                double Vn = 0.0;
                double alphaB = 0.0;
                double chem = 0.0;
                double baryon_enthalpy_ratio = 0.0;

                if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                {
                  alphaB = alphaB_d[icell_glb];
                  chem = baryon * alphaB;
                  nB = nB_d[icell_glb];
                  Vx = Vx_d[icell_glb];
                  Vy = Vy_d[icell_glb];
                  Vn = Vn_d[icell_glb];
                  Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

                  alphaB = muB / T;
                  baryon_enthalpy_ratio = nB / (E + P);
                }

                // df coefficients
                //double shear_coeff = 0.5 / (T * T * (E + P));  // (see df_shear)
                double shear_coeff = 0.0;

                double pdotdsigma_f_eta_sum = 0.0; // eta integral

                // sum over eta
                for(long ieta = 0; ieta < eta_tab_length; ieta++)
                {
                  double eta = etaValues_d[ieta];
                  double eta_weight = etaWeights_d[ieta];

                  double ptau = mT * cosh(y - eta);           // p^tau
                  double pn = mT_over_tau * sinh(y - eta);    // p^eta
                  double tau2_pn = tau2 * pn;

                  double pdotdsigma = eta_weight * (ptau * dat  +  px * dax  +  py * day  +  pn * dan);

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                  double pdotu = ptau * ut  -  px * ux  -  py * uy  -  tau2_pn * un;  // u.p
                  double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                  double feqbar = 1.0  -  sign * feq;

                  // pi^munu.p_mu.p_nu (double check)
                  double pimunu_pmu_pnu = pitt * ptau * ptau  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                  + 2.0 * (-(pitx * px  +  pity * py) * ptau  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * ptau));
                  // V^mu.p_mu
                  double Vmu_pmu = Vt * ptau  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;


                  double df;  // compute df correction
                  switch(DF_MODE)
                  {
                    case 1: // 14 moment
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu;
                      double df_bulk = 0.0;
                      double df_diff = 0.0;
                      //double df_bulk = (bulk0_coeff * mass2  +  (bulk1_coeff * baryon  +  bulk2_coeff * pdotu) * pdotu) * bulkPi;
                      //double df_diff = (c3 * baryon  +  c4 * pdotu) * Vmu_pmu;

                      df = feqbar * (df_shear + df_bulk + df_diff);
                      break;
                    }
                    case 2: // Chapman enskog
                    {
                      double df_shear = 0.0;
                      double df_bulk = 0.0;
                      double df_diff = 0.0;
                      //double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      //double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                      //double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;

                      df = feqbar * (df_shear + df_bulk + df_diff);
                      break;
                    }
                    default:
                    {
                      printf("Error: set df_mode = (1,2) in parameters.dat\n"); //exit(-1);
                    }
                  } // DF_MODE

                  if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                  pdotdsigma_f_eta_sum += pdotdsigma * feq * (1.0 + df);

                } // ieta

                dN_pTdpTdphidy_cell = constant * pdotdsigma_f_eta_sum;

              } // icell < endFO

              // add contribution of cells assigned to thread
              dN_pTdpTdphidy_thread[idx_local] += dN_pTdpTdphidy_cell;

            } // add chunk cells



            // perform reduction over threads in block:
            //::::::::::::::::::::::::::::::::::::::::::::::::::::::

            int N = Nthreads;    // number of threads in block (must be power of 2)

            __syncthreads();     // prepare threads for reduction

            while(N != 1)
            {
              N /= 2;

              // reduce thread pairs
              if(idx_local < N)
              {
                dN_pTdpTdphidy_thread[idx_local] += dN_pTdpTdphidy_thread[idx_local + N];
              }

              __syncthreads();   // test if this is needed
            }

            //::::::::::::::::::::::::::::::::::::::::::::::::::::::


            // will you accidentally overadd?

            // store block's contribution to the spectra
            if(idx_local == 0)
            {
              long iS3D_block = iS3D  +  blockIdx.x * spectra_length;

              //dN_pTdpTdphidy_d_blocks[iS3D_block] = dN_pTdpTdphidy_thread[0];
            }

          } // iy

        } // iphip

      } // ipT

    } // ipart

  } // end function





//Does a block reduction, where the previous kernel did a thread reduction.
//test if chunking is necessary for this kernel ?
__global__ void blockReduction(double* dN_pTdpTdphidy_d_blocks, double* dN_pTdpTdphidy_d, long spectra_length, long blocks_ker1)
{
  long local_idx = threadIdx.x  +  blockDim.x * blockIdx.x;

  // each thread is assigned a momentum / particle coordinate
  if(local_idx < spectra_length)
  { 
    // sum spectra contributions of the blocks from first kernel
    for(long iblock_ker1 = 0; iblock_ker1 < blocks_ker1; iblock_ker1++)
    {
      dN_pTdpTdphidy_d[local_idx] += dN_pTdpTdphidy_d_blocks[local_idx  +  iblock_ker1 * spectra_length];
    }
    if(isnan(dN_pTdpTdphidy_d[local_idx])) 
    {
      printf("found dN_pTdpTdphidy_d nan \n");
    }
  }
}



/*
//Does a block reduction, where the previous kernel did a thread reduction.
//test if chunking is necessary for this kernel ?
__global__ void blockReduction(double* dN_pTdpTdphidy_d, int final_spectrum_size, int blocks_ker1)
{
  long idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < final_spectrum_size)
  {
    if (blocks_ker1 == 1) return; //Probably will never happen, but best to be careful
    //Need to start at i=1, since adding everything to i=0

    for (int i = 1; i < blocks_ker1; i++)
    {
      dN_pTdpTdphidy_d[idx] += dN_pTdpTdphidy_d[idx + i * final_spectrum_size];
      if (isnan(dN_pTdpTdphidy_d[idx])) printf("found dN_pTdpTdphidy_d nan \n");
    }
  }
}
*/

__global__ void calculate_dN_pTdpTdphidy_VAH_PL( long FO_length, long number_of_chosen_particles, long pT_tab_length, long phi_tab_length, long y_pts, long eta_pts,
  double* dN_pTdpTdphidy_d, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaDeltaWeights_d,
  double *Mass_d, double *Sign_d, double *Degen_d, double *Baryon_d,
  double *T_d, double *P_d, double *E_d, double *tau_d, double *eta_d, double *ut_d, double *ux_d, double *uy_d, double *un_d,
  double *dat_d, double *dax_d, double *day_d, double *dan_d,
  double *pitt_d, double *pitx_d, double *pity_d, double *pitn_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *pinn_d, double *bulkPi_d,
  double *Wx_d, double *Wy_d, double *Lambda_d, double *aL_d, double *c0_d, double *c1_d, double *c2_d, double *c3_d, double *c4_d, double prefactor,
  int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int DIMENSION)
  {
    for (long n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      //This array is a shared array that will contain the integration contributions from each cell.
      __shared__ double temp[threadsPerBlock];
      //Assign a global index and a local index
      int icell = threadIdx.x + blockDim.x * blockIdx.x; //global idx
      int idx_local = threadIdx.x; //local idx
      long endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds

      __syncthreads();
      for (long imm = 0; imm < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts; imm++) //each thread <-> FO cell , and each thread loops over all momenta and species
      {
        //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
        temp[idx_local] = 0.0;

        if (icell < endFO)
        {
          //get Freezeout cell info
          /******************************************************************/
          long icell_glb = icell + (n * FO_chunk); //this runs over the entire FO surface
          double tau = tau_d[icell_glb];         // longitudinal proper time
          double tau2 = tau * tau;
          if(DIMENSION == 3)
          {
            etaValues_d[0] = eta_d[icell_glb];     // spacetime rapidity from surface file
          }
          //double eta = eta_d[icell_glb];         // spacetime rapidity

          double dat = dat_d[icell_glb];         // covariant normal surface vector
          double dax = dax_d[icell_glb];
          double day = day_d[icell_glb];
          double dan = dan_d[icell_glb];

          //double ut = ut_d[icell_glb];           // contravariant fluid velocity
          double ux = ux_d[icell_glb];
          double uy = uy_d[icell_glb];
          double un = un_d[icell_glb];
          double ut = sqrt(fabs(1.0 + ux * ux + uy * uy + tau2 * un * un));

          double u0 = sqrt(1.0 + ux*ux + uy*uy);
          double zt = tau * un / u0;
          double zn = ut / (u0 * tau);            // z basis vector

          double T = T_d[icell_glb];             // temperature
          double E = E_d[icell_glb];             // energy density
          double P = P_d[icell_glb];             // pressure

          double pitt = pitt_d[icell_glb];       // piperp^munu
          double pitx = pitx_d[icell_glb];
          double pity = pity_d[icell_glb];
          double pitn = pitn_d[icell_glb];
          double pixx = pixx_d[icell_glb];
          double pixy = pixy_d[icell_glb];
          double pixn = pixn_d[icell_glb];
          double piyy = piyy_d[icell_glb];
          double piyn = piyn_d[icell_glb];
          double pinn = pinn_d[icell_glb];

          double bulkPi = bulkPi_d[icell_glb];   // residual bulk pressure

          double Wx = Wx_d[icell_glb];           // W^mu
          double Wy = Wy_d[icell_glb];           // reinforce orthogonality
          double Wt = (ux * Wx  +  uy * Wy) * ut / (u0 * u0);
          double Wn = Wt * un / ut;

          double Lambda = Lambda_d[icell_glb];   // anisotropic variables
          double aL = aL_d[icell_glb];

          double c0 = c0_d[icell_glb];           // 14-moment coefficients
          double c1 = c1_d[icell_glb];
          double c2 = c2_d[icell_glb];
          double c3 = c3_d[icell_glb];
          double c4 = c4_d[icell_glb];

          /**************************************************************/

          //get particle info and momenta
          //imm = ipT + (iphip * (pT_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (ipart * (pT_tab_length * phi_tab_length * y_pts))
          long ipart       = imm / (pT_tab_length * phi_tab_length * y_pts);
          long iy          = (imm - (ipart * pT_tab_length * phi_tab_length * y_pts) ) / (pT_tab_length * phi_tab_length);
          long iphip       = (imm - (ipart * pT_tab_length * phi_tab_length * y_pts) - (iy * pT_tab_length * phi_tab_length) ) / pT_tab_length;
          long ipT         = imm - ( (ipart * (pT_tab_length * phi_tab_length * y_pts)) + (iy * (pT_tab_length * phi_tab_length)) + (iphip * (pT_tab_length)) );

          // set particle properties
          double mass = Mass_d[ipart];    // (GeV)
          double mass2 = mass * mass;
          double sign = Sign_d[ipart];
          double degeneracy = Degen_d[ipart];

          double px       = pT_d[ipT] * trig_d[ipT + phi_tab_length];
          double py       = pT_d[ipT] * trig_d[ipT];
          double mT       = sqrt(mass2 + pT_d[ipT] * pT_d[ipT]);
          double y        = y_d[iy];

          double pdotdsigma_f_eta_sum = 0.0; //accumulator
          //sum over eta
          for (int ieta = 0; ieta < eta_pts; ieta++)
          {
            double eta = etaValues_d[ieta];
            double delta_eta_weight = etaDeltaWeights_d[ieta];

            double pt = mT * cosh(y - eta); // contravariant
            double pn = (mT / tau) * sinh(y - eta); // contravariant
            // useful expression
            double tau2_pn = tau2 * pn;

            //momentum vector is contravariant, surface normal vector is COVARIANT
            double pdotdsigma = pt * dat + px * dax + py * day + pn * dan;

            //thermal equilibrium distributions - for viscous hydro
            double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;    // u.p = LRF energy
            double pdotz = pt * zt  -  tau2_pn * zn;                            // z.p = - LRF longitudinal momentum

            double xiL = 1.0 / (aL * aL)  -  1.0;

            #ifdef FORCE_F0
            Lambda = T;
            xiL = 0.0;
            #endif

            double Ea = sqrt(pdotu * pdotu  +  xiL * pdotz * pdotz);

            //leading order romatschke strickland distn
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

          } //for (ieta)

          temp[idx_local] += (prefactor * degeneracy * pdotdsigma_f_eta_sum);
        }//if(icell < FO_chunk)

        int N = blockDim.x;
        __syncthreads(); //Make sure threads are prepared for reduction
        do
        {
          //Here N must be a power of two. Try reducing by powers of 2, 4, 6 etc...
          N /= 2;
          if (idx_local < N) temp[idx_local] += temp[idx_local + N];
          __syncthreads();//Test if this is needed
        } while(N != 1);

        long final_spectrum_size = number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts;
        //if (idx_local == 0) dN_pTdpTdphidy_d[blockIdx.x * final_spectrum_size + imm] = temp[0];
        if (idx_local == 0) dN_pTdpTdphidy_d[blockIdx.x * final_spectrum_size + imm] += temp[0]; //add up results for each chunk
      } //for (long imm)
    } //for (int n) chunk loop
  } //calculate_dN_pTdpTdphidy

void EmissionFunctionArray::write_dN_pTdpTdphidy_toFile()
{
  printf("writing to file\n");
  //write 3D spectra in block format, different blocks for different species,
  //different sublocks for different values of rapidity
  //rows corespond to phip and columns correspond to pT

  int y_pts = y_tab_length;     // default 3+1d pts
  if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

  char filename[255] = "";

  sprintf(filename, "results/dN_pTdpTdphidy.dat");
  ofstream spectraFile(filename, ios_base::app);
  for (int ipart = 0; ipart < npart; ipart++)
  {
    for (int iy = 0; iy < y_pts; iy++)
    {
      double y;
      if(DIMENSION == 2) y = 0.0;
      else y = y_tab->get(1,iy + 1);

      for (int iphip = 0; iphip < phi_tab_length; iphip++)
      {
        double phip = phi_tab->get(1,iphip + 1);

        for (int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pT_tab->get(1,ipT + 1);
          //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
          long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
          //if (dN_pTdpTdphidy[is] < 1.0e-40) spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << 0.0 << "\n";
          //else spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[is] << "\n";
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
        } //ipT
        spectraFile << "\n";
      } //iphip
    } //iy
  }//ipart
  spectraFile.close();
}


void EmissionFunctionArray::calculate_spectra()
{
  cudaDeviceSynchronize();    // synchronize device at beginning
  cudaError_t err;            // error handling

  cout << "calculate_spectra() has started... " << endl;
  Stopwatch sw;
  sw.tic();

  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error at very beginning: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  double prefactor = pow(2.0 * M_PI * hbarC, -3);

  long blocks_ker1 = (FO_chunk + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the first kernel (thread reduction)
  long spectrum_size = blocks_ker1 * spectra_length; //the size of the spectrum which has been intrablock reduced, but not interblock reduced
  long final_spectrum_size = spectra_length; //size of final array for all particles , as a function of particle, pT and phi and y
  long blocks_ker2 = (final_spectrum_size + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the second kernel (block reduction)

  cout << "________________________|______________" << endl;
  cout << "pT_tab_length           |\t" << pT_tab_length << endl;
  cout << "phi_tab_length          |\t" << phi_tab_length << endl;
  cout << "y_tab_length            |\t" << y_tab_length << endl;
  cout << "eta_tab_length          |\t" << eta_tab_length << endl;
  cout << "unreduced spectrum size |\t" << spectrum_size << endl;
  cout << "reduced spectrum size   |\t" << spectra_length << endl;
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

  for(int ipart = 0; ipart < npart; ipart++)
  {
    int pdg_index = chosen_particles_table[ipart];
    particle = &particles[pdg_index];

    Mass[ipart] = particle->mass;
    Sign[ipart] = particle->sign;
    Degen[ipart] = particle->gspin;
    Baryon[ipart] = particle->baryon;
  }


  // freezeout surface info
  FO_surf *surf;
  double *T, *P, *E, *alphaB, *nB;            // thermodynamic properties
  double *tau, *eta;                          // position
  double *ux, *uy, *un;                       // u^mu
  double *dat, *dax, *day, *dan;              // dsigma_mu
  double *pixx, *pixy, *pixn, *piyy, *piyn;   // pi^munu
  double *bulkPi;                             // bulk pressure
  double *Vx, *Vy, *Vn;                       // V^mu

  // callocate memory
  T = (double*)calloc(FO_length, sizeof(double));
  P = (double*)calloc(FO_length, sizeof(double)); // replace with df coefficients
  E = (double*)calloc(FO_length, sizeof(double));
  tau = (double*)calloc(FO_length, sizeof(double));

  if(DIMENSION == 3)
  {
    eta = (double*)calloc(FO_length, sizeof(double));
  }

  ux = (double*)calloc(FO_length, sizeof(double));
  uy = (double*)calloc(FO_length, sizeof(double));
  un = (double*)calloc(FO_length, sizeof(double));

  dat = (double*)calloc(FO_length, sizeof(double));
  dax = (double*)calloc(FO_length, sizeof(double));
  day = (double*)calloc(FO_length, sizeof(double));
  dan = (double*)calloc(FO_length, sizeof(double));

  if(INCLUDE_SHEAR_DELTAF)
  {
    pixx = (double*)calloc(FO_length, sizeof(double));
    pixy = (double*)calloc(FO_length, sizeof(double));
    pixn = (double*)calloc(FO_length, sizeof(double));
    piyy = (double*)calloc(FO_length, sizeof(double));
    piyn = (double*)calloc(FO_length, sizeof(double));
  }

  if(INCLUDE_BULK_DELTAF)
  {
    bulkPi = (double*)calloc(FO_length, sizeof(double));
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    alphaB = (double*)calloc(FO_length, sizeof(double));  // muB / T
    nB = (double*)calloc(FO_length, sizeof(double));
    Vx = (double*)calloc(FO_length, sizeof(double));
    Vy = (double*)calloc(FO_length, sizeof(double));
    Vn = (double*)calloc(FO_length, sizeof(double));
  }

  /*
  // freezeout surface info exclusive for VAH_PL
  double *PL, *Wx, *Wy;
  double *Lambda, *aL;
  double *c0, *c1, *c2, *c3, *c4; //delta-f coeffs for vah

  if(MODE == 2)
  {
    PL = (double*)calloc(FO_length, sizeof(double));

    //Wt = (double*)calloc(FO_length, sizeof(double));
    Wx = (double*)calloc(FO_length, sizeof(double));
    Wy = (double*)calloc(FO_length, sizeof(double));
    //Wn = (double*)calloc(FO_length, sizeof(double));

    Lambda = (double*)calloc(FO_length, sizeof(double));
    aL = (double*)calloc(FO_length, sizeof(double));

    // 14-moment coefficients (VAH_PL)
    if (DF_MODE == 4)
    {
      c0 = (double*)calloc(FO_length, sizeof(double));
      c1 = (double*)calloc(FO_length, sizeof(double));
      c2 = (double*)calloc(FO_length, sizeof(double));
      c3 = (double*)calloc(FO_length, sizeof(double));
      c4 = (double*)calloc(FO_length, sizeof(double));
    }
  }
  */

  // Reading freezeout surface info
  for(long icell = 0; icell < FO_length; icell++)
  {
    surf = &surf_ptr[icell];

    T[icell] = surf->T;
    P[icell] = surf->P;
    E[icell] = surf->E;

    tau[icell] = surf->tau;
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
      nB[icell] = surf->nB;
      Vx[icell] = surf->Vx;
      Vy[icell] = surf->Vy;
      Vn[icell] = surf->Vn;
    }

    /*
    if(MODE == 2)
    {
      PL[icell] = surf->PL;
      //Wt[icell] = surf->Wt;
      Wx[icell] = surf->Wx;
      Wy[icell] = surf->Wy;
      //Wn[icell] = surf->Wn;

      Lambda[icell] = surf->Lambda;
      aL[icell] = surf->aL;

      if (DF_MODE == 4)
      {
        c0[icell] = surf->c0;
        c1[icell] = surf->c1;
        c2[icell] = surf->c2;
        c3[icell] = surf->c3;
        c4[icell] = surf->c4;
      }
    }
    */
  } // icell


  // set up momentum tables to pass to GPU
  double *pT = (double*)calloc(pT_tab_length, sizeof(double));
  double *trig = (double*)calloc(2 * phi_tab_length, sizeof(double));
  double yValues[y_tab_length];
  double etaValues[eta_tab_length];
  double etaWeights[eta_tab_length];

  for(int ipT = 0; ipT < pT_tab_length; ipT++)
  {
    pT[ipT] = pT_tab->get(1, ipT + 1);
  }

  for(int iphip = 0; iphip < phi_tab_length; iphip++)
  {
    double phip = phi_tab->get(1, iphip + 1);

    trig[iphip] = sin(phip);
    trig[iphip + phi_tab_length] = cos(phip);
  }


  if(DIMENSION == 2)
  {
    yValues[0] = 0.0;
    for(int ieta = 0; ieta < eta_tab_length; ieta++)
    {
      etaValues[ieta] = eta_tab->get(1, ieta + 1);
      etaWeights[ieta] = eta_tab->get(2, ieta + 1);
    }
  }
  else if(DIMENSION == 3)
  {
    etaValues[0] = 0.0;
    etaWeights[0] = 1.0;
    for(int iy = 0; iy < y_tab_length; iy++)
    {
      yValues[iy] = y_tab->get(1, iy + 1);
    }
  }



  cout << "Declaring and allocating device arrays " << endl;

  double *Mass_d, *Sign_d, *Degen_d, *Baryon_d;

  double *T_d, *P_d, *E_d, *alphaB_d, *nB_d;
  double *tau_d, *eta_d;
  double *ux_d, *uy_d, *un_d;
  double *dat_d, *dax_d, *day_d, *dan_d;
  double *pixx_d, *pixy_d, *pixn_d, *piyy_d, *piyn_d;
  double *bulkPi_d;
  double *Vx_d, *Vy_d, *Vn_d;

  double *pT_d, *trig_d, *y_d, *etaValues_d, *etaWeights_d;

  double *dN_pTdpTdphidy_d;
  double *dN_pTdpTdphidy_d_blocks;

  //vah quantities
  //double *PL_d, *Wx_d, *Wy_d, *Lambda_d, *aL_d;
  //double *c0_d, *c1_d, *c2_d, *c3_d, *c4_d;


  // allocate memory on device
  cudaMalloc((void**) &Mass_d,   npart * sizeof(double));
  cudaMalloc((void**) &Sign_d,   npart * sizeof(double));
  cudaMalloc((void**) &Degen_d,  npart * sizeof(double));
  cudaMalloc((void**) &Baryon_d, npart * sizeof(double));


  cudaMalloc((void**) &T_d, FO_length * sizeof(double));
  cudaMalloc((void**) &P_d, FO_length * sizeof(double));
  cudaMalloc((void**) &E_d, FO_length * sizeof(double));

  cudaMalloc((void**)   &tau_d, FO_length * sizeof(double));
  if(DIMENSION == 3)
  {
    cudaMalloc((void**) &eta_d, FO_length * sizeof(double));
  }

  cudaMalloc((void**) &ux_d,  FO_length * sizeof(double));
  cudaMalloc((void**) &uy_d,  FO_length * sizeof(double));
  cudaMalloc((void**) &un_d,  FO_length * sizeof(double));

  cudaMalloc((void**) &dat_d, FO_length * sizeof(double));
  cudaMalloc((void**) &dax_d, FO_length * sizeof(double));
  cudaMalloc((void**) &day_d, FO_length * sizeof(double));
  cudaMalloc((void**) &dan_d, FO_length * sizeof(double));

  if(INCLUDE_SHEAR_DELTAF)
  {
    cudaMalloc((void**) &pixx_d, FO_length * sizeof(double));
    cudaMalloc((void**) &pixy_d, FO_length * sizeof(double));
    cudaMalloc((void**) &pixn_d, FO_length * sizeof(double));
    cudaMalloc((void**) &piyy_d, FO_length * sizeof(double));
    cudaMalloc((void**) &piyn_d, FO_length * sizeof(double));
  }

  if(INCLUDE_BULK_DELTAF)
  {
    cudaMalloc((void**) &bulkPi_d, FO_length * sizeof(double));
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    cudaMalloc((void**) &alphaB_d, FO_length * sizeof(double));
    cudaMalloc((void**) &nB_d,     FO_length * sizeof(double));
    cudaMalloc((void**) &Vx_d,     FO_length * sizeof(double));
    cudaMalloc((void**) &Vy_d,     FO_length * sizeof(double));
    cudaMalloc((void**) &Vn_d,     FO_length * sizeof(double));
  }

  /*
  if (MODE == 2)
  {
    cudaMalloc( (void**) &PL_d,     FO_length * sizeof(double) );
    cudaMalloc( (void**) &Wx_d,     FO_length * sizeof(double) );
    cudaMalloc( (void**) &Wy_d,     FO_length * sizeof(double) );
    cudaMalloc( (void**) &Lambda_d, FO_length * sizeof(double) );
    cudaMalloc( (void**) &aL_d,     FO_length * sizeof(double) );
    if (DF_MODE == 4)
    {
      cudaMalloc( (void**) &c0_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &c1_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &c2_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &c3_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &c4_d, FO_length * sizeof(double) );
    }
  }
  */

  cudaMalloc((void**) &pT_d,   pT_tab_length * sizeof(double));
  cudaMalloc((void**) &trig_d, 2 * phi_tab_length * sizeof(double));
  cudaMalloc((void**) &etaValues_d,  eta_tab_length * sizeof(double));
  cudaMalloc((void**) &etaWeights_d, eta_tab_length * sizeof(double));
  cudaMalloc((void**) &y_d, y_tab_length * sizeof(double));
 
  cudaMalloc((void**) &dN_pTdpTdphidy_d, spectra_length * sizeof(double));
  cudaMalloc((void**) &dN_pTdpTdphidy_d_blocks, spectrum_size * sizeof(double));

  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error in device memory allocation: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }



  cout << "Copying particle, momentum and freezeout data from host to device" << endl;
  
  cudaMemcpy(Mass_d,   Mass,   npart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Sign_d,   Sign,   npart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Degen_d,  Degen,  npart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Baryon_d, Baryon, npart * sizeof(double), cudaMemcpyHostToDevice);



  cudaMemcpy(T_d,   T,   FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(P_d,   P,   FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(E_d,   E,   FO_length * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(tau_d,   tau, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  if(DIMENSION == 3)
  {
    cudaMemcpy(eta_d, eta, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  }

  cudaMemcpy(ux_d,  ux,  FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(uy_d,  uy,  FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(un_d,  un,  FO_length * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(dat_d, dat, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dax_d, dax, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(day_d, day, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dan_d, dan, FO_length * sizeof(double), cudaMemcpyHostToDevice);

  if(INCLUDE_SHEAR_DELTAF)
  {
    cudaMemcpy(pixx_d, pixx, FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pixy_d, pixy, FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pixn_d, pixn, FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(piyy_d, piyy, FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(piyn_d, piyn, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  }

  if(INCLUDE_BULK_DELTAF)
  {
    cudaMemcpy(bulkPi_d, bulkPi, FO_length * sizeof(double), cudaMemcpyHostToDevice);
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    cudaMemcpy(alphaB_d, alphaB, FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(nB_d,     nB,     FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Vx_d,     Vx,     FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Vy_d,     Vy,     FO_length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Vn_d,     Vn,     FO_length * sizeof(double), cudaMemcpyHostToDevice);
  }

  /*
  if (MODE == 2)
  {
    cudaMemcpy( PL_d, PL, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( Wx_d, Wx, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( Wy_d, Wy, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( Lambda_d, Lambda, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( aL_d, aL, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    if (DF_MODE == 4)
    {
      cudaMemcpy( c0_d, c0, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( c1_d, c1, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( c2_d, c2, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( c3_d, c3, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( c4_d, c4, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    }
  }
  */

  //cudaMemcpy( dN_pTdpTdphidy_d, dN_pTdpTdphidy, spectrum_size * sizeof(double), cudaMemcpyHostToDevice ); //this is an empty array, we shouldn't need to memcpy it?


  cudaMemcpy(pT_d,   pT,   pT_tab_length * sizeof(double),      cudaMemcpyHostToDevice);
  cudaMemcpy(trig_d, trig, 2 * phi_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(etaValues_d,  etaValues,  eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(etaWeights_d, etaWeights, eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, yValues, y_tab_length * sizeof(double), cudaMemcpyHostToDevice);


  // dimensions?
  cudaMemset(dN_pTdpTdphidy_d, 0.0, spectra_length * sizeof(double));
  cudaMemset(dN_pTdpTdphidy_d_blocks, 0.0, spectrum_size * sizeof(double));

  // but dN_pTdpTdphidy_d and dN_pTdpTdphidy don't have same dimensions...

  double df_coeff[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  // copy df coefficients to the device array
  double *df_coeff_d;
  cudaMalloc((void**) &df_coeff_d, 5 * sizeof(double));
  cudaMemcpy(df_coeff_d, df_coeff, 5 * sizeof(double), cudaMemcpyHostToDevice);



  // check if memory copy error
  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error in a cudaMemcpy: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  //Perform kernels, first calculation of integrand and reduction across threads, then a reduction across blocks

  if(debug) cout << "Starting first Kernel: thread reduction" << endl;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaDeviceSynchronize();


 
  calculate_dN_pTdpTdphidy<<<blocks_ker1, threadsPerBlock>>>(FO_length, npart, pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length, dN_pTdpTdphidy_d_blocks, pT_d, trig_d, y_d, etaValues_d, etaWeights_d, Mass_d, Sign_d, Degen_d, Baryon_d, T_d, P_d, E_d, tau_d, eta_d, ux_d, uy_d, un_d, dat_d, dax_d, day_d, dan_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, bulkPi_d, alphaB_d, nB_d, Vx_d, Vy_d, Vn_d, prefactor, df_coeff_d, INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DIMENSION, OUTFLOW, DF_MODE);
  

  /*
  else if (MODE == 2) calculate_dN_pTdpTdphidy_VAH_PL<<<blocks_ker1, threadsPerBlock>>>(FO_length, number_of_chosen_particles, pT_tab_length, phi_tab_length, y_pts, eta_pts,
                                                                                        dN_pTdpTdphidy_d, pT_d, trig_d, y_d, eta_values_d, etaDeltaWeights_d,
                                                                                        Mass_d, Sign_d, Degen_d, Baryon_d,
                                                                                        T_d, P_d, E_d, tau_d, eta_d, ut_d, ux_d, uy_d, un_d,
                                                                                        dat_d, dax_d, day_d, dan_d,
                                                                                        pitt_d, pitx_d, pity_d, pitn_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, pinn_d, bulkPi_d,
                                                                                        Wx_d, Wy_d, Lambda_d, aL_d, c0_d, c1_d, c2_d, c3_d, c4_d, prefactor,
                                                                                        REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, DIMENSION);
  */


  // synchonize kernels after calculation finished
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("Error in first kernel: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }


  // not sure what's going on here
  cout << "Starting second kernel: block reduction" << endl;
  cudaDeviceSynchronize();
  blockReduction<<<blocks_ker2, threadsPerBlock>>>(dN_pTdpTdphidy_d_blocks, dN_pTdpTdphidy_d, spectra_length, blocks_ker1);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds * 1000.0;
  cout << "Finished in " << seconds << " seconds." << endl;

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in second kernel: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }


  // Copy final spectra for all particles from device to host
  cout << "Copying spectra from device to host" << endl;
  //cudaMemcpy( dN_pTdpTdphidy, dN_pTdpTdphidy_d, final_spectrum_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaMemcpy(dN_pTdpTdphidy, dN_pTdpTdphidy_d, spectra_length * sizeof(double), cudaMemcpyDeviceToHost);

  // Write the results to disk
  write_dN_pTdpTdphidy_toFile();

  // Clean up on host and device
  cout << "Deallocating host and device memory" << endl;

  free(Mass);
  free(Sign);
  free(Degen);
  free(Baryon);

  free(T);
  free(P);
  free(E);

  free(tau);
  if(DIMENSION == 3)
  {
    free(eta);
  }

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
    free(nB);
    free(Vx);
    free(Vy);
    free(Vn);
  }

  /*
  if (MODE == 2)
  {
    free(PL);
    free(Wx);
    free(Wy);
    free(Lambda);
    free(aL);
    if (DF_MODE == 4)
    {
      free(c0);
      free(c1);
      free(c2);
      free(c3);
      free(c4);
    }
  }
  */

  // cudaFree = deallocate memory on device
  cudaFree(Mass_d);
  cudaFree(Sign_d);
  cudaFree(Degen_d);
  cudaFree(Baryon_d);

  cudaFree(T_d);
  cudaFree(P_d);
  cudaFree(E_d);

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
    cudaFree(nB_d);
    cudaFree(Vx_d);
    cudaFree(Vy_d);
    cudaFree(Vn_d);
  }

  cudaFree(pT_d);
  cudaFree(trig_d);
  cudaFree(etaValues_d);
  cudaFree(etaWeights_d);
  cudaFree(y_d);

  cudaFree(dN_pTdpTdphidy_d);

  /*
  if (MODE == 2)
  {
    cudaFree(PL);
    cudaFree(Wx);
    cudaFree(Wy);
    cudaFree(Lambda);
    cudaFree(aL);
    if (DF_MODE == 4)
    {
      cudaFree(c0);
      cudaFree(c1);
      cudaFree(c2);
      cudaFree(c3);
      cudaFree(c4);
    }
  }
  */
}
