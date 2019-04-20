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
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in,
                                              particle_info* particles_in, int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in, deltaf_coefficients df_in)
{
  paraRdr = paraRdr_in;
  pT_tab = pT_tab_in;
  pT_tab_length = pT_tab->getNumberOfRows();
  phi_tab = phi_tab_in;
  phi_tab_length = phi_tab->getNumberOfRows();
  y_tab = y_tab_in;
  y_tab_length = y_tab->getNumberOfRows();
  eta_tab = eta_tab_in;
  eta_tab_length = eta_tab->getNumberOfRows();

  // get control parameters
  MODE = paraRdr->getVal("mode");
  DF_MODE = paraRdr->getVal("df_mode");
  DIMENSION = paraRdr->getVal("dimension");
  INCLUDE_BARYON = paraRdr->getVal("include_baryon");
  INCLUDE_BULK_DELTAF = paraRdr->getVal("include_bulk_deltaf");
  INCLUDE_SHEAR_DELTAF = paraRdr->getVal("include_shear_deltaf");
  INCLUDE_BARYONDIFF_DELTAF = paraRdr->getVal("include_baryondiff_deltaf");
  REGULATE_DELTAF = paraRdr->getVal("regulate_deltaf");
  GROUP_PARTICLES = paraRdr->getVal("group_particles");
  PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");

  particles = particles_in;
  Nparticles = Nparticles_in;
  surf_ptr = surf_ptr_in;
  FO_length = FO_length_in;
  df = df_in;
  number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

  chosen_particles_01_table = new int[Nparticles];
  //a class member to hold 3D spectra for all chosen particles

  int y_pts = y_tab_length;
  if (DIMENSION == 2) y_pts = 1;
  dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts];
  //zero the array
  for (long iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts; iSpectra++)
  {
    dN_pTdpTdphidy[iSpectra] = 0.0;
  }

  for (int n = 0; n < Nparticles; n++) chosen_particles_01_table[n] = 0;

  //only grab chosen particles from the table
  for (int m = 0; m < number_of_chosen_particles; m++)
  { //loop over all chosen particles
    int mc_id = chosen_particles_in->get(1, m + 1);

    for (int n = 0; n < Nparticles; n++)
    {
      if (particles[n].mc_id == mc_id)
      {
        chosen_particles_01_table[n] = 1;
        break;
      }
    }
  }

  // next, for sampling processes
  chosen_particles_sampling_table = new int[number_of_chosen_particles];
  // first copy the chosen_particles table, but now using indices instead of mc_id
  int current_idx = 0;
  for (int m = 0; m < number_of_chosen_particles; m++)
  {
    int mc_id = chosen_particles_in->get(1, m + 1);
    for (int n = 0; n < Nparticles; n++)
    {
      if (particles[n].mc_id == mc_id)
      {
        chosen_particles_sampling_table[current_idx] = n;
        current_idx ++;
        break;
      }
    }
  }

  // next re-order them so that particles with similar mass are adjacent
  if (GROUP_PARTICLES == 1) // sort particles according to their mass; bubble-sorting
  {
    for (int m = 0; m < number_of_chosen_particles; m++)
    {
      for (int n = 0; n < number_of_chosen_particles - m - 1; n++)
      {
        if (particles[chosen_particles_sampling_table[n]].mass > particles[chosen_particles_sampling_table[n + 1]].mass)
        {
          // swap them
          int particle_idx = chosen_particles_sampling_table[n + 1];
          chosen_particles_sampling_table[n + 1] = chosen_particles_sampling_table[n];
          chosen_particles_sampling_table[n] = particle_idx;
        }
      }
    }
  }
}


EmissionFunctionArray::~EmissionFunctionArray()
{
  delete[] chosen_particles_01_table;
  delete[] chosen_particles_sampling_table;
  delete[] dN_pTdpTdphidy; //for holding 3d spectra of all chosen particles
}

__global__ void calculate_dN_pTdpTdphidy( long FO_length, long number_of_chosen_particles, long pT_tab_length, long phi_tab_length, long y_pts, long eta_pts,
  double* dN_pTdpTdphidy_d, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaDeltaWeights_d,
  double *Mass_d, double *Sign_d, double *Degen_d, double *Baryon_d,
  double *T_d, double *P_d, double *E_d, double *tau_d, double *eta_d, double *ut_d, double *ux_d, double *uy_d, double *un_d,
  double *dat_d, double *dax_d, double *day_d, double *dan_d,
  double *pitt_d, double *pitx_d, double *pity_d, double *pitn_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *pinn_d, double *bulkPi_d,
  double *muB_d, double *nB_d, double *Vt_d, double *Vx_d, double *Vy_d, double *Vn_d, double prefactor, double *df_coeff_d,
  int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION)
  {

    // loop over freezeout surface chunks
    for(long n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      // set the chunk size
      long endFO = FO_chunk;                            // default size
      if(n == (FO_length / FO_chunk)) 
      {
        endFO = FO_length - (n * FO_chunk);             // reduce size of last chunk
      }

      // this array is a shared array that will contain the integration contributions from each cell.
      __shared__ double temp[threadsPerBlock];

      int icell = threadIdx.x + blockDim.x * blockIdx.x; // global index of thread in kernel
      int idx_local = threadIdx.x;                       // local index of thread in block
      

      __syncthreads();                                   // sync threads for each block


      // each thread is assigned icell in chunk and loops over all momenta and species
      for(long imm = 0; imm < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts; imm++) 
      {
        temp[idx_local] = 0.0;                          // reset temp to zero
        // why not move this loop inside the if(icell < endFO)?

        // only do perform calculation for chunk cells (may be some extraneous threads)
        if(icell < endFO)
        {
          // get freezeout cell info:
          long icell_glb = icell + n * FO_chunk;  // global freezeout cell index

          double tau = tau_d[icell_glb];          // longitudinal proper time
          double tau2 = tau * tau;
          if(DIMENSION == 3)
          {
            etaValues_d[0] = eta_d[icell_glb];    // spacetime rapidity from surface file
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

          if(udsigma <= 0.0)                      // skip cells with u.dsigma < 0
          {
            break;               
          }

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
          double baryon_enthalpy_ratio = 0.0; 

          if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
          {
            muB = muB_d[icell_glb];
            nB = nB_d[icell_glb];
            Vx = Vx_d[icell_glb];
            Vy = Vy_d[icell_glb];
            Vn = Vn_d[icell_glb];
            Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

            alphaB = muB / T;
            baryon_enthalpy_ratio = nB / (E + P);
          }

          // df coefficients
          double shear_coeff = 0.5 / (T * T * (E + P));  // (see df_shear)


          // let's forget about df coefficients for now

          // if I need to call a function here
          // do I need to put __device__ ? 
          // what about classes?


          // get particle info and momenta (ask Derek he got this)
          // why not just make separate loops? 

          //imm = ipT + (iphip * (pT_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (ipart * (pT_tab_length * phi_tab_length * y_pts))
          long ipart = imm / (pT_tab_length * phi_tab_length * y_pts);
          long iy = (imm - (ipart * pT_tab_length * phi_tab_length * y_pts) ) / (pT_tab_length * phi_tab_length);
          long iphip = (imm - (ipart * pT_tab_length * phi_tab_length * y_pts) - (iy * pT_tab_length * phi_tab_length) ) / pT_tab_length;
          long ipT = imm - ( (ipart * (pT_tab_length * phi_tab_length * y_pts)) + (iy * (pT_tab_length * phi_tab_length)) + (iphip * (pT_tab_length)) );


          // set particle properties
          double mass = Mass_d[ipart];   
          double sign = Sign_d[ipart];
          double degeneracy = Degen_d[ipart];
          double baryon = Baryon_d[ipart];

          double mass2 = mass * mass;
          double chem = baryon * alphaB;          


          // set momentum
          double pT = pT_d[ipT];
          double sinphip = trig_d[ipT];
          double cosphip = trig_d[ipT + phi_tab_length];
          double y = y_d[iy];

          double px = pT * cosphip;
          double py = pT * sinphip;
          double mT = sqrt(mass2  +  pT * pT);
          double mT_over_tau = mT / tau;
          

          double pdotdsigma_f_eta_sum = 0.0; // accumulator

          // sum over eta
          for(int ieta = 0; ieta < eta_pts; ieta++)
          {
            double eta = etaValues_d[ieta];
            double eta_weight = etaDeltaWeights_d[ieta];

            double ptau = mT * cosh(y - eta);         // p^tau
            double pn = mT_over_tau * sinh(y - eta);  // p^eta
            double tau2_pn = tau2 * pn;

            double pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan);

            if(true && pdotdsigma <= 0.0)          // enforce outflow (temporary)
            {
              continue;
            }
          
            double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;  // u.p
            double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
            double feqbar = 1.0  -  sign * feq;

            // pi^munu.p_mu.p_nu
            double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
            + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

            // V^mu.p_mu
            double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;


            double df;

            // compute df correction
            switch(1)
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
                printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
              }
            } // DF_MODE

            if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

            double f = feq * (1.0 + df);  

            pdotdsigma_f_eta_sum += pdotdsigma * f;
        
          } // ieta  

          temp[idx_local] += (prefactor * degeneracy * pdotdsigma_f_eta_sum);

        } // if(icell < FO_chunk)




        // perform reduction over threads in each block:
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::

        int N = blockDim.x;     // number of threads in block
        __syncthreads();        // make sure threads are prepared for reduction

        do
        {
          // here N must be a power of two. (try reducing by powers of 2, 4, 6 etc...)
          N /= 2;

          if(idx_local < N) 
          {
            temp[idx_local] += temp[idx_local + N];
          }

          __syncthreads();      // test if this is needed

        } while(N != 1);

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::



        long final_spectrum_size = number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts;

        //if (idx_local == 0) dN_pTdpTdphidy_d[blockIdx.x * final_spectrum_size + imm] = temp[0];

        // will you accidentally overadd?
        if(idx_local == 0) 
        {
          dN_pTdpTdphidy_d[blockIdx.x * final_spectrum_size + imm] += temp[0]; // add up results for each chunk
        }

      } // imm (species/momenta loop)

    } // n (chunk loop)

  } // end of calculate_dN_pTdpTdphidy(...)


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
  int npart = number_of_chosen_particles;
  char filename[255] = "";

  sprintf(filename, "results/dN_pTdpTdphidy.dat");
  ofstream spectraFile(filename, ios_base::app);
  for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
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
  cudaDeviceSynchronize();    // synchronize the kernel at beginning
  cudaError_t err;            // error handling

  cout << "calculate_spectra() has started... " << endl;
  Stopwatch sw;
  sw.tic();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error at very beginning: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
    // why not exit?
  }

  // here we want to set up the tables to pass to device
  // rather than constructing them locally
  long y_pts = y_tab_length;    // defaults pts for 3+1d
  long eta_pts = 1;

  if(DIMENSION == 2)
  {
    y_pts = 1;
    eta_pts = eta_tab_length;
  }

  // number of blocks in kernels for reduction and ...


  //int  blocks_ker1 = (FO_length + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the first kernel (thread reduction)
  long  blocks_ker1 = (FO_chunk + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the first kernel (thread reduction)
  long spectrum_size = blocks_ker1 * number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts; //the size of the spectrum which has been intrablock reduced, but not interblock reduced
  long  final_spectrum_size = number_of_chosen_particles * pT_tab_length * phi_tab_length * y_pts; //size of final array for all particles , as a function of particle, pT and phi and y
  long  blocks_ker2 = (final_spectrum_size + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the second kernel (block reduction)

  cout << "# of chosen particles   = " << number_of_chosen_particles << endl;
  cout << "FO_length               = " << FO_length            << endl;
  cout << "pT_tab_length           = " << pT_tab_length        << endl;
  cout << "phi_tab_length          = " << phi_tab_length       << endl;
  cout << "y_pts                   = " << y_pts                << endl;
  cout << "unreduced spectrum size = " << spectrum_size        << endl;
  cout << "reduced spectrum size   = " << final_spectrum_size  << endl;
  cout << "threads per block       = " << threadsPerBlock      << endl;
  cout << "blocks in first kernel  = " << blocks_ker1          << endl;
  cout << "blocks in second kernel = " << blocks_ker2          << endl;

  //Declare and fill arrays on host
  cout << "Declaring and filling host arrays with particle and freezeout surface info" << endl;
  //particle info
  particle_info *particle;
  double *Mass, *Sign, *Degen, *Baryon;
  Mass   = (double*)calloc(number_of_chosen_particles, sizeof(double));
  Sign   = (double*)calloc(number_of_chosen_particles, sizeof(double));
  Degen  = (double*)calloc(number_of_chosen_particles, sizeof(double));
  Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));

  for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
  {
    int particle_idx = chosen_particles_sampling_table[ipart];
    particle = &particles[particle_idx];
    Mass[ipart]   = particle->mass;
    Sign[ipart]   = particle->sign;
    Degen[ipart]  = particle->gspin;
    Baryon[ipart] = particle->baryon;
  }

  //freezeout surface info
  FO_surf *surf = &surf_ptr[0];
  double *T, *P, *E, *tau, *eta, *ut, *ux, *uy, *un;
  double *dat, *dax, *day, *dan;
  double *pitt, *pitx, *pity, *pitn, *pixx, *pixy, *pixn, *piyy, *piyn, *pinn, *bulkPi;
  double *muB, *nB, *Vt, *Vx, *Vy, *Vn;
  //temperature, pressure, energy density, tau, eta
  T   = (double*)calloc(FO_length, sizeof(double));
  P   = (double*)calloc(FO_length, sizeof(double));
  E   = (double*)calloc(FO_length, sizeof(double));
  tau = (double*)calloc(FO_length, sizeof(double));
  eta = (double*)calloc(FO_length, sizeof(double));
  //contravariant flow velocity
  ut  = (double*)calloc(FO_length, sizeof(double));
  ux  = (double*)calloc(FO_length, sizeof(double));
  uy  = (double*)calloc(FO_length, sizeof(double));
  un  = (double*)calloc(FO_length, sizeof(double));
  //contravariant surface normal vector
  dat = (double*)calloc(FO_length, sizeof(double));
  dax = (double*)calloc(FO_length, sizeof(double));
  day = (double*)calloc(FO_length, sizeof(double));
  dan = (double*)calloc(FO_length, sizeof(double));
  //ten contravariant shear stress components
  pitt = (double*)calloc(FO_length, sizeof(double));
  pitx = (double*)calloc(FO_length, sizeof(double));
  pity = (double*)calloc(FO_length, sizeof(double));
  pitn = (double*)calloc(FO_length, sizeof(double));
  pixx = (double*)calloc(FO_length, sizeof(double));
  pixy = (double*)calloc(FO_length, sizeof(double));
  pixn = (double*)calloc(FO_length, sizeof(double));
  piyy = (double*)calloc(FO_length, sizeof(double));
  piyn = (double*)calloc(FO_length, sizeof(double));
  pinn = (double*)calloc(FO_length, sizeof(double));
  //bulk pressure
  bulkPi = (double*)calloc(FO_length, sizeof(double));

  if (INCLUDE_BARYON)
  {
    //baryon chemical potential
    muB = (double*)calloc(FO_length, sizeof(double));

    if (INCLUDE_BARYONDIFF_DELTAF)
    {
      nB = (double*)calloc(FO_length, sizeof(double));
      Vt = (double*)calloc(FO_length, sizeof(double));
      Vx = (double*)calloc(FO_length, sizeof(double));
      Vy = (double*)calloc(FO_length, sizeof(double));
      Vn = (double*)calloc(FO_length, sizeof(double));
    }
  }

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

  for (long icell = 0; icell < FO_length; icell++)
  {
    //reading info from surface
    surf = &surf_ptr[icell];
    T[icell]   = surf->T;
    P[icell]   = surf->P;
    E[icell]   = surf->E;
    tau[icell] = surf->tau;
    eta[icell] = surf->eta;

    ut[icell] = surf->ut;
    ux[icell] = surf->ux;
    uy[icell] = surf->uy;
    un[icell] = surf->un;

    dat[icell] = surf->dat;
    dax[icell] = surf->dax;
    day[icell] = surf->day;
    dan[icell] = surf->dan;

    pitt[icell] = surf->pitt;
    pitx[icell] = surf->pitx;
    pity[icell] = surf->pity;
    pitn[icell] = surf->pitn;
    pixx[icell] = surf->pixx;
    pixy[icell] = surf->pixy;
    pixn[icell] = surf->pixn;
    piyy[icell] = surf->piyy;
    piyn[icell] = surf->piyn;
    pinn[icell] = surf->pinn;

    bulkPi[icell] = surf->bulkPi;

    if (INCLUDE_BARYON)
    {
      muB[icell] = surf->muB;
      if (INCLUDE_BARYONDIFF_DELTAF)
      {
        nB[icell] = surf->nB;
        Vt[icell] = surf->Vt;
        Vx[icell] = surf->Vx;
        Vy[icell] = surf->Vy;
        Vn[icell] = surf->Vn;
      }
    }

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
  }
  // fill arrays with values points in y and eta


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

  double *pT, *trig;
  pT   = (double*)calloc(pT_tab_length,      sizeof(double));
  trig = (double*)calloc(2 * phi_tab_length, sizeof(double));

  for (int i = 0; i < pT_tab_length; i++) pT[i] = pT_tab->get(1, i+1);
  for (int i = 0; i < phi_tab_length; i++)
  {
    trig[i] = sin( phi_tab->get(1, i+1) );
    trig[i + phi_tab_length] = cos( phi_tab->get(1, i+1) );
  }

  cout << "Declaring and Allocating device arrays " << endl;
  double *Mass_d, *Sign_d, *Degen_d, *Baryon_d;
  double *pT_d, *trig_d, *y_d, *eta_values_d, *etaDeltaWeights_d;
  double *T_d, *P_d, *E_d, *tau_d, *eta_d, *ut_d, *ux_d, *uy_d, *un_d;
  double *dat_d, *dax_d, *day_d, *dan_d;
  double *pitt_d, *pitx_d, *pity_d, *pitn_d, *pixx_d, *pixy_d, *pixn_d, *piyy_d, *piyn_d, *pinn_d, *bulkPi_d;
  double *muB_d, *nB_d, *Vt_d, *Vx_d, *Vy_d, *Vn_d;

  //vah quantities
  double *PL_d, *Wx_d, *Wy_d, *Lambda_d, *aL_d;
  double *c0_d, *c1_d, *c2_d, *c3_d, *c4_d;

  double *dN_pTdpTdphidy_d;

  //Allocate memory on device
  cudaMalloc( (void**) &Mass_d,   number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Sign_d,   number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Degen_d,  number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Baryon_d, number_of_chosen_particles * sizeof(double) );

  cudaMalloc( (void**) &pT_d,     pT_tab_length * sizeof(double)              );
  cudaMalloc( (void**) &trig_d,   2 * phi_tab_length * sizeof(double)         );
  cudaMalloc( (void**) &y_d,      y_pts        * sizeof(double)               );
  cudaMalloc( (void**) &eta_values_d,  eta_pts * sizeof(double)               );
  cudaMalloc( (void**) &etaDeltaWeights_d,  eta_pts * sizeof(double)          );

  cudaMalloc( (void**) &T_d,      FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &P_d,      FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &E_d,      FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &tau_d,    FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &eta_d,    FO_length * sizeof(double)                  );

  cudaMalloc( (void**) &ut_d,     FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &ux_d,     FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &uy_d,     FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &un_d,     FO_length * sizeof(double)                  );

  cudaMalloc( (void**) &dat_d,    FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &dax_d,    FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &day_d,    FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &dan_d,    FO_length * sizeof(double)                  );

  cudaMalloc( (void**) &pitt_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pitx_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pity_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pitn_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pixx_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pixy_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pixn_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &piyy_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &piyn_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &pinn_d,   FO_length * sizeof(double)                  );
  cudaMalloc( (void**) &bulkPi_d, FO_length * sizeof(double)                  );

  if (INCLUDE_BARYON)
  {
    cudaMalloc( (void**) &muB_d,  FO_length * sizeof(double) );
    if (INCLUDE_BARYONDIFF_DELTAF)
    {
      cudaMalloc( (void**) &nB_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vt_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vx_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vy_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vn_d, FO_length * sizeof(double) );
    }
  }

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
  cudaMalloc( (void**) &dN_pTdpTdphidy_d, spectrum_size * sizeof(double) );


  // check if any errors in device memory allocation  
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in device memory allocation: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;  // reset to success by default?
  }

  //copy info from host to device
  cout << "Copying Freezeout and Particle data from host to device" << endl;

  cudaMemcpy( Mass_d,   	Mass,   number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Sign_d,   	Sign,   number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Degen_d,   	Degen,  number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Baryon_d,   Baryon, number_of_chosen_particles * sizeof(double),      cudaMemcpyHostToDevice );

  cudaMemcpy( pT_d,   	pT,   pT_tab_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( trig_d, trig,   2 * phi_tab_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( y_d,   	   yValues,   y_pts  * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( eta_values_d,   	   etaValues,   eta_pts * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( etaDeltaWeights_d,   	   etaDeltaWeights,   eta_pts * sizeof(double),   cudaMemcpyHostToDevice );

  cudaMemcpy( T_d,     T, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( P_d,     P, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( E_d,     E, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( tau_d, tau, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( eta_d, eta, FO_length * sizeof(double),   cudaMemcpyHostToDevice );

  cudaMemcpy( ut_d,   ut, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( ux_d,   ux, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( uy_d,   uy, FO_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( un_d,   un, FO_length * sizeof(double),   cudaMemcpyHostToDevice );

  cudaMemcpy( dat_d,  dat, FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( dax_d,  dax, FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( day_d,  day, FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( dan_d,  dan, FO_length * sizeof(double), cudaMemcpyHostToDevice );

  cudaMemcpy( pitt_d, pitt,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pitx_d, pitx,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pity_d, pity,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pitn_d, pitn,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pixx_d, pixx,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pixy_d, pixy,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pixn_d, pixn,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( piyy_d, piyy,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( piyn_d, piyn,FO_length * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( pinn_d, pinn,FO_length * sizeof(double), cudaMemcpyHostToDevice );

  cudaMemcpy( bulkPi_d,bulkPi, FO_length * sizeof(double), cudaMemcpyHostToDevice );

  if (INCLUDE_BARYON)
  {
    cudaMemcpy( muB_d,    muB,  FO_length * sizeof(double),   cudaMemcpyHostToDevice );
    if (INCLUDE_BARYONDIFF_DELTAF)
    {
      cudaMemcpy( nB_d, nB, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vt_d, Vt, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vx_d, Vx, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vy_d, Vy, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vn_d, Vn, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    }
  }

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

  //cudaMemcpy( dN_pTdpTdphidy_d, dN_pTdpTdphidy, spectrum_size * sizeof(double), cudaMemcpyHostToDevice ); //this is an empty array, we shouldn't need to memcpy it?
  cudaMemset(dN_pTdpTdphidy_d, 0.0, spectrum_size * sizeof(double));

  // df_coeff array:
  //  - holds {c0,c1,c2,c3,c4}              (14-moment vhydro)
  //      or  {F,G,betabulk,betaV,betapi}   (CE vhydro, modified vhydro)

  double df_coeff[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  // 14-moment vhydro
  if(DF_MODE == 1)
  {
    df_coeff[0] = df.c0;
    df_coeff[1] = df.c1;
    df_coeff[2] = df.c2;
    df_coeff[3] = df.c3;
    df_coeff[4] = df.c4;

    // print coefficients
    cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << "\tc3 = " << df_coeff[3] << "\tc4 = " << df_coeff[4] << endl;
  }

  //copy delta_f coefficients to the device array
  double *df_coeff_d;
  cudaMalloc( (void**) &df_coeff_d, 5 * sizeof(double) );
  cudaMemcpy( df_coeff_d, df_coeff, 5 * sizeof(double), cudaMemcpyHostToDevice );



  // check if memory copy error
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in a cudaMemcpy: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  //Perform kernels, first calculation of integrand and reduction across threads, then a reduction across blocks
  double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
  if(debug) cout << "Starting First Kernel : thread reduction" << endl;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaDeviceSynchronize();

  if (MODE == 1) calculate_dN_pTdpTdphidy<<<blocks_ker1, threadsPerBlock>>>(FO_length, number_of_chosen_particles, pT_tab_length, phi_tab_length, y_pts, eta_pts,
                                                              dN_pTdpTdphidy_d, pT_d, trig_d, y_d, eta_values_d, etaDeltaWeights_d,
                                                              Mass_d, Sign_d, Degen_d, Baryon_d,
                                                              T_d, P_d, E_d, tau_d, eta_d, ut_d, ux_d, uy_d, un_d,
                                                              dat_d, dax_d, day_d, dan_d,
                                                              pitt_d, pitx_d, pity_d, pitn_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, pinn_d, bulkPi_d,
                                                              muB_d, nB_d, Vt_d, Vx_d, Vy_d, Vn_d, prefactor, df_coeff_d,
                                                              INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DIMENSION);

  else if (MODE == 2) calculate_dN_pTdpTdphidy_VAH_PL<<<blocks_ker1, threadsPerBlock>>>(FO_length, number_of_chosen_particles, pT_tab_length, phi_tab_length, y_pts, eta_pts,
                                                                                        dN_pTdpTdphidy_d, pT_d, trig_d, y_d, eta_values_d, etaDeltaWeights_d,
                                                                                        Mass_d, Sign_d, Degen_d, Baryon_d,
                                                                                        T_d, P_d, E_d, tau_d, eta_d, ut_d, ux_d, uy_d, un_d,
                                                                                        dat_d, dax_d, day_d, dan_d,
                                                                                        pitt_d, pitx_d, pity_d, pitn_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, pinn_d, bulkPi_d,
                                                                                        Wx_d, Wy_d, Lambda_d, aL_d, c0_d, c1_d, c2_d, c3_d, c4_d, prefactor,
                                                                                        REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, DIMENSION);


  // synchonize kernels after calculation finished
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in first kernel: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }


  // not sure what's going on here
  cout << "Starting Second kernel : block reduction" << endl;
  cudaDeviceSynchronize();
  blockReduction<<<blocks_ker2, threadsPerBlock>>>(dN_pTdpTdphidy_d, final_spectrum_size, blocks_ker1);
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


  //Copy final spectra for all particles from device to host
  cout << "Copying spectra from device to host" << endl;
  //cudaMemcpy( dN_pTdpTdphidy, dN_pTdpTdphidy_d, final_spectrum_size * sizeof(double), cudaMemcpyDeviceToHost );
  cudaMemcpy( dN_pTdpTdphidy, dN_pTdpTdphidy_d, final_spectrum_size * sizeof(double), cudaMemcpyDeviceToHost );

  //write the results to disk
  write_dN_pTdpTdphidy_toFile();

  //Clean up on host and device
  cout << "Deallocating host and device memory" << endl;

  free(Mass);
  free(Sign);
  free(Degen);
  free(Baryon);

  free(T);
  free(P);
  free(E);
  free(tau);
  free(eta);

  free(ut);
  free(ux);
  free(uy);
  free(un);

  free(dat);
  free(dax);
  free(day);
  free(dan);

  free(pitt);
  free(pitx);
  free(pity);
  free(pitn);
  free(pixx);
  free(pixy);
  free(pixn);
  free(piyy);
  free(piyn);
  free(pinn);

  free(bulkPi);

  if (INCLUDE_BARYON)
  {
    free(muB);
    if (INCLUDE_BARYONDIFF_DELTAF)
    {
      free(nB);
      free(Vt);
      free(Vx);
      free(Vy);
      free(Vn);
    }
  }

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

  // cudaFree = deallocate memory on device

  cudaFree(Mass_d);
  cudaFree(Sign_d);
  cudaFree(Degen_d);
  cudaFree(Baryon_d);

  cudaFree(T_d);
  cudaFree(P_d);
  cudaFree(E_d);
  cudaFree(tau_d);
  cudaFree(eta_d);

  cudaFree(ut_d);
  cudaFree(ux_d);
  cudaFree(uy_d);
  cudaFree(un_d);

  cudaFree(dat_d);
  cudaFree(dax_d);
  cudaFree(day_d);
  cudaFree(dan_d);

  cudaFree(pitt_d);
  cudaFree(pitx_d);
  cudaFree(pity_d);
  cudaFree(pitn_d);
  cudaFree(pixx_d);
  cudaFree(pixy_d);
  cudaFree(pixn_d);
  cudaFree(piyy_d);
  cudaFree(piyn_d);
  cudaFree(pinn_d);

  cudaFree(bulkPi_d);

  cudaFree( dN_pTdpTdphidy_d );

  if (INCLUDE_BARYON)
  {
    cudaFree(muB_d);
    if (INCLUDE_BARYONDIFF_DELTAF)
    {
      cudaFree(nB_d);
      cudaFree(Vt_d);
      cudaFree(Vx_d);
      cudaFree(Vy_d);
      cudaFree(Vn_d);
    }
  }

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
}
