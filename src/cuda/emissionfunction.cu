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

#define AMOUNT_OF_OUTPUT 0    // smaller value means less outputs
#define THREADS_PER_BLOCK 128  // try optimizing this, also we can define two different threads/block for the separate kernels
#define FO_CHUNK 12800

using namespace std;

// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in, Deltaf_Data * df_data_in)
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
      printf("Chosen particles error: %d not found in pdg.dat\n", mc_id);
      exit(-1);
    }
  }


  // df coefficients
  df_data = df_data_in;


  // particle spectra of chosen particle species
  momentum_length = pT_tab_length * phi_tab_length * y_tab_length;
  spectra_length = npart * pT_tab_length * phi_tab_length * y_tab_length;
  dN_pTdpTdphidy_momentum = (double*)calloc(momentum_length, sizeof(double));
  dN_pTdpTdphidy_spectra = (double*)calloc(spectra_length, sizeof(double));
}


EmissionFunctionArray::~EmissionFunctionArray()
{

}

__global__ void calculate_dN_pTdpTdphidy_threadReduction(double* dN_pTdpTdphidy_d_blocks, long endFO, long momentum_length, long pT_tab_length, long phi_tab_length, long y_tab_length, long eta_tab_length, double* pT_d, double* trig_d, double* y_d, double* etaValues_d, double* etaWeights_d, double mass_squared, double sign, double prefactor, double baryon, double *T_d, double *P_d, double *E_d, double *tau_d, double *eta_d, double *ux_d, double *uy_d, double *un_d, double *dat_d, double *dax_d, double *day_d, double *dan_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *bulkPi_d, double *alphaB_d, double *nB_d, double *Vx_d, double *Vy_d, double *Vn_d, deltaf_coefficients *df_coeff_d, int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DIMENSION, int OUTFLOW, int DF_MODE)
{

  // contribution of dN_pTdpTdphidy from each thread
  __shared__ double dN_pTdpTdphidy_thread[THREADS_PER_BLOCK];

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
    double E = E_d[icell];            // energy density
    double P = P_d[icell];            // pressure

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
    double nB = 0.0;                  // net baryon density
    double Vt = 0.0;                  // V^mu
    double Vx = 0.0;                  // enforce orthogonality V.u = 0
    double Vy = 0.0;
    double Vn = 0.0;
    double chem = 0.0;
    double baryon_enthalpy_ratio = 0.0;

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      alphaB = alphaB_d[icell];
      chem = baryon * alphaB;
      nB = nB_d[icell];
      Vx = Vx_d[icell];
      Vy = Vy_d[icell];
      Vn = Vn_d[icell];
      Vt = (Vx * ux  +  Vy * uy  +  Vn * tau2_un) / ut;
      baryon_enthalpy_ratio = nB / (E + P);
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

        double Vx_px = Vx * px;
        double Vy_py = Vy * py;

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
            long iP_block = iP  +  blockIdx.x * momentum_length;

            dN_pTdpTdphidy_d_blocks[iP_block] += dN_pTdpTdphidy_thread[0];
          }

        } // iy

      } // iphip

    } // ipT

  } // icell < endFO

} // end function


// does a block reduction, where the previous kernel did a thread reduction.
__global__ void calculate_dN_pTdpTdphidy_blockReduction(double* dN_pTdpTdphidy_d_blocks, long spectra_length, long blocks_ker1)
{
  long ithread = threadIdx.x  +  blockDim.x * blockIdx.x;

  // each thread is assigned a momentum / particle coordinate
  if(ithread < spectra_length)
  {
    // sum spectra contributions of the blocks from first kernel
    for(long iblock_ker1 = 1; iblock_ker1 < blocks_ker1; iblock_ker1++)
    {
      dN_pTdpTdphidy_d_blocks[ithread] += dN_pTdpTdphidy_d_blocks[ithread  +  iblock_ker1 * spectra_length];
    }
    if(isnan(dN_pTdpTdphidy_d_blocks[ithread]))
    {
      printf("found dN_pTdpTdphidy_d nan \n");
    }
  }
}



void EmissionFunctionArray::write_dN_2pipTdpTdy_toFile(long *MCID)
{
  char filename[255] = "";
  printf("Writing thermal dN_2pipTdpTdy to file...\n");

  for(long ipart = 0; ipart < npart; ipart++)
  {
    sprintf(filename, "results/continuous/dN_2pipTdpTdy_%d.dat", MCID[ipart]);
    ofstream spectra(filename, ios_base::out);

    for(long iy = 0; iy < y_tab_length; iy++)
    {
      double y;
      if(DIMENSION == 2) y = 0.0;
      else y = y_tab->get(1, iy + 1);

      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        double pT =  pT_tab->get(1, ipT + 1);

        double dN_2pipTdpTdy = 0.0;

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1, iphip + 1);
          double phip_weight = phi_tab->get(2, iphip + 1);

          long iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

          dN_2pipTdpTdy += phip_weight * dN_pTdpTdphidy_spectra[iS3D] / two_pi;

        } //iphip

        spectra << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT << "\t" << dN_2pipTdpTdy << "\n";
      } //ipT

      if(iy < y_tab_length - 1) spectra << "\n";
    } //iy

    spectra.close();
  }
}


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
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy_spectra[iS3D] << "\n";
        } //ipT
        spectraFile << "\n";
      } //iphip
    } //iy
  }//ipart
  spectraFile.close();
}


void EmissionFunctionArray::calculate_spectra()
{
  cout << "calculate_spectra() has started... " << endl;

  cudaDeviceSynchronize();    // test synchronize device
  
  if(cudaGetLastError() != cudaSuccess)
  {
    printf("Error at very beginning: %s\n", cudaGetErrorString(cudaGetLastError()));
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

  long threadsPerBlock = THREADS_PER_BLOCK;
  long FO_chunk = FO_CHUNK;

  long blocks_ker1 = (FO_chunk + threadsPerBlock - 1) / threadsPerBlock;        // number of blocks in kernel 1 (thread reduction)
  long blocks_ker2 = (momentum_length + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in kernel 2 (block reduction)

  long chunks = (FO_length + FO_chunk - 1) / FO_chunk;
  long partial = FO_length % FO_chunk;

  long block_momentum_length = blocks_ker1 * momentum_length;

  cout << "________________________|______________" << endl;
  cout << "chosen_particles        |\t" << npart << endl;
  cout << "pT_tab_length           |\t" << pT_tab_length << endl;
  cout << "phi_tab_length          |\t" << phi_tab_length << endl;
  cout << "y_tab_length            |\t" << y_tab_length << endl;
  cout << "eta_tab_length          |\t" << eta_tab_length << endl;
  cout << "total spectra length    |\t" << spectra_length << endl;
  cout << "momentum length         |\t" << momentum_length << endl;
  cout << "block momentum length   |\t" << block_momentum_length << endl;
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
  double *T, *P, *E, *alphaB, *nB;            // thermodynamic properties
  double *tau, *eta;                          // position
  double *ux, *uy, *un;                       // u^mu
  double *dat, *dax, *day, *dan;              // dsigma_mu
  double *pixx, *pixy, *pixn, *piyy, *piyn;   // pi^munu
  double *bulkPi;                             // bulk pressure
  double *Vx, *Vy, *Vn;                       // V^mu

  deltaf_coefficients *df_coeff;

  // callocate memory
  T = (double*)calloc(FO_chunk, sizeof(double));
  P = (double*)calloc(FO_chunk, sizeof(double)); // replace with df coefficients
  E = (double*)calloc(FO_chunk, sizeof(double));
  tau = (double*)calloc(FO_chunk, sizeof(double));

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
    nB = (double*)calloc(FO_chunk, sizeof(double));
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

  for(long ipT = 0; ipT < pT_tab_length; ipT++)
  {
    pT[ipT] = pT_tab->get(1, ipT + 1);
  }

  for(long iphip = 0; iphip < phi_tab_length; iphip++)
  {
    double phip = phi_tab->get(1, iphip + 1);

    trig[iphip] = cos(phip);
    trig[iphip + phi_tab_length] = sin(phip);
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


  cout << "Declaring and allocating device arrays " << endl;

  double *T_d, *P_d, *E_d, *alphaB_d, *nB_d;
  double *tau_d, *eta_d;
  double *ux_d, *uy_d, *un_d;
  double *dat_d, *dax_d, *day_d, *dan_d;
  double *pixx_d, *pixy_d, *pixn_d, *piyy_d, *piyn_d;
  double *bulkPi_d;
  double *Vx_d, *Vy_d, *Vn_d;

  deltaf_coefficients *df_coeff_d;

  double *pT_d, *trig_d, *yp_d, *etaValues_d, *etaWeights_d;

  double *dN_pTdpTdphidy_d_blocks;


  // allocate memory on device
  cudaMalloc((void**) &T_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &P_d, FO_chunk * sizeof(double));
  cudaMalloc((void**) &E_d, FO_chunk * sizeof(double));

  cudaMalloc((void**) &tau_d, FO_chunk * sizeof(double));
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
    cudaMalloc((void**) &nB_d, FO_chunk * sizeof(double));
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

  cudaMalloc((void**) &dN_pTdpTdphidy_d_blocks, block_momentum_length * sizeof(double));

  if(cudaGetLastError() != cudaSuccess)
  {
    printf("Error in device memory allocation: %s\n", cudaGetErrorString(cudaGetLastError()));
  }

  printf("Copying momentum tables from host to device...\n");
  cudaMemcpy(pT_d, pT, pT_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(trig_d, trig, 2 * phi_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(yp_d, yp, y_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(etaValues_d, etaValues, eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(etaWeights_d, etaWeights, eta_tab_length * sizeof(double), cudaMemcpyHostToDevice);


  if(cudaGetLastError() != cudaSuccess)
  {
    printf("Error in memory copy from host to device: %s\n", cudaGetErrorString(cudaGetLastError()));
  }


  printf("\n");
  Stopwatch sw;
  sw.tic();

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

      // evaluate the df coefficients
      double T1 = T[icell];
      double E1 = E[icell];
      double P1 = P[icell];
      double muB1 = 0.0;
      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        muB1 = T1 * alphaB[icell];
      }
      double bulkPi1 = 0.0;
      if(INCLUDE_BULK_DELTAF)
      {
        bulkPi1 = bulkPi[icell];
      }

      df_coeff[icell] = df_data->evaluate_df_coefficients(T1, muB1, E1, P1, bulkPi1);

    } // icell


    // copy memory from host to device
    cudaMemcpy(T_d, T, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(E_d, E, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(tau_d, tau, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
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
      cudaMemcpy(nB_d, nB, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(Vx_d, Vx, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(Vy_d, Vy, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(Vn_d, Vn, FO_chunk * sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(df_coeff_d, df_coeff, FO_chunk * sizeof(deltaf_coefficients), cudaMemcpyHostToDevice);

    if(cudaGetLastError() != cudaSuccess)
    {
      printf("Error in memory copy from host to device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    // loop over particles
    for(int ipart = 0; ipart < npart; ipart++)
    {
      // reset momentum spectra of particle to zero
      cudaMemset(dN_pTdpTdphidy_d_blocks, 0.0, block_momentum_length * sizeof(double));

      double mass = Mass[ipart];
      double sign = Sign[ipart];
      double degen = Degen[ipart];
      double baryon = Baryon[ipart];

      double mass_squared = mass * mass;
      double prefactor = degen / h3;

      // launch thread reduction kernel
      cudaDeviceSynchronize();
      calculate_dN_pTdpTdphidy_threadReduction<<<blocks_ker1, threadsPerBlock>>>(dN_pTdpTdphidy_d_blocks, endFO, momentum_length, pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length, pT_d, trig_d, yp_d, etaValues_d, etaWeights_d, mass_squared, sign, prefactor, baryon, T_d, P_d, E_d, tau_d, eta_d, ux_d, uy_d, un_d, dat_d, dax_d, day_d, dan_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, bulkPi_d, alphaB_d, nB_d, Vx_d, Vy_d, Vn_d, df_coeff_d, INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DIMENSION, OUTFLOW, DF_MODE);
      cudaDeviceSynchronize();

      if(cudaGetLastError() != cudaSuccess)
      {
        printf("Error in thread reduction kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
      }

      // launch block reduction kernel
      cudaDeviceSynchronize();
      calculate_dN_pTdpTdphidy_blockReduction<<<blocks_ker2, threadsPerBlock>>>(dN_pTdpTdphidy_d_blocks, momentum_length, blocks_ker1);
      cudaDeviceSynchronize();

      if(cudaGetLastError() != cudaSuccess)
      {
        printf("Error in block reduction kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
      }

      // copy chunk contribution to particle's spectra from device to host
      cudaMemcpy(dN_pTdpTdphidy_momentum, dN_pTdpTdphidy_d_blocks, momentum_length * sizeof(double), cudaMemcpyDeviceToHost);

      if(cudaGetLastError() != cudaSuccess)
      {
        printf("Error in memory copy from device to host: %s\n", cudaGetErrorString(cudaGetLastError()));
      }

      // amend particle spectra to the total
      for(int ipT = 0; ipT < pT_tab_length; ipT++)
      {
        for(int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          for(int iy = 0; iy < y_tab_length; iy++)
          {
            long iP = ipT  +  pT_tab_length * (iphip  +  phi_tab_length * iy);

            long iS3D = ipart  +  npart * iP;

            dN_pTdpTdphidy_spectra[iS3D] += dN_pTdpTdphidy_momentum[iP];
          }
        }
      } // iP

    } // ipart

    printf("Progress: finished chunk %ld of %ld", n + 1, chunks);
    printf("\n");

  } // loop over chunks

  sw.toc();
  cout << "\nKernel launches took " << sw.takeTime() << " seconds.\n" << endl;


  // write the results to disk
  write_dN_2pipTdpTdy_toFile(MCID);


  cout << "Deallocating host and device memory" << endl;

  free(dN_pTdpTdphidy_spectra);
  free(dN_pTdpTdphidy_momentum);
  free(chosen_particles_table);

  free(Mass);
  free(Sign);
  free(Degen);
  free(Baryon);
  free(MCID);

  free(T);
  free(P);
  free(E);

  free(tau);
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
    free(nB);
    free(Vx);
    free(Vy);
    free(Vn);
  }

  free(df_coeff);

  // cudaFree = deallocate memory on device
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

  cudaFree(df_coeff_d);

  cudaFree(pT_d);
  cudaFree(trig_d);
  cudaFree(yp_d);
  cudaFree(etaValues_d);
  cudaFree(etaWeights_d);
  cudaFree(dN_pTdpTdphidy_d_blocks);

}


