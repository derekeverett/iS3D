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

#include <cuda.h>
#include <cuda_runtime.h>

#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs
#define threadsPerBlock 512 //try optimizing this, also we can define two different threads/block for the separate kernels
#define debug 1	//1 for debugging, 0 otherwise

using namespace std;

// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, particle_info* particles_in, int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in)
{
  paraRdr = paraRdr_in;
  pT_tab = pT_tab_in;
  pT_tab_length = pT_tab->getNumberOfRows();
  phi_tab = phi_tab_in;
  phi_tab_length = phi_tab->getNumberOfRows();
  y_tab = y_tab_in;
  y_tab_length = y_tab->getNumberOfRows();

  // get control parameters
  INCLUDE_BARYON = paraRdr->getVal("include_baryon");
  INCLUDE_BULK_DELTAF = paraRdr->getVal("include_bulk_deltaf");
  INCLUDE_SHEAR_DELTAF = paraRdr->getVal("include_shear_deltaf");
  INCLUDE_BARYONDIFF_DELTAF = paraRdr->getVal("include_baryondiff_deltaf");
  GROUP_PARTICLES = paraRdr->getVal("group_particles");
  PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");

  particles = particles_in;
  Nparticles = Nparticles_in;
  surf_ptr = surf_ptr_in;
  FO_length = FO_length_in;
  number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

  chosen_particles_01_table = new int[Nparticles];
  //a class member to hold 3D spectra for all chosen particles
  dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];

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

__global__ void calculate_dN_pTdpTdphidy( long FO_length, int number_of_chosen_particles, int pT_tab_length, int phi_tab_length, int y_tab_length,
  double* dN_pTdpTdphidy_d, double* pT_d, double* trig_d, double* y_d,
  double *Mass_d, double *Sign_d, double *Degen_d, double *Baryon_d,
  double *T_d, double *P_d, double *E_d, double *tau_d, double *eta_d, double *ut_d, double *ux_d, double *uy_d, double *un_d,
  double *dat_d, double *dax_d, double *day_d, double *dan_d,
  double *pitt_d, double *pitx_d, double *pity_d, double *pitn_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *pinn_d, double *bulkPi_d,
  double *muB_d, double *Vt_d, double *Vx_d, double *Vy_d, double *Vn_d, double prefactor, int INCLUDE_BARYON, int INCLUDE_BARYONDIFF_DELTAF)
  {
    //This array is a shared array that will contain the integration contributions from each cell.
    __shared__ double temp[threadsPerBlock];
    //Assign a global index and a local index
    //int idx_glb = threadIdx.x + blockDim.x * blockIdx.x;
    int icell = threadIdx.x;
    __syncthreads();

    for (long imm = 0; imm < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; imm++) //each thread <-> FO cell , and each thread loops over all momenta and species
    {
      //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT

      temp[icell] = 0.0;
      if (icell < FO_length) //this index corresponds to the freezeout cell
      {
        //imm = ipT + (iphip * (pT_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (ipart * (pT_tab_length * phi_tab_length * y_tab_length))
        int ipart       = imm / (pT_tab_length * phi_tab_length * y_tab_length);
        int iy          = (imm - (ipart * pT_tab_length * phi_tab_length * y_tab_length) ) / (pT_tab_length * phi_tab_length);
        int iphip       = (imm - (ipart * pT_tab_length * phi_tab_length * y_tab_length) - (iy * pT_tab_length * phi_tab_length) ) / pT_tab_length;
        int ipT         = imm - ( (ipart * (pT_tab_length * phi_tab_length * y_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (iphip * (pT_tab_length)) );
        double px       = pT_d[ipT] * trig_d[ipT + phi_tab_length];
        double py       = pT_d[ipT] * trig_d[ipT];
        double mT       = sqrt(Mass_d[ipart] * Mass_d[ipart] + pT_d[ipT] * pT_d[ipT]);
        double y        = y_d[iy];
        double pt       = mT * cosh(y - eta_d[icell]); //contravariant
        double pn       = (1.0 / tau_d[icell]) * mT * sinh(y - eta_d[icell]); //contravariant

        //thermal equilibrium distributions - for viscous hydro
        double pdotu = pt * ut_d[icell] - px * ux_d[icell] - py * uy_d[icell] - (tau_d[icell] * tau_d[icell]) * pn * un_d[icell]; //watch factors of tau from metric!
        double baryon_factor = 0.0;
        if (INCLUDE_BARYON) baryon_factor = Baryon_d[ipart] * muB_d[icell];
        double exponent = (pdotu - baryon_factor) / T_d[icell];
        double f0 = 1. / (exp(exponent) + Sign_d[ipart]);
        double pdotdsigma = pt * dat_d[icell] + px * dax_d[icell] + py * day_d[icell] + pn * dan_d[icell]; //CHECK THIS !

        //viscous corrections
        double delta_f_shear = 0.0;
        double tau2 = tau_d[icell] * tau_d[icell];
        double tau4 = tau2 * tau2;
        //double pimunu_pmu_pnu = (pt * pt * pitt[icell_glb] - 2.0 * pt * px * pitx[icell_glb] - 2.0 * pt * py * pity[icell_glb] + px * px * pixx[icell_glb] + 2.0 * px * py * pixy[icell_glb] + py * py * piyy[icell_glb] + pn * pn * pinn[icell_glb]);
        double shear_deltaf_prefactor = 1.0 / (2.0 * T_d[icell] * T_d[icell] * (E_d[icell] + P_d[icell]));
        double pimunu_pmu_pnu = pitt_d[icell] * pt * pt + pixx_d[icell] * px * px + piyy_d[icell] * py * py + pinn_d[icell] * pn * pn * (1.0 / tau4)
        + 2.0 * (-pitx_d[icell] * pt * px - pity_d[icell] * pt * py - pitn_d[icell] * pt * pn * (1.0 / tau2) + pixy_d[icell] * px * py + pixn_d[icell] * px * pn * (1.0 / tau2) + piyn_d[icell] * py * pn * (1.0 / tau2));
        delta_f_shear = ((1.0 - Sign_d[ipart] * f0) * pimunu_pmu_pnu * shear_deltaf_prefactor);

        double delta_f_bulk = 0.0;
        //put fourteen moment expression for bulk viscosity (\delta)f here

        double delta_f_baryondiff = 0.0;
        //put fourteen moment expression for baryon diffusion (\delta)f here

        //WHAT IS RATIO - CHANGE THIS ?
        double ratio = min(1., fabs(1. / (delta_f_shear + delta_f_bulk + delta_f_baryondiff)));
        double result = (prefactor * Degen_d[ipart] * pdotdsigma * tau_d[icell] * f0 * (1. + (delta_f_shear + delta_f_bulk + delta_f_baryondiff) * ratio));
        temp[icell] += result;

      }//if(icell < FO_length)

      int N = blockDim.x;
      __syncthreads(); //Make sure threads are prepared for reduction
      do
      {
        //Here N must be a power of two. Try reducing by powers of 2, 4, 6 etc...
        N /= 2;
        if (icell < N) temp[icell] += temp[icell + N];
        __syncthreads();//Test if this is needed
      } while(N != 1);

      long spectra_size = number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length;
      if (icell == 0) dN_pTdpTdphidy_d[blockIdx.x * spectra_size + imm] = temp[0];
    }
  }

//Does a block reduction, where the previous kernel did a thread reduction.
__global__ void blockReduction(double* dN_pTdpTdphidy_d, int final_spectrum_size, int blocks_ker1)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < final_spectrum_size)
  {
    if (blocks_ker1 == 1) return; //Probably will never happen, but best to be careful
    //Need to start at i=1, since adding everything to i=0
    for (int i = 1; i < blocks_ker1; i++) dN_pTdpTdphidy_d[idx] += dN_pTdpTdphidy_d[idx + i * final_spectrum_size];
  }
}


void EmissionFunctionArray::write_dN_pTdpTdphidy_toFile()
{
  printf("writing to file\n");
  //write 3D spectra in block format, different blocks for different species,
  //different sublocks for different values of rapidity
  //rows corespond to phip and columns correspond to pT

  //is there a better / less confusing format for this file?
  int npart = number_of_chosen_particles;
  char filename[255] = "";
  sprintf(filename, "results/dN_pTdpTdphidy_block.dat");
  ofstream spectraFileBlock(filename, ios_base::app);
  for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
  {
    for (int iy = 0; iy < y_tab_length; iy++)
    {
      for (int iphip = 0; iphip < phi_tab_length; iphip++)
      {
        for (int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
          spectraFileBlock << scientific <<  setw(15) << setprecision(8) << dN_pTdpTdphidy[is] << "\t";
        } //ipT
        spectraFileBlock << "\n";
      } //iphip
    } //iy
  }//ipart
  spectraFileBlock.close();

  sprintf(filename, "results/dN_pTdpTdphidy.dat");
  ofstream spectraFile(filename, ios_base::app);
  for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
  {
    for (int iy = 0; iy < y_tab_length; iy++)
    {
      for (int iphip = 0; iphip < phi_tab_length; iphip++)
      {
        for (int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double y = y_tab->get(1,iy + 1);
          double pT = pT_tab->get(1,ipT + 1);
          double phip = phi_tab->get(1,iphip + 1);
          long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[is] << "\n";
        } //ipT
        spectraFile << "\n";
      } //iphip
    } //iy
  }//ipart
  spectraFile.close();
}


void EmissionFunctionArray::calculate_spectra()
{
  cudaDeviceSynchronize();
  cudaError_t err;

  cout << "calculate_spectra() has started... " << endl;
  Stopwatch sw;
  sw.tic();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error at very beginning: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  int  blocks_ker1 = (FO_length + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the first kernel (thread reduction)
  long spectrum_size = blocks_ker1 * number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; //the size of the spectrum which has been intrablock reduced, but not interblock reduced
  int  final_spectrum_size = number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; //size of final array for all particles , as a function of particle, pT and phi and y
  int  blocks_ker2 = (final_spectrum_size + threadsPerBlock - 1) / threadsPerBlock; // number of blocks in the second kernel (block reduction)

  cout << "# of chosen particles   = " << number_of_chosen_particles << endl;
  cout << "FO_length               = " << FO_length            << endl;
  cout << "pT_tab_length           = " << pT_tab_length        << endl;
  cout << "phi_tab_length          = " << phi_tab_length       << endl;
  cout << "y_tab_length            = " << y_tab_length         << endl;
  cout << "unreduced spectrum size = " << spectrum_size        << endl; //?
  cout << "reduced spectrum size   = " << final_spectrum_size  << endl; //?
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
  double *muB, *Vt, *Vx, *Vy, *Vn;
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
      //contravariant baryon diffusion current
      Vt = (double*)calloc(FO_length, sizeof(double));
      Vx = (double*)calloc(FO_length, sizeof(double));
      Vy = (double*)calloc(FO_length, sizeof(double));
      Vn = (double*)calloc(FO_length, sizeof(double));
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
        Vt[icell] = surf->Vt;
        Vx[icell] = surf->Vx;
        Vy[icell] = surf->Vy;
        Vn[icell] = surf->Vn;
      }
    }
  }

  //this array holds the 3D spectra for all species
  double *dN_pTdpTdphidy;
  dN_pTdpTdphidy = (double*)malloc(spectrum_size * sizeof(double));
  for (int i = 0; i < spectrum_size; i++) dN_pTdpTdphidy[i] = 0.0;

  double *pT, *trig, *y;
  pT   = (double*)calloc(pT_tab_length,      sizeof(double));
  trig = (double*)calloc(2 * phi_tab_length, sizeof(double));
  y    = (double*)calloc(y_tab_length,       sizeof(double));

  for (int i = 0; i < pT_tab_length; i++) pT[i] = pT_tab->get(1, i+1);
  for (int i = 0; i < phi_tab_length; i++)
  {
    trig[i] = sin( phi_tab->get(1, i+1) );
    trig[i + phi_tab_length] = cos( phi_tab->get(1, i+1) );
  }
  for (int i = 0; i < y_tab_length; i++) y[i] = y_tab->get(1, i+1);

  cout << "Declaring and Allocating device arrays " << endl;
  double *Mass_d, *Sign_d, *Degen_d, *Baryon_d;
  double *pT_d, *trig_d, *y_d;
  double *T_d, *P_d, *E_d, *tau_d, *eta_d, *ut_d, *ux_d, *uy_d, *un_d;
  double *dat_d, *dax_d, *day_d, *dan_d;
  double *pitt_d, *pitx_d, *pity_d, *pitn_d, *pixx_d, *pixy_d, *pixn_d, *piyy_d, *piyn_d, *pinn_d, *bulkPi_d;
  double *muB_d, *Vt_d, *Vx_d, *Vy_d, *Vn_d;
  double *dN_pTdpTdphidy_d;

  //Allocate memory on device
  cudaMalloc( (void**) &Mass_d,   number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Sign_d,   number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Degen_d,  number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Baryon_d, number_of_chosen_particles * sizeof(int)    );

  cudaMalloc( (void**) &pT_d,     pT_tab_length * sizeof(double)              );
  cudaMalloc( (void**) &trig_d,   2 * phi_tab_length * sizeof(double)         );
  cudaMalloc( (void**) &y_d,      y_tab_length * sizeof(double)               );

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
      cudaMalloc( (void**) &Vt_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vx_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vy_d, FO_length * sizeof(double) );
      cudaMalloc( (void**) &Vn_d, FO_length * sizeof(double) );
    }
  }

  cudaMalloc( (void**) &dN_pTdpTdphidy_d, spectrum_size * sizeof(double) );
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in device memory allocation: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  //copy info from host to device
  cout << "Copying data from host to device" << endl;

  cudaMemcpy( Mass_d,   	Mass,   number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Sign_d,   	Sign,   number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Degen_d,   	Degen,  number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Baryon_d,   Baryon, number_of_chosen_particles * sizeof(int),      cudaMemcpyHostToDevice );

  cudaMemcpy( pT_d,   	pT,   pT_tab_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( trig_d, trig,   2 * phi_tab_length * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( y_d,   	   y,   y_tab_length * sizeof(double),   cudaMemcpyHostToDevice );

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
      cudaMemcpy( Vt_d, Vt, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vx_d, Vx, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vy_d, Vy, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vn_d, Vn, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    }
  }
  cudaMemcpy( dN_pTdpTdphidy_d, dN_pTdpTdphidy, spectrum_size * sizeof(double), cudaMemcpyHostToDevice ); //this is an empty array, we shouldn't need to memcpy it?

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

  calculate_dN_pTdpTdphidy<<<blocks_ker1, threadsPerBlock>>>(FO_length, number_of_chosen_particles, pT_tab_length, phi_tab_length, y_tab_length,
                                                              dN_pTdpTdphidy_d, pT_d, trig_d, y_d,
                                                              Mass_d, Sign_d, Degen_d, Baryon_d,
                                                              T_d, P_d, E_d, tau_d, eta_d, ut_d, ux_d, uy_d, un_d,
                                                              dat_d, dax_d, day_d, dan_d,
                                                              pitt_d, pitx_d, pity_d, pitn_d, pixx_d, pixy_d, pixn_d, piyy_d, piyn_d, pinn_d, bulkPi_d,
                                                              muB_d, Vt_d, Vx_d, Vy_d, Vn_d, prefactor, INCLUDE_BARYON, INCLUDE_BARYONDIFF_DELTAF);


  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in first kernel: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

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
  cudaMemcpy( dN_pTdpTdphidy, dN_pTdpTdphidy_d, spectrum_size * sizeof(double), cudaMemcpyDeviceToHost );

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

  free(dN_pTdpTdphidy);

  if (INCLUDE_BARYON)
  {
    free(muB);
    if (INCLUDE_BARYONDIFF_DELTAF)
    {
      free(Vt);
      free(Vx);
      free(Vy);
      free(Vn);
    }
  }

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
      cudaFree(Vt_d);
      cudaFree(Vx_d);
      cudaFree(Vy_d);
      cudaFree(Vn_d);
    }
  }
}
