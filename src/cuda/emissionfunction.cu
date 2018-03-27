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
  MODE = paraRdr->getVal("mode");
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
  number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

  chosen_particles_01_table = new int[Nparticles];
  //a class member to hold 3D spectra for all chosen particles
  dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
  //zero the array
  for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
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

__global__ void calculate_dN_pTdpTdphidy( long FO_length, int number_of_chosen_particles, int pT_tab_length, int phi_tab_length, int y_tab_length,
  double* dN_pTdpTdphidy_d, double* pT_d, double* trig_d, double* y_d,
  double *Mass_d, double *Sign_d, double *Degen_d, double *Baryon_d,
  double *T_d, double *P_d, double *E_d, double *tau_d, double *eta_d, double *ut_d, double *ux_d, double *uy_d, double *un_d,
  double *dat_d, double *dax_d, double *day_d, double *dan_d,
  double *pitt_d, double *pitx_d, double *pity_d, double *pitn_d, double *pixx_d, double *pixy_d, double *pixn_d, double *piyy_d, double *piyn_d, double *pinn_d, double *bulkPi_d,
  double *muB_d, double *nB_d, double *Vt_d, double *Vx_d, double *Vy_d, double *Vn_d, double prefactor, double *df_coeff_d,
  int INCLUDE_BARYON, int REGULATE_DELTAF, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF)
  {
    //This array is a shared array that will contain the integration contributions from each cell.
    __shared__ double temp[threadsPerBlock];
    //Assign a global index and a local index
    int icell = threadIdx.x + blockDim.x * blockIdx.x; //global idx
    int idx_local = threadIdx.x; //local idx
    __syncthreads();

    for (long imm = 0; imm < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; imm++) //each thread <-> FO cell , and each thread loops over all momenta and species
    {
      //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT

      temp[idx_local] = 0.0;
      if (icell < FO_length) //this index corresponds to the freezeout cell
      {

        //get Freezeout cell info
        /******************************************************************/
        double tau = tau_d[icell];         // longitudinal proper time
        double eta = eta_d[icell];         // spacetime rapidity

        double dat = dat_d[icell];         // covariant normal surface vector
        double dax = dax_d[icell];
        double day = day_d[icell];
        double dan = dan_d[icell];

        double ut = ut_d[icell];           // contravariant fluid velocity
        double ux = ux_d[icell];
        double uy = uy_d[icell];
        double un = un_d[icell];

        double T = T_d[icell];             // temperature
        double E = E_d[icell];             // energy density
        double P = P_d[icell];             // pressure

        double pitt = pitt_d[icell];       // pi^munu
        double pitx = pitx_d[icell];
        double pity = pity_d[icell];
        double pitn = pitn_d[icell];
        double pixx = pixx_d[icell];
        double pixy = pixy_d[icell];
        double pixn = pixn_d[icell];
        double piyy = piyy_d[icell];
        double piyn = piyn_d[icell];
        double pinn = pinn_d[icell];

        double bulkPi = bulkPi_d[icell];   // bulk pressure

        double muB = 0.0;                         // baryon chemical potential
        double nB = 0.0;                          // net baryon density
        double Vt = 0.0;                          // baryon diffusion
        double Vx = 0.0;
        double Vy = 0.0;
        double Vn = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_d[icell];

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_d[icell];
            Vt = Vt_d[icell];
            Vx = Vx_d[icell];
            Vy = Vy_d[icell];
            Vn = Vn_d[icell];
          }
        }
        /**************************************************************/

        //get particle info and momenta
        //imm = ipT + (iphip * (pT_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (ipart * (pT_tab_length * phi_tab_length * y_tab_length))
        int ipart       = imm / (pT_tab_length * phi_tab_length * y_tab_length);
        int iy          = (imm - (ipart * pT_tab_length * phi_tab_length * y_tab_length) ) / (pT_tab_length * phi_tab_length);
        int iphip       = (imm - (ipart * pT_tab_length * phi_tab_length * y_tab_length) - (iy * pT_tab_length * phi_tab_length) ) / pT_tab_length;
        int ipT         = imm - ( (ipart * (pT_tab_length * phi_tab_length * y_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (iphip * (pT_tab_length)) );

        // set particle properties
        double mass = Mass_d[ipart];    // (GeV)
        double mass2 = mass * mass;
        double sign = Sign_d[ipart];
        double degeneracy = Degen_d[ipart];
        double baryon = 0.0;
        double chem = 0.0;           // chemical potential term in feq
        if(INCLUDE_BARYON)
        {
          baryon = Baryon_d[ipart];
          chem = baryon * muB;
        }

        double px       = pT_d[ipT] * trig_d[ipT + phi_tab_length];
        double py       = pT_d[ipT] * trig_d[ipT];
        double mT       = sqrt(mass2 + pT_d[ipT] * pT_d[ipT]);
        double y        = y_d[iy];
        double pt       = mT * cosh(y - eta); //contravariant
        double pn       = (1.0 / tau) * mT * sinh(y - eta); //contravariant

        // useful expressions
        double tau2 = tau * tau;
        double tau2_pn = tau2 * pn;
        double shear_coeff = 0.5 / (T * T * ( E + P));  // (see df_shear)

        //momentum vector is contravariant, surface normal vector is COVARIANT
        double pdotdsigma = pt * dat + px * dax + py * day + pn * dan;

        //thermal equilibrium distributions - for viscous hydro
        double pdotu = pt * ut - px * ux - py * uy - tau2_pn * un;    // u.p = LRF energy
        double feq = 1.0 / ( exp(( pdotu - chem ) / T) + sign);

        //viscous corrections
        double feqbar = 1.0 - sign*feq;
        // shear correction:
        double df_shear = 0.0;
        // pi^munu * p_mu * p_nu (factorized and corrected the tau factors I think)
        double pimunu_pmu_pnu = pitt * pt * pt + pixx * px * px + piyy * py * py + pinn * tau2_pn * tau2_pn
        + 2.0 * (-(pitx * px + pity * py) * pt + pixy * px * py + tau2_pn * (pixn * px + piyn * py - pitn * pt));

        if (INCLUDE_SHEAR_DELTAF) df_shear = shear_coeff * pimunu_pmu_pnu;  // df / (feq*feqbar)

        // bulk correction:
        double df_bulk = 0.0;
        if (INCLUDE_BULK_DELTAF)
        {
          double c0 = df_coeff_d[0];
          double c2 = df_coeff_d[2];
          if (INCLUDE_BARYON)
          {
            double c1 = df_coeff_d[1];
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
          double c3 = df_coeff_d[3];
          double c4 = df_coeff_d[4];
          // V^mu * p_mu
          double Vmu_pmu = Vt * pt - Vx * px - Vy * py - Vn * tau2_pn;
          df_baryondiff = (baryon * c3 + c4 * pdotu) * Vmu_pmu;
        }
        double df = df_shear + df_bulk + df_baryondiff;

        double result = 0.0;
        if (REGULATE_DELTAF)
        {
          double reg_df = max( -1.0, min( feqbar * df, 1.0 ) );
          result = (prefactor * degeneracy * pdotdsigma * feq * (1.0 + reg_df));
        }
        else result = (prefactor * degeneracy * pdotdsigma * feq * (1.0 + feqbar * df));
        temp[idx_local] += result;

      }//if(icell < FO_length)

      int N = blockDim.x;
      __syncthreads(); //Make sure threads are prepared for reduction
      do
      {
        //Here N must be a power of two. Try reducing by powers of 2, 4, 6 etc...
        N /= 2;
        if (idx_local < N) temp[idx_local] += temp[idx_local + N];
        __syncthreads();//Test if this is needed
      } while(N != 1);

      long final_spectrum_size = number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length;
      if (idx_local == 0) dN_pTdpTdphidy_d[blockIdx.x * final_spectrum_size + imm] = temp[0];
    } //for (long imm)
  } //calculate_dN_pTdpTdphidy

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
          if (dN_pTdpTdphidy[is] < 1.0e-40) spectraFileBlock << scientific <<  setw(15) << setprecision(8) << 0.0 << "\t";
          else spectraFileBlock << scientific <<  setw(15) << setprecision(8) << dN_pTdpTdphidy[is] << "\t";
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
          long imm = ipT + (iphip * (pT_tab_length)) + (iy * (pT_tab_length * phi_tab_length)) + (ipart * (pT_tab_length * phi_tab_length * y_tab_length));
          //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
          //if (dN_pTdpTdphidy[is] < 1.0e-40) spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << 0.0 << "\n";
          //else spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[is] << "\n";
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[imm] << "\n";
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
  }

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
  double *muB_d, *nB_d, *Vt_d, *Vx_d, *Vy_d, *Vn_d;
  double *dN_pTdpTdphidy_d;

  //Allocate memory on device
  cudaMalloc( (void**) &Mass_d,   number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Sign_d,   number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Degen_d,  number_of_chosen_particles * sizeof(double) );
  cudaMalloc( (void**) &Baryon_d, number_of_chosen_particles * sizeof(double)    );

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
      cudaMalloc( (void**) &nB_d, FO_length * sizeof(double) );
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
  cout << "Copying Freezeout and Particle data from host to device" << endl;

  cudaMemcpy( Mass_d,   	Mass,   number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Sign_d,   	Sign,   number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Degen_d,   	Degen,  number_of_chosen_particles * sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( Baryon_d,   Baryon, number_of_chosen_particles * sizeof(double),      cudaMemcpyHostToDevice );

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
      cudaMemcpy( nB_d, nB, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vt_d, Vt, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vx_d, Vx, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vy_d, Vy, FO_length * sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( Vn_d, Vn, FO_length * sizeof(double), cudaMemcpyHostToDevice );
    }
  }
  cudaMemcpy( dN_pTdpTdphidy_d, dN_pTdpTdphidy, spectrum_size * sizeof(double), cudaMemcpyHostToDevice ); //this is an empty array, we shouldn't need to memcpy it?


  // read in tables of delta_f coefficients (we could make a separate file)

  // coefficient files:
  //  - temperature and chemical potential column ~ fm^-1
  //  - coefficient column ~ fm^n (see their headers)
  // recall freezeout temperature and chemical potential in surface struct ~ GeV

  printf("Reading in delta_f coefficient tables \n");
  double T_invfm = T[0] / hbarC;  // freezeout temperature in fm^-1 (all T[i]'s assumed same)
  double muB_invfm = 0.0;         // added chemical potential in fm^-1
  if(INCLUDE_BARYON) muB_invfm = muB[0] / hbarC;

  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  //                       Coefficients of 14 moment approximation                       ::
  //                                                                                     ::
  // df ~ (c0 + b*c1*(u.p) + c2*(u.p)^2)*Pi + (b*c3 + c4(u.p))p_u*V^u + c5*p_u*p_v*pi^uv ::
  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  // df_coefficient array:
  //  - holds {c0,c1,c2,c3,c4}(T_FO,muB_FO)
  double df_coeff[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  int n_T = 100;                  // number of points in temperature (in coefficient files)
  int n_muB = 1;                  // default number of points in baryon chemical potential
  if(INCLUDE_BARYON) n_muB = 100; // (can change resolution later)

  double T_array[n_T][n_muB];     // temperature (fm^-1) array (increasing with n_T)
  double muB_array[n_T][n_muB];   // baryon chemical potential (fm^-1) array (increasing with n_muB)
  double c0[n_T][n_muB];          // coefficient table
  double c2[n_T][n_muB];

  FILE * c0_file;                 // coefficient files
  FILE * c2_file;
  char c0_name[255];              // coefficient file names
  char c2_name[255];
  char header[300];               // for skipping file header

  // c0, c2 coefficients:
  if(INCLUDE_BARYON)
  {
    sprintf(c0_name, "%s", "deltaf_coefficients/c0_df14_vh.dat");
    sprintf(c2_name, "%s", "deltaf_coefficients/c2_df14_vh.dat");
  }
  else
  {
    sprintf(c0_name, "%s", "deltaf_coefficients/c0_df14_vh_muB_0.dat");
    sprintf(c2_name, "%s", "deltaf_coefficients/c2_df14_vh_muB_0.dat");
  }

  c0_file = fopen(c0_name, "r");
  c2_file = fopen(c2_name, "r");
  if(c0_file == NULL) printf("Couldn't open c0 coefficient file!\n");
  if(c2_file == NULL) printf("Couldn't open c2 coefficient file!\n");

  // skip the headers
  fgets(header, 100, c0_file);
  fgets(header, 100, c2_file);
  cout << "Loading c0 and c2 coefficients.." << endl;

  // scan c0, c2 files
  for(int iB = 0; iB < n_muB; iB++)
  {
    for(int iT = 0; iT < n_T; iT++)
    {
      // set temperature (fm^-1) array from file
      if(INCLUDE_BARYON)
      {
        fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c0[iT][iB]);
        fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c2[iT][iB]);
        //cout << T_array[iT][iB] << "\t" << muB_array[iT][iB] << "\t" << c0[iT][iB] << "\t" << endl;
        //cout << T_array[iT][iB] << "\t" << muB_array[iT][iB] << "\t" << c2[iT][iB] << "\t" << endl;
      }
      else
      {
        fscanf(c0_file, "%lf\t\t%lf\n", &T_array[iT][iB], &c0[iT][iB]);
        fscanf(c2_file, "%lf\t\t%lf\n", &T_array[iT][iB], &c2[iT][iB]);
        //cout << T_array[iT][0] << "\t" << c0[iT][0] << "\t" << iT << endl;
        //cout << T_array[iT][0] << "\t" << c2[iT][0] << "\t" << iT << endl;
      }

      // check if cross freezeout temperature (T_array increasing)
      if (iT > 0 && T_invfm < T_array[iT][iB])
      {
        // linear-interpolate wrt temperature (c0,c2) at T_invfm
        df_coeff[0] = c0[iT-1][iB] + c0[iT][iB] * ((T_invfm - T_array[iT-1][iB]) / T_array[iT-1][iB]);
        df_coeff[2] = c2[iT-1][iB] + c2[iT][iB] * ((T_invfm - T_array[iT-1][iB]) / T_array[iT-1][iB]);
        break;
      }
    } //iT
  } //iB

  // include c1 coefficient
  if(INCLUDE_BARYON)
  {
    double c1[n_T][n_muB];
    FILE * c1_file;
    char c1_name[255];
    sprintf(c1_name, "%s", "deltaf_coefficients/c1_df14_vh.dat");
    c1_file = fopen(c1_name, "r");
    if(c1_file == NULL) printf("Couldn't open c1 coefficient file!\n");
    fgets(header, 100, c1_file);
    cout << "Loading c1 coefficient.." << endl;
    // scan c1 file
    for(int iB = 0; iB < n_muB; iB++)
    {
      for(int iT = 0; iT < n_T; iT++)
      {
        fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c1[iT][iB]);
        //cout << T_array[iT][iB] << "\t" << muB_array[iT][iB] << "\t" << c1[iT][iB] << "\t" << endl;
        if (iT > 0 && T_invfm < T_array[iT][iB])
        {
          // linear-interpolate w.r.t. temperature c1 at T_invfm
          df_coeff[1] = c1[iT-1][iB] + c1[iT][iB] * ((T_invfm - T_array[iT-1][iB]) / T_array[iT-1][iB]);
          break;
        }
      } //iT
    } //iB
    // include c3 and c4 coefficients
    if(INCLUDE_BARYONDIFF_DELTAF)
    {
      double c3[n_T][n_muB];
      double c4[n_T][n_muB];
      FILE * c3_file;
      FILE * c4_file;
      char c3_name[255];
      char c4_name[255];
      sprintf(c3_name, "%s", "deltaf_coefficients/c3_df14_vh.dat");
      sprintf(c4_name, "%s", "deltaf_coefficients/c4_df14_vh.dat");
      c3_file = fopen(c3_name, "r");
      c4_file = fopen(c4_name, "r");
      if(c3_file == NULL) printf("Couldn't open c3 coefficient file!\n");
      if(c4_file == NULL) printf("Couldn't open c4 coefficient file!\n");
      fgets(header, 100, c3_file);
      fgets(header, 100, c4_file);
      cout << "Loading c3 and c4 coefficients.." << endl;
      // scan c3,c4 files
      for(int iB = 0; iB < n_muB; iB++)
      {
        for(int iT = 0; iT < n_T; iT++)
        {
          fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c3[iT][iB]);
          fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c4[iT][iB]);
          //cout << T_array[iT][iB] << "\t" << muB_array[iT][iB] << "\t" << c3[iT][iB] << "\t" << endl;
          //cout << T_array[iT][iB] << "\t" << muB_array[iT][iB] << "\t" << c4[iT][iB] << "\t" << endl;
          if(iT > 0 && T_invfm < T_array[iT][iB])
          {
            // linear-interpolate wrt temperature (c3,c4) at T_invfm
            df_coeff[3] = c3[iT-1][iB] + c3[iT][iB] * ((T_invfm - T_array[iT-1][iB]) / T_array[iT-1][iB]);
            df_coeff[4] = c4[iT-1][iB] + c4[iT][iB] * ((T_invfm - T_array[iT-1][iB]) / T_array[iT-1][iB]);
            break;
          }
        } //iT
      } //iB
    }
  }

  // convert 14-momentum coefficients to real-life units (hbarC ~ 0.2 GeV * fm)
  df_coeff[0] /= (hbarC * hbarC * hbarC);
  df_coeff[1] /= (hbarC * hbarC);
  df_coeff[2] /= (hbarC * hbarC * hbarC);
  df_coeff[3] /= (hbarC * hbarC);
  df_coeff[4] /= (hbarC * hbarC * hbarC);

  //copy delta_f coefficients to the device array
  double *df_coeff_d;
  cudaMalloc( (void**) &df_coeff_d, 5 * sizeof(double) );
  cudaMemcpy( df_coeff_d, df_coeff, 5 * sizeof(double), cudaMemcpyHostToDevice );
  //for testing
  cout << "delta_f coefficients: "<< endl;
  cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << "\tc3 = " << df_coeff[3] << "\tc4 = " << df_coeff[4] << endl;

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
                                                              muB_d, nB_d, Vt_d, Vx_d, Vy_d, Vn_d, prefactor, df_coeff_d,
                                                              INCLUDE_BARYON, REGULATE_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF);


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
}
