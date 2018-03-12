#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#ifdef _OMP
  #include <omp.h>
#endif
#include "main.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "Stopwatch.h"
#include "arsenal.h"
#include "ParameterReader.h"
#ifdef _OPENACC
#include <accelmath.h>
#endif
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

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

  //is this necessary???
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

//this is where the magic happens - and the speed bottleneck
//this is the function that needs to be parallelized
//try acceleration via omp threads and simd, as well as openacc, CUDA?
void EmissionFunctionArray::calculate_dN_ptdptdphidy(double *Mass, double *Sign, double *Degen, double *Baryon,
  double *T, double *P, double *E, double *tau, double *eta, double *ut, double *ux, double *uy, double *un,
  double *dat, double *dax, double *day, double *dan,
  double *pitt, double *pitx, double *pity, double *pitn, double *pixx, double *pixy, double *pixn, double *piyy, double *piyn, double *pinn, double *bulkPi,
  double *muB, double *Vt, double *Vx, double *Vy, double *Vn, double *df_coeff)

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
  double yValues[y_tab_length];
  for (int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);
  for (int iy = 0; iy < y_tab_length; iy++) yValues[iy] = y_tab->get(1, iy + 1);

  //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
  //to hold the spectra for each surface cell in a chunk, for all particle species
  //in this way, we are limited to small values of FO_chunk, because our array holds values for all species...
  //try putting loop over particle species as outermost loop so that we don't need such a huge array?
  int npart = number_of_chosen_particles;
  double *dN_pTdpTdphidy_all;
  dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

  //loop over bite size chunks of FO surface

  for (int n = 0; n < FO_length / FO_chunk + 1; n++)
  {
    printf("Progress : Finished chunk %d of %d \n", n, FO_length / FO_chunk);
    int endFO = FO_chunk;
    if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
    //this section of code takes majority of time (compared to the reduction ) when using omp with 20 threads
    #pragma omp parallel for
    #pragma acc kernels
    //#pragma acc loop independent
    for (int icell = 0; icell < endFO; icell++) //cell index inside each chunk
    {
      int icell_glb = n * FO_chunk + icell; //global FO cell index

      //now loop over all particle species and momenta - consider particle loop outside of loop over cells?
      for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
      {
        for (int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pTValues[ipT];
          double mT = sqrt(Mass[ipart] * Mass[ipart] + pT * pT);

          for (int iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double px = pT * trig_phi_table[iphip][0]; //contravariant
            double py = pT * trig_phi_table[iphip][1]; //contravariant

            for (int iy = 0; iy < y_tab_length; iy++)
            {
              //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
              double y = yValues[iy];
              double shear_deltaf_prefactor = 1.0 / (2.0 * T[icell_glb] * T[icell_glb] * (E[icell_glb] + P[icell_glb]));
              double pt = mT * cosh(y - eta[icell_glb]); //contravariant
              double pn = (-1.0 / tau[icell_glb]) * mT * sinh(y - eta[icell_glb]); //contravariant

              //thermal equilibrium distributions - for viscous hydro
              double pdotu = pt * ut[icell_glb] - px * ux[icell_glb] - py * uy[icell_glb] - (tau[icell_glb] * tau[icell_glb]) * pn * un[icell_glb]; //watch factors of tau from metric!
              double baryon_factor = 0.0;
              if (INCLUDE_BARYON) baryon_factor = Baryon[ipart] * muB[icell_glb];
              double exponent = (pdotu - baryon_factor) / T[icell_glb];
              double f0 = 1. / (exp(exponent) + Sign[ipart]);

              //momentum vector is contravariant, surface normal vector is COVARIANT
              double pdotdsigma = pt * dat[icell_glb] + px * dax[icell_glb] + py * day[icell_glb] + pn * dan[icell_glb]; //CHECK THIS !

              //viscous corrections
              double delta_f_shear = 0.0;
              double tau2 = tau[icell_glb] * tau[icell_glb];
              double tau4 = tau2 * tau2;
              //double pimunu_pmu_pnu = (pt * pt * pitt[icell_glb] - 2.0 * pt * px * pitx[icell_glb] - 2.0 * pt * py * pity[icell_glb] + px * px * pixx[icell_glb] + 2.0 * px * py * pixy[icell_glb] + py * py * piyy[icell_glb] + pn * pn * pinn[icell_glb]);
              double pimunu_pmu_pnu = pitt[icell_glb] * pt * pt + pixx[icell_glb] * px * px + piyy[icell_glb] * py * py + pinn[icell_glb] * pn * pn * (1.0 / tau4)
              + 2.0 * (-pitx[icell_glb] * pt * px - pity[icell_glb] * pt * py - pitn[icell_glb] * pt * pn * (1.0 / tau2) + pixy[icell_glb] * px * py + pixn[icell_glb] * px * pn * (1.0 / tau2) + piyn[icell_glb] * py * pn * (1.0 / tau2));

              delta_f_shear = ((1.0 - Sign[ipart] * f0) * pimunu_pmu_pnu * shear_deltaf_prefactor);

              double delta_f_bulk = 0.0;
              //put fourteen moment expression for bulk viscosity (\delta)f here

              //this expression is just for testing , not correct!
              delta_f_bulk = bulkPi[icell_glb] * (df_coeff[0] + pdotu * df_coeff[1] + pdotu * pdotu * df_coeff[2]);

              double delta_f_baryondiff = 0.0;
              //put fourteen moment expression for baryon diffusion (\delta)f here

              //WHAT IS RATIO - CHANGE THIS ?
              double ratio = min(1., fabs(1. / (delta_f_shear + delta_f_bulk + delta_f_baryondiff)));
              long long int ir = icell + (FO_chunk * ipart) + (FO_chunk * npart * ipT) + (FO_chunk * npart * pT_tab_length * iphip) + (FO_chunk * npart * pT_tab_length * phi_tab_length * iy);
              dN_pTdpTdphidy_all[ir] = (prefactor * Degen[ipart] * pdotdsigma * tau[icell_glb] * f0 * (1. + (delta_f_shear + delta_f_bulk + delta_f_baryondiff) * ratio));

            } //iy
          } //iphip
        } //ipT
      } //ipart
    } //icell
    //now perform the reduction over cells
    //this section of code is quite fast compared to filling dN_xxx_all when using multiple threads openmp
    #pragma omp parallel for collapse(3)
    #pragma acc kernels
    for (int ipart = 0; ipart < npart; ipart++)
    {
      for (int ipT = 0; ipT < pT_tab_length; ipT++)
      {
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          for (int iy = 0; iy < y_tab_length; iy++)
          {
            long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
            double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
            #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
            for (int icell = 0; icell < FO_chunk; icell++)
            {
              long long int ir = icell + (FO_chunk * ipart) + (FO_chunk * npart * ipT) + (FO_chunk * npart * pT_tab_length * iphip) + (FO_chunk * npart * pT_tab_length * phi_tab_length * iy);
              dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[ir];
            }//icell
            dN_pTdpTdphidy[is] += dN_pTdpTdphidy_tmp; //sum over all chunks
          }//iy
        }//iphip
      }//ipT
    }//ipart species
  }//n FO chunk

  //free memory
  free(dN_pTdpTdphidy_all);

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


//*********************************************************************************************
void EmissionFunctionArray::calculate_spectra()
// Calculate dNArrays and flows for all particles given in chosen_particle file.
{
  cout << "calculate_spectra() has started... " << endl;
  Stopwatch sw;
  sw.tic();

  //fill arrays with all particle info and freezeout info to pass to function which will perform the integral
  //this should only be done once, not for each particle... then launch calculation on accelerator w/o unecessary copies

  //particle info
  particle_info *particle;
  double *Mass, *Sign, *Degen, *Baryon;
  Mass = (double*)calloc(number_of_chosen_particles, sizeof(double));
  Sign = (double*)calloc(number_of_chosen_particles, sizeof(double));
  Degen = (double*)calloc(number_of_chosen_particles, sizeof(double));
  Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));

  for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
  {
    int particle_idx = chosen_particles_sampling_table[ipart];
    particle = &particles[particle_idx];
    Mass[ipart] = particle->mass;
    Sign[ipart] = particle->sign;
    Degen[ipart] = particle->gspin;
    Baryon[ipart] = particle->baryon;
  }

  //freezeout surface info
  FO_surf *surf = &surf_ptr[0];
  double *T, *P, *E, *tau, *eta, *ut, *ux, *uy, *un;
  double *dat, *dax, *day, *dan;
  double *pitt, *pitx, *pity, *pitn, *pixx, *pixy, *pixn, *piyy, *piyn, *pinn, *bulkPi;
  double *muB, *Vt, *Vx, *Vy, *Vn;

  T = (double*)calloc(FO_length, sizeof(double));
  P = (double*)calloc(FO_length, sizeof(double));
  E = (double*)calloc(FO_length, sizeof(double));
  tau = (double*)calloc(FO_length, sizeof(double));
  eta = (double*)calloc(FO_length, sizeof(double));
  ut = (double*)calloc(FO_length, sizeof(double));
  ux = (double*)calloc(FO_length, sizeof(double));
  uy = (double*)calloc(FO_length, sizeof(double));
  un = (double*)calloc(FO_length, sizeof(double));

  dat = (double*)calloc(FO_length, sizeof(double));
  dax = (double*)calloc(FO_length, sizeof(double));
  day = (double*)calloc(FO_length, sizeof(double));
  dan = (double*)calloc(FO_length, sizeof(double));

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
  bulkPi = (double*)calloc(FO_length, sizeof(double));

  if (INCLUDE_BARYON)
  {
    muB = (double*)calloc(FO_length, sizeof(double));

    if (INCLUDE_BARYONDIFF_DELTAF)
    {
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
    T[icell] = surf->T;
    P[icell] = surf->P;
    E[icell] = surf->E;
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

  //read in tables of delta_f coefficients
  //write now these tables are in units fm^x, e.g. Temperature in fm^(-1)
  //recall our FO variables have different units, e.g. T[] has units GeV
  //put c0, c1, c2 into an array of three values ; extend to include c3, c4?
  printf("Reading in delta_f coefficient tables \n");
  double df_coeff[3] = {0.0, 0.0, 0.0}; // array holding values of c_0, c_1, and c_2
  double T_invfm = T[0] / hbarC; //FO Temp in units fm^(-1)

  int n_T = 100; //number of points in Temperature
  int n_muB = 1; //number of points in baryon chem pot
  if (INCLUDE_BARYON) n_muB = 500;

  double table_T[n_T]; //values of temperature

  FILE *fileIn;
  char fname[255];

  //c0 coefficient
  sprintf(fname, "%s", "deltaf_coefficients/c0_df14_vh_muB_0.dat");
  fileIn = fopen(fname, "r");
  if (fileIn == NULL) printf("Couldn't open deltaf_coefficients table !\n");
  double table_c0[n_T]; //values of c_0
  //fill the array from file
  char buffer[300];
  fgets(buffer, 100, fileIn); //skip the header
  for (int iT = 0; iT < n_T; iT++)
  {
    fscanf(fileIn, "%lf\t\t%lf\n", &table_T[iT], &table_c0[iT]);
  }
  //now linearly interpolate to find c0 at freezeout temperature
  //freezeout temperature is determined by the first cell of freezeout surface, T[0]...
  for (int iT = 0; iT < n_T; iT++)
  {
    if (T_invfm < table_T[iT]) //find where we cross the FO temperature
    {
      df_coeff[0] = table_c0[iT - 1] + table_c0[iT] * ( (T_invfm - table_T[iT - 1]) / table_T[iT - 1]); //set the value of c_0
      break;
    }
  }

  //c1 coefficient - only nonzero for nonzero baryon number
  /*
  sprintf(fname, "%s", "deltaf_coefficients/c1_df14_vh_muB_0.dat");
  fileIn = fopen(fname, "r");
  if (fileIn == NULL) printf("Couldn't open deltaf_coefficients table !\n");
  double table_c1[n_T]; //values of c_1
  //fill the array from file
  fgets(buffer, 100, fileIn); //skip the header
  for (int iT = 0; iT < n_T; iT++)
  {
    fscanf(fileIn, "%lf\t\t%lf\n", &table_T[iT], &table_c1[iT]);
  }
  //now linearly interpolate to find c0 at freezeout temperature
  //freezeout temperature is determined by the first cell of freezeout surface, T[0]...
  for (int iT = 0; iT < n_T; iT++)
  {
    if (T_invfm < table_T[iT]) //find where we cross the FO temperature
    {
      bulk_df_c[1] = table_c1[iT - 1] + table_c1[iT] * ( (T_invfm - table_T[iT - 1]) / table_T[iT - 1]); //set the value of c_1
      break;
    }
  }
  */

  //c2 coefficient
  sprintf(fname, "%s", "deltaf_coefficients/c2_df14_vh_muB_0.dat");
  fileIn = fopen(fname, "r");
  if (fileIn == NULL) printf("Couldn't open deltaf_coefficients table !\n");
  double table_c2[n_T]; //values of c_1
  //fill the array from file
  fgets(buffer, 100, fileIn); //skip the header
  for (int iT = 0; iT < n_T; iT++)
  {
    fscanf(fileIn, "%lf\t\t%lf\n", &table_T[iT], &table_c2[iT]);
  }
  //now linearly interpolate to find c0 at freezeout temperature
  //freezeout temperature is determined by the first cell of freezeout surface, T[0]...
  for (int iT = 0; iT < n_T; iT++)
  {
    if (T_invfm < table_T[iT]) //find where we cross the FO temperature
    {
      df_coeff[2] = table_c2[iT - 1] + table_c2[iT] * ( (T_invfm - table_T[iT - 1]) / table_T[iT - 1]); //set the value of c_2
      break;
    }
  }


  //how should we convert units of c0, c1, c2? they are dimensionless right ...

  //for testing
  cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << endl;

  //launch function to perform integrations - this should be readily parallelizable
  calculate_dN_ptdptdphidy(Mass, Sign, Degen, Baryon,
                          T, P, E, tau, eta, ut, ux, uy, un,
                          dat, dax, day, dan,
                          pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
                          muB, Vt, Vx, Vy, Vn, df_coeff);

  //write the results to file
  write_dN_pTdpTdphidy_toFile();
  //free memory
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
      free(Vt);
      free(Vx);
      free(Vy);
      free(Vn);
    }
  }
  sw.toc();
  cout << " -- calculate_spectra() finished " << sw.takeTime() << " seconds." << endl;
}
/*
bool EmissionFunctionArray::particles_are_the_same(int idx1, int idx2)
{
    if (particles[idx1].sign!=particles[idx2].sign) return false;
    if (particles[idx1].baryon!=particles[idx2].baryon && INCLUDE_MUB) return false;
    if (abs(particles[idx1].mass-particles[idx2].mass) / (particles[idx2].mass+1e-30) > PARTICLE_DIFF_TOLERANCE) return false;
    for (long l=0; l<FO_length; l++)
    {
        double chem1 = FOsurf_ptr[l].particle_mu[idx1], chem2 = FOsurf_ptr[l].particle_mu[idx2];
        if (abs(chem1-chem2)/(chem2+1e-30) > PARTICLE_DIFF_TOLERANCE)
        {
          return false;
        }

    }

    return true;
}
*/
