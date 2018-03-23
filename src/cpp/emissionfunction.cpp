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

//this is where the magic happens - and the speed bottleneck
//this is the function that needs to be parallelized
//try acceleration via omp threads and simd, as well as openacc, CUDA?
void EmissionFunctionArray::calculate_dN_ptdptdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff)

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
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      printf("Progress : Finished chunk %d of %d \n", n, FO_length / FO_chunk);
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
      #pragma omp parallel for
      #pragma acc kernels
      //#pragma acc loop independent
      for (int icell = 0; icell < endFO; icell++) //cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell; //global FO cell index

        // set freezeout info to local varibles to reduce(?) memory access outside cache :
        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double eta = eta_fo[icell_glb];         // spacetime rapidity

        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ut = ut_fo[icell_glb];           // contravariant fluid velocity
        double ux = ux_fo[icell_glb];
        double uy = uy_fo[icell_glb];
        double un = un_fo[icell_glb];

        double T = T_fo[icell_glb];             // temperature
        double E = E_fo[icell_glb];             // energy density
        double P = P_fo[icell_glb];             // pressure

        double pitt = pitt_fo[icell_glb];       // pi^munu
        double pitx = pitx_fo[icell_glb];
        double pity = pity_fo[icell_glb];
        double pitn = pitn_fo[icell_glb];
        double pixx = pixx_fo[icell_glb];
        double pixy = pixy_fo[icell_glb];
        double pixn = pixn_fo[icell_glb];
        double piyy = piyy_fo[icell_glb];
        double piyn = piyn_fo[icell_glb];
        double pinn = pinn_fo[icell_glb];

        double bulkPi = bulkPi_fo[icell_glb];   // bulk pressure

        double muB = 0.0;                         // baryon chemical potential
        double nB = 0.0;                          // net baryon density
        double Vt = 0.0;                          // baryon diffusion
        double Vx = 0.0;
        double Vy = 0.0;
        double Vn = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell_glb];

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_fo[icell_glb];
            Vt = Vt_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
          }
        }

        // useful expressions
        double tau2 = tau * tau;
        //double tau4 = tau2 * tau2;
        double shear_coeff = 0.5 / (T * T * ( E + P));  // (see df_shear)

        //now loop over all particle species and momenta
        for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
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

              for (int iy = 0; iy < y_tab_length; iy++)
              {
                //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                double y = yValues[iy];
                //double shear_deltaf_prefactor = 1.0 / (2.0 * T[icell_glb] * T[icell_glb] * (E[icell_glb] + P[icell_glb]));
                double pt = mT * cosh(y - eta); //contravariant
                double pn = mT_over_tau * sinh(y - eta); //contravariant
                // useful expression
                double tau2_pn = tau2 * pn;

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
                double c0 = df_coeff[0];
                double c2 = df_coeff[2];

                if (INCLUDE_BULK_DELTAF)
                {
                  if (INCLUDE_BARYON)
                  {
                    double c1 = df_coeff[1];
                    df_bulk = (c0 + (baryon * c1 + c2 * pdotu) * pdotu) * bulkPi;
                  }
                  else
                  {
                    df_bulk = (c0 + c2 * pdotu * pdotu) * bulkPi;
                  }
                }

                // baryon diffusion correction:
                double df_baryondiff = 0.0;

                if (INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                {
                  double c3 = df_coeff[3];
                  double c4 = df_coeff[4];
                  // V^mu * p_mu
                  double Vmu_pmu = Vt * pt - Vx * px - Vy * py - Vn * tau2_pn;
                  df_baryondiff = (baryon * c3 + c4 * pdotu) * Vmu_pmu;
                }
                double df = df_shear + df_bulk + df_baryondiff;
                //long long int ir = icell + (FO_chunk * ipart) + (FO_chunk * npart * ipT) + (FO_chunk * npart * pT_tab_length * iphip) + (FO_chunk * npart * pT_tab_length * phi_tab_length * iy);
                long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                //check that this expression is correct
                if (REGULATE_DELTAF)
                {
                  double reg_df = max( -1.0, min( feqbar * df, 1.0 ) );
                  dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma * feq * (1.0 + reg_df));
                }
                else dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma * feq * (1.0 + feqbar * df));
              } //iy
            } //iphip
          } //ipT
        } //ipart
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
              for (int iy = 0; iy < y_tab_length; iy++)
              {
                long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
                double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                }//icell
                dN_pTdpTdphidy[is] += dN_pTdpTdphidy_tmp; //sum over all chunks
              }//iy
            }//iphip
          }//ipT
        }//ipart species
      } //if (endFO != 0 )
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
            long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
            //if (dN_pTdpTdphidy[is] < 1.0e-40) spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << 0.0 << "\n";
            //else spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[is] << "\n";
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
    double *muB, *nB, *Vt, *Vx, *Vy, *Vn;

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
        nB = (double*) calloc(FO_length, sizeof(double));
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
          nB[icell] = surf->nB;
          Vt[icell] = surf->Vt;
          Vx[icell] = surf->Vx;
          Vy[icell] = surf->Vy;
          Vn[icell] = surf->Vn;
        }
      }
    }

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
    df_coeff[0] /= hbarC;
    df_coeff[1] /= (hbarC * hbarC);
    df_coeff[2] /= (hbarC * hbarC * hbarC);
    df_coeff[3] /= (hbarC * hbarC);
    df_coeff[4] /= (hbarC * hbarC * hbarC);

    //for testing
    cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << "\tc3 = " << df_coeff[3] << "\tc4 = " << df_coeff[4] << endl;

    //launch function to perform integrations - this should be readily parallelizable
    calculate_dN_ptdptdphidy(Mass, Sign, Degen, Baryon,
      T, P, E, tau, eta, ut, ux, uy, un,
      dat, dax, day, dan,
      pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
      muB, nB, Vt, Vx, Vy, Vn, df_coeff);

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
          free(nB);
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
