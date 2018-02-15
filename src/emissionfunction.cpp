#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <omp.h>
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
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, particle_info* particles_in, FO_surf* surf_ptr_in, long FO_length_in)
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
  surf_ptr = surf_ptr_in;
  FO_length = FO_length_in;
  number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

  chosen_particles_table = new int[Nparticles];
  //a class member to hold 3D spectra for all chosen particles
  dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];

  for (int n = 0; n < Nparticles; n++) chosen_particles_table[n] = 0;

  //only grab chosen particles from the table
  for (int m = 0; m < number_of_chosen_particles; m++)
  { //loop over all chosen particles
    int mc_id = chosen_particles_in->get(1, m + 1);

    for (int n = 0; n < Nparticles; n++)
    {
      if (particles[n].mc_id == mc_id)
      {
        chosen_particles_table[n] = 1;
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
  last_particle_idx = -1;
}


EmissionFunctionArray::~EmissionFunctionArray()
{
  delete[] chosen_particles_table;
  delete[] chosen_particles_sampling_table;
  delete[] dN_pTdpTdphidy; //for holding 3d spectra of all chosen particles
}

//this is where the magic happens
//try acceleration via omp threads and simd, as well as openacc
void EmissionFunctionArray::calculate_dN_ptdptdphidy(double *Mass, double *Sign, double *Degen, double *Baryon,
  double *T, double *P, double *E, double *tau, double *eta, double *ut, double *ux, double *uy, double *un,
  double *dat, double *dax, double *day, double *dan,
  double *pitt, double *pitx, double *pity, double *pitn, double *pixx, double *pixy, double *pixn, double *piyy, double *piyn, double *pinn, double *bulkPi,
  double *muB, double *Vt, double *Vx, double *Vy, double *Vn)
{
  double prefactor = 1.0 / (8.0 double * (M_PI *M _PI * M_PI)) / hbarC / hbarC / hbarC;
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
    printf("start filling big array \n");
    int endFO = FO_chunk;
    if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
    //this section of code takes majority of time (compared to the reduction ) when using omp with 20 threads
    #pragma omp parallel for
    #pragma acc kernels
    #pragma acc loop independent
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
            double px = pT * trig_phi_table[iphip][0];
            double py = pT * trig_phi_table[iphip][1];

            for (int iy = 0; iy < y_tab_length; iy++)
            {
              double y = yValues[iy];
              double shear_deltaf_prefactor = 1.0 / (2.0 * T[icell_glb] * T[icell_glb] * (E[icell_glb] + P[icell_glb]));
              double pt = mT * cosh(y - eta[icell_glb]); //contravariant
              double pn = (-1.0 / tau[icell_glb]) * mT * sinh(y - eta[icell_glb]); //contravariant

              //thermal equilibrium distributions - for viscous hydro
              double pdotu = pt * ut[icell_glb] - px * ux[icell_glb] - py * uy[icell_glb] - (tau[icell_glb] * tau[icell_glb]) * pn * un[icell_glb]; //watch factors of tau from metric! is ueta read in as contravariant?
              double baryon_factor = 0.0;
              if (INCLUDE_BARYON) baryon_factor = Baryon[ipart] * muB[icell_glb];
              double exponent = (pdotu - baryon_factor) / T[icell_glb];
              double f0 = 1. / (exp(exponent) + sign);
              double pdotdsigma = pt * dat[icell_glb] + px * dax[icell_glb] + py * day[icell_glb] + pn * dan[icell_glb]; //are these dax, day etc. the covariant components?

              //viscous corrections
              double delta_f_shear = 0.0;
              double tau2 = tau[icell_glb] * tau[icell_glb];
              double tau4 = tau[icell_glb] * tau[icell_glb] * tau[icell_glb] * tau[icell_glb];
              //double pimunu_pmu_pnu = (pt * pt * pitt[icell_glb] - 2.0 * pt * px * pitx[icell_glb] - 2.0 * pt * py * pity[icell_glb] + px * px * pixx[icell_glb] + 2.0 * px * py * pixy[icell_glb] + py * py * piyy[icell_glb] + pn * pn * pinn[icell_glb]);
              pimunu_pmu_pnu = pitt[icell_glb] * pt * pt + pixx[icell_glb] * px * px + piyy[icell_glb] * py * py + pinn[icell_glb] * pn * pn * (1.0 / tau4)
              + 2.0 * (-pitx[icell_glb] * pt * px - pity[icell_glb] * pt * py - pitn[icell_glb] * pt * pn * (1.0 / tau2) + pixy[icell_glb] * px * py + pixn[icell_glb] * px * pn * (1.0 / tau2) + piyn[icell_glb] * py * pn * (1.0 / tau2));
              delta_f_shear = ((1.0 - Sign[ipart] * f0) * pimunu_pmu_pnu * shear_deltaf_prefactor);
              double delta_f_bulk = 0.0;
              //put fourteen moment expression for bulk viscosity (\delta)f here
              double delta_f_baryondiff = 0.0;
              //put fourteen moment expression for baryon diffusion (\delta)f here
              double ratio = min(1., fabs(1. / (delta_f_shear + delta_f_bulk + delta_f_baryondiff)));
              long long int ir = icell + (FO_chunk * ipart) + (FO_chunk * npart * ipT) + (FO_chunk * npart * pT_tab_length * iphip) + (FO_chunk * npart * pT_tab_length * phi_tab_length * iy);
              dN_pTdpTdphidy_all[ir] = (prefactor * Degen[ipart] * pdotdsigma * tau[icell] * f0 * (1. + (delta_f_shear + delta_f_bulk + delta_f_baryondiff) * ratio));
            } //iy
          } //iphip
        } //ipT
      } //ipart
    } //icell
    printf("performing reduction over cells\n");
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
  char filename[255] = "";
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
          long long int is = ipT + (pT_tab_length * iphip) + (pT_tab_length * phi_tab_length * iy);
          spectra3DFile << scientific <<  setw(15) << setprecision(8) << dN_pTdpTdphidy_3D[is] << "\t";
        } //ipT
        spectra3DFile << "\n";
      } //iphip
    } //iy
  }//ipart
  spectra3DFile.close();
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

  //launch function to perform integrations - this should be readily parallelizable
  calculate_dN_ptdptdphidy(Mass, Sign, Degen, Baryon,
                          T, P, E, tau, eta, ut, ux, uy, un,
                          dat, dax, day, dan,
                          pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
                          muB, Vt, Vx, Vy, Vn);

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
//#pragma acc routine
void EmissionFunctionArray::getbulkvisCoefficients(double Tdec, double* bulkvisCoefficients)
{
   double Tdec_fm = Tdec/hbarC;  // [1/fm]
   double Tdec_fm_power[11];    // cache the polynomial power of Tdec_fm
   Tdec_fm_power[1] = Tdec_fm;
   for(int ipower = 2; ipower < 11; ipower++)
       Tdec_fm_power[ipower] = Tdec_fm_power[ipower-1]*Tdec_fm;
   if(bulk_deltaf_kind == 0)       // 14 moment expansion
   {
        // load from file
        bulkvisCoefficients[0] = bulkdf_coeff->interp(1, 2, Tdec_fm, 5)/pow(hbarC, 3);  //B0 [fm^3/GeV^3]
        bulkvisCoefficients[1] = bulkdf_coeff->interp(1, 3, Tdec_fm, 5)/pow(hbarC, 2);  // D0 [fm^3/GeV^2]
        bulkvisCoefficients[2] = bulkdf_coeff->interp(1, 4, Tdec_fm, 5)/pow(hbarC, 3);  // E0 [fm^3/GeV^3]
        // parameterization for mu = 0
        //bulkvisCoefficients[0] = exp(-15.04512474*Tdec_fm + 11.76194266)/pow(hbarC, 3); //B0[fm^3/GeV^3]
        //bulkvisCoefficients[1] = exp( -12.45699277*Tdec_fm + 11.4949293)/hbarC/hbarC;  // D0 [fm^3/GeV^2]
        //bulkvisCoefficients[2] = -exp(-14.45087586*Tdec_fm + 11.62716548)/pow(hbarC, 3);  // E0 [fm^3/GeV^3]
   }
   else if(bulk_deltaf_kind == 1)  // relaxation type
   {
       // parameterization from JF
       // A Polynomial fit to each coefficient -- X is the temperature in fm^-1
       // Both fits are reliable between T=100 -- 180 MeV , do not trust it beyond
       bulkvisCoefficients[0] = (  642096.624265727
                                 - 8163329.49562861*Tdec_fm_power[1]
                                 + 47162768.4292073*Tdec_fm_power[2]
                                 - 162590040.002683*Tdec_fm_power[3]
                                 + 369637951.096896*Tdec_fm_power[4]
                                 - 578181331.809836*Tdec_fm_power[5]
                                 + 629434830.225675*Tdec_fm_power[6]
                                 - 470493661.096657*Tdec_fm_power[7]
                                 + 230936465.421*Tdec_fm_power[8]
                                 - 67175218.4629078*Tdec_fm_power[9]
                                 + 8789472.32652964*Tdec_fm_power[10]);

       bulkvisCoefficients[1] = (  1.18171174036192
                                 - 17.6740645873717*Tdec_fm_power[1]
                                 + 136.298469057177*Tdec_fm_power[2]
                                 - 635.999435106846*Tdec_fm_power[3]
                                 + 1918.77100633321*Tdec_fm_power[4]
                                 - 3836.32258307711*Tdec_fm_power[5]
                                 + 5136.35746882372*Tdec_fm_power[6]
                                 - 4566.22991441914*Tdec_fm_power[7]
                                 + 2593.45375240886*Tdec_fm_power[8]
                                 - 853.908199724349*Tdec_fm_power[9]
                                 + 124.260460450113*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 2)
   {
       // A Polynomial fit to each coefficient -- Tfm is the temperature in fm^-1
       // Both fits are reliable between T=100 -- 180 MeV , do not trust it beyond
       bulkvisCoefficients[0] = (
               21091365.1182649 - 290482229.281782*Tdec_fm_power[1]
             + 1800423055.01882*Tdec_fm_power[2] - 6608608560.99887*Tdec_fm_power[3]
             + 15900800422.7138*Tdec_fm_power[4] - 26194517161.8205*Tdec_fm_power[5]
             + 29912485360.2916*Tdec_fm_power[6] - 23375101221.2855*Tdec_fm_power[7]
             + 11960898238.0134*Tdec_fm_power[8] - 3618358144.18576*Tdec_fm_power[9]
             + 491369134.205902*Tdec_fm_power[10]);

       bulkvisCoefficients[1] = (
               4007863.29316896 - 55199395.3534188*Tdec_fm_power[1]
             + 342115196.396492*Tdec_fm_power[2] - 1255681487.77798*Tdec_fm_power[3]
             + 3021026280.08401*Tdec_fm_power[4] - 4976331606.85766*Tdec_fm_power[5]
             + 5682163732.74188*Tdec_fm_power[6] - 4439937810.57449*Tdec_fm_power[7]
             + 2271692965.05568*Tdec_fm_power[8] - 687164038.128814*Tdec_fm_power[9]
             + 93308348.3137008*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 3)
   {
       bulkvisCoefficients[0] = (
               160421664.93603 - 2212807124.97991*Tdec_fm_power[1]
             + 13707913981.1425*Tdec_fm_power[2] - 50204536518.1767*Tdec_fm_power[3]
             + 120354649094.362*Tdec_fm_power[4] - 197298426823.223*Tdec_fm_power[5]
             + 223953760788.288*Tdec_fm_power[6] - 173790947240.829*Tdec_fm_power[7]
             + 88231322888.0423*Tdec_fm_power[8] - 26461154892.6963*Tdec_fm_power[9]
             + 3559805050.19592*Tdec_fm_power[10]);
       bulkvisCoefficients[1] = (
               33369186.2536556 - 460293490.420478*Tdec_fm_power[1]
             + 2851449676.09981*Tdec_fm_power[2] - 10443297927.601*Tdec_fm_power[3]
             + 25035517099.7809*Tdec_fm_power[4] - 41040777943.4963*Tdec_fm_power[5]
             + 46585225878.8723*Tdec_fm_power[6] - 36150531001.3718*Tdec_fm_power[7]
             + 18353035766.9323*Tdec_fm_power[8] - 5504165325.05431*Tdec_fm_power[9]
             + 740468257.784873*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 4)
   {
       bulkvisCoefficients[0] = (
               1167272041.90731 - 16378866444.6842*Tdec_fm_power[1]
             + 103037615761.617*Tdec_fm_power[2] - 382670727905.111*Tdec_fm_power[3]
             + 929111866739.436*Tdec_fm_power[4] - 1540948583116.54*Tdec_fm_power[5]
             + 1767975890298.1*Tdec_fm_power[6] - 1385606389545*Tdec_fm_power[7]
             + 709922576963.213*Tdec_fm_power[8] - 214726945096.326*Tdec_fm_power[9]
             + 29116298091.9219*Tdec_fm_power[10]);
       bulkvisCoefficients[1] = (
               5103633637.7213 - 71612903872.8163*Tdec_fm_power[1]
             + 450509014334.964*Tdec_fm_power[2] - 1673143669281.46*Tdec_fm_power[3]
             + 4062340452589.89*Tdec_fm_power[4] - 6737468792456.4*Tdec_fm_power[5]
             + 7730102407679.65*Tdec_fm_power[6] - 6058276038129.83*Tdec_fm_power[7]
             + 3103990764357.81*Tdec_fm_power[8] - 938850005883.612*Tdec_fm_power[9]
             + 127305171097.249*Tdec_fm_power[10]);
   }
   return;
}
