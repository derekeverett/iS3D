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
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, particle_info* particles_in, int Nparticles_in, FO_surf* FOsurf_ptr_in, long FO_length_in)
{
  paraRdr = paraRdr_in;
  pT_tab = pT_tab_in; pT_tab_length = pT_tab->getNumberOfRows();
  phi_tab = phi_tab_in; phi_tab_length = phi_tab->getNumberOfRows();
  y_tab = y_tab_in; y_tab_length = y_tab->getNumberOfRows();

  //a class member to hold 3D spectra for one species
  dN_pTdpTdphidy = new double [pT_tab_length * phi_tab_length * y_tab_length];
  dN_ptdptdphidy_filename = "results/dN_ptdptdphidy.dat";

  // get control parameters
  CALCULATEDED3P = paraRdr->getVal("calculate_dEd3p");
  INCLUDE_BULKDELTAF = paraRdr->getVal("turn_on_bulk");
  INCLUDE_MUB = paraRdr->getVal("turn_on_muB");
  INCLUDE_DELTAF = paraRdr->getVal("turn_on_shear");
  GROUPING_PARTICLES = paraRdr->getVal("grouping_particles");
  PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");
  USE_HISTORIC_FORMAT = paraRdr->getVal("use_historic_format");
  F0_IS_NOT_SMALL = paraRdr->getVal("f0_is_not_small");
  bulk_deltaf_kind = paraRdr->getVal("bulk_deltaf_kind");

  particles = particles_in;
  Nparticles = Nparticles_in;

  FOsurf_ptr = FOsurf_ptr_in;
  FO_length = FO_length_in;

  number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

  chosen_particles_01_table = new int[Nparticles];
  for (int n=0; n<Nparticles; n++) chosen_particles_01_table[n]=0;
  for (int m=0; m<number_of_chosen_particles; m++)
  {
    int monval = chosen_particles_in->get(1,m+1);
    for (int n=0; n<Nparticles; n++)
    {
      if (particles[n].monval==monval)
      {
        chosen_particles_01_table[n]=1;
        break;
      }
    }
  }
  // next, for sampling processes
  chosen_particles_sampling_table = new int[number_of_chosen_particles];
  // first copy the chosen_particles table, but now using indices instead of monval
  int current_idx = 0;
  for (int m=0; m<number_of_chosen_particles; m++)
  {
    int monval = chosen_particles_in->get(1,m+1);
    for (int n=0; n<Nparticles; n++)
    {
      if (particles[n].monval==monval)
      {
        chosen_particles_sampling_table[current_idx] = n;
        current_idx ++;
        break;
      }
    }
  }
  // next re-order them so that particles with similar mass are adjacent
  if (GROUPING_PARTICLES == 1) // sort particles according to their mass; bubble-sorting
  {
    for (int m=0; m<number_of_chosen_particles; m++)
      for (int n=0; n<number_of_chosen_particles-m-1; n++)
        if (particles[chosen_particles_sampling_table[n]].mass > particles[chosen_particles_sampling_table[n+1]].mass)
        {
          // swap them
          int particle_idx = chosen_particles_sampling_table[n+1];
          chosen_particles_sampling_table[n+1] = chosen_particles_sampling_table[n];
          chosen_particles_sampling_table[n] = particle_idx;
        }
  }
  last_particle_idx = -1;

  //arrays for bulk delta f coefficients
  bulkdf_coeff = new Table ("tables/BulkDf_Coefficients_Hadrons_s95p-v0-PCE.dat");
}


EmissionFunctionArray::~EmissionFunctionArray()
{
  delete dN_ptdptdphidy;
  if(CALCULATEDED3P == 1) delete dE_ptdptdphidy;
  delete[] chosen_particles_01_table;
  delete[] chosen_particles_sampling_table;
  delete bulkdf_coeff;
  delete[] dN_pTdpTdphidy_3D; //for holding 3d spectra of one species
}

//this function reorganizes the loop structure again
//try acceleration via omp threads and simd, as well as openacc
void EmissionFunctionArray::calculate_dN_ptdptdphidy(int particle_idx)
{
  last_particle_idx = particle_idx;
  particle_info* particle;
  particle = &particles[particle_idx];

  double mass = particle->mass;
  double sign = particle->sign;
  double degen = particle->gspin;
  int baryon = particle->baryon;

  double prefactor = 1.0/(8.0*(M_PI*M_PI*M_PI))/hbarC/hbarC/hbarC;

  FO_surf* surf = &FOsurf_ptr[0];

  int FO_chunk = 10000;

  double *bulkvisCoefficients;
  if (bulk_deltaf_kind == 0) bulkvisCoefficients = new double [3];
  else bulkvisCoefficients = new double [2];

  double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
  for (int j = 0; j < phi_tab_length; j++)
  {
    double phi = phi_tab->get(1,j+1);
    trig_phi_table[j][0] = cos(phi);
    trig_phi_table[j][1] = sin(phi);
  }
  //fill arrays with values points in pT and y, don't need phip values but cos(phip)/sin(phip) which are in trig table
  double pTValues[pT_tab_length];
  double yValues[y_tab_length];

  for (int ipT = 0; ipT < pT_tab_length; ipT++)
  {
    pTValues[ipT] = pT_tab->get(1, ipT + 1);
  }

  for (int iy = 0; iy < y_tab_length; iy++)
  {
    yValues[iy] = y_tab->get(1, iy + 1);
  }

  //now fill arrays with surface info
  double *Tdec, *Pdec, *Edec, *mu, *tau, *eta, *utau, *ux, *uy, *ueta;
  double *datau, *dax, *day, *daeta, *pi00, *pi01, *pi02, *pi11, *pi12, *pi22, *pi33;
  double *muB, *bulkPi;

  Tdec = (double*)calloc(FO_length, sizeof(double));
  Pdec = (double*)calloc(FO_length, sizeof(double));
  Edec = (double*)calloc(FO_length, sizeof(double));
  mu = (double*)calloc(FO_length, sizeof(double));
  tau = (double*)calloc(FO_length, sizeof(double));
  eta = (double*)calloc(FO_length, sizeof(double));
  utau = (double*)calloc(FO_length, sizeof(double));
  ux = (double*)calloc(FO_length, sizeof(double));
  uy = (double*)calloc(FO_length, sizeof(double));
  ueta = (double*)calloc(FO_length, sizeof(double));

  datau = (double*)calloc(FO_length, sizeof(double));
  dax = (double*)calloc(FO_length, sizeof(double));
  day = (double*)calloc(FO_length, sizeof(double));
  daeta = (double*)calloc(FO_length, sizeof(double));
  pi00 = (double*)calloc(FO_length, sizeof(double));
  pi01 = (double*)calloc(FO_length, sizeof(double));
  pi02 = (double*)calloc(FO_length, sizeof(double));
  pi11 = (double*)calloc(FO_length, sizeof(double));
  pi12 = (double*)calloc(FO_length, sizeof(double));
  pi22 = (double*)calloc(FO_length, sizeof(double));
  pi33 = (double*)calloc(FO_length, sizeof(double));

  muB = (double*)calloc(FO_length, sizeof(double));
  bulkPi = (double*)calloc(FO_length, sizeof(double));

  //this should be moved outside this function so that it is not done redundantly for every particle inside particle loop
  //also it should only be copied once to the gpu ... !
  //#pragma omp parallel for
  for (int icell = 0; icell < FO_length; icell++)
  {
    //reading info from surface
    surf = &FOsurf_ptr[icell];
    Tdec[icell] = surf->Tdec;
    Pdec[icell] = surf->Pdec;
    Edec[icell] = surf->Edec;
    mu[icell] = surf->particle_mu[last_particle_idx];
    tau[icell] = surf->tau;
    eta[icell] = surf->eta;
    utau[icell] = surf->u0;
    ux[icell] = surf->u1;
    uy[icell] = surf->u2;
    ueta[icell] = surf->u3;
    datau[icell] = surf->da0;
    dax[icell] = surf->da1;
    day[icell] = surf->da2;
    daeta[icell] = surf->da3;
    pi00[icell] = surf->pi00;
    pi01[icell] = surf->pi01;
    pi02[icell] = surf->pi02;
    pi11[icell] = surf->pi11;
    pi12[icell] = surf->pi12;
    pi22[icell] = surf->pi22;
    pi33[icell] = surf->pi33;
    muB[icell] = surf->muB;
    bulkPi[icell] = surf->bulkPi;
  }

  //declare a huge array of size FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
  //to hold the spectra for each surface cell in a chunk
  double *dN_pTdpTdphidy_all;
  dN_pTdpTdphidy_all = (double*)calloc(FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

  //loop over bite size chunks of FO surface
  for (int n = 0; n < FO_length / FO_chunk + 1; n++)
  {
    printf("start filling big array\n");
    int end = FO_chunk;
    if (n == (FO_length / FO_chunk)) end = FO_length - (n * FO_chunk); //don't go out of array bounds
    //this section of code takes majority of time (compared to the reduction ) when using omp with 20 threads
    #pragma omp parallel for
    #pragma acc kernels
    #pragma acc loop independent
    for (int icell = 0; icell < FO_chunk; icell++) //cell index inside each chunk
    {
      int icell_glb = n * FO_chunk + icell; //global FO cell index
      for (int ipT = 0; ipT < pT_tab_length; ipT++)
      {
        double pT = pTValues[ipT];
        double mT = sqrt(mass * mass + pT * pT);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double px = pT * trig_phi_table[iphip][0];
          double py = pT * trig_phi_table[iphip][1];

          for (int iy = 0; iy < y_tab_length; iy++)
          {
            double y = yValues[iy];

            double deltaf_prefactor = 0.0;

            if (INCLUDE_DELTAF) deltaf_prefactor = 1.0/(2.0 * Tdec[icell_glb] * Tdec[icell_glb] * (Edec[icell_glb] + Pdec[icell_glb]));
            //NEED TO MOVE getbulkvisCoefficients OUTSIDE OF ACC ROUTINE SOMEHOW
            if (INCLUDE_BULKDELTAF == 1)
            {
              //FIX THIS - NEED BULK VIS COEFFICIENTS ON GPU
              getbulkvisCoefficients(Tdec[icell_glb], bulkvisCoefficients);
            }
            double ptau = mT * cosh(y - eta[icell_glb]); //contravariant
            double peta = (-1.0 / tau[icell_glb]) * mT * sinh(y - eta[icell_glb]); //contravariant

            //thermal equilibrium distributions
            double pdotu = ptau * utau[icell_glb] - px * ux[icell_glb] - py * uy[icell_glb] - (tau[icell_glb] * tau[icell_glb]) * peta * ueta[icell_glb]; //watch factors of tau from metric! is ueta read in as contravariant?
            double expon = (pdotu - mu[icell_glb] - baryon * muB[icell_glb]) / Tdec[icell_glb];
            double f0 = 1./(exp(expon) + sign);
            // Must adjust this to be correct for the p*del \tau term.
            double pdotdsigma = ptau * datau[icell_glb] + px * dax[icell_glb] + py * day[icell_glb] + peta * daeta[icell_glb]; //are these dax, day etc. the covariant components?
            //viscous corrections
            double delta_f_shear = 0.0;
            if (INCLUDE_DELTAF)
            {
              double Wfactor = (ptau * ptau * pi00[icell_glb] - 2.0 * ptau * px * pi01[icell_glb] - 2.0 * ptau * py * pi02[icell_glb] + px * px * pi11[icell_glb] + 2.0 * px * py * pi12[icell_glb] + py * py * pi22[icell_glb] + peta * peta *pi33[icell_glb]);
              delta_f_shear = ((1 - F0_IS_NOT_SMALL*sign*f0) * Wfactor * deltaf_prefactor);
            }
            double delta_f_bulk = 0.0;
            if (INCLUDE_BULKDELTAF == 1)
            {
              if (bulk_deltaf_kind == 0) delta_f_bulk = (- (1. - F0_IS_NOT_SMALL * sign * f0) * bulkPi[icell_glb] * (bulkvisCoefficients[0] * mass * mass + bulkvisCoefficients[1] * pdotu + bulkvisCoefficients[2] * pdotu * pdotu));
              else if (bulk_deltaf_kind == 1)
              {
                double E_over_T = pdotu / Tdec[icell_glb];
                double mass_over_T = mass / Tdec[icell_glb];
                delta_f_bulk = (-1.0 * (1. - sign * f0)/E_over_T * bulkvisCoefficients[0] * (mass_over_T * mass_over_T/3. - bulkvisCoefficients[1] * E_over_T * E_over_T) * bulkPi[icell_glb]);
              }
              else if (bulk_deltaf_kind == 2)
              {
                double E_over_T = pdotu / Tdec[icell_glb];
                delta_f_bulk = (-1.*(1.-sign * f0) * (-bulkvisCoefficients[0] + bulkvisCoefficients[1] * E_over_T) * bulkPi[icell_glb]);
              }
              else if (bulk_deltaf_kind == 3)
              {
                double E_over_T = pdotu / Tdec[icell_glb];
                delta_f_bulk = (-1.0*(1.-sign * f0) / sqrt(E_over_T) * (-bulkvisCoefficients[0] + bulkvisCoefficients[1] * E_over_T) * bulkPi[icell_glb]);
              }
              else if (bulk_deltaf_kind == 4)
              {
                double E_over_T = pdotu / Tdec[icell_glb];
                delta_f_bulk = (-1.0*(1.-sign * f0) * (bulkvisCoefficients[0] - bulkvisCoefficients[1] / E_over_T) * bulkPi[icell_glb]);
              }
            }

            double ratio = min(1., fabs(1. / (delta_f_shear + delta_f_bulk)));
            long long int ir = icell + (FO_chunk * ipT) + (FO_chunk * pT_tab_length * iphip) + (FO_chunk * pT_tab_length * phi_tab_length * iy);
            dN_pTdpTdphidy_all[ir] = (prefactor * degen * pdotdsigma * tau[icell] * f0 * (1. + (delta_f_shear + delta_f_bulk) * ratio));
          } //iy
        } //iphip
      } //ipT
    } //icell
    printf("performing reduction over cells\n");
    //now perform the reduction over cells
    //this section of code is quite fast compared to filling dN_xxx_all when using multiple threads openmp
    #pragma omp parallel for collapse(3)
    #pragma acc kernels
    for (int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      for (int iphip = 0; iphip < phi_tab_length; iphip++)
      {
        for (int iy = 0; iy < y_tab_length; iy++)
        {
          long long int is = ipT + (pT_tab_length * iphip) + (pT_tab_length * phi_tab_length * iy);
          double dN_pTdpTdphidy_3D_tmp = 0.0; //reduction variable
          #pragma omp simd reduction(+:dN_pTdpTdphidy_3D_tmp)
          for (int icell = 0; icell < FO_chunk; icell++)
          {
            long long int ir = icell + (FO_chunk * ipT) + (FO_chunk * pT_tab_length * iphip) + (FO_chunk * pT_tab_length * phi_tab_length * iy);
            dN_pTdpTdphidy_3D_tmp += dN_pTdpTdphidy_all[ir];
          }//icell
           dN_pTdpTdphidy_3D[is] += dN_pTdpTdphidy_3D_tmp; //cumulative sum over all chunks
        }//iy
      }//iphip
    }//ipT
  }//n FO chunk

  //free memory
  free(dN_pTdpTdphidy_all);
  free(Tdec);
  free(Pdec);
  free(Edec);
  free(mu);
  free(tau);
  free(eta);
  free(utau);
  free(ux);
  free(uy);
  free(ueta);
  free(datau);
  free(dax);
  free(day);
  free(daeta);
  free(pi00);
  free(pi01);
  free(pi02);
  free(pi11);
  free(pi12);
  free(pi22);
  free(pi33);
  free(muB);
  free(bulkPi);
}

void EmissionFunctionArray::write_dN_ptdptdphidy_toFile(int monval) //pass monte carlo ID as argument for filename
// Append the dN_xxx results to file.
{
  printf("writing to file\n");
  //write 3D spectra in block format, different blocks for different values of rapidity,
  //rows corespond to phip and columns correspond to pT
  char filename[255] = "";
  sprintf(filename, "results/%d_spectra_3D.dat", monval);
  ofstream spectra3DFile(filename, ios_base::app);
  //this writes in block format , different blocks correspond to different values of y
  //in each block, the row corresponds to the value of phip and the column to the value of pT
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
  spectra3DFile.close();

}


//*********************************************************************************************
void EmissionFunctionArray::calculate_dN_ptdptdphidy_4all(int to_order)
// Calculate dNArrays and flows for all particles given in chosen_particle file.
{
  cout << endl
  << "****************************************************************"
  << endl
  << "Function calculate_dN_ptdptdphidy_4all started... " << endl;
  Stopwatch sw;
  sw.tic();

  // loop over chosen particles
  particle_info* particle = NULL;
  for (int m=0; m<number_of_chosen_particles; m++)
  {
    int particle_idx = chosen_particles_sampling_table[m];
    particle = &particles[particle_idx];
    int monval = particle->monval;
    cout << "Index: " << m << ", Name: " << particle->name << ", Monte-carlo index: " << monval << endl;
    // Calculate dN / (ptdpt dphi dy)
    if (m>0 && particles_are_the_same(particle_idx, chosen_particles_sampling_table[m-1]))
    {
      cout << " -- Using dN_ptdptdphidy from previous calculation... " << endl;
      last_particle_idx = particle_idx; // fake a calculation
    }
    else
    {
      cout << " -- Calculating dN_ptdptdphidy..." << endl;
      //calculate_dN_ptdptdphidy(particle_idx);  //for 2D spectra
      calculate_dN_ptdptdphidy_parallel_3(particle_idx); //for 3D spectra
      //then write to file here
      write_dN_ptdptdphidy3d_toFile(monval); //write the spectra for one species
    }

  }
  sw.toc();
  cout << " -- Calculate_dN_ptdptdphidy_and_flows_4all finishes " << sw.takeTime() << " seconds." << endl;

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
