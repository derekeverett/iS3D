#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <complex>
#include <array>
#include <sys/time.h>
//#ifdef _OMP
#include <omp.h>
//#endif
#include "iS3D.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "Stopwatch.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include "gaussThermal.h"
#include "particle.h"

using namespace std;

// Class Lab_Momentum (can I move this to sampling kernel?)
//------------------------------------------
Lab_Momentum::Lab_Momentum(LRF_Momentum pLRF_in)
{
    E_LRF = pLRF_in.E;
    px_LRF = pLRF_in.px;
    py_LRF = pLRF_in.py;
    pz_LRF = pLRF_in.pz;
}

void Lab_Momentum::boost_pLRF_to_lab_frame(Milne_Basis basis_vectors, double ut, double ux, double uy, double un)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    ptau  = E_LRF * ut  +  px_LRF * Xt  +  pz_LRF * Zt;
    px    = E_LRF * ux  +  px_LRF * Xx  +  py_LRF * Yx;
    py    = E_LRF * uy  +  px_LRF * Xy  +  py_LRF * Yy;
    pn    = E_LRF * un  +  px_LRF * Xn  +  pz_LRF * Zn;
}
//------------------------------------------

/*
double equilibrium_particle_density(double mass, double degeneracy, double sign, double T, double chem)
{
  // for cross-checking particle density calculation

  double neq = 0.0;
  double sign_factor = -sign;
  int jmax = 20;
  double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);
  double mbar = mass / T;

  for(int j = 1; j < jmax; j++)            // sum truncated expansion of Bose-Fermi distribution
  {
    double k = (double)j;
    sign_factor *= (-sign);
    neq += sign_factor * exp(k * chem) * gsl_sf_bessel_Kn(2, k * mbar) / k;
  }
  neq *= degeneracy * mass * mass * T / two_pi2_hbarC3;

  return neq;
}
*/

double compute_detA(Shear_Stress pimunu, double shear_mod, double bulk_mod)
{
  double pixx_LRF = pimunu.pixx_LRF;  double piyy_LRF = pimunu.piyy_LRF;
  double pixy_LRF = pimunu.pixy_LRF;  double piyz_LRF = pimunu.piyz_LRF;
  double pixz_LRF = pimunu.pixz_LRF;  double pizz_LRF = pimunu.pizz_LRF;

  double Axx = 1.0  +  pixx_LRF * shear_mod  +  bulk_mod;
  double Axy = pixy_LRF * shear_mod;
  double Axz = pixz_LRF * shear_mod;
  double Ayy = 1.0  +  piyy_LRF * shear_mod  +  bulk_mod;
  double Ayz = piyz_LRF * shear_mod;
  double Azz = 1.0  +  pizz_LRF * shear_mod  +  bulk_mod;

  // assume Aij is symmetric (need to change this formula if include diffusion)
  double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);

  return detA;
}

bool is_linear_pion0_density_negative(double T, double neq_pion0, double J20_pion0, double bulkPi, double F, double betabulk)
{
  // determine if linear pion0 density goes negative

  double dn_pion0 = bulkPi * (neq_pion0  +  J20_pion0 * F / T / T) / betabulk;

  double nlinear_pion0 = neq_pion0 + dn_pion0;

  if(nlinear_pion0 < 0.0) return true;

  return false;
}

bool does_feqmod_breakdown(double mass_pion0, double T, double F, double bulkPi, double betabulk, double detA, double detA_min, double z, Gauss_Laguerre * laguerre, int df_mode)
{
  if(df_mode == 3)
  {
    const int laguerre_pts = laguerre->points;
    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];
    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    // calculate linearized pion density
    double mbar_pion0 = mass_pion0 / T;

    double neq_fact = T * T * T / two_pi2_hbarC3;
    double J20_fact = T * neq_fact;

    double neq_pion0 = neq_fact * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar_pion0, 0., 0., -1.);
    double J20_pion0 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, laguerre_pts, mbar_pion0, 0., 0., -1.);

    bool pion_density_negative = is_linear_pion0_density_negative(T, neq_pion0, J20_pion0, bulkPi, F, betabulk);

    if(detA <= detA_min || pion_density_negative) return true;
  }
  else if(df_mode == 4)
  {
    if(z < 0.0) printf("Error: z should be positive");

    if(detA <= detA_min || z < 0.0) return true;
  }

  return false;
}



// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in,
  Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in,
  int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in, Deltaf_Data * df_data_in)
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
    OPERATION = paraRdr->getVal("operation");
    MODE = paraRdr->getVal("mode");
    DF_MODE = paraRdr->getVal("df_mode");
    DIMENSION = paraRdr->getVal("dimension");
    INCLUDE_BARYON = paraRdr->getVal("include_baryon");
    INCLUDE_BULK_DELTAF = paraRdr->getVal("include_bulk_deltaf");
    INCLUDE_SHEAR_DELTAF = paraRdr->getVal("include_shear_deltaf");
    INCLUDE_BARYONDIFF_DELTAF = paraRdr->getVal("include_baryondiff_deltaf");

    REGULATE_DELTAF = paraRdr->getVal("regulate_deltaf");
    OUTFLOW = paraRdr->getVal("outflow");

    DETA_MIN = paraRdr->getVal("deta_min");
    GROUP_PARTICLES = paraRdr->getVal("group_particles");
    PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");

    MASS_PION0 = paraRdr->getVal("mass_pion0");

    LIGHTEST_PARTICLE = paraRdr->getVal("lightest_particle");
    DO_RESONANCE_DECAYS = paraRdr->getVal("do_resonance_decays");

    OVERSAMPLE = paraRdr->getVal("oversample");
    MIN_NUM_HADRONS = paraRdr->getVal("min_num_hadrons");
    SAMPLER_SEED = paraRdr->getVal("sampler_seed");
    if (OPERATION == 2) printf("Sampler seed set to %d \n", SAMPLER_SEED);

    TEST_SAMPLER = paraRdr->getVal("test_sampler");

    Nevents = 1;    // default value for number of sampled events

    // for binning the sampled particles (for sampler tests)
    PT_LOWER_CUT = paraRdr->getVal("pT_lower_cut");
    PT_UPPER_CUT = paraRdr->getVal("pT_upper_cut");
    PT_BINS = paraRdr->getVal("pT_bins");

    Y_CUT = paraRdr->getVal("y_cut");
    Y_BINS = paraRdr->getVal("y_bins");

    ETA_CUT = paraRdr->getVal("eta_cut");
    ETA_BINS = paraRdr->getVal("eta_bins");

    // for binning the sampled particles (for sampler tests)
    TAU_MIN = paraRdr->getVal("tau_min");
    TAU_MAX = paraRdr->getVal("tau_max");
    TAU_BINS = paraRdr->getVal("tau_bins");

    R_MIN = paraRdr->getVal("r_min");
    R_MAX = paraRdr->getVal("r_max");
    R_BINS = paraRdr->getVal("r_bins");

    particles = particles_in;
    Nparticles = Nparticles_in;
    surf_ptr = surf_ptr_in;
    FO_length = FO_length_in;
    df_data = df_data_in;
    number_of_chosen_particles = chosen_particles_in->getNumberOfRows();


    // allocate memory for sampled distributions / spectra (for sampler testing)
    dN_dy_count = (double *)calloc(number_of_chosen_particles, sizeof(double));

    total_count = (double *)calloc(number_of_chosen_particles, sizeof(double));
    sampled_pT_PDF = (double **)calloc(number_of_chosen_particles, sizeof(double));

    pT_count_vn = (double **)calloc(number_of_chosen_particles, sizeof(double));
    sampled_vn_real = (double ***)calloc(K_MAX, sizeof(double));
    sampled_vn_imag = (double ***)calloc(K_MAX, sizeof(double));

    sampled_dN_taudtaudy = (double **)calloc(number_of_chosen_particles, sizeof(double));
    sampled_dN_twopirdrdy = (double **)calloc(number_of_chosen_particles, sizeof(double));

    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      pT_count_vn[ipart] = (double *)calloc(PT_BINS, sizeof(double));

      sampled_pT_PDF[ipart] = (double *)calloc(PT_BINS, sizeof(double));

      sampled_dN_taudtaudy[ipart] = (double *)calloc(TAU_BINS, sizeof(double));
      sampled_dN_twopirdrdy[ipart] = (double *)calloc(R_BINS, sizeof(double));
    }

    for(int k = 0; k < K_MAX; k++)
    {
      sampled_vn_real[k] = (double **)calloc(number_of_chosen_particles, sizeof(double));
      sampled_vn_imag[k] = (double **)calloc(number_of_chosen_particles, sizeof(double));

      for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
      {
        sampled_vn_real[k][ipart] = (double *)calloc(PT_BINS, sizeof(double));
        sampled_vn_imag[k][ipart] = (double *)calloc(PT_BINS, sizeof(double));
      }
    }




    chosen_particles_01_table = new int[Nparticles];

    //a class member to hold 3D smooth CF spectra for all chosen particles
    dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
    // holds smooth CF spectra of a given parent resonance
    logdN_PTdPTdPhidY = new double [pT_tab_length * phi_tab_length * y_tab_length];

    //zero the array
    for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
    {
      dN_pTdpTdphidy[iSpectra] = 0.0;
    }
    for(int iS_parent = 0; iS_parent < pT_tab_length * phi_tab_length * y_tab_length; iS_parent++)
    {
      logdN_PTdPTdPhidY[iS_parent] = 0.0; // is it harmful to have a y_tab_length =/= 1 if DIMENSION = 2 (waste of memory?)
    }

    if (MODE == 5)
    {
      //class member to hold polarization vector of chosen particles
      St = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      Sx = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      Sy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      Sn = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      //holds the normalization of the polarization vector of chosen particles
      Snorm = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];

      for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
      {
        St[iSpectra] = 0.0;
        Sx[iSpectra] = 0.0;
        Sy[iSpectra] = 0.0;
        Sn[iSpectra] = 0.0;
        Snorm[iSpectra] = 0.0;
      }
    } // if (MODE == 5)

    for (int n = 0; n < Nparticles; n++) chosen_particles_01_table[n] = 0;

    //only grab chosen particles from the table
    for (int m = 0; m < number_of_chosen_particles; m++)
    { //loop over all chosen particles
      int mc_id = chosen_particles_in->get(1, m + 1);

      // store chosen index of pion0
      /*
      if(mc_id == 111)
      {
        chosen_pion0.push_back(m);
      }
      */

      for (int n = 0; n < Nparticles; n++)
      {
        if (particles[n].mc_id == mc_id)
        {
          chosen_particles_01_table[n] = 1;
          break;
        }
      }
    } // for (int m = 0; m < number_of_chosen_particles; m++)

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
    } //for (int m = 0; m < number_of_chosen_particles; m++)

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
        } // for (int n = 0; n < number_of_chosen_particles - m - 1; n++)
      } // for (int m = 0; m < number_of_chosen_particles; m++)
    } // if (GROUP_PARTICLES == 1)
  } // EmissionFunctionArray::EmissionFunctionArray

  EmissionFunctionArray::~EmissionFunctionArray()
  {
    delete[] chosen_particles_01_table;
    delete[] chosen_particles_sampling_table;
    delete[] dN_pTdpTdphidy; //for holding 3d spectra of all chosen particles
    delete[] logdN_PTdPTdPhidY;
  }


  void EmissionFunctionArray::write_dN_pTdpTdphidy_toFile(int *MCID)
  {
    printf("Writing thermal spectra to file...\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts

    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    sprintf(filename, "results/dN_pTdpTdphidy.dat");
    ofstream spectraFile(filename, ios_base::app);
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;

        if (DIMENSION == 2) y = 0.0;

        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();

    //now write a separate file for each species
    for (int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/dN_pTdpTdphidy_%d.dat", mcid);
      ofstream spectraFile(filename, ios_base::app);
      //write the header
      spectraFile << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_pTdpTdphidy" << "\n";
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;

        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
      spectraFile.close();
    }
  }

    void EmissionFunctionArray::write_dN_pTdpTdphidy_with_resonance_decays_toFile()
  {
    printf("Writing thermal + resonance decays spectra to file...\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    sprintf(filename, "results/dN_pTdpTdphidy_resonance_decays.dat");
    ofstream spectraFile(filename, ios_base::app);
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }

  void EmissionFunctionArray::write_dN_dpTdphidy_toFile(int *MCID)
  {
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";
    int y_pts = y_tab_length;     // default 3+1d pts
    if (DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)
    sprintf(filename, "results/dN_dpTdphidy.dat");
    ofstream spectraFile(filename, ios_base::app);
    //write the header
    spectraFile << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_dpTdphidy" << "\n";
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            double value = dN_pTdpTdphidy[iS3D] * pT;
            spectraFile << scientific << setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << value << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();

    //now write a separate file for each species
    for (int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/dN_dpTdphidy_%d.dat", mcid);
      ofstream spectraFile(filename, ios_base::app);
      //write the header
      spectraFile << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_dpTdphidy" << "\n";
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            double value = dN_pTdpTdphidy[iS3D] * pT;
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << value << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    } //ipart
  }

  void EmissionFunctionArray::write_dN_dpTdphidy_with_resonance_decays_toFile()
  {
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";
    int y_pts = y_tab_length;     // default 3+1d pts
    if (DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)
    sprintf(filename, "results/dN_dpTdphidy_resonance_decays.dat");
    ofstream spectraFile(filename, ios_base::app);
    //write the header
    spectraFile << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_dpTdphidy" << "\n";
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            double value = dN_pTdpTdphidy[iS3D] * pT;
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << value << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }


  void EmissionFunctionArray::write_dN_dphidy_toFile(int *MCID)
  {
    printf("Writing thermal dN_dphidy to file...\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    // write a separate file for each species
    for (int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/dN_dphidy_%d.dat", mcid);
      ofstream spectraFile(filename, ios_base::app);
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          double dN_dphidy = 0.0;

          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            double pT_gauss_weight = pT_tab->get(2, ipT + 1);

            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

            dN_dphidy += pT_gauss_weight * pT * dN_pTdpTdphidy[iS3D];
          } //ipT
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << dN_dphidy << "\n";
        } //iphip
        spectraFile << "\n";
      } //iy
      spectraFile.close();
    }
  }

  void EmissionFunctionArray::write_dN_twopipTdpTdy_toFile(int *MCID)
  {
    printf("Writing thermal dN_twopipTdpTdy to file...\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    // write a separate file for each species
    for(int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/dN_twopipTdpTdy_%d.dat", mcid);
      ofstream spectraFile(filename, ios_base::app);
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y = y_tab->get(1, iy + 1);
        if(DIMENSION == 2) y = 0.0;

        for(int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pT_tab->get(1, ipT + 1);
          double dN_twopipTdpTdy = 0.0;

          for (int iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double phip = phi_tab->get(1, iphip + 1);
            double phip_gauss_weight = phi_tab->get(2, iphip + 1);

            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

            dN_twopipTdpTdy += phip_gauss_weight * dN_pTdpTdphidy[iS3D] / (2.0 * M_PI);
          } //iphip
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT << "\t" << dN_twopipTdpTdy << "\n";
        } //ipT
        spectraFile << "\n";
      } //iy
      spectraFile.close();
    }
  }

  void EmissionFunctionArray::write_dN_twopidpTdy_toFile(int *MCID)
  {
    printf("Writing thermal dN_twopidpTdy to file...\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    // write a separate file for each species
    for(int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/dN_twopidpTdy_%d.dat", mcid);
      ofstream spectraFile(filename, ios_base::app);
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y = y_tab->get(1, iy + 1);;
        if(DIMENSION == 2) y = 0.0;

        for(int ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pT_tab->get(1, ipT + 1);
          double dN_twopipTdpTdy = 0.0;

          for (int iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double phip = phi_tab->get(1, iphip + 1);
            double phip_gauss_weight = phi_tab->get(2, iphip + 1);

            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

            dN_twopipTdpTdy += phip_gauss_weight * dN_pTdpTdphidy[iS3D] / (2.0 * M_PI);
          } //iphip
          spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT << "\t" << dN_twopipTdpTdy * pT << "\n";
        } //ipT
        spectraFile << "\n";
      } //iy
      spectraFile.close();
    }
  }

  void EmissionFunctionArray::write_dN_dy_toFile(int *MCID)
  {
    printf("Writing thermal dN_dy to file...\n");

    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    //write a separate file for each species
    for(int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/dN_dy_%d.dat", mcid);
      ofstream spectraFile(filename, ios_base::app);
      for(int iy = 0; iy < y_pts; iy++)
      {
        double y = y_tab->get(1, iy + 1);
        if(DIMENSION == 2) y = 0.0;

        double dN_dy = 0.0;

        for(int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1, iphip + 1);
          double phip_gauss_weight = phi_tab->get(2, iphip + 1);

          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1, ipT + 1);
            double pT_gauss_weight = pT_tab->get(2, ipT + 1);

            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

            dN_dy += phip_gauss_weight * pT_gauss_weight * dN_pTdpTdphidy[iS3D];
          } //ipT

        } //iphip
        spectraFile << setw(5) << setprecision(8) << y << "\t" << dN_dy << "\n";
      } //iy
      spectraFile.close();
    }
  }


  void EmissionFunctionArray::write_polzn_vector_toFile()
  {
    printf("Writing polarization vector to file...\n");
    int npart = number_of_chosen_particles;
    char filename_t[255] = "";
    char filename_x[255] = "";
    char filename_y[255] = "";
    char filename_n[255] = "";
    sprintf(filename_t, "results/St.dat");
    sprintf(filename_x, "results/Sx.dat");
    sprintf(filename_y, "results/Sy.dat");
    sprintf(filename_n, "results/Sn.dat");
    ofstream StFile(filename_t, ios_base::app);
    ofstream SxFile(filename_x, ios_base::app);
    ofstream SyFile(filename_y, ios_base::app);
    ofstream SnFile(filename_n, ios_base::app);

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));
            StFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (St[iS3D] / Snorm[iS3D]) << "\n";
            SxFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (Sx[iS3D] / Snorm[iS3D]) << "\n";
            SyFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (Sy[iS3D] / Snorm[iS3D]) << "\n";
            SnFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (Sn[iS3D] / Snorm[iS3D]) << "\n";

          } //ipT
          StFile << "\n";
          SxFile << "\n";
          SyFile << "\n";
          SnFile << "\n";
        } //iphip
      } //iy
    }//ipart
    StFile.close();
    SxFile.close();
    SyFile.close();
    SnFile.close();
  }

  void EmissionFunctionArray::write_particle_list_toFile()
  {
    printf("Writing sampled particles list to file...\n");

    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      char filename[255] = "";
      sprintf(filename, "results/particle_list_%d.dat", ievent + 1);

      //ofstream spectraFile(filename, ios_base::app);
      ofstream spectraFile(filename, ios_base::out);

      int num_particles = particle_event_list[ievent].size();

      //write the header
      spectraFile << "mcid" << "," << "tau" << "," << "x" << "," << "y" << "," << "eta" << "," << "E" << "," << "px" << "," << "py" << "," << "pz" << "\n";
      for (int ipart = 0; ipart < num_particles; ipart++)
      {
        int mcid = particle_event_list[ievent][ipart].mcID;
        double tau = particle_event_list[ievent][ipart].tau;
        double x = particle_event_list[ievent][ipart].x;
        double y = particle_event_list[ievent][ipart].y;
        double eta = particle_event_list[ievent][ipart].eta;
        double E = particle_event_list[ievent][ipart].E;
        double px = particle_event_list[ievent][ipart].px;
        double py = particle_event_list[ievent][ipart].py;
        double pz = particle_event_list[ievent][ipart].pz;
        spectraFile << scientific <<  setw(5) << setprecision(8) << mcid << "," << tau << "," << x << "," << y << "," << eta << "," << E << "," << px << "," << py << "," << pz << "\n";
      }//ipart
      spectraFile.close();
    } // ievent
  }

  //write particle list in oscar format for UrQMD/SMASH afterburner
  void EmissionFunctionArray::write_particle_list_OSC()
  {
    printf("Writing sampled particles list to OSCAR File...\n");

    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      char filename[255] = "";
      sprintf(filename, "results/particle_list_osc_%d.dat", ievent + 1);

      ofstream spectraFile(filename, ios_base::out);

      int num_particles = particle_event_list[ievent].size();

      //write the header
      //spectraFile << "#" << " " << num_particles << "\n";
      spectraFile << "n pid px py pz E m x y z t" << "\n";
      for (int ipart = 0; ipart < num_particles; ipart++)
      {
        int mcid = particle_event_list[ievent][ipart].mcID;
        double x = particle_event_list[ievent][ipart].x;
        double y = particle_event_list[ievent][ipart].y;
        double t = particle_event_list[ievent][ipart].t;
        double z = particle_event_list[ievent][ipart].z;

        double m  = particle_event_list[ievent][ipart].mass;
        double E  = particle_event_list[ievent][ipart].E;
        double px = particle_event_list[ievent][ipart].px;
        double py = particle_event_list[ievent][ipart].py;
        double pz = particle_event_list[ievent][ipart].pz;
        spectraFile << ipart << " " << mcid << " " << scientific <<  setw(5) << setprecision(16) << px << " " << py << " " << pz << " " << E << " " << m << " " << x << " " << y << " " << z << " " << t << "\n";
      }//ipart
      spectraFile.close();
    } // ievent
  }

  void EmissionFunctionArray::write_sampled_dN_dy_average_to_file_test(int * MCID)
  {
    printf("Writing event-averaged (1/N) * dN/dy of each species to file...\n");

    const int npart = number_of_chosen_particles;

    // write (1/N)dN/dy distribution each species
    for(int ipart = 0; ipart < npart; ipart++)
    {
      char filename[255] = "";
      int mcid = MCID[ipart]; // we can just use this in place
      sprintf(filename, "results/dN_dy/dN_dy_%d_test.dat", mcid);
      ofstream spectra(filename, ios_base::out);

      // dN/dy averaged over rapidity window (assumes boost invariance in this window)
      spectra << dN_dy_count[ipart] / (Nevents * 2.0 * Y_CUT) << endl;

      spectra.close();

    } // ipart

    // free memory
    free(dN_dy_count);
  }



  void EmissionFunctionArray::write_sampled_pT_PDF_to_file_test(int * MCID)
  {
    printf("Writing event-averaged (1/N) * dN/dpT(pT) of each species to file...\n");

    // set up the pT grid (midpoints of each bin)
    const double pTbinwidth = (PT_UPPER_CUT - PT_LOWER_CUT) / (double)PT_BINS;

    double pT_mid[PT_BINS];
    for(int ipT = 0; ipT < PT_BINS; ipT++)
    {
      pT_mid[ipT] = PT_LOWER_CUT  +  pTbinwidth * ((double)ipT + 0.5);
    }


    const int npart = number_of_chosen_particles;

    // write (1/N)dN/dpT distribution each species
    for(int ipart = 0; ipart < npart; ipart++)
    {
      char filename[255] = "";
      int mcid = MCID[ipart]; // we can just use this in place
      sprintf(filename, "results/momentum_distribution/pT_pdf_%d_test.dat", mcid);
      ofstream spectra(filename, ios_base::out);

      // header = total number of particles of species ipart
      spectra << total_count[ipart] << "\n";

      for(int ipT = 0; ipT < PT_BINS; ipT++)
      {
        // normalize dN/dpT
        sampled_pT_PDF[ipart][ipT] /= (pTbinwidth * total_count[ipart]);

        spectra << setprecision(6) << scientific << pT_mid[ipT] << "\t" << sampled_pT_PDF[ipart][ipT] << "\n";
      }

      spectra.close();

    } // ipart

    // free memory
    free(total_count);
    free_2D(sampled_pT_PDF, npart);
  }


  void EmissionFunctionArray::write_sampled_pT_pdf_toFile(int * MCID)
  {
    printf("Writing event-averaged pT probability density function ~ dNdpT(pT) of each species to file...\n");

    int npart = number_of_chosen_particles;

    // first set up the pT bins
    double pT_lower_cut = PT_LOWER_CUT;       // transverse momentum cuts
    double pT_upper_cut = PT_UPPER_CUT;
    double y_cut = Y_CUT;                     // rapidity cut

    int pTbins = PT_BINS;
    double pTbinwidth = (pT_upper_cut - pT_lower_cut) / (double)pTbins;
    double pT_pdf[npart][pTbins];             // event averaged pT probability distribution of the particles
    long number_of_sampled_particles[npart];  // number of particles of each species sampled from all events

    for(int ipart = 0; ipart < npart; ipart++)
    {
      // initialize to zero
      number_of_sampled_particles[ipart] = 0;
    }

    double pT_midpoint[pTbins];   // pT grid (the bin midpoints)

    // set the pT grid
    for(int ipT = 0; ipT < pTbins; ipT++)
    {
      pT_midpoint[ipT] = pT_lower_cut + pTbinwidth * ((double)ipT + 0.5);

      // initialize pdfs to zero
      for(int ipart = 0; ipart < npart; ipart++)
      {
        pT_pdf[ipart][ipT] = 0.0;
      }
    }


    // now go through all the events
    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      int N = particle_event_list[ievent].size();

      // number of particles of a given event
      for(int n = 0; n < N; n++)
      {
        int ipart = particle_event_list[ievent][n].chosen_index; // particle index of chosen particle file (oh yeah forgot to tell Derek)
        double E = particle_event_list[ievent][n].E;
        double px = particle_event_list[ievent][n].px;
        double py = particle_event_list[ievent][n].py;
        double pz = particle_event_list[ievent][n].pz;

        double y = 0.5 * log((E + pz) / (E - pz));
        double phi_over_pi = atan2(py, px) / M_PI;
        if(phi_over_pi < 0.0) phi_over_pi += 2.0;
        double pT = sqrt(px * px + py * py);

        // pT bin index
        int ipT = (int)floor(pT / pTbinwidth);

        if(fabs(y) <= y_cut)
        {
          if(ipT < pTbins) pT_pdf[ipart][ipT] += 1.0;   // add counts to each bin

          number_of_sampled_particles[ipart] += 1;      // count number for all pT
        } // rapidity cut
      }// n
    } // ievent

    // now normalize dNdpT to unity and write them to file
    for(int ipart = 0; ipart < npart; ipart++)
    {
      char filename[255] = "";
      int mcid = MCID[ipart]; // we can just use this in place
      sprintf(filename, "results/pT_pdf_%d.dat", mcid);
      ofstream spectra(filename, ios_base::out);

      // total number of particles of species ipart at top of file (header)
      spectra << number_of_sampled_particles[ipart] << "\n";

      for(int ipT = 0; ipT < pTbins; ipT++)
      {
        // normalization factor
        pT_pdf[ipart][ipT] /= (pTbinwidth * (double)number_of_sampled_particles[ipart]);
        spectra << setprecision(6) << scientific << pT_midpoint[ipT] << "\t" << pT_pdf[ipart][ipT] << "\n";
      } // ipT
      spectra.close();
    } // ipart
  }


  void EmissionFunctionArray::write_continuous_vn_toFile(int *MCID)
  {
    printf("Writing continuous vn(pT,y) to file (for testing vn's)...\n");

    char filename[255] = "";

    int npart = number_of_chosen_particles;

    int y_pts = y_tab_length;           // default 3+1d
    if(DIMENSION == 2) y_pts = 1;       // 2+1d (y = 0)

    const complex<double> I(0.0,1.0);   // imaginary i

    const int k_max = 7;                // v_n = {v_1, ..., v_7}

    // write a separate file for each species
    for(int ipart  = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];
      sprintf(filename, "results/vn_continuous/vn_%d.dat", mcid);
      ofstream vn_File(filename, ios_base::app);

      for(int iy = 0; iy < y_pts; iy++)
      {
        double y = y_tab->get(1, iy + 1);
        if(DIMENSION == 2) y = 0.0;

        for(int ipT = 0; ipT < pT_tab_length; ipT++)
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

          // gauss legendre phip integration
          for(int iphip = 0; iphip < phi_tab_length; iphip++)
          {
            // phip root/weight
            double phip = phi_tab->get(1, iphip + 1);
            double phip_weight = phi_tab->get(2, iphip + 1);

            long long int iS3D = (long long int)ipart + (long long int)npart * ((long long int)ipT + (long long int)pT_tab_length * ((long long int)iphip + (long long int)phi_tab_length * (long long int)iy));

            for(int k = 0; k < k_max; k++)
            {
              Vn_real_numerator[k] += cos(((double)k + 1.0) * phip) * phip_weight * dN_pTdpTdphidy[iS3D];
              Vn_imag_numerator[k] += sin(((double)k + 1.0) * phip) * phip_weight * dN_pTdpTdphidy[iS3D];
            }
            vn_denominator += phip_weight * dN_pTdpTdphidy[iS3D];

          } //iphip

          vn_File << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT;

          for(int k = 0; k < k_max; k++)
          {
            double vn = abs(Vn_real_numerator[k]  +  I * Vn_imag_numerator[k]) / vn_denominator;

            if(vn_denominator < 1.e-15) vn = 0.0;

            vn_File << "\t" << vn;
          }

          vn_File << "\n";

        } //ipT

        vn_File << "\n";

      } //iy

      vn_File.close();

    }

  }

  void EmissionFunctionArray::write_sampled_vn_to_file_test(int * MCID)
  {
    printf("Writing event-averaged v_n(pT) (rapidity averaged) of each species to file...\n");

    // imaginary i
    const complex<double> I(0,1.0);


    // set up the pT grid (midpoints of each bin)
    const double pTbinwidth = (PT_UPPER_CUT - PT_LOWER_CUT) / (double)PT_BINS;

    double pT_mid[PT_BINS];
    for(int ipT = 0; ipT < PT_BINS; ipT++)
    {
      pT_mid[ipT] = PT_LOWER_CUT  +  pTbinwidth * ((double)ipT + 0.5);
    }


    // write (1/N)dN/dpT distribution each species
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char filename[255] = "";
      sprintf(filename, "results/vn/vn_%d_test.dat", MCID[ipart]);
      ofstream spectra(filename, ios_base::out);

      for(int ipT = 0; ipT < PT_BINS; ipT++)
      {
        spectra << setprecision(6) << scientific << pT_mid[ipT];
        for(int k = 0; k < K_MAX; k++)
        {
          double vn_abs = abs(sampled_vn_real[k][ipart][ipT]  +  I * sampled_vn_imag[k][ipart][ipT]) / pT_count_vn[ipart][ipT];
          if(std::isnan(vn_abs) || std::isinf(vn_abs)) vn_abs = 0.0;
          spectra << "\t" << vn_abs;
        }
        spectra << "\n";
      } // ipT
      spectra.close();
    } // ipart

    // free memory
    free_2D(pT_count_vn, number_of_chosen_particles);
    free_3D(sampled_vn_real, K_MAX, number_of_chosen_particles);
    free_3D(sampled_vn_imag, K_MAX, number_of_chosen_particles);
  }


  void EmissionFunctionArray::write_sampled_vn_toFile(int * MCID)
  {
    printf("Writing event-averaged v_n(pT) (rapidity averaged) of each species to file...\n");

    const complex<double> I(0,1.0);           // imaginary i

    const int k_max = 7;                      // v_n = {v_1, ..., v_7}

    const int npart = number_of_chosen_particles;

    const double pT_lower_cut = PT_LOWER_CUT; // transverse momentum cuts
    const double pT_upper_cut = PT_UPPER_CUT;
    const double y_cut = Y_CUT;               // rapidity cut

    const int pTbins = PT_BINS;
    double pTbinwidth = (pT_upper_cut - pT_lower_cut) / (double)pTbins;

    double pT_midpoint[pTbins];               // pT grid (bins are centered at midpoints)

    double vn_real[k_max][npart][pTbins];     // event & rapidity averaged vn(pT) (real part) of each species
    double vn_imag[k_max][npart][pTbins];     // event & rapidity averaged vn(pT) (imaginary part) of each species

    long pT_count[npart][pTbins];             // number of sampled particles of each species in each pT bin for all events


    for(int ipT = 0; ipT < pTbins; ipT++)
    {
      // set the pT grid
      pT_midpoint[ipT] = pT_lower_cut + pTbinwidth * ((double)ipT + 0.5);

      // initialize vn and pT_count to zero
      for(int ipart = 0; ipart < npart; ipart++)
      {
        pT_count[ipart][ipT] = 0.0;

        for(int k = 0; k < k_max; k++)
        {
          vn_real[k][ipart][ipT] = 0.0;
          vn_imag[k][ipart][ipT] = 0.0;
        }
      }
    }

    // now go through all the events
    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      int N = particle_event_list[ievent].size();

      // number of particles of a given event
      for(int n = 0; n < N; n++)
      {
        int ipart = particle_event_list[ievent][n].chosen_index;
        double E = particle_event_list[ievent][n].E;
        double px = particle_event_list[ievent][n].px;
        double py = particle_event_list[ievent][n].py;
        double pz = particle_event_list[ievent][n].pz;

        double y = 0.5 * log((E + pz) / (E - pz));
        double phi = atan2(py, px);
        if(phi < 0.0) phi += 2.0 * M_PI;
        double pT = sqrt(px * px + py * py);

        // pT bin index
        int ipT = (int)floor(pT / pTbinwidth);

        if(fabs(y) <= y_cut)
        {
          if(ipT < pTbins)
          {
            // add counts to each pT bin
            pT_count[ipart][ipT] += 1;

            for(int k = 0; k < k_max; k++)
            {
              // add exponential weights to each bin
              vn_real[k][ipart][ipT] += cos(((double)k + 1.0) * phi);
              vn_imag[k][ipart][ipT] += sin(((double)k + 1.0) * phi);
            }
          }
        } // rapidity cut
      }// n
    } // ievent

    // now normalize v_n's by the number of particles in each pT bin and write to file;
    for(int ipart = 0; ipart < npart; ipart++)
    {
      char filename[255] = "";
      int mcid = MCID[ipart]; // we can just use this in place
      sprintf(filename, "results/vn/vn_%d.dat", mcid);
      ofstream spectra(filename, ios_base::out);

      for(int ipT = 0; ipT < pTbins; ipT++)
      {
        spectra << setprecision(6) << scientific << pT_midpoint[ipT];
        for(int k = 0; k < k_max; k++)
        {
          double result = abs(vn_real[k][ipart][ipT]  +  I * vn_imag[k][ipart][ipT]) / (double)pT_count[ipart][ipT];
          if(std::isnan(result)) result = 0.0;
          spectra << "\t" << result;
        }
        spectra << "\n";
      } // ipT
      spectra.close();
    } // ipart
  }


  void EmissionFunctionArray::write_sampled_dN_dX_to_file_test(int * MCID)
  {
    printf("Writing event-averaged boost invariant spacetime distributions dN_dX of each species to file...\n");

    // dX = tau.dtau.deta or 2.pi.r.dr.deta
    // only have boost invariance in mind right now so deta <-> dyp

    const double taubinwidth = (TAU_MAX - TAU_MIN) / (double)TAU_BINS;
    double tau_mid[TAU_BINS];
    for(int itau = 0; itau < TAU_BINS; itau++)
    {
      tau_mid[itau] = TAU_MIN + taubinwidth * ((double)itau + 0.5);
    }

    const double rbinwidth = (R_MAX - R_MIN) / (double)R_BINS;
    double r_mid[R_BINS];
    for(int ir = 0; ir < R_BINS; ir++)
    {
      r_mid[ir] = R_MIN + rbinwidth * ((double)ir + 0.5);
    }


    // now event-average dN_dXdy and normalize to dNdy and write them to file
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file_time[255] = "";
      char file_radial[255] = "";

      sprintf(file_time, "results/spacetime_distribution/dN_taudtaudy_sampled_%d_test.dat", MCID[ipart]);
      sprintf(file_radial, "results/spacetime_distribution/dN_twopirdrdy_sampled_%d_test.dat", MCID[ipart]);

      ofstream time_distribution(file_time, ios_base::out);
      ofstream radial_distribution(file_radial, ios_base::out);

      // normalize spacetime distributions by the binwidth, jacobian factor, events and rapidity cut range
      for(int ir = 0; ir < R_BINS; ir++)
      {
        sampled_dN_twopirdrdy[ipart][ir] /= (2.0 * M_PI * r_mid[ir] * rbinwidth * (double)Nevents * 2.0 * Y_CUT);

        radial_distribution << setprecision(6) << scientific << r_mid[ir] << "\t" << sampled_dN_twopirdrdy[ipart][ir] << "\n";
      }

      for(int itau = 0; itau < TAU_BINS; itau++)
      {
        sampled_dN_taudtaudy[ipart][itau] /= (tau_mid[itau] * taubinwidth * (double)Nevents * 2.0 * Y_CUT);

        time_distribution << setprecision(6) << scientific << tau_mid[itau] << "\t" << sampled_dN_taudtaudy[ipart][itau] << "\n";
      }

      time_distribution.close();
      radial_distribution.close();
    } // ipart

    // free memory
    free_2D(sampled_dN_taudtaudy, number_of_chosen_particles);
    free_2D(sampled_dN_twopirdrdy, number_of_chosen_particles);
  }


  void EmissionFunctionArray::write_sampled_dN_dX_toFile(int * MCID)
  {
    printf("Writing event-averaged boost invariant spacetime distributions dN_dX of each species to file...\n");

    // dX = tau.dtau.deta, 2.pi.r.dr.deta or 2.pi.tau.r.dtau.dr.deta
    // only have boost invariance in mind right now so deta = dy (rapidity)

    const int npart = number_of_chosen_particles;

    // tau grid (bin midpoints)
    const int taubins = TAU_BINS;
    const double taubinwidth = (TAU_MAX - TAU_MIN) / (double)TAU_BINS;
    double tau_midpoint[taubins];
    for(int itau = 0; itau < taubins; itau++)
    {
      tau_midpoint[itau] = TAU_MIN + taubinwidth * ((double)itau + 0.5);
    }

    // r grid (bin midpoints)
    const int rbins = R_BINS;
    const double rbinwidth = (R_MAX - R_MIN) / (double)R_BINS;
    double r_midpoint[rbins];
    for(int ir = 0; ir < rbins; ir++)
    {
      r_midpoint[ir] = R_MIN + rbinwidth * ((double)ir + 0.5);
    }

    double ** dN_taudtaudy = (double **)calloc(npart, sizeof(double));           // event averaged dN_tau.dtau.dy distribution of each species
    double ** dN_twopirdrdy = (double **)calloc(npart, sizeof(double));          // event averaged dN_twopi.r.dr.dy distribution of each species
    double *** dN_twopitaurdtaudrdy = (double ***)calloc(npart, sizeof(double)); // event averaged dN_twopi.tau.r.dtau.dr.dy distribution of each species
    long * count = (long*)calloc(npart, sizeof(long));                           // number of counts of each species from all events

    for(int ipart = 0; ipart < npart; ipart++)
    {
      dN_taudtaudy[ipart] = (double *)calloc(taubins, sizeof(double));
      dN_twopirdrdy[ipart] = (double *)calloc(taubins, sizeof(double));

      dN_twopitaurdtaudrdy[ipart] = (double **)calloc(taubins, sizeof(double));
      for(int itau = 0; itau < taubins; itau++) dN_twopitaurdtaudrdy[ipart][itau] = (double *)calloc(rbins, sizeof(double));
    }

    // now go through all the events
    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      int N = particle_event_list[ievent].size();

      // number of particles of a given event
      for(int n = 0; n < N; n++)
      {
        int ipart = particle_event_list[ievent][n].chosen_index;
        double tau = particle_event_list[ievent][n].tau;
        double x = particle_event_list[ievent][n].x;
        double y = particle_event_list[ievent][n].y;
        double E = particle_event_list[ievent][n].E;
        double pz = particle_event_list[ievent][n].pz;

        double r = sqrt(x * x  +  y * y);
        double yp = 0.5 * log((E + pz) / (E - pz));

        int itau = (int)floor(tau / taubinwidth); // tau bin index
        int ir = (int)floor(r / rbinwidth);       // r bin index

        if(itau < 0) printf("Error: tau bin index is negative\n");
        if(ir < 0) printf("Error: r bin index is negative\n");

        if(fabs(yp) <= Y_CUT)
        {
          // total count
          count[ipart] += 1;

          // add count to the corresponding bin(s)
          if(itau >= 0 && itau < taubins)
          {
            dN_taudtaudy[ipart][itau] += 1.0;

            if(ir >= 0 && ir < rbins)
            {
              dN_twopitaurdtaudrdy[ipart][itau][ir] += 1.0;
            }
          }
          if(ir < rbins)
          {
            dN_twopirdrdy[ipart][ir] += 1.0;
          }
        }

      } // n
    } // ievent

    // now event-average dN_dXdy and normalize to dNdy and write them to file
    for(int ipart = 0; ipart < npart; ipart++)
    {
      int mcid = MCID[ipart];

      char file_time[255] = "";
      char file_radial[255] = "";
      char file_timeradial[255] = "";

      sprintf(file_time, "results/spacetime_distribution/dN_taudtaudy_sampled_%d.dat", mcid);
      sprintf(file_radial, "results/spacetime_distribution/dN_twopirdrdy_sampled_%d.dat", mcid);
      sprintf(file_timeradial, "results/spacetime_distribution/dN_twopitaurdtaudrdy_sampled_%d.dat", mcid);

      ofstream time_distribution(file_time, ios_base::out);
      ofstream radial_distribution(file_radial, ios_base::out);
      ofstream timeradial_distribution(file_timeradial, ios_base::out);

      // normalize by the binwidth(s), jacobian factor(s), events and rapidity cut
      for(int ir = 0; ir < rbins; ir++)
      {
        double r_mid = r_midpoint[ir];

        dN_twopirdrdy[ipart][ir] /= (2.0 * M_PI * r_mid * rbinwidth * (double)Nevents * 2.0 * Y_CUT);

        radial_distribution << setprecision(6) << scientific << r_midpoint[ir] << "\t" << dN_twopirdrdy[ipart][ir] << "\n";

        for(int itau = 0; itau < taubins; itau++)
        {
          double tau_mid = tau_midpoint[itau];

          dN_twopitaurdtaudrdy[ipart][itau][ir] /= (2.0 * M_PI * tau_mid * r_mid * rbinwidth * taubinwidth * (double)Nevents * 2.0 * Y_CUT);

          timeradial_distribution << setprecision(6) << scientific << tau_mid << "\t" << r_mid << "\t" << dN_twopitaurdtaudrdy[ipart][itau][ir] << "\n";
        }
      }

      // normalize by the binwidth, jacobian factors(s), events and rapidity cut
      for(int itau = 0; itau < taubins; itau++)
      {
        double tau_mid = tau_midpoint[itau];

        dN_taudtaudy[ipart][itau] /= (tau_mid * taubinwidth * (double)Nevents * 2.0 * Y_CUT);

        time_distribution << setprecision(6) << scientific << tau_mid << "\t" << dN_taudtaudy[ipart][itau] << "\n";
      }

      time_distribution.close();
      radial_distribution.close();
      timeradial_distribution.close();
    } // ipart

    // free memory
    free_2D(dN_taudtaudy, npart);
    free_2D(dN_twopirdrdy, npart);
    free_3D(dN_twopitaurdtaudrdy, npart, taubins);
    free(count);
  }


  void EmissionFunctionArray::write_yield_list_toFile()
  {
    printf("Writing mean yield and sampled yield list to file...\n");

    ofstream average_yield("results/mean_yield.dat", ios_base::out);
    average_yield << mean_yield << endl;
    average_yield.close();

    ofstream yield_list("results/yield_list.dat", ios_base::out);
    yield_list << "sampled particle yield\n"; // write the header

    for(int ievent = 0; ievent < Nevents; ievent++) yield_list << particle_yield_list[ievent] << endl;

    yield_list.close();
  }

  //*********************************************************************************************
  void EmissionFunctionArray::calculate_spectra(std::vector<Sampled_Particle> &particle_event_list_in)
  {
    cout << "calculate_spectra() has started:\n\n";
    Stopwatch sw;
    sw.tic();

    //double t1 = omp_get_wtime();

    //struct timeval t1, t2;
    //gettimeofday(&t1, NULL);

    //fill arrays with all particle info and freezeout info to pass to function which will perform the integral

    printf("Loading particle and freezeout surface arrays...\n");

    //particle info
    particle_info *particle;
    double *Mass, *Sign, *Degeneracy, *Baryon;
    int *MCID;
    double *Equilibrium_Density, *Bulk_Density, *Diffusion_Density;

    Mass = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Sign = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Degeneracy = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));

    MCID = (int*)calloc(number_of_chosen_particles, sizeof(int));

    Equilibrium_Density = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Bulk_Density = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Diffusion_Density = (double*)calloc(number_of_chosen_particles, sizeof(double));

    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      int particle_idx = chosen_particles_sampling_table[ipart];

      particle = &particles[particle_idx];

      Mass[ipart] = particle->mass;
      Sign[ipart] = particle->sign;
      Degeneracy[ipart] = particle->gspin;
      Baryon[ipart] = particle->baryon;
      MCID[ipart] = particle->mc_id;
      Equilibrium_Density[ipart] = particle->equilibrium_density;
      Bulk_Density[ipart] = particle->bulk_density;
      Diffusion_Density[ipart] = particle->diff_density;
    }

    // gauss laguerre roots and weights
    Gauss_Laguerre * gla = new Gauss_Laguerre;
    gla->load_roots_and_weights("tables/gla_roots_weights_20_points.txt");

    // gauss legendre roots and weights
    Gauss_Legendre * legendre = new Gauss_Legendre;
    legendre->load_roots_and_weights("tables/gauss_legendre_48pts.dat");

    // averaged thermodynamic quantities
    Plasma * QGP = new Plasma;
    QGP->load_thermodynamic_averages();

    // freezeout info
    FO_surf *surf = &surf_ptr[0];

    // freezeout surface info exclusive for VH
    double *E, *T, *P;
    if (MODE == 0 || MODE == 1 || MODE == 4 || MODE == 5 || MODE == 6 || MODE == 7)
    {
      E = (double*)calloc(FO_length, sizeof(double));
      P = (double*)calloc(FO_length, sizeof(double));
    }

    T = (double*)calloc(FO_length, sizeof(double));

    // freezeout surface info common for VH / VAH
    double *tau, *x, *y, *eta;
    double *ux, *uy, *un;
    double *dat, *dax, *day, *dan;
    double *pixx, *pixy, *pixn, *piyy, *piyn, *bulkPi;
    double *muB, *nB, *Vx, *Vy, *Vn;

    tau = (double*)calloc(FO_length, sizeof(double));
    x = (double*)calloc(FO_length, sizeof(double));
    y = (double*)calloc(FO_length, sizeof(double));
    eta = (double*)calloc(FO_length, sizeof(double));

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
      muB = (double*)calloc(FO_length, sizeof(double));
      nB = (double*)calloc(FO_length, sizeof(double));
      Vx = (double*)calloc(FO_length, sizeof(double));
      Vy = (double*)calloc(FO_length, sizeof(double));
      Vn = (double*)calloc(FO_length, sizeof(double));
    }

    //thermal vorticity tensor for polarization studies
    double *wtx, *wty, *wtn, *wxy, *wxn, *wyn;
    if (MODE == 5)
    {
      wtx = (double*)calloc(FO_length, sizeof(double));
      wty = (double*)calloc(FO_length, sizeof(double));
      wtn = (double*)calloc(FO_length, sizeof(double));
      wxy = (double*)calloc(FO_length, sizeof(double));
      wxn = (double*)calloc(FO_length, sizeof(double));
      wyn = (double*)calloc(FO_length, sizeof(double));
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

      if (MODE == 0 || MODE == 1 || MODE == 4 || MODE == 5 || MODE == 6 || MODE == 7)
      {
        E[icell] = surf->E;
        P[icell] = surf->P;
      }

      T[icell] = surf->T;

      tau[icell] = surf->tau;
      x[icell] = surf->x;
      y[icell] = surf->y;
      eta[icell] = surf->eta;

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
        muB[icell] = surf->muB;
        nB[icell] = surf->nB;
        Vx[icell] = surf->Vx;
        Vy[icell] = surf->Vy;
        Vn[icell] = surf->Vn;
      }

      if (MODE == 5)
      {
        wtx[icell] = surf->wtx;
        wty[icell] = surf->wty;
        wtn[icell] = surf->wtn;
        wxy[icell] = surf->wxy;
        wxn[icell] = surf->wxn;
        wyn[icell] = surf->wyn;
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

    // compute the particle spectra

    if (MODE == 0 || MODE == 1 || MODE == 4 || MODE == 5 || MODE == 6 || MODE == 7) // viscous hydro
    {
      switch(DF_MODE)
      {
        case 1: // 14 moment
        case 2: // Chapman Enskog
        {
          switch(OPERATION)
          {
            case 0: // smooth CFF spacetime distribution
            {
              calculate_dN_dX(MCID, Mass, Sign, Degeneracy, Baryon, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data);
              break;
            }
            case 1: // smooth CFF momentum distribution
            {
              calculate_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data);
              break;
            }
            case 2: // sample CFF
            {
              if(OVERSAMPLE)
              {
                // average particle yield
                double Ntotal = calculate_total_yield(Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla);

                // number of events to sample
                Nevents = (int)ceil(MIN_NUM_HADRONS / Ntotal);
              }

              printf("Sampling %d event(s)\n", Nevents);

              particle_event_list.resize(Nevents);
              particle_yield_list.resize(Nevents, 0);

              if(DF_MODE == 1) printf("Sampling particles with Grad 14 moment df...\n");
              if(DF_MODE == 2) printf("Sampling particles with Chapman Enskog df...\n");

              sample_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, MCID, Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);

              if(TEST_SAMPLER) // only for testing the sampler
              {
                write_sampled_dN_dy_average_to_file_test(MCID);
                //write_sampled_pT_pdf_toFile(MCID);
                write_sampled_pT_PDF_to_file_test(MCID);
                //write_sampled_vn_toFile(MCID);
                write_sampled_vn_to_file_test(MCID);
                //write_sampled_dN_dX_toFile(MCID);
                write_sampled_dN_dX_to_file_test(MCID);
                write_yield_list_toFile();
              }
              else // do for actual runs
              {
                write_particle_list_OSC();
              }

              particle_event_list_in = particle_event_list[0];  // only one event per core
              break;
            }
            default:
            {
              cout << "Set operation to 1 or 2" << endl;
              exit(-1);
            }
          }
          break;
        }
        case 3: // modified (Mike)
        case 4: // modified (Jonah)
        {
          switch(OPERATION)
          {
            case 0: // smooth CFF spacetime distribution
            {
              calculate_dN_dX_feqmod(MCID, Mass, Sign, Degeneracy, Baryon, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, gla, df_data);
              break;
            }
            case 1: // smooth CF
            {
              calculate_dN_ptdptdphidy_feqmod(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, gla, df_data);
              break;
            }
            case 2: // sampler
            {
              if(OVERSAMPLE)
              {
                double Ntotal = calculate_total_yield(Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla);

                Nevents = (int)ceil(MIN_NUM_HADRONS / Ntotal);
              }

              printf("Sampling %d event(s)\n", Nevents);

              particle_event_list.resize(Nevents);
              particle_yield_list.resize(Nevents, 0);

              if(DF_MODE == 3) printf("Sampling particles with Mike's modified distribution...\n");
              if(DF_MODE == 4) printf("Sampling particles with Jonah's modified distribution...\n");

              sample_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, MCID, Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);


              if(TEST_SAMPLER) // only for testing the sampler
              {
                write_sampled_dN_dy_average_to_file_test(MCID);
                //write_sampled_pT_pdf_toFile(MCID);
                write_sampled_pT_PDF_to_file_test(MCID);
                //write_sampled_vn_toFile(MCID);
                write_sampled_vn_to_file_test(MCID);
                //write_sampled_dN_dX_toFile(MCID);
                write_sampled_dN_dX_to_file_test(MCID);
                write_yield_list_toFile();
              }
              else // do for actual runs
              {
                write_particle_list_OSC();
              }

              particle_event_list_in = particle_event_list[0];

              break;
            }
            default:
            {
              cout << "Set operation to 1 or 2" << endl;
              exit(-1);
            }
          } //switch(OPERATION)
          break;
        } //case 3
        default:
        {
          cout << "Please specify df_mode = (1,2,3,4) in parameters.dat..." << endl;
          exit(-1);
        }
      } //switch(DF_MODE)
    } // if (MODE == 1 || MODE == 4 || MODE == 5 || MODE == 6)
    else if (MODE == 2)
    {
      switch(OPERATION)
      {
        case 1: // smooth CF
        {
          // calculate_dN_pTdpTdphidy_VAH_PL(Mass, Sign, Degeneracy,
          //     tau, eta, ux, uy, un,
          //     dat, dax, day, dan, T,
          //     pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
          //     Wx, Wy, Lambda, aL, c0, c1, c2, c3, c4);

          break;
        }
        case 2: // sampler
        {
          // sample_dN_pTdpTdphidy_VAH_PL(Mass, Sign, Degeneracy,
          //       tau, eta, ux, uy, un,
          //       dat, dax, day, dan, T,
          //       pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
          //       Wx, Wy, Lambda, aL, c0, c1, c2, c3, c4);
          break;
        }
        default:
        {
          cout << "Please specify df_mode = (1,2,3) in parameters.dat..." << endl;
          exit(-1);
        }
      } //switch(OPERATION)
    } //else if(MODE == 2)

    else if (MODE == 5) calculate_spin_polzn(Mass, Sign, Degeneracy, tau, eta, ux, uy, un, dat, dax, day, dan, wtx, wty, wtn, wxy, wxn, wyn, QGP);

    //write the results to file
    if (OPERATION == 1)
    {
      write_dN_pTdpTdphidy_toFile(MCID);
      //write_dN_dpTdphidy_toFile(MCID);
      write_continuous_vn_toFile(MCID);
      //write_dN_twopipTdpTdy_toFile(MCID);
      //write_dN_twopidpTdy_toFile(MCID);
      //write_dN_dphidy_toFile(MCID);
      write_dN_dy_toFile(MCID);

      // option to do resonance decays option
      if(DO_RESONANCE_DECAYS)
      {
        // call resonance decays routine
        do_resonance_decays(particles);

        // write amended spectra from resonance decays to file
        // currently, unstable particles are still included (should change)
        write_dN_pTdpTdphidy_with_resonance_decays_toFile();
        write_dN_dpTdphidy_with_resonance_decays_toFile();
      }
    }

    if (MODE == 5) write_polzn_vector_toFile();

    cout << "Freeing freezeout surface memory..." << endl;
    // free memory
    free(Mass);
    free(Sign);
    free(Baryon);

    if (MODE == 0 || MODE == 1 || MODE == 4 || MODE == 5 || MODE == 6 || MODE == 7)
    {
      free(E);
      free(P);
    }

    free(T);

    free(tau);
    free(eta);

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

    if(INCLUDE_BULK_DELTAF) free(bulkPi);

    if (MODE == 5)
    {
      free(wtx);
      free(wty);
      free(wtn);
      free(wxy);
      free(wxn);
      free(wyn);
    }

    if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
    {
      free(muB);
      free(nB);
      free(Vx);
      free(Vy);
      free(Vn);
    }

    if(MODE == 2)
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
    sw.toc();

    //double t2 = omp_get_wtime();
    cout << "\ncalculate_spectra() took " << sw.takeTime() << " seconds." << endl;
    //cout << "\ncalculate_spectra() took " << t2.tv_sec - t1.tv_sec << " seconds." << endl;
    //cout << "\ncalculate_spectra() took " << (t2 - t1) << " seconds." << endl;
  }







