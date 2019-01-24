#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <array>
#ifdef _OMP
#include <omp.h>
#endif
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

// Class Lab_Momentum
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

double equilibrium_particle_density(double mass, double degeneracy, double sign, double T, double chem)
{
  // may want to use a direct gauss quadrature instead

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

  // Aij is symmetric
  double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);

  return detA;
}

bool is_linear_pion0_density_negative(double T, double neq_pion0, double J20_pion0, double bulkPi, double F, double betabulk)
{
  // determine if linear pion0 density goes negative

  // calculate the pion-0 density by itself: that way I don't need to include multiple particles when testing single particle spectra

  double dn_pion0 = bulkPi * (neq_pion0  +  J20_pion0 * F / T / T) / betabulk;

  double nlinear_pion0 = neq_pion0 + dn_pion0;

  if(nlinear_pion0 < 0.0) return true;

  return false;
}

bool does_feqmod_breakdown(double detA, double detA_min, bool pion_density_negative)
{
  if(detA <= detA_min || pion_density_negative) return true;

  return false;
}

// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in,
  Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in,
  int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in,  deltaf_coefficients * df_in, Deltaf_Data * df_data_in)
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

    // for binning the sampled particles (for sampler tests)
    PT_LOWER_CUT = paraRdr->getVal("pT_lower_cut");
    PT_UPPER_CUT = paraRdr->getVal("pT_upper_cut");
    PT_BINS = paraRdr->getVal("pT_bins");
    Y_CUT = paraRdr->getVal("y_cut");


    DYNAMICAL = paraRdr->getVal("dynamical");

    Nevents = 1;    // default value for number of sampled events

    particles = particles_in;
    Nparticles = Nparticles_in;
    surf_ptr = surf_ptr_in;
    FO_length = FO_length_in;
    df = df_in;
    df_data = df_data_in;
    number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

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
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << value << "\n";
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

            dN_dy += pT * phip_gauss_weight * pT_gauss_weight * dN_pTdpTdphidy[iS3D];
          } //ipT

        } //iphip
        spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << dN_dy << "\n";
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

      //ofstream spectraFile(filename, ios_base::app);
      ofstream spectraFile(filename, ios_base::out);

      int num_particles = particle_event_list[ievent].size();

      //write the header
      spectraFile << "#" << " " << num_particles << "\n";
      for (int ipart = 0; ipart < num_particles; ipart++)
      {
        int mcid = particle_event_list[ievent][ipart].mcID;
        double x = particle_event_list[ievent][ipart].x;
        double y = particle_event_list[ievent][ipart].y;
        double t = particle_event_list[ievent][ipart].t;
        double z = particle_event_list[ievent][ipart].z;

        //double mass = particle_list[ipart].mass;
        double E = particle_event_list[ievent][ipart].E;
        double px = particle_event_list[ievent][ipart].px;
        double py = particle_event_list[ievent][ipart].py;
        double pz = particle_event_list[ievent][ipart].pz;
        spectraFile << mcid << "," << scientific <<  setw(5) << setprecision(16) << t << "," << x << "," << y << "," << z << "," << E << "," << px << "," << py << "," << pz << "\n";
      }//ipart
      spectraFile.close();
    } // ievent
  }

  void EmissionFunctionArray::write_momentum_list_toFile()
  {
    printf("Writing sampled momentum list to file...\n");

    ofstream spectraFile("results/momentum_list.dat", ios_base::out);

    //write the header
    spectraFile << "y" << "\t" << "phi_over_pi" << "\t" << "pT" << "\n";

    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      int N = particle_event_list[ievent].size();

      for(int n = 0; n < N; n++)
      {
        double E = particle_event_list[ievent][n].E;
        double px = particle_event_list[ievent][n].px;
        double py = particle_event_list[ievent][n].py;
        double pz = particle_event_list[ievent][n].pz;

        double y = 0.5 * log((E + pz) / (E - pz));
        double phi_over_pi = atan2(py, px) / M_PI;
        if(phi_over_pi < 0.0) phi_over_pi += 2.0;
        double pT = sqrt(px * px + py * py);

        double yCut = 5.0;  // rapidity cut
        if(fabs(y) <= yCut)
        {
          //spectraFile << scientific <<  setw(5) << setprecision(6) << pT << "\n";
          spectraFile << scientific <<  setw(5) << setprecision(6) << y << "\t" << phi_over_pi << "\t" << pT << "\n";
        }
      }//ipart
    } // ievent
    spectraFile.close();
  }

  void EmissionFunctionArray::write_sampled_pT_pdf_toFile(int * MCID)
  {
    printf("Writing event-averaged pT probability density (pdfs) of each species to file...\n");

    //ofstream spectraFile("results/momentum_list.dat", ios_base::out);
    int npart = number_of_chosen_particles;
    //write the header
    //spectraFile << "y" << "\t" << "phi_over_pi" << "\t" << "pT" << "\n";

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
        int ipart = particle_event_list[ievent][n].chosen_index; // particle index of chosen particle file
        //int mcID = particle_event_list[ievent][n].mcID;
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
          if(ipT < pTbins)
          {
            pT_pdf[ipart][ipT] += 1.0;                // add counts to each bin
            number_of_sampled_particles[ipart] += 1;  // count number

          } // pT cut

        } // rapidity cut

      }// n

    } // ievent

    // now normalize the pdfs to unity and write them to file
    for(int ipart = 0; ipart < npart; ipart++)
    {
      char filename[255] = "";

      int mcid = MCID[ipart]; // we can just use this in place

      sprintf(filename, "results/pT_pdf_%d.dat", mcid);

      ofstream spectra(filename, ios_base::out);

      // total number of particles of species ipart at top of file
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

  void EmissionFunctionArray::write_yield_list_toFile()
  {
    printf("Writing mean yield and sampled yield list to file...\n");

    ofstream average_yield("results/mean_yield.dat", ios_base::out);
    average_yield << mean_yield << endl;
    average_yield.close();

    ofstream yield_list("results/yield_list.dat", ios_base::out);
    yield_list << "sampled particle yield\n"; // write the header

    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      yield_list << particle_yield_list[ievent] << endl;
    }

    yield_list.close();
  }

  //*********************************************************************************************
  void EmissionFunctionArray::calculate_spectra(std::vector<Sampled_Particle> &particle_event_list_in)
  {
    cout << "calculate_spectra() has started:\n\n";
    #ifdef _OPENMP
    //double sec = omp_get_wtime();
    #endif
    Stopwatch sw;
    sw.tic();

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
    gla->load_roots_and_weights("tables/gla_roots_weights_32_points.txt");


    // gauss legendre roots and weights
    Gauss_Legendre * legendre = new Gauss_Legendre;
    legendre->load_roots_and_weights("tables/gauss_legendre_24pts.dat");


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
            case 1: // smooth CFF
            {
              calculate_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data);
              break;
            }
            case 2: // sample CFF
            {
              if(OVERSAMPLE)
              {
                // average particle yield
                double Ntotal = calculate_total_yield(Mass, Sign, Degeneracy, Baryon, Equilibrium_Density, Bulk_Density,Diffusion_Density, T, P, E, tau, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);

                // number of events to sample
                Nevents = (int)ceil(MIN_NUM_HADRONS / Ntotal);
              }

              particle_event_list.resize(Nevents);
              particle_yield_list.resize(Nevents, 0);

              if(DF_MODE == 1) printf("Sampling particles with df 14 moment...\n");
              if(DF_MODE == 2) printf("Sampling particles with df Chapman Enskog...\n");

              sample_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, MCID, Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);

              write_particle_list_toFile();
              write_particle_list_OSC();
              //write_momentum_list_toFile();
              write_sampled_pT_pdf_toFile(MCID);
              write_yield_list_toFile();

              particle_event_list_in = particle_event_list[0];
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
            case 1: // smooth CF
            {
              calculate_dN_ptdptdphidy_feqmod(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, gla, df_data);
              break;
            }
            case 2: // sampler
            {
              if(OVERSAMPLE)
              {
                double Ntotal = calculate_total_yield(Mass, Sign, Degeneracy, Baryon, Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);

                Nevents = (int)ceil(MIN_NUM_HADRONS / Ntotal);
              }

              particle_event_list.resize(Nevents);
              particle_yield_list.resize(Nevents, 0);

              printf("Sampling particles with feqmod...\n");

              sample_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, MCID, Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);

              write_particle_list_toFile();
              write_particle_list_OSC();
              //write_momentum_list_toFile();
              write_sampled_pT_pdf_toFile(MCID);
              write_yield_list_toFile();

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
      write_dN_dpTdphidy_toFile(MCID);

      write_dN_dphidy_toFile(MCID);
      write_dN_dy_toFile(MCID);
      write_dN_twopipTdpTdy_toFile(MCID);
      write_dN_twopidpTdy_toFile(MCID);

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
    cout << "\ncalculate_spectra() took " << sw.takeTime() << " seconds." << endl;
  }
