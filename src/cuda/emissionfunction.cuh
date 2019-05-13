#ifndef EMISSIONFUNCTION_H
#define EMISSIONFUNCTION_H

#include<string>
#include<vector>
#include "Table.cuh"
#include "main.cuh"
#include "ParameterReader.cuh"
#include "deltafReader.cuh"
using namespace std;

class EmissionFunctionArray
{
private:
  // kernel paremeters
  long threadsPerBlock;   // number of threads per block
  long FO_chunk;          // surface chunk size


  // control parameters
  int OPERATION, MODE, DF_MODE, DIMENSION;
  int INCLUDE_BARYON, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF;
  int OUTFLOW, REGULATE_DELTAF;
  

  // freezeout surface
  long FO_length;
  FO_surf *surf_ptr;
  

  // momentum tables
  long pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length, y_minus_eta_tab_length;
  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  

  // particle info
  particle_info* particles;  
  long Nparticles;              // number of pdg particles
  long npart;                   // number of chosen particles
  long *chosen_particles_table; // stores the pdg index of the chosen particle (to access chosen particle properties)
  
  
  // df coefficients
  Deltaf_Data *df_data;


  // particle spectra of chosen particle species
  long momentum_length, spectra_length;   
  double *dN_pTdpTdphidy;    

  // spacetime grid
  double tau_min, r_min, eta_min;
  double tau_max, r_max, eta_max;
  long tau_bins, r_bins, phi_bins, eta_bins;
  double tau_width, r_width, phi_width, eta_width;

  // spacetime distributions
  long spacetime_length, X_length;
  double *dN_dX;


public:
  EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in,
                        particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in, Deltaf_Data *df_data_in);
  ~EmissionFunctionArray();

  // void calculate_dN_ptdptdphidy(double *, double *, double *, double *,
  //   double *, double *, double *, double *, double *, double *, double *, double *, double *,
  //   double *, double *, double *, double *,
  //   double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  //   double *, double *, double *, double *, double *, double*, double*);

  // void calculate_dN_ptdptdphidy_VAH_PL(double *, double *, double *,
  // double *, double *, double *, double *, double *,
  // double *, double *, double *, double *, double *,
  // double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  // double *, double *, double *, double *, double *, double *, double *, double *, double *);

  void write_dN_2pipTdpTdy_toFile(long *MCID);
  void write_vn_toFile(long *MCID);
  void write_dN_dphidy_toFile(long *MCID);
  void write_dN_dy_toFile(long *MCID);

  void write_dN_taudtaudeta_toFile(long *MCID);
  void write_dN_2pirdrdeta_toFile(long *MCID);
  void write_dN_dphideta_toFile(long *MCID);
  
  void calculate_spectra();

};

#endif
