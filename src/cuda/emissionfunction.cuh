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

  // control parameters
  int MODE;      
  int DF_MODE;   
  int DIMENSION; 
  int INCLUDE_BARYON;
  int INCLUDE_SHEAR_DELTAF;
  int INCLUDE_BULK_DELTAF;
  int INCLUDE_BARYONDIFF_DELTAF;
  int OUTFLOW;
  int REGULATE_DELTAF;
  

  // freezeout surface
  FO_surf *surf_ptr;
  long FO_length;


  // momentum tables
  Table *pT_tab;
  Table *phi_tab;
  Table *y_tab;
  Table *eta_tab;

  int pT_tab_length;
  int phi_tab_length;
  int y_tab_length;
  int eta_tab_length;


  // particle info
  particle_info* particles;  
  int Nparticles;              // number of pdg particles
  int npart;                   // number of chosen particles
  int *chosen_particles_table; // stores the pdg index of the chosen particle (to access chosen particle properties)
  
  
  // df coefficients
  deltaf_coefficients df;


  // particle spectra of chosen particle species
  long spectra_length;
  double *dN_pTdpTdphidy;      


public:
  EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in,
                        particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in, deltaf_coefficients df_in);
  ~EmissionFunctionArray();

  void calculate_dN_ptdptdphidy(double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double*, double*);

  void calculate_dN_ptdptdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);

  void write_dN_pTdpTdphidy_toFile(); // write 3D spectra to file
  void calculate_spectra();

};

#endif
