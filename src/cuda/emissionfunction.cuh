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
  ParameterReader* paraRdr;

  int MODE; //vh or vah , ...
  int DF_MODE;  // delta-f type
  int DIMENSION; // hydro d+1 dimensions (2+1 or 3+1)
  int INCLUDE_BULK_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BARYONDIFF_DELTAF;
  int REGULATE_DELTAF;
  int INCLUDE_BARYON;
  int GROUP_PARTICLES;
  double PARTICLE_DIFF_TOLERANCE;

  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  int pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length;
  long FO_length;
  double *dN_pTdpTdphidy; //to hold 3D spectra of all species
  int *chosen_particles_01_table; // has length Nparticle, 0 means miss, 1 means include
  int *chosen_particles_sampling_table; // store particle index; the sampling process follows the order specified by this table
  int Nparticles;
  int number_of_chosen_particles;
  particle_info* particles;
  FO_surf* surf_ptr;
  deltaf_coefficients df;
  bool particles_are_the_same(int, int);

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

  void write_dN_pTdpTdphidy_toFile(); //write 3D spectra to file
  void calculate_spectra();

};

#endif
