#ifndef EMISSIONFUNCTION_H
#define EMISSIONFUNCTION_H

#include<string>
#include<vector>
#include "Table.h"
#include "main.h"
#include "ParameterReader.h"
using namespace std;

class EmissionFunctionArray
{
private:
  ParameterReader* paraRdr;

  int INCLUDE_BULKDELTAF, INCLUDE_DELTAF;
  int INCLUDE_MUB;
  int bulk_deltaf_kind;
  int GROUPING_PARTICLES;
  double PARTICLE_DIFF_TOLERANCE;
  int F0_IS_NOT_SMALL;

  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  int pT_tab_length, phi_tab_length, y_tab_length;
  long FO_length;
  Table *dN_ptdptdphidy;
  double *dN_pTdpTdphidy; //to hold 3D spectra of one species
  int number_of_chosen_particles;
  int Nparticles;
  particle_info* particles;
  FO_surf* FOsurf_ptr;
  int last_particle_idx; // store the last particle index being used by calculate_dN_ptdptdphidy function
  bool particles_are_the_same(int, int);

  //array for bulk delta f coefficients
  Table *bulkdf_coeff;

public:
  EmissionFunctionArray(ParameterReader* paraRdr_in, double particle_y_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in);
  ~EmissionFunctionArray();

  void calculate_dN_ptdptdphidy(int); //again changed array handling
  void write_dN_ptdptdphidy_toFile(int); //write 3D spectra to file

  void calculate_dN_ptdptdphidy_4all(int to_order=9);
  void getbulkvisCoefficients(double Tdec, double* bulkvisCoefficients);

};

#endif
