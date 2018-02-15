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

  int INCLUDE_BULK_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BARYONDIFF_DELTAF;
  int INCLUDE_BARYON;
  int GROUP_PARTICLES;
  double PARTICLE_DIFF_TOLERANCE;

  Table *pT_tab, *phi_tab, *y_tab;
  int pT_tab_length, phi_tab_length, y_tab_length;
  long FO_length;
  double *dN_pTdpTdphidy; //to hold 3D spectra of all species
  int number_of_chosen_particles;
  particle_info* particles;
  FO_surf* surf_ptr;
  bool particles_are_the_same(int, int);

public:
  EmissionFunctionArray(ParameterReader* paraRdr_in, double particle_y_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, particle_info* particles_in, FO_surf* FOsurf_ptr_in, long FO_length_in);
  ~EmissionFunctionArray();

  void calculate_dN_ptdptdphidy(int); //again changed array handling
  void write_dN_ptdptdphidy_toFile(int); //write 3D spectra to file
  void calculate_spectra();

};

#endif
