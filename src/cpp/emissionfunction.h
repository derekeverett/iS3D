#ifndef EMISSIONFUNCTION_H
#define EMISSIONFUNCTION_H

#include<string>
#include<vector>
#include "Table.h"
#include "main.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include "particle.h"

using namespace std;

typedef struct
{
  double x;   // pLRF.x
  double y;   // pLRF.y
  double z;   // pLRF.z
} lrf_momentum;

typedef struct
{
  // fit parameters of y = exp(constant + slope * mT)
  // purpose is to extrapolate the distribution dN_dymTdmTdphi at large mT
  // for the resonance decay integration routines
  double constant;
  double slope;
} MT_fit_parameters;


lrf_momentum Sample_momentum(double mass, double T, double alphaB);

class EmissionFunctionArray
{
private:
  ParameterReader* paraRdr;

  int OPERATION; // calculate smooth spectra or sample distributions
  int MODE; //vh or vah , ...
  int DF_MODE;  // delta-f type
  int DIMENSION; // hydro d+1 dimensions (2+1 or 3+1)
  int INCLUDE_BULK_DELTAF, INCLUDE_SHEAR_DELTAF, INCLUDE_BARYONDIFF_DELTAF;
  int REGULATE_DELTAF;
  int INCLUDE_BARYON;
  int GROUP_PARTICLES;
  double PARTICLE_DIFF_TOLERANCE;
  int LIGHTEST_PARTICLE; //mcid of lightest resonance to calculate in decay feed-down
  int DO_RESONANCE_DECAYS; // smooth resonance decays option

  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  int pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length;
  long FO_length;
  double *dN_pTdpTdphidy; //to hold smooth CF 3D spectra of all species

  std::vector<Sampled_Particle> particle_list; //to hold sampled particle list

  int *chosen_particles_01_table;       // has length Nparticle, 0 means miss, 1 means include
  int *chosen_particles_sampling_table; // store particle index; the sampling process follows the order specified by this table
  int Nparticles;
  int number_of_chosen_particles;
  particle_info* particles;       // contains all the particle info from pdg.dat
  FO_surf* surf_ptr;
  deltaf_coefficients df;
  bool particles_are_the_same(int, int);

public:
  EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in,
                        particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in, deltaf_coefficients df_in);
  ~EmissionFunctionArray();

  void sample_dN_pTdpTdphidy(double *, double *, double *, double *, int *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *,
    int, double *, double *, double *, double *, double *, double *);


  void sample_dN_pTdpTdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2);



   void calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
 double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, const int pbar_pts, double * pbar_root1, double * pbar_root2, double * pbar_weight1, double * pbar_weight2);


  void sample_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);

  void calculate_dN_pTdpTdphidy(double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double*, double*);

  void calculate_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);

  void write_dN_pTdpTdphidy_toFile(); // write invariant 3D spectra to file
  void write_dN_dpTdphidy_toFile();   // write 3D spectra to file in experimental bins
  void write_dN_pTdpTdphidy_with_resonance_decays_toFile(); // write invariant 3D spectra to file (w/ resonance decay effects)
  void write_dN_dpTdphidy_with_resonance_decays_toFile();   // write 3D spectra to file in experimental bins (w/ resonance decay effects)
  void write_particle_list_toFile();  // write sampled particle list
  void calculate_spectra();

  // resonance decay functions:
  void do_resonance_decays(particle_info * particle_data);

  int particle_chosen_index(int particle_index);

  void resonance_decay_channel(particle_info * particle_data, int parent_index, int parent_chosen_index, int channel, vector<int> decays_index_vector);

  void two_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, double mass_1, double mass_2, double mass_parent);

  void three_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, int particle_3, double mass_1, double mass_2, double mass_3, double mass_parent);

  MT_fit_parameters estimate_MT_function_of_dNdypTdpTdphi(int iy, int iphip, int parent_chosen_index, double mass_parent);

  double dN_dYMTdMTdPhi_boost_invariant(int parent_chosen_index, double * MTValues, double * PhipValues, double MT, double Phip1, double Phip2, double Phip_min, double Phip_max, double MTmax, MT_fit_parameters ** MT_params);

  double dN_dYMTdMTdPhi_non_boost_invariant(int parent_chosen_index, double * MTValues, double * PhipValues, double * YValues, double MT, double Phip1, double Phip2, double Y, double MTmax, MT_fit_parameters ** MT_params);

};

#endif
