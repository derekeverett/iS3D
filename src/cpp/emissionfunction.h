#ifndef EMISSIONFUNCTION_H
#define EMISSIONFUNCTION_H

#include<string>
#include<vector>
#include "Table.h"
#include "main.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include "particle.h"
#include "viscous_correction.h"

using namespace std;

typedef struct
{
  double x;   // pLRF.x
  double y;   // pLRF.y
  double z;   // pLRF.z
} lrf_momentum;

//sample momentum with linear viscous correction
lrf_momentum Sample_Momentum_deltaf(double mass, double T, double alphaB, Shear_Tensor pimunu, double bulkPi, double eps, double pressure, double tau2,
                            double sign, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DF_MODE);

//sample momentum with modified equil viscous correction
lrf_momentum Sample_Momentum_mod(double mass, double T, double alphaB);

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

  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  int pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length;
  long FO_length;
  double *dN_pTdpTdphidy; //to hold smooth CF 3D spectra of all species

  double *St, *Sx, *Sy, *Sn; //to hold the polarization vector of all species
  double *Snorm; //the normalization of the polarization vector of all species

  std::vector<Sampled_Particle> particle_list; //to hold sampled particle list

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

  void calculate_spin_polzn(double *Mass, double *Sign, double *Degeneracy,
    double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
    double *wtx_fo, double *wty_fo, double *wtn_fo, double *wxy_fo, double *wxn_fo, double *wyn_fo);

  void calculate_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);

  void write_dN_pTdpTdphidy_toFile(); //write invariant 3D spectra to file
  void write_polzn_vector_toFile(); //write components of spin polarization vector to file
  void write_dN_dpTdphidy_toFile(); //write 3D spectra to file in experimental bins
  void write_particle_list_toFile(); //write sampled particle list
  void calculate_spectra();

  void do_resonance_decays();

};

#endif
