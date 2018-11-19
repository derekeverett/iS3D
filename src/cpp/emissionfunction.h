#ifndef EMISSIONFUNCTION_H
#define EMISSIONFUNCTION_H

#include<string>
#include<vector>
#include<random>
#include "Table.h"
#include "main.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include "particle.h"
#include "viscous_correction.h"

using namespace std;

typedef struct
{
  double E;    // u.p
  double px;   // -X.p
  double py;   // -Y.p
  double pz;   // -Z.p
} LRF_Momentum;

class Lab_Momentum
{
  private:          // LRF momentum components
    double E_LRF;
    double px_LRF;
    double py_LRF;
    double pz_LRF;

  public:           // contravariant lab frame momentum p^mu (milne)
    double ptau;    // p^tau
    double px;      // p^x
    double py;      // p^y
    double pn;      // p^eta

    Lab_Momentum(LRF_Momentum pLRF_in);
    void boost_pLRF_to_lab_frame(Milne_Basis basis_vectors, double ut, double ux, double uy, double un);
};

typedef struct
{
  // fit parameters of y = exp(constant + slope * mT)
  // purpose is to extrapolate the distribution dN_dymTdmTdphi at large mT
  // for the resonance decay integration routines
  double constant;
  double slope;
} MT_fit_parameters;

// thermal particle density (just for crosschecking)
double equilibrium_particle_density(double mass, double degeneracy, double sign, double T, double chem);

// thermal particle density with outflow only (needed if enforce p.dsigma > 0)
//double equilibrium_density_outflow(double mbar_squared, double sign, double chem, double dsigmaTime_over_dsigmaSpace, double * pbar_root1, double * pbar_exp_weight1, const int pbar_pts);

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
  double DETA_MIN;
  int GROUP_PARTICLES;
  double PARTICLE_DIFF_TOLERANCE;
  int LIGHTEST_PARTICLE; //mcid of lightest resonance to calculate in decay feed-down
  int DO_RESONANCE_DECAYS; // smooth resonance decays option

  int OVERSAMPLE; // whether or not to iteratively oversample surface
  long int MIN_NUM_HADRONS; //min number of particles summed over all samples
  long int SAMPLER_SEED; //the seed for the particle sampler. If chosen < 0, seed set with clocktime
  int Nevents;              // number of sampled events
  double mean_yield;        // mean number of particles emitted from freezeout surface (includes backflow)

  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  int pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length;
  long FO_length;
  double *dN_pTdpTdphidy; //to hold smooth CF 3D spectra of all species
  double *logdN_PTdPTdPhidY; // hold log of smooth CF 3D spectra of parent (set in res decay for linear interpolation)

  double *St, *Sx, *Sy, *Sn; //to hold the polarization vector of all species
  double *Snorm; //the normalization of the polarization vector of all species

  std::vector<Sampled_Particle> particle_list;                        // to hold sampled particle list (inactive)
  std::vector< std::vector<Sampled_Particle> > particle_event_list;   // holds sampled particle list of all events
  std::vector<int> particle_yield_list;                               // particle yield for all events (includes p.dsigma particles)

  vector<int> chosen_pion0;             // stores chosen particle index of pion0 (for tracking feqmod breakdown)

  int *chosen_particles_01_table;       // has length Nparticle, 0 means miss, 1 means include
  int *chosen_particles_sampling_table; // store particle index; the sampling process follows the order specified by this table
  int Nparticles;
  int number_of_chosen_particles;
  particle_info* particles;       // contains all the particle info from pdg.dat
  FO_surf* surf_ptr;
  deltaf_coefficients df;
  bool particles_are_the_same(int, int);

public:

  // constructor / destructor
  EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in, deltaf_coefficients df_in);
  ~EmissionFunctionArray();

  // main function
  void calculate_spectra();

  // continuous spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // continuous spectra with feq + df
  void calculate_dN_pTdpTdphidy(double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double*, double*, double *);

  // continuous spectra with feqmod
  void calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
    double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
    double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
    double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, const int pbar_pts, double * pbar_root1, double * pbar_root2, double * pbar_weight1, double * pbar_weight2, double *thermodynamic_average);

  // continuous spectra with fa + dft
  void calculate_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
    double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *);

  //:::::::::::::::::::::::::::::::::::::::::::::::::


  // sampling spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  double particle_density(double mass, double degeneracy, double sign, double baryon, double T, double alphaB, double bulkPi, double ds_time, double * df_coeff, double * pbar_root1, double * pbar_exp_weight1, const int pbar_pts);

  double particle_density_outflow(double mass, double degeneracy, double sign, double baryon, double T, double alphaB, double bulkPi, double ds_space, double ds_time_over_ds_space, double * df_coeff, double * pbar_root1, double * pbar_exp_weight1, const int pbar_pts);

  double particle_density_outflow_new(double mass, double degeneracy, double sign, double baryon, double T, double alphaB, double bulkPi, double ds_space, double ds_time_over_ds_space, double * df_coeff);

  // estimate total particle yield from freezeout surface -> number of events to sample
  double estimate_total_yield(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *tau_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *bulkPi_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *thermodynamic_average, double * df_coeff, const int pbar_pts, double * pbar_root1, double * pbar_exp_weight1);

  // sample momentum with feq + df
  LRF_Momentum sample_momentum(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear14_coeff, double baryon_enthalpy_ratio);

  // sample momentum with feqmod
  LRF_Momentum sample_momentum_feqmod(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T_mod, double alphaB_mod, Surface_Element_Vector dsigma, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio);

  // computes rvisc weight df correction
  double compute_df_weight(LRF_Momentum pLRF, double mass_squared, double sign, double baryon, double T, double alphaB, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear_coeff, double baryon_enthalpy_ratio);

  // momentum rescaling (used by feqmod momentum sampler)
  LRF_Momentum rescale_momentum(LRF_Momentum pmod, double mass_squared, double baryon, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio);


  // sample particles with feq + df or feqmod
  void sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, double *thermodynamic_average, const int pbar_pts, double * pbar_root1, double * pbar_exp_weight1);
  // sample particles with feqmod
  /*
  void sample_dN_pTdpTdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2);
  */

  void sample_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);

  //:::::::::::::::::::::::::::::::::::::::::::::::::


  // spin polarization:
  void calculate_spin_polzn(double *Mass, double *Sign, double *Degeneracy,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *wtx_fo, double *wty_fo, double *wtn_fo, double *wxy_fo, double *wxn_fo, double *wyn_fo);


  // write to file functions:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  void write_dN_pTdpTdphidy_toFile(int *MCID); // write invariant 3D spectra to file
  void write_dN_pTdpTdphidy_with_resonance_decays_toFile(); // write invariant 3D spectra to file (w/ resonance decay effects)
  void write_dN_dphidy_toFile(int *MCID);
  void write_dN_pTdpTdy_toFile(int *MCID);
  void write_dN_dy_toFile(int *MCID);
  void write_polzn_vector_toFile(); //write components of spin polarization vector to file

  void write_dN_dpTdphidy_toFile(int *MCID);   // write 3D spectra to file in experimental bins
  void write_dN_dpTdphidy_with_resonance_decays_toFile();   // write 3D spectra to file in experimental bins (w/ resonance decay effects)
  void write_particle_list_toFile();  // write sampled particle list
  void write_particle_list_OSC(); //write sampled particle list in OSCAR format for UrQMD/SMASH
  void write_momentum_list_toFile();  // write sampled momentum list
  void write_yield_list_toFile();     // write mean yield and sampled yield list to files

  //:::::::::::::::::::::::::::::::::::::::::::::::::


  // resonance decay routine:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // main function
  void do_resonance_decays(particle_info * particle_data);

  // switch statement for n-body routines
  void resonance_decay_channel(particle_info * particle_data, int parent_index, int parent_chosen_index, int channel, vector<int> decays_index_vector);

  // n-body integration routines
  void two_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, double mass_1, double mass_2, double mass_parent);

  void three_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, int particle_3, double mass_parent);

  // parent spectra linear interpolator
  double dN_dYMTdMTdPhi_boost_invariant(int parent_chosen_index, double * MTValues, double PhipValues[], double MT, double Phip1, double Phip2, double Phip_min, double Phip_max, double MTmax, MT_fit_parameters ** MT_params);

  double dN_dYMTdMTdPhi_non_boost_invariant(int parent_chosen_index, double * MTValues, double * PhipValues, int iYL, int iYR, double YL, double YR, double MT, double Phip1, double Phip2, double Y, double Phip_min, double Phip_max, double MTmax, MT_fit_parameters ** MT_params);

  // MT fit function
  MT_fit_parameters estimate_MT_function_of_dNdypTdpTdphi(int iy, int iphip, double mass_parent);

  // other
  int particle_chosen_index(int particle_index);

  //:::::::::::::::::::::::::::::::::::::::::::::::::

};

#endif
