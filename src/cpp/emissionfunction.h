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
  double t;   // dsigmaLRF.t
  double x;   // dsigmaLRF.x
  double y;   // dsigmaLRF.y
  double z;   // dsigmaLRF.z
} lrf_dsigma;

typedef struct
{
  double E;   // u.p
  double x;   // pLRF.x
  double y;   // pLRF.y
  double z;   // pLRF.z
} lrf_momentum;

class Lab_Momentum
{
  private:            // momentum LRF components:
    double E_LRF;     // u.p
    double px_LRF;    // -X.p
    double py_LRF;    // -Y.p
    double pz_LRF;    // -Z.p

  public:             // contravariant lab frame momentum p^mu (milne):
    double ptau;      // p^tau
    double px;        // p^x
    double py;        // p^y
    double pn;        // p^eta

    // constructor
    Lab_Momentum(lrf_momentum pLRF_in);
    // boost pLRF to the lab frame
    void boost_pLRF_to_lab_frame(Milne_Basis_Vectors basis_vectors, double ut, double ux, double uy, double un);

};

typedef struct
{
  // fit parameters of y = exp(constant + slope * mT)
  // purpose is to extrapolate the distribution dN_dymTdmTdphi at large mT
  // for the resonance decay integration routines
  double constant;
  double slope;
} MT_fit_parameters;

// thermal particle density (expanding BE/FD distributions)
double equilibrium_particle_density(double mass, double degeneracy, double sign, double T, double chem, double mbar, int jmax, double two_pi2_hbarC3);

double compute_deltaf_weight(lrf_momentum pLRF, double mass, double sign, double baryon, double T, double alphaB, Shear_Stress_Tensor pimunu, double bulkPi, Baryon_Diffusion_Current Vmu, double shear_coeff, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DF_MODE);

//sample momentum with linear viscous correction
lrf_momentum Sample_Momentum_feq_plus_deltaf(double mass, double sign, double baryon, double T, double alphaB, dsigma_Vector ds, Shear_Stress_Tensor pimunu, double bulkPi, Baryon_Diffusion_Current Vmu, double shear_coeff, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DF_MODE);

// momentum rescaling
lrf_momentum Rescale_Momentum(lrf_momentum pLRF_mod, double mass_squared, double baryon, Shear_Stress_Tensor pimunu, Baryon_Diffusion_Current Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio);

//sample momentum with modified equil viscous correction
lrf_momentum Sample_Momentum_feqmod(double mass, double sign, double baryon, double T_mod, double alphaB_mod, dsigma_Vector ds, Shear_Stress_Tensor pimunu, Baryon_Diffusion_Current Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio);


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
  int Nevents;              // number of sampled events

  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  int pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length;
  long FO_length;
  double *dN_pTdpTdphidy; //to hold smooth CF 3D spectra of all species
  double *logdN_PTdPTdPhidY; // hold log of smooth CF 3D spectra of parent (set in res decay for linear interpolation)

  double *St, *Sx, *Sy, *Sn; //to hold the polarization vector of all species
  double *Snorm; //the normalization of the polarization vector of all species

  std::vector<Sampled_Particle> particle_list; // to hold sampled particle list

  std::vector< std::vector<Sampled_Particle> > particle_event_list;   // holds sampled particle list of all events

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
  EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in, deltaf_coefficients df_in);
  ~EmissionFunctionArray();

  // highlight function
  void calculate_spectra();


  // continuous spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::
  void calculate_dN_pTdpTdphidy(double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double*, double*);

   void calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
 double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, const int pbar_pts, double * pbar_root1, double * pbar_root2, double * pbar_weight1, double * pbar_weight2);

  void calculate_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);
  //:::::::::::::::::::::::::::::::::::::::::::::::::


  // sampling spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::
  double estimate_total_yield(double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  int, double *, double *, double *, double *, double *, double *);

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
  void write_polzn_vector_toFile(); //write components of spin polarization vector to file


  void write_dN_dpTdphidy_toFile(int *MCID);   // write 3D spectra to file in experimental bins
  void write_dN_dpTdphidy_with_resonance_decays_toFile();   // write 3D spectra to file in experimental bins (w/ resonance decay effects)
  void write_particle_list_toFile(int sample);  // write sampled particle list
  void write_particle_list_OSC(int sample); //write sampled particle list in OSCAR format for UrQMD/SMASH
  void write_momentum_list_toFile(int sample);  // write sampled momentum list
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // resonance decay routine:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // highlight
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
