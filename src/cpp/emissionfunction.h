#ifndef EMISSIONFUNCTION_H
#define EMISSIONFUNCTION_H

#include<string>
#include<vector>
#include<random>
#include "Table.h"
#include "iS3D.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include "particle.h"
#include "viscous_correction.h"

using namespace std;

// local rest frame momentum
typedef struct
{
  double E;    // u.p
  double px;   // -X.p
  double py;   // -Y.p
  double pz;   // -Z.p
} LRF_Momentum;

// lab frame momentum
class Lab_Momentum
{
  private:          // LRF momentum components
    double E_LRF;
    double px_LRF;
    double py_LRF;
    double pz_LRF;

  public:           // contravariant lab frame momentum p^mu (milne components)
    double ptau;    // p^tau
    double px;      // p^x
    double py;      // p^y
    double pn;      // p^eta

    Lab_Momentum(LRF_Momentum pLRF_in);
    void boost_pLRF_to_lab_frame(Milne_Basis basis_vectors, double ut, double ux, double uy, double un);
};

// three distributions to sample heavy hadrons
//typedef enum {heavy, moderate, light} heavy_distribution;

typedef struct
{
  // fit parameters of y = exp(constant + slope * mT)
  // purpose is to extrapolate the distribution dN_dymTdmTdphi at large mT
  // for the resonance decay integration routines
  double constant;
  double slope;
} MT_fit_parameters;


// thermal particle density (just for crosschecking)
//double equilibrium_particle_density(double mass, double degeneracy, double sign, double T, double chem);

double compute_detA(Shear_Stress pimunu, double betapi, double bulk_mod);

bool is_linear_pion0_density_negative(double T, double neq_pion0, double J20_pion0, double bulkPi, double F, double betabulk);
bool does_feqmod_breakdown(double mass_pion0, double T, double F, double bulkPi, double betabulk, double detA, double detA_min, double z, Gauss_Laguerre * laguerre, int df_mode, int fast, double Tavg, double F_avg, double betabulk_avg);

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
  int INCLUDE_BULK_DELTAF;
  int INCLUDE_SHEAR_DELTAF;
  int INCLUDE_BARYONDIFF_DELTAF;

  int REGULATE_DELTAF;
  int OUTFLOW;

  int INCLUDE_BARYON;
  double DETA_MIN;
  int GROUP_PARTICLES;
  double PARTICLE_DIFF_TOLERANCE;

  double MASS_PION0;

  int LIGHTEST_PARTICLE; //mcid of lightest resonance to calculate in decay feed-down
  int DO_RESONANCE_DECAYS; // smooth resonance decays option

  int OVERSAMPLE; // whether or not to iteratively oversample surface
  int FAST;                 // switch to compute mean hadron number quickly using an averaged (T,muB)
  long int MIN_NUM_HADRONS; //min number of particles summed over all samples
  long int SAMPLER_SEED; //the seed for the particle sampler. If chosen < 0, seed set with clocktime

  int TEST_SAMPLER;

  int Nevents;              // number of sampled events
  double mean_yield;        // mean number of particles emitted from freezeout surface (includes backflow)

  // for binning sampled particles (for sampler tests)
  double PT_LOWER_CUT;
  double PT_UPPER_CUT;
  int PT_BINS;
  double Y_CUT;
  int Y_BINS;
  double ETA_CUT;
  int ETA_BINS;

  double TAU_MIN;
  double TAU_MAX;
  int TAU_BINS;

  double R_MIN;
  double R_MAX;
  int R_BINS;

  // for sampler test (2+1d)
  double **dN_dy_count;         // holds event-averaged sampled dN/dy for each species
  double **dN_deta_count;       // holds event-averaged sampled dN/deta for each species
  double **dN_2pipTdpTdy_count; // holds event-averaged sampled (1/N) dN/dpT for each species

  double ***sampled_vn_real;  // holds event-averaged sampled Re(Vn) for each species
  double ***sampled_vn_imag;  // holds event-averaged sampled Im(Vn) for each species
  const int K_MAX = 7;        // {v_1, ..., v7}
  double **pT_count_vn;       // tracks count in each pT bin for each species (for vn calculation)

  double **sampled_dN_taudtaudy;
  double **sampled_dN_twopirdrdy;


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

  int *chosen_particles_01_table;       // has length Nparticle, 0 means miss, 1 means include
  int *chosen_particles_sampling_table; // store particle index; the sampling process follows the order specified by this table
  int Nparticles;
  int number_of_chosen_particles;
  particle_info* particles;       // contains all the particle info from pdg.dat
  FO_surf* surf_ptr;
  Deltaf_Data * df_data;
  bool particles_are_the_same(int, int);

public:

  // constructor / destructor
  EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particle, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles, FO_surf* FOsurf_ptr_in, long FO_length_in, Deltaf_Data * df_data_in);
  ~EmissionFunctionArray();

  // main function
  void calculate_spectra(std::vector<Sampled_Particle> &particle_event_list_in);


  // continuous spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // continuous spectra with feq + df
  void calculate_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data);

  // continuous spectra with feqmod
  void calculate_dN_ptdptdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data * df_data);

  void calculate_dN_dX(int *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data);

   void calculate_dN_dX_feqmod(int *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data * df_data);

  // continuous spectra with fa + dft
  void calculate_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
    double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *, double *, double *, double *);

  //:::::::::::::::::::::::::::::::::::::::::::::::::


  // sampling spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // calculate average total particle yield from freezeout surface to determine number of events to sample
  double calculate_total_yield(double * Equilibrium_Density, double * Bulk_Density, double * Diffusion_Density, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB, double *nB, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data * df_data, Gauss_Laguerre * laguerre);

  // sample particles with feq + df or feqmod
  void sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data, Gauss_Laguerre * laguerre, Gauss_Legendre * legendre);


  // add counts for sampled spectra / spacetime distributions (normalization in the write to file function)
  void sample_dN_2pipTdpTdy(Sampled_Particle new_particle, double yp);
  void sample_dN_dy(Sampled_Particle new_particle, double yp);
  void sample_dN_deta(Sampled_Particle new_particle, double eta);
  void sample_vn(Sampled_Particle new_particle, double yp);
  void sample_dN_dX(Sampled_Particle new_particle, double yp);



  void sample_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  double *, double *, double *, double *, double *, double *, double *, double *, double *);

  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // spin polarization:
  void calculate_spin_polzn(double *Mass, double *Sign, double *Degeneracy,
  double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *wtx_fo, double *wty_fo, double *wtn_fo, double *wxy_fo, double *wxn_fo, double *wyn_fo, Plasma * QGP);


  // write to file functions:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  void write_dN_pTdpTdphidy_toFile(int *MCID); // write invariant 3D spectra to file
  void write_dN_pTdpTdphidy_with_resonance_decays_toFile(); // write invariant 3D spectra to file (w/ resonance decay effects)
  void write_dN_dphidy_toFile(int *MCID);
  void write_dN_twopipTdpTdy_toFile(int *MCID);
  void write_dN_twopidpTdy_toFile(int *MCID);
  void write_dN_dy_toFile(int *MCID);
  void write_continuous_vn_toFile(int *MCID);
  void write_polzn_vector_toFile(); //write components of spin polarization vector to file

  void write_dN_dpTdphidy_toFile(int *MCID);   // write 3D spectra to file in experimental bins
  void write_dN_dpTdphidy_with_resonance_decays_toFile();   // write 3D spectra to file in experimental bins (w/ resonance decay effects)
  void write_particle_list_toFile();              // write sampled particle list
  void write_particle_list_OSC();                 // write sampled particle list in OSCAR format for UrQMD/SMASH
  void write_yield_list_toFile();                 // write mean yield and sampled yield list to files

  // for sampler test
  void write_sampled_dN_dy_to_file_test(int * MCID);
  void write_sampled_dN_deta_to_file_test(int * MCID);
  void write_sampled_dN_2pipTdpTdy_to_file_test(int * MCID);
  void write_sampled_vn_to_file_test(int * MCID);
  void write_sampled_dN_dX_to_file_test(int * MCID);



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
