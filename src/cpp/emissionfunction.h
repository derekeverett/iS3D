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
  double E;     // u.p
  double px;    // -X.p
  double py;    // -Y.p
  double pz;    // -Z.p
  double feq; // thermal distribution
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


class EmissionFunctionArray
{
private:   
  long CORES;                     // number of threads for openmp

  // mode parameters
  int OPERATION;                  // calculate smooth spectra or sample distributions
  int MODE;                       // hydro mode
  int DF_MODE;                    // df type
  int DIMENSION;                  // 2+1 or 3+1 dimensions 
  int INCLUDE_BARYON;             // include baryon chemical potential 
  int INCLUDE_SHEAR_DELTAF;       // shear df 
  int INCLUDE_BULK_DELTAF;        // bulk df
  int INCLUDE_BARYONDIFF_DELTAF;  // baryon diffusion df
  int REGULATE_DELTAF;            // regulate |df| <= feq
  int OUTFLOW;                    // enforce outflow p.dsigma > 0


  // momentum tables
  Table *pT_tab, *phi_tab, *y_tab, *eta_tab;
  long pT_tab_length, phi_tab_length, y_tab_length, eta_tab_length, y_minus_eta_tab_length;


  // freezeout surface
  long FO_length;
  FO_surf *surf_ptr;


  // particle info
  particle_info* particles;      
  long Nparticles;              // number of pdg particles
  long npart;                   // number of chosen particles
  long *chosen_particles_table; // stores the pdg index of the chosen particle (to access chosen particle properties)

 
  // df coefficients
  Deltaf_Data * df_data;


  // feqmod breakdown parameters
  double DETA_MIN;                // min value of detA
  double MASS_PION0;              // lightest pion mass in pdg table


  // resonance decay parameters (code not tested)
  //int LIGHTEST_PARTICLE;          // mcid of lightest resonance
  //int DO_RESONANCE_DECAYS;        // smooth resonance decays option


  // sampler parameters
  int OVERSAMPLE;                 // option to oversample surface
  int FAST;                       // switch to compute mean hadron number quickly using an averaged (T,muB)
  double MIN_NUM_HADRONS;         // min total number of particles sampled (determines estimate for Nevents)
  double MAX_NUM_SAMPLES;         // max number of events (Nevents <= max events)
  long int SAMPLER_SEED;          // set seed for particle sampler with fixed value (if >= 0) or clocktime (if < 0)
  int TEST_SAMPLER;               // test mode (bin particles during runtime instead making event lists)

  long Nevents;                                                     // number of events sampled
  std::vector<std::vector<Sampled_Particle>> particle_event_list;   // holds sampled particle list of all events


  // histogram parameters
  double Y_CUT, ETA_CUT;
  double PT_MIN, TAU_MIN, R_MIN;
  double PT_MAX, TAU_MAX, R_MAX;
  long PT_BINS, PHIP_BINS, Y_BINS, TAU_BINS, R_BINS, ETA_BINS;
  double PT_WIDTH, PHIP_WIDTH, Y_WIDTH, TAU_WIDTH, R_WIDTH, ETA_WIDTH;


  // event-averaged momentum distributions (so far only 2+1d)
  double **dN_dy_count, **dN_2pipTdpTdy_count, **dN_dphipdy_count;  

  // event-averaged vn
  double ***vn_real_count, ***vn_imag_count;    
  double **pT_count;             // count in each pT bin 
  const int K_MAX = 7;           // {v1, ..., v7}       

  // event-averaged spacetime distribution    
  double **dN_taudtaudy_count, **dN_twopirdrdy_count, **dN_dphisdy_count, **dN_deta_count; 


  // continuous distributnios
  long spectra_length, spacetime_length;
  double *dN_pTdpTdphidy;       // momentum spectra of all species
  double *dN_dX;                // spacetime distributions of all species
  //double *logdN_PTdPTdPhidY;    // momentum spectra of parent resonance


  // spin polarization
  double *St, *Sx, *Sy, *Sn;    // polarization vector of all species
  double *Snorm;                // polarization vector norm of all species

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

  void calculate_dN_dX(long *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data);

   void calculate_dN_dX_feqmod(long *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data * df_data);

  // continuous spectra with fa + dft
  // void calculate_dN_pTdpTdphidy_VAH_PL(double *, double *, double *,
  //   double *, double *, double *, double *, double *,
  //   double *, double *, double *, double *, double *,
  //   double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *,
  //   double *, double *, double *, double *, double *, double *, double *, double *, double *);

  //:::::::::::::::::::::::::::::::::::::::::::::::::


  // sampling spectra routines:
  //:::::::::::::::::::::::::::::::::::::::::::::::::

  // calculate average total particle yield from freezeout surface to determine number of events to sample
  double calculate_total_yield(double * Equilibrium_Density, double * Bulk_Density, double * Diffusion_Density, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB, double *nB, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data * df_data, Gauss_Laguerre * laguerre);

  // sample particles with feq + df or feqmod
  void sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, long *MCID, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data, Gauss_Laguerre * laguerre, Gauss_Legendre * legendre);


  // add counts for sampled distributions
  void sample_dN_dy(int chosen_index, double y);
  void sample_dN_deta(int chosen_index, double eta);
  void sample_dN_dphipdy(int chosen_index, double px, double py);
  void sample_dN_2pipTdpTdy(int chosen_index, double px, double py);
  void sample_vn(int chosen_index, double px, double py);
  void sample_dN_dX(int chosen_index, double tau, double x, double y);



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

  void write_dN_pTdpTdphidy_toFile(long *MCID); // write invariant 3D spectra to file
  void write_dN_dphidy_toFile(long *MCID);
  void write_dN_twopipTdpTdy_toFile(long *MCID);
  void write_dN_dy_toFile(long *MCID);
  void write_continuous_vn_toFile(long *MCID);
  void write_polzn_vector_toFile(); //write components of spin polarization vector to file

  void write_dN_taudtaudeta_toFile(long *MCID);
  void write_dN_2pirdrdeta_toFile(long *MCID);
  void write_dN_dphideta_toFile(long *MCID);

  void write_particle_list_toFile();              // write sampled particle list
  void write_particle_list_OSC();                 // write sampled particle list in OSCAR format for UrQMD/SMASH

  // for sampler test
  void write_sampled_dN_dy_to_file_test(long *MCID);
  void write_sampled_dN_deta_to_file_test(long *MCID);
  void write_sampled_dN_2pipTdpTdy_to_file_test(long *MCID);
  void write_sampled_dN_dphipdy_to_file_test(long *MCID);
  void write_sampled_vn_to_file_test(long *MCID);
  void write_sampled_dN_dX_to_file_test(long *MCID);



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
