//ver 1.1
#ifndef READINDATA_H
#define READINDATA_H

#include "iS3D.h"
#include "ParameterReader.h"
#include <fstream>

using namespace std;

class Gauss_Laguerre
{
  private:
    int alpha;   // # generalized powers (0 : alpha - 1)

  public:
    int points;  // # quadrature points
    double ** root;
    double ** weight;

    Gauss_Laguerre();
    void load_roots_and_weights(string file_name);
};

class Gauss_Legendre
{
  private:

  public:
    int points;  // # quadrature points
    double * root;
    double * weight;

    Gauss_Legendre();
    void load_roots_and_weights(string file_name);
};

class Plasma
{
  private:
    // I just wanted a cool name
  public:                               // average thermo quantities over freezeout surface
    double temperature;                 // GeV
    double energy_density;              // GeV / fm^3
    double pressure;                    // GeV / fm^3
    double baryon_chemical_potential;   // GeV
    double net_baryon_density;          // fm^-3

    Plasma();
    void load_thermodynamic_averages();
};

typedef struct
{
  int mc_id; // Monte Carlo number according PDG
  string name;
  double mass;
  double width;
  int gspin; // spin degeneracy
  int baryon;
  int strange;
  int charm;
  int bottom;
  int gisospin; // isospin degeneracy
  int charge;
  int decays; // amount of decays listed for this resonance
  int stable; // defines whether this particle is considered as stable
  int decays_Npart[Maxdecaychannel];
  double decays_branchratio[Maxdecaychannel];
  int decays_part[Maxdecaychannel][Maxdecaypart];
  int sign; //Bose-Einstein or Dirac-Fermi statistics

  // ~ particle number / cell volume (for sampler routine)
  double equilibrium_density; // equilibrium density  (thermal number / u.dsigma)
  double bulk_density;        // bulk correction      (bulk number / u.dsigma / bulkPi)
  double diff_density;        // diffusion correction (diffusion number / V.dsigma)

} particle_info;

typedef struct
{
   double tau, x, y, eta; //contravariant spacetime position
   double dat, dax, day, dan; //COVARIANT surface normal vector
   double ut, ux, uy, un; //contravariant flow velocity
   double E, T, P; //energy density, Temperature and Isotropic Thermal Pressure
   double pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn; //contravariant components of shear stress tensor, or pi_perp^(\mu\nu) in case of VAH
   double bulkPi; //bulk pressure, or residual bulk pressure in case of VAH
   double muB, muS; //baryon chemical potential, strangeness chem. pot.
   double nB, Vt, Vx, Vy, Vn; //baryon number density, contravariant baryon diffusion current, or in case of VAH transverse baryon diffusion vector

   double wtx, wty, wtn, wxy, wxn, wyn; //the 6 components of the antisymmetric thermal vorticity with contravariant components w^\mu\nu

   //quantities exclusive to VAH
   double PL; //longitudinal pressure
   double PT; //transverse pressure
   double Wt, Wx, Wy, Wn; //contraviariant longitudinal momentum diffusion current W^\mu = W_perpz^\mu
   double Lambda; // effective Temperature
   double aL, aT; // longitudinal and transverse momentum anisotropy parameters
   double upsilonB; //effective baryon chemical potential
   double nBL; //LRF longitudinal baryon diffusion

   double c0,c1,c2,c3,c4; // for vah every FO has different delta-f coefficients

} FO_surf;

typedef struct
{
  //  Coefficients of 14 moment approximation (vhydro)
  // df ~ ((c0-c2)m^2 + b.c1(u.p) + (4c2-c0)(u.p)^2).Pi + (b.c3 + c4(u.p))p_u.V^u + c5.p_u.p_v.pi^uv
  double c0;
  double c1;
  double c2;
  double c3;
  double c4;
  double shear14_coeff;

  //  Coefficients of Chapman Enskog expansion (vhydro)
  // df ~ ((c0-c2)m^2 + b.c1(u.p) + (4c2-c0)(u.p)^2).Pi + (b.c3 + c4(u.p))p_u.V^u + c5.p_u.p_v.pi^uv
  double F;
  double G;
  double betabulk;
  double betaV;
  double betapi;

} deltaf_coefficients;



// JONAH'S MODIFIED BULK COEFFICIENT METHOD
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::
const int jonah_points = 301;               // # interpolation points

typedef struct
{
  const int points = jonah_points;

  double lambda[jonah_points];              // isotropic momentum scale
  double z[jonah_points];                   // renormalization factor (apart from detLambda)
  double bulkPi_over_Peq[jonah_points];     // bulk pressure output

  double bulkPi_over_Peq_max;               // the maximum bulk pressure in the array

  cubic_spline_coefficients cubic_spline[jonah_points-1];

} jonah_coefficients;


jonah_coefficients compute_jonah_bulk_coefficients(particle_info * particle_data, int Nparticle);
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::



class FO_data_reader
{
    private:
        ParameterReader* paraRdr;
        string pathToInput;
        int mode; //type of freezeout surface, VH or VAH
        int dimension; // added on 7/16
        int df_mode;
        int include_baryon; //switch to turn on/off baryon chemical potential
        int include_bulk_deltaf; //switch to turn on/off (\delta)f correction from bulk viscosity
        int include_shear_deltaf; //switch to turn on/off (\delta)f correction from shear viscosity
        int include_baryondiff_deltaf; //switch to turn on/off (\delta)f correction from baryon diffusion

    public:
        FO_data_reader(ParameterReader * paraRdr_in, string pathToInput);
        ~FO_data_reader();

        int get_number_cells();
        void read_surf_switch(long length, FO_surf * surf_ptr);
        void read_surf_VH(long length, FO_surf * surf_ptr);
        void read_surf_VH_Vorticity(long length, FO_surf * surf_ptr);
        void read_surf_VAH_PLMatch(long length, FO_surf * surf_ptr);
        void read_surf_VAH_PLPTMatch(long length, FO_surf * surf_ptr);
        void read_surf_VH_MUSIC(long length, FO_surf * surf_ptr);
        void read_surf_VH_MUSIC_New(long length, FO_surf* surf_ptr);
        int read_resonances_list(particle_info * particle, FO_surf * surf_ptr, deltaf_coefficients * df);
};

#endif
