//ver 1.1
#ifndef READINDATA_H
#define READINDATA_H

#include "main.cuh"
#include "ParameterReader.cuh"
#include<fstream>

using namespace std;

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

class FO_data_reader
{
    private:
        ParameterReader* paraRdr;
        string pathToInput;
        int mode; //type of freezeout surface, VH or VAH
        int include_baryon; //switch to turn on/off baryon chemical potential
        int include_bulk_deltaf; //switch to turn on/off (\delta)f correction from bulk viscosity
        int include_shear_deltaf; //switch to turn on/off (\delta)f correction from shear viscosity
        int include_baryondiff_deltaf; //switch to turn on/off (\delta)f correction from baryon diffusion

    public:
        FO_data_reader(ParameterReader* paraRdr_in, string pathToInput);
        ~FO_data_reader();

        int get_number_cells();
        void read_surf_switch(long length, FO_surf* surf_ptr);
        void read_surf_VH(long length, FO_surf* surf_ptr);
        void read_surf_VAH_PLMatch(long length, FO_surf* surf_ptr);
        void read_surf_VAH_PLPTMatch(long length, FO_surf* surf_ptr);
        void read_surf_VH_MUSIC(long length, FO_surf* surf_ptr);
        int read_resonances_list(particle_info* particle);
};

#endif
