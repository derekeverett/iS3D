//ver 1.1
#ifndef READINDATA_H
#define READINDATA_H

#include "main.h"
#include "ParameterReader.h"
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
   double tau, x, y, eta; //spacetime position
   double dat, dax, day, dan; //covariant surface normal vector
   double ut, ux, uy, un; //covariant flow velocity
   double E, T, P; //energy density, Temperature and Pressure
   double Bn, muB, muS; //what is Bn? Baryon Chemical potential and Strangeness chem. pot.
   double pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn; //covariant components of shear stress
   double bulkPi; //bulk pressure
} FO_surf;

class read_FOdata
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
        read_FOdata(ParameterReader* paraRdr_in, string pathToInput);
        ~read_FOdata();

        int get_number_cells();
        void read_surf_switch(long length, FO_surf* surf_ptr);
        void read_surf_VH(long length, FO_surf* surf_ptr);
        void read_surf_VAH(long length, FO_surf* surf_ptr);
        int read_resonances_list(particle_info* particle);
};

#endif
