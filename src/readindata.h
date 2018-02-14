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
   double tau, x, y, eta;
   double dat, dax, day, dan;
   double ut, ux, uy, un;
   double E, T, P;
   double Bn, muB, muS; //what is Bn? we won't need muS ...
   double pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn;
   double bulkPi;
   double particle_mu[Maxparticle]; //WHAT IS THIS?
} FO_surf;

class read_FOdata
{
    private:
        ParameterReader* paraRdr;
        string pathToInput;
        int mode;
        int turn_on_bulk;
        int turn_on_muB;

    public:
        read_FOdata(ParameterReader* paraRdr_in, string pathToInput);
        ~read_FOdata();

        int get_number_of_freezeout_cells();
        void read_in_freeze_out_data(int length, FO_surf* surf_ptr);
        int read_in_chemical_potentials(string pathToInput, int FO_length, FO_surf* surf_ptr, particle_info* particle_ptr); //what chemical potentials are these?
        void read_decdat(int length, FO_surf* surf_ptr);
        void read_surfdat(int length, FO_surf* surf_ptr);
        void read_FOsurfdat(int length, FO_surf* surf_ptr);
        int read_resonances_list(particle_info* particle);
        void calculate_particle_mu(int Nparticle, FO_surf* FOsurf_ptr, int FO_length, particle_info* particle, double** particle_mu);
};

#endif
