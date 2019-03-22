#ifndef READINDATA_H
#define READINDATA_H

#include <fstream>

using namespace std;

const int Maxparticle = 500;     // max size of array for particle info storage (make sure it's big enough)
const int Maxdecaychannel = 20;  // max number of decays channels for given resonance (make sure it's big enough)
const int Maxdecaypart = 5;      // don't change this (it's the default number of decay particle entries in pdg.dat)

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


typedef struct
{
  long int mc_id; // Monte Carlo number according PDG
  string name;    // name
  double mass;    // mass (GeV)
  double width;   // resonance width (GeV)
  int gspin;      // spin degeneracy
  int baryon;     // baryon number
  int strange;    // strangeness
  int charm;      // charmness
  int bottom;     // bottomness
  int gisospin;   // isospin degeneracy
  int charge;     // electric charge
  int decays;     // amount of decays listed for this resonance
  int stable;     // defines whether this particle is considered as stable under strong interactions

  int decays_Npart[Maxdecaychannel];
  double decays_branchratio[Maxdecaychannel];
  int decays_part[Maxdecaychannel][Maxdecaypart];

  int sign;     // Bose-Einstein or Dirac-Fermi statistics

} particle_info;


int read_resonances_list(particle_info* particle);




const int max_digits = 10;  // max number of mcid digits (goes up to nuclei)

class read_mcid
{
  // determine the remaining particle properties based on mcid
  // (borrows some functionality from smash's pdgcode.hpp)

  // note: only have hadrons in mind (print errors if have leptons, etc)

  private:
    long int mcid;          // mcid of particle
  public:
    bool is_deuteron;       // label particle as deuteron (nothing yet for fake dibaryon d')
    bool is_hadron;         // label particle as hadron
    bool is_meson;          // label particle as meson
    bool is_baryon;         // label particle as baryon
    bool has_antiparticle;  // does particle have a distinct antiparticle?
    int baryon;             // baryon number
    int spin;               // spin x 2
    int gspin;              // spin degeneracy
    int sign;               // quantum statistics sign (BE, FD) = (-1, 1)

    uint32_t nJ  : 4;       // spin quantum number nJ = 2J + 1
    uint32_t nq3 : 4;       // third quark field
    uint32_t nq2 : 4;       // second quark field
    uint32_t nq1 : 4;       // first quark field, 0 for mesons
    uint32_t nL  : 4;       // "angular momentum"
    uint32_t nR  : 4;       // "radial excitation"
    uint32_t n   : 4, :3;   // first field: "counter"
    uint32_t n8;
    uint32_t n9;
    uint32_t n10;           // nuclei have 10-digits

    read_mcid(long int mcid_in);

    void is_particle_a_deuteron();// determine if particle is a deuteron
    void is_particle_a_hadron();  // determine if particle is a hadron
    void is_particle_a_meson();
    void is_particle_a_baryon();  // determine if the hadron is a baryon
    void get_baryon();            // get the baryon number
    void get_spin();              // get the spin x 2
    void get_gspin();             // get the spin degeneracy
    void get_sign();              // get the quantum statistics sign
    void does_particle_have_distinct_antiparticle();  // determine if there's a distinct antiparticle

};


int read_resonances_list_smash_box(particle_info * particle);


#endif
