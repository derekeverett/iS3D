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


#endif


