#ifndef PARTICLE_H
#define PARTICLE_H


class particle
{
public:
  long int 	mcID;     //Montecarlo number according PDG
  char	name[26];
  double	mass;
  double	width;
  int	    gspin;      //spin degeneracy
  int	    baryon;
  int	    strange;
  int	    charm;
  int	    bottom;
  int	    gisospin;  // isospin degeneracy
  int	    charge;
  int	    decays;    // amount of decays listed for this resonance
  int	    stable;     // defines whether this particle is considered as stable
};

class particleDecay
{
public:
  int	reso;       // Montecarlo number of decaying resonance
  int	numpart;    // number of daughter particles after decay
  double branch;  // branching ratio
  int	part[5];    // array of daughter particles Montecarlo numbers
};

//a class to store basic particle ID, position and momenta for a sampled particle list
class Sampled_Particle
{
public:

  //PDG ID number
  int mcID = 0;
  //mass
  double mass = 0.0;
  //milne coordinate four vector of particle production point
  double tau = 0.0;
  double x = 0.0;
  double y = 0.0;
  double eta = 0.0;

  //cartesian coordinates
  double t = 0.0;
  double z = 0.0;

  //cartesian momentum four vector
  double E = 0.0;
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
};

#endif
