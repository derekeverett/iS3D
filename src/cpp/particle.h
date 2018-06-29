#ifndef PARTICLE_H
#define PARTICLE_H
class particle
{
public:
  long int 	mcID;     /* Montecarlo number according PDG */
  char	name[26];
  double	mass;
  double	width;
  int	    gspin;      /* spin degeneracy */
  int	    baryon;
  int	    strange;
  int	    charm;
  int	    bottom;
  int	    gisospin;  /* isospin degeneracy */
  int	    charge;
  int	    decays;    /* amount of decays listed for this resonance */
  int	    stable;     /* defines whether this particle is considered as stable */
};
#endif

class particleDecay
{
public:
  int	reso;       /* Montecarlo number of decaying resonance */
  int	numpart;    /* number of daughter particles after decay */
  double branch;  /* branching ratio */
  int	part[5];    /* array of daughter particles Montecarlo numbers */
};
