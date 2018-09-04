#ifndef VISCOUS_CORRECTION_H
#define VISCOUS_CORRECTION_H

class Shear_Tensor
{
public:
  //contravariant components pi ^ \mu \nu milne coordinates
  double pitt;
  double pitx;
  double pity;
  double pitn;
  double pixx;
  double pixy;
  double pixn;
  double piyy;
  double piyn;
  double pinn;

  //spatial cartesian components
  double pixz;
  double piyz;
  double pizz;

  //boosts components of shear stress from lab frame to LRF
  void boost_to_lrf(double Xt, double Xx, double Xy, double Xn, double Yx, double Yy, double Zt, double Zn, double tau2);

  //calculates the upper bound sum_{\mu,\nu} | pi^\mu\nu |
  double compute_max();
};

#endif
