#ifndef VISCOUS_CORRECTION_H
#define VISCOUS_CORRECTION_H


class Milne_Basis_Vectors
{
  public:
    // spatial basis vectors: X^mu, Y^mu, Z^mu (nonzero components)
    double Xt;
    double Xx;
    double Xy;
    double Xn;

    double Yx;
    double Yy;

    double Zt;
    double Zn;

    Milne_Basis_Vectors(double ut, double ux, double uy, double un, double uperp, double utperp, double tau);
};


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


class Baryon_Diffusion_Current
{
  public:
    double Vt;      // V^mu milne components
    double Vx;
    double Vy;
    double Vn;

    double Vx_LRF;  // LRF components
    double Vy_LRF;
    double Vz_LRF;

    Baryon_Diffusion_Current(double Vt_in, double Vx_in, double Vy_in, double Vn_in);
    void boost_to_lrf(double Xt, double Xx, double Xy, double Xn, double Yx, double Yy, double Zt, double Zn, double tau2);
};

#endif
