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

    // constructor 
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

class Shear_Stress_Tensor // (Mike's version)
{
  private:            // pi^munu contravariant milne components:
    double pitt;      // pi^tautau
    double pitx;      // pi^taux
    double pity;      // pi^tauy
    double pitn;      // pi^taueta
    double pixx;      // pi^xx
    double pixy;      // pi^xy
    double pixn;      // pi^xeta
    double piyy;      // pi^yy
    double piyn;      // pi^yeta
    double pinn;      // pi^etaeta

  public:             // LRF components:
    double pixx_LRF;  // X.pi.X
    double pixy_LRF;  // X.pi.Y
    double pixz_LRF;  // X.pi.Z
    double piyy_LRF;  // Y.pi.Y
    double piyz_LRF;  // Y.pi.Z
    double pizz_LRF;  // Z.pi.Z

    // constructor
    Shear_Stress_Tensor(double pitt_in, double pitx_in, double pity_in, double pitn_in, double pixx_in, double pixy_in, double pixn_in, double piyy_in, double piyn_in, double pinn_in);

    // computes the LRF components pi_ij
    void boost_shear_stress_to_lrf(Milne_Basis_Vectors basis_vectors, double tau2);

    //calculates the upper bound sum_{\mu,\nu} | pi^\mu\nu |
    //double compute_max();
};


class Baryon_Diffusion_Current
{
  private:            // V^mu contravariant milne components:
    double Vt;        // V^tau
    double Vx;        // V^x
    double Vy;        // V^y
    double Vn;        // V^eta

  public:             // LRF components:
    double Vx_LRF;    // -X.V
    double Vy_LRF;    // -Y.V
    double Vz_LRF;    // -Z.V

    // constructor
    Baryon_Diffusion_Current(double Vt_in, double Vx_in, double Vy_in, double Vn_in);

    // computes the LRF components V_i
    void boost_baryon_diffusion_to_lrf(Milne_Basis_Vectors basis_vectors, double tau2);
};


class dsigma_Vector
{
  private:                    // dsigma_mu covariant milne components: (delta_eta_weight factored out)
    double dsigmat;           // dsigma_tau
    double dsigmax;           // dsigma_x
    double dsigmay;           // dsigma_y
    double dsigman;           // dsigma_eta

  public:                     // LRF components:  (no tau2 factor needed since dsigma is covariant)
    double dsigmat_LRF;       // u.dsigma  = u^mu . dsigma_mu
    double dsigmax_LRF;       // -X.dsigma = - X^mu . dsigma_mu
    double dsigmay_LRF;       // -Y.dsigma = - Y^mu . dsigma_mu
    double dsigmaz_LRF;       // -Z.dsigma = - Z^mu . dsigma_mu
    double dsigma_magnitude;  // |u.dsigma| + sqrt((u.dsigma)^2 - dsigma.dsigma)  (for rideal)

    // constructor
    dsigma_Vector(double dsigmat_in, double dsigmax_in, double dsigmay_in, double dsigman_in);

    // computes the LRF components
    void boost_dsigma_to_lrf(Milne_Basis_Vectors basis_vectors, double ut, double ux, double uy, double un);
    void compute_dsigma_max();

};

#endif
