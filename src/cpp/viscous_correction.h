#ifndef VISCOUS_CORRECTION_H
#define VISCOUS_CORRECTION_H

class Milne_Basis_Vectors
{
  public:
    // spatial basis vectors: X^mu, Y^mu, Z^mu (nonzero components)
    double Xt, Xx, Xy, Xn;
    double Yx, Yy;
    double Zt, Zn;

    Milne_Basis_Vectors(double ut, double ux, double uy, double un, double uperp, double utperp, double tau);
};

class Shear_Stress_Tensor
{
  private:  // pi^munu contravariant milne components:
    double pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn;

  public:   // LRF components: pi_ij = Xi.pi.Xj
    double pixx_LRF, pixy_LRF, pixz_LRF, piyy_LRF, piyz_LRF, pizz_LRF;
    double pi_magnitude;  // sqrt(pi.pi)

    Shear_Stress_Tensor(double pitt_in, double pitx_in, double pity_in, double pitn_in, double pixx_in, double pixy_in, double pixn_in, double piyy_in, double piyn_in, double pinn_in);
    void boost_pimunu_to_lrf(Milne_Basis_Vectors basis_vectors, double tau2);
    void compute_pimunu_max();
};


class Baryon_Diffusion_Current
{
  private:  // V^mu contravariant milne components:
    double Vt, Vx, Vy, Vn;

  public:   // LRF components: V_i = - Xi.V
    double Vx_LRF, Vy_LRF, Vz_LRF;
    double V_magnitude; // sqrt(V.V)

    Baryon_Diffusion_Current(double Vt_in, double Vx_in, double Vy_in, double Vn_in);
    void boost_Vmu_to_lrf(Milne_Basis_Vectors basis_vectors, double tau2);
    void compute_Vmu_max();
};


class dsigma_Vector
{
  private:  // dsigma_mu covariant milne components: (delta_eta_weight factored out)
    double dsigmat, dsigmax, dsigmay, dsigman;

  public:   // dsigma LRF components:
    double dsigmat_LRF;       // u^mu . dsigma_mu
    double dsigmax_LRF;       // - X^mu . dsigma_mu
    double dsigmay_LRF;       // - Y^mu . dsigma_mu
    double dsigmaz_LRF;       // - Z^mu . dsigma_mu
    double dsigma_magnitude;  // |u.dsigma| + sqrt((u.dsigma)^2 - dsigma.dsigma)

    dsigma_Vector(double dsigmat_in, double dsigmax_in, double dsigmay_in, double dsigman_in);
    void boost_dsigma_to_lrf(Milne_Basis_Vectors basis_vectors, double ut, double ux, double uy, double un);
    void compute_dsigma_max();
};

#endif
