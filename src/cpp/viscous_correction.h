#ifndef VISCOUS_CORRECTION_H
#define VISCOUS_CORRECTION_H

class Milne_Basis
{
  public:
    // milne basis vectors: U^mu, X^mu, Y^mu, Z^mu (nonzero components)
    double Ut, Ux, Uy, Un;
    double Xt, Xx, Xy, Xn;
    double Yx, Yy;
    double Zt, Zn;

    Milne_Basis(double ut, double ux, double uy, double un, double uperp, double utperp, double tau);
    void test_orthonormality(double tau2);
};

class Shear_Stress
{
  private:  // pi^munu contravariant milne components:
    double pitt, pitx, pity, pitn;
    double pixx, pixy, pixn;
    double piyy, piyn;
    double pinn;

  public:   // LRF components pi_ij = Xi.pi.Xj
    double pixx_LRF, pixy_LRF, pixz_LRF;
    double piyy_LRF, piyz_LRF;
    double pizz_LRF;
    double pi_magnitude;

    // diagonalized components of pi_ij
    double pixx_D;
    double piyy_D;
    double pizz_D;
    double delta_piperp;    // transverse asymmetry measure = [-1,1]
    double azi_plus;
    double azi_minus;

    Shear_Stress(double pitt_in, double pitx_in, double pity_in, double pitn_in, double pixx_in, double pixy_in, double pixn_in, double piyy_in, double piyn_in, double pinn_in);
    void boost_pimunu_to_lrf(Milne_Basis basis_vectors, double tau2);
    void diagonalize_pimunu_in_lrf();
    void compute_pi_magnitude(); 
    void test_pimunu_orthogonality_and_tracelessness(double ut, double ux, double uy, double un, double tau2);
};


class Baryon_Diffusion
{
  private:  // V^mu contravariant milne components:
    double Vt;
    double Vx, Vy, Vn;

  public:   // LRF components: V_i = - Xi.V
    double Vx_LRF, Vy_LRF, Vz_LRF;

    Baryon_Diffusion(double Vt_in, double Vx_in, double Vy_in, double Vn_in);
    void boost_Vmu_to_lrf(Milne_Basis basis_vectors, double tau2);
    void test_Vmu_orthogonality(double ut, double ux, double uy, double un, double tau2);
};


class Surface_Element_Vector
{
  private:  // dsigma_mu covariant milne components: (delta_eta_weight factored out)
    double dsigmat;
    double dsigmax, dsigmay, dsigman;

  public:   // dsigma LRF components:
    double dsigmat_LRF;       // u^mu . dsigma_mu
    double dsigmax_LRF;       // - X^mu . dsigma_mu
    double dsigmay_LRF;       // - Y^mu . dsigma_mu
    double dsigmaz_LRF;       // - Z^mu . dsigma_mu
    double dsigma_magnitude;  // |u.dsigma| + sqrt((u.dsigma)^2 - dsigma.dsigma)
    double dsigma_space;      // sqrt((u.dsigma)^2 - dsigma.dsigma)
    double costheta_LRF;      // cosine of polar angle of dsigma-3-vector in LRF

    Surface_Element_Vector(double dsigmat_in, double dsigmax_in, double dsigmay_in, double dsigman_in);
    void boost_dsigma_to_lrf(Milne_Basis basis_vectors, double ut, double ux, double uy, double un);
    void compute_dsigma_magnitude();
    void compute_dsigma_lrf_polar_angle();
};

#endif
