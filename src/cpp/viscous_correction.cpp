#include "viscous_correction.h"
#include <math.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

Milne_Basis::Milne_Basis(double ut, double ux, double uy, double un, double uperp, double utperp, double tau)
{
    Ut = ut;    Uy = uy;
    Ux = ux;    Un = un;

    double sinhL = tau * un / utperp;
    double coshL = ut / utperp;

    Xt = uperp * coshL;         Zt = sinhL;
    Xn = uperp * sinhL / tau;   Zn = coshL / tau;

    Xx = 1.0;   Yx = 0.0;
    Xy = 0.0;   Yy = 1.0;

    if(uperp > 1.e-5) // stops (ux=0)/(uperp=0) nans for first FO cells with no initial transverse flow
    {
        Xx = utperp * ux / uperp;   Yx = - uy / uperp;
        Xy = utperp * uy / uperp;   Yy = ux / uperp;
    }
}

void Milne_Basis::test_orthonormality(double tau2)
{
    // test whether the basis vectors are orthonormal to numerical precision

    double U_normal = fabs(Ut * Ut  -  Ux * Ux  -  Uy * Uy  -  tau2 * Un * Un  -  1.0); // U.U - 1
    double X_normal = fabs(Xt * Xt  -  Xx * Xx  -  Xy * Xy  -  tau2 * Xn * Xn  +  1.0); // X.X + 1
    double Y_normal = fabs(- Yx * Yx  -  Yy * Yy  +  1.0);                              // Y.Y + 1
    double Z_normal = fabs(Zt * Zt  -  tau2 * Zn * Zn  +  1.0);                         // Z.Z + 1

    double U_dot_X = fabs(Xt * Ut  -  Xx * Ux  -  Xy * Uy  -  tau2 * Xn * Un);          // U.X
    double U_dot_Y = fabs(- Yx * Ux  -  Yy * Uy);                                       // U.Y
    double U_dot_Z = fabs(Zt * Ut  -  tau2 * Zn * Un);                                  // U.Z

    double X_dot_Y = fabs(- Xx * Yx  -  Xy * Yy);                                       // X.Y
    double X_dot_Z = fabs(Xt * Zt  -  tau2 * Xn * Zn);                                  // X.Z
    // Y.Z = 0 automatically (no components to multiply..)                              // Y.Z

    double U_orthogonal = max(U_dot_X, max(U_dot_Y, U_dot_Z));
    double X_orthogonal = max(X_dot_Y, X_dot_Z);

    double epsilon = 1.e-14;    // best order of magnitude I could get why?...

    if(U_normal > epsilon) printf("U is not normalized to 1 (%.6g)\n", U_normal);
    if(X_normal > epsilon) printf("X is not normalized to -1 (%.6g)\n", X_normal);
    if(Y_normal > epsilon) printf("Y is not normalized to -1 (%.6g)\n", Y_normal);
    if(Z_normal > epsilon) printf("Z is not normalized to -1 (%.6g)\n", Z_normal);
    if(U_orthogonal > epsilon) printf("U is not orthogonal (%.6g)\n", U_orthogonal);
    if(X_orthogonal > epsilon) printf("X is not orthogonal (%.6g)\n", X_orthogonal);
}

Surface_Element_Vector::Surface_Element_Vector(double dsigmat_in, double dsigmax_in, double dsigmay_in, double dsigman_in)
{
    dsigmat = dsigmat_in;
    dsigmax = dsigmax_in;
    dsigmay = dsigmay_in;
    dsigman = dsigman_in;
}

void Surface_Element_Vector::boost_dsigma_to_lrf(Milne_Basis basis_vectors, double ut, double ux, double uy, double un)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    dsigmat_LRF = dsigmat * ut  +  dsigmax * ux  +  dsigmay * uy  +  dsigman * un;
    dsigmax_LRF = - (dsigmat * Xt  +  dsigmax * Xx  +  dsigmay * Xy  +  dsigman * Xn);
    dsigmay_LRF = - (dsigmax * Yx  +  dsigmay * Yy);
    dsigmaz_LRF = - (dsigmat * Zt  +  dsigman * Zn);
}

void Surface_Element_Vector::compute_dsigma_magnitude()
{
    dsigma_space = sqrt(dsigmax_LRF * dsigmax_LRF  +  dsigmay_LRF * dsigmay_LRF  +  dsigmaz_LRF * dsigmaz_LRF);
    dsigma_magnitude = fabs(dsigmat_LRF) + dsigma_space;
}

void Surface_Element_Vector::compute_dsigma_lrf_polar_angle()
{
    costheta_LRF = dsigmaz_LRF / dsigma_space;
}


Shear_Stress::Shear_Stress(double pitt_in, double pitx_in, double pity_in, double pitn_in, double pixx_in, double pixy_in, double pixn_in, double piyy_in, double piyn_in, double pinn_in)
{
    pitt = pitt_in;
    pitx = pitx_in;
    pity = pity_in;
    pitn = pitn_in;
    pixx = pixx_in;
    pixy = pixy_in;
    pixn = pixn_in;
    piyy = piyy_in;
    piyn = piyn_in;
    pinn = pinn_in;
}

void Shear_Stress::test_pimunu_orthogonality_and_tracelessness(double ut, double ux, double uy, double un, double tau2)
{
    double tau2_un = tau2 * un;

    double pitmu_umu = fabs(pitt * ut  -  pitx * ux  -  pity * uy  -  pitn * tau2_un);    // pi^\tau\mu.u_mu
    double pixmu_umu = fabs(pitx * ut  -  pixx * ux  -  pixy * uy  -  pixn * tau2_un);    // pi^x\mu.u_mu
    double piymu_umu = fabs(pity * ut  -  pixy * ux  -  piyy * uy  -  piyn * tau2_un);    // pi^x\mu.u_mu
    double pinmu_umu = fabs(pitn * ut  -  pixn * ux  -  piyn * uy  -  pinn * tau2_un);    // pi^\eta\mu.u_mu
    double Tr_pimunu = fabs(pitt  -  pixx  -  piyy  -  tau2 * pinn);                      // Tr(pimunu)

    double pimunu_umu = max(pitmu_umu, max(pixmu_umu, max(piymu_umu, pinmu_umu)));        // max pi.u component

    double epsilon = 1.e-15;

    if(pimunu_umu > epsilon) printf("pimunu is not orthogonal (%.6g)\n", pimunu_umu);
    if(Tr_pimunu > epsilon) printf("pimunu is not traceless (%.6g)\n", Tr_pimunu);
}

void Shear_Stress::boost_pimunu_to_lrf(Milne_Basis basis_vectors, double tau2)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
            + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));

    //pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xy + pixy*Xx + piyy*Xy + tau2*piyn*Xn); // wrong formula
    //pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xn - piyn*Xy - tau2*pinn*Xn);  // wrong formula

    pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xt + pixy*Xx + piyy*Xy + tau2*piyn*Xn); // fixed on 10/16

    pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xx - piyn*Xy - tau2*pinn*Xn);  // fixed on 10/16

    piyy_LRF = pixx*Yx*Yx + 2.0*pixy*Yx*Yy + piyy*Yy*Yy;

    piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);

    pizz_LRF = - (pixx_LRF + piyy_LRF);
}

Baryon_Diffusion::Baryon_Diffusion(double Vt_in, double Vx_in, double Vy_in, double Vn_in)
{
    Vt = Vt_in;
    Vx = Vx_in;
    Vy = Vy_in;
    Vn = Vn_in;
}

void Baryon_Diffusion::test_Vmu_orthogonality(double ut, double ux, double uy, double un, double tau2)
{
    double Vmu_umu = fabs(Vt * ut  -  Vx * ux  -  Vy * uy  -  tau2 * Vn * un);    // V.u

    double epsilon = 1.e-15;

    if(Vmu_umu > epsilon) printf("Vmu is not orthogonal\n");
}

void Baryon_Diffusion::boost_Vmu_to_lrf(Milne_Basis basis_vectors, double tau2)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    Vx_LRF = - Vt * Xt  +  Vx * Xx  +  Vy * Xy  +  tau2 * Vn * Xn;
    Vy_LRF = Vx * Yx  +  Vy * Yy;
    Vz_LRF = - Vt * Zt  +  tau2 * Vn * Zn;
}
