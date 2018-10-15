#include "viscous_correction.h"
#include <math.h>
#include <iostream>
#include <stdlib.h>
using namespace std;



Milne_Basis::Milne_Basis(double ut, double ux, double uy, double un, double uperp, double utperp, double tau)
{
      Xx = 1.0; Xy = 0.0;
      Yx = 0.0; Yy = 1.0;

      double sinhL = tau * un / utperp;
      double coshL = ut / utperp;

      Xt = uperp * coshL;
      Xn = uperp * sinhL / tau;
      Zt = sinhL;
      Zn = coshL / tau;

      if(uperp > 1.e-5) // stops (ux=0)/(uperp=0) nans for first FO cells with no initial transverse flow
      {
        Xx = utperp * ux / uperp;
        Xy = utperp * uy / uperp;
        Yx = - uy / uperp;
        Yy = ux / uperp;
      }
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
    dsigmax_LRF = -(dsigmat * Xt  +  dsigmax * Xx  +  dsigmay * Xy  +  dsigman * Xn);
    dsigmay_LRF = -(dsigmax * Yx  +  dsigmay * Yy);
    dsigmaz_LRF = -(dsigmat * Zt  +  dsigman * Zn);
}

void Surface_Element_Vector::compute_dsigma_max()
{
    dsigma_magnitude = fabs(dsigmat_LRF) + sqrt(dsigmax_LRF * dsigmax_LRF  +  dsigmay_LRF * dsigmay_LRF  +  dsigmaz_LRF * dsigmaz_LRF);
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

void Shear_Stress::boost_pimunu_to_lrf(Milne_Basis basis_vectors, double tau2)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
            + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));
    pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xy + pixy*Xx + piyy*Xy + tau2*piyn*Xn);
    pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xn - piyn*Xy - tau2*pinn*Xn);
    piyy_LRF = pixx*Yx*Yx + piyy*Yy*Yy + 2.0*pixy*Yx*Yy;
    piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);
    pizz_LRF = - (pixx_LRF + piyy_LRF);
}

void Shear_Stress::compute_pimunu_max()
{
    pi_magnitude = sqrt(pixx_LRF * pixx_LRF  +  piyy_LRF * piyy_LRF  +  pizz_LRF * pizz_LRF  +  2.0 * (pixy_LRF * pixy_LRF  +  pixz_LRF * pixz_LRF  +  piyz_LRF * piyz_LRF));
}

Baryon_Diffusion::Baryon_Diffusion(double Vt_in, double Vx_in, double Vy_in, double Vn_in)
{
    Vt = Vt_in;
    Vx = Vx_in;
    Vy = Vy_in;
    Vn = Vn_in;
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

void Baryon_Diffusion::compute_Vmu_max()
{
    V_magnitude = sqrt(Vx_LRF * Vx_LRF  +  Vy_LRF * Vy_LRF  +  Vz_LRF * Vz_LRF);
}
