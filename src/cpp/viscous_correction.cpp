#include "viscous_correction.h"
#include <math.h>


Milne_Vector_Basis::Milne_Vector_Basis(double ut, double ux, double uy, double un, double uperp, double utperp, double tau)
{
      Xx = 1.0; Xy = 0.0;
      Yx = 0.0; Yy = 1.0;

      double sinhL = tau * un / utperp;
      double coshL = ut / utperp;

      Xt = uperp * coshL;
      Xn = uperp * sinhL / tau;
      Zt = sinhL;
      Zn = coshL / tau;

      if(uperp > 1.e-5) // stops (ux=0)/(uperp=0) nans
      {
        Xx = utperp * ux / uperp;
        Xy = utperp * uy / uperp;
        Yx = - uy / uperp;
        Yy = ux / uperp;
      }
}



Baryon_Diffusion_Current::Baryon_Diffusion_Current()
{
    Vt = 0.0;
    Vx = 0.0;
    Vy = 0.0;
    Vn = 0.0;

    Vx_LRF = 0.0;
    Vy_LRF = 0.0;
    Vz_LRF = 0.0;
}

void Shear_Tensor::boost_to_lrf(double Xt, double Xx, double Xy, double Xn, double Yx, double Yy, double Zt, double Zn, double tau2)
{
  // pimunu in the LRF: piij = Xi.pi.Xj
  double pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
          + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));
  double pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xy + pixy*Xx + piyy*Xy + tau2*piyn*Xn);
  double pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xn - piyn*Xy - tau2*pinn*Xn);
  double piyy_LRF = pixx*Yx*Yx + piyy*Yy*Yy + 2.0*pixy*Yx*Yy;
  double piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);
  double pizz_LRF = - (pixx_LRF + piyy_LRF);

  //reset components to LRF components
  pixx = pixx_LRF;
  pixy = pixy_LRF;
  pixz = pixz_LRF;
  piyy = piyy_LRF;
  piyz = piyz_LRF;
  pizz = pizz_LRF;

}

//note this max is frame dependent, make sure pimunu has first been boosted to correct frame first
double Shear_Tensor::compute_max()
{
  // probably want to make LRF variables
  double max = 2.0 * ( fabs(pixx) + fabs(pixy) + fabs(pixz) + fabs(piyy) + fabs(piyz) + fabs(pizz) );
  return max;
}


void Baryon_Diffusion_Current::boost_to_lrf(Milne_Vector_Basis basis, double tau2)
{
    double Xt = basis.Xt;
    double Xx = basis.Xx;
    double Xy = basis.Xy;
    double Xn = basis.Xn;

    double Yx = basis.Yx;
    double Yy = basis.Yy;

    double Zt = basis.Zt;
    double Zn = basis.Zn;

    Vx_LRF = - Vt * Xt  +  Vx * Xx  +  Vy * Xy  +  tau2 * Vn * Xn;
    Vy_LRF = Vx * Yx + Vy * Yy;
    Vz_LRF = - Vt * Zt  +  tau2 * Vn * Zn;
}


