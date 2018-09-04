#include "viscous_correction.h"
#include <math.h>

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
  double max = 2.0 * ( fabs(pixx) + fabs(pixy) + fabs(pixz) + fabs(piyy) + fabs(piyz) + fabs(pizz) );
  return max;
}
