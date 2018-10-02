#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <chrono>
#include <limits>
#include <array>
#ifdef _OMP
#include <omp.h>
#endif
#include "main.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "Stopwatch.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include "gaussThermal.h"
#include "particle.h"
#include "viscous_correction.h"

#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

using namespace std;

double equilibrium_particle_density(double mass, double degeneracy, double sign, double T, double chem, double mbar, int jmax, double two_pi2_hbarC3)
{
  // may want to use a direct gauss quadrature instead

  double neq = 0.0;
  double sign_factor = -sign;

  for(int j = 1; j < jmax; j++)            // sum truncated expansion of Bose-Fermi distribution
  {
    double k = (double)j;
    sign_factor *= (-sign);
    neq += sign_factor * exp(k * chem) * gsl_sf_bessel_Kn(2, k * mbar) / k;
  }
  neq *= degeneracy * mass * mass * T / two_pi2_hbarC3;

  return neq;
}

double compute_deltaf_weight(lrf_momentum pLRF, double mass, double sign, double baryon, double T, double alphaB, Shear_Stress_Tensor pimunu, double bulkPi, Baryon_Diffusion_Current Vmu, double shear_coeff, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DF_MODE)
{
  // bosons have sign = -1, fermions have sign = 1

  // pLRF components
  double E = pLRF.E;
  double px = pLRF.x;
  double py = pLRF.y;
  double pz = pLRF.z;

  // could pass df_coeff vector possibly...

  // pimunu LRF components
  double pixx = pimunu.pixx_LRF;
  double pixy = pimunu.pixy_LRF;
  double pixz = pimunu.pixz_LRF;
  double piyy = pimunu.piyy_LRF;
  double piyz = pimunu.piyz_LRF;
  double pizz = pimunu.pizz_LRF;
  double pi_magnitude = pimunu.pi_magnitude;

  double feqbar = 1.0 - sign / (exp(E / T) + sign);   // 1 - sign . feq
  double sign_factor = (3.0 - sign) / 2.0;            // 1 for fermions, 2 for bosons

  //if(feq > 1.0) {printf("Error: feq > 1\n"); exit(-1);}
  //if(feqbar > 2.0) {printf("Error: feqbar > 2\n"); exit(-1);}

  // default values for the viscous weights
  double shear_weight = 1.0;
  double bulk_weight = 1.0;
  double diffusion_weight = 1.0;

  if(INCLUDE_SHEAR_DELTAF)
  {
    double A = shear_coeff;
    double pimunu_pmu_pnu = px * px * pixx  + py * py * piyy  +  pz * pz * pizz  +  2.0 * (px * py * pixy  +  px * pz * pixz  +  py * pz * piyz);

    double df_shear = feqbar * pimunu_pmu_pnu / A;
    double df_shear_max = sign_factor * E * E * pi_magnitude / A;

    // default
    shear_weight = (1.0 + df_shear) / (1.0 + df_shear_max);

    // regulate
    if(fabs(df_shear) > 1.0)
    {
      shear_weight = (1.0 + max(-1.0, min(df_shear, 1.0))) / (1.0 + min(df_shear_max, 1.0));
    }
  }
  /*
  if(INCLUDE_BULK_DELTAF)
  {
    double H = 1.0;
    double f0 = 1.0 / ( exp(E / T) + sign );
    double num = 1.0 + ( 1.0 + sign * f0 ) * H * bulkPi;
    double Hmax = 1.0;
    double den = 1.0 + ( 1.0 + sign * f0 ) * Hmax * bulkPi;
    //bulk_weight = num / den;
    bulk_weight = 1.0;
  }
  if(INCLUDE_BARYONDIFF_DELTAF)
  {
    diffusion_weight = 1.0;
  }
  */

  // enforced regulation?
  //return fabs(shear_weight * bulk_weight * diffusion_weight);
  return shear_weight;
}


lrf_momentum Sample_Momentum_feq_plus_deltaf(double mass, double sign, double baryon, double T, double alphaB, dsigma_Vector ds, Shear_Stress_Tensor pimunu, double bulkPi, Baryon_Diffusion_Current Vmu, double shear_coeff, int INCLUDE_SHEAR_DELTAF, int INCLUDE_BULK_DELTAF, int INCLUDE_BARYONDIFF_DELTAF, int DF_MODE)
{
  // sampled LRF momentum
  lrf_momentum pLRF;

  // for acceptance / rejection loop
  bool rejected = true;

  // random generator
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);

  // uniform distributions in (phi, costheta) on standby:
  uniform_real_distribution<double> phi_uniform_distribution(0.0, 2.0 * M_PI);
  uniform_real_distribution<double> costheta_uniform_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  double mass_squared = mass * mass;

  // dsigma LRF components
  double dst = ds.dsigmat_LRF;
  double dsx = ds.dsigmax_LRF;
  double dsy = ds.dsigmay_LRF;
  double dsz = ds.dsigmaz_LRF;
  double ds_magnitude = ds.dsigma_magnitude;

  // set df coefficients (viscous corrections)
  double c0 = 0.0;
  double c1 = 0.0;
  double c2 = 0.0;
  double c3 = 0.0;
  double c4 = 0.0;
  double F = 0.0;
  double G = 0.0;
  double betabulk = 0.0;
  double betaV = 0.0;
  double betapi = 0.0;

  /*
  switch(DF_MODE)
  {
    case 1: // 14-moment
    {
      // bulk coefficients
      c0 = df_coeff[0];
      c1 = df_coeff[1];
      c2 = df_coeff[2];
      // diffusion coefficients
      c3 = df_coeff[3];
      c4 = df_coeff[4];
      // shear coefficient evaluated later
      break;
    }
    case 2: // Chapman-Enskog
    {
      // bulk coefficients
      F = df_coeff[0];
      G = df_coeff[1];
      betabulk = df_coeff[2];
      // diffusion coefficient
      betaV = df_coeff[3];
      // shear coefficient
      betapi = df_coeff[4];
      break;
    }
    default:
    {
      cout << "Please choose df_mode = 1 or 2 in parameters.dat" << endl;
      exit(-1);
    }
  }
  */

  // pion sampling routine:
  // not adaptive rejection sampling, but naive method here...
  if(mass / T < 1.67)
  {
    double p;   // radial momentum
    double E;   // energy

    // sample p until accepted
    while(rejected)
    {
      // draw p from distribution p^2.exp(-p/T) by
      // sampling three variables (r1,r2,r3) uniformly from [0,1)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      //p = - T * log(r1 * r2 * r3);
      p = - T * (l1 + l2 + l3);
      E = sqrt(p * p + mass_squared);

      //here the pion weight should include the Bose enhancement factor
      //this formula assumes zero chemical potential mu = 0
      //TO DO - generalize for nonzero chemical potential
      double weight_light = exp(p/T) / (exp(E/T) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(!(weight_light >= 0.0 && weight_light <= 1.0)){printf("Error: weight = %f out of bounds\n", weight_light);exit(-1);}

      // check p acceptance
      if(propose < weight_light) break;

    } // p rejection loop

    // sample angles uniformly until accepted
    while(rejected)
    {
      double phi = phi_uniform_distribution(generator);
      double costheta = costheta_uniform_distribution(generator);

      double cosphi = cos(phi);
      double sinphi = sqrt(fabs(1.0 - cosphi * cosphi));
      double sintheta = sqrt(fabs(1.0 - costheta * costheta));

      // pLRF components
      pLRF.E = E;
      pLRF.x = p * sintheta * cosphi;
      pLRF.y = p * sintheta * sinphi;
      pLRF.z = p * costheta;

      double pdsigma_abs = fabs(E * dst  -  pLRF.x * dsx  -  pLRF.y * dsy  -  pLRF.z * dsz);
      double rideal = pdsigma_abs / (E * ds_magnitude);

      double rvisc = compute_deltaf_weight(pLRF, mass, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, shear_coeff, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DF_MODE);

      double weight_angle = rideal * rvisc;
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(!(rideal >= 0.0 && rideal <= 1.0)){printf("Error: rideal = %f out of bounds\n", rideal);exit(-1);}
      if(!(rvisc >= 0.0 && rvisc <= 1.0)){printf("Error: rvisc = %f out of bounds\n", rvisc);exit(-1);}

      // check angles' acceptance
      if(propose < weight_angle) break;

    } // angles rejection loop

  } // sampling light pions

  // heavy hadron sampling routine:
  // use variable transformation described in LongGang's notes
  else
  {
    // determine which part of integrand dominates
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T;
    double I3 = 2.0 * T * T;
    double Itot = I1 + I2 + I3;

    double p;   // radial momentum
    double E;   // energy
    double k;   // kinetic energy

    // sample k until accepted
    while(rejected)
    {
      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // randomly select distributions to sample k from based on integrated weights
      if(propose_distribution < I1 / Itot)
      {
        // draw k from distribution exp(-k/T) . dk by sampling r1 uniformly from [0,1):
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        k = - T * log(r1);

      } // distribution 1 (very heavy)
      else if(propose_distribution < (I1 + I2) / Itot)
      {
        // draw k from k.exp(-k/T).dk by sampling (r1,r2) uniformly from [0,1):
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k = - T * (l1 + l2);

      } // distribution 2 (moderately heavy)
      else
      {
        // draw k from k^2.exp(-k/T).dk by sampling (r1,r2,r3) uniformly from [0,1):
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k = - T * (l1 + l2 + l3);

      } // distribution 3 (light)

      E = k + mass;
      p = sqrt(fabs(E * E - mass_squared));

      double weight_heavy = (p/E) * exp(E/T) / (exp(E/T) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(!(weight_heavy >= 0.0 && weight_heavy <= 1.0)){printf("Error: weight = %f out of bounds\n", weight_heavy);exit(-1);}

      // check k acceptance
      if(propose < weight_heavy) break;

    } // k rejection loop

    // sample angles uniformly until accepted
    while(rejected)
    {
      double phi = phi_uniform_distribution(generator);
      double costheta = costheta_uniform_distribution(generator);

      double cosphi = cos(phi);
      double sinphi = sqrt(fabs(1.0 - cosphi * cosphi));
      double sintheta = sqrt(fabs(1.0 - costheta * costheta));

      // pLRF components
      pLRF.E = E;
      pLRF.x = p * sintheta * cosphi;
      pLRF.y = p * sintheta * sinphi;
      pLRF.z = p * costheta;

      double pdsigma_abs = fabs(E * dst  -  pLRF.x * dsx  -  pLRF.y * dsy  -  pLRF.z * dsz);
      double rideal = pdsigma_abs / (E * ds_magnitude);

      double rvisc = compute_deltaf_weight(pLRF, mass, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, shear_coeff, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DF_MODE);

      double weight_angle = rideal * rvisc;
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(!(rideal >= 0.0 && rideal <= 1.0)){printf("Error: rideal = %f out of bounds\n", rideal);exit(-1);}
      if(!(rvisc >= 0.0 && rvisc <= 1.0)){printf("Error: rvisc = %f out of bounds\n", rvisc);exit(-1);}

      // check angles acceptance
      if(propose < weight_angle) break;

    } // angles rejection loop

  } // sampling heavy hadrons

  return pLRF;

}


lrf_momentum Rescale_Momentum(lrf_momentum pLRF_mod, double mass_squared, double baryon, Shear_Stress_Tensor pimunu, Baryon_Diffusion_Current Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio)
{
    lrf_momentum pLRF;

    // rescale modified momentum p_mod with map M
    double E_mod = pLRF_mod.E;
    double px_mod = pLRF_mod.x;
    double py_mod = pLRF_mod.y;
    double pz_mod = pLRF_mod.z;

    // LRF momentum components (ideal by default or no viscous corrections)
    double px = px_mod;
    double py = py_mod;
    double pz = pz_mod;

    // LRF shear stress components
    double pixx = pimunu.pixx_LRF;
    double pixy = pimunu.pixy_LRF;
    double pixz = pimunu.pixz_LRF;
    double piyy = pimunu.piyy_LRF;
    double piyz = pimunu.piyz_LRF;
    double pizz = pimunu.pizz_LRF;

    // LRF baryon diffusion components
    double Vx = Vmu.Vx_LRF;
    double Vy = Vmu.Vy_LRF;
    double Vz = Vmu.Vz_LRF;
    double diffusion_factor = diff_coeff * (E_mod * baryon_enthalpy_ratio + baryon);

    // rescale momentum:
    // add shear corrections
    px += shear_coeff * (pixx * px_mod  +  pixy * py_mod  +  pixz * pz_mod);
    py += shear_coeff * (pixy * px_mod  +  piyy * py_mod  +  piyz * pz_mod);
    pz += shear_coeff * (pixz * px_mod  +  piyz * py_mod  +  pizz * pz_mod);

    // // add bulk corrections
    // px += bulk_coeff * px_mod;
    // py += bulk_coeff * py_mod;
    // pz += bulk_coeff * pz_mod;

    // // add diffusion corrections
    // px += diffusion_factor * Vx;
    // py += diffusion_factor * Vy;
    // pz += diffusion_factor * Vz;

    // set LRF momentum
    pLRF.x = px;
    pLRF.y = py;
    pLRF.z = pz;
    pLRF.E = sqrt(mass_squared  +  px * px  +  py * py  +  pz * pz);

    return pLRF;
}


lrf_momentum Sample_Momentum_feqmod(double mass, double sign, double baryon, double T_mod, double alphaB_mod, dsigma_Vector ds, Shear_Stress_Tensor pimunu, Baryon_Diffusion_Current Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio)
{
  double two_pi = 2.0 * M_PI;

  lrf_momentum pLRF;                          // sampled LRF momentum
  lrf_momentum pLRF_mod;                      // sampled modified LRF momentum

  bool rejected = true;                       // for acceptance / rejection loop

  // momentum generator
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);

  double mass_squared = mass * mass;          // mass squared

  double dst = ds.dsigmat_LRF;                // dsigma LRF components
  double dsx = ds.dsigmax_LRF;
  double dsy = ds.dsigmay_LRF;
  double dsz = ds.dsigmaz_LRF;
  double ds_magnitude = ds.dsigma_magnitude;

  // pion routine (not adaptive rejection sampling, but naive method here.. ~ massless Fermi distribution)
  if(mass / T_mod < 1.67)
  {
    while(rejected)
    {
      // draw (p_mod, phi_mod, costheta_mod) from distribution p_mod^2 * exp(-p_mod / T_mod) by
      // sampling three variables (r1,r2,r3) uniformly from [0,1)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p_mod = - T_mod * (l1 + l2 + l3);                  // modified 3-momentum magnitude
      double E_mod = sqrt(fabs(p_mod * p_mod + mass_squared));  // modified energy

      // calculate modified angles
      double phi_mod = two_pi * pow(l1 + l2, 2) / pow(l1 + l2 + l3, 2);
      double costheta_mod = (l1 - l2) / (l1 + l2);
      double sintheta_mod = sqrt(fabs(1.0 - costheta_mod * costheta_mod));

      // modified pLRF components
      pLRF_mod.E = E_mod;
      pLRF_mod.x = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.y = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.z = p_mod * costheta_mod;

      // momentum rescaling (* highlight *)
      pLRF = Rescale_Momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

      double pdsigma_abs = fabs(pLRF.E * dst - pLRF.x * dsx - pLRF.y * dsy - pLRF.z * dsz);
      double rideal = pdsigma_abs / (pLRF.E * ds_magnitude);


      //here the pion weight should include the Bose enhancement factor
      //this formula assumes zero chemical potential mu = 0
      //TO DO - generalize for nonzero chemical potential
      // I might want to pass sign since T_mod can vary ("light particle" not necessarily a pion)
      double weight_light = rideal * exp(p_mod / T_mod) / (exp(E_mod / T_mod) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      //if(!(rideal >= 0.0 && rideal <= 1.0)){printf("Error: rideal = %f out of bounds\n", rideal);exit(-1);}
      if(!(weight_light >= 0.0 && weight_light <= 1.0)){printf("Error: weight = %f out of bounds\n", weight_light);exit(-1);}

      // check pLRF acceptance
      if(propose < weight_light)
      {
        break;  // exit rejection loop
      }

    } // rejection loop
  } // sampling pions
  // heavy hadrons (use variable transformation described in LongGang's Sampler Notes)
  else
  {
    // determine which part of integrand dominates
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T_mod;
    double I3 = 2.0 * T_mod * T_mod;
    double Itot = I1 + I2 + I3;

    double I1_over_Itot = I1 / Itot;
    double I1_plus_I2_over_Itot = (I1 + I2) / Itot;

    // uniform distributions in (phi_mod, costheta_mod) on standby:
    uniform_real_distribution<double> phi_uniform_distribution(0.0, two_pi);
    uniform_real_distribution<double> costheta_uniform_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

    while(rejected)
    {
      // random variables to be sampled
      double k_mod, phi_mod, costheta_mod;

      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // randomly select distributions to sample momentum from based on integrated weights
      if(propose_distribution < I1_over_Itot)
      {
        // draw k_mod from distribution exp(-k_mod / T_mod) by sampling r1 uniformly from [0,1):
        //    k_mod = -T_mod * log(r1)
        // sample modified LRF angles costheta_mod = [-1,1] and phi_mod = [0,2pi) uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k_mod = - T_mod * log(r1);
        phi_mod = phi_uniform_distribution(generator);
        costheta_mod = costheta_uniform_distribution(generator);

      } // distribution 1 (very heavy)
      else if(propose_distribution < I1_plus_I2_over_Itot)
      {
        // draw (k_mod,phi_mod) from distribution k_mod * exp(-k_mod/T_mod) by sampling (r1,r2) uniformly from [0,1):
        //    k_mod = - T_mod * log(r1)
        //    phi_mod = 2 * pi * log(r1) / (log(r1) + log(r2));
        // sample LRF angle costheta_mod = [-1,1] uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k_mod = - T_mod * (l1 + l2);
        phi_mod = two_pi * l1 / (l1 + l2);
        costheta_mod = costheta_uniform_distribution(generator);

      } // distribution 2 (moderately heavy)
      else
      {
        // draw (k_mod,phi_mod,costheta_mod) from k_mod^2 * exp(-k_mod / T_mod) distribution by sampling (r1,r2,r3) uniformly from [0,1):
        //    k_mod = - T_mod * (log(r1) + log(r2) + log(r3))
        //    phi_mod = 2 * \pi * (log(r1) + log(r2))^2 / (log(r1) + log(r2) + log(r3))^2
        //    costheta_mod = (log(r1) - log(r2)) / (log(r1) + log(r2)
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k_mod = - T_mod * (l1 + l2 + l3);
        phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
        costheta_mod = (l1 - l2) / (l1 + l2);

      } // distribution 3 (light)

      double sintheta_mod = sqrt(fabs(1.0 - costheta_mod * costheta_mod));
      double E_mod = k_mod + mass;
      double p_mod = sqrt(fabs(E_mod * E_mod - mass_squared));

      pLRF_mod.E = E_mod;
      pLRF_mod.x = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.y = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.z = p_mod * costheta_mod;

      // momentum rescaling (* highlight *)
      pLRF = Rescale_Momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

      double pdsigma_abs = fabs(pLRF.E * dst - pLRF.x * dsx - pLRF.y * dsy - pLRF.z * dsz);
      double rideal = pdsigma_abs / (pLRF.E * ds_magnitude);

      double weight_heavy = rideal * (p_mod / E_mod) * exp(E_mod / T_mod) / (exp(E_mod / T_mod) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      //if(!(rideal >= 0.0 && rideal <= 1.0)){printf("Error: rideal = %f out of bounds\n", rideal);exit(-1);}
      if(!(weight_heavy >= 0.0 && weight_heavy <= 1.0)){printf("Error: weight = %f out of bounds\n", weight_heavy);exit(-1);}

      // check pLRF acceptance
      if(propose < weight_heavy)
      {
        break;  // exit rejection loop
      }

    } // rejection loop
  } // sampling heavy hadrons
  return pLRF;
}


void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2, double *pbar_root3, double *pbar_weight3)
  {
    printf("sampling particles from vhydro with df...\n");
    int npart = number_of_chosen_particles;
    double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);

    //set seed of random devices using clock time to produce nonidentical events
    unsigned seed_poisson = chrono::system_clock::now().time_since_epoch().count();
    unsigned seed_discrete = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen_poisson(seed_poisson);
    default_random_engine gen_discrete(seed_discrete);

    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts];    // eta_weight * delta_eta
    double delta_eta = fabs((eta_tab->get(1,2)) - (eta_tab->get(1,1)));  // assume uniform grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
    }

    // set df coefficients (viscous corrections)
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;
    double F = 0.0;
    double G = 0.0;
    double betabulk = 0.0;
    double betaV = 0.0;
    double betapi = 0.0;

    switch(DF_MODE)
    {
      case 1: // 14-moment
      {
        // bulk coefficients
        c0 = df_coeff[0];
        c1 = df_coeff[1];
        c2 = df_coeff[2];
        // diffusion coefficients
        c3 = df_coeff[3];
        c4 = df_coeff[4];
        // shear coefficient evaluated later
        break;
      }
      case 2: // Chapman-Enskog
      {
        // bulk coefficients
        F = df_coeff[0];
        G = df_coeff[1];
        betabulk = df_coeff[2];
        // diffusion coefficient
        betaV = df_coeff[3];
        // shear coefficient
        betapi = df_coeff[4];
        break;
      }
      default:
      {
        printf("Please choose df_mode = 1 or 2 in parameters.dat\n");
        exit(-1);
      }
    }

    //loop over all freezeout cells
    #pragma omp parallel for
    for (int icell = 0; icell < FO_length; icell++)   // may want to change int to long
    {
      double tau = tau_fo[icell];         // longitudinal proper time
      double x = x_fo[icell];             // x position
      double y = y_fo[icell];             // y position
      double tau2 = tau * tau;

      if(DIMENSION == 3)
      {
        etaValues[0] = eta_fo[icell];     // spacetime rapidity from surface file
      }

      double dat = dat_fo[icell];         // covariant normal surface vector dsigma_mu
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];         // dan should be 0 in 2+1d case

      double ux = ux_fo[icell];           // contravariant fluid velocity u^mu
      double uy = uy_fo[icell];           // enforce normalization
      double un = un_fo[icell];           // u^eta (fm^-1)
      double ut = sqrt(fabs(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un)); //u^tau

      double ut2 = ut * ut;               // useful expressions
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux2 + uy2);
      double utperp = sqrt(1.0  +  uperp * uperp);

      // set milne basis vectors
      Milne_Basis_Vectors basis_vectors(ut, ux, uy, un, uperp, utperp, tau);

      double T = T_fo[icell];             // temperature (GeV)
      double E = E_fo[icell];             // energy density (GeV/fm^3)
      double P = P_fo[icell];             // pressure (GeV/fm^3)

      double T3 = T * T * T;              // useful expressions
      double T4 = T3 * T;
      double T5 = T4 * T;

      double pitt = 0.0;                  // contravariant shear stress tensor pi^munu (GeV/fm^3)
      double pitx = 0.0;                  // enforce orthogonality and tracelessness
      double pity = 0.0;
      double pitn = 0.0;
      double pixx = 0.0;
      double pixy = 0.0;
      double pixn = 0.0;
      double piyy = 0.0;
      double piyn = 0.0;
      double pinn = 0.0;

      if(INCLUDE_SHEAR_DELTAF)
      {
        pixx = pixx_fo[icell];
        pixy = pixy_fo[icell];
        pixn = pixn_fo[icell];
        piyy = piyy_fo[icell];
        piyn = piyn_fo[icell];
        pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
      }

      double bulkPi = 0.0;

      if(INCLUDE_BULK_DELTAF)
      {
        bulkPi = bulkPi_fo[icell];        // bulk pressure (GeV/fm^3)
      }

      double muB = 0.0;                   // baryon chemical potential (GeV)
      double alphaB = 0.0;                // muB / T
      double nB = 0.0;                    // net baryon density (fm^-3)
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu (fm^-3)
      double Vx = 0.0;                    // enforce orthogonality
      double Vy = 0.0;
      double Vn = 0.0;
      double baryon_enthalpy_ratio = 0.0; // nB / (E + P)

      if(INCLUDE_BARYON)
      {
        muB = muB_fo[icell];
        alphaB = muB / T;

        if(INCLUDE_BARYONDIFF_DELTAF)
        {
          nB = nB_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
          baryon_enthalpy_ratio = nB / (E + P);
        }
      }


      // shear coefficient (14 moment; temporary)
      double shear_coeff = 2.0 * T * T * (E + P);           // change later if needed


      // dsigma class (delta_eta_weight factored out of 2+1d volume element since it drops out in rideal)
      dsigma_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_max();

      // shear stress class
      Shear_Stress_Tensor pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.boost_shear_stress_to_lrf(basis_vectors, tau2);
      pimunu.compute_pimunu_max();

      // baryon diffusion class
      Baryon_Diffusion_Current Vmu(Vt, Vx, Vy, Vn);
      Vmu.boost_baryon_diffusion_to_lrf(basis_vectors, tau2);

      // udotdsigma and Vdotdsigma w/o delta_eta_weight factor
      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;
      double Vdsigma = Vt * dat  +  Vx * dax  +  Vy * day  +  Vn * dan;


      std::vector<double> density_list;          // holds mean number / delta_eta_weight of each species in FO cell
      density_list.resize(npart);
      double dn_tot = 0.0;                       // total mean number  / delta_eta_weight of hadrons in FO cell

      // loop over hadron species
      for(int ipart = 0; ipart < npart; ipart++)
      {
        if(udsigma <= 0.0)
        {
          break;                                 // skip over cells with u.dsigma < 0
        }
        // particle's properties:
        double mass = Mass[ipart];               // mass in GeV
        double mbar = mass / T;                  // mass / temperature (equilibrium)
        double sign = Sign[ipart];               // quantum statistics sign
        double degeneracy = Degeneracy[ipart];   // spin degeneracy
        double baryon = Baryon[ipart];           // baryon number
        double chem = baryon * alphaB;           // baryon chemical potential term in feq

        // thermal particles:
        int jmax = 6;                            // truncation term of Bose-Fermi expansion = jmax - 1
        if(mbar < 2.0) jmax = 20;                // light particles need more terms than the leading order term

        double neq = equilibrium_particle_density(mass, degeneracy, sign, T, chem, mbar, jmax, two_pi2_hbarC3);
        double dn_thermal = udsigma * neq;       // mean number of thermal particles / delta_eta_weight of type ipart in FO cell


        // bulk and diffusion corrections to particle number in FO cell:
        double dn_bulk = 0.0;
        double dn_diff = 0.0;

        if(INCLUDE_BULK_DELTAF)
        {
          switch(DF_MODE)
          {
            case 1: // 14 moment (not sure what the status is)
            {
              double J10_fact = degeneracy * T3 / two_pi2_hbarC3;
              double J20_fact = degeneracy * T4 / two_pi2_hbarC3;
              double J30_fact = degeneracy * T5 / two_pi2_hbarC3;
              double J10 =  J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
              double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
              double J30 = J30_fact * GaussThermal(J30_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);

              dn_bulk = udsigma * bulkPi * ((c0 - c2) * mass * mass * J10 + (c1 * baryon * J20) + (4.0 * c2 - c0) * J30);
              break;
            }
            case 2: // Chapman-Enskog
            {
              double J10_fact = degeneracy * T3 / two_pi2_hbarC3;
              double J20_fact = degeneracy * T4 / two_pi2_hbarC3;
              double J10 = J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
              double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
              dn_bulk = udsigma * (bulkPi / betabulk) * (neq + (baryon * J10 * G) + (J20 * F / T / T));

              break;
            }
            default:
            {
              cout << "Please choose df_mode = 1 or 2 in parameters.dat" << endl;
              exit(-1);
            }
          } //switch(DF_MODE)
        } //if(INCLUDE_BULK_DELTAF)
        if(INCLUDE_BARYONDIFF_DELTAF)
        {
          switch(DF_MODE)
          {
            case 1: // 14 moment
            {
              // it's probably faster to use J31
              double J31_fact = degeneracy * T5 / two_pi2_hbarC3 / 3.0;
              double J31 = J31_fact * GaussThermal(J31_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);
              // these coefficients need to be loaded.
              // c3 ~ cV / V
              // c4 ~ 2cW / V
              dn_diff = - (Vdsigma / betaV) * ((baryon * c3 * neq * T) + (c4 * J31));
              break;
            }
            case 2: // Chapman Enskog
            {
              double J11_fact = degeneracy * T3 / two_pi2_hbarC3 / 3.0;
              double J11 = J11_fact * GaussThermal(J11_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
              dn_diff = - (Vdsigma / betaV) * (neq * T * baryon_enthalpy_ratio - baryon * J11);
              break;
            }
            default:
            {
              cout << "Please choose df_mode = 1 or 2 in parameters.dat" << endl;
              exit(-1);
            }
          } //switch(DF_MODE)
        } //if(INCLUDE_BARYONDIFF_DELTAF)

        double dn = dn_thermal + dn_bulk + dn_diff;    // mean number of particles of species ipart / delta_eta_weight
        density_list[ipart] = dn;                      // store in density list to sample individual species later
        dn_tot += dn;                                  // add to total mean number of hadrons / delta_eta_weight

      } // loop over hadrons (ipart)


      // loop over eta points
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        if(udsigma <= 0.0)
        {
          break;                                       // skip over cells with u.dsigma < 0
        }

        double delta_eta_weight = etaDeltaWeights[ieta];
        double dN_tot = delta_eta_weight * dn_tot;     // total mean number of hadrons in FO cell

        // SAMPLE PARTICLES:

        // probability distribution for number of hadrons
        std::poisson_distribution<> poisson_hadrons(dN_tot);

        int N_hadrons = poisson_hadrons(gen_poisson);               // sample total number of hadrons in FO cell

        for (int n = 0; n < N_hadrons; n++)
        {
          // sample particle type with discrete distribution (weights = dn_ipart / dn_tot)
          std::discrete_distribution<int> discrete_particle_type(density_list.begin(), density_list.end());

          int idx_sampled = discrete_particle_type(gen_discrete);   // chosen index of sampled particle type

          double mass = Mass[idx_sampled];                          // mass of sampled particle in GeV
          double sign = Sign[idx_sampled];                          // quantum statistics sign
          double baryon = Baryon[idx_sampled];                      // baryon number
          int mcid = MCID[idx_sampled];                             // MC ID of sampled particle



          // should make sample_momentum functions class members

          // sample LRF Momentum with Scott Pratt's Trick - See LongGang's Sampler Notes
          // perhap divide this into sample_momentum_from_distribution_x()
          lrf_momentum pLRF = Sample_Momentum_feq_plus_deltaf(mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, shear_coeff, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DF_MODE);



          // lab frame momentum p^mu (milne components)
          Lab_Momentum pmu(pLRF);
          pmu.boost_pLRF_to_lab_frame(basis_vectors, ut, ux, uy, un);


          // pdsigma has delta_eta_weight factored out (irrelevant for sign)
          double pdsigma = pmu.ptau * dat  +  pmu.px * dax  +  pmu.py * day  +  pmu.pn * dan;

          // add sampled particle to particle_list
          if (pdsigma >= 0.0)
          {
            double eta = etaValues[ieta];
            double sinheta = sinh(eta);
            double cosheta = sqrt(1.0 + sinheta * sinheta);

            // add sampled particle to particle_list
            Sampled_Particle new_particle;
            new_particle.mcID = mcid;
            new_particle.tau = tau;
            new_particle.x = x;
            new_particle.y = y;
            new_particle.eta = eta;

            new_particle.t = tau * cosheta;
            new_particle.z = tau * sinheta;

            new_particle.mass = mass;
            new_particle.px = pmu.px;
            new_particle.py = pmu.py;

            double pz = tau * pmu.pn * cosheta  +  pmu.ptau * sinheta;
            new_particle.pz = pz;
            //new_particle.E = pmu.ptau * cosheta  +  tau * pmu.pn * sinheta;
            //is it possible for this to be too small or negative?
            //try enforcing mass shell condition on E
            new_particle.E = sqrtf(mass * mass  +  pmu.px * pmu.px  +  pmu.py * pmu.py  +  pz * pz);

            //add to particle list
            //CAREFUL push_back is not a thread-safe operation
            //how should we modify for GPU version?
            #pragma omp critical
            particle_list.push_back(new_particle);
          } // if(pdsigma >= 0)
        } // for (int n = 0; n < N_hadrons; n++)
      } // for(int ieta = 0; ieta < eta_pts; ieta++)
  } // for (int icell = 0; icell < FO_length; icell++)
}


void EmissionFunctionArray::sample_dN_pTdpTdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
  int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2)
  {
    int npart = number_of_chosen_particles;
    double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);

    //set seed of random devices using clock time to produce nonidentical events
    unsigned seed1 = chrono::system_clock::now().time_since_epoch().count();
    unsigned seed2 = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen1(seed1);
    default_random_engine gen2(seed2);

    int eta_pts = 1;
    if(DIMENSION == 2)
    {
      eta_pts = eta_tab_length;
    }
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts];                               // eta_weight * delta_eta
    double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // placeholder for eta_fo below
      etaDeltaWeights[0] = 1.0; // default 1.0 for 3+1d
    }

    // set df coefficients:
    double F = df_coeff[0];
    double G = df_coeff[1];
    double betabulk = df_coeff[2];
    double betaV = df_coeff[3];
    double betapi = df_coeff[4];

    double shear_coeff = 0.5 / betapi;
    double three_betabulk = 3.0 * betabulk;
    double F_over_betabulk = F / betabulk;
    double G_over_betabulk = G / betabulk;

    // loop over all freezeout cells (icell)
    #pragma omp parallel for
    for(int icell = 0; icell < FO_length; icell++)
    {
      // FO cell coordinates:
      double tau = tau_fo[icell];         // longitudinal proper time
      double x = x_fo[icell];             // x coordinate
      double y = y_fo[icell];             // y coordinate
      double tau2 = tau * tau;

      if(DIMENSION == 3)
      {
        etaValues[0] = eta_fo[icell];     // spacetime rapidity from surface file
      }

      double dat = dat_fo[icell];         // covariant normal surface vector dsigma_mu
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];

      double ux = ux_fo[icell];           // contravariant fluid velocity u^mu
      double uy = uy_fo[icell];           // enforce normalization
      double un = un_fo[icell];           // u^eta
      double ut = sqrt(fabs(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un)); //u^tau

      double ut2 = ut * ut;               // useful expressions
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux2 + uy2);
      double utperp = sqrt(1.0  +  uperp * uperp);

      // set milne basis vectors
      Milne_Basis_Vectors basis_vectors(ut, ux, uy, un, uperp, utperp, tau);

      double T = T_fo[icell];             // temperature
      double E = E_fo[icell];             // energy density
      double P = P_fo[icell];             // pressure

      double T3 = T * T * T;              // useful expressions
      double T4 = T3 * T;

      double pitt = 0.0;                  // contravariant shear stress tensor pi^munu
      double pitx = 0.0;                  // enforce orthogonality and tracelessness
      double pity = 0.0;
      double pitn = 0.0;
      double pixx = 0.0;
      double pixy = 0.0;
      double pixn = 0.0;
      double piyy = 0.0;
      double piyn = 0.0;
      double pinn = 0.0;

      if(INCLUDE_SHEAR_DELTAF)
      {
        pixx = pixx_fo[icell];
        pixy = pixy_fo[icell];
        pixn = pixn_fo[icell];
        piyy = piyy_fo[icell];
        piyn = piyn_fo[icell];
        pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
      }

      double bulkPi = 0.0;

      if(INCLUDE_BULK_DELTAF)
      {
        bulkPi = bulkPi_fo[icell];        // bulk pressure
      }

      double muB = 0.0;                   // baryon chemical potential
      double alphaB = 0.0;                // muB / T
      double nB = 0.0;                    // net baryon density
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu
      double Vx = 0.0;
      double Vy = 0.0;
      double Vn = 0.0;
      double baryon_enthalpy_ratio = 0.0; // nB / (E + P)

      if(INCLUDE_BARYON)
      {
        muB = muB_fo[icell];
        alphaB = muB / T;

        if(INCLUDE_BARYONDIFF_DELTAF)
        {
          nB = nB_fo[icell];
          Vx = Vx_fo[icell];
          Vy = Vy_fo[icell];
          Vn = Vn_fo[icell];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
          baryon_enthalpy_ratio = nB / (E + P);
        }
      }

      // set modified coefficients, temperature, chemical potential
      double bulk_coeff = bulkPi / three_betabulk;
      double diff_coeff = T / betaV;

      double T_mod = T  +  bulkPi * F_over_betabulk;
      double alphaB_mod =  alphaB  +  bulkPi * G_over_betabulk;

      // dsigma class (delta_eta_weight factored out of 2+1d volume element since it drops out in rideal)
      dsigma_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_max();

      // shear stress class
      Shear_Stress_Tensor pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.boost_shear_stress_to_lrf(basis_vectors, tau2);
      pimunu.compute_pimunu_max();

      // baryon diffusion class
      Baryon_Diffusion_Current Vmu(Vt, Vx, Vy, Vn);
      Vmu.boost_baryon_diffusion_to_lrf(basis_vectors, tau2);

      // udotdsigma and Vdotdsigma w/o delta_eta_weight factor
      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;
      double Vdsigma = Vt * dat  +  Vx * dax  +  Vy * day  +  Vn * dan;


      std::vector<double> density_list;          // holds mean number / delta_eta_weight of each species in FO cell
      density_list.resize(npart);
      double dn_tot = 0.0;                       // total mean number  / delta_eta_weight of hadrons in FO cell

      // loop over particles
      for(int ipart = 0; ipart < npart; ipart++)
      {
        if(udsigma <= 0.0)
        {
          break;                                 // skip over cells with u.dsigma < 0
        }

        // particle's properties:
        double mass = Mass[ipart];               // mass in GeV
        double mbar = mass / T;                  // mass / temperature (equilibrium)
        double sign = Sign[ipart];               // quantum statistics sign
        double degeneracy = Degeneracy[ipart];   // spin degeneracy
        double baryon = Baryon[ipart];           // baryon number
        double chem_eq = baryon * alphaB;        // equilibrium baryon chemical potential term in feq

        // thermal particles:
        int jmax = 6;                            // truncation term of Bose-Fermi expansion = jmax - 1
        if(mbar < 2.0) jmax = 20;                // light particles need more terms than the leading order term

        double neq = equilibrium_particle_density(mass, degeneracy, sign, T, chem_eq, mbar, jmax, two_pi2_hbarC3);
        double dn_thermal = udsigma * neq;       // mean number of particles / delta_eta_weight of type ipart in FO cell


        // bulk and diffusion corrections:
        double dn_bulk = 0.0;                    // dn_bulk mod same as df b/c of renorm factor Z)
        double dn_diff = 0.0;                    // assumes mod (not baryon) diffusion current of each species ~ same as df one)

        if(INCLUDE_BULK_DELTAF)
        {
          double J10_fact = degeneracy * T3 / two_pi2_hbarC3;
          double J20_fact = degeneracy * T4 / two_pi2_hbarC3;
          double J10 = J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
          double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
          dn_bulk = udsigma * (bulkPi / betabulk) * (neq + (baryon * J10 * G) + (J20 * F / T / T));
        }
        if(INCLUDE_BARYONDIFF_DELTAF)
        {
          double J11_fact = degeneracy * T3 / two_pi2_hbarC3 / 3.0;
          double J11 = J11_fact * GaussThermal(J11_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
          dn_diff = - (Vdsigma / betaV) * (neq * T * baryon_enthalpy_ratio - baryon * J11);
        }


        double dn = dn_thermal + dn_bulk + dn_diff;    // mean number of particles of species ipart / delta_eta_weight
        density_list[ipart] = dn;                      // store in density list to sample individual species later
        dn_tot += dn;                                  // add to total mean number of hadrons / delta_eta_weight

      } // loop over particles (ipart)


      // loop over eta points
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        if(udsigma <= 0.0)
        {
          break;                                       // skip over cells with u.dsigma < 0
        }

        double delta_eta_weight = etaDeltaWeights[ieta];
        double dN_tot = delta_eta_weight * dn_tot;     // total mean number of hadrons in FO cell

        // SAMPLE PARTICLES:

        // probability distribution for number of hadrons
        std::poisson_distribution<> poisson_hadrons(dN_tot);

        int N_hadrons = poisson_hadrons(gen1);              // sample total number of hadrons in FO cell

        // sample momentum of N_hadrons
        for(int n = 0; n < N_hadrons; n++)
        {
          // sample particle type with discrete distribution (weights = dn_ipart / dn_tot)
          std::discrete_distribution<int> discrete_particle_type(density_list.begin(), density_list.end());

          int idx_sampled = discrete_particle_type(gen2);   // chosen index of sampled particle type

          double mass = Mass[idx_sampled];                  // mass of sampled particle in GeV
          double sign = Sign[idx_sampled];                  // quantum statistics sign
          double baryon = Baryon[idx_sampled];              // baryon number
          int mcid = MCID[idx_sampled];                     // MC ID of sampled particle
          //:::::::::::::::::::::::::::::::::::


          // sample particle's momentum with the modified equilibrium distribution
         //lrf_momentum pLRF = Sample_Momentum_feqmod(mass, sign, baryon, T_mod, alphaB_mod, dsigma, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

          double shear_coeff_14 = 2.0 * T * T * (E + P);
          lrf_momentum pLRF = Sample_Momentum_feq_plus_deltaf(mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, shear_coeff_14, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DF_MODE);



          // lab frame momentum p^mu (milne components)
          Lab_Momentum pmu(pLRF);
          pmu.boost_pLRF_to_lab_frame(basis_vectors, ut, ux, uy, un);


          // pdsigma has delta_eta_weight factored out (irrelevant for sign)
          double pdsigma = pmu.ptau * dat  +  pmu.px * dax  +  pmu.py * day  +  pmu.pn * dan;

          // only keep particles with p.dsigma >= 0
          if(pdsigma >= 0.0)
          {
            double eta = etaValues[ieta];
            double sinheta = sinh(eta);
            double cosheta = sqrt(1.0 + sinheta * sinheta);

            // add sampled particle to particle_list
            Sampled_Particle new_particle;
            new_particle.mcID = mcid;
            new_particle.tau = tau;
            new_particle.x = x;
            new_particle.y = y;
            new_particle.eta = eta;

            new_particle.t = tau * cosheta;
            new_particle.z = tau * sinheta;

            new_particle.mass = mass;
            new_particle.px = pmu.px;
            new_particle.py = pmu.py;

            double pz = tau * pmu.pn * cosheta  +  pmu.ptau * sinheta;
            new_particle.pz = pz;
            //new_particle.E = pmu.ptau * cosheta  +  tau * pmu.pn * sinheta;
            //is it possible for this to be too small or negative?
            //try enforcing mass shell condition on E
            new_particle.E = sqrtf(mass * mass  +  pmu.px * pmu.px  +  pmu.py * pmu.py  +  pz * pz);

            //add to particle list
            //CAREFUL push_back is not a thread safe operation
            //how should we modify for gpu version ?
            #pragma omp critical
            particle_list.push_back(new_particle);
          } // keep p.dsigma >= 0 particles
        } // sampled hadrons (n)
      } // eta points (ieta)
  } // freezeout cells (icell)
}

void EmissionFunctionArray::sample_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
  double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
  double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
{
  printf("Sampling particles from VAH \n");
  printf("NOTHING HERE YET \n");
}
