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

double EmissionFunctionArray::compute_df_weight(lrf_momentum pLRF, double mass_squared, double sign, double baryon, double T, double alphaB, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear14_coeff, double baryon_enthalpy_ratio)
{
  // pLRF components
  double E = pLRF.E;
  double px = pLRF.px;
  double py = pLRF.py;
  double pz = pLRF.pz;

  double feqbar = 1.0 - sign / (exp(E/T) + sign);   // 1 - sign . feq (bosons, fermions have sign = -1, 1)
  double sign_factor = (3.0 - sign) / 2.0;          // 1 for fermions, 2 for bosons

  if(fabs(feqbar - 1.0) > 1.0) {printf("Error: feqbar too large\n");}

  // df corrections
  double df_shear = 0.0;
  double df_shear_max = 0.0;
  double df_bulk = 0.0;
  double df_bulk_max = 0.0;
  double df_diff = 0.0;
  double df_diff_max = 0.0;

  if(INCLUDE_SHEAR_DELTAF)
  {
    // pimunu LRF components
    double pixx = pimunu.pixx_LRF;
    double pixy = pimunu.pixy_LRF;
    double pixz = pimunu.pixz_LRF;
    double piyy = pimunu.piyy_LRF;
    double piyz = pimunu.piyz_LRF;
    double pizz = pimunu.pizz_LRF;
    double pi_magnitude = pimunu.pi_magnitude;

    double pimunu_pmu_pnu = px * px * pixx + py * py * piyy + pz * pz * pizz + 2.0 * (px * py * pixy + px * pz * pixz + py * pz * piyz);

    if(DF_MODE == 1)
    {
      df_shear = pimunu_pmu_pnu / shear14_coeff;
      df_shear_max = E * E * pi_magnitude / shear14_coeff;
    }
    else if(DF_MODE == 2)
    {
      double betapi = df_coeff[4];

      df_shear = pimunu_pmu_pnu / (2.0 * E * betapi * T);
      df_shear_max = E * pi_magnitude / (2.0 * fabs(betapi) * T);
    }
  }

  if(INCLUDE_BULK_DELTAF)
  {
    double bulkPi_magnitude = fabs(bulkPi); // this might not work for bulk since it's isotropic in angle

    if(DF_MODE == 1)
    {
      double c0 = df_coeff[0];
      double c1 = df_coeff[1];
      double c2 = df_coeff[2];

      df_bulk = ((c0 - c2) * mass_squared + (baryon * c1 + (4.0 * c2 - c0) * E) * E) * bulkPi;
      df_bulk_max = (fabs((c0 - c2)) * mass_squared + (fabs(baryon * c1) + fabs((4.0 * c2 - c0)) * E) * E) * bulkPi_magnitude;
    }
    else if(DF_MODE == 2)
    {
      double F = df_coeff[0];
      double G = df_coeff[1];
      double betabulk = df_coeff[2];

      double T2 = T * T;
      df_bulk = (baryon * G + F * E / T2 + (E - mass_squared / E) / (3.0 * T)) * bulkPi / betabulk;
      df_bulk_max = (fabs(baryon * G) + fabs(F) * E / T2 + (E - mass_squared / E) / (3.0 * T)) * bulkPi_magnitude / fabs(betabulk);
    }
  }

  if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
  {
    // Vmu LRF components
    double Vx = Vmu.Vx_LRF;
    double Vy = Vmu.Vy_LRF;
    double Vz = Vmu.Vz_LRF;
    double V_magnitude = Vmu.V_magnitude;

    double Vmu_pmu = - (px * Vx + py * Vy + pz * Vz);

    if(DF_MODE == 1)
    {
      double c3 = df_coeff[3];
      double c4 = df_coeff[4];

      df_diff = (baryon * c3 + c4 * E) * Vmu_pmu;
      df_diff_max = (fabs(baryon * c3) + fabs(c4) * E) * E * V_magnitude;
    }
    else if(DF_MODE == 2)
    {
      double betaV = df_coeff[3];

      df_diff = (baryon_enthalpy_ratio - baryon / E) * Vmu_pmu / betaV;
      df_diff_max = (fabs(baryon_enthalpy_ratio) * E - fabs(baryon)) * V_magnitude / fabs(betaV);
    }
  }

  double df = feqbar * (df_shear + df_bulk + df_diff);
  //double df_max = sign_factor * (df_shear_max + df_bulk_max + df_diff_max);


  //double viscous_weight = (1.0 + df) / (1.0 + df_max);

  if(fabs(df) > 1.0) df = max(-1.0, min(df, 1.0));
  

  // regulate df
  //if(fabs(df) > 1.0) viscous_weight = (1.0 + max(-1.0, min(df, 1.0))) / (1.0 + min(df_max, 1.0));

  return (1.0 + df) / 2.0;

}

lrf_momentum EmissionFunctionArray::sample_momentum(default_random_engine& generator, long * pk_acceptances, long * pk_samples, long * angle_acceptances, long * angle_samples, double mass, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear14_coeff, double baryon_enthalpy_ratio)
{
  // sampled LRF momentum
  lrf_momentum pLRF;

  // for accept / reject loop
  bool rejected = true;

  double mass_squared = mass * mass;

  // uniform distributions in (phi, costheta) on standby:
  uniform_real_distribution<double> phi_distribution(0.0, 2.0 * M_PI);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;

  if(mass / T < 1.67)
  {
    // sample p until accepted
    while(rejected)
    {
      *pk_samples = (*pk_samples) + 1;
      // draw p from distribution p^2.exp(-p/T) by
      // sampling three variables (r1,r2,r3) uniformly from [0,1)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p = - T * (l1 + l2 + l3);
      double phi = 2.0 * M_PI * pow((l1 + l2) / (l1 + l2 + l3), 2);   
      double costheta = (l1 - l2) / (l1 + l2);

      double E = sqrt(p * p + mass_squared);
      double sintheta = sqrt(1.0 - costheta * costheta);

      // debugging...
      if(std::isnan(p)) printf("Error: p is nan\n");
      if(std::isnan(phi)) printf("Error: phi is nan\n");
      if(std::isnan(costheta)) printf("Error: costheta is nan\n");

      if(std::isnan(E)) printf("Error: E is nan\n");
      if(std::isnan(sintheta)) printf("Error: sintheta is nan\n");

      if(p < 0.0) printf("Error: p = %lf is out of bounds\n", p);
      if(!(phi >= 0.0 && phi < 2.0 * M_PI)) printf("Error: phi = %lf is out of bounds\n", phi);
      if(fabs(costheta) > 1.0) printf("Error: costheta = %lf is out of bounds\n", costheta);

      pLRF.E = E;
      pLRF.px = p * sintheta * cos(phi);
      pLRF.py = p * sintheta * sin(phi);
      pLRF.pz = p * costheta;

      double pdsigma_abs = fabs(E * dst - pLRF.px * dsx - pLRF.py * dsy - pLRF.pz * dsz);
      double rideal = pdsigma_abs / (E * ds_magnitude);

      double rvisc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);

  

      double weight_light = exp(p/T) / (exp(E/T) + sign) * rideal * rvisc;
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_light - 0.5) > 0.5)
      {
        //printf("Error: weight_light = %f out of bounds\n", weight_light);
      }

      // check p acceptance
      if(propose < weight_light)
      {
        *pk_acceptances = (*pk_acceptances) + 1;
        break;
      }

    } // p rejection loop

  } // sampling light hadron p



  // sample angles uniformly until accepted
  // while(rejected)
  // {
  //   *angle_samples = (*angle_samples) + 1;

  //   double phi = phi_distribution(generator);
  //   double costheta = costheta_distribution(generator);
  //   double sintheta = sqrt(1.0 - costheta * costheta);

  //   // pLRF components
  //   pLRF.E = E;
  //   pLRF.px = p * sintheta * cos(phi);
  //   pLRF.py = p * sintheta * sin(phi);
  //   pLRF.pz = p * costheta;

  //   double pdsigma_abs = fabs(E * dst - pLRF.px * dsx - pLRF.py * dsy - pLRF.pz * dsz);
  //   double rideal = pdsigma_abs / (E * ds_magnitude);

  //   double rvisc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);

  //   double weight_angle = rideal * rvisc;
  //   double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

  //   if(fabs(weight_angle - 0.5) > 0.5)
  //   {
  //     printf("Error: weight_angle = %f out of bounds\n", weight_angle);
  //   }

  //   // check angles' acceptance
  //   if(propose < weight_angle)
  //   {
  //     *angle_acceptances = (*angle_acceptances) + 1;
  //     break;
  //   }

  // } // angles rejection loop

  return pLRF;

}

/*
lrf_momentum EmissionFunctionArray::sample_momentum(default_random_engine& generator, long * pk_acceptances, long * pk_samples, long * angle_acceptances, long * angle_samples, double mass, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear14_coeff, double baryon_enthalpy_ratio)
{
  // sampled LRF momentum
  lrf_momentum pLRF;

  // for accept / reject loop
  bool rejected = true;

  double mass_squared = mass * mass;

  // uniform distributions in (phi, costheta) on standby:
  uniform_real_distribution<double> phi_distribution(0.0, 2.0 * M_PI);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;

  double p, E;   // radial momentum and energy

  // pion sampling routine:
  // not adaptive rejection sampling, but naive method here...
  if(mass / T < 1.67)
  {
    // sample p until accepted
    while(rejected)
    {
      *pk_samples = (*pk_samples) + 1;
      // draw p from distribution p^2.exp(-p/T) by
      // sampling three variables (r1,r2,r3) uniformly from [0,1)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      p = - T * log(r1 * r2 * r3);
      E = sqrt(p * p + mass_squared);

      //here the pion weight should include the Bose enhancement factor
      //this formula assumes zero chemical potential mu = 0
      //TO DO - generalize for nonzero chemical potential
      double weight_light = exp(p/T) / (exp(E/T) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_light - 0.5) > 0.5)
      {
        printf("Error: weight_light = %f out of bounds\n", weight_light);
      }

      // check p acceptance
      if(propose < weight_light)
      {
        *pk_acceptances = (*pk_acceptances) + 1;
        break;
      }

    } // p rejection loop

  } // sampling light hadron p

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // determine which part of integrand dominates
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T;
    double I3 = 2.0 * T * T;
    double Itot = I1 + I2 + I3;

    double I1_over_Itot = I1 / Itot;
    double I1_plus_I2_over_Itot = (I1 + I2) / Itot;

    double k;   // kinetic energy

    // sample k until accepted
    while(rejected)
    {
      *pk_samples = (*pk_samples) + 1;

      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // randomly select distributions to sample k from based on integrated weights
      if(propose_distribution < I1_over_Itot)
      {
        // draw k from distribution exp(-k/T) . dk by sampling r1 uniformly from [0,1):
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        k = - T * log(r1);

      } // distribution 1 (very heavy)
      else if(propose_distribution < I1_plus_I2_over_Itot)
      {
        // draw k from k.exp(-k/T).dk by sampling (r1,r2) uniformly from [0,1):
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k = - T * log(r1 * r2);

      } // distribution 2 (moderately heavy)
      else
      {
        // draw k from k^2.exp(-k/T).dk by sampling (r1,r2,r3) uniformly from [0,1):
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k = - T * log(r1 * r2 * r3);

      } // distribution 3 (light)

      E = k + mass;
      p = sqrt(E * E - mass_squared);

      double weight_heavy = p/E * exp(E/T) / (exp(E/T) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_heavy - 0.5) > 0.5)
      {
        printf("Error: weight_heavy = %f out of bounds\n", weight_heavy);
      }

      // check k acceptance
      if(propose < weight_heavy)
      {
        *pk_acceptances = (*pk_acceptances) + 1;
        break;
      }

    } // k rejection loop

  } // sampling heavy hadron k

  // sample angles uniformly until accepted
  while(rejected)
  {
    *angle_samples = (*angle_samples) + 1;

    double phi = phi_distribution(generator);
    double costheta = costheta_distribution(generator);
    double sintheta = sqrt(1.0 - costheta * costheta);

    // pLRF components
    pLRF.E = E;
    pLRF.px = p * sintheta * cos(phi);
    pLRF.py = p * sintheta * sin(phi);
    pLRF.pz = p * costheta;

    double pdsigma_abs = fabs(E * dst - pLRF.px * dsx - pLRF.py * dsy - pLRF.pz * dsz);
    double rideal = pdsigma_abs / (E * ds_magnitude);

    double rvisc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);

    double weight_angle = rideal * rvisc;
    double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

    if(fabs(weight_angle - 0.5) > 0.5)
    {
      printf("Error: weight_angle = %f out of bounds\n", weight_angle);
    }

    // check angles' acceptance
    if(propose < weight_angle)
    {
      *angle_acceptances = (*angle_acceptances) + 1;
      break;
    }

  } // angles rejection loop

  return pLRF;

}
*/

lrf_momentum EmissionFunctionArray::rescale_momentum(lrf_momentum pmod, double mass_squared, double baryon, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio)
{
    double Emod = pmod.E;
    double pxmod = pmod.px;
    double pymod = pmod.py;
    double pzmod = pmod.pz;

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
    diff_coeff *= (Emod * baryon_enthalpy_ratio + baryon);

    lrf_momentum pLRF;

    // Default ideal (no viscous corrections)
    pLRF.px = pxmod;
    pLRF.py = pymod;
    pLRF.pz = pzmod;

    // rescale momentum with viscous transformation M(pmod):
    pLRF.px += (shear_coeff * (pixx * pxmod + pixy * pymod + pixz * pzmod) + bulk_coeff * pxmod + diff_coeff * Vx);
    pLRF.py += (shear_coeff * (pixy * pxmod + piyy * pymod + piyz * pzmod) + bulk_coeff * pymod + diff_coeff * Vy);
    pLRF.pz += (shear_coeff * (pixz * pxmod + piyz * pymod + pizz * pzmod) + bulk_coeff * pzmod + diff_coeff * Vz);
    pLRF.E = sqrt(mass_squared + pLRF.px * pLRF.px + pLRF.py * pLRF.py + pLRF.pz * pLRF.pz);

    return pLRF;
}


lrf_momentum EmissionFunctionArray::sample_momentum_feqmod(default_random_engine& generator, long * mod_acceptances, long * mod_samples, double mass, double sign, double baryon, double T_mod, double alphaB_mod, Surface_Element_Vector dsigma, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio)
{
  double two_pi = 2.0 * M_PI;

  lrf_momentum pLRF;                          // sampled LRF momentum
  lrf_momentum pLRF_mod;                      // sampled modified LRF momentum

  bool rejected = true;                       // for accept/reject loop

  double mass_squared = mass * mass;

   // uniform distributions in (phi_mod, costheta_mod) on standby
  uniform_real_distribution<double> phi_distribution(0.0, two_pi);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;

  // pion routine (not adaptive rejection sampling, but naive method here.. ~ massless Fermi distribution)
  if(mass / T_mod < 1.67)
  {
    while(rejected)
    {
      *mod_samples = (*mod_samples) + 1;
      // draw (p_mod, phi_mod, costheta_mod) from distribution p_mod^2 * exp(-p_mod / T_mod) by
      // sampling three variables (r1,r2,r3) uniformly from [0,1)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p_mod = - T_mod * (l1 + l2 + l3);                        // modified 3-momentum magnitude
      double E_mod = sqrt(p_mod * p_mod + mass_squared);              // modified energy

      double phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);   // modified angles
      double costheta_mod = (l1 - l2) / (l1 + l2);

      double sintheta_mod = sqrt(1.0 - costheta_mod * costheta_mod);

      // modified pLRF components
      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // momentum rescaling
      pLRF = rescale_momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

      double pdsigma_abs = fabs(pLRF.E * dst - pLRF.px * dsx - pLRF.py * dsy - pLRF.pz * dsz);
      double rideal = pdsigma_abs / (pLRF.E * ds_magnitude);

      //here the pion weight should include the Bose enhancement factor
      //this formula assumes zero chemical potential mu = 0
      //TO DO - generalize for nonzero chemical potential
      double weight_light = rideal * exp(p_mod/T_mod) / (exp(E_mod/T_mod) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_light - 0.5) > 0.5)
      {
        printf("Error: weight = %f out of bounds\n", weight_light);
      }

      // check pLRF acceptance
      if(propose < weight_light)
      {
        *mod_acceptances = (*mod_acceptances) + 1;
        break;
      }

    } // rejection loop

  } // sampling light hadrons

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // determine which part of integrand dominates
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T_mod;
    double I3 = 2.0 * T_mod * T_mod;
    double Itot = I1 + I2 + I3;

    double I1_over_Itot = I1 / Itot;
    double I1_plus_I2_over_Itot = (I1 + I2) / Itot;

    double k_mod, phi_mod, costheta_mod;    // modified kinetic energy and angles

    while(rejected)
    {
      *mod_samples = (*mod_samples) + 1;

      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // randomly select distributions to sample momentum from based on integrated weights
      if(propose_distribution < I1_over_Itot)
      {
        // draw k_mod from exp(-k_mod/T_mod) by sampling r1 uniformly from [0,1):
        // sample (costheta_mod, phi_mod) uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k_mod = - T_mod * log(r1);
        phi_mod = phi_distribution(generator);
        costheta_mod = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      else if(propose_distribution < I1_plus_I2_over_Itot)
      {
        // draw (k_mod, phi_mod) from k_mod.exp(-k_mod/T_mod) by sampling (r1,r2) uniformly from [0,1):
        // sample costheta_mod uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k_mod = - T_mod * (l1 + l2);
        phi_mod = two_pi * l1 / (l1 + l2);
        costheta_mod = costheta_distribution(generator);

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

      double E_mod = k_mod + mass;
      double p_mod = sqrt(E_mod * E_mod - mass_squared);
      double sintheta_mod = sqrt(1.0 - costheta_mod * costheta_mod);

      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // momentum rescaling
      pLRF = rescale_momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

      double pdsigma_abs = fabs(pLRF.E * dst - pLRF.px * dsx - pLRF.py * dsy - pLRF.pz * dsz);
      double rideal = pdsigma_abs / (pLRF.E * ds_magnitude);

      double weight_heavy = rideal * (p_mod / E_mod) * exp(E_mod / T_mod) / (exp(E_mod / T_mod) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_heavy - 0.5) > 0.5)
      {
        printf("Error: weight = %f out of bounds\n", weight_heavy);
      }

      // check pLRF acceptance
      if(propose < weight_heavy)
      {
        *mod_acceptances = (*mod_acceptances) + 1;
        break;
      }

    } // rejection loop

  } // sampling heavy hadrons

  return pLRF;

}



double EmissionFunctionArray::estimate_total_yield(double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *tau_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *bulkPi_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo)
  {
    printf("Estimating total particle yield...\n");

    int npart = number_of_chosen_particles;

    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaDeltaWeights[eta_pts];                                      // eta_weight * delta_eta
    double delta_eta = fabs((eta_tab->get(1,2)) - (eta_tab->get(1,1)));   // assume uniform grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta_weight = eta_tab->get(2, ieta + 1);
        etaDeltaWeights[ieta] = eta_weight * delta_eta;
      }
    }
    else if(DIMENSION == 3) etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d

    double neq_tot = 0.0;                 // total equilibrium number / udsigma
    double dn_bulk_tot = 0.0;             // bulk correction / udsigma / bulkPi
    double dn_diff_tot = 0.0;             // diffusion correction / Vdsigma

    for(int ipart = 0; ipart < npart; ipart++)
    {
      neq_tot += Equilibrium_Density[ipart];
      dn_bulk_tot += Bulk_Density[ipart];
      dn_diff_tot += Diffusion_Density[ipart];

    } // hadron species (ipart)

    double Ntot = 0.0;                    // total particle yield (includes p.dsigma < 0 particles)

    #pragma omp parallel for
    for (long icell = 0; icell < FO_length; icell++)
    {
      double tau = tau_fo[icell];         // longitudinal proper time

      double dat = dat_fo[icell];         // covariant normal surface vector dsigma_mu
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];         // dan should be 0 in 2+1d case

      double ux = ux_fo[icell];           // contravariant fluid velocity u^mu
      double uy = uy_fo[icell];           // enforce normalization
      double un = un_fo[icell];           // u^eta (fm^-1)
      double ut = sqrt(1.0 + ux * ux + uy * uy + tau * tau * un * un); //u^tau

      double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

      if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu (fm^-3)
      double Vx = 0.0;                    // enforce orthogonality
      double Vy = 0.0;
      double Vn = 0.0;

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        Vx = Vx_fo[icell];
        Vy = Vy_fo[icell];
        Vn = Vn_fo[icell];
        Vt = (Vx * ux + Vy * uy + tau * tau * Vn * un) / ut;
      }

      // udotdsigma and Vdotdsigma w/o delta_eta_weight factor
      double udsigma = ut * dat + ux * dax + uy * day + un * dan;
      double Vdsigma = Vt * dat + Vx * dax + Vy * day + Vn * dan;

      // total number of hadrons / delta_eta_weight in FO_cell
      double dn_tot = udsigma * (neq_tot + bulkPi * dn_bulk_tot) - Vdsigma * dn_diff_tot;

      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        if(udsigma <= 0.0) break;
        // add mean number of hadrons in FO cell to total yield
        Ntot += dn_tot * etaDeltaWeights[ieta];

      } // eta points (ieta)

    } // freezeout cells (icell)

    return 0.99 * Ntot;   // assume ~ 99% of particles have p.dsigma > 0

  }





void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo,double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, double *thermodynamic_average)
  {
    int npart = number_of_chosen_particles;
    double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);


    // fix the seed and generator
    //unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    unsigned seed = 1;
    default_random_engine generator(seed);


    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;                   // extension in eta
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts];                               // eta_weight * delta_eta
    double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * fabs(delta_eta);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
    }

    // average thermodynamic quantities
    double Tavg = thermodynamic_average[0];
    double Eavg = thermodynamic_average[1];
    double Pavg = thermodynamic_average[2];
    double muBavg = thermodynamic_average[3];
    double nBavg = thermodynamic_average[4];

    // set df coefficients
    double c0, c1, c2, c3, c4;            // 14 moment
    double F, G, betabulk, betaV, betapi; // Chapman Enskog

    switch(DF_MODE)
    {
      case 1: // 14-moment
      {
        c0 = df_coeff[0];         // bulk coefficients
        c1 = df_coeff[1];
        c2 = df_coeff[2];
        c3 = df_coeff[3];         // diffusion coefficients
        c4 = df_coeff[4];
        // shear coefficient evaluated later
        break;
      }
      case 2: // Chapman-Enskog
      case 3: // Modified
      {
        F = df_coeff[0];           // bulk coefficients
        G = df_coeff[1];
        betabulk = df_coeff[2];
        betaV = df_coeff[3];       // diffusion coefficient
        betapi = df_coeff[4];      // shear coefficient
        break;
      }
      default:
      {
        printf("\nError: please set df_mode = (1,2,3)\n");
        exit(-1);
      }
    }

    // compute contributions to total mean hadron number / cell volume
    double neq_tot = 0.0;                      // total equilibrium number / udsigma
    double dn_bulk_tot = 0.0;                  // total bulk number / bulkPi / udsigma
    double dn_diff_tot = 0.0;                  // - total diff number / Vdsigma

    for(int ipart = 0; ipart < npart; ipart++)
    {
      neq_tot += Equilibrium_Density[ipart];
      dn_bulk_tot += Bulk_Density[ipart];
      dn_diff_tot += Diffusion_Density[ipart];
    }

    // for benchmarking momentum sampling efficiency
    long pk_acceptances = 0;
    long pk_samples = 0;
    long angle_acceptances = 0;
    long angle_samples = 0;

    long mod_acceptances = 0;
    long mod_samples = 0;

    //loop over all freezeout cells
    #pragma omp parallel for
    for(long icell = 0; icell < FO_length; icell++)
    {
      double tau = tau_fo[icell];         // FO cell coordinates
      double tau2 = tau * tau;
      double x = x_fo[icell];
      double y = y_fo[icell];
      if(DIMENSION == 3) etaValues[0] = eta_fo[icell];

      double dat = dat_fo[icell];         // covariant normal surface vector dsigma_mu
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];         // dan should be 0 in 2+1d case

      double ux = ux_fo[icell];           // contravariant fluid velocity u^mu
      double uy = uy_fo[icell];           // enforce normalization u.u = 1
      double un = un_fo[icell];           // u^eta (fm^-1)
      double ut = sqrt(1.0 + ux * ux + uy * uy + tau2 * un * un); //u^tau


      double ut2 = ut * ut;               // useful expressions
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux2 + uy2);
      double utperp = sqrt(1.0 + uperp * uperp);

      double T = Tavg;                    // temperature (GeV)
      double E = Eavg;                    // energy density (GeV/fm^3)
      double P = Pavg;                    // equilibrium pressure (GeV/fm^3)

      double pitt = 0.0;                  // contravariant shear stress tensor pi^munu (GeV/fm^3)
      double pitx = 0.0;                  // enforce orthogonality pi.u = 0
      double pity = 0.0;                  // and tracelessness Tr(pi) = 0
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
        pinn = (pixx * (ux2 - ut2) + piyy * (uy2 - ut2) + 2.0 * (pixy * ux * uy + tau2 * un * (pixn * ux + piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux + piyn * uy + tau2 * pinn * un) / ut;
        pity = (pixy * ux + piyy * uy + tau2 * piyn * un) / ut;
        pitx = (pixx * ux + pixy * uy + tau2 * pixn * un) / ut;
        pitt = (pitx * ux + pity * uy + tau2 * pitn * un) / ut;
      }

      // testing orthogonality and tracelessness (checked)
      //cout << setprecision(15) << pitt * ut - pitx * ux - pity * uy - tau2 * pitn * un << endl;
      //cout << setprecision(15) << pitx * ut - pixx * ux - pixy * uy - tau2 * pixn * un << endl;
      //cout << setprecision(15) << pity * ut - pixy * ux - piyy * uy - tau2 * piyn * un << endl;
      //cout << setprecision(15) << pitn * ut - pixn * ux - piyn * uy - tau2 * pinn * un << endl;
      //cout << setprecision(15) << pitt - pixx - piyy - tau2 * pinn << endl;


      double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

      if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

      double muB = muBavg;                // baryon chemical potential (GeV)
      double alphaB = muB / T;            // muB / T
      double nB = nBavg;                  // net baryon density
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu (fm^-3)
      double Vx = 0.0;                    // enforce orthogonality V.u = 0
      double Vy = 0.0;
      double Vn = 0.0;
      double baryon_enthalpy_ratio = nB / (E + P);

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        Vx = Vx_fo[icell];
        Vy = Vy_fo[icell];
        Vn = Vn_fo[icell];
        Vt = (Vx * ux + Vy * uy + tau2 * Vn * un) / ut;
      }

      // set milne basis vectors
      Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);

      double Xt = basis_vectors.Xt;
      double Xx = basis_vectors.Xx;
      double Xy = basis_vectors.Xy;
      double Xn = basis_vectors.Xn;
      double Yx = basis_vectors.Yx;
      double Yy = basis_vectors.Yy;
      double Zt = basis_vectors.Zt;
      double Zn = basis_vectors.Zn;


      //cout << setprecision(15) << Xt * Xt - Xx * Xx - Xy * Xy - tau2 * Xn * Xn << endl;
      //cout << setprecision(15) << - Yx * Yx - Yy * Yy << endl;
      //cout << setprecision(15) << Zt * Zt - tau2 * Zn * Zn << endl;

      // dsigma / delta_eta_weight class
      Surface_Element_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_max();

      // shear stress class
      Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);
      pimunu.compute_pimunu_max();

      // baryon diffusion class
      Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
      Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);
      Vmu.compute_Vmu_max();

      // (udotdsigma, Vdotdsigma) / delta_eta_weight
      double udsigma = ut * dat + ux * dax + uy * day + un * dan;
      double Vdsigma = Vt * dat + Vx * dax + Vy * day + Vn * dan;

      // total mean number  / delta_eta_weight of hadrons in FO cell
      double dn_tot = udsigma * (neq_tot + bulkPi * dn_bulk_tot) - Vdsigma * dn_diff_tot;

      // loop over eta points
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        if(udsigma <= 0.0) break;                               // skip over cells with u.dsigma < 0

        double eta = etaValues[ieta];
        double delta_eta_weight = etaDeltaWeights[ieta];
        double dN_tot = delta_eta_weight * dn_tot;              // total mean number of hadrons in FO cell

        std::poisson_distribution<> poisson_hadrons(dN_tot);    // probability distribution for number of hadrons

        if(dN_tot <= 0.0) printf("Negative hadrons\n");

        // sample events for each FO cell
        for(int ievent = 0; ievent < Nevents; ievent++)
        {
          int N_hadrons = poisson_hadrons(generator);           // sample total number of hadrons in FO cell

          for(int n = 0; n < N_hadrons; n++)
          {
            // mean number of particles of species ipart / delta_eta_weight
            std::vector<double> density_list;
            density_list.resize(npart);

            for(int ipart = 0; ipart < npart; ipart++)
            {
              double neq = Equilibrium_Density[ipart];
              double dn_bulk = Bulk_Density[ipart];
              double dn_diff = Diffusion_Density[ipart];

              density_list[ipart] = udsigma * (neq + bulkPi * dn_bulk) - Vdsigma * dn_diff;
            }

            // discrete probability distribution for particle types (weights ~ dN_ipart / dN_tot)
            std::discrete_distribution<int> particle_type(density_list.begin(), density_list.end());



            int chosen_index = particle_type(generator);        // chosen index of sampled particle type
            double mass = Mass[chosen_index];                   // mass of sampled particle in GeV
            double sign = Sign[chosen_index];                   // quantum statistics sign
            double baryon = Baryon[chosen_index];               // baryon number
            int mcid = MCID[chosen_index];                      // mc id

            lrf_momentum pLRF;

            // sample local rest frame momentum
            switch(DF_MODE)
            {
              case 1: // 14 moment
              case 2: // Chapman Enskog
              {
                double shear14_coeff = 2.0 * T * T * (E + P); // 14 moment shear coefficient

                pLRF = sample_momentum(generator, &pk_acceptances, &pk_samples, &angle_acceptances, &angle_samples, mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);
                break;
              }
              case 3: // Modified
              {
                double shear_coeff = 0.5 / betapi;
                double bulk_coeff = bulkPi / (3.0 * betabulk);
                double diff_coeff = T / betaV;

                double T_mod = T + bulkPi * F / betabulk;
                double alphaB_mod = alphaB + bulkPi * G / betabulk;

                // somehow I can switch to df case under certain conditions (goto 2)

                pLRF = sample_momentum_feqmod(generator, &mod_acceptances, &mod_samples, mass, sign, baryon, T_mod, alphaB_mod, dsigma, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

                break;
              }
              default:
              {
                printf("\nError: for vhydro momentum sampling please set df_mode = (1,2,3)\n");
                exit(-1);
              }
            }

            // lab frame momentum
            Lab_Momentum pLab(pLRF);
            pLab.boost_pLRF_to_lab_frame(basis_vectors, ut, ux, uy, un);

            // p.dsigma / delta_eta_weight
            double pdsigma = pLab.ptau * dat + pLab.px * dax + pLab.py * day + pLab.pn * dan;

            // add sampled particle to particle_event_list
            if(pdsigma >= 0.0)
            {
              double sinheta = sinh(eta);
              double cosheta = sqrt(1.0 + sinheta * sinheta);

              // add sampled particle to particle_event_list
              Sampled_Particle new_particle;
              new_particle.mcID = mcid;
              new_particle.tau = tau;
              new_particle.x = x;
              new_particle.y = y;
              new_particle.eta = eta;

              new_particle.t = tau * cosheta;
              new_particle.z = tau * sinheta;

              new_particle.mass = mass;
              new_particle.px = pLab.px;
              new_particle.py = pLab.py;

              double pz = tau * pLab.pn * cosheta + pLab.ptau * sinheta;
              new_particle.pz = pz;
              //new_particle.E = pmu.ptau * cosheta  +  tau * pmu.pn * sinheta;
              //is it possible for this to be too small or negative?
              //try enforcing mass shell condition on E
              new_particle.E = sqrt(mass * mass + pLab.px * pLab.px + pLab.py * pLab.py + pz * pz);

              //add to particle_event_list
              //CAREFUL push_back is not a thread-safe operation
              //how should we modify for GPU version?
              #pragma omp critical
              particle_event_list[ievent].push_back(new_particle);
            } // if(pdsigma >= 0)

          } // sampled hadrons (n)

        } // sampled events (ievent)

      } // eta points (ieta)

    } // freezeout cells (icell)

    if(DF_MODE == 1 || DF_MODE == 2)
    {
      printf("Momentum sampling efficiency (p,k) = %lf\n", (double)pk_acceptances / (double)pk_samples);
      printf("Momentum sampling efficiency (phi,theta) = %lf\n\n", (double)angle_acceptances / (double)angle_samples);
    }
    else if(DF_MODE == 3)
    {
      printf("Modified momentum sampling efficiency = %lf\n\n", (double)mod_acceptances / (double)mod_samples);
    }

}

/*
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
          lrf_momentum pLRF = sample_momentum_feqmod(mass, sign, baryon, T_mod, alphaB_mod, dsigma, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

          //double shear_coeff_14 = 2.0 * T * T * (E + P);
          //lrf_momentum pLRF = Sample_Momentum_feq_plus_deltaf(mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, shear_coeff_14, INCLUDE_SHEAR_DELTAF, INCLUDE_BULK_DELTAF, INCLUDE_BARYONDIFF_DELTAF, DF_MODE);



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
*/

void EmissionFunctionArray::sample_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
  double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
  double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
{
  printf("Sampling particles from VAH \n");
  printf("NOTHING HERE YET \n");
}
