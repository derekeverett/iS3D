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
#include "iS3D.h"
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

using namespace std;

const double pi_over2 = M_PI / 2.0;
const double pi_over4 = M_PI / 4.0;
const double solution_tolerance = 1.e-10;


void EmissionFunctionArray::sample_dN_dpT(Sampled_Particle new_particle)
{
  // construct the dN/dpT distribution by adding counts from all events

  double E = new_particle.E;
  double pz = new_particle.pz;
  double yp = 0.5 * log((E + pz) / (E - pz));

  if(fabs(yp) <= Y_CUT)
  {
    int ipart = new_particle.chosen_index;

    total_count[ipart] += 1.0;              // add counts within rapidity cut

    double px = new_particle.px;
    double py = new_particle.py;
    double pT = sqrt(px * px  +  py * py);

    double pTbinwidth = (PT_UPPER_CUT - PT_LOWER_CUT) / (double)PT_BINS;

    int ipT = (int)floor((pT - PT_LOWER_CUT) / pTbinwidth);  // pT bin index

    if(ipT >= 0 && ipT < PT_BINS)
    {
      sampled_pT_PDF[ipart][ipT] += 1.0;    // add counts to each pT bin
    }
  }
}

void EmissionFunctionArray::sample_vn(Sampled_Particle new_particle)
{
  // construct the vn's by adding counts from all events

  double E = new_particle.E;
  double pz = new_particle.pz;
  double yp = 0.5 * log((E + pz) / (E - pz));

  if(fabs(yp) <= Y_CUT)
  {
    int ipart = new_particle.chosen_index;

    double px = new_particle.px;
    double py = new_particle.py;
    double pz = new_particle.pz;

    double phi = atan2(py, px);
    if(phi < 0.0)
    {
      phi += 2.0 * M_PI;
    }
    double pT = sqrt(px * px  +  py * py);

    double pTbinwidth = (PT_UPPER_CUT - PT_LOWER_CUT) / (double)PT_BINS;

    int ipT = (int)floor((pT - PT_LOWER_CUT) / pTbinwidth);  // pT bin index

    if(ipT >= 0 && ipT < PT_BINS)
    {
      // add counts to each pT bin
      pT_count_vn[ipart][ipT] += 1.0;

      for(int k = 0; k < K_MAX; k++)
      {
        // add complex exponential weights to each bin
        sampled_vn_real[k][ipart][ipT] += cos(((double)k + 1.0) * phi);
        sampled_vn_imag[k][ipart][ipT] += sin(((double)k + 1.0) * phi);
      }
    }
  }
}

void EmissionFunctionArray::sample_dN_dX(Sampled_Particle new_particle)
{
  // construct spacetime distributions by adding counts from all events

  double E = new_particle.E;
  double pz = new_particle.pz;
  double yp = 0.5 * log((E + pz) / (E - pz));

  if(fabs(yp) <= Y_CUT)
  {
    int ipart = new_particle.chosen_index;

    double taubinwidth = (TAU_MAX - TAU_MIN) / (double)TAU_BINS;
    double rbinwidth = (R_MAX - R_MIN) / (double)R_BINS;

    double tau = new_particle.tau;
    double x = new_particle.x;
    double y = new_particle.y;
    double r = sqrt(x * x  +  y * y);

    int itau = (int)floor((tau - TAU_MIN) / taubinwidth); // tau bin index
    int ir = (int)floor((r - R_MIN) / rbinwidth);         // r bin index

    if(itau >= 0 && itau < TAU_BINS)
    {
      sampled_dN_taudtaudy[ipart][itau] += 1.0; // add count to tau bin
    }
    if(ir >= 0 && ir < R_BINS)
    {
      sampled_dN_twopirdrdy[ipart][ir] += 1.0;  // add count to rbin
    }
  }
}


double canonical(default_random_engine & generator)
{
  // random number between [0,1)
  return generate_canonical<double, numeric_limits<double>::digits>(generator);
}


double pion_thermal_weight_max(double mass, double T, double chem)
{
  // rescale the pion thermal weight w_eq by the max value if m/T < 0.8554 (the max is local)
  // we assume no pion chemical potential (i.e. ignore non-equilibrium chemical potential EoS models)

  if(chem != 0.0) printf("Error: pion has chemical potential\n");

  double x = mass / T;  // x < 0.8554
  double x2 = x * x;
  double x3 = x2 * x;
  double x4 = x3 * x;

  // rational polynomial fit of the max value
  double max = (143206.88623164667 - 95956.76008684626*x - 21341.937407169076*x2 + 14388.446116867359*x3 -
     6083.775788504437*x4)/
    (-0.3541350577684533 + 143218.69233952634*x - 24516.803600065778*x2 - 115811.59391199696*x3 +
     35814.36403387459*x4);

  if(x < 0.1) printf("Out of data interpolation range: extrapolating fit...\n");

  double buffer = 1.00001; // ensures rescaled w_eq <= 1.0 numerically

  return buffer * max;
}


double equilibrium_particle_density(double mbar, double chem, double sign)
{
  // compute the thermal particle density (w/o the degeneracy * mass^2 * T / two_pi2_hbarC3 prefactor)
  double neq = 0.0;

  double sign_factor = -sign;

  int n_min = 5;    // min number of iterations
  int n_max = 30;   // max number of iterations

  int n;

  double tolerance = 1.e-5;  // 0.001% tolarance
  double error;

  // sum truncated expansion of Bose-Fermi distribution
  for(n = 1; n < n_min; n++)
  {
    double k = (double)n;
    sign_factor *= (-sign);
    neq += sign_factor * exp(k * chem) * gsl_sf_bessel_Kn(2, k * mbar) / k;
  }
  do
  {
    double neq_prev = neq;

    double k = (double)n;
    sign_factor *= (-sign);
    neq += sign_factor * exp(k * chem) * gsl_sf_bessel_Kn(2, k * mbar) / k;

    error = (neq - neq_prev) / neq_prev;

    n++;  // keep iterating until acheived desired precision (I think n++ goes here; double check)

  } while(n < n_max || fabs(error) <= tolerance);

  return neq;
}

double compute_outflow_integrand(double dst, double dsperp, double dsz, double p, double E)
{
  double a = dsz;
  double b = dsperp;
  double c = dst * E / p;

  // I think the current issue I have the wrong weights for the amplitudes
  // bc the square wave can be shifted
  // solve for the root(s) of f(x) = c - a.x - b.sqrt(1-x^2)
  double a2 = a * a;
  double b2 = b * b;
  double b4 = b2 * b2;
  double c2 = c * c;
  double sqrt_term = sqrt(a2*b2 + b4 - b2*c2);

  if(isnan(sqrt_term))
  {
    // no solution and f(x) > 0 or Theta(f(x)) = 1 (since c > 0)
    //printf("Solutions are complex\n");
    return 2.0 * dst;
  }

  // solutions (within costheta domain = [-1,1])
  double x1 = max(-1.0, min(1.0, (a*c - sqrt_term) / (a2 + b2)));
  double x2 = max(-1.0, min(1.0, (a*c + sqrt_term) / (a2 + b2)));

  if(x1 > x2) printf("Error solutions ordered wrong\n");

  // check if one of the solutions is extraneous


  // positive part of azimuthal square wave
  // b -> + ds_perp_amp
  // f(x) = convex
  double sol1_plus = c - a*x1 - b*sqrt(1.-x1*x1); // proposed solutions
  double sol2_plus = c - a*x2 - b*sqrt(1.-x2*x2);

  double xp = x1;

  int sols_plus = 2;

  if(fabs(sol1_plus) > solution_tolerance)        // check if solutions are extraneous
  {
    xp = x2;
    sols_plus--;
  }
  if(fabs(sol2_plus) > solution_tolerance)
  {
    sols_plus--;
  }

  double integrand_plus;

  if(sols_plus == 0)
  {
     // Theta(f(x)) = 1 in [-1, 1]
    integrand_plus = 2.*dst - p/E*dsperp*pi_over2;
  }
  else if(sols_plus == 1 && a > 0.0)
  {
    // Theta(f(x)) = 1 in [-1, xp]
    integrand_plus = dst*(1.+xp) - p/E*(dsperp*(0.5*(xp*sqrt(1.-xp*xp) + asin(xp) + pi_over2)) + 0.5*dsz*(xp*xp-1.));
  }
  else if(sols_plus == 1 && a < 0.0)
  {
    // Theta(f(x)) = 1 in [xp, 1]
    integrand_plus = dst*(1.-xp) - p/E*(dsperp*(0.5*(pi_over2 - xp*sqrt(1.-xp*xp) - asin(xp))) + 0.5*dsz*(1.-xp*xp));
  }
  else if(sols_plus == 2)
  {
    // Theta(f(x)) = 1 in [-1, x1] + [x2, 1]
    integrand_plus = dst*(2.+x1-x2) - p/E*(dsperp*(pi_over2 + 0.5*(x1*sqrt(1.-x1*x1) - x2*sqrt(1.-x2*x2) + asin(x1) - asin(x2))) + 0.5*dsz*(x1*x1 - x2*x2));
  }



  // negative part of azimuthal square wave
  // b -> - ds_perp_amp
  // f(x) = concave
  double sol1_minus = c - a*x1 + b*sqrt(1.-x1*x1); // proposed solutions
  double sol2_minus = c - a*x2 + b*sqrt(1.-x2*x2);

  double xm = x1;

  int sols_minus = 2;

  if(fabs(sol1_minus) > solution_tolerance)        // check if solutions are extraneous
  {
    xm = x2;
    sols_minus--;
  }
  if(fabs(sol2_minus) > solution_tolerance)
  {
    sols_minus--;
  }

  double integrand_minus;

  if(sols_minus == 0)
  {
     // Theta(f(x)) = 1 in [-1, 1]
    integrand_minus = 2.*dst + p/E*dsperp*pi_over2;
  }
  else if(sols_minus == 1 && a > 0.0)
  {
    // Theta(f(x)) = 1 in [-1, xm]
    integrand_minus = dst*(1.+xm) - p/E*(-dsperp*(0.5*(xm*sqrt(1.-xm*xm) + asin(xm) - pi_over2)) + 0.5*dsz*(xm*xm-1.));
  }
  else if(sols_minus == 1 && a < 0.0)
  {
    // Theta(f(x)) = 1 in [xm, 1]
    integrand_minus = dst*(1.-xm) - p/E*(-dsperp*(0.5*(pi_over2 - xm*sqrt(1.-xm*xm) - asin(xm))) + 0.5*dsz*(1.-xm*xm));
  }
  else if(sols_minus == 2)
  {
    // Theta(f(x)) = 1 in [x1,x2]
    integrand_minus = dst*(x2-x1) - p/E*(-dsperp*(0.5*(x2*sqrt(1.-x2*x2) - x1*sqrt(1.-x1*x1) + asin(x2) - asin(x1))) + 0.5*dsz*(x2*x2 - x1*x1));
  }

  return (integrand_plus + integrand_minus) / 2.0;
}


// boost-invariant approximation
double compute_outflow_integrand_2(double dst, double dsperp, double p, double E)
{
  double b = dsperp;
  double c = dst * E / p;

  double sqrt_term = sqrt(b*b - c*c);

  if(isnan(sqrt_term)) return 2.*dst;

  // solutions (within costheta domain = [-1,1])
  double x1 = max(-1.0, min(1.0, - sqrt_term / b));
  double x2 = max(-1.0, min(1.0, sqrt_term / b));

  if(x1 > x2) printf("Error solutions ordered wrong\n");

  // positive part of azimuthal square wave
  // b -> + ds_perp_amp
  // f(x) = convex
  double sol_plus = c - b*sqrt(1.-x1*x1); // either 2 solutions or no solution

  int sols_plus = 2;
  if(fabs(sol_plus) > solution_tolerance) // check if solutions are extraneous
  {
    sols_plus = 0;
  }

  double integrand_plus;

  if(sols_plus == 0)
  {
     // Theta(f(x)) = 1 in [-1, 1]
    integrand_plus = 2.*dst - p/E*dsperp*pi_over2;
  }
  else if(sols_plus == 2)
  {
    // Theta(f(x)) = 1 in [-1, x1] + [x2, 1]
    integrand_plus = 2.*dst*(1.+x1) - p/E*(dsperp*(pi_over2 + x1*sqrt(1.-x1*x1) + asin(x1)));
  }

  // f(x) is always positive
  double integrand_minus = 2.*dst + p/E*dsperp*pi_over2;

  return (integrand_plus + integrand_minus) / 2.0;
}

double compute_outflow_integrand_3(double dst, double dsperp, double p, double E)
{
  double b = dsperp;
  double c = dst * E / p;


  double integrand_plus;

  if(b > c)
  {
    return 2.0;
  }
}


double compute_angular_phase_space_integral(double feq, double h, double g, double pizz_D, double piperp_amp)
{
  // evaluate angular integral ~ int dx.dphi feq(p).max(0.0, min(1.0 + h(p) + g(p).k(x, phi), 2.0)
  //    x = costheta (polar angle)
  //    phi = azimuthal angle
  //    k(x, phi) = pizz_D.x^2 + (1-x^2)(piperp + delta_piperp.cos(2phi))

  // we evaluate the x-integral by piecewise integration after approximating the azimuthal
  // term in k(x,phi) { piperp + delta_piperp.cos(2phi) } as a square wave with mean amplitudes
  //    piperp_plus = piperp + 2.delta_piperp/pi   (corresponds to f_plus)
  //    piperp_minus = piperp - 2.delta_piperp/pi  (corresponds to f_minus)

  // index notation
  //    xL = roots of lower bound feq(1+df) = 0
  //    xH = roots of upper bound feq(1+df) = 2feq
  //    plus = square wave region w/ piperp_plus
  //    minus = square wave region w/ piperp_minus

  // curvature of the k(x) wpt x
  double C = pizz_D - piperp_amp;

  double xL = sqrt(max(0.0, min( (-piperp_amp - (1.0 + h)/g) / C, 1.0)));
  double xH = sqrt(max(0.0, min( (-piperp_amp + (1.0 - h)/g) / C, 1.0)));

  double f_angular_integral;

  // feq + df_reg is concave
  if(C < 0.0)
  {
    if(!(xL >= xH)) printf("Error in concave piecewise function..\n");

    f_angular_integral = feq * (xL + xH + h*(xL - xH) + g*(C*(xL*xL*xL - xH*xH*xH)/3.0 + piperp_amp*(xL - xH)));
  }
  // feq + df_reg is convex
  else if(C > 0.0)
  {
    if(!(xH >= xL)) printf("Error in convex piecewise function..\n");
    f_angular_integral = feq * (2.0 - xL - xH + h*(xH - xL) + g*(C*(xH*xH*xH - xL*xL*xL)/3.0 + piperp_amp*(xH - xL)));
  }
  // f + df_reg is const
  else
  {
    f_angular_integral = feq * max(0.0, min(1.0 + h + g*pizz_D, 2.0));
  }

  return f_angular_integral;
}




double mean_particle_number(double mass, double degeneracy, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, deltaf_coefficients df, double T_mod, double alphaB_mod, bool feqmod_breaks_down, Gauss_Laguerre * laguerre, const int legendre_pts, double * pbar_root, double * pbar_weight, int df_mode, int outflow, int include_baryon, double detA, double detA_bulk, int sample_with_max_volume)
{
  // computes the particle number from the CFF with Theta(p.dsigma) (outflow) and feq + df_regulated (or feqmod)
  // for spacelike cells: (neq + dn_regulated)_outflow > neq + dn_regulated
  // for timelike cells:  (neq + dn_regulated)_outflow = neq + dn_regulated
  // the reason we do this: self-consistency between sampled and smooth CFF for more precise testing of the sampler

  // the feqmod formula is an approximation due to the complexity of the integral

  double mass_squared = mass * mass;

  double mbar = mass / T;
  double mbar_squared = mbar * mbar;

  double chem = baryon * alphaB;

  // pimunu LRF components
  double pixx_LRF = pimunu.pixx_LRF;
  double piyy_LRF = pimunu.piyy_LRF;
  double pizz_LRF = pimunu.pizz_LRF;
  double pixz_LRF = pimunu.pixz_LRF;


  double pi_magnitude = pimunu.pi_magnitude;

  double pixx = pixx_LRF;
  double piyy = piyy_LRF;
  double pixy = pimunu.pixy_LRF;
  double pixz = pixz_LRF;
  double piyz = pimunu.piyz_LRF;


  // diagonalized components and amplitudes of the azimuthal square wave approximation
  double pizz_D = pimunu.pizz_D;
  double piperp_plus = pimunu.piperp_plus;
  double piperp_minus = pimunu.piperp_minus;

  // temporary
  //azi_plus = 1.0;
  //azi_minus = 1.0;


  // df coefficients
  double c0 = df.c0;
  double c1 = df.c1;
  double c2 = df.c2;
  double shear14_coeff = df.shear14_coeff;

  double F = df.F;
  double G = df.G;
  double betabulk = df.betabulk;
  double betapi = df.betapi;

  double lambda = df.lambda;
  double z = df.z;
  double delta_lambda = df.delta_lambda;
  double delta_z = df.delta_z;

  // feqmod terms
  double mbar_mod = mass / T_mod;
  double chem_mod;
  double bulk_mod;

  double equilibrium_density;
  double linearized_density;
  double modified_density;
  double renorm;

  if(df_mode == 3)
  {
    chem_mod = baryon * alphaB_mod;
    bulk_mod = bulkPi / (3.0 * betabulk);

    // compute the equilibrium, linearized and modified densities
    const int laguerre_pts = laguerre->points;
    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];
    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    double T2 = T * T;

    double neq_fact = T2 * T / two_pi2_hbarC3;
    double J20_fact = T * neq_fact;
    double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

    equilibrium_density = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, alphaB, baryon, sign);

    double J10 = 0.0;
    if(include_baryon)
    {
      J10 = neq_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, alphaB, baryon, sign);
    }
    double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, laguerre_pts, mbar, alphaB, baryon, sign);

    double bulk_density = (equilibrium_density + (baryon * J10 * G) + (J20 * F / T2)) / betabulk;

    linearized_density = equilibrium_density  +  bulkPi * bulk_density;
    modified_density = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar_mod, alphaB_mod, baryon, sign);

    renorm = linearized_density / modified_density;
  }
  else if(df_mode == 4)
  {
    chem_mod = 0.0;
    bulk_mod = lambda;

    const int laguerre_pts = laguerre->points;
    double * pbar_root1 = laguerre->root[1];
    double * pbar_weight1 = laguerre->weight[1];

    double neq_fact = T * T * T / two_pi2_hbarC3;

    equilibrium_density = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, 0.0, 0.0, sign);
    renorm = z;
  }

  double detA_ratio = detA_bulk / detA;

  // dsigma terms
  double ds_time = dsigma.dsigmat_LRF;
  double ds_space = dsigma.dsigma_space;
  double ds_time_over_ds_space = ds_time / ds_space;
  double ds_max = dsigma.dsigma_magnitude;

  // rotation angles
  double costheta_ds = dsigma.costheta_LRF;
  double costheta_ds2 = costheta_ds * costheta_ds;
  double sintheta_ds = sqrt(fabs(1.0 - costheta_ds2));

  double shear_outflow = (0.5 * piyy_LRF  +  (pixx_LRF + 0.5 * piyy_LRF) * (2.0 * costheta_ds2 - 1.0) -  2.0 * pixz_LRF * sintheta_ds * costheta_ds);

  // mean particle number of a given species emitted
  // from freezeout cell with outflow and df corrections
  double particle_number = 0.0;

  switch(df_mode)
  {
    /*
    case 0:
    {
      if(!outflow || ds_time >= ds_space)
      {
        const int laguerre_pts = laguerre->points;
        double * pbar_root1 = laguerre->root[1];
        double * pbar_weight1 = laguerre->weight[1];

        double neq_fact = T * T * T / two_pi2_hbarC3;
        double neq = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, alphaB, baryon, sign);

        particle_number = ds_time * neq;
      }
      else
      {
        double dsx = dsigma.dsigmax_LRF;
        double dsy = dsigma.dsigmay_LRF;
        double dsz = dsigma.dsigmaz_LRF;

        double ds_perp = sqrt(dsx * dsx + dsy * dsy); // these should be stored in the dsigma class
        //double phi_perp = atan2(dsy / dsx);
        //if(phi_perp < 0.0) phi_perp += 2.0 * M_PI;
        double ds_perp_amp = 2.0 * ds_perp / M_PI;

        for(int i = 0; i < legendre_pts; i++)
        {
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];
          double Ebar = sqrt(pbar * pbar +  mbar_squared);

          double feq = 1.0 / (exp(Ebar - chem) + sign);

          //double integrand = feq * compute_outflow_integrand(ds_time, ds_perp_amp, dsz, pbar, Ebar);

          double integrand = feq * compute_outflow_integrand_2(ds_time, ds_perp_amp, pbar, Ebar);

          particle_number += weight * integrand;
        }
        particle_number *= degeneracy * T * T * T / four_pi2_hbarC3;
      }
      break;
    }
    */
    case 1: // 14 moment
    {
      if(!outflow || ds_time >= ds_space || sample_with_max_volume)
      {
        // radial momentum pbar integral (pbar = p / T)
        for(int i = 0; i < legendre_pts; i++)
        {
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar = sqrt(pbar * pbar +  mbar_squared);

          double p = pbar * T;
          double E = Ebar * T;

          double feq = 1.0 / (exp(Ebar - chem) + sign);
          double feqbar = 1.0 - sign * feq;

          double df_bulk = feqbar * ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2 - c0) * E) * E) * bulkPi;

          /*
          // rms regulation method
          double df_shear_rms = sqrt(2.0 / 15.0) * feqbar * p * p * pi_magnitude / shear14_coeff;
          double df = max(-1.0 + 0.5 * df_shear_rms, min(df_bulk, 1.0 - 0.5 * df_shear_rms));
          particle_number += 2.0 * weight * feq * (1.0 + df);
          */

          // new regulation method
          double g = feqbar * p * p / shear14_coeff;
          double h = df_bulk;

          double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
          double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

          particle_number += weight * (f_plus + f_minus);

        } // i
        if(sample_with_max_volume)
        {
          particle_number *= ds_max * degeneracy * T * T * T / four_pi2_hbarC3;
        }
        else
        {
          particle_number *= ds_time * degeneracy * T * T * T / four_pi2_hbarC3;
        }

      } // timelike cell or max volume cell
      else
      {
        // radial momentum pbar integral (pbar = p / T)
        for(int i = 0; i < legendre_pts; i++)
        {
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar = sqrt(pbar * pbar +  mbar_squared);
          double E = Ebar * T;
          double p = pbar * T;

          double feq = 1.0 / (exp(Ebar - chem) + sign);
          double feqbar = 1.0 - sign * feq;

          double costheta_star = min(1.0, Ebar * ds_time_over_ds_space / pbar);
          double costheta_star2 = costheta_star * costheta_star;
          double costheta_star4 = costheta_star2 * costheta_star2;

          double df_bulk = feqbar * ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2 - c0) * E) * E) * bulkPi;

          double g = feqbar * p * p / shear14_coeff;
          double h = df_bulk;

          double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
          double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

          // take the isotropic average wpt polar angle
          double fiso = (f_plus + f_minus) / 2.0;

          double fbulk = feq * max(0.0, min(1.0, 1.0 + df_bulk));

          double favg = (fiso + fbulk) / 2.0;

          // include the df_shear contribution by itself
          double df_shear_time = 0.5 * feq * g * costheta_star * (1.0 - costheta_star2) * shear_outflow;
          double df_shear_space = 0.125 * feq * g * (1.0  +  2.0 * costheta_star2  -  3.0 * costheta_star4) * shear_outflow;

          double f_time = fiso * (1.0 + costheta_star)  +  df_shear_time;
          double f_space = 0.5 * fiso * (costheta_star2 - 1.0)  +  df_shear_space;

          double integrand = ds_time * f_time  -  ds_space * pbar / Ebar * f_space;

          particle_number += weight * integrand;


          //double costheta_star3 = costheta_star2 * costheta_star;
          //double costheta_star4 = costheta_star3 * costheta_star;
          //double z = costheta_star;
          //double z2 = z * z;


          //    df_shear_time_mean = z*(1-z)*(piyy + (2*pixx + piyy)*(2*costheta_ds2 - 1) - 4*pixz*sintheta_ds*costheta_ds)/4.;

          //    df_shear_space_mean = -(1 + 3*z*z)*(piyy + (2*pixx + piyy)*(2*costheta_ds2 - 1) - 4*pixz*sintheta_ds*costheta_ds)/8.;

          //    df_shear_time_sq = (256*(pow(pixy,2) + pow(pixz,2) + pow(piyz,2)) - 4*(-1 + z)*z*(4*pow(pixy,2)*(-1 + 9*pow(z,2)) + 4*pow(piyz,2)*(-1 + 9*pow(z,2)) - pow(pixz,2)*(19 + 9*pow(z,2))) +
          // 4*pow(pixx,2)*(64 + (-1 + z)*z*(19 + 9*pow(z,2))) + 4*pixx*piyy*(64 + (-1 + z)*z*(19 + 9*pow(z,2))) + pow(piyy,2)*(256 + (-1 + z)*z*(-119 + 171*pow(z,2))) +
          // 15*(-1 + z)*z*(4*(2*pixx*piyy + pow(piyy,2) + 4*pow(piyz,2) + (3*piyy*(2*pixx + piyy) - 4*pow(piyz,2))*pow(z,2) + 4*pow(pixy,2)*(-1 + pow(z,2)))*(2*pow(costheta_ds, 2)-1) +
          //    (2*pixx - 2*pixz + piyy)*(2*(pixx + pixz) + piyy)*(-3 + 7*pow(z,2))*(pow(sintheta_ds, 4)+pow(costheta_ds, 4)-6.0*pow(sintheta_ds*costheta_ds,2)) - 8*(4*pixy*piyz*(-1 + pow(z,2)) + pixz*(piyy + 3*piyy*pow(z,2)))*Sin(2*\[Theta]) -
          //    4*pixz*(2*pixx + piyy)*(-3 + 7*pow(z,2))*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1)))/960.;


          //    // for space divided out (-1 + pow(z,2))

          //    df_shear_space_sq = (60*pow(pixx,2) + 48*pow(pixy,2) + 60*pow(pixz,2) + 60*pixx*piyy + 45*pow(piyy,2) + 48*pow(piyz,2) + 24*pow(pixx,2)*pow(z,2) + 24*pow(pixz,2)*pow(z,2) +
          // 24*pixx*piyy*pow(z,2) - 30*pow(piyy,2)*pow(z,2) + 12*pow(pixx,2)*pow(z,4) - 48*pow(pixy,2)*pow(z,4) + 12*pow(pixz,2)*pow(z,4) + 12*pixx*piyy*pow(z,4) +
          // 57*pow(piyy,2)*pow(z,4) - 48*pow(piyz,2)*pow(z,4) + 4*(3*pow(piyy,2) + 4*pow(piyz,2) + 6*pow(piyy,2)*pow(z,2) + 16*pow(piyz,2)*pow(z,2) + 15*pow(piyy,2)*pow(z,4) -
          //    20*pow(piyz,2)*pow(z,4) + 4*pow(pixy,2)*(-1 - 4*pow(z,2) + 5*pow(z,4)) + 6*pixx*piyy*(1 + 2*pow(z,2) + 5*pow(z,4)))*(2*pow(costheta_ds, 2)-1) +
          // (4*pow(pixx,2) - 4*pow(pixz,2) + 4*pixx*piyy + pow(piyy,2))*(-1 - 10*pow(z,2) + 35*pow(z,4))*(pow(sintheta_ds, 4)+pow(costheta_ds, 4)-6.0*pow(sintheta_ds*costheta_ds,2)) - 24*pixz*piyy*Sin(2*\[Theta]) + 32*pixy*piyz*Sin(2*\[Theta]) -
          // 48*pixz*piyy*pow(z,2)*Sin(2*\[Theta]) + 128*pixy*piyz*pow(z,2)*Sin(2*\[Theta]) - 120*pixz*piyy*pow(z,4)*Sin(2*\[Theta]) - 160*pixy*piyz*pow(z,4)*Sin(2*\[Theta]) + 8*pixx*pixz*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1) +
          // 4*pixz*piyy*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1) + 80*pixx*pixz*pow(z,2)*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1) + 40*pixz*piyy*pow(z,2)*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1) - 280*pixx*pixz*pow(z,4)*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1) - 140*pixz*piyy*pow(z,4)*4.0*sintheta_ds*costheta_ds*(2.0*costheta_ds2-1))/192.;


          //    double df_shear_time_std = sqrt(df_shear_time_sq - df_shear_time_mean * df_shear_time_mean);
          //    double df_shear_space_std = sqrt(df_shear_space_sq - df_shear_space_mean * df_shear_space_mean);

          // (phi, costheta) integrated feq + df time and space terms

          //double df_bulk = feqbar * ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2 - c0) * E) * E) * bulkPi;
          //double df_shear_rms = sqrt(2.0 / 15.0) * feqbar * p * p * pi_magnitude / shear14_coeff;

          // regulate df isotropic term (df_shear_rms refines the bounds averaged wpt angles)
          //double df_isotropic = df_bulk;

          //df_isotropic = max(-1.0 + 0.25 * df_shear_rms, min(df_isotropic, 1.0 - 0.25 * df_shear_rms));

          //double f_isotropic = feq * (1.0 + df_isotropic); // regulated isotropic part of distribution


          //double df_shear_time = 0.5 * feq * feqbar / shear14_coeff * p * p * costheta_star * (1.0 - costheta_star2) * shear_outflow;

          //double df_shear_space = 0.5 * feq * feqbar / shear14_coeff * p * p * costheta_star * 0.25 * (1.0  +  2.0 * costheta_star2  -  3.0 * costheta_star4) * shear_outflow;

          //double df_shear_time = 0.5 * feq * feqbar / shear14_coeff * p * p * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star - costheta_star3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star3) / 3.0)  +  2.0 * pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);

          //double df_shear_space = 0.5 * feq * feqbar / shear14_coeff * p * p * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star2 - 1.0) * (costheta_star2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star4 - 1.0))  +  0.5 * pixz_LRF * (3.0 * costheta_star4 -  2.0 * costheta_star2  -  1.0) * sintheta_ds * costheta_ds);

          //double f_time = f_isotropic * (1.0 + costheta_star)  +  0.0*df_shear_time;
          //double f_space = 0.5 * f_isotropic * (costheta_star2 - 1.0)  +  0.0*df_shear_space;

          //double f_max = feq * (2.0 - 0.5 * df_shear_rms);
          //double f_min = 0.5 * feq * df_shear_rms;

          // double f_max = 2.0 * feq;
          // double f_min = 0.0;

          // double f_time_max = f_max * (1.0 + costheta_star);
          // double f_space_min = 0.5 * f_max * (costheta_star2 - 1.0);

          // double f_time_min = f_min * (1.0 + costheta_star);
          // double f_space_max = 0.5 * f_min * (costheta_star2 - 1.0);

          // f_time = max(f_time_min, min(f_time, f_time_max));
          // f_space = max(f_space_min, min(f_space, f_space_max));

          //double integrand = ds_time * f_time  -  ds_space * pbar / Ebar * f_space;

          //double integrand_max = f_max * (ds_time * (1.0 + costheta_star)  -  0.5 * ds_space * (costheta_star2 - 1.0) * pbar / Ebar);
          //double integrand_min = f_min * (ds_time * (1.0 + costheta_star)  -  0.5 * ds_space * (costheta_star2 - 1.0) * pbar / Ebar);
          //integrand = max(integrand_min, min(integrand, integrand_max));

          //particle_number += weight * integrand;

        } // i

        particle_number *= degeneracy * T * T * T / four_pi2_hbarC3;
      } // spacelike cell

      break;
    }
    case 2: // Chapman Enskog
    {
      chapman_enskog:

      if(!outflow || ds_time >= ds_space || sample_with_max_volume)
      {
        // radial momentum pbar integral (pbar = p / T)
        for(int i = 0; i < legendre_pts; i++)
        {
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar = sqrt(pbar * pbar +  mbar_squared);
          double E = Ebar * T;

          double feq = 1.0 / (exp(Ebar - chem) + sign);
          double feqbar = 1.0 - sign * feq;

          double df_bulk = feqbar * (F / T * Ebar  +  baryon * G  +  (Ebar  -  mbar_squared / Ebar) / 3.0) * bulkPi / betabulk;

          /*
          // rms regulation method
          double df_shear_rms = sqrt(2.0 / 15.0) * feqbar * pbar * pbar * pi_magnitude / (2.0 * betapi * Ebar);
          double df = max(-1.0 + 0.5 * df_shear_rms, min(df_bulk, 1.0 - 0.5 * df_shear_rms));
          particle_number += 2.0 * weight * feq * (1.0 + df);
          */

          // new regulation method
          double g = feqbar * pbar * pbar / (2.0 * betapi * Ebar);
          double h = df_bulk;

          double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
          double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

          particle_number += weight * (f_plus + f_minus);

        } // i

        if(sample_with_max_volume)
        {
          particle_number *= ds_max * degeneracy * T * T * T / four_pi2_hbarC3;
        }
        else
        {
          particle_number *= ds_time * degeneracy * T * T * T / four_pi2_hbarC3;
        }

      } // timelike cell or max volume cell
      else
      {
        // radial momentum pbar integral (pbar = p / T)
        for(int i = 0; i < legendre_pts; i++)
        {
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar = sqrt(pbar * pbar +  mbar_squared);
          double E = Ebar * T;
          double p = pbar * T;

          double feq = 1.0 / (exp(Ebar - chem) + sign);
          double feqbar = 1.0 - sign * feq;

          double costheta_star = min(1.0, Ebar * ds_time_over_ds_space / pbar);
          double costheta_star2 = costheta_star * costheta_star;
          double costheta_star3 = costheta_star2 * costheta_star;
          double costheta_star4 = costheta_star2 * costheta_star2;

          double df_bulk = feqbar * (F / T * Ebar  +  baryon * G  +  (Ebar  -  mbar_squared / Ebar) / 3.0) * bulkPi / betabulk;

          double g = feqbar * pbar * pbar / (2.0 * betapi * Ebar);
          double h = df_bulk;

          double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
          double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

          // take the isotropic average wpt polar angle
          double fiso = (f_plus + f_minus) / 2.0;

          // include the df_shear contribution by itself
          double df_shear_time = 0.5 * feq * g * costheta_star * (1.0 - costheta_star2) * shear_outflow;
          double df_shear_space = 0.125 * feq * g * (1.0  +  2.0 * costheta_star2  -  3.0 * costheta_star4) * shear_outflow;

          double f_time = fiso * (1.0 + costheta_star) + 0.5*df_shear_time;
          double f_space = 0.5 * fiso * (costheta_star2 - 1.0) + 0.5*df_shear_space;

          double integrand = ds_time * f_time  -  ds_space * pbar / Ebar * f_space;

          particle_number += weight * integrand;

          /*
          double df_shear_rms = sqrt(2.0 / 15.0) * feqbar * pbar * pbar * pi_magnitude / (2.0 * betapi * Ebar);

          // regulate df isotropic term (df_shear_rms refines the bounds averaged wpt angles)
          double df_isotropic = df_bulk;

          df_isotropic = max(-1.0 + 0.25 * df_shear_rms, min(df_isotropic, 1.0 - 0.25 * df_shear_rms));

          double f_isotropic = feq * (1.0 + df_isotropic); // regulated isotropic part of distribution


          double df_shear_time = 0.25 * feq * feqbar / Ebar / betapi * pbar * pbar * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star - costheta_star3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star3) / 3.0)  +  2.0 * pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);

          double df_shear_space = 0.25 * feq * feqbar / Ebar / betapi * pbar * pbar * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star2 - 1.0) * (costheta_star2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star4 - 1.0))  +  0.5 * pixz_LRF * (3.0 * costheta_star4 -  2.0 * costheta_star2  -  1.0) * sintheta_ds * costheta_ds);


          double f_time = f_isotropic * (1.0 + costheta_star)  +  0.0*df_shear_time;
          double f_space = 0.5 * f_isotropic * (costheta_star2 - 1.0)  +  0.0*df_shear_space;

          //double f_max = feq * (2.0 - 0.5 * df_shear_rms);
          //double f_min = 0.5 * feq * df_shear_rms;

          // double f_time_max = f_max * (1.0 + costheta_star);
          // double f_space_min = 0.5 * f_max * (costheta_star2 - 1.0);

          // double f_time_min = f_min * (1.0 + costheta_star);
          // double f_space_max = 0.5 * f_min * (costheta_star2 - 1.0);

          // f_time = max(f_time_min, min(f_time, f_time_max));
          // f_space = max(f_space_min, min(f_space, f_space_max));

          double integrand = ds_time * f_time  -  ds_space * pbar / Ebar * f_space;

          //double integrand_max = f_max * (ds_time * (1.0 + costheta_star)  -  0.5 * ds_space * (costheta_star2 - 1.0) * pbar / Ebar);
          //double integrand_min = f_min * (ds_time * (1.0 + costheta_star)  -  0.5 * ds_space * (costheta_star2 - 1.0) * pbar / Ebar);

          //integrand = max(integrand_min, min(integrand, integrand_max));

          particle_number += weight * integrand;
          */

        } // i

        particle_number *= degeneracy * T * T * T / four_pi2_hbarC3;
      } // spacelike cell

      break;
    }
    case 3: // modified (Mike)
    {
      if(feqmod_breaks_down) goto chapman_enskog;

      if(!outflow || ds_time >= ds_space || sample_with_max_volume)
      {
        if(linearized_density < 0.0) printf("Error: linearized density is negative. Adjust the mass_pion0 parameter...\n");

        if(sample_with_max_volume)
        {
          particle_number = ds_max * linearized_density;
        }
        else
        {
          particle_number = ds_time * linearized_density;
        }

      } // timelike cell or max volume cell
      else
      {
        // radial momentum pbar integral (pbar = p / T)
        for(int i = 0; i < legendre_pts; i++)
        {
          // old method
          /*
          double pbar_mod = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar_mod = sqrt(pbar_mod * pbar_mod  +  mbar_mod * mbar_mod);

          double feqmod = 1.0 / (exp(Ebar_mod - chem_mod) + sign);

          double mbar_bulk = mbar_mod / (1.0 + bulk_mod);
          double Ebar_bulk = sqrt(pbar_mod * pbar_mod  +  mbar_bulk * mbar_bulk);

          // fixed minor bug on 3/21/19
          double costheta_star_mod = min(1.0, Ebar_bulk * ds_time_over_ds_space / pbar_mod);
          double costheta_star_mod2 = costheta_star_mod * costheta_star_mod;

          particle_number += weight * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * pbar_mod / Ebar_bulk * (costheta_star_mod * costheta_star_mod - 1.0)) * feqmod;
          */


          // new method

          double pbar_mod = pbar_root[i];
          double weight = pbar_weight[i];

          double mbar_bulk = mbar_mod / (1.0 + bulk_mod);

          double Ebar_mod = sqrt(pbar_mod * pbar_mod  +  mbar_mod * mbar_mod);
          double Ebar_bulk = sqrt(pbar_mod * pbar_mod  +  mbar_bulk * mbar_bulk);

          // fixed minor bug on 3/21/19
          double costheta_star_mod = min(1.0, Ebar_bulk * ds_time_over_ds_space / pbar_mod);
          double costheta_star_mod2 = costheta_star_mod * costheta_star_mod;
          double costheta_star_mod3 = costheta_star_mod2 * costheta_star_mod;
          double costheta_star_mod4 = costheta_star_mod3 * costheta_star_mod;

          double feqmod = 1.0 / (exp(Ebar_mod - chem_mod) + sign);

          double feqmodbar = 1.0 - sign * feqmod;

          //double pbar = pbar_mod * (1.0 + bulk_mod);
          //double Ebar = sqrt(pbar * pbar  +  mbar_mod * mbar_mod);
          //double feq = 1.0 / (exp(Ebar * T_mod / T - chem_mod) + sign);
          //double feqbar = 1.0 - sign * feq;
          //double df_shear_rms_mod = sqrt(2.0 / 15.0) * feq * feqbar * pbar * pbar * pi_magnitude / (2.0 * betapi * Ebar) * T_mod / T;

          double df_shear_rms_mod = sqrt(2.0 / 15.0) * feqmod * feqmodbar * pbar_mod * pbar_mod * pi_magnitude / (2.0 * (1.0 + bulk_mod) * betapi * Ebar_mod);

          double fmod_isotropic = feqmod;

          //fmod_isotropic = max(0.5 * df_shear_rms_mod, min(fmod_isotropic, 2.0 * feqmod  -  0.5 * df_shear_rms_mod));


          double df_shear_mod_time = 0.25 * feqmod * feqmodbar / Ebar_mod / betapi / (1.0 + bulk_mod) * pbar_mod * pbar_mod * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star_mod)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star_mod3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star_mod - costheta_star_mod3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star_mod)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star_mod3) / 3.0)  +  2.0 * pixz_LRF * costheta_star_mod * (costheta_star_mod2 - 1.0) * sintheta_ds * costheta_ds);

          double df_shear_mod_space = 0.25 * feqmod * feqmodbar / Ebar_mod / betapi / (1.0 + bulk_mod) * pbar_mod * pbar_mod * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star_mod2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star_mod4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star_mod2 - 1.0) * (costheta_star_mod2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star_mod2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star_mod4 - 1.0))  +  0.5 * pixz_LRF * (3.0 * costheta_star_mod4 -  2.0 * costheta_star_mod2  -  1.0) * sintheta_ds * costheta_ds);


          double f_time = fmod_isotropic * (1.0 + costheta_star_mod)  +  df_shear_mod_time;
          double f_space = 0.5 * fmod_isotropic * (costheta_star_mod2 - 1.0)  +  df_shear_mod_space;

          //double f_max = 2.0 * feqmod  -  0.5 * df_shear_rms_mod;
          //double f_min = 0.5 * df_shear_rms_mod;

          // double f_time_max = f_max * (1.0 + costheta_star_mod);
          // double f_space_max = 0.5 * f_min * (costheta_star_mod2 - 1.0);

          // double f_time_min = f_min * (1.0 + costheta_star_mod);
          // double f_space_min = 0.5 * f_max * (costheta_star_mod2 - 1.0);

          // f_time = max(f_time_min, min(f_time, f_time_max));
          // f_space = max(f_space_min, min(f_space, f_space_max));

          double integrand = ds_time * f_time  -  ds_space * f_space * pbar_mod / Ebar_bulk;

          //double integrand_max = f_max * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * (costheta_star_mod2 - 1.0) * pbar_mod / Ebar_bulk);
          //double integrand_min = f_min * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * (costheta_star_mod2 - 1.0) * pbar_mod / Ebar_bulk);
          //integrand = max(integrand_min, min(integrand, integrand_max));

          particle_number += weight * integrand;


          /*
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar_mod = sqrt(pbar * pbar  +  mbar_mod * mbar_mod);

          double costheta_star_mod = min(1.0, Ebar_mod * ds_time_over_ds_space / pbar);
          double costheta_star_mod2 = costheta_star_mod * costheta_star_mod;

          double pbar_bulk = pbar / (1.0 + bulk_mod);
          double Ebar_bulk = sqrt(pbar_bulk * pbar_bulk  +  mbar_mod * mbar_mod);

          double feqmod = 1.0 / (exp(Ebar_bulk - chem_mod) + sign);

          particle_number += weight * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * pbar / Ebar_mod * (costheta_star_mod2 - 1.0)) * feqmod;
          */
        } // i

        //particle_number *= detA_ratio * degeneracy * renorm * T_mod * T_mod * T_mod / four_pi2_hbarC3;
        particle_number *= degeneracy * renorm * T_mod * T_mod * T_mod / four_pi2_hbarC3;

        // I may need detA or not depending on how I write the formula with shear
        //particle_number *= degeneracy * renorm * T_mod * T_mod * T_mod / four_pi2_hbarC3 / pow(1.0 + bulk_mod, 3);
      } // spacelike cell

      break;
    }
    case 4: // modified (Jonah)
    {
      if(!feqmod_breaks_down)
      {
        if(!outflow || ds_time >= ds_space || sample_with_max_volume)
        {
          if(z < 0.0) printf("Error: z is negative\n");

          if(sample_with_max_volume)
          {
            particle_number = ds_max * z * equilibrium_density;
          }
          else
          {
            particle_number = ds_time * z * equilibrium_density;
          }

        } // timelike cell or max volume cell
        else
        {
          // radial momentum pbar integral (pbar = p / T)
          for(int i = 0; i < legendre_pts; i++)
          {
            /*
            double pbar_mod = pbar_root[i];
            double weight = pbar_weight[i];

            double Ebar_mod = sqrt(pbar_mod * pbar_mod  +  mbar_mod * mbar_mod);

            double feqmod = 1.0 / (exp(Ebar_mod - chem_mod) + sign);


            double mbar_bulk = mbar_mod / (1.0 + bulk_mod);
            double Ebar_bulk = sqrt(pbar_mod * pbar_mod  +  mbar_bulk * mbar_bulk);

            double costheta_star_mod = min(1.0, Ebar_bulk * ds_time_over_ds_space / pbar_mod);
            double costheta_star_mod2 = costheta_star_mod * costheta_star_mod;

            particle_number += weight * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * pbar_mod / Ebar_bulk * (costheta_star_mod2 - 1.0)) * feqmod;
            */

            // new method

          double pbar_mod = pbar_root[i];
          double weight = pbar_weight[i];

          double mbar_bulk = mbar_mod / (1.0 + bulk_mod);

          double Ebar_mod = sqrt(pbar_mod * pbar_mod  +  mbar_mod * mbar_mod);
          double Ebar_bulk = sqrt(pbar_mod * pbar_mod  +  mbar_bulk * mbar_bulk);

          // fixed minor bug on 3/21/19
          double costheta_star_mod = min(1.0, Ebar_bulk * ds_time_over_ds_space / pbar_mod);
          double costheta_star_mod2 = costheta_star_mod * costheta_star_mod;
          double costheta_star_mod3 = costheta_star_mod2 * costheta_star_mod;
          double costheta_star_mod4 = costheta_star_mod3 * costheta_star_mod;

          double feqmod = 1.0 / (exp(Ebar_mod - chem_mod) + sign);

          double feqmodbar = 1.0 - sign * feqmod;


          double df_shear_rms_mod = sqrt(2.0 / 15.0) * feqmod * feqmodbar * pbar_mod * pbar_mod * pi_magnitude / (2.0 * (1.0 + bulk_mod) * betapi * Ebar_mod);

          double fmod_isotropic = feqmod;

          //fmod_isotropic = max(0.5 * df_shear_rms_mod, min(fmod_isotropic, 2.0 * feqmod  -  0.5 * df_shear_rms_mod));


          double df_shear_mod_time = 0.25 * feqmod * feqmodbar / Ebar_mod / betapi / (1.0 + bulk_mod) * pbar_mod * pbar_mod * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star_mod)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star_mod3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star_mod - costheta_star_mod3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star_mod)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star_mod3) / 3.0)  +  2.0 * pixz_LRF * costheta_star_mod * (costheta_star_mod2 - 1.0) * sintheta_ds * costheta_ds);

          double df_shear_mod_space = 0.25 * feqmod * feqmodbar / Ebar_mod / betapi / (1.0 + bulk_mod) * pbar_mod * pbar_mod * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star_mod2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star_mod4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star_mod2 - 1.0) * (costheta_star_mod2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star_mod2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star_mod4 - 1.0))  +  0.5 * pixz_LRF * (3.0 * costheta_star_mod4 -  2.0 * costheta_star_mod2  -  1.0) * sintheta_ds * costheta_ds);


          double f_time = fmod_isotropic * (1.0 + costheta_star_mod)  +  df_shear_mod_time;
          double f_space = 0.5 * fmod_isotropic * (costheta_star_mod2 - 1.0)  +  df_shear_mod_space;

          //double f_max = 2.0 * feqmod  -  0.5 * df_shear_rms_mod;
          //double f_min = 0.5 * df_shear_rms_mod;

          // double f_time_max = f_max * (1.0 + costheta_star_mod);
          // double f_space_max = 0.5 * f_min * (costheta_star_mod2 - 1.0);

          // double f_time_min = f_min * (1.0 + costheta_star_mod);
          // double f_space_min = 0.5 * f_max * (costheta_star_mod2 - 1.0);

          // f_time = max(f_time_min, min(f_time, f_time_max));
          // f_space = max(f_space_min, min(f_space, f_space_max));

          double integrand = ds_time * f_time  -  ds_space * f_space * pbar_mod / Ebar_bulk;

          //double integrand_max = f_max * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * (costheta_star_mod2 - 1.0) * pbar_mod / Ebar_bulk);
          //double integrand_min = f_min * (ds_time * (1.0 + costheta_star_mod)  -  0.5 * ds_space * (costheta_star_mod2 - 1.0) * pbar_mod / Ebar_bulk);
          //integrand = max(integrand_min, min(integrand, integrand_max));

          particle_number += weight * integrand;

          } // i

          particle_number *= degeneracy * renorm * T_mod * T_mod * T_mod / four_pi2_hbarC3;
        }
      }
      else
      {
        if(!outflow || ds_time >= ds_space)
        {
          // radial momentum pbar integral (pbar = p / T)
          for(int i = 0; i < legendre_pts; i++)
          {
            double pbar = pbar_root[i];
            double weight = pbar_weight[i];

            double Ebar = sqrt(pbar * pbar +  mbar_squared);

            double feq = 1.0 / (exp(Ebar - chem) + sign);
            double feqbar = 1.0 - sign * feq;

            // df_bulk correction key difference from chapman enskog
            //::::::::::::::::::::::::::::::::::::::::::
            double df_bulk = delta_z  - 3.0 * delta_lambda  +  feqbar * delta_lambda * (Ebar  -  mbar_squared / Ebar);
            //::::::::::::::::::::::::::::::::::::::::::

            // rms regulation method
            /*
            double df_shear_rms = sqrt(2.0 / 15.0) * feqbar * pbar * pbar * pi_magnitude / (2.0 * betapi * Ebar);
            double df = max(-1.0 + 0.5 * df_shear_rms, min(df_bulk, 1.0 - 0.5 * df_shear_rms));
            particle_number += 2.0 * weight * feq * (1.0 + df);
            */

            // new regulation method
            double g = feqbar * pbar * pbar / (2.0 * betapi * Ebar);
            double h = df_bulk;

            double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
            double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

            particle_number += weight * (f_plus + f_minus);

          } // i

          particle_number *= ds_time * degeneracy * T * T * T / four_pi2_hbarC3;
        } // timelike cell
        else
        {
          // radial momentum pbar integral (pbar = p / T)
          for(int i = 0; i < legendre_pts; i++)
          {
            double pbar = pbar_root[i];
            double weight = pbar_weight[i];

            double Ebar = sqrt(pbar * pbar +  mbar_squared);
            double E = Ebar * T;
            double p = pbar * T;

            double feq = 1.0 / (exp(Ebar - chem) + sign);
            double feqbar = 1.0 - sign * feq;

            double costheta_star = min(1.0, Ebar * ds_time_over_ds_space / pbar);
            double costheta_star2 = costheta_star * costheta_star;
            double costheta_star3 = costheta_star2 * costheta_star;
            double costheta_star4 = costheta_star3 * costheta_star;

            // df_bulk correction is key difference from chapman enskog
            //::::::::::::::::::::::::::::::::::::::::::
            double df_bulk = delta_z  -  3.0 * delta_lambda  +  feqbar * delta_lambda * (Ebar  -  mbar_squared / Ebar);
            //::::::::::::::::::::::::::::::::::::::::::

            double df_shear_rms = 0.5 * sqrt(2.0 / 15.0) * feqbar * pbar * pbar * pi_magnitude / betapi / Ebar;

            // regulate isotropic df term (df_shear_rms refines the bounds averaged wpt angles)
            double df_isotropic = df_bulk;

            df_isotropic = max(-1.0 + 0.5 * df_shear_rms, min(df_isotropic, 1.0 - 0.5 * df_shear_rms));

            double f_isotropic = feq * (1.0 + df_isotropic); // regulated isotropic part of distribution


            // (phi, costheta) integrated feq + df time and space terms
            double df_shear_time = 0.25 * feq * feqbar / Ebar / betapi * pbar * pbar * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star - costheta_star3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star3) / 3.0)  +  2.0 * pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);

            double df_shear_space = 0.25 * feq * feqbar / Ebar / betapi * pbar * pbar * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star2 - 1.0) * (costheta_star2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star4 - 1.0))  +  0.5 * pixz_LRF * (3.0 * costheta_star4 -  2.0 * costheta_star2  -  1.0) * sintheta_ds * costheta_ds);


            double f_time = f_isotropic * (1.0 + costheta_star)  +  0.0*df_shear_time;
            double f_space = 0.5 * f_isotropic * (costheta_star2 - 1.0)  +  0.0*df_shear_space;


            //double f_max = feq * (2.0 - 0.5 * df_shear_rms);
            //double f_min = 0.5 * feq * df_shear_rms;

            // double f_time_max = f_max * (1.0 + costheta_star);
            // double f_space_min = 0.5 * f_max * (costheta_star2 - 1.0);

            // double f_time_min = f_min * (1.0 + costheta_star);
            // double f_space_max = 0.5 * f_min * (costheta_star2 - 1.0);

            // f_time = max(f_time_min, min(f_time, f_time_max));
            // f_space = max(f_space_min, min(f_space, f_space_max));

            double integrand = ds_time * f_time  -  ds_space * pbar / Ebar * f_space;

            // double integrand_max = f_max * (ds_time * (1.0 + costheta_star)  -  0.5 * ds_space * (costheta_star2 - 1.0) * pbar / Ebar);
            // double integrand_min = f_min * (ds_time * (1.0 + costheta_star)  -  0.5 * ds_space * (costheta_star2 - 1.0) * pbar / Ebar);
            // integrand = max(integrand_min, min(integrand, integrand_max));

            particle_number += weight * integrand;

          } // i

          particle_number *= degeneracy * T * T * T / four_pi2_hbarC3;
        } // spacelike cell
      }

      break;
    }
    default:
    {
      printf("\nParticle density outflow error: please set df_mode = (1,2,3,4)\n");
      exit(-1);
    }
  } // df_mode

  if(particle_number < 0.0) printf("Error: particle number is negative\n");

  return particle_number;
}


double compute_df_weight(LRF_Momentum pLRF, double mass_squared, double sign, double baryon, double T, double alphaB, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, deltaf_coefficients df, double baryon_enthalpy_ratio, int df_mode)
{
  // pLRF components
  double E = pLRF.E;
  double px = pLRF.px;
  double py = pLRF.py;
  double pz = pLRF.pz;

  // pimunu LRF components
  double pixx = pimunu.pixx_LRF;
  double pixy = pimunu.pixy_LRF;
  double pixz = pimunu.pixz_LRF;
  double piyy = pimunu.piyy_LRF;
  double piyz = pimunu.piyz_LRF;
  double pizz = pimunu.pizz_LRF;

  double pimunu_pmu_pnu = px * px * pixx  +  py * py * piyy  +  pz * pz * pizz  +  2.0 * (px * py * pixy  +  px * pz * pixz  +  py * pz * piyz);

  // Vmu LRF components
  double Vx = Vmu.Vx_LRF;
  double Vy = Vmu.Vy_LRF;
  double Vz = Vmu.Vz_LRF;

  double Vmu_pmu = - (px * Vx  +  py * Vy  +  pz * Vz);

  // total df correction
  double df_tot = 0.0;

  switch(df_mode)
  {
    case 1: // 14 moment
    {
      double chem = baryon * alphaB;
      double feqbar = 1.0 - sign / (exp(E/T - chem) + sign);

      double c0 = df.c0;
      double c1 = df.c1;
      double c2 = df.c2;
      double c3 = df.c3;
      double c4 = df.c4;
      double shear14_coeff = df.shear14_coeff;

      double df_shear = pimunu_pmu_pnu / shear14_coeff;
      double df_bulk = ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2  -  c0) * E) * E) * bulkPi;
      double df_diff = (baryon * c3  +  c4 * E) * Vmu_pmu;

      df_tot = feqbar * (df_shear + df_bulk + df_diff);
      break;
    }
    case 2: // Chapman Enskog
    case 3: // Modified (Mike) (switch to linearized df)
    {
      double chem = baryon * alphaB;
      double feqbar = 1.0 - sign / (exp(E/T - chem) + sign);

      double F = df.F;
      double G = df.G;
      double betabulk = df.betabulk;
      double betaV = df.betaV;
      double betapi = df.betapi;

      double df_shear = pimunu_pmu_pnu / (2.0 * E * betapi * T);
      double df_bulk = (baryon * G  +  F * E / T / T  +  (E  -  mass_squared / E) / (3.0 * T)) * bulkPi / betabulk;
      double df_diff = (baryon_enthalpy_ratio  -  baryon / E) * Vmu_pmu / betaV;

      df_tot = feqbar * (df_shear + df_bulk + df_diff);
      break;
    }
    case 4: // Modified (Jonah) (switch to linearized df)
    {
      double feqbar = 1.0 - sign / (exp(E/T) + sign);

      double delta_lambda = df.delta_lambda;
      double delta_z = df.delta_z;
      double betapi = df.betapi;

      double df_shear = feqbar * pimunu_pmu_pnu / (2.0 * E * betapi * T);
      double df_bulk = (delta_z  -  3.0 * delta_lambda)  +  feqbar * delta_lambda * (E  -  mass_squared / E) / T;

      df_tot = df_shear + df_bulk;
      break;
    }
    default:
    {
      printf("Error: use df_mode = (1,2,3,4)\n");
      exit(-1);
    }
  } // df_mode

  df_tot = max(-1.0, min(df_tot, 1.0));  // regulate |df| <= 1

  return (1.0 + df_tot) / 2.0;
}


LRF_Momentum sample_momentum(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, deltaf_coefficients df, double baryon_enthalpy_ratio, int df_mode, int sample_with_max_volume)
{
  // samples the local rest frame momentum from a
  // distribution with linear viscous corrections (either 14 moment or Chapman Enskog)

  double mass_squared = mass * mass;
  double chem = baryon * alphaB;

  // currently the momentum sampler does not work for photons and bosons with nonzero chemical potential
  // so non-equilibrium or electric / strange charge chemical potentials are not considered
  if(sign == -1.0 && chem != 0.0)
  {
    printf("Error: bosons have chemical potential. Exiting...\n");
    exit(-1);
  }

  // uniform distributions in (phi,costheta) on standby:
  uniform_real_distribution<double> phi_distribution(0.0, two_pi);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;

  // local rest frame momentum
  LRF_Momentum pLRF;

  bool rejected = true;

  if(mass == 0.0)
  {
    printf("Error: cannot sample photons with this method. Exiting...\n");
    exit(-1);
  }
  else if(mass / T < 1.008)
  {
    double weq_max = 1.0; // default value if don't need to rescale weq = exp(p/T) / (exp(E/T) + sign)

    if(mass / T < 0.8554 && sign == -1.0) weq_max = pion_thermal_weight_max(mass, T, chem);

    while(rejected)
    {
      *samples = (*samples) + 1;
      // draw (p,phi,costheta) from p^2.exp(-p/T).dp.dphi.dcostheta by sampling (r1,r2,r3)
      double r1 = 1.0 - canonical(generator);
      double r2 = 1.0 - canonical(generator);
      double r3 = 1.0 - canonical(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p = - T * (l1 + l2 + l3);
      double phi = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
      double costheta = (l1 - l2) / (l1 + l2);

      double E = sqrt(p * p  +  mass_squared);              // energy
      double sintheta = sqrt(1.0  -  costheta * costheta);  // sin(theta)

      // pLRF components
      pLRF.E = E;
      pLRF.px = p * sintheta * cos(phi);
      pLRF.py = p * sintheta * sin(phi);
      pLRF.pz = p * costheta;

      double pdsigma_Theta = max(0.0, E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

      // compute the weights
      double w_eq = exp(p/T) / (exp(E/T) + sign) / weq_max;
      double w_flux = pdsigma_Theta / (E * ds_magnitude);
      double w_visc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df, baryon_enthalpy_ratio, df_mode);

      if(sample_with_max_volume) w_flux = 1.0;

      double weight_light = w_eq * w_flux * w_visc;

      // check if 0 <= weight <= 1
      if(fabs(weight_light - 0.5) > 0.5) printf("Error: df weight_light = %lf out of bounds\n", weight_light);

      // check pLRF acceptance
      if(canonical(generator) < weight_light) break;

    } // rejection loop

  } // sampling pions

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // integrated weights
    std::vector<double> K_weight;
    K_weight.resize(3);

    K_weight[0] = mass_squared;    // heavy (sampled_distribution = 0)
    K_weight[1] = 2.0 * mass * T;  // moderate (sampled_distribution = 1)
    K_weight[2] = 2.0 * T * T;     // light (sampled_distribution = 2)

    // for some reason intel compiler didn't accept enum templates...
    //discrete_distribution<heavy_distribution> K_distribution(K_weight.begin(), K_weight.end());
    discrete_distribution<int> K_distribution(K_weight.begin(), K_weight.end());

    double k, phi, costheta;       // kinetic energy, angles

    while(rejected)
    {
      *samples = (*samples) + 1;

      //heavy_distribution sampled_distribution = K_distribution(generator);
      int sampled_distribution = K_distribution(generator);

      // select distribution to sample from based on integrated weights
      //if(sampled_distribution == heavy)
      if(sampled_distribution == 0)
      {
        // draw k from exp(-k/T).dk by sampling r1
        // sample direction uniformly
        double r1 = 1.0 - canonical(generator);

        k = - T * log(r1);
        phi = phi_distribution(generator);
        costheta = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      //else if(sampled_distribution == moderate)
      else if(sampled_distribution == 1)
      {
        // draw (k,phi) from k.exp(-k/T).dk.dphi by sampling (r1,r2)
        // sample costheta uniformly
        double r1 = 1.0 - canonical(generator);
        double r2 = 1.0 - canonical(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k = - T * (l1 + l2);
        phi = two_pi * l1 / (l1 + l2);
        costheta = costheta_distribution(generator);

      } // distribution 2 (moderately heavy)
      //else if(sampled_distribution == light)
      else if(sampled_distribution == 2)
      {
        // draw (k,phi,costheta) from k^2.exp(-k/T).dk.dphi.dcostheta by sampling (r1,r2,r3)
        double r1 = 1.0 - canonical(generator);
        double r2 = 1.0 - canonical(generator);
        double r3 = 1.0 - canonical(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k = - T * (l1 + l2 + l3);
        phi = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
        costheta = (l1 - l2) / (l1 + l2);

      } // distribution 3 (light)
      else
      {
        printf("Error: did not sample any K distribution\n");
        exit(-1);
      }

      double E = k + mass;                                    // energy
      double p = sqrt(E * E  -  mass_squared);                // momentum magnitude
      double sintheta = sqrt(1.0  -  costheta * costheta);    // sin(theta)

      // pLRF components
      pLRF.E = E;
      pLRF.px = p * sintheta * cos(phi);
      pLRF.py = p * sintheta * sin(phi);
      pLRF.pz = p * costheta;

      double pdsigma_Theta = max(0.0, E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

      // compute the weight
      double w_eq = p/E * exp(E/T - chem) / (exp(E/T - chem) + sign);
      double w_flux = pdsigma_Theta / (E * ds_magnitude);
      double w_visc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df, baryon_enthalpy_ratio, df_mode);

      if(sample_with_max_volume) w_flux = 1.0;

      double weight_heavy = w_eq * w_flux * w_visc;

      // check if 0 <= weight <= 1
      if(fabs(weight_heavy - 0.5) > 0.5) printf("Error: df weight_heavy = %f out of bounds\n", weight_heavy);

      // check pLRF acceptance
      if(canonical(generator) < weight_heavy) break;

    } // rejection loop

  } // sampling heavy hadrons

  *acceptances = (*acceptances) + 1;

  return pLRF;

}

LRF_Momentum rescale_momentum(LRF_Momentum pLRF_mod, double mass_squared, double baryon, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_mod, double bulk_mod, double diff_mod, double baryon_enthalpy_ratio)
{
    double E_mod = pLRF_mod.E;
    double px_mod = pLRF_mod.px;
    double py_mod = pLRF_mod.py;
    double pz_mod = pLRF_mod.pz;

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
    diff_mod *= (E_mod * baryon_enthalpy_ratio  +  baryon);

    LRF_Momentum pLRF;

    // local momentum transformation
    // p_i = A_ij * p_mod_j  +  E_mod * q_i  +  b * T * a_i
    pLRF.px = (1.0 + bulk_mod) * px_mod  +  shear_mod * (pixx * px_mod  +  pixy * py_mod  +  pixz * pz_mod)  +  diff_mod * Vx;
    pLRF.py = (1.0 + bulk_mod) * py_mod  +  shear_mod * (pixy * px_mod  +  piyy * py_mod  +  piyz * pz_mod)  +  diff_mod * Vy;
    pLRF.pz = (1.0 + bulk_mod) * pz_mod  +  shear_mod * (pixz * px_mod  +  piyz * py_mod  +  pizz * pz_mod)  +  diff_mod * Vz;
    pLRF.E = sqrt(mass_squared  +  pLRF.px * pLRF.px  +  pLRF.py * pLRF.py  +  pLRF.pz * pLRF.pz);

    return pLRF;
}


LRF_Momentum sample_momentum_feqmod(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T_mod, double alphaB_mod, Surface_Element_Vector dsigma, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_mod, double bulk_mod, double diff_mod, double baryon_enthalpy_ratio, int sample_with_max_volume)
{
  // samples the local rest frame momentum
  // from a modified equilibrium distribution (either Mike or Jonah)

  double mass_squared = mass * mass;
  double chem_mod = baryon * alphaB_mod;

  // currently the momentum sampler does not work for photons and bosons with nonzero chemical potential
  // so non-equilibrium or electric / strange charge chemical potentials are not considered
  if(sign == -1.0 && chem_mod != 0.0)
  {
    printf("Error: bosons have chemical potential. Exiting...\n");
    exit(-1);
  }

   // uniform distributions in (phi_mod, costheta_mod) on standby
  uniform_real_distribution<double> phi_distribution(0.0, two_pi);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;

  LRF_Momentum pLRF;                          // local rest frame momentum
  LRF_Momentum pLRF_mod;                      // modified local rest frame momentum

  bool rejected = true;

  if(mass == 0.0)
  {
    printf("Error: cannot sample photons with this method. Exiting...\n");
    exit(-1);
  }
  if(mass / T_mod < 1.008)
  {
    double w_mod_max = 1.0; // default value if don't rescale w_mod = exp(p'/T') / (exp(E'/T') - 1)

    // modified temperature increases, so it's possible pions are too light w/o rescaling w_mod
    if(mass / T_mod <= 0.8554 && sign == -1.0) w_mod_max = pion_thermal_weight_max(mass, T_mod, chem_mod);

    while(rejected)
    {
      *samples = (*samples) + 1;
      // draw (p_mod, phi_mod, costheta_mod) from p_mod^2.exp(-p_mod/T_mod).dp_mod.dphi_mod.dcostheta_mod by sampling (r1,r2,r3)
      double r1 = 1.0 - canonical(generator);
      double r2 = 1.0 - canonical(generator);
      double r3 = 1.0 - canonical(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p_mod = - T_mod * (l1 + l2 + l3);                        // modified momentum magnitude
      double phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);   // modified angles
      double costheta_mod = (l1 - l2) / (l1 + l2);

      double E_mod = sqrt(p_mod * p_mod  +  mass_squared);            // modified energy
      double sintheta_mod = sqrt(1.0 - costheta_mod * costheta_mod);  // sin(theta)

      // modified pLRF components
      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // local momentum transformation
      pLRF = rescale_momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_mod, bulk_mod, diff_mod, baryon_enthalpy_ratio);

      double pdsigma_Theta = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

      // compute the weight
      double w_mod = exp(p_mod/T_mod - chem_mod) / (exp(E_mod/T_mod - chem_mod) + sign) / w_mod_max;
      double w_flux = pdsigma_Theta / (pLRF.E * ds_magnitude);

      if(sample_with_max_volume) w_flux = 1.0;

      double weight_light = w_mod * w_flux;

      // check if 0 <= weight <= 1
      if(fabs(weight_light - 0.5) > 0.5) printf("Error: modified light weight = %f out of bounds\n", weight_light);

      // check pLRF acceptance
      if(canonical(generator) < weight_light) break;

    } // rejection loop

  } // sampling pions

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // modified integrated weights
    std::vector<double> K_mod_weight;
    K_mod_weight.resize(3);

    K_mod_weight[0] = mass_squared;         // heavy (sampled_distribution = 0)
    K_mod_weight[1] = 2.0 * mass * T_mod;   // moderate (sampled_distribution = 1)
    K_mod_weight[2] = 2.0 * T_mod * T_mod;  // light (sampled_distribution = 2)

    //discrete_distribution<heavy_distribution> K_mod_distribution(K_mod_weight.begin(), K_mod_weight.end());
    discrete_distribution<int> K_mod_distribution(K_mod_weight.begin(), K_mod_weight.end());

    double k_mod, phi_mod, costheta_mod;    // modified kinetic energy and angles

    while(rejected)
    {
      *samples = (*samples) + 1;

      //heavy_distribution sampled_distribution = K_mod_distribution(generator);
      int sampled_distribution = K_mod_distribution(generator);

      // sample one of the 3 distributions based on integrated weights
      if(sampled_distribution == 0)
      {
        // draw k_mod from exp(-k_mod/T_mod).dk_mod by sampling r1
        // sample modified direction uniformly
        double r1 = 1.0 - canonical(generator);

        k_mod = - T_mod * log(r1);
        phi_mod = phi_distribution(generator);
        costheta_mod = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      else if(sampled_distribution == 1)
      {
        // draw (k_mod, phi_mod) from k_mod.exp(-k_mod/T_mod).dk_mod.dphi_mod by sampling (r1,r2)
        // sample costheta_mod uniformly
        double r1 = 1.0 - canonical(generator);
        double r2 = 1.0 - canonical(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k_mod = - T_mod * (l1 + l2);
        phi_mod = two_pi * l1 / (l1 + l2);
        costheta_mod = costheta_distribution(generator);

      } // distribution 2 (moderately heavy)
      else if(sampled_distribution == 2)
      {
        // draw (k_mod,phi_mod,costheta_mod) from k_mod^2.exp(-k_mod/T_mod).dk_mod.dphi_mod.dcostheta_mod by sampling (r1,r2,r3)
        double r1 = 1.0 - canonical(generator);
        double r2 = 1.0 - canonical(generator);
        double r3 = 1.0 - canonical(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k_mod = - T_mod * (l1 + l2 + l3);
        phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
        costheta_mod = (l1 - l2) / (l1 + l2);

      } // distribution 3 (light)
      else
      {
        printf("Error: did not sample any K_mod distribution\n");
        exit(-1);
      }

      double E_mod = k_mod + mass;
      double p_mod = sqrt(E_mod * E_mod  -  mass_squared);
      double sintheta_mod = sqrt(1.0  -  costheta_mod * costheta_mod);

      // modified pLRF components
      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // local momentum transformation
      pLRF = rescale_momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_mod, bulk_mod, diff_mod, baryon_enthalpy_ratio);

      double pdsigma_Theta = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

      // compute the weight
      double w_mod = (p_mod/E_mod) * exp(E_mod/T_mod - chem_mod) / (exp(E_mod/T_mod - chem_mod) + sign);
      double w_flux = pdsigma_Theta / (pLRF.E * ds_magnitude);

      if(sample_with_max_volume) w_flux = 1.0;

      double weight_heavy = w_mod * w_flux;

      // check if 0 <= weight <= 1
      if(fabs(weight_heavy - 0.5) > 0.5) printf("Error: modified heavy weight = %f out of bounds\n", weight_heavy);

      // check pLRF acceptance
      if(canonical(generator) < weight_heavy) break;

    } // rejection loop

  } // sampling heavy hadrons

  *acceptances = (*acceptances) + 1;

  return pLRF;

}

double EmissionFunctionArray::calculate_total_yield(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double * Equilibrium_Density, double * Bulk_Density, double * Diffusion_Density, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data * df_data, Gauss_Laguerre * laguerre, Gauss_Legendre * legendre)
  {
    // calculate the total mean particle yield from the freezeout surface
    // to determine the number of events you want to sample

    printf("Total particle yield = ");

    int npart = number_of_chosen_particles;

    double detA_min = DETA_MIN; // default value

    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaWeights[eta_pts];
    etaWeights[0] = 1.0;  // default for 3+1d

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
        //cout << ieta << "\t" << etaWeights[ieta] << endl;
      }
    }

    // set up root and weight arrays for pbar integrals in particle_number_outflow()
    const int legendre_pts = legendre->points;
    double * legendre_root = legendre->root;
    double * legendre_weight = legendre->weight;

    double pbar_root_outflow[legendre_pts];
    double pbar_weight_outflow[legendre_pts];

    for(int i = 0; i < legendre_pts; i++)
    {
      double x = legendre_root[i];  // x should never be 1.0 (or else pbar = inf)
      double s = (1.0 - x) / 2.0;   // variable transformations
      double pbar = (1.0 - s) / s;

      if(std::isinf(pbar))
      {
        printf("Error: none of legendre roots should be 1.0\n");
        exit(-1);
      }

      pbar_root_outflow[i] = pbar;
      pbar_weight_outflow[i] = 0.5 * pbar * pbar * legendre_weight[i] / (s * s);
    }

    /*
    // these are default densities if no outflow or regulated dfcorrection
    //double neq_tot = 0.0;                 // total equilibrium number / udsigma
    //double dn_bulk_tot = 0.0;             // bulk correction / udsigma / bulkPi
    //double dn_diff_tot = 0.0;             // diffusion correction / Vdsigma

    for(int ipart = 0; ipart < npart; ipart++)
    {
      neq_tot += Equilibrium_Density[ipart];
      dn_bulk_tot += Bulk_Density[ipart];
      dn_diff_tot += Diffusion_Density[ipart];

    } // hadron species (ipart)
    */

    double Ntot = 0.0;                    // total particle yield

    std::vector<double> N_list;           // particle yield of each species
    N_list.resize(npart);

    //#pragma omp parallel for
    for(long icell = 0; icell < FO_length; icell++)
    {
      double tau = tau_fo[icell];         // longitudinal proper time
      double tau2 = tau * tau;

      double dat = dat_fo[icell];         // covariant normal surface vector dsigma_mu
      double dax = dax_fo[icell];
      double day = day_fo[icell];
      double dan = dan_fo[icell];         // dan should be 0 in 2+1d case

      double ux = ux_fo[icell];           // contravariant fluid velocity u^mu
      double uy = uy_fo[icell];           // enforce normalization
      double un = un_fo[icell];           // u^eta (fm^-1)
      double ut = sqrt(1.0 +  ux * ux  + uy * uy  +  tau2 * un * un);   // u^tau

      double uperp =  sqrt(ux * ux  +  uy * uy);                        // useful expressions
      double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double ut2 = ut * ut;

      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan; // udotdsigma / eta_weight

      if(udsigma <= 0.0) continue;        // skip over cells with u.dsigma < 0

      double T = T_fo[icell];             // temperature (GeV)
      double P = P_fo[icell];             // equilibrium pressure (GeV/fm^3)
      double E = E_fo[icell];             // energy density (GeV/fm^3)

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
        pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
      }

      double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

      if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

      double muB = 0.0;
      double alphaB = 0.0;
      double nB = 0.0;
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu
      double Vx = 0.0;                    // enforce orthogonality
      double Vy = 0.0;
      double Vn = 0.0;
      double Vdsigma = 0.0;               // Vdotdsigma / delta_eta_weight
      double baryon_enthalpy_ratio = 0.0;

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        muB = muB_fo[icell];
        nB = nB_fo[icell];
        Vx = Vx_fo[icell];
        Vy = Vy_fo[icell];
        Vn = Vn_fo[icell];
        Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
        Vdsigma = Vt * dat  +  Vx * dax  +  Vy * day  +  Vn * dan;

        alphaB = muB / T;
        baryon_enthalpy_ratio = nB / (E + P);
      }

      // regulate bulk pressure if goes out of bounds given
      // by Jonah's feqmod to avoid gsl interpolation errors
      if (DF_MODE == 4)
      {
        double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

        if(bulkPi < - P) bulkPi = - (1.0 - 1.e-5) * P;
        else if(bulkPi / P > bulkPi_over_Peq_max) bulkPi = P * (bulkPi_over_Peq_max - 1.e-5);
      }

      // evaluate df coefficients
      deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

      // modified coefficients (Mike / Jonah)
      double F = df.F;
      double G = df.G;
      double betabulk = df.betabulk;
      double betapi = df.betapi;
      double lambda = df.lambda;
      double z = df.z;

      // milne basis vectors
      Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
      basis_vectors.test_orthonormality(tau2);

      // surface element class
      Surface_Element_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_magnitude();
      dsigma.compute_dsigma_lrf_polar_angle();

      // shear stress class
      Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
      pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);
      pimunu.diagonalize_pimunu_in_lrf();
      pimunu.compute_pi_magnitude();


      // modified temperature / chemical potential and rescaling coefficients
      double T_mod = T;
      double alphaB_mod = alphaB;

      double shear_mod = 0.0;
      double bulk_mod = 0.0;

      if(DF_MODE == 3)
      {
        T_mod = T  +  bulkPi * F / betabulk;
        alphaB_mod = alphaB  +  bulkPi * G / betabulk;

        shear_mod = 0.5 / betapi;
        bulk_mod = bulkPi / (3.0 * betabulk);
      }
      else if(DF_MODE == 4)
      {
        shear_mod = 0.5 / betapi;
        bulk_mod = lambda;
      }

      double detA = compute_detA(pimunu, shear_mod, bulk_mod);
      double detA_bulk = pow(1.0 + bulk_mod, 3);

      // determine if feqmod breaks down
      bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, detA_min, z, laguerre, DF_MODE);

      // total number of hadrons / eta_weight in FO_cell
      double dn_tot = 0.0;

      std::vector<double> dn_list;
      dn_list.resize(npart);

      // sum over hadrons
      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];              // particle properties
        double degeneracy = Degeneracy[ipart];
        double sign = Sign[ipart];
        double baryon = Baryon[ipart];

        dn_list[ipart] = mean_particle_number(mass, degeneracy, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, df, T_mod, alphaB_mod,feqmod_breaks_down, laguerre, legendre_pts, pbar_root_outflow, pbar_weight_outflow, DF_MODE, OUTFLOW, INCLUDE_BARYON, detA, detA_bulk, 0);

        dn_tot += dn_list[ipart];
      }

      // add mean number of hadrons in FO cell to total yield
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        Ntot += dn_tot * etaWeights[ieta];

        for(int ipart = 0; ipart < npart; ipart++)
        {
          N_list[ipart] += dn_list[ipart] * etaWeights[ieta];
        }
      }

    } // freezeout cells (icell)

    mean_yield = Ntot;
    printf("Total dN_dy = %lf\n\n", Ntot/14.0); // dN/deta (for the eta_trapezoid_table_57pt.dat)

    // for(int ipart = 0; ipart < npart; ipart++)
    // {
    //   cout << setprecision(5) << N_list[ipart]/14.0 << endl;
    // }
    // exit(-1);


    return Ntot;

  }

void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data, Gauss_Laguerre * laguerre, Gauss_Legendre * legendre)
  {
    int npart = number_of_chosen_particles;

    long breakdown = 0;

    double detA_min = DETA_MIN; // default


    // All of this should be made more global for case of using sampler across multiple cores
    //::::::::::::::::::::::::::
    // set seed
    unsigned seed;
    if (SAMPLER_SEED < 0) seed = chrono::system_clock::now().time_since_epoch().count();
    else seed = SAMPLER_SEED;

    default_random_engine generator_poisson(seed);
    default_random_engine generator_type(seed);
    default_random_engine generator_momentum(seed);
    //:::::::::::::::::::::::::::

    // set up extension in eta
    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaValues[eta_pts];
    double etaWeights[eta_pts];

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0; // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
    }

    // set up root and weight arrays for pbar integrals in particle_number_outflow()
    const int legendre_pts = legendre->points;
    double * legendre_root = legendre->root;
    double * legendre_weight = legendre->weight;

    double pbar_root_outflow[legendre_pts];
    double pbar_weight_outflow[legendre_pts];

    for(int i = 0; i < legendre_pts; i++)
    {
      double x = legendre_root[i];  // x should never be 1.0 (or else pbar = inf)
      double s = (1.0 - x) / 2.0;   // variable transformations
      double pbar = (1.0 - s) / s;

      if(std::isinf(pbar)){printf("Error: none of legendre roots should be -1\n"); exit(-1);}

      pbar_root_outflow[i] = pbar;
      pbar_weight_outflow[i] = 0.5 * pbar * pbar * legendre_weight[i] / (s * s);
    }


    // compute contributions to total mean hadron number / cell volume
    /*
    double neq_tot = 0.0;                      // total equilibrium number / udsigma
    double dn_bulk_tot = 0.0;                  // total bulk number / bulkPi / udsigma
    double dn_diff_tot = 0.0;                  // - total diff number / Vdsigma

    for(int ipart = 0; ipart < npart; ipart++)
    {
      neq_tot += Equilibrium_Density[ipart];
      dn_bulk_tot += Bulk_Density[ipart];
      dn_diff_tot += Diffusion_Density[ipart];
    }
    */

    // for benchmarking momentum sampling efficiency
    long acceptances = 0;
    long samples = 0;

    // loop over all freezeout cells
    //#pragma omp parallel for
    for(long icell = 0; icell < FO_length; icell++)
    {
      double tau = tau_fo[icell];         // freezeout cell coordinates
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
      double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un); // u^tau

      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan; // udotdsigma / eta_weight

      if(udsigma <= 0.0) continue;        // skip over cells with u.dsigma < 0

      double ut2 = ut * ut;               // useful expressions
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux * ux  +  uy * uy);
      double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

      double T = T_fo[icell];             // temperature (GeV)
      double P = P_fo[icell];             // pressure (GeV/fm^3)
      double E = E_fo[icell];             // energy density (GeV/fm^3)

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
        pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
      }

      double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

      if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

      double muB = 0.0;
      double alphaB = 0.0;
      double nB = 0.0;
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu
      double Vx = 0.0;                    // enforce orthogonality
      double Vy = 0.0;
      double Vn = 0.0;
      double Vdsigma = 0.0;               // Vdotdsigma / delta_eta_weight
      double baryon_enthalpy_ratio = 0.0;

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        muB = muB_fo[icell];
        nB = nB_fo[icell];
        Vx = Vx_fo[icell];
        Vy = Vy_fo[icell];
        Vn = Vn_fo[icell];
        Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
        Vdsigma = Vt * dat  +  Vx * dax  +  Vy * day  +  Vn * dan;

        alphaB = muB / T;
        baryon_enthalpy_ratio = nB / (E + P);
      }

      // regulate bulk pressure if goes out of bounds given
      // by Jonah's feqmod to avoid gsl interpolation errors
      if (DF_MODE == 4)
      {
        double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

        if(bulkPi < - P) bulkPi = - (1.0 - 1.e-6) * P;
        else if(bulkPi / P > bulkPi_over_Peq_max) bulkPi = P * (bulkPi_over_Peq_max - 1.e-6);
      }

      deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

      // modified coefficients (Mike / Jonah)
      double F = df.F;
      double G = df.G;
      double betabulk = df.betabulk;
      double betaV = df.betaV;
      double betapi = df.betapi;
      double lambda = df.lambda;
      double z = df.z;

      // milne basis class
      Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
      basis_vectors.test_orthonormality(tau2);

      // dsigma / eta_weight class
      Surface_Element_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_magnitude();
      dsigma.compute_dsigma_lrf_polar_angle();

      // LRF components
      double dst = dsigma.dsigmat_LRF;
      double dsx = dsigma.dsigmax_LRF;
      double dsy = dsigma.dsigmay_LRF;
      double dsz = dsigma.dsigmaz_LRF;
      double ds_max = dsigma.dsigma_magnitude;

      // shear stress class
      Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
      pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);
      pimunu.diagonalize_pimunu_in_lrf();
      pimunu.compute_pi_magnitude();

      // baryon diffusion class
      Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
      Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
      Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);

      // modified temperature / chemical potential and rescaling coefficients
      double T_mod = T;
      double alphaB_mod = alphaB;

      double shear_mod = 0.0;
      double bulk_mod = 0.0;
      double diff_mod = 0.0;

      if(DF_MODE == 3)
      {
        T_mod = T  +  bulkPi * F / betabulk;
        alphaB_mod = alphaB  +  bulkPi * G / betabulk;

        shear_mod = 0.5 / betapi;
        bulk_mod = bulkPi / (3.0 * betabulk);
        diff_mod = T / betaV;
      }
      else if(DF_MODE == 4)
      {
        shear_mod = 0.5 / betapi;
        bulk_mod = lambda;
        diff_mod = 0.0;
      }


      double detA = compute_detA(pimunu, shear_mod, bulk_mod);
      double detA_bulk = pow(1.0 + bulk_mod, 3);

      // determine if feqmod breaks down
      bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, detA_min, z, laguerre, DF_MODE);

      if(feqmod_breaks_down)
      {
        breakdown++;
        cout << setw(5) << setprecision(4) << "feqmod breaks down at " << breakdown << " / " << FO_length << " cell at tau = " << tau << " fm/c:" << "\t detA = " << detA << "\t detA_min = " << detA_min << endl;
      }

      // holds mean particle number / eta_weight of all species
      std::vector<double> dn_list;
      dn_list.resize(npart);

      // total mean number  / eta_weight of hadrons
      // emitted from freezeout cell of max volume
      double dn_tot = 0.0;

      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];              // mass (GeV)
        double degeneracy = Degeneracy[ipart];  // spin degeneracy
        double sign = Sign[ipart];              // quantum statistics sign
        double baryon = Baryon[ipart];          // baryon number

        dn_list[ipart] = mean_particle_number(mass, degeneracy, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, df, T_mod, alphaB_mod,feqmod_breaks_down, laguerre, legendre_pts, pbar_root_outflow, pbar_weight_outflow, DF_MODE, OUTFLOW, INCLUDE_BARYON, detA, detA_bulk, SAMPLE_WITH_MAX_VOLUME);

        dn_tot += dn_list[ipart];
      }

      // construct discrete probability distribution for particle types (weight[ipart] ~ dn_list[ipart] / dn_tot)
      std::discrete_distribution<int> particle_type(dn_list.begin(), dn_list.end());

      // loop over eta points
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta = etaValues[ieta];
        double sinheta = sinh(eta);
        double cosheta = sqrt(1.0  +  sinheta * sinheta);

        double dN_tot = dn_tot * etaWeights[ieta];              // total mean number of hadrons in FO cell

        if(dN_tot <= 0.0) continue;                             // error: poisson mean value out of bounds

        // construct poisson probability distribution for number of hadrons
        std::poisson_distribution<int> poisson_hadrons(dN_tot);

        // sample events for each FO cell
        for(int ievent = 0; ievent < Nevents; ievent++)
        {
          int N_hadrons = poisson_hadrons(generator_poisson);   // sample total number of hadrons in FO cell

          for(int n = 0; n < N_hadrons; n++)
          {
            int chosen_index = particle_type(generator_type);   // chosen index of sampled particle type
            double mass = Mass[chosen_index];                   // mass of sampled particle in GeV
            double sign = Sign[chosen_index];                   // quantum statistics sign
            double baryon = Baryon[chosen_index];               // baryon number
            int mcid = MCID[chosen_index];                      // mc id

            LRF_Momentum pLRF;

            // sample the local rest frame momentum
            switch(DF_MODE)
            {
              case 1: // 14 moment
              case 2: // Chapman Enskog
              {
                switch_to_linear_df:

                pLRF = sample_momentum(generator_momentum, &acceptances, &samples, mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, df, baryon_enthalpy_ratio, DF_MODE, SAMPLE_WITH_MAX_VOLUME);
                break;
              }
              case 3: // Modified (Mike)
              {
                if(feqmod_breaks_down) goto switch_to_linear_df;

                pLRF = sample_momentum_feqmod(generator_momentum, &acceptances, &samples, mass, sign, baryon, T_mod, alphaB_mod, dsigma, pimunu, Vmu, shear_mod, bulk_mod, diff_mod, baryon_enthalpy_ratio, SAMPLE_WITH_MAX_VOLUME);

                break;
              }
              case 4: // Modified (Jonah)
              {
                if(feqmod_breaks_down) goto switch_to_linear_df;

                pLRF = sample_momentum_feqmod(generator_momentum, &acceptances, &samples, mass, sign, 0.0, T, 0.0, dsigma, pimunu, Vmu, shear_mod, bulk_mod, 0.0, 0.0, SAMPLE_WITH_MAX_VOLUME);

                break;
              }
              default:
              {
                printf("\nError: for viscous hydro momentum sampling please set df_mode = (1,2,3,4)\n");
                exit(-1);
              }
            }

            // lab frame momentum
            Lab_Momentum pLab(pLRF);
            pLab.boost_pLRF_to_lab_frame(basis_vectors, ut, ux, uy, un);

            // new sampled particle info
            Sampled_Particle new_particle;

            new_particle.chosen_index = chosen_index;
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

            double pz = tau * pLab.pn * cosheta  +  pLab.ptau * sinheta;
            new_particle.pz = pz;
            new_particle.E = sqrt(mass * mass  +  pLab.px * pLab.px  +  pLab.py * pLab.py  +  pz * pz);

            if(SAMPLE_WITH_MAX_VOLUME)
            {
              double pdsigma_Theta = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

              double w_flux = pdsigma_Theta / (pLRF.E * ds_max);   // flux weight

              if(canonical(generator_momentum) < w_flux)
              {
                if(TEST_SAMPLER)
                {
                  // main purpose is to avoid memory bottleneck
                  sample_dN_dpT(new_particle);
                  sample_vn(new_particle);
                  sample_dN_dX(new_particle);
                }
                else
                {
                  //CAREFUL push_back is not a thread-safe operation
                  //how should we modify for GPU version?
                  #pragma omp critical
                  particle_event_list[ievent].push_back(new_particle);
                }
                particle_yield_list[ievent] += 1;
              }
            } // sampling particles from max volume
            else
            {
              if(TEST_SAMPLER)
              {
                sample_dN_dpT(new_particle);
                sample_vn(new_particle);
                sample_dN_dX(new_particle);
              }
              else
              {
                //CAREFUL push_back is not a thread-safe operation
                //how should we modify for GPU version?
                #pragma omp critical
                particle_event_list[ievent].push_back(new_particle);
              }
              particle_yield_list[ievent] += 1;
            } // always add particle if computed mean hadron number exactly

          } // sampled hadrons (n)
        } // sampled events (ievent)
      } // eta points (ieta)
      //cout << "\r" << "Finished " << setw(4) << setprecision(3) << (double)(icell + 1) / (double)FO_length * 100.0 << " \% of freezeout cells" << flush;
    } // freezeout cells (icell)
    printf("\nMomentum sampling efficiency = %f %%\n", (float)(100.0 * (double)acceptances / (double)samples));
}







///////////////////////////////////////////////////////////////



/*


double max_particle_number_jonah(double mass, double degeneracy, double sign, double T, double ds_max, double z, Gauss_Laguerre * laguerre)
{
  // mean particle number of a given species emitted from freezeout cell w/ max volume
  // - max mean number ~ particle_density * ds_max for feqmod if it doesn't break down
  double mbar = mass / T;

  const int gauss_pts = laguerre->points;
  double * pbar_root1 = laguerre->root[1];
  double * pbar_weight1 = laguerre->weight[1];

  double neq_fact = T * T * T / two_pi2_hbarC3;
  double equilibrium_density = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, gauss_pts, mbar, 0.0, 0.0, sign);

  return ds_max * z * equilibrium_density;
}




LRF_Momentum rescale_momentum_jonah(LRF_Momentum pLRF_mod, double mass_squared, Shear_Stress pimunu, double shear_mod, double bulk_mod)
{
    double E_mod = pLRF_mod.E;
    double px_mod = pLRF_mod.px;
    double py_mod = pLRF_mod.py;
    double pz_mod = pLRF_mod.pz;

    // LRF shear stress components
    double pixx = pimunu.pixx_LRF;
    double pixy = pimunu.pixy_LRF;
    double pixz = pimunu.pixz_LRF;
    double piyy = pimunu.piyy_LRF;
    double piyz = pimunu.piyz_LRF;
    double pizz = pimunu.pizz_LRF;

    LRF_Momentum pLRF;

    // local momentum transformation: p_i = A_ij * p_mod_j
    pLRF.px = (1.0 + bulk_mod) * px_mod  +  shear_mod * (pixx * px_mod  +  pixy * py_mod  +  pixz * pz_mod);
    pLRF.py = (1.0 + bulk_mod) * py_mod  +  shear_mod * (pixy * px_mod  +  piyy * py_mod  +  piyz * pz_mod);
    pLRF.pz = (1.0 + bulk_mod) * pz_mod  +  shear_mod * (pixz * px_mod  +  piyz * py_mod  +  pizz * pz_mod);
    pLRF.E = sqrt(mass_squared  +  pLRF.px * pLRF.px  +  pLRF.py * pLRF.py  +  pLRF.pz * pLRF.pz);

    return pLRF;
}



LRF_Momentum sample_momentum_feqmod_jonah(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double T, Shear_Stress pimunu, double shear_mod, double bulk_mod)
{
  // samples the local rest frame momentum from
  // Jonah's distribution (the flux weight is separated)
  double mass_squared = mass * mass;

  uniform_real_distribution<double> phi_distribution(0.0, two_pi);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  LRF_Momentum pLRF;
  LRF_Momentum pLRF_mod;

  bool rejected = true;

  if(mass == 0.0)
  {
    printf("Error: cannot sample photons with this method. Exiting...\n");
    exit(-1);
  }
  if(mass / T < 1.008)
  {
    double w_mod_max = 1.0;
    if(mass / T <= 0.8554 && sign == -1.0) w_mod_max = pion_thermal_weight_max(mass, T, 0.0);

    while(rejected)
    {
      *samples = (*samples) + 1;
      double r1 = 1.0 - canonical(generator);
      double r2 = 1.0 - canonical(generator);
      double r3 = 1.0 - canonical(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p_mod = - T * (l1 + l2 + l3);
      double phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
      double costheta_mod = (l1 - l2) / (l1 + l2);

      double E_mod = sqrt(p_mod * p_mod  +  mass_squared);
      double sintheta_mod = sqrt(1.0 - costheta_mod * costheta_mod);

      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      pLRF = rescale_momentum_jonah(pLRF_mod, mass_squared, pimunu, shear_mod, bulk_mod);

      double weight_light = exp(p_mod/T) / (exp(E_mod/T) + sign) / w_mod_max;

      if(fabs(weight_light - 0.5) > 0.5) printf("Error: modified light weight = %f out of bounds\n", weight_light);

      if(canonical(generator) < weight_light) break;

    } // rejection loop

  } // sampling pions

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    std::vector<double> K_mod_weight;
    K_mod_weight.resize(3);

    K_mod_weight[0] = mass_squared;
    K_mod_weight[1] = 2.0 * mass * T;
    K_mod_weight[2] = 2.0 * T * T;

    discrete_distribution<int> K_mod_distribution(K_mod_weight.begin(), K_mod_weight.end());

    double k_mod, phi_mod, costheta_mod;

    while(rejected)
    {
      *samples = (*samples) + 1;

      int sampled_distribution = K_mod_distribution(generator);

      if(sampled_distribution == 0)
      {
        double r1 = 1.0 - canonical(generator);

        k_mod = - T * log(r1);
        phi_mod = phi_distribution(generator);
        costheta_mod = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      else if(sampled_distribution == 1)
      {
        double r1 = 1.0 - canonical(generator);
        double r2 = 1.0 - canonical(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k_mod = - T * (l1 + l2);
        phi_mod = two_pi * l1 / (l1 + l2);
        costheta_mod = costheta_distribution(generator);

      } // distribution 2 (moderately heavy)
      else if(sampled_distribution == 2)
      {
        double r1 = 1.0 - canonical(generator);
        double r2 = 1.0 - canonical(generator);
        double r3 = 1.0 - canonical(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k_mod = - T * (l1 + l2 + l3);
        phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
        costheta_mod = (l1 - l2) / (l1 + l2);

      } // distribution 3 (light)
      else
      {
        printf("Error: did not sample any K_mod distribution\n");
        exit(-1);
      }

      double E_mod = k_mod + mass;
      double p_mod = sqrt(E_mod * E_mod  -  mass_squared);
      double sintheta_mod = sqrt(1.0  -  costheta_mod * costheta_mod);

      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      pLRF = rescale_momentum_jonah(pLRF_mod, mass_squared, pimunu, shear_mod, bulk_mod);

      double weight_heavy = (p_mod/E_mod) * exp(E_mod/T) / (exp(E_mod/T) + sign);

      if(fabs(weight_heavy - 0.5) > 0.5) printf("Error: modified heavy weight = %f out of bounds\n", weight_heavy);

      if(canonical(generator) < weight_heavy) break;

    } // rejection loop

  } // sampling heavy hadrons

  *acceptances = (*acceptances) + 1;

  return pLRF;

}



void EmissionFunctionArray::sample_dN_pTdpTdphidy_jonah(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, Deltaf_Data *df_data, Gauss_Laguerre * laguerre)
  {
    // sampling by using Jonah's method:
    // 1. sample particles from a max volume
    // 2. keep with probability = flux weight

    if(DF_MODE != 4)
    {
      printf("Error: to sample jonah, set df mode = 4\n");
      exit(-1);
    }

    int npart = number_of_chosen_particles;

    // All of this should be made more global for case of using sampler across multiple cores
    //::::::::::::::::::::::::::
    // set seed
    unsigned seed;
    if (SAMPLER_SEED < 0) seed = chrono::system_clock::now().time_since_epoch().count();
    else seed = SAMPLER_SEED;

    default_random_engine generator_poisson(seed);
    default_random_engine generator_type(seed);
    default_random_engine generator_momentum(seed);
    //:::::::::::::::::::::::::::

    // set up extension in eta
    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaValues[eta_pts];
    double etaWeights[eta_pts];

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0; // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
    }

    long acceptances = 0;
    long samples = 0;

    // loop over all freezeout cells
    //#pragma omp parallel for
    for(long icell = 0; icell < FO_length; icell++)
    {
      double tau = tau_fo[icell];         // freezeout cell coordinates
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
      double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un); // u^tau

      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan; // udotdsigma / eta_weight

      if(udsigma <= 0.0) continue;        // skip over cells with u.dsigma < 0

      double ut2 = ut * ut;               // useful expressions
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux * ux  +  uy * uy);
      double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

      double T = T_fo[icell];             // temperature (GeV)
      double P = P_fo[icell];             // pressure (GeV/fm^3)
      double E = E_fo[icell];             // energy density (GeV/fm^3)

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
        pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
      }

      double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

      if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

      double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

      if(bulkPi < - P) bulkPi = - (1.0 - 1.e-6) * P;
      else if(bulkPi / P > bulkPi_over_Peq_max) bulkPi = P * (bulkPi_over_Peq_max - 1.e-6);

      deltaf_coefficients df = df_data->evaluate_df_coefficients(T, 0.0, E, P, bulkPi);

      // jonah modified coefficients
      double betapi = df.betapi;
      double lambda = df.lambda;
      double z = df.z;

      if(z < 0.0) printf("Error in Jonah's feqmod: the normalization factor z is negative\n");

      // milne basis class
      Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
      basis_vectors.test_orthonormality(tau2);

      // dsigma / eta_weight class
      Surface_Element_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_magnitude();

      double dst = dsigma.dsigmat_LRF;
      double dsx = dsigma.dsigmax_LRF;
      double dsy = dsigma.dsigmay_LRF;
      double dsz = dsigma.dsigmaz_LRF;
      double ds_max = dsigma.dsigma_magnitude;

      // shear stress class
      Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
      pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

      double shear_mod = 0.5 / betapi;
      double bulk_mod = lambda;
      double diff_mod = 0.0;

      // holds mean particle number / eta_weight of all species
      std::vector<double> dn_list;
      dn_list.resize(npart);

      double dn_tot = 0.0;

      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];
        double degeneracy = Degeneracy[ipart];
        double sign = Sign[ipart];

        dn_list[ipart] = max_particle_number_jonah(mass, degeneracy, sign, T, ds_max, z, laguerre);

        dn_tot += dn_list[ipart];
      }

      std::discrete_distribution<int> particle_type(dn_list.begin(), dn_list.end());

      // loop over eta points
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta = etaValues[ieta];
        double sinheta = sinh(eta);
        double cosheta = sqrt(1.0  +  sinheta * sinheta);

        double dN_tot = dn_tot * etaWeights[ieta];

        if(dN_tot <= 0.0) continue;

        std::poisson_distribution<int> poisson_hadrons(dN_tot);

        // sample events for each FO cell
        for(int ievent = 0; ievent < Nevents; ievent++)
        {
          int N_hadrons = poisson_hadrons(generator_poisson);

          for(int n = 0; n < N_hadrons; n++)
          {
            int chosen_index = particle_type(generator_type);   // chosen index of sampled particle type
            double mass = Mass[chosen_index];                   // mass of sampled particle in GeV
            double sign = Sign[chosen_index];                   // quantum statistics sign
            double baryon = Baryon[chosen_index];               // baryon number
            int mcid = MCID[chosen_index];                      // mc id

            // sample LRF momentum
            LRF_Momentum pLRF = sample_momentum_feqmod_jonah(generator_momentum, &acceptances, &samples, mass, sign, T, pimunu, shear_mod, bulk_mod);

            // lab frame momentum
            Lab_Momentum pLab(pLRF);
            pLab.boost_pLRF_to_lab_frame(basis_vectors, ut, ux, uy, un);

            // new sampled particle info
            Sampled_Particle new_particle;

            new_particle.chosen_index = chosen_index;
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

            double pz = tau * pLab.pn * cosheta  +  pLab.ptau * sinheta;
            new_particle.pz = pz;
            new_particle.E = sqrt(mass * mass  +  pLab.px * pLab.px  +  pLab.py * pLab.py  +  pz * pz);

            double pdsigma_Theta = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

            // flux weight
            double w_flux = pdsigma_Theta / (pLRF.E * ds_max);

            double propose = canonical(generator_momentum);

            // determine whether keep particle based on flux weight
            if(propose < w_flux)
            {
              if(TEST_SAMPLER)
              {
                // main purpose is to avoid memory bottleneck
                sample_dN_dpT(new_particle);
                sample_vn(new_particle);
                sample_dN_dX(new_particle);
              }
              else
              {
                //CAREFUL push_back is not a thread-safe operation
                //how should we modify for GPU version?
                #pragma omp critical
                particle_event_list[ievent].push_back(new_particle);
              }

              particle_yield_list[ievent] += 1;
            } // keep particle

          } // sampled hadrons (n)
        } // sampled events (ievent)
      } // eta points (ieta)
      //cout << "\r" << "Finished " << setw(4) << setprecision(3) << (double)(icell + 1) / (double)FO_length * 100.0 << " \% of freezeout cells" << flush;
    } // freezeout cells (icell)
    printf("\nMomentum sampling efficiency = %f %%\n", (float)(100.0 * (double)acceptances / (double)samples));
}

*/
///////////////////////////////////////////



void EmissionFunctionArray::sample_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
  double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
  double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
{
  printf("Sampling particles from VAH \n");
  printf("NOTHING HERE YET \n");
}
