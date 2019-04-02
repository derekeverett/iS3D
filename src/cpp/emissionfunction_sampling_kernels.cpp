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

void EmissionFunctionArray::sample_dN_dy_average(Sampled_Particle new_particle)
{
  // construct the dN/dy distribution averaged over rapidity window = 2.Y_CUT by adding counts from all events

  double E = new_particle.E;
  double pz = new_particle.pz;
  double yp = 0.5 * log((E + pz) / (E - pz));

  if(fabs(yp) <= Y_CUT)
  {
    int ipart = new_particle.chosen_index;
    dN_dy_count[ipart] += 1.0;              // add counts within rapidity cut
  }
}

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




double mean_particle_number(double mass, double degeneracy, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, deltaf_coefficients df, bool feqmod_breaks_down, Gauss_Laguerre * laguerre, const int legendre_pts, double * pbar_root, double * pbar_weight, int df_mode, int include_baryon, int use_max_volume)
{
  // computes mean particle number emitted from freezeout cell
  // with Theta(p.dsigma) (outflow) and feq + df_regulated (or feqmod)

  double mbar = mass / T;
  double mbar_squared = mbar * mbar;
  double chem = baryon * alphaB;

  // dsigma volume used
  double ds_time = dsigma.dsigmat_LRF;      // for estimating number of events
  double ds_max = dsigma.dsigma_magnitude;  // for actual sampling

  double ds_volume;
  if(use_max_volume) ds_volume = ds_max;
  else ds_volume = ds_time;
 
  double particle_number = 0.0;

  switch(df_mode)
  {
    case 1: // 14 moment
    {
      double mass_squared = mass * mass;

      double c0 = df.c0;
      double c1 = df.c1;
      double c2 = df.c2;
      double shear14_coeff = df.shear14_coeff;

      // diagonalized components and amplitudes of the azimuthal square wave approximation
      double pizz_D = pimunu.pizz_D;
      double piperp_plus = pimunu.piperp_plus;
      double piperp_minus = pimunu.piperp_minus;

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

        double g = feqbar * p * p / shear14_coeff;
        double h = df_bulk;

        double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
        double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

        particle_number += weight * (f_plus + f_minus);

      } // i

      particle_number *= ds_volume * degeneracy * T * T * T / four_pi2_hbarC3;

      break;
    }
    case 2: // Chapman Enskog
    {
      chapman_enskog:

      double F = df.F;
      double G = df.G;
      double betabulk = df.betabulk;
      double betapi = df.betapi;

      // diagonalized components and amplitudes of the azimuthal square wave approximation
      double pizz_D = pimunu.pizz_D;
      double piperp_plus = pimunu.piperp_plus;
      double piperp_minus = pimunu.piperp_minus;
    
      for(int i = 0; i < legendre_pts; i++)
      {
        double pbar = pbar_root[i];
        double weight = pbar_weight[i];

        double Ebar = sqrt(pbar * pbar +  mbar_squared);
        double E = Ebar * T;

        double feq = 1.0 / (exp(Ebar - chem) + sign);
        double feqbar = 1.0 - sign * feq;

        double df_bulk = feqbar * (F / T * Ebar  +  baryon * G  +  (Ebar  -  mbar_squared / Ebar) / 3.0) * bulkPi / betabulk;

        double g = feqbar * pbar * pbar / (2.0 * betapi * Ebar);
        double h = df_bulk;

        double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
        double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

        particle_number += weight * (f_plus + f_minus);

      } // i

      particle_number *= ds_volume * degeneracy * T * T * T / four_pi2_hbarC3;

      break;
    }
    case 3: // modified (Mike)
    {
      if(feqmod_breaks_down) goto chapman_enskog;

      double G = df.G;
      double F = df.F;
      double betabulk = df.betabulk;

      // compute the linearized density
      const int laguerre_pts = laguerre->points;
      double * pbar_root1 = laguerre->root[1];
      double * pbar_root2 = laguerre->root[2];
      double * pbar_weight1 = laguerre->weight[1];
      double * pbar_weight2 = laguerre->weight[2];

      double T2 = T * T;
      double neq_fact = T2 * T / two_pi2_hbarC3;
      double J20_fact = T * neq_fact;

      double equilibrium_density = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, alphaB, baryon, sign);
      double J10 = 0.0;
      if(include_baryon)
      {
        J10 = neq_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, alphaB, baryon, sign);
      }
      double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, laguerre_pts, mbar, alphaB, baryon, sign);
      double bulk_density = (equilibrium_density + (baryon * J10 * G) + (J20 * F / T2)) / betabulk;

      double linearized_density = equilibrium_density  +  bulkPi * bulk_density;
     

      if(linearized_density < 0.0) printf("Error: linearized density is negative. Adjust the mass_pion0 parameter...\n");

      //:::::::::::::::
      particle_number = ds_volume * linearized_density;
      //:::::::::::::::

      break;
    }
    case 4: // modified (Jonah)
    {
      if(!feqmod_breaks_down)
      {
        double z = df.z;

        if(z < 0.0) printf("Error: z is negative\n");

        const int laguerre_pts = laguerre->points;
        double * pbar_root1 = laguerre->root[1];
        double * pbar_weight1 = laguerre->weight[1];

        double neq_fact = T * T * T / two_pi2_hbarC3;
        double equilibrium_density = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar, 0.0, 0.0, sign);

        //:::::::::::::::
        particle_number = ds_volume * z * equilibrium_density;
        //:::::::::::::::
      }
      else
      {
        double betapi = df.betapi;
        double delta_z = df.delta_z;
        double delta_lambda = df.delta_lambda;

        // diagonalized components and amplitudes of the azimuthal square wave approximation
        double pizz_D = pimunu.pizz_D;
        double piperp_plus = pimunu.piperp_plus;
        double piperp_minus = pimunu.piperp_minus;
    
        // radial momentum pbar integral (pbar = p / T)
        for(int i = 0; i < legendre_pts; i++)
        {
          double pbar = pbar_root[i];
          double weight = pbar_weight[i];

          double Ebar = sqrt(pbar * pbar +  mbar_squared);

          double feq = 1.0 / (exp(Ebar - chem) + sign);
          double feqbar = 1.0 - sign * feq;

          double df_bulk_jonah = delta_z  - 3.0 * delta_lambda  +  feqbar * delta_lambda * (Ebar  -  mbar_squared / Ebar);
        
          double g = feqbar * pbar * pbar / (2.0 * betapi * Ebar);
          double h = df_bulk_jonah;

          double f_plus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_plus);
          double f_minus = compute_angular_phase_space_integral(feq, h, g, pizz_D, piperp_minus);

          particle_number += weight * (f_plus + f_minus);

        } // i

        particle_number *= ds_volume * degeneracy * T * T * T / four_pi2_hbarC3;
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


LRF_Momentum sample_momentum(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, deltaf_coefficients df, double baryon_enthalpy_ratio, int df_mode)
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

      // compute the weights
      double w_eq = exp(p/T) / (exp(E/T) + sign) / weq_max;
      double w_visc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df, baryon_enthalpy_ratio, df_mode);

      double weight_light = w_eq * w_visc;

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


      // compute the weight
      double w_eq = p/E * exp(E/T - chem) / (exp(E/T - chem) + sign);
      double w_visc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df, baryon_enthalpy_ratio, df_mode);

      double weight_heavy = w_eq * w_visc;

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


LRF_Momentum sample_momentum_feqmod(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T_mod, double alphaB_mod, Surface_Element_Vector dsigma, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_mod, double bulk_mod, double diff_mod, double baryon_enthalpy_ratio)
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

      // compute the weight
      double weight_light = exp(p_mod/T_mod - chem_mod) / (exp(E_mod/T_mod - chem_mod) + sign) / w_mod_max;

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

      // compute the weight
      double weight_heavy = (p_mod/E_mod) * exp(E_mod/T_mod - chem_mod) / (exp(E_mod/T_mod - chem_mod) + sign);

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
    // estimate the total mean particle yield from the freezeout surface
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

    double Ntot = 0.0;                    // total particle yield

    #pragma omp parallel for reduction(+:Ntot)
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

      // sum over hadrons
      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];              // particle properties
        double degeneracy = Degeneracy[ipart];
        double sign = Sign[ipart];
        double baryon = Baryon[ipart];

        dn_tot += mean_particle_number(mass, degeneracy, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, df, feqmod_breaks_down, laguerre, legendre_pts, pbar_root_outflow, pbar_weight_outflow, DF_MODE, INCLUDE_BARYON, 0);
      }

      // add mean number of hadrons in FO cell to total yield
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        Ntot += dn_tot * etaWeights[ieta];
      }

    } // freezeout cells (icell)

    mean_yield = Ntot;
    printf("dN_dy ~ %lf\n\n", Ntot/14.0); // dN/deta (for the eta_trapezoid_table_57pt.dat)

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

      #pragma omp parallel for reduction(+:dn_tot)
      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];              // mass (GeV)
        double degeneracy = Degeneracy[ipart];  // spin degeneracy
        double sign = Sign[ipart];              // quantum statistics sign
        double baryon = Baryon[ipart];          // baryon number

        dn_list[ipart] = mean_particle_number(mass, degeneracy, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, df, feqmod_breaks_down, laguerre, legendre_pts, pbar_root_outflow, pbar_weight_outflow, DF_MODE, INCLUDE_BARYON, 1);

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
        //#pragma omp parallel for
        for(int ievent = 0; ievent < Nevents; ievent++)
        {
          //#pragma omp critical
          int N_hadrons = poisson_hadrons(generator_poisson);   // sample total number of hadrons in FO cell

          for(int n = 0; n < N_hadrons; n++)
          {
            //#pragma omp critical
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

                //#pragma omp critical
                pLRF = sample_momentum(generator_momentum, &acceptances, &samples, mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, df, baryon_enthalpy_ratio, DF_MODE);
                break;
              }
              case 3: // Modified (Mike)
              {
                if(feqmod_breaks_down) goto switch_to_linear_df;

                //#pragma omp critical
                pLRF = sample_momentum_feqmod(generator_momentum, &acceptances, &samples, mass, sign, baryon, T_mod, alphaB_mod, dsigma, pimunu, Vmu, shear_mod, bulk_mod, diff_mod, baryon_enthalpy_ratio);

                break;
              }
              case 4: // Modified (Jonah)
              {
                if(feqmod_breaks_down) goto switch_to_linear_df;

                //#pragma omp critical
                pLRF = sample_momentum_feqmod(generator_momentum, &acceptances, &samples, mass, sign, 0.0, T, 0.0, dsigma, pimunu, Vmu, shear_mod, bulk_mod, 0.0, 0.0);

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

            double pdsigma_Theta = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

            double w_flux = pdsigma_Theta / (pLRF.E * ds_max);   // flux weight

            if(canonical(generator_momentum) < w_flux)
            {
              if(TEST_SAMPLER)
              {
                // main purpose is to avoid memory bottleneck
                sample_dN_dy_average(new_particle);
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

          } // sampled hadrons (n)
        } // sampled events (ievent)
      } // eta points (ieta)
      //cout << "\r" << "Finished " << setw(4) << setprecision(3) << (double)(icell + 1) / (double)FO_length * 100.0 << " \% of freezeout cells" << flush;
    } // freezeout cells (icell)
    printf("\nMomentum sampling efficiency = %f %%\n", (float)(100.0 * (double)acceptances / (double)samples));
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
