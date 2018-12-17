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

#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

using namespace std;


double EmissionFunctionArray::particle_number_outflow(double mass, double degeneracy, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, double * df_coeff, double shear14_coeff, double equilibrium_density, double bulk_density, double T_mod, double alphaB_mod, double detA, double modified_density, bool feqmod_breaks_down)
{
  // computes the particle density from the CFF with Theta(p.dsigma) (outflow) and feq + df_regulated (or feqmod)
  // for spacelike cells: (neq + dn_regulated)_outflow > neq + dn_regulated
  // for timelike cells:  (neq + dn_regulated)_outflow = neq + dn_regulated
  // the reason we do this: self-consistency between sampled and smooth CFF for more precise testing of the sampler

  // the feqmod formula is an approximation due to the complexity of the integral


  double four_pi2_hbarC3 = 4.0 * pow(M_PI, 2) * pow(hbarC, 3);


  // don't use odd # of points 
  const int gauss_pts = 24;

  //hard coded roots/weights 

  // x_i
  double gauss_legendre_root[gauss_pts] = {-0.99518721999702, -0.97472855597131, -0.93827455200273, -0.8864155270044, -0.8200019859739, -0.74012419157855, -0.64809365193698, -0.54542147138884, -0.43379350762605, -0.31504267969616, -0.19111886747362, -0.064056892862606, 0.06405689286261, 0.19111886747362, 0.31504267969616, 0.43379350762605, 0.54542147138884, 0.64809365193698, 0.74012419157855, 0.8200019859739, 0.8864155270044, 0.93827455200273, 0.97472855597131, 0.99518721999702};

  // w_i
  double gauss_legendre_weight[gauss_pts] = {0.01234122979999, 0.02853138862893, 0.0442774388174, 0.059298584915437, 0.0733464814111, 0.08619016153195, 0.0976186521041, 0.107444270116, 0.11550566805373, 0.1216704729278, 0.12583745634683, 0.1279381953468, 0.1279381953468, 0.1258374563468, 0.1216704729278, 0.1155056680537, 0.107444270116, 0.09761865210411, 0.08619016153195, 0.07334648141108, 0.05929858491544, 0.04427743881742, 0.02853138862893, 0.01234122979999};


//   double gauss_legendre_root[gauss_pts] = {-0.99877100725243,-0.99353017226635,-0.98412458372283,-0.97059159254625,-0.95298770316043,-0.93138669070655,-0.90587913671557,-0.8765720202742,-0.84358826162439,-0.80706620402944,-0.76715903251574,-0.72403413092382,-0.67787237963266,-0.62886739677651,-0.57722472608397,-0.52316097472223,-0.46690290475096,-0.40868648199072,-0.34875588629216,-0.28736248735546,-0.22476379039469,-0.16122235606889,-0.09700469920946,-0.032380170962869,0.03238017096287,0.0970046992095,0.1612223560689,0.2247637903947,0.2873624873555,0.3487558862922,0.40868648199072,0.46690290475096,0.52316097472223,0.57722472608397,0.62886739677651,0.67787237963266,0.72403413092382,0.76715903251574,0.80706620402944,0.84358826162439,
// 0.87657202027425,0.90587913671557,0.93138669070655,0.95298770316043,0.97059159254625,0.98412458372283,0.99353017226635,0.99877100725243};

//   double gauss_legendre_weight[gauss_pts] = {0.00315334605231,0.00732755390128,0.01147723457923,0.01557931572294,0.019616160457356,0.02357076083932,0.0274265097084,0.0311672278328,0.0347772225648,0.03824135106583,0.04154508294346,0.04467456085669,0.04761665849249,0.05035903555385,0.052890189485194,0.05519950369998,0.057277292100403,0.0591148396984,0.06070443916589,
// 0.06203942315989,0.06311419228625,0.06392423858465,0.06446616443595,0.0647376968127,0.0647376968127,0.064466164436,0.06392423858465,0.06311419228625,0.06203942315989,0.060704439165894,
// 0.0591148396984,0.0572772921004,0.05519950369998,0.0528901894852,0.0503590355539,0.0476166584925,0.044674560856694,0.04154508294346,0.03824135106583,0.03477722256477,0.0311672278328,0.0274265097084,0.02357076083932,0.019616160457356,0.01557931572294,0.01147723457923,0.00732755390128,0.003153346052306};


  double mass_squared = mass * mass;

  double mbar = mass / T;
  double mbar_squared = mbar * mbar;

  double chem = baryon * alphaB;

  // pimunu LRF components
  double pixx_LRF = pimunu.pixx_LRF;  
  double piyy_LRF = pimunu.piyy_LRF;
  double pizz_LRF = pimunu.pizz_LRF;
  double pixz_LRF = pimunu.pixz_LRF;  


  // Lipei: include Vmu LRF components
  // ???



  // Lipei: fill in diffusion coefficents here (they're in df_coeff array)

  // df_coeff is one of them
  double c0 = df_coeff[0];    double F = df_coeff[0];
  double c1 = df_coeff[1];    double G = df_coeff[1];
  double c2 = df_coeff[2];    double betabulk = df_coeff[2];
                              double betapi = df_coeff[4];





  // feqmod terms (Lipei doesn't care about feqmod)
  double bulk_coeff = bulkPi / (3.0 * betabulk);
  double mbar_mod = mass / T_mod;
  double chem_mod = baryon * alphaB_mod;

  double n_linear = equilibrium_density  +  bulkPi * bulk_density;
  double renorm = n_linear / modified_density;







  // dsigma terms
  double ds_time = dsigma.dsigmat_LRF;                  // ds_t
  double ds_space = dsigma.dsigma_space;                // |ds_i|
  double ds_time_over_ds_space = ds_time / ds_space;

  // rotation angles
  double costheta_ds = dsigma.costheta_LRF;             // polar angle of ds_i 3-vector
  double costheta_ds2 = costheta_ds * costheta_ds;
  double sintheta_ds = sqrt(fabs(1.0 - costheta_ds2));

  // temporary test
  // pizz_LRF = -0.008934/0.014314/sqrt(1.5)*shear14_coeff;
  // pixx_LRF = - 0.5 * pizz_LRF;
  // piyy_LRF = pixx_LRF;
  // pixz_LRF = 0.0;

  // ds_time_over_ds_space = 1.0/1.4;
  // ds_space = 1.4;
  // costheta_ds = 1.0;
  // costheta_ds2 = 1.0;
  // sintheta_ds = 0.0;

  // particle density with outflow and corrections
  double n_outflow = 0.0;

  // radial momentum integration routine
  for(int i = 0; i < gauss_pts; i++)
  {
    double x = gauss_legendre_root[i];  // x should never be 1.0 (or else pbar = inf)
    double s = (1.0 - x) / 2.0;         // variable transformations
    double pbar = (1.0 - s) / s;

    switch(DF_MODE)
    {
      case 1: // 14 moment
      {
        double Ebar = sqrt(pbar * pbar +  mbar_squared);
        double E = Ebar * T;
        double p = pbar * T;

        double feq = 1.0 / (exp(Ebar - chem) + sign);
        double feqbar = 1.0 - sign * feq;


        // timelike cells
        if(!OUTFLOW || ds_time_over_ds_space >= 1.0)
        {
          // easy formula (eq. 30 for feq + df_bulk)




          double df_bulk = ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2 - c0) * E) * E) * bulkPi;

          double df = feqbar * (df_bulk);

          df = max(-1.0, min(df, 1.0)); // regulate df <= |feq|

          n_outflow += gauss_legendre_weight[i] * ds_time_over_ds_space * pbar * pbar * feq * (1.0 + df) / (s * s);




          // n_outflow = sum of two terms (one is a * (feq + df_bulk)  +  b * (df_diff))

          // a ~ u.ds
          // b ~ V.ds

        }
        // spacelike cells
        else
        {
          double costheta_star = min(1.0, Ebar * ds_time_over_ds_space / pbar);     // cosO*

          double costheta_star2 = costheta_star * costheta_star;
          double costheta_star3 = costheta_star2 * costheta_star;
          double costheta_star4 = costheta_star3 * costheta_star;

          // (phi, costheta) integrated feq + df time and space terms




          double feq_time = feq * (1.0 + costheta_star);
          double feq_space = 0.5 * feq * (costheta_star2 - 1.0);





          double df_bulk = feq * feqbar * ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2 - c0) * E) * E) * bulkPi;
          double df_bulk_time = df_bulk * (1.0 + costheta_star);
          double df_bulk_space = 0.5 * df_bulk * (costheta_star2 - 1.0);





          double df_shear_time = 0.5 * feq * feqbar / shear14_coeff * p * p * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star - costheta_star3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star3) / 3.0)  +  pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);

          double df_shear_space = 0.5 * feq * feqbar / shear14_coeff * p * p * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star2 - 1.0) * (costheta_star2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star4 - 1.0))  +  pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);







          double f_time = feq_time + df_shear_time + df_bulk_time;

          double f_space = feq_space + df_shear_space + df_bulk_space; 



          double integrand = ds_time_over_ds_space * f_time  -  pbar / Ebar * f_space;





          double integrand_upper_bound = 2.0 * (ds_time_over_ds_space * feq_time  -  pbar / Ebar * feq_space);

          integrand = max(0.0, min(integrand, integrand_upper_bound));  // regulate integrand (corresponds to regulating df)




          n_outflow += 0.5 * gauss_legendre_weight[i] * pbar * pbar * integrand / (s * s);

        }

        break;
      }
      case 2: // Chapman Enskog
      {
        chapman_enskog:

        double Ebar = sqrt(pbar * pbar +  mbar_squared);
        double E = Ebar * T;
        double p = pbar * T;

        double feq = 1.0 / (exp(Ebar - chem) + sign);
        double feqbar = 1.0 - sign * feq;

        // timelike cells
        if(!OUTFLOW || ds_time_over_ds_space >= 1.0)
        {
          double df_bulk = (F / T * Ebar  +  baryon * G  +  (Ebar  -  mbar_squared / Ebar) / 3.0) * bulkPi / betabulk;

          double df = feqbar * df_bulk;

          df = max(-1.0, min(df, 1.0));

          n_outflow += gauss_legendre_weight[i] * ds_time_over_ds_space * pbar * pbar * feq * (1.0 + df) / (s * s);
        }
        // spacelike cells
        else
        {
          double costheta_star = min(1.0, Ebar * ds_time_over_ds_space / pbar);
          double costheta_star2 = costheta_star * costheta_star;
          double costheta_star3 = costheta_star2 * costheta_star;
          double costheta_star4 = costheta_star3 * costheta_star;

          // (phi, costheta) integrated feq + df time and space terms
          double feq_time = feq * (1.0 + costheta_star);
          double feq_space = 0.5 * feq * (costheta_star2 - 1.0);

          
          double df_bulk = feq * feqbar * (F / T * Ebar  +  baryon * G  +  (Ebar  -  mbar_squared / Ebar) / 3.0) * bulkPi / betabulk;
          double df_bulk_time = df_bulk * (1.0 + costheta_star);
          double df_bulk_space = 0.5 * df_bulk * (costheta_star2 - 1.0);


          double df_shear_time = 0.25 * feq * feqbar / Ebar / betapi * pbar * pbar * (pixx_LRF * (costheta_ds2 * (1.0 + costheta_star)  +  (2.0 - 3.0 * costheta_ds2) * (1.0 + costheta_star3) / 3.0)  +  piyy_LRF * (2.0 / 3.0 + costheta_star - costheta_star3 / 3.0)  +  pizz_LRF * ((1.0 - costheta_ds2) * (1.0 + costheta_star)  +  (3.0 * costheta_ds2 - 1.0) * (1.0 + costheta_star3) / 3.0)  +  pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);

          double df_shear_space = 0.25 * feq * feqbar / Ebar / betapi * pbar * pbar * (pixx_LRF * (0.5 * costheta_ds2 * (costheta_star2 - 1.0)  +  0.25 * (2.0 - 3.0 * costheta_ds2) * (costheta_star4 - 1.0))  -  0.25 * piyy_LRF * (costheta_star2 - 1.0) * (costheta_star2 - 1.0)  +  pizz_LRF * (0.5 * (1.0 - costheta_ds2) * (costheta_star2 - 1.0)  +  0.25 * (3.0 * costheta_ds2 - 1.0) * (costheta_star4 - 1.0))  +  pixz_LRF * costheta_star * (costheta_star2 - 1.0) * sintheta_ds * costheta_ds);


          double f_time = feq_time + df_shear_time + df_bulk_time;
          double f_space = feq_space + df_shear_space + df_bulk_space; 

          double integrand = ds_time_over_ds_space * f_time  -  pbar / Ebar * f_space;
          double integrand_upper_bound = 2.0 * (ds_time_over_ds_space * feq_time  -  pbar / Ebar * feq_space);

          integrand = max(0.0, min(integrand, integrand_upper_bound));  // regulate integrand (corresponds to regulating df)

          n_outflow += 0.5 * gauss_legendre_weight[i] * pbar * pbar * integrand / (s * s);
        }

        break;
      }
      case 3: // modified (not yet complete)
      {
        if(feqmod_breaks_down) goto chapman_enskog;

        double pbar_mod = pbar; // use the root
        double Ebar_mod = sqrt(pbar_mod * pbar_mod  +  mbar_mod * mbar_mod);

        double feqmod = 1.0 / (exp(Ebar_mod - chem_mod) + sign);  // effectively..

        double Ebar = sqrt(pbar * pbar +  mbar_squared);

        if(!OUTFLOW || ds_time_over_ds_space >= 1.0)
        {
          n_outflow += gauss_legendre_weight[i] * ds_time_over_ds_space * pbar_mod * pbar_mod * feqmod / (s * s);
        }
        else
        {
          double costheta_star_mod = min(1.0, Ebar_mod * ds_time_over_ds_space / pbar);
          double mbar_bulk = mbar_mod / (1.0 + bulk_coeff);
          double Ebar_bulk = sqrt(pbar_mod * pbar_mod  +  mbar_bulk * mbar_bulk);

          n_outflow += 0.5 * gauss_legendre_weight[i] * (ds_time_over_ds_space * pbar_mod * pbar_mod * (1.0  +  costheta_star_mod) - 0.5 * pbar_mod * pbar_mod * pbar_mod / Ebar_bulk * (costheta_star_mod * costheta_star_mod  -  1.0)) * feqmod / (s * s);
        }

        break;
      }
      default:
      {
        printf("\nParticle density outflow error: please set df_mode = (1,2,3)\n"); exit(-1);
      }
    }

  }


  // resume


  // external prefactor = degeneracy.ds_space.T^3 / (4.pi^2.hbar^3) (or T_mod^3)
  if(DF_MODE == 3 && !feqmod_breaks_down)
  {
    return n_outflow * degeneracy * ds_space * detA * renorm * T_mod * T_mod * T_mod / four_pi2_hbarC3;
  }
  else
  {
    //cout << setprecision(15) << n_outflow * degeneracy * ds_space * T * T * T / four_pi2_hbarC3 << endl;
    //exit(-1);


    return n_outflow * degeneracy * ds_space * T * T * T / four_pi2_hbarC3;


  }

}


double EmissionFunctionArray::compute_df_weight(LRF_Momentum pLRF, double mass_squared, double sign, double baryon, double T, double alphaB, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear14_coeff, double baryon_enthalpy_ratio)
{
  // pLRF components
  double E = pLRF.E;    double py = pLRF.py;
  double px = pLRF.px;  double pz = pLRF.pz;

  // pimunu LRF components
  double pixx = pimunu.pixx_LRF;  double piyy = pimunu.piyy_LRF;
  double pixy = pimunu.pixy_LRF;  double piyz = pimunu.piyz_LRF;
  double pixz = pimunu.pixz_LRF;  double pizz = pimunu.pizz_LRF;

  double pimunu_pmu_pnu = px * px * pixx  +  py * py * piyy  +  pz * pz * pizz  +  2.0 * (px * py * pixy  +  px * pz * pixz  +  py * pz * piyz);

  // Vmu LRF components
  double Vx = Vmu.Vx_LRF;
  double Vy = Vmu.Vy_LRF;
  double Vz = Vmu.Vz_LRF;

  double Vmu_pmu = - (px * Vx  +  py * Vy  +  pz * Vz);

  // df corrections
  double df_shear = 0.0;
  double df_bulk = 0.0;
  double df_diff = 0.0;

  switch(DF_MODE)
  {
    case 1: // 14 moment
    {
      double c0 = df_coeff[0];
      double c1 = df_coeff[1];
      double c2 = df_coeff[2];
      double c3 = df_coeff[3];
      double c4 = df_coeff[4];

      df_shear = pimunu_pmu_pnu / shear14_coeff;
      df_bulk = ((c0 - c2) * mass_squared  +  (baryon * c1  +  (4.0 * c2  -  c0) * E) * E) * bulkPi;
      df_diff = (baryon * c3  +  c4 * E) * Vmu_pmu;
      break;
    }
    case 2: // Chapman Enskog
    {
      double F = df_coeff[0];
      double G = df_coeff[1];
      double betabulk = df_coeff[2];
      double betaV = df_coeff[3];
      double betapi = df_coeff[4];

      df_shear = pimunu_pmu_pnu / (2.0 * E * betapi * T);
      df_bulk = (baryon * G  +  F * E / T / T  +  (E  -  mass_squared / E) / (3.0 * T)) * bulkPi / betabulk;
      df_diff = (baryon_enthalpy_ratio  -  baryon / E) * Vmu_pmu / betaV;
      break;
    }
    default:
    {
      //
    }
  } // DF_MODE

  double chem = baryon * alphaB;
  double feqbar = 1.0 - sign / (exp(E/T - chem) + sign);   // 1 - sign.feq/degeneracy

  double df = feqbar * (df_shear + df_bulk + df_diff);

  df = max(-1.0, min(df, 1.0));  // regulate |df| <= 1

  return (1.0 + df) / 2.0;
}

LRF_Momentum EmissionFunctionArray::sample_momentum(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T, double alphaB, Surface_Element_Vector dsigma, Shear_Stress pimunu, double bulkPi, Baryon_Diffusion Vmu, double * df_coeff, double shear14_coeff, double baryon_enthalpy_ratio)
{
  double two_pi = 2.0 * M_PI;
  double mass_squared = mass * mass;
  double chem = baryon * alphaB;

  // uniform distributions in (phi,costheta) on standby:
  uniform_real_distribution<double> phi_distribution(0.0, two_pi);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;


  LRF_Momentum pLRF;


  bool rejected = true;

  if(mass / T < 1.67)
  {
    while(rejected)
    {
      *samples = (*samples) + 1;
      // draw (p,phi,costheta) from p^2.exp(-p/T).dp.dphi.dcostheta by sampling (r1,r2,r3)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p = - T * (l1 + l2 + l3);
      double phi = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
      double costheta = (l1 - l2) / (l1 + l2);

      double E = sqrt(p * p  +  mass_squared);
      double sintheta = sqrt(1.0  -  costheta * costheta);

      // if(std::isnan(p)) printf("Error: p is nan\n");
      // if(std::isnan(phi)) printf("Error: phi is nan\n");
      // if(std::isnan(costheta)) printf("Error: costheta is nan\n");
      // if(std::isnan(E)) printf("Error: E is nan\n");
      // if(std::isnan(sintheta)) printf("Error: sintheta is nan\n");
      // if(p < 0.0) printf("Error: p = %lf is out of bounds\n", p);
      // if(!(phi >= 0.0 && phi < two_pi)) printf("Error: phi = %lf is out of bounds\n", phi);
      // if(fabs(costheta) > 1.0) printf("Error: costheta = %lf is out of bounds\n", costheta);

      pLRF.E = E;
      pLRF.px = p * sintheta * cos(phi);
      pLRF.py = p * sintheta * sin(phi);
      pLRF.pz = p * costheta;

      // p.dsigma * Heaviside(p.dsigma)
      double pdsigma_Heaviside = max(0.0, E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

      double rideal = pdsigma_Heaviside / (E * ds_magnitude);
      double rvisc = 1.0;

      if(INCLUDE_SHEAR_DELTAF || INCLUDE_BULK_DELTAF || INCLUDE_BARYONDIFF_DELTAF)
      {
        rvisc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);
      }

      double weight_light = exp(p/T) / (exp(E/T) + sign) * rideal * rvisc;
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_light - 0.5) > 0.5) printf("Error: weight_light = %lf out of bounds\n", weight_light);

      // check pLRF acceptance
      if(propose < weight_light) break;

    } // rejection loop

  } // sampling pions

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // integrated weights
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T;
    double I3 = 2.0 * T * T;
    double Itot = I1 + I2 + I3;

    double I1_over_Itot = I1 / Itot;
    double I1_plus_I2_over_Itot = (I1 + I2) / Itot;

    while(rejected)
    {
      *samples = (*samples) + 1;

      double k, phi, costheta;   // kinetic energy, angles

      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // select distribution to sample from based on integrated weights
      if(propose_distribution < I1_over_Itot)
      {
        // draw k from exp(-k/T).dk by sampling r1
        // sample direction uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k = - T * log(r1);
        phi = phi_distribution(generator);
        costheta = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      else if(propose_distribution < I1_plus_I2_over_Itot)
      {
        // draw (k,phi) from k.exp(-k/T).dk.dphi by sampling (r1,r2)
        // sample costheta uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k = - T * (l1 + l2);
        phi = two_pi * l1 / (l1 + l2);
        costheta = costheta_distribution(generator);

      } // distribution 2 (moderately heavy)
      else
      {
        // draw (k,phi,costheta) from k^2.exp(-k/T).dk.dphi.dcostheta by sampling (r1,r2,r3)
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k = - T * (l1 + l2 + l3);
        phi = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
        costheta = (l1 - l2) / (l1 + l2);

      } // distribution 3 (light)

      double E = k + mass;
      double p = sqrt(E * E  -  mass_squared);
      double sintheta = sqrt(1.0  -  costheta * costheta);

      // if(std::isnan(k)) printf("Error: k is nan\n");
      // if(std::isnan(phi)) printf("Error: phi is nan\n");
      // if(std::isnan(costheta)) printf("Error: costheta is nan\n");
      // if(std::isnan(E)) printf("Error: E is nan\n");
      // if(std::isnan(p)) printf("Error: p is nan\n");
      // if(std::isnan(sintheta)) printf("Error: sintheta is nan\n");
      // if(k < 0.0) printf("Error: k = %lf is out of bounds\n", k);
      // if(!(phi >= 0.0 && phi < two_pi)) printf("Error: phi = %lf is out of bounds\n", phi);
      // if(fabs(costheta) > 1.0) printf("Error: costheta = %lf is out of bounds\n", costheta);

      pLRF.E = E;
      pLRF.px = p * sintheta * cos(phi);
      pLRF.py = p * sintheta * sin(phi);
      pLRF.pz = p * costheta;

      // p.dsigma * Heaviside(p.dsigma)
      double pdsigma_Heaviside = max(0.0, E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);

      double rideal = pdsigma_Heaviside / (E * ds_magnitude);
      double rvisc = 1.0;

      if(INCLUDE_SHEAR_DELTAF || INCLUDE_BULK_DELTAF || INCLUDE_BARYONDIFF_DELTAF)
      {
        rvisc = compute_df_weight(pLRF, mass_squared, sign, baryon, T, alphaB, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);
      }

      double weight_heavy = p/E * exp(E/T - chem) / (exp(E/T - chem) + sign) * rideal * rvisc;
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_heavy - 0.5) > 0.5) printf("Error: weight_heavy = %f out of bounds\n", weight_heavy);

      // check pLRF acceptance
      if(propose < weight_heavy) break;

    } // rejection loop

  } // sampling heavy hadrons

  *acceptances = (*acceptances) + 1;

  return pLRF;

}


LRF_Momentum EmissionFunctionArray::rescale_momentum(LRF_Momentum pmod, double mass_squared, double baryon, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio)
{
    double E_mod = pmod.E;    double py_mod = pmod.py;
    double px_mod = pmod.px;  double pz_mod = pmod.pz;

    // LRF shear stress components
    double pixx = pimunu.pixx_LRF;  double piyy = pimunu.piyy_LRF;
    double pixy = pimunu.pixy_LRF;  double piyz = pimunu.piyz_LRF;
    double pixz = pimunu.pixz_LRF;  double pizz = pimunu.pizz_LRF;

    // LRF baryon diffusion components
    double Vx = Vmu.Vx_LRF;
    double Vy = Vmu.Vy_LRF;
    double Vz = Vmu.Vz_LRF;
    diff_coeff *= (E_mod * baryon_enthalpy_ratio  +  baryon);

    LRF_Momentum pLRF;

    // local momentum transformation
    pLRF.px = (1.0 + bulk_coeff) * px_mod  +  shear_coeff * (pixx * px_mod  +  pixy * py_mod  +  pixz * pz_mod)  +  diff_coeff * Vx;
    pLRF.py = (1.0 + bulk_coeff) * py_mod  +  shear_coeff * (pixy * px_mod  +  piyy * py_mod  +  piyz * pz_mod)  +  diff_coeff * Vy;
    pLRF.pz = (1.0 + bulk_coeff) * pz_mod  +  shear_coeff * (pixz * px_mod  +  piyz * py_mod  +  pizz * pz_mod)  +  diff_coeff * Vz;
    pLRF.E = sqrt(mass_squared  +  pLRF.px * pLRF.px  +  pLRF.py * pLRF.py  +  pLRF.pz * pLRF.pz);

    return pLRF;
}


LRF_Momentum EmissionFunctionArray::sample_momentum_feqmod(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double baryon, double T_mod, double alphaB_mod, Surface_Element_Vector dsigma, Shear_Stress pimunu, Baryon_Diffusion Vmu, double shear_coeff, double bulk_coeff, double diff_coeff, double baryon_enthalpy_ratio)
{
  double two_pi = 2.0 * M_PI;
  double mass_squared = mass * mass;
  double chem_mod = baryon * alphaB_mod;

   // uniform distributions in (phi_mod, costheta_mod) on standby
  uniform_real_distribution<double> phi_distribution(0.0, two_pi);
  uniform_real_distribution<double> costheta_distribution(-1.0, nextafter(1.0, numeric_limits<double>::max()));

  // dsigma LRF components
  double dst = dsigma.dsigmat_LRF;
  double dsx = dsigma.dsigmax_LRF;
  double dsy = dsigma.dsigmay_LRF;
  double dsz = dsigma.dsigmaz_LRF;
  double ds_magnitude = dsigma.dsigma_magnitude;

  LRF_Momentum pLRF;                          // LRF momentum
  LRF_Momentum pLRF_mod;                      // modified LRF momentum

  bool rejected = true;

  if(mass / T_mod < 1.67)   // the modified temperature increases, so it's possible heavy hadrons can be effectively light
  {
    if(mass > 0.140) printf("Error: heavy hadrons are effectively light\n"); // is this a big deal if it happens?
    while(rejected)
    {
      *samples = (*samples) + 1;
      // draw (p_mod, phi_mod, costheta_mod) from p_mod^2.exp(-p_mod/T_mod).dp_mod.dphi_mod.dcostheta_mod by sampling (r1,r2,r3)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p_mod = - T_mod * (l1 + l2 + l3);                        // modified momentum magnitude
      double phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);   // modified angles
      double costheta_mod = (l1 - l2) / (l1 + l2);

      double E_mod = sqrt(p_mod * p_mod  +  mass_squared);            // modified energy
      double sintheta_mod = sqrt(1.0 - costheta_mod * costheta_mod);

      // modified pLRF components
      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // local momentum transformation
      pLRF = rescale_momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

      // p.dsigma * Heaviside(p.dsigma)
      double pdsigma_Heaviside = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);
      double rideal = pdsigma_Heaviside / (pLRF.E * ds_magnitude);

      double weight_light = rideal * exp(p_mod/T_mod - chem_mod) / (exp(E_mod/T_mod - chem_mod) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_light - 0.5) > 0.5) printf("Error: weight = %f out of bounds\n", weight_light);

      // check pLRF acceptance
      if(propose < weight_light) break;

    } // rejection loop

  } // sampling pions

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // modified integrated weights
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T_mod;
    double I3 = 2.0 * T_mod * T_mod;
    double Itot = I1 + I2 + I3;

    double I1_over_Itot = I1 / Itot;
    double I1_plus_I2_over_Itot = (I1 + I2) / Itot;

    while(rejected)
    {
      double k_mod, phi_mod, costheta_mod;    // modified kinetic energy and angles

      *samples = (*samples) + 1;

      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // select distribution to sample from based on integrated weights
      if(propose_distribution < I1_over_Itot)
      {
        // draw k_mod from exp(-k_mod/T_mod).dk_mod by sampling r1
        // sample modified direction uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k_mod = - T_mod * log(r1);
        phi_mod = phi_distribution(generator);
        costheta_mod = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      else if(propose_distribution < I1_plus_I2_over_Itot)
      {
        // draw (k_mod, phi_mod) from k_mod.exp(-k_mod/T_mod).dk_mod.dphi_mod by sampling (r1,r2)
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
        // draw (k_mod,phi_mod,costheta_mod) from k_mod^2.exp(-k_mod/T_mod).dk_mod.dphi_mod.dcostheta_mod by sampling (r1,r2,r3)
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
      double p_mod = sqrt(E_mod * E_mod  -  mass_squared);
      double sintheta_mod = sqrt(1.0  -  costheta_mod * costheta_mod);

      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // local momentum transformation
      pLRF = rescale_momentum(pLRF_mod, mass_squared, baryon, pimunu, Vmu, shear_coeff, bulk_coeff, diff_coeff, baryon_enthalpy_ratio);

      // p.dsigma * Heaviside(p.dsigma)
      double pdsigma_Heaviside = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);
      double rideal = pdsigma_Heaviside / (pLRF.E * ds_magnitude);

      double weight_heavy = rideal * (p_mod/E_mod) * exp(E_mod/T_mod - chem_mod) / (exp(E_mod/T_mod - chem_mod) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_heavy - 0.5) > 0.5) printf("Error: weight = %f out of bounds\n", weight_heavy);

      // check pLRF acceptance
      if(propose < weight_heavy) break;

    } // rejection loop

  } // sampling heavy hadrons

  *acceptances = (*acceptances) + 1;

  return pLRF;

}


LRF_Momentum EmissionFunctionArray::rescale_momentum_jonah(LRF_Momentum pmod, double mass_squared, Shear_Stress pimunu, double shear_coeff, double lambda)
{
    double E_mod = pmod.E;    double py_mod = pmod.py;
    double px_mod = pmod.px;  double pz_mod = pmod.pz;

    // LRF shear stress components
    double pixx = pimunu.pixx_LRF;  double piyy = pimunu.piyy_LRF;
    double pixy = pimunu.pixy_LRF;  double piyz = pimunu.piyz_LRF;
    double pixz = pimunu.pixz_LRF;  double pizz = pimunu.pizz_LRF;

    LRF_Momentum pLRF;

    // local momentum transformation
    pLRF.px = (1.0 + lambda) * px_mod  +  shear_coeff * (pixx * px_mod  +  pixy * py_mod  +  pixz * pz_mod);
    pLRF.py = (1.0 + lambda) * py_mod  +  shear_coeff * (pixy * px_mod  +  piyy * py_mod  +  piyz * pz_mod);
    pLRF.pz = (1.0 + lambda) * pz_mod  +  shear_coeff * (pixz * px_mod  +  piyz * py_mod  +  pizz * pz_mod);
    pLRF.E = sqrt(mass_squared  +  pLRF.px * pLRF.px  +  pLRF.py * pLRF.py  +  pLRF.pz * pLRF.pz);

    return pLRF;
}



LRF_Momentum EmissionFunctionArray::sample_momentum_feqmod_jonah(default_random_engine& generator, long * acceptances, long * samples, double mass, double sign, double T, Surface_Element_Vector dsigma, Shear_Stress pimunu, double shear_coeff, double lambda)
{
  double two_pi = 2.0 * M_PI;
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

  LRF_Momentum pLRF;                          // LRF momentum
  LRF_Momentum pLRF_mod;                      // modified LRF momentum

  bool rejected = true;

  if(mass / T < 1.67)
  {
    while(rejected)
    {
      *samples = (*samples) + 1;
      // draw (p_mod, phi_mod, costheta_mod) from p_mod^2.exp(-p_mod/T).dp_mod.dphi_mod.dcostheta_mod by sampling (r1,r2,r3)
      double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
      double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

      double l1 = log(r1);
      double l2 = log(r2);
      double l3 = log(r3);

      double p_mod = - T * (l1 + l2 + l3);                            // modified momentum magnitude
      double phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);   // modified angles
      double costheta_mod = (l1 - l2) / (l1 + l2);

      double E_mod = sqrt(p_mod * p_mod  +  mass_squared);            // modified energy
      double sintheta_mod = sqrt(1.0 - costheta_mod * costheta_mod);

      // modified pLRF components
      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // local momentum transformation
      pLRF = rescale_momentum_jonah(pLRF_mod, mass_squared, pimunu, shear_coeff, lambda);

      // p.dsigma * Heaviside(p.dsigma)
      double pdsigma_Heaviside = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);
      double rideal = pdsigma_Heaviside / (pLRF.E * ds_magnitude);

      double weight_light = rideal * exp(p_mod/T) / (exp(E_mod/T) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_light - 0.5) > 0.5) printf("Error: weight = %f out of bounds\n", weight_light);

      // check pLRF acceptance
      if(propose < weight_light) break;

    } // rejection loop

  } // sampling pions

  // heavy hadrons (use variable transformation described in LongGang's notes)
  else
  {
    // modified integrated weights
    double I1 = mass_squared;
    double I2 = 2.0 * mass * T;
    double I3 = 2.0 * T * T;
    double Itot = I1 + I2 + I3;

    double I1_over_Itot = I1 / Itot;
    double I1_plus_I2_over_Itot = (I1 + I2) / Itot;

    while(rejected)
    {
      double k_mod, phi_mod, costheta_mod;    // modified kinetic energy and angles

      *samples = (*samples) + 1;

      double propose_distribution = generate_canonical<double, numeric_limits<double>::digits>(generator);

      // select distribution to sample from based on integrated weights
      if(propose_distribution < I1_over_Itot)
      {
        // draw k_mod from exp(-k_mod/T).dk_mod by sampling r1
        // sample modified direction uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        k_mod = - T * log(r1);
        phi_mod = phi_distribution(generator);
        costheta_mod = costheta_distribution(generator);

      } // distribution 1 (very heavy)
      else if(propose_distribution < I1_plus_I2_over_Itot)
      {
        // draw (k_mod, phi_mod) from k_mod.exp(-k_mod/T).dk_mod.dphi_mod by sampling (r1,r2)
        // sample costheta_mod uniformly
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);

        k_mod = - T * (l1 + l2);
        phi_mod = two_pi * l1 / (l1 + l2);
        costheta_mod = costheta_distribution(generator);

      } // distribution 2 (moderately heavy)
      else
      {
        // draw (k_mod,phi_mod,costheta_mod) from k_mod^2.exp(-k_mod/T).dk_mod.dphi_mod.dcostheta_mod by sampling (r1,r2,r3)
        double r1 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r2 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);
        double r3 = 1.0 - generate_canonical<double, numeric_limits<double>::digits>(generator);

        double l1 = log(r1);
        double l2 = log(r2);
        double l3 = log(r3);

        k_mod = - T * (l1 + l2 + l3);
        phi_mod = two_pi * pow((l1 + l2) / (l1 + l2 + l3), 2);
        costheta_mod = (l1 - l2) / (l1 + l2);

      } // distribution 3 (light)

      double E_mod = k_mod + mass;
      double p_mod = sqrt(E_mod * E_mod  -  mass_squared);
      double sintheta_mod = sqrt(1.0  -  costheta_mod * costheta_mod);

      pLRF_mod.E = E_mod;
      pLRF_mod.px = p_mod * sintheta_mod * cos(phi_mod);
      pLRF_mod.py = p_mod * sintheta_mod * sin(phi_mod);
      pLRF_mod.pz = p_mod * costheta_mod;

      // local momentum transformation
      pLRF = rescale_momentum_jonah(pLRF_mod, mass_squared, pimunu, shear_coeff, lambda);

      // p.dsigma * Heaviside(p.dsigma)
      double pdsigma_Heaviside = max(0.0, pLRF.E * dst  -  pLRF.px * dsx  -  pLRF.py * dsy  -  pLRF.pz * dsz);
      double rideal = pdsigma_Heaviside / (pLRF.E * ds_magnitude);

      double weight_heavy = rideal * (p_mod/E_mod) * exp(E_mod/T) / (exp(E_mod/T) + sign);
      double propose = generate_canonical<double, numeric_limits<double>::digits>(generator);

      if(fabs(weight_heavy - 0.5) > 0.5) printf("Error: weight = %f out of bounds\n", weight_heavy);

      // check pLRF acceptance
      if(propose < weight_heavy) break;

    } // rejection loop

  } // sampling heavy hadrons

  *acceptances = (*acceptances) + 1;

  return pLRF;

}


double EmissionFunctionArray::calculate_total_yield(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *tau_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *thermodynamic_average, double * df_coeff, const int pbar_pts, double * pbar_root1, double * pbar_weight1, double * pbar_root2, double * pbar_weight2)
  {
    printf("Estimated particle yield = ");

    double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);
    double four_pi2_hbarC3 = 4.0 * pow(M_PI,2) * pow(hbarC,3);

    double detA_min = DETA_MIN; // default value

    int npart = number_of_chosen_particles;

    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;
    double etaTrapezoidWeights[eta_pts];                                  // eta_weight * delta_eta
    double delta_eta = fabs((eta_tab->get(1,2)) - (eta_tab->get(1,1)));   // assume uniform grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta_weight = eta_tab->get(2, ieta + 1);
        etaTrapezoidWeights[ieta] = eta_weight * delta_eta;
      }
    }
    else if(DIMENSION == 3)
    {
      etaTrapezoidWeights[0] = 1.0; // 1.0 for 3+1d
    }

    double Tavg = thermodynamic_average[0];
    double Eavg = thermodynamic_average[1];
    double Pavg = thermodynamic_average[2];
    double muBavg = thermodynamic_average[3];

    // set df coefficients
    double c0, c1, c2, c3, c4;            // 14 moment
    double F, G, betabulk, betaV, betapi; // Chapman Enskog

    //deltaf_coefficients df;

    double shear14_coeff = 2.0 * (Eavg + Pavg) * Tavg * Tavg;

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

    double mbar_pion0 = MASS_PION0 / Tavg;
    double T3_over_two_pi2_hbar3 = pow(Tavg, 3) / two_pi2_hbarC3;
    double T4_over_two_pi2_hbar3 = pow(Tavg, 4) / two_pi2_hbarC3;

    double neq_pion0 = T3_over_two_pi2_hbar3 * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_pion0, 0.0, 0.0, -1.0);
    double J20_pion0 = T4_over_two_pi2_hbar3 * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar_pion0, 0.0, 0.0, -1.0);

    //double neq_tot = 0.0;                 // total equilibrium number / udsigma
    //double dn_bulk_tot = 0.0;             // bulk correction / udsigma / bulkPi
    //double dn_diff_tot = 0.0;             // diffusion correction / Vdsigma

    /*
    for(int ipart = 0; ipart < npart; ipart++)
    {
      neq_tot += Equilibrium_Density[ipart];    // these are default densities if no outflow correction
      dn_bulk_tot += Bulk_Density[ipart];
      dn_diff_tot += Diffusion_Density[ipart];

    } // hadron species (ipart)
    */

    double Ntot = 0.0;                    // total particle yield

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

      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan; // udotdsigma / delta_eta_weight

      if(udsigma <= 0.0) continue;        // skip over cells with u.dsigma < 0

      double T = Tavg;                    // temperature (GeV)

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

      double muB = muBavg;                // baryon chemical potential (GeV)
      double alphaB = muB / T;
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu
      double Vx = 0.0;                    // enforce orthogonality
      double Vy = 0.0;
      double Vn = 0.0;

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        Vx = Vx_fo[icell];
        Vy = Vy_fo[icell];
        Vn = Vn_fo[icell];
        Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
      }

      // Vdotdsigma / delta_eta_weight
      double Vdsigma = Vt * dat  +  Vx * dax  +  Vy * day  +  Vn * dan;

      // milne basis class
      Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
      basis_vectors.test_orthonormality(tau2);

      // shear stress class
      Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
      pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);


      // feqmod modified temperature / chemical potential
      double T_mod = T  +  bulkPi * F / betabulk;
      double alphaB_mod = alphaB  +  bulkPi * G / betabulk;


      // feqmod breakdown switching criteria
      bool pion_density_negative = is_linear_pion0_density_negative(T, neq_pion0, J20_pion0, bulkPi, F, betabulk);

      double detA = compute_detA(pimunu, bulkPi, betapi, betabulk);
      double detA_bulk = pow(1.0 + bulkPi / (3.0 * betabulk), 3);

      if(pion_density_negative) detA_min = max(detA_min, detA_bulk);

      bool feqmod_breaks_down = does_feqmod_breakdown(detA, detA_min, pion_density_negative);

      double nmod_fact = detA * T_mod * T_mod * T_mod / two_pi2_hbarC3;


      // surface element class
      Surface_Element_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_magnitude();
      dsigma.compute_dsigma_lrf_polar_angle();


      // total number of hadrons / delta_eta_weight in FO_cell
      double dn_tot = 0.0;

      // sum over hadrons
      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];              // particle properties
        double degeneracy = Degeneracy[ipart];
        double sign = Sign[ipart];
        double baryon = Baryon[ipart];

        double equilibrium_density = Equilibrium_Density[ipart];
        double bulk_density = Bulk_Density[ipart];

        double modified_density = equilibrium_density;
        if(DF_MODE == 3)
        {
          double mbar_mod = mass / T_mod;

          modified_density = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);
        }

        dn_tot += particle_number_outflow(mass, degeneracy, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, df_coeff, shear14_coeff, equilibrium_density, bulk_density, T_mod, alphaB_mod, detA, modified_density, feqmod_breaks_down);
      }


      // add mean number of hadrons in FO cell to total yield
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        Ntot += dn_tot * etaTrapezoidWeights[ieta];
      }

    } // freezeout cells (icell)

    mean_yield = Ntot;
    printf("%lf\n", Ntot/14.0);

    return Ntot;

  }


void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, int *MCID, double *Equilibrium_Density, double *Bulk_Density, double *Diffusion_Density, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff, double *thermodynamic_average, const int pbar_pts, double * pbar_root1, double * pbar_weight1, double * pbar_root2, double * pbar_weight2)
  {
    int npart = number_of_chosen_particles;
    double two_pi2_hbarC3 = 2.0 * pow(M_PI,2) * pow(hbarC,3);
    double four_pi2_hbarC3 = 4.0 * pow(M_PI,2) * pow(hbarC,3);

    long feqmod_breakdown_count = 0;

    double detA_min = DETA_MIN;   // default value

    // set seed
    unsigned seed;
    if (SAMPLER_SEED < 0) seed = chrono::system_clock::now().time_since_epoch().count();
    else seed = SAMPLER_SEED;

    default_random_engine generator_poisson(seed);
    default_random_engine generator_type(seed);
    default_random_engine generator_momentum(seed);

    int eta_pts = 1;
    if(DIMENSION == 2) eta_pts = eta_tab_length;                   // extension in eta
    double etaValues[eta_pts];
    double etaTrapezoidWeights[eta_pts];                           // eta_weight * delta_eta
    double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform eta grid

    if(DIMENSION == 2)
    {
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaTrapezoidWeights[ieta] = (eta_tab->get(2, ieta + 1)) * fabs(delta_eta);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;           // below, will load eta_fo
      etaTrapezoidWeights[0] = 1.0; // 1.0 for 3+1d
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

    double shear14_coeff = 2.0 * Tavg * Tavg * (Eavg + Pavg);

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



    // calculate linear pion0 density terms (neq_pion0, J20_pion0)
    // this is for the function is_linear_pion0_density_negative()
    double mbar_pion0 = MASS_PION0 / Tavg;
    double T3_over_two_pi2_hbar3 = pow(Tavg, 3) / two_pi2_hbarC3;
    double T4_over_two_pi2_hbar3 = pow(Tavg, 4) / two_pi2_hbarC3;

    double neq_pion0 = T3_over_two_pi2_hbar3 * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_pion0, 0.0, 0.0, -1.0);
    double J20_pion0 = T4_over_two_pi2_hbar3 * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar_pion0, 0.0, 0.0, -1.0);




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

    //loop over all freezeout cells
    //#pragma omp parallel for
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
      double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un); // u^tau

      double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan; // udotdsigma / delta_eta_weight

      if(udsigma <= 0.0) continue;        // skip over cells with u.dsigma < 0

      double ut2 = ut * ut;               // useful expressions
      double ux2 = ux * ux;
      double uy2 = uy * uy;
      double uperp =  sqrt(ux * ux  +  uy * uy);
      double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

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
        pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
      }

      double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

      if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell];

      double muB = muBavg;                // baryon chemical potential (GeV)
      double alphaB = muB / T;            // muB / T
      double nB = nBavg;                  // net baryon density
      double baryon_enthalpy_ratio = nB / (E + P);
      double Vt = 0.0;                    // contravariant net baryon diffusion current V^mu (fm^-3)
      double Vx = 0.0;                    // enforce orthogonality V.u = 0
      double Vy = 0.0;
      double Vn = 0.0;
      double Vdsigma = 0.0;               // Vdotdsigma / delta_eta_weight

      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
      {
        Vx = Vx_fo[icell];
        Vy = Vy_fo[icell];
        Vn = Vn_fo[icell];
        Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
        Vdsigma = Vt * dat  +  Vx * dax  +  Vy * day  +  Vn * dan;
      }

      // milne basis class
      Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
      basis_vectors.test_orthonormality(tau2);

      // dsigma / delta_eta_weight class
      Surface_Element_Vector dsigma(dat, dax, day, dan);
      dsigma.boost_dsigma_to_lrf(basis_vectors, ut, ux, uy, un);
      dsigma.compute_dsigma_magnitude();
      dsigma.compute_dsigma_lrf_polar_angle();

      // shear stress class
      Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
      pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
      pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

      // baryon diffusion class
      Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
      Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
      Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);


      // feqmod coefficients
      double shear_mod_coeff = 0.5 / betapi;
      double bulk_mod_coeff = bulkPi / (3.0 * betabulk);
      double diff_mod_coeff = T / betaV;

      // modified temperature, chemical potential
      double T_mod = T  +  bulkPi * F / betabulk;
      double alphaB_mod = alphaB  +  bulkPi * G / betabulk;


      // feqmod breakdown switching criteria
      bool pion_density_negative = is_linear_pion0_density_negative(T, neq_pion0, J20_pion0, bulkPi, F, betabulk);

      double detA = compute_detA(pimunu, bulkPi, betapi, betabulk);
      double detA_bulk = pow(1.0 + bulkPi / (3.0 * betabulk), 3);

      if(pion_density_negative) detA_min = max(detA_min, detA_bulk);

      bool feqmod_breaks_down = does_feqmod_breakdown(detA, detA_min, pion_density_negative);

      double nmod_fact = detA * T_mod * T_mod * T_mod / two_pi2_hbarC3;


      // holds mean particle number / delta_eta_weight of all species
      std::vector<double> dn_list;
      dn_list.resize(npart);

      // total mean number  / delta_eta_weight of hadrons in FO cell
      double dn_tot = 0.0;

      for(int ipart = 0; ipart < npart; ipart++)
      {
        double mass = Mass[ipart];              // mass (GeV)
        double degeneracy = Degeneracy[ipart];  // spin degeneracy
        double sign = Sign[ipart];              // quantum statistics sign
        double baryon = Baryon[ipart];          // baryon number
        double chem = baryon * alphaB;          // chemical potential term in feq

        double equilibrium_density = Equilibrium_Density[ipart];
        double bulk_density = Bulk_Density[ipart];

        double modified_density = equilibrium_density;
        if(DF_MODE == 3)
        {
          double mbar_mod = mass / T_mod;

          modified_density = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);
        }

        dn_list[ipart] = particle_number_outflow(mass, degeneracy, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, df_coeff, shear14_coeff, equilibrium_density, bulk_density, T_mod, alphaB_mod, detA, modified_density, feqmod_breaks_down);

        dn_tot += dn_list[ipart];

      }

      // discrete probability distribution for particle types (weight[ipart] ~ dn_list[ipart] / dn_tot)
      std::discrete_distribution<int> particle_type(dn_list.begin(), dn_list.end());


      // the new formula is different from this one because the timelike cells use the same gauss-legendre integration as the spacelike one
      /*
      if(ds_time / ds_space >= 1.0) // null/timelike cell
      {
        dn_tot = udsigma * (neq_tot + bulkPi * dn_bulk_tot) - Vdsigma * dn_diff_tot;  // this might be wrong bc it could be negative?
      }
      else // spacelike cell
      {
        for(int ipart = 0; ipart < npart; ipart++)
        {
          double mass = Mass[ipart];              // mass (GeV)
          double degeneracy = Degeneracy[ipart];  // spin degeneracy
          double sign = Sign[ipart];              // quantum statistics sign
          double baryon = Baryon[ipart];          // baryon number
          double chem = baryon * alphaB;          // chemical potential term in feq

          double equilibrium_density = Equilibrium_Density[ipart];
          double bulk_density = Bulk_Density[ipart];

          double modified_density = 1.0;
          // should I pass detA of detA_bulk (if I end up using an approximation)

          dn_tot += particle_density_outflow(mass, degeneracy, sign, baryon, T, alphaB, pimunu, bulkPi, ds_space, ds_time_over_ds_space, df_coeff, shear14_coeff, equilibrium_density, bulk_density, T_mod, alphaB_mod, detA, modified_density, feqmod_breaks_down);
        }

      } // outflow condition
      */


      // loop over eta points
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta = etaValues[ieta];
        double sinheta = sinh(eta);
        double cosheta = sqrt(1.0  +  sinheta * sinheta);

        double delta_eta_weight = etaTrapezoidWeights[ieta];
        double dN_tot = delta_eta_weight * dn_tot;              // total mean number of hadrons in FO cell

        // poisson mean value out of bounds
        if(dN_tot <= 0.0)
        {
          printf("Error: the total number of hadrons / cell = %lf should be positive\n", dN_tot);
          continue;
        }


        std::poisson_distribution<int> poisson_hadrons(dN_tot); // probability distribution for number of hadrons

        // sample events for each FO cell
        for(int ievent = 0; ievent < Nevents; ievent++)
        {
          int N_hadrons = poisson_hadrons(generator_poisson);           // sample total number of hadrons in FO cell

          particle_yield_list[ievent] += N_hadrons;             // add sampled hadrons to yield list (includes p.dsigma < 0)

          for(int n = 0; n < N_hadrons; n++)
          {
            int chosen_index = particle_type(generator_type);   // chosen index of sampled particle type
            double mass = Mass[chosen_index];                   // mass of sampled particle in GeV
            double sign = Sign[chosen_index];                   // quantum statistics sign
            double baryon = Baryon[chosen_index];               // baryon number
            int mcid = MCID[chosen_index];                      // mc id

            LRF_Momentum pLRF;

            // sample local rest frame momentum
            switch(DF_MODE)
            {
              case 1: // 14 moment
              case 2: // Chapman Enskog
              {
                chapman_enskog:

                pLRF = sample_momentum(generator_momentum, &acceptances, &samples, mass, sign, baryon, T, alphaB, dsigma, pimunu, bulkPi, Vmu, df_coeff, shear14_coeff, baryon_enthalpy_ratio);
                break;
              }
              case 3: // Modified
              {
                if(feqmod_breaks_down) goto chapman_enskog;

                pLRF = sample_momentum_feqmod(generator_momentum, &acceptances, &samples, mass, sign, baryon, T_mod, alphaB_mod, dsigma, pimunu, Vmu, shear_mod_coeff, bulk_mod_coeff, diff_mod_coeff, baryon_enthalpy_ratio);

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

            double pz = tau * pLab.pn * cosheta  +  pLab.ptau * sinheta;
            new_particle.pz = pz;
            //new_particle.E = pmu.ptau * cosheta  +  tau * pmu.pn * sinheta;
            //is it possible for this to be too small or negative?
            //try enforcing mass shell condition on E
            new_particle.E = sqrt(mass * mass  +  pLab.px * pLab.px  +  pLab.py * pLab.py  +  pz * pz);

            //add to particle_event_list
            //CAREFUL push_back is not a thread-safe operation
            //how should we modify for GPU version?
            #pragma omp critical
            particle_event_list[ievent].push_back(new_particle);

          } // sampled hadrons (n)

        } // sampled events (ievent)

      } // eta points (ieta)

    } // freezeout cells (icell)

    printf("Momentum sampling efficiency = %lf\n", (double)acceptances / (double)samples);

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
