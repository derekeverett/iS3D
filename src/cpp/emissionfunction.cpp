#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
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
#include "gaussThermal.h"
#ifdef _OPENACC
#include <accelmath.h>
#endif
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

using namespace std;


// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in, Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in, int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in,  deltaf_coefficients df_in)
{
  paraRdr = paraRdr_in;
  pT_tab = pT_tab_in;
  pT_tab_length = pT_tab->getNumberOfRows();
  phi_tab = phi_tab_in;
  phi_tab_length = phi_tab->getNumberOfRows();
  y_tab = y_tab_in;
  y_tab_length = y_tab->getNumberOfRows();
  eta_tab = eta_tab_in;
  eta_tab_length = eta_tab->getNumberOfRows();

  // get control parameters
  HYDRO_MODE = paraRdr->getVal("hydro_mode");
  DF_MODE = paraRdr->getVal("df_mode");
  DIMENSION = paraRdr->getVal("dimension");
  REINFORCE = paraRdr->getVal("reinforce");
  INCLUDE_BARYON = paraRdr->getVal("include_baryon");
  INCLUDE_BULK_DELTAF = paraRdr->getVal("include_bulk_deltaf");
  INCLUDE_SHEAR_DELTAF = paraRdr->getVal("include_shear_deltaf");
  INCLUDE_BARYONDIFF_DELTAF = paraRdr->getVal("include_baryondiff_deltaf");
  REGULATE_DELTAF = paraRdr->getVal("regulate_deltaf");
  GROUP_PARTICLES = paraRdr->getVal("group_particles");
  PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");

  particles = particles_in;
  Nparticles = Nparticles_in;
  surf_ptr = surf_ptr_in;
  FO_length = FO_length_in;
  df = df_in;
  number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

  chosen_particles_01_table = new int[Nparticles];
  //a class member to hold 3D spectra for all chosen particles
  dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
  //zero the array
  for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
  {
    dN_pTdpTdphidy[iSpectra] = 0.0;
  }

  for (int n = 0; n < Nparticles; n++) chosen_particles_01_table[n] = 0;

  //only grab chosen particles from the table
  for (int m = 0; m < number_of_chosen_particles; m++)
  { //loop over all chosen particles
    int mc_id = chosen_particles_in->get(1, m + 1);

    for (int n = 0; n < Nparticles; n++)
    {
      if (particles[n].mc_id == mc_id)
      {
        chosen_particles_01_table[n] = 1;
        break;
      }
    }
  }

  // next, for sampling processes
  chosen_particles_sampling_table = new int[number_of_chosen_particles];
  // first copy the chosen_particles table, but now using indices instead of mc_id
  int current_idx = 0;
  for (int m = 0; m < number_of_chosen_particles; m++)
  {
    int mc_id = chosen_particles_in->get(1, m + 1);
    for (int n = 0; n < Nparticles; n++)
    {
      if (particles[n].mc_id == mc_id)
      {
        chosen_particles_sampling_table[current_idx] = n;
        current_idx ++;
        break;
      }
    }
  }

  // next re-order them so that particles with similar mass are adjacent
  if (GROUP_PARTICLES == 1) // sort particles according to their mass; bubble-sorting
  {
    for (int m = 0; m < number_of_chosen_particles; m++)
    {
      for (int n = 0; n < number_of_chosen_particles - m - 1; n++)
      {
        if (particles[chosen_particles_sampling_table[n]].mass > particles[chosen_particles_sampling_table[n + 1]].mass)
        {
          // swap them
          int particle_idx = chosen_particles_sampling_table[n + 1];
          chosen_particles_sampling_table[n + 1] = chosen_particles_sampling_table[n];
          chosen_particles_sampling_table[n] = particle_idx;
        }
      }
    }
  }
}


EmissionFunctionArray::~EmissionFunctionArray()
{
  delete[] chosen_particles_01_table;
  delete[] chosen_particles_sampling_table;
  delete[] dN_pTdpTdphidy; //for holding 3d spectra of all chosen particles
}

//this is where the magic happens - and the speed bottleneck
//this is the function that needs to be parallelized
//try acceleration via omp threads and simd, as well as openacc, CUDA?
void EmissionFunctionArray::calculate_dN_ptdptdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *EquilibriumDensity, double *LinearDensityCorrection,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff)

  {
    double prefactor = pow(2.0 * M_PI * hbarC, -3);
    int FO_chunk = 10000;


    // Set momentum, spacetime rapidity (2+1d) points from tables

    // cos(phi), sin(phi) arrays
    double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
    for (int j = 0; j < phi_tab_length; j++)
    {
      double phi = phi_tab->get(1,j+1);
      trig_phi_table[j][0] = cos(phi);
      trig_phi_table[j][1] = sin(phi);
    }

    // pT array
    double pTValues[pT_tab_length];
    for (int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      double pT = pT_tab->get(1,ipT+1);
      pTValues[ipT] = pT;
    }

    // y and eta arrays (default 3+1d)
    int y_pts = y_tab_length;
    int eta_pts = 1;
    if(DIMENSION == 2)  // pts for 2+1d
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts]; // eta_weight * delta_eta

    // eta interval (assume uniform)
    double eta1 = eta_tab->get(1,1);
    double eta2 = eta_tab->get(1,2);
    double delta_eta = eta2 - eta1;
    //cout << delta_eta << endl;

    if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // will load eta_fo later
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_pts; iy++)
      {
        double y = y_tab->get(1,iy+1);
        yValues[iy] = y;
      }
    }
    else if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        double eta = eta_tab->get(1,ieta+1);
        double eta_weight = eta_tab->get(2,ieta+1);

        etaValues[ieta] = eta;
        etaDeltaWeights[ieta] = eta_weight * delta_eta;
        //cout << etaValues[ieta] << "\t" << etaDeltaWeights[ieta] << endl;
      }
    }

    // Momentum rescaling matrix (for modified equilibrium mode)
    // perhaps I can find the inverse matrix with a gsl function
    // that way I don't have to solve the matrix equation every time
    double ** M;
    if(DF_MODE == 3)
    {
      M = (double**)calloc(3, sizeof(double*));
      for(int i = 0; i < 3; i++) M[i] = (double*)calloc(3, sizeof(double));
    }
    int permutation[3]; // permutation vector for LUP decomposition

    // for modified particle density (temp)
    const int pbar_pts = 32;

    double pbar_root1[pbar_pts] = {0.111258307817857,0.373218875635789,0.785645933548703,1.34945557597191,2.06595254243689,2.9368251505102,3.96416490487103,5.15049470795513,6.49880481850242,8.01259753651012,9.69594226244322,11.5535431802497,13.5908225576041,15.8140236536506,18.2303386010357,20.848068568997,23.6768263015307,26.727795211377,30.0140653370291,33.5510758837993,37.3572089427003,41.4546032327439,45.8702977224496,50.6378873345034,55.8000071146054,61.4122254946366,67.5494874881775,74.3175519062769,81.8752902019347,90.484348981827,100.645181531231,113.644990137437};

    double pbar_weight1[pbar_pts] = {0.0186059727905476,0.0866173173954464,0.174736939714111,0.223981707776893,0.207665649180844,0.147764609849031,0.0832606473486164,0.0378197718165418,0.0139927393633244,0.004241112030074,0.00105579297396034,0.000215935110576046,3.62266132038311e-05,4.96936839587084e-06,5.54740690315405e-07,5.00813828447151e-08,3.62776022875082e-09,2.08821684084769e-10,9.44037161938962e-12,3.30457415819565e-13,8.80420243357802e-15,1.74829776065748e-16,2.5216841514199e-18,2.55814264797865e-20,1.75187894103143e-22,7.67671322813603e-25,2.00250351976227e-27,2.80862372202544e-30,1.81852286568561e-33,4.23375360281789e-37,2.20632410063702e-41,7.76055673224384e-47};

    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    //in this way, we are limited to small values of FO_chunk, because our array holds values for all species...
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));


    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      printf("Progress : Finished chunk %d of %ld \n", n, FO_length / FO_chunk);
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); //don't go out of array bounds
      #pragma omp parallel for
      #pragma acc kernels
      //#pragma acc loop independent
      for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell;     // global FO cell index

        // set freezeout info to local varibles to reduce(?) memory access outside cache :
        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double tau2 = tau * tau;
        double tau4 = tau2 * tau2;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from freezeout cell
        }
        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ux = ux_fo[icell_glb];           // contravariant fluid velocity
        double uy = uy_fo[icell_glb];
        double un = un_fo[icell_glb];
        double ut;

        double T = T_fo[icell_glb];             // temperature
        double P = P_fo[icell_glb];             // pressure
        double E = E_fo[icell_glb];             // energy density

        double pixx = pixx_fo[icell_glb];       // pi^munu
        double pixy = pixy_fo[icell_glb];
        double pixn = pixn_fo[icell_glb];
        double piyy = piyy_fo[icell_glb];
        double piyn = piyn_fo[icell_glb];
        double pitt, pitx, pity, pitn, pinn;

        double u0 = sqrt(1.0  +  ux * ux  +  uy * uy);

        if(REINFORCE)
        {
          // u.u = 1, pi.u = 0, Tr(pi) = 0
          ut = sqrt(u0 * u0 +  tau2 * un * un);
          double ut2 = ut * ut;
          pinn = (pixx*(ux*ux - ut2) + piyy*(uy*uy - ut2) + 2.0*(pixy*ux*uy + tau2*un*(pixn*ux + piyn*uy))) / (tau2 * u0 * u0);
          pitn = (pixn*ux + piyn*uy + tau2*pinn*un) / ut;
          pity = (pixy*ux + piyy*uy + tau2*piyn*un) / ut;
          pitx = (pixx*ux + pixy*uy + tau2*pixn*un) / ut;
          pitt = (pitx*ux + pity*uy + tau2*pitn*un) / ut;
        }
        else
        {
          ut = ut_fo[icell_glb];
          pitt = pitt_fo[icell_glb];
          pitx = pitx_fo[icell_glb];
          pity = pity_fo[icell_glb];
          pitn = pitn_fo[icell_glb];
          pinn = pinn_fo[icell_glb];
        }

        double bulkPi = bulkPi_fo[icell_glb];   // bulk pressure

        double muB = 0.0;                       // baryon chemical potential
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density
        double Vt = 0.0;                        // baryon diffusion
        double Vx = 0.0;
        double Vy = 0.0;
        double Vn = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell_glb];
          alphaB = muB / T;

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
            if(REINFORCE)
            {
              // V.u = 0
              Vt = (Vx * ux  + Vy * uy  +  tau2 * Vn * un) / ut;
            }
            else Vt = Vt_fo[icell_glb];
          }
        }

        // Milne basis vectors:
        double Xt, Xx, Xy, Xn;  // X^mu
        double Yx, Yy;          // Y^mu
        double Zt, Zn;          // Z^mu
        if(DF_MODE == 3)
        {
          // u0 = sqrt(1.0 + ux * ux + uy * uy_)
          double sinhL = tau * un / u0;
          double coshL = ut / u0;
          double uperp = sqrt(ux * ux  +  uy * uy);

          Xt = uperp * coshL;
          Xn = uperp * sinhL / tau;
          Zt = sinhL;
          Zn = coshL / tau;

          // stops (ux=0)/(uperp=0) nans
          if(uperp < 1.e-5)
          {
            Xx = 1.0; Xy = 0.0;
            Yx = 0.0; Yy = 1.0;
          }
          else
          {
            Xx = u0 * ux / uperp;
            Xy = u0 * uy / uperp;
            Yx = - uy / uperp;
            Yy = ux / uperp;
          }
        } // set basis vectors

        // Set df coefficients: (efficient for openmp, etc?)
        // 14 moment
        double c0 = 0.0;
        double c1 = 0.0;
        double c2 = 0.0;
        double c3 = 0.0;
        double c4 = 0.0;
        // Chapman enskog or modified equilibrium
        double F = 0.0;
        double G = 0.0;
        double betabulk, betaV, betapi;
        double dT = 0.0;
        double dalphaB = 0.0;
        double T_mod = T;
        double alphaB_mod = alphaB;
        double baryon_enthalpy_ratio = 0.0;
        // shear/bulk coefficients
        double shear_coeff = 0.0;
        double bulk0_coeff = 0.0;
        double bulk1_coeff = 0.0;
        double bulk2_coeff = 0.0;

        switch(DF_MODE)
        {
          case 1: // 14 moment
          {
            if(INCLUDE_SHEAR_DELTAF)
            {
              shear_coeff = 0.5 / (T * T * (E + P));
            }
            if(INCLUDE_BULK_DELTAF)
            {
              c0 = df_coeff[0];
              c2 = df_coeff[2];
              bulk0_coeff = c0 - c2;
              bulk2_coeff = 4.0*c2 - c0;
              if(INCLUDE_BARYON)
              {
                c1 = df_coeff[1];
                bulk1_coeff = c1;
              }
            }
            if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
            {
              c3 = df_coeff[3];
              c4 = df_coeff[4];
            }
            break;
          } // case 1
          case 2: // Chapman enskog
          case 3: // Modified equilibrium
          {
            if(INCLUDE_SHEAR_DELTAF)
            {
              betapi = df_coeff[4];
              shear_coeff = 0.5 / (betapi * T);
            }
            if(INCLUDE_BULK_DELTAF)
            {
              F = df_coeff[0];
              betabulk = df_coeff[2];
              bulk0_coeff = F / (T * T * betabulk);
              bulk2_coeff = 1.0 / (3.0 * T * betabulk);
              // modified temperature
              dT = F * bulkPi / betabulk;
              T_mod = T + dT;

              if(INCLUDE_BARYON)
              {
                G = df_coeff[1];
                bulk1_coeff = G / betabulk;
                // modified muB/T
                dalphaB = G * bulkPi / betabulk;
                alphaB_mod = alphaB + dalphaB;
              }
            }
            if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
            {
              betaV = df_coeff[3];
              baryon_enthalpy_ratio = nB / (E + P);
            }
            break;
          } // case 2 or 3
          default:
          {
            cout << "Please specify df_mode = (1,2,3) in parameters.dat..." << endl;
            exit(-1);
          }
        } // df coefficients

        // set momentum rescaling matrix: (w/ shear and bulk corrections)
        double detA = 1.0;  // determinant of Aij = (ideal + shear + bulk) part of Mij
        double nlinear_fact = 0.0;
        double nmod_fact = 0.0;
        // pimunu in the LRF: piij = Xi.pi.Xj
        double pixx_LRF, pixy_LRF, pixn_LRF, piyy_LRF, piyn_LRF, pinn_LRF;

        if(DF_MODE == 3)
        {
          // default ideal: Mij = deltaij
          // - Aij is symmetric, Mij is not symmetric if include baryon diffusion
          double Mxx = 1.0,  Mxy = 0.0,  Mxn = 0.0;
          double Myx = 0.0,  Myy = 1.0,  Myn = 0.0;
          double Mnx = 0.0,  Mny = 0.0,  Mnn = 1.0;

          if(INCLUDE_SHEAR_DELTAF)
          {
            // evaluate LRF piij
            pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau4*pinn*Xn*Xn
                    + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));
            pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xy + pixy*Xx + piyy*Xy + tau2*piyn*Xn);
            pixn_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xn - piyn*Xy - tau2*pinn*Xn);
            piyy_LRF = pixx*Yx*Yx + piyy*Yy*Yy + 2.0*pixy*Yx*Yy;
            piyn_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);
            pinn_LRF = - (pixx_LRF + piyy_LRF);

            // add shear matrix elements ~ piij
            double shear_fact = 0.5 / betapi;
            Mxx += (pixx_LRF * shear_fact);
            Mxy += (pixy_LRF * shear_fact);
            Mxn += (pixn_LRF * shear_fact);
            Myx = Mxy;
            Myy += (piyy_LRF * shear_fact);
            Myn += (piyn_LRF * shear_fact);
            Mnx = Mxn;
            Mny = Myn;
            Mnn += (pinn_LRF * shear_fact);
          } // shear elements
          if(INCLUDE_BULK_DELTAF)
          {
            // add bulk matrix elements ~ deltaij
            double bulk_term = bulkPi / betabulk / 3.0;
            Mxx += bulk_term;
            Myy += bulk_term;
            Mnn += bulk_term;
          } // bulk elements

          // baryon diffusion correction is more complicated b/c it depends on pprime

          // evaluate detA (only depends on ideal + shear + bulk)
          detA = Mxx * (Myy * Mnn - Myn * Myn)  -  Mxy * (Mxy * Mnn - Myn * Mxn)  +  Mxn * (Mxy * Myn - Myy * Mxn);
          //cout << setw(10) << setprecision(8) << piabs << "\t\t" << 2.0 * betapi << "\t\t" << detA << endl;
          //cout << detA << endl;

          // set M-matrix
          M[0][0] = Mxx;  M[0][1] = Mxy;  M[0][2] = Mxn;
          M[1][0] = Myx;  M[1][1] = Myy;  M[1][2] = Myn;
          M[2][0] = Mnx;  M[2][1] = Mny;  M[2][2] = Mnn;
          // LUP decompose M once
          LUP_decomposition(M, 3, permutation);

          // prefactors for modified density and linear density correction
          nmod_fact = detA * T_mod * T_mod * T_mod / (2.0 * M_PI * M_PI * hbarC * hbarC * hbarC);

          if(INCLUDE_BULK_DELTAF) nlinear_fact = bulkPi / betabulk;

        } // set momentum matrix





        //now loop over all particle species and momenta
        for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          // set particle properties
          double mass = Mass[ipart];    // (GeV)
          double mass2 = mass * mass;
          double sign = Sign[ipart];
          double degeneracy = Degeneracy[ipart];
          double baryon = 0.0;
          double chem = 0.0;            // chemical potential term in feq or feqmod
          double chem_mod = 0.0;
          if(INCLUDE_BARYON)
          {
            baryon = Baryon[ipart];
            chem = baryon * alphaB;   // default
            if(DF_MODE == 3 && INCLUDE_BULK_DELTAF)
            {
              chem_mod = baryon * alphaB_mod;
            }
          }
          // modified renormalization factor
          double renorm = 1.0;
          if(DF_MODE == 3)
          {
            if(INCLUDE_BULK_DELTAF)
            {
              double neq = EquilibriumDensity[ipart];
              double nlinear_correction = nlinear_fact * LinearDensityCorrection[ipart];
              double n_linear = neq + nlinear_correction;

              double mbar_mod = mass / T_mod;
              double n_mod = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);

              renorm = n_linear / n_mod;
            }
            else renorm = 1.0 / detA;
          }



          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            // transverse radial momentum and transverse mass
            double pT = pTValues[ipT];
            double mT = sqrt(mass2 + pT * pT);
            double mT_over_tau = mT / tau;

            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double cosphi = trig_phi_table[iphip][0];
              double sinphi = trig_phi_table[iphip][1];
              double px = pT * cosphi;  // p^x
              double py = pT * sinphi;  // p^y

              for (int iy = 0; iy < y_pts; iy++)
              {
                double y = yValues[iy];             // y
                double pdotdsigma_f_eta_sum = 0.0;  // reset to zero

                // integrate over eta (flexible for 2+1 or 3+1 hydro)
                for(int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];                     // eta point
                  double delta_eta_weight = etaDeltaWeights[ieta];  // delta_eta * eta_weight (trapezoid rule)

                  // all vector components are CONTRAVARIANT EXCEPT the surface
                  // normal vector dat, dax, day, dan, which are COVARIANT
                  double pt = mT * cosh(y - eta);           // p^tau
                  double pn = mT_over_tau * sinh(y - eta);  // p^eta
                  double tau2_pn = tau2 * pn;
                  // momentum vector is contravariant, surface normal vector is COVARIANT
                  double pdotdsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;
                  // distribution: feq(1+df) or feqmod
                  double f;

                  // calculate f
                  switch(DF_MODE)
                  {
                    case 1: // 14 moment
                    {
                      // u.p LRF energy
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      // thermal distribution
                      double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                      // shear correction
                      double df_shear = 0.0;
                      if(INCLUDE_SHEAR_DELTAF)
                      {
                        // pi^munu.p_mu.p_nu
                        double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                        + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));
                        // df_shear / feqbar
                        df_shear = shear_coeff * pimunu_pmu_pnu;
                      }
                      // bulk correction
                      double df_bulk = 0.0;
                      if(INCLUDE_BULK_DELTAF)
                      {
                        if(!INCLUDE_BARYON) df_bulk = (bulk0_coeff * mass2  +  bulk2_coeff * pdotu * pdotu) * bulkPi;
                        else df_bulk = (bulk0_coeff * mass2  +  (baryon * bulk1_coeff + bulk2_coeff * pdotu) * pdotu) * bulkPi;
                      }
                      // baryon diffusion correction
                      double df_baryondiff = 0.0;
                      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                      {
                        // V^mu * p_mu
                        double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;
                        df_baryondiff = (baryon * c3  +  c4 * pdotu) * Vmu_pmu;
                      }
                      // viscous corrections
                      double feqbar = 1.0  -  sign * feq;
                      double df = feqbar * (df_shear + df_bulk + df_baryondiff);
                      // regulate df
                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

                      f = feq * (1.0 + df);
                      break;
                    } // case 1
                    case 2: // Chapman enskog
                    {
                      // goto label (see case 3)
                      chapman_enskog:

                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                      double df_shear = 0.0;
                      if(INCLUDE_SHEAR_DELTAF)
                      {
                        double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                        + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));
                        df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      }
                      double df_bulk = 0.0;
                      if(INCLUDE_BULK_DELTAF)
                      {
                        if(!INCLUDE_BARYON) df_bulk = (bulk0_coeff * pdotu  +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                        else df_bulk = (bulk0_coeff * pdotu  +  baryon * bulk1_coeff  +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                      }
                      double df_baryondiff = 0.0;
                      if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                      {
                        double Vmu_pmu = Vt * pt   -   Vx * px   -   Vy * py   -   Vn * tau2_pn;
                        df_baryondiff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;
                      }
                      double feqbar = 1.0  -  sign * feq;
                      double df = feqbar * (df_shear + df_bulk + df_baryondiff);
                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

                      f = feq * (1.0 + df);
                      break;
                    } // case 2
                    case 3: // modified distribution
                    {
                      // if feqmod breaks down, go back to case 2 statement (chapman enskog)
                      if(detA < 0.01) goto chapman_enskog;
                      // local momentum
                      double pX = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;  // -x.p
                      double pY = Yx * px  +  Yy * py;                                // -y.p
                      double pZ = -Zt * pt  +  Zn * tau2_pn;                          // -z.p
                      double p[3] = {pX, pY, pZ};
                      // solve the matrix equation: pi = Mij.pmodj
                      LUP_solve(M, 3, permutation, p);
                      // pmod stored in p
                      double pXmod2 = p[0] * p[0];
                      double pYmod2 = p[1] * p[1];
                      double pZmod2 = p[2] * p[2];
                      // effective LRF energy
                      double E_mod = sqrt(mass2 + pXmod2 + pYmod2 + pZmod2);

                      // modified equilibrium distribution
                      f = renorm / (exp(E_mod / T_mod  -  chem_mod) + sign);
                      break;
                    } // case 3
                    default:
                    {
                      cout << "Please set df_mode = (1,2,3) in parameters.dat..." << endl;
                      exit(-1);
                    } // default error
                  } // df_mode

                  pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * f);

                } // ieta

                long long int iSpectra = icell + endFO * (ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy)));

                dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);

              } //iy
            } //iphip
          } //ipT
        } //ipart
      } //icell
      if(endFO != 0)
      {
        //now perform the reduction over cells
        #pragma omp parallel for collapse(3)
        #pragma acc kernels
        for (int ipart = 0; ipart < npart; ipart++)
        {
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              for (int iy = 0; iy < y_pts; iy++)
              {
                //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
                long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
                double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  long long int iSpectra = icell + endFO * iS3D;
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                }//icell
                dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; //sum over all chunks
              }//iy
            }//iphip
          }//ipT
        }//ipart species
      } //if (endFO != 0 )
    }//n FO chunk

    //free memory
    if(DF_MODE == 3) free_2D(M,3);
    free(dN_pTdpTdphidy_all);
  }

// haven't defined in the header yet...
void EmissionFunctionArray::calculate_dN_ptdptdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
  double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo, double *Wt_fo, double *Wx_fo,double *Wy_fo, double *Wn_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
  {
    double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
    int FO_chunk = 10000;

    double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
    for (int j = 0; j < phi_tab_length; j++)
    {
      double phi = phi_tab->get(1,j+1);
      trig_phi_table[j][0] = cos(phi);
      trig_phi_table[j][1] = sin(phi);
    }

    //fill arrays with values points in pT and y
    double pTValues[pT_tab_length];
    for (int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

    // fill arrays with values points in y and eta
    int y_pts = y_tab_length;    // defaults pts for 3+1d
    int eta_pts = 1;

    if(DIMENSION == 2)
    {
      y_pts = 1;
      eta_pts = eta_tab_length;
    }

    double yValues[y_pts];
    double etaValues[eta_pts];
    double etaDeltaWeights[eta_pts];    // eta_weight * delta_eta

    double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid
    //cout << delta_eta << endl;

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
      for(int ieta = 0; ieta < eta_pts; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaDeltaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
        //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaDeltaWeights[0] = 1.0; // 1.0 for 3+1d
      for (int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);
    }


    //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
    //to hold the spectra for each surface cell in a chunk, for all particle species
    //in this way, we are limited to small values of FO_chunk, because our array holds values for all species...
    int npart = number_of_chosen_particles;
    double *dN_pTdpTdphidy_all;
    dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

    //loop over bite size chunks of FO surface
    for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
    {
      printf("Progress : Finished chunk %d of %ld \n", n, FO_length / FO_chunk);
      int endFO = FO_chunk;
      if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); // don't go out of array bounds
      #pragma omp parallel for
      #pragma acc kernels
      //#pragma acc loop independent
      for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
      {
        int icell_glb = n * FO_chunk + icell; // global FO cell index

        // set freezeout info to local varibles
        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double tau2 = tau * tau;                // useful expression
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
        }
        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ux = ux_fo[icell_glb];           // contravariant fluid velocity
        double uy = uy_fo[icell_glb];           // enforce normalization: u.u = 1
        double un = un_fo[icell_glb];
        double ut;

        double u0 = sqrt(1.0  +  ux * ux  +  uy * uy);

        double pixx = pixx_fo[icell_glb];       // piperp^munu
        double pixy = pixy_fo[icell_glb];
        double pitt, pitx, pity, pitn, pixn, piyy, piyn, pinn;

        double Wx = Wx_fo[icell_glb];           // W^mu
        double Wy = Wy_fo[icell_glb];
        double Wt, Wn;

        if(REINFORCE)
        {
          // reinforce normalization
          ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un);

          // reinforce piperp^munu properties (solve eqs. by row)
          // x
          pitx = (pixx * ux  +  pixy * uy) * ut / (u0 * u0);
          pixn = un * pitx / ut;
          // y
          piyy = (-pixx * (1.0 + uy * uy)  +  2.0 * pixy * ux * uy) / (1.0 + ux * ux);
          pity = (pixy * ux  +  piyy * uy) * ut / (u0 * u0);
          piyn = un * pity / ut;
          // n
          pitn = (pixn * ux  +  piyn * uy) * ut / (u0 * u0);
          pinn = un * pitn / ut;
          // t
          pitt = ut * pitn / un;

          // reinforce W^mu properties
          Wt = (ux * Wx  +  uy * Wy) * ut / (u0 * u0);
          Wn = Wt * un / ut;
        }
        else
        {
          ut = ut_fo[icell_glb];

          pitt = pitt_fo[icell_glb];
          pitx = pitx_fo[icell_glb];
          pity = pity_fo[icell_glb];
          pitn = pitn_fo[icell_glb];
          pixn = pixn_fo[icell_glb];
          piyy = piyy_fo[icell_glb];
          piyn = piyn_fo[icell_glb];
          pinn = pinn_fo[icell_glb];

          Wt = Wt_fo[icell_glb];
          Wn = Wn_fo[icell_glb];
        }

        double zt = tau * un / u0;
        double zn = ut / (u0 * tau);            // z basis vector

        double bulkPi = bulkPi_fo[icell_glb];   // residual bulk pressure

        double Lambda = Lambda_fo[icell_glb];   // anisotropic variables
        double aL = aL_fo[icell_glb];
        double xiL = 1.0 / (aL * aL)  -  1.0;   // xi

        double c0 = c0_fo[icell_glb];           // 14-moment coefficients
        double c1 = c1_fo[icell_glb];
        double c2 = c2_fo[icell_glb];
        double c3 = c3_fo[icell_glb];
        double c4 = c4_fo[icell_glb];


        //now loop over all particle species and momenta
        for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          // set particle properties
          double mass = Mass[ipart];    // (GeV)
          double mass2 = mass * mass;
          double sign = Sign[ipart];
          double degeneracy = Degeneracy[ipart];


          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            // set transverse radial momentum and transverse mass (GeV)
            double pT = pTValues[ipT];
            double mT = sqrt(mass2 + pT * pT);
            double mT_over_tau = mT / tau;

            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double px = pT * trig_phi_table[iphip][0]; // contravariant
              double py = pT * trig_phi_table[iphip][1]; // contravariant

              for (int iy = 0; iy < y_pts; iy++)
              {
                // all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                double y = yValues[iy];

                double pdotdsigma_f_eta_sum = 0.0;

                // integrate over eta
                for (int ieta = 0; ieta < eta_pts; ieta++)
                {
                  double eta = etaValues[ieta];
                  double delta_eta_weight = etaDeltaWeights[ieta];

                  double pt = mT * cosh(y - eta);
                  double pn = mT_over_tau * sinh(y - eta);
                  double tau2_pn = tau2 * pn;

                  // momentum vector is contravariant, surface normal vector is COVARIANT
                  double pdotdsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;

                  // anisotropic distribution - for vahydro
                  double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;    // u.p = LRF energy
                  double pdotz = pt * zt  -  tau2_pn * zn;                            // z.p = - LRF longitudinal momentum

                  double Ea = sqrt(pdotu * pdotu  +  xiL * pdotz * pdotz);

                  double fa = 1.0 / (exp(Ea / Lambda) + sign);

                  // residual viscous corrections
                  double fabar = 1.0 - sign*fa;

                  // residual shear correction:
                  double df_shear = 0.0;

                  if (INCLUDE_SHEAR_DELTAF)
                  {
                    // - (-z.p)p_{mu}.W^mu
                    double Wmu_pmu_pz = pdotz * (Wt * pt  -  Wx * px  -  Wy * py  -  Wn * tau2_pn);

                    // p_{mu.p_nu}.piperp^munu
                    double pimunu_pmu_pnu = pitt * pt * pt + pixx * px * px + piyy * py * py + pinn * tau2_pn * tau2_pn
                  + 2.0 * (-(pitx * px + pity * py) * pt + pixy * px * py + tau2_pn * (pixn * px + piyn * py - pitn * pt));

                    df_shear = c3 * Wmu_pmu_pz  +  c4 * pimunu_pmu_pnu;  // df / (feq*feqbar)
                  }

                  // residual bulk correction:
                  double df_bulk = 0.0;
                  if (INCLUDE_BULK_DELTAF) df_bulk = (c0 * mass2  +  c1 * pdotz * pdotz  +  c2 * pdotu * pdotu) * bulkPi;

                  double df = fabar * (df_shear + df_bulk);

                  if (REGULATE_DELTAF)
                  {
                    double reg_df = max(-1.0, min(df, 1.0));
                    pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * fa * (1.0 + reg_df));
                  }
                  else pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * fa * (1.0 + df));

                } // ieta

                long long int iSpectra = icell + endFO * (ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy)));

                dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);
              } //iy
            } //iphip
          } //ipT
        } //ipart
      } //icell
      if(endFO != 0)
      {
        //now perform the reduction over cells
        #pragma omp parallel for collapse(3)
        #pragma acc kernels
        for (int ipart = 0; ipart < npart; ipart++)
        {
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            for (int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              for (int iy = 0; iy < y_pts; iy++)
              {
                //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
                long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
                double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                for (int icell = 0; icell < endFO; icell++)
                {
                  //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                  long long int iSpectra = icell + endFO * iS3D;
                  dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                }//icell
                dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; //sum over all chunks
              }//iy
            }//iphip
          }//ipT
        }//ipart species
      } //if (endFO != 0 )
    }//n FO chunk

    //free memory
    free(dN_pTdpTdphidy_all);
  }

  void EmissionFunctionArray::write_dN_pTdpTdphidy_toFile()
  {
    printf("writing to file\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT

    int npart = number_of_chosen_particles;
    char filename[255] = "";
    sprintf(filename, "results/dN_pTdpTdphidy_block.dat");
    ofstream spectraFileBlock(filename, ios_base::app);

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
            long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
            if (dN_pTdpTdphidy[iS3D] < 1.0e-40) spectraFileBlock << scientific <<  setw(15) << setprecision(8) << 0.0 << "\t";
            else spectraFileBlock << scientific <<  setw(15) << setprecision(8) << dN_pTdpTdphidy[iS3D] << "\t";
          } //ipT
          spectraFileBlock << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFileBlock.close();

    sprintf(filename, "results/dN_pTdpTdphidy.dat");
    ofstream spectraFile(filename, ios_base::app);
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if(DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);

          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
            long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
            //if (dN_pTdpTdphidy[is] < 1.0e-40) spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << 0.0 << "\n";
            //else spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[is] << "\n";
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t{" << pT << ",\t" << dN_pTdpTdphidy[iS3D] << "},\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }




  //*********************************************************************************************
  void EmissionFunctionArray::calculate_spectra()
  // Calculate dNArrays and flows for all particles given in chosen_particle file.
  {
    cout << "calculate_spectra() has started... " << endl;
    Stopwatch sw;
    sw.tic();

    //fill arrays with all particle info and freezeout info to pass to function which will perform the integral
    //this should only be done once, not for each particle... then launch calculation on accelerator w/o unecessary copies

    //particle info
    particle_info *particle;
    double *Mass, *Sign, *Degen, *Baryon;
    Mass = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Sign = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Degen = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));

     // added modified density info (equilibrium density and linear density correction in modified vhydro)
    double *EquilibriumDensity, *LinearDensityCorrection;
    if(DF_MODE == 3)
    {
      EquilibriumDensity = (double*)calloc(number_of_chosen_particles, sizeof(double));
      LinearDensityCorrection = (double*)calloc(number_of_chosen_particles, sizeof(double));
    }

    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      int particle_idx = chosen_particles_sampling_table[ipart];
      particle = &particles[particle_idx];
      Mass[ipart] = particle->mass;
      Sign[ipart] = particle->sign;
      Degen[ipart] = particle->gspin;
      Baryon[ipart] = particle->baryon;
      if(DF_MODE == 3)
      {
        EquilibriumDensity[ipart] = particle->equilibrium_density;
        LinearDensityCorrection[ipart] = particle->linearized_density_correction;
      }
    }

    // freezeout info
    FO_surf *surf = &surf_ptr[0];

    // freezeout surface info exclusive for VH
    double *E, *T, *P;
    if(HYDRO_MODE == 1)
    {
      E = (double*)calloc(FO_length, sizeof(double));
      P = (double*)calloc(FO_length, sizeof(double));
    }

    T = (double*)calloc(FO_length, sizeof(double));

    // freezeout surface info common for VH / VAH
    double *tau, *eta, *ut, *ux, *uy, *un;
    double *dat, *dax, *day, *dan;
    double *pitt, *pitx, *pity, *pitn, *pixx, *pixy, *pixn, *piyy, *piyn, *pinn, *bulkPi;
    double *muB, *nB, *Vt, *Vx, *Vy, *Vn;

    tau = (double*)calloc(FO_length, sizeof(double));
    eta = (double*)calloc(FO_length, sizeof(double));

    ut = (double*)calloc(FO_length, sizeof(double));
    ux = (double*)calloc(FO_length, sizeof(double));
    uy = (double*)calloc(FO_length, sizeof(double));
    un = (double*)calloc(FO_length, sizeof(double));

    dat = (double*)calloc(FO_length, sizeof(double));
    dax = (double*)calloc(FO_length, sizeof(double));
    day = (double*)calloc(FO_length, sizeof(double));
    dan = (double*)calloc(FO_length, sizeof(double));

    pitt = (double*)calloc(FO_length, sizeof(double));
    pitx = (double*)calloc(FO_length, sizeof(double));
    pity = (double*)calloc(FO_length, sizeof(double));
    pitn = (double*)calloc(FO_length, sizeof(double));
    pixx = (double*)calloc(FO_length, sizeof(double));
    pixy = (double*)calloc(FO_length, sizeof(double));
    pixn = (double*)calloc(FO_length, sizeof(double));
    piyy = (double*)calloc(FO_length, sizeof(double));
    piyn = (double*)calloc(FO_length, sizeof(double));
    pinn = (double*)calloc(FO_length, sizeof(double));

    bulkPi = (double*)calloc(FO_length, sizeof(double));

    if(INCLUDE_BARYON)
    {
      muB = (double*)calloc(FO_length, sizeof(double));

      if(INCLUDE_BARYONDIFF_DELTAF)
      {
        nB = (double*)calloc(FO_length, sizeof(double));
        Vt = (double*)calloc(FO_length, sizeof(double));
        Vx = (double*)calloc(FO_length, sizeof(double));
        Vy = (double*)calloc(FO_length, sizeof(double));
        Vn = (double*)calloc(FO_length, sizeof(double));
      }
    }


    // freezeout surface info exclusive for VAH_PL
    double *Wt, *Wx, *Wy, *Wn;
    double *Lambda, *aL;
    double *c0, *c1, *c2, *c3, *c4; //delta-f coeffs for vah

    if(HYDRO_MODE == 2)
    {
      Wt = (double*)calloc(FO_length, sizeof(double));
      Wx = (double*)calloc(FO_length, sizeof(double));
      Wy = (double*)calloc(FO_length, sizeof(double));
      Wn = (double*)calloc(FO_length, sizeof(double));

      Lambda = (double*)calloc(FO_length, sizeof(double));
      aL = (double*)calloc(FO_length, sizeof(double));

      // 14-moment coefficients (VAH_PL)
      if (DF_MODE == 4)
      {
        c0 = (double*)calloc(FO_length, sizeof(double));
        c1 = (double*)calloc(FO_length, sizeof(double));
        c2 = (double*)calloc(FO_length, sizeof(double));
        c3 = (double*)calloc(FO_length, sizeof(double));
        c4 = (double*)calloc(FO_length, sizeof(double));
      }
    }



    for(long icell = 0; icell < FO_length; icell++)
    {
      // reading info from surface
      surf = &surf_ptr[icell];

      if(HYDRO_MODE == 1)
      {
        P[icell] = surf->P;
        E[icell] = surf->E;
      }

      T[icell] = surf->T;

      tau[icell] = surf->tau;
      eta[icell] = surf->eta;

      ut[icell] = surf->ut;
      ux[icell] = surf->ux;
      uy[icell] = surf->uy;
      un[icell] = surf->un;

      dat[icell] = surf->dat;
      dax[icell] = surf->dax;
      day[icell] = surf->day;
      dan[icell] = surf->dan;

      pitt[icell] = surf->pitt;
      pitx[icell] = surf->pitx;
      pity[icell] = surf->pity;
      pitn[icell] = surf->pitn;
      pixx[icell] = surf->pixx;
      pixy[icell] = surf->pixy;
      pixn[icell] = surf->pixn;
      piyy[icell] = surf->piyy;
      piyn[icell] = surf->piyn;
      pinn[icell] = surf->pinn;

      bulkPi[icell] = surf->bulkPi;

      if(INCLUDE_BARYON)
      {
        muB[icell] = surf->muB;

        if(INCLUDE_BARYONDIFF_DELTAF)
        {
          nB[icell] = surf->nB;
          Vt[icell] = surf->Vt;
          Vx[icell] = surf->Vx;
          Vy[icell] = surf->Vy;
          Vn[icell] = surf->Vn;
        }
      }

      if(HYDRO_MODE == 2)
      {
        Wt[icell] = surf->Wt;
        Wx[icell] = surf->Wx;
        Wy[icell] = surf->Wy;
        Wn[icell] = surf->Wn;

        Lambda[icell] = surf->Lambda;
        aL[icell] = surf->aL;

        if(DF_MODE == 4)
        {
          c0[icell] = surf->c0;
          c1[icell] = surf->c1;
          c2[icell] = surf->c2;
          c3[icell] = surf->c3;
          c4[icell] = surf->c4;
        }
      }
    }

    // df_coeff array:
    //  - holds {c0,c1,c2,c3,c4}              (14-moment vhydro)
    //      or  {F,G,betabulk,betaV,betapi}   (CE vhydro, modified vhydro)

    double df_coeff[5] = {0.0, 0.0, 0.0, 0.0, 0.0};


    if(HYDRO_MODE == 1)
    {
      // 14-moment vhydro
      switch(DF_MODE)
        {
          case 1: // 14 moment
          {
            df_coeff[0] = df.c0;
            df_coeff[1] = df.c1;
            df_coeff[2] = df.c2;
            df_coeff[3] = df.c3;
            df_coeff[4] = df.c4;

            // print coefficients
            cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << "\tc3 = " << df_coeff[3] << "\tc4 = " << df_coeff[4] << endl;
            break;
          }
          case 2: // Chapman enskog
          case 3: // Modified equilibrium
          {
            df_coeff[0] = df.F;
            df_coeff[1] = df.G;
            df_coeff[2] = df.betabulk;
            df_coeff[3] = df.betaV;
            df_coeff[4] = df.betapi;

            // print coefficients
            cout << "F = " << df_coeff[0] << "\tG = " << df_coeff[1] << "\tbetabulk = " << df_coeff[2] << "\tbetaV = " << df_coeff[3] << "\tbetapi = " << df_coeff[4] << endl;
            break;
          }
          default:
          {
            cout << "Please specify df_mode = (1,2,3) in parameters.dat..." << endl;
            exit(-1);
          }
        }

      // launch function to perform integrations - this should be readily parallelizable (VH)
      calculate_dN_ptdptdphidy(Mass, Sign, Degen, Baryon, EquilibriumDensity, LinearDensityCorrection,
        T, P, E, tau, eta, ut, ux, uy, un,
        dat, dax, day, dan,
        pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
        muB, nB, Vt, Vx, Vy, Vn, df_coeff);

    }

    if(HYDRO_MODE == 2)
    {

      // launch function to perform integrations (VAH_PL)
      calculate_dN_ptdptdphidy_VAH_PL(Mass, Sign, Degen,
        tau, eta, ut, ux, uy, un,
        dat, dax, day, dan,
        pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
        Wt, Wx, Wy, Wn, Lambda, aL, c0, c1, c2, c3, c4);
    }



    //write the results to file
    write_dN_pTdpTdphidy_toFile();

    cout << "Freeing memory" << endl;
    // free memory
    free(Mass);
    free(Sign);
    free(Baryon);

    if(DF_MODE == 3)
    {
       free(EquilibriumDensity);
      free(LinearDensityCorrection);
    }

    if(HYDRO_MODE == 1)
    {
      free(E);
      free(P);
    }

    free(T);

    free(tau);
    free(eta);

    free(ut);
    free(ux);
    free(uy);
    free(un);

    free(dat);
    free(dax);
    free(day);
    free(dan);

    free(pitt);
    free(pitx);
    free(pity);
    free(pitn);
    free(pixx);
    free(pixy);
    free(pixn);
    free(piyy);
    free(piyn);
    free(pinn);

    free(bulkPi);

    if(INCLUDE_BARYON)
    {
      free(muB);
      if(INCLUDE_BARYONDIFF_DELTAF)
      {
        free(nB);
        free(Vt);
        free(Vx);
        free(Vy);
        free(Vn);
      }
    }

    if(HYDRO_MODE == 2)
    {
      free(Wt);
      free(Wx);
      free(Wy);
      free(Wn);
      free(Lambda);
      free(aL);
      if(DF_MODE == 4)
      {
        free(c0);
        free(c1);
        free(c2);
        free(c3);
        free(c4);
      }
    }

    sw.toc();
    cout << " -- calculate_spectra() finished " << sw.takeTime() << " seconds." << endl;
  }
    /*
    bool EmissionFunctionArray::particles_are_the_same(int idx1, int idx2)
    {
    if (particles[idx1].sign!=particles[idx2].sign) return false;
    if (particles[idx1].baryon!=particles[idx2].baryon && INCLUDE_MUB) return false;
    if (abs(particles[idx1].mass-particles[idx2].mass) / (particles[idx2].mass+1e-30) > PARTICLE_DIFF_TOLERANCE) return false;
    for (long l=0; l<FO_length; l++)
    {
    double chem1 = FOsurf_ptr[l].particle_mu[idx1], chem2 = FOsurf_ptr[l].particle_mu[idx2];
    if (abs(chem1-chem2)/(chem2+1e-30) > PARTICLE_DIFF_TOLERANCE)
    {
    return false;
  }

}

return true;
}
*/
