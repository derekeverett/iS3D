#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
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
#ifdef _OPENACC
#include <accelmath.h>
#endif
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

//#define FORCE_F0

using namespace std;


// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in,
  Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in,
  int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in,  deltaf_coefficients df_in)
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
    OPERATION = paraRdr->getVal("operation");
    MODE = paraRdr->getVal("mode");
    DF_MODE = paraRdr->getVal("df_mode");
    DIMENSION = paraRdr->getVal("dimension");
    INCLUDE_BARYON = paraRdr->getVal("include_baryon");
    INCLUDE_BULK_DELTAF = paraRdr->getVal("include_bulk_deltaf");
    INCLUDE_SHEAR_DELTAF = paraRdr->getVal("include_shear_deltaf");
    INCLUDE_BARYONDIFF_DELTAF = paraRdr->getVal("include_baryondiff_deltaf");
    REGULATE_DELTAF = paraRdr->getVal("regulate_deltaf");
    GROUP_PARTICLES = paraRdr->getVal("group_particles");
    PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");

    LIGHTEST_PARTICLE = paraRdr->getVal("lightest_particle");

    particles = particles_in;
    Nparticles = Nparticles_in;
    surf_ptr = surf_ptr_in;
    FO_length = FO_length_in;
    df = df_in;
    number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

    chosen_particles_01_table = new int[Nparticles];

    //a class member to hold 3D smooth CF spectra for all chosen particles
    dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
    //zero the array
    for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
    {
      dN_pTdpTdphidy[iSpectra] = 0.0;
    }

    //a class member to hold the particle list for sampling
    //how big does this need to be? probably need to make it a vector instead so we can resize it...
    particle_list = new sampled_particle [10000];

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
    delete[] particle_list; //for holding particle list of sampled particles
  }

  void EmissionFunctionArray::sample_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
    double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
    double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
    double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff,
    int pbar_pts, double *pbar_root1, double *pbar_weight1, double *pbar_root2, double *pbar_weight2)
    {
      printf("Sampling particles from VH \n");

      //double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
      int npart = number_of_chosen_particles;
      int particle_index = 0; //runs over all particles in final sampled particle list

      int eta_pts = 1;
      if(DIMENSION == 2) eta_pts = eta_tab_length;
      double etaValues[eta_pts];
      double etaDeltaWeights[eta_pts];    // eta_weight * delta_eta
      double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

      if(DIMENSION == 2)
      {
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
      }

      //get viscous correction coefficients
      double F        = df_coeff[0];
      double G        = df_coeff[1];
      double betabulk = df_coeff[2];
      double betaV    = df_coeff[3];
      double betapi   = df_coeff[4];

      //loop over all freezeout cells
      for (int icell = 0; icell < FO_length; icell++)
      {
        // set freezeout info to local varibles to reduce(?) memory access outside cache :
        double tau = tau_fo[icell];         // longitudinal proper time
        double tau2 = tau * tau;

        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell];     // spacetime rapidity from surface file
        }

        //FIX THIS - NEED A LOOP OVER ETA FOR CASE OF 2D SURFACE
        double eta = etaValues[0];
        //FIX THIS

        double dat = dat_fo[icell];         // covariant normal surface vector
        double dax = dax_fo[icell];
        double day = day_fo[icell];
        double dan = dan_fo[icell];

        double ux = ux_fo[icell];           // contravariant fluid velocity
        double uy = uy_fo[icell];           // reinforce normalization
        double un = un_fo[icell];
        double ut = sqrt(fabs(1.0 + ux*ux + uy*uy + tau2*un*un));

        double T = T_fo[icell];             // temperature
        double E = E_fo[icell];             // energy density
        double P = P_fo[icell];             // pressure

        double pitt = pitt_fo[icell];       // pi^munu
        double pitx = pitx_fo[icell];
        double pity = pity_fo[icell];
        double pitn = pitn_fo[icell];
        double pixx = pixx_fo[icell];
        double pixy = pixy_fo[icell];
        double pixn = pixn_fo[icell];
        double piyy = piyy_fo[icell];
        double piyn = piyn_fo[icell];
        double pinn = pinn_fo[icell];

        double bulkPi = bulkPi_fo[icell];   // bulk pressure

        double muB = 0.0;
        double alphaB = 0.0;                       // baryon chemical potential
        double nB = 0.0;                        // net baryon density
        double Vt = 0.0;                        // baryon diffusion
        double Vx = 0.0;
        double Vy = 0.0;
        double Vn = 0.0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell];
          alphaB = muB / T;

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = nB_fo[icell];
            Vt = Vt_fo[icell];
            Vx = Vx_fo[icell];
            Vy = Vy_fo[icell];
            Vn = Vn_fo[icell];
          }
        }

        //common prefactor
        double udotdsigma = dat*ut + dax*ux + day*uy + dan*un;
        double num_factor = 1.0 / 2.0 / M_PI / M_PI / hbarC / hbarC / hbarC;
        double prefactor = num_factor * udotdsigma * T;

        //the total number of particles (all species) for the FO cell
        double dN_tot = 0.0;

        //a list with species dependent densities
        std::vector<double> density_list;
        density_list.resize(npart);

        //loop over all species
        for (int ipart = 0; ipart < npart; ipart++)
        {
          //printf("ipart = %d \n", ipart);
          // set particle properties
          double mass = Mass[ipart];    // (GeV)
          double mass2 = mass * mass;
          double mbar = mass / T;
          double sign = Sign[ipart];
          double degeneracy = Degeneracy[ipart];
          double baryon = 0.0;
          double chem = 0.0;           // chemical potential term in feq
          if(INCLUDE_BARYON)
          {
            baryon = Baryon[ipart];
            chem = baryon * muB;
          }

          //first calculate thermal equil. number of particles
          double sum = 0.0;
          double dN_thermal = 0.0;

          //for light particles need more than leading order term in BE or FD expansion
          int sum_max = 2;
          if ( mass / T < 2.0 ) sum_max = 10;

          for (int kk = 1; kk < sum_max; kk++) //truncate expansion of BE or FD distribution
          {
            double k = (double)kk;
            sum += pow(-sign, kk+1) * exp(k * chem / T) * gsl_sf_bessel_Kn(2, k*mbar) / k;
          }
          dN_thermal = prefactor * sum * degeneracy * mass2;

          //correction from bulk pressure
          double dN_bulk = 0.0;
          if (INCLUDE_BULK_DELTAF)
          {
            //printf("starting bulk\n");
            double N10_fact = pow(T,3) / (2.0 * M_PI * M_PI) / hbarC / hbarC / hbarC;
            double J20_fact = pow(T,4) / (2.0 * M_PI * M_PI) / hbarC / hbarC / hbarC;
            double N10 = degeneracy * N10_fact * GaussThermal(N10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
            double J20 = degeneracy * J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
            dN_bulk = (bulkPi / betabulk) * ( dN_thermal + (N10 * G) + (J20 * F / pow(T,2)) );
            //printf("finished bulk\n");
          }

          //correction from diffusion current
          double dN_diff = 0.0;
          if (INCLUDE_BARYONDIFF_DELTAF)
          {

          }

          //add them up
          //this is the number of particles of species ipart
          double dN = dN_thermal + dN_bulk + dN_diff;

          //save to a list to later sample inidividual species
          density_list[ipart] = dN;

          //add this to the total mean number of hadrons for the FO cell
          dN_tot += dN;

      }

      //now sample the number of particles (all species)
      //non-deterministic random number generator
      std::random_device gen1;
      std::random_device gen2;

      //deterministic random number generator
      //std::mt19937 gen(1701);

      std::poisson_distribution<> poisson_distr(dN_tot);

      //total number of hadrons in FO cell
      int N_hadrons = poisson_distr(gen1);

      //make a discrete distribution with weights according to particle densities
      std::discrete_distribution<int> particle_numbers ( density_list.begin(), density_list.end() );

      for (int n = 0; n < N_hadrons; n++)
      {
        //printf("n = %d \n", n);
        int idx_sampled = particle_numbers(gen2); //this gives the index of the sampled particle
        //printf("idx_sampled = %d \n", idx_sampled);

        //get the mass of the sampled particle
        double mass = Mass[idx_sampled];  // (GeV)

        //get the MC ID of the sampled particle
        //TO DO : pass an array that holds this info
        //int mcid = mcID[idx_sampled];

        //sample the momentum from distribution using Scott Pratt Trick
        //stuff

        //printf("particle_index = %d \n", particle_index);
        //set coordinates of production to FO cell coords
        particle_list[particle_index].tau = tau;
        //TO DO
        //particle_list[particle_index].x = x;
        //particle_list[particle_index].y = y;
        particle_list[particle_index].eta = eta;

        //FIX MOMENTA TO NONTRIVIAL VALUES
        //set particle momentum to sampled values
        particle_list[particle_index].E = 0.0;
        particle_list[particle_index].px = 0.0;
        particle_list[particle_index].py = 0.0;
        particle_list[particle_index].pz = 0.0;
        // FIX MOMENTA TO NONTRIVIAL VALUES

        particle_index += 1;
        //printf("particle_index = %d\n", particle_index);

      } // for (int n = 0; n < N_hadrons; n++)
    } // for (int icell = 0; icell < FO_length; icell++)
  }

  void EmissionFunctionArray::sample_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
    double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
    double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
    double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
  {
    printf("Sampling particles from VAH \n");

  }

  void EmissionFunctionArray::calculate_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
    double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ut_fo, double *ux_fo, double *uy_fo, double *un_fo,
    double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
    double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo, double *bulkPi_fo,
    double *muB_fo, double *nB_fo, double *Vt_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, double *df_coeff)

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

      //fill arrays with values points in pT
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

          if(DIMENSION == 3)
          {
            etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
          }
          double dat = dat_fo[icell_glb];         // covariant normal surface vector
          double dax = dax_fo[icell_glb];
          double day = day_fo[icell_glb];
          double dan = dan_fo[icell_glb];

          double ux = ux_fo[icell_glb];           // contravariant fluid velocity
          double uy = uy_fo[icell_glb];           // reinforce normalization
          double un = un_fo[icell_glb];
          double ut = sqrt(fabs(1.0 + ux*ux + uy*uy + tau2*un*un));

          double T = T_fo[icell_glb];             // temperature
          double E = E_fo[icell_glb];             // energy density
          double P = P_fo[icell_glb];             // pressure

          double pitt = pitt_fo[icell_glb];       // pi^munu
          double pitx = pitx_fo[icell_glb];
          double pity = pity_fo[icell_glb];
          double pitn = pitn_fo[icell_glb];
          double pixx = pixx_fo[icell_glb];
          double pixy = pixy_fo[icell_glb];
          double pixn = pixn_fo[icell_glb];
          double piyy = piyy_fo[icell_glb];
          double piyn = piyn_fo[icell_glb];
          double pinn = pinn_fo[icell_glb];

          double bulkPi = bulkPi_fo[icell_glb];   // bulk pressure

          double muB = 0.0;                       // baryon chemical potential
          double nB = 0.0;                        // net baryon density
          double Vt = 0.0;                        // baryon diffusion
          double Vx = 0.0;
          double Vy = 0.0;
          double Vn = 0.0;

          if(INCLUDE_BARYON)
          {
            muB = muB_fo[icell_glb];

            if(INCLUDE_BARYONDIFF_DELTAF)
            {
              nB = nB_fo[icell_glb];
              Vt = Vt_fo[icell_glb];
              Vx = Vx_fo[icell_glb];
              Vy = Vy_fo[icell_glb];
              Vn = Vn_fo[icell_glb];
            }
          }

          double shear_coeff = 0.5 / (T * T * (E + P));  // (see df_shear)

          //now loop over all particle species and momenta
          for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
          {
            // set particle properties
            double mass = Mass[ipart];    // (GeV)
            double mass2 = mass * mass;
            double sign = Sign[ipart];
            double degeneracy = Degeneracy[ipart];
            double baryon = 0.0;
            double chem = 0.0;           // chemical potential term in feq
            if(INCLUDE_BARYON)
            {
              baryon = Baryon[ipart];
              chem = baryon * muB;
            }

            for (int ipT = 0; ipT < pT_tab_length; ipT++)
            {
              // set transverse radial momentum and transverse mass (GeV)
              double pT = pTValues[ipT];
              double mT = sqrt(mass2 + pT * pT);
              // useful expression
              double mT_over_tau = mT / tau;

              for (int iphip = 0; iphip < phi_tab_length; iphip++)
              {
                double px = pT * trig_phi_table[iphip][0]; //contravariant
                double py = pT * trig_phi_table[iphip][1]; //contravariant

                for (int iy = 0; iy < y_pts; iy++)
                {
                  //all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                  double y = yValues[iy];

                  double pdotdsigma_f_eta_sum = 0.0;

                  // sum over eta
                  for (int ieta = 0; ieta < eta_pts; ieta++)
                  {
                    double eta = etaValues[ieta];
                    double delta_eta_weight = etaDeltaWeights[ieta];

                    double pt = mT * cosh(y - eta); // contravariant
                    double pn = mT_over_tau * sinh(y - eta); // contravariant
                    // useful expression
                    double tau2_pn = tau2 * pn;

                    //momentum vector is contravariant, surface normal vector is COVARIANT
                    double pdotdsigma = pt * dat + px * dax + py * day + pn * dan;

                    //thermal equilibrium distributions - for viscous hydro
                    double pdotu = pt * ut - px * ux - py * uy - tau2_pn * un;    // u.p = LRF energy
                    double feq = 1.0 / ( exp(( pdotu - chem ) / T) + sign);

                    //viscous corrections
                    double feqbar = 1.0 - sign*feq;

                    // shear correction:
                    double df_shear = 0.0;
                    if (INCLUDE_SHEAR_DELTAF)
                    {
                      // pi^munu * p_mu * p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt + pixx * px * px + piyy * py * py + pinn * tau2_pn * tau2_pn
                      + 2.0 * (-(pitx * px + pity * py) * pt + pixy * px * py + tau2_pn * (pixn * px + piyn * py - pitn * pt));

                      df_shear = shear_coeff * pimunu_pmu_pnu;  // df / (feq*feqbar)
                    }

                    // bulk correction:
                    double df_bulk = 0.0;
                    if (INCLUDE_BULK_DELTAF)
                    {
                      double c0 = df_coeff[0];
                      double c2 = df_coeff[2];
                      if (INCLUDE_BARYON)
                      {
                        double c1 = df_coeff[1];
                        df_bulk = ((c0-c2)*mass2 + c1 * baryon * pdotu + (4.0*c2-c0)*pdotu*pdotu) * bulkPi;
                      }
                      else
                      {
                        df_bulk = ((c0-c2)*mass2 + (4.0*c2-c0)*pdotu*pdotu) * bulkPi;
                      }
                    }

                    // baryon diffusion correction:
                    double df_baryondiff = 0.0;
                    if (INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
                    {
                      double c3 = df_coeff[3];
                      double c4 = df_coeff[4];
                      // V^mu * p_mu
                      double Vmu_pmu = Vt * pt - Vx * px - Vy * py - Vn * tau2_pn;
                      df_baryondiff = (baryon * c3 + c4 * pdotu) * Vmu_pmu;
                    }
                    double df = df_shear + df_bulk + df_baryondiff;


                    if (REGULATE_DELTAF)
                    {
                      double reg_df = max( -1.0, min( feqbar * df, 1.0 ) );
                      pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * feq * (1.0 + reg_df));
                    }
                    else pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * feq * (1.0 + df));

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
          #pragma omp parallel for collapse(4)
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

    // haven't defined in the header yet...
    void EmissionFunctionArray::calculate_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
      double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
      double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
      double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
      double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
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

            double T = T_fo[icell_glb];

            double ux = ux_fo[icell_glb];           // contravariant fluid velocity
            double uy = uy_fo[icell_glb];           // enforce normalization: u.u = 1
            double un = un_fo[icell_glb];
            double ut = sqrt(1.0 + ux*ux + uy*uy + tau2*un*un);

            double u0 = sqrt(1.0 + ux*ux + uy*uy);
            double zt = tau * un / u0;
            double zn = ut / (u0 * tau);            // z basis vector

            double pitt = pitt_fo[icell_glb];       // piperp^munu
            double pitx = pitx_fo[icell_glb];
            double pity = pity_fo[icell_glb];
            double pitn = pitn_fo[icell_glb];
            double pixx = pixx_fo[icell_glb];
            double pixy = pixy_fo[icell_glb];
            double pixn = pixn_fo[icell_glb];
            double piyy = piyy_fo[icell_glb];
            double piyn = piyn_fo[icell_glb];
            double pinn = pinn_fo[icell_glb];

            double bulkPi = bulkPi_fo[icell_glb];   // residual bulk pressure

            double Wx = Wx_fo[icell_glb];           // W^mu
            double Wy = Wy_fo[icell_glb];           // reinforce orthogonality
            double Wt = (ux * Wx  +  uy * Wy) * ut / (u0 * u0);
            double Wn = Wt * un / ut;

            double Lambda = Lambda_fo[icell_glb];   // anisotropic variables
            double aL = aL_fo[icell_glb];

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
                // useful expression
                double mT_over_tau = mT / tau;

                for (int iphip = 0; iphip < phi_tab_length; iphip++)
                {
                  double px = pT * trig_phi_table[iphip][0]; //contravariant
                  double py = pT * trig_phi_table[iphip][1]; //contravariant

                  for (int iy = 0; iy < y_pts; iy++)
                  {
                    // all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                    double y = yValues[iy];

                    double pdotdsigma_f_eta_sum = 0.0;

                    // sum over eta
                    for (int ieta = 0; ieta < eta_pts; ieta++)
                    {
                      double eta = etaValues[ieta];
                      double delta_eta_weight = etaDeltaWeights[ieta];

                      double pt = mT * cosh(y - eta);
                      double pn = mT_over_tau * sinh(y - eta);

                      // useful expression
                      double tau2_pn = tau2 * pn;

                      // momentum vector is contravariant, surface normal vector is COVARIANT
                      double pdotdsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;

                      //thermal equilibrium distributions - for viscous hydro
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;    // u.p = LRF energy
                      double pdotz = pt * zt  -  tau2_pn * zn;                            // z.p = - LRF longitudinal momentum

                      double xiL = 1.0 / (aL * aL)  -  1.0;

                      #ifdef FORCE_F0
                      Lambda = T;
                      xiL = 0.0;
                      #endif

                      double Ea = sqrt(pdotu * pdotu  +  xiL * pdotz * pdotz);

                      double fa = 1.0 / ( exp(Ea / Lambda) + sign);

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

                      double df = df_shear + df_bulk;

                      if (REGULATE_DELTAF)
                      {
                        double reg_df = max( -1.0, min( fabar * df, 1.0 ) );
                        pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * fa * (1.0 + reg_df));
                      }
                      else pdotdsigma_f_eta_sum += (delta_eta_weight * pdotdsigma * fa * (1.0 + fabar * df));

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
            #pragma omp parallel for collapse(4)
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

        int y_pts = y_tab_length;     // default 3+1d pts
        if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

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
                spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
              } //ipT
              spectraFile << "\n";
            } //iphip
          } //iy
        }//ipart
        spectraFile.close();
      }


      //*********************************************************************************************
      void EmissionFunctionArray::calculate_spectra()
      {
        cout << "calculate_spectra() has started... " << endl;
        Stopwatch sw;
        sw.tic();

        //fill arrays with all particle info and freezeout info to pass to function which will perform the integral

        //particle info
        particle_info *particle;
        double *Mass, *Sign, *Degen, *Baryon;
        Mass = (double*)calloc(number_of_chosen_particles, sizeof(double));
        Sign = (double*)calloc(number_of_chosen_particles, sizeof(double));
        Degen = (double*)calloc(number_of_chosen_particles, sizeof(double));
        Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));

        for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
        {
          int particle_idx = chosen_particles_sampling_table[ipart];
          particle = &particles[particle_idx];
          Mass[ipart] = particle->mass;
          Sign[ipart] = particle->sign;
          Degen[ipart] = particle->gspin;
          Baryon[ipart] = particle->baryon;
        }

        // freezeout info
        FO_surf *surf = &surf_ptr[0];

        // freezeout surface info exclusive for VH
        double *E, *T, *P;
        if(MODE == 1)
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

        if (INCLUDE_BARYON)
        {
          muB = (double*)calloc(FO_length, sizeof(double));

          if (INCLUDE_BARYONDIFF_DELTAF)
          {
            nB = (double*)calloc(FO_length, sizeof(double));
            Vt = (double*)calloc(FO_length, sizeof(double));
            Vx = (double*)calloc(FO_length, sizeof(double));
            Vy = (double*)calloc(FO_length, sizeof(double));
            Vn = (double*)calloc(FO_length, sizeof(double));

          }
        }

        // freezeout surface info exclusive for VAH_PL
        double *PL, *Wx, *Wy;
        double *Lambda, *aL;
        double *c0, *c1, *c2, *c3, *c4; //delta-f coeffs for vah

        if(MODE == 2)
        {
          PL = (double*)calloc(FO_length, sizeof(double));

          //Wt = (double*)calloc(FO_length, sizeof(double));
          Wx = (double*)calloc(FO_length, sizeof(double));
          Wy = (double*)calloc(FO_length, sizeof(double));
          //Wn = (double*)calloc(FO_length, sizeof(double));

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

        for (long icell = 0; icell < FO_length; icell++)
        {
          //reading info from surface
          surf = &surf_ptr[icell];

          if(MODE == 1)
          {
            E[icell] = surf->E;
            P[icell] = surf->P;
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

          if (INCLUDE_BARYON)
          {
            muB[icell] = surf->muB;

            if (INCLUDE_BARYONDIFF_DELTAF)
            {
              nB[icell] = surf->nB;
              Vt[icell] = surf->Vt;
              Vx[icell] = surf->Vx;
              Vy[icell] = surf->Vy;
              Vn[icell] = surf->Vn;
            }
          }

          if(MODE == 2)
          {
            PL[icell] = surf->PL;
            //Wt[icell] = surf->Wt;
            Wx[icell] = surf->Wx;
            Wy[icell] = surf->Wy;
            //Wn[icell] = surf->Wn;

            Lambda[icell] = surf->Lambda;
            aL[icell] = surf->aL;

            if (DF_MODE == 4)
            {
              c0[icell] = surf->c0;
              c1[icell] = surf->c1;
              c2[icell] = surf->c2;
              c3[icell] = surf->c3;
              c4[icell] = surf->c4;
            }
          }
        }

        //for sampling, read in gla roots and weights
        FILE * gla_file;
        char header[300];
        gla_file = fopen("tables/gla12_roots_weights_64_pts.dat", "r");
        if(gla_file == NULL) printf("Couldn't open Gauss-Laguerre roots/weights file\n");

        int pbar_pts;

        // get # quadrature pts
        fscanf(gla_file, "%d\n", &pbar_pts);

        // Gauss Laguerre roots-weights
        double pbar_root1[pbar_pts];
        double pbar_weight1[pbar_pts];
        double pbar_root2[pbar_pts];
        double pbar_weight2[pbar_pts];

        // skip the next 2 headers
        fgets(header, 100, gla_file);
        fgets(header, 100, gla_file);

        // load roots/weights
        for (int i = 0; i < pbar_pts; i++) fscanf(gla_file, "%lf\t\t%lf\t\t%lf\t\t%lf\n", &pbar_root1[i], &pbar_weight1[i], &pbar_root2[i], &pbar_weight2[i]);

        // close file
        fclose(gla_file);

        // df_coeff array:
        //  - holds {c0,c1,c2,c3,c4}              (14-moment vhydro)
        //      or  {F,G,betabulk,betaV,betapi}   (CE vhydro, modified vhydro)

        double df_coeff[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

        if(MODE == 1)
        {
          // 14-moment vhydro
          if(DF_MODE == 1)
          {
            df_coeff[0] = df.c0;
            df_coeff[1] = df.c1;
            df_coeff[2] = df.c2;
            df_coeff[3] = df.c3;
            df_coeff[4] = df.c4;

            // print coefficients
            cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << "\tc3 = " << df_coeff[3] << "\tc4 = " << df_coeff[4] << endl;
          }
          else if (DF_MODE == 2 || DF_MODE == 3)

          {
            df_coeff[0] = df.F;
            df_coeff[1] = df.G;
            df_coeff[2] = df.betabulk;
            df_coeff[3] = df.betaV;
            df_coeff[4] = df.betapi;

            // print coefficients
            cout << "F = " << df_coeff[0] << "\tG = " << df_coeff[1] << "\tbetabulk = " << df_coeff[2] << "\tbetaV = " << df_coeff[3] << "\tbetapi = " << df_coeff[4] << endl;
          }

          if (OPERATION == 1)
          {
            // launch function to perform integrations - this should be readily parallelizable (VH)
            calculate_dN_pTdpTdphidy(Mass, Sign, Degen, Baryon,
              T, P, E, tau, eta, ut, ux, uy, un,
              dat, dax, day, dan,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              muB, nB, Vt, Vx, Vy, Vn, df_coeff);
            }

            else if (OPERATION == 2)
            {
              sample_dN_pTdpTdphidy(Mass, Sign, Degen, Baryon,
                T, P, E, tau, eta, ut, ux, uy, un,
                dat, dax, day, dan,
                pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
                muB, nB, Vt, Vx, Vy, Vn, df_coeff,
                pbar_pts, pbar_root1, pbar_weight1, pbar_root2, pbar_weight2);
            }

          } //if (MODE == 1)

          else if(MODE == 2)
          {
            if (OPERATION == 1)
            {
              // launch function to perform integrations (VAH_PL)
              calculate_dN_pTdpTdphidy_VAH_PL(Mass, Sign, Degen,
                tau, eta, ux, uy, un,
                dat, dax, day, dan, T,
                pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
                Wx, Wy, Lambda, aL, c0, c1, c2, c3, c4);
              }

              else if (OPERATION == 2)
              {
                sample_dN_pTdpTdphidy_VAH_PL(Mass, Sign, Degen,
                  tau, eta, ux, uy, un,
                  dat, dax, day, dan, T,
                  pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
                  Wx, Wy, Lambda, aL, c0, c1, c2, c3, c4);
              }

            }

            //write the results to file
            if (OPERATION == 1) write_dN_pTdpTdphidy_toFile();

            cout << "Freeing memory" << endl;
            // free memory
            free(Mass);
            free(Sign);
            free(Baryon);

            if(MODE == 1)
            {
              free(E);
              free(T);
              free(P);
            }

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

            if (INCLUDE_BARYON)
            {
              free(muB);
              if (INCLUDE_BARYONDIFF_DELTAF)
              {
                free(nB);
                free(Vt);
                free(Vx);
                free(Vy);
                free(Vn);
              }
            }
            if(MODE == 2)
            {
              free(PL);
              free(Wx);
              free(Wy);
              free(Lambda);
              free(aL);
              if (DF_MODE == 4)
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

          void EmissionFunctionArray::do_resonance_decays()
          {
            printf("Starting resonance decays \n");

            //declare a particle struct to hold all the particle info
            particle particles[10];

            //read the in the resonance info
            //int max, maxdecay;
            //readParticleData("PDG/pdg.dat", &max, &maxdecay, particles)
            //calc_reso_decays(max, maxdecay, LIGHTEST_PARTICLE);
            printf("Resonance decays finished \n");
          }
