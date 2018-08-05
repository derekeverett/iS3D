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
#define AMOUNT_OF_OUTPUT 0 // smaller value means less outputs

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

    DO_RESONANCE_DECAYS = paraRdr->getVal("do_resonance_decays");

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


  void EmissionFunctionArray::write_dN_pTdpTdphidy_toFile()
  {
    printf("Writing thermal spectra to file...\n");
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
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }

    void EmissionFunctionArray::write_dN_pTdpTdphidy_with_resonance_decays_toFile()
  {
    printf("Writing thermal + resonance decays spectra to file...\n");
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";

    int y_pts = y_tab_length;     // default 3+1d pts
    if(DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)

    sprintf(filename, "results/dN_pTdpTdphidy_resonance_decays.dat");
    ofstream spectraFile(filename, ios_base::app);
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);

        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }

  void EmissionFunctionArray::write_dN_dpTdphidy_toFile()
  {
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";
    int y_pts = y_tab_length;     // default 3+1d pts
    if (DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)
    sprintf(filename, "results/dN_dpTdphidy.dat");
    ofstream spectraFile(filename, ios_base::app);
    //write the header
    spectraFile << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_dpTdphidy" << "\n";
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
            double value = dN_pTdpTdphidy[iS3D] * pT;
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << value << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }

    void EmissionFunctionArray::write_dN_dpTdphidy_with_resonance_decays_toFile()
  {
    //write 3D spectra in block format, different blocks for different species,
    //different sublocks for different values of rapidity
    //rows corespond to phip and columns correspond to pT
    int npart = number_of_chosen_particles;
    char filename[255] = "";
    int y_pts = y_tab_length;     // default 3+1d pts
    if (DIMENSION == 2) y_pts = 1; // 2+1d pts (y = 0)
    sprintf(filename, "results/dN_dpTdphidy_resonance_decays.dat");
    ofstream spectraFile(filename, ios_base::app);
    //write the header
    spectraFile << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_dpTdphidy" << "\n";
    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for (int iy = 0; iy < y_pts; iy++)
      {
        double y;
        if (DIMENSION == 2) y = 0.0;
        else y = y_tab->get(1,iy + 1);
        for (int iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for (int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
            double value = dN_pTdpTdphidy[iS3D] * pT;
            spectraFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << value << "\n";
          } //ipT
          spectraFile << "\n";
        } //iphip
      } //iy
    }//ipart
    spectraFile.close();
  }

  void EmissionFunctionArray::write_particle_list_toFile()
  {
    printf("Writing sampled particles list to file...\n");
    char filename[255] = "";
    sprintf(filename, "results/particle_list.dat");
    ofstream spectraFile(filename, ios_base::app);
    int num_particles = particle_list.size();
    //write the header
    spectraFile << "mcid" << "," << "tau" << "," << "x" << "," << "y" << "," << "eta" << "," << "E" << "," << "px" << "," << "py" << "," << "pz" << "\n";
    for (int ipart = 0; ipart < num_particles; ipart++)
    {
      int mcid = particle_list[ipart].mcID;
      double tau = particle_list[ipart].tau;
      double x = particle_list[ipart].x;
      double y = particle_list[ipart].y;
      double eta = particle_list[ipart].eta;
      double E = particle_list[ipart].E;
      double px = particle_list[ipart].px;
      double py = particle_list[ipart].py;
      double pz = particle_list[ipart].pz;
      spectraFile << scientific <<  setw(5) << setprecision(8) << mcid << "," << tau << "," << x << "," << y << "," << eta << "," << E << "," << px << "," << py << "," << pz << "\n";
    }//ipart
    spectraFile.close();
  }


  //*********************************************************************************************
  void EmissionFunctionArray::calculate_spectra()
  {
    cout << "calculate_spectra() has started ";
    Stopwatch sw;
    sw.tic();

    //fill arrays with all particle info and freezeout info to pass to function which will perform the integral

    //particle info
    particle_info *particle;
    double *Mass, *Sign, *Degen, *Baryon;
    int *MCID;
    Mass = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Sign = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Degen = (double*)calloc(number_of_chosen_particles, sizeof(double));
    Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));
    MCID = (int*)calloc(number_of_chosen_particles, sizeof(int));

    for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      int particle_idx = chosen_particles_sampling_table[ipart];
      particle = &particles[particle_idx];
      Mass[ipart] = particle->mass;
      Sign[ipart] = particle->sign;
      Degen[ipart] = particle->gspin;
      Baryon[ipart] = particle->baryon;
      MCID[ipart] = particle->mc_id;
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
    double *tau, *x, *y, *eta;
    double *ut, *ux, *uy, *un;
    double *dat, *dax, *day, *dan;
    double *pitt, *pitx, *pity, *pitn, *pixx, *pixy, *pixn, *piyy, *piyn, *pinn, *bulkPi;
    double *muB, *nB, *Vt, *Vx, *Vy, *Vn;

    tau = (double*)calloc(FO_length, sizeof(double));
    x = (double*)calloc(FO_length, sizeof(double));
    y = (double*)calloc(FO_length, sizeof(double));
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
      x[icell] = surf->x;
      y[icell] = surf->y;
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

    // for sampling and modified thermal spectra
    // read in gauss laguerre roots and weights
    // for gauss laguerre quadrature

    FILE * gla_file;
    char header[300];
    gla_file = fopen("tables/gla123_roots_weights_32_pts.dat", "r");
    if(gla_file == NULL) printf("Couldn't open Gauss-Laguerre roots/weights file\n");

    int pbar_pts;

    // get # quadrature pts
    fscanf(gla_file, "%d\n", &pbar_pts);

    // Gauss Laguerre roots-weights
    double pbar_root1[pbar_pts];
    double pbar_weight1[pbar_pts];
    double pbar_root2[pbar_pts];
    double pbar_weight2[pbar_pts];
    double pbar_root3[pbar_pts];
    double pbar_weight3[pbar_pts];

    // skip the next 2 headers
    fgets(header, 100, gla_file);
    fgets(header, 100, gla_file);

    // load roots/weights
    for (int i = 0; i < pbar_pts; i++) fscanf(gla_file, "%lf\t\t%lf\t\t%lf\t\t%lf\t\t%lf\t\t%lf\n", &pbar_root1[i], &pbar_weight1[i], &pbar_root2[i], &pbar_weight2[i], &pbar_root3[i], &pbar_weight3[i]);

    // close file
    fclose(gla_file);

    // df_coeff array:
    //  - holds {c0,c1,c2,c3,c4}              (14-moment vhydro)
    //      or  {F,G,betabulk,betaV,betapi}   (CE vhydro, modified vhydro)

    double df_coeff[5] = {0.0, 0.0, 0.0, 0.0, 0.0};


    if(MODE == 1) // viscous hydro
    {
      switch(DF_MODE)
      {
        case 1: // 14-moment
        {
          df_coeff[0] = df.c0;
          df_coeff[1] = df.c1;
          df_coeff[2] = df.c2;
          df_coeff[3] = df.c3;
          df_coeff[4] = df.c4;

          // print coefficients
          //cout << "c0 = " << df_coeff[0] << "\tc1 = " << df_coeff[1] << "\tc2 = " << df_coeff[2] << "\tc3 = " << df_coeff[3] << "\tc4 = " << df_coeff[4] << endl;

          switch(OPERATION)
          {
            case 1: // thermal
            {
              calculate_dN_pTdpTdphidy(Mass, Sign, Degen, Baryon,
              T, P, E, tau, eta, ut, ux, uy, un,
              dat, dax, day, dan,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              muB, nB, Vt, Vx, Vy, Vn, df_coeff);
              break;
            }
            case 2: // sample
            {
              sample_dN_pTdpTdphidy(Mass, Sign, Degen, Baryon, MCID,
              T, P, E, tau, x, y, eta, ut, ux, uy, un,
              dat, dax, day, dan,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              muB, nB, Vt, Vx, Vy, Vn, df_coeff,
              pbar_pts, pbar_root1, pbar_weight1, pbar_root2, pbar_weight2, pbar_root3, pbar_weight3);
              break;
            }
            default:
            {
              cout << "Set operation to 1 or 2" << endl;
              exit(-1);
            }
          }
          break;
        }
        case 2: // Chapman Enskog
        {
          df_coeff[0] = df.F;
          df_coeff[1] = df.G;
          df_coeff[2] = df.betabulk;
          df_coeff[3] = df.betaV;
          df_coeff[4] = df.betapi;

          // print coefficients
          //cout << "F = " << df_coeff[0] << "\tG = " << df_coeff[1] << "\tbetabulk = " << df_coeff[2] << "\tbetaV = " << df_coeff[3] << "\tbetapi = " << df_coeff[4] << endl;

          switch(OPERATION)
          {
            case 1: // thermal
            {
              calculate_dN_pTdpTdphidy(Mass, Sign, Degen, Baryon,
              T, P, E, tau, eta, ut, ux, uy, un,
              dat, dax, day, dan,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              muB, nB, Vt, Vx, Vy, Vn, df_coeff);
              break;
            }
            case 2: // sample
            {
              sample_dN_pTdpTdphidy(Mass, Sign, Degen, Baryon, MCID,
              T, P, E, tau, x, y, eta, ut, ux, uy, un,
              dat, dax, day, dan,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              muB, nB, Vt, Vx, Vy, Vn, df_coeff,
              pbar_pts, pbar_root1, pbar_weight1, pbar_root2, pbar_weight2, pbar_root3, pbar_weight3);
              break;
            }
            default:
            {
              cout << "Set operation to 1 or 2" << endl;
              exit(-1);
            }
          }
          break;
        }
        case 3: // modified
        {
          df_coeff[0] = df.F;
          df_coeff[1] = df.G;
          df_coeff[2] = df.betabulk;
          df_coeff[3] = df.betaV;
          df_coeff[4] = df.betapi;

          // print coefficients
          //cout << "F = " << df_coeff[0] << "\tG = " << df_coeff[1] << "\tbetabulk = " << df_coeff[2] << "\tbetaV = " << df_coeff[3] << "\tbetapi = " << df_coeff[4] << endl;

          switch(OPERATION)
          {
            case 1: // thermal
            {
              calculate_dN_ptdptdphidy_feqmod(Mass, Sign, Degen, Baryon,
              T, P, E, tau, eta, ux, uy, un,
              dat, dax, day, dan,
              pixx, pixy, pixn, piyy, piyn, bulkPi,
              muB, nB, Vx, Vy, Vn, df_coeff, pbar_pts, pbar_root1, pbar_root2, pbar_weight1, pbar_weight2);
                break;
            }
            case 2: // sample
            {

              sample_dN_pTdpTdphidy_feqmod(Mass, Sign, Degen, Baryon, MCID,
              T, P, E, tau, x, y, eta, ut, ux, uy, un,
              dat, dax, day, dan,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              muB, nB, Vt, Vx, Vy, Vn, df_coeff,
              pbar_pts, pbar_root1, pbar_weight1, pbar_root2, pbar_weight2);

              break;
            }
            default:
            {
              cout << "Set operation to 1 or 2" << endl;
              exit(-1);
            }
          }
          break;
        }
        default:
        {
          cout << "Please specify df_mode = (1,2,3) in parameters.dat..." << endl;
          exit(-1);
        }
      }
    } //if (MODE == 1)
    else if(MODE == 2)
    {
      switch(OPERATION)
      {
        case 1: // thermal
        {
          calculate_dN_pTdpTdphidy_VAH_PL(Mass, Sign, Degen,
              tau, eta, ux, uy, un,
              dat, dax, day, dan, T,
              pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
              Wx, Wy, Lambda, aL, c0, c1, c2, c3, c4);

          break;
        }
        case 2: // sample
        {
          sample_dN_pTdpTdphidy_VAH_PL(Mass, Sign, Degen,
                tau, eta, ux, uy, un,
                dat, dax, day, dan, T,
                pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, bulkPi,
                Wx, Wy, Lambda, aL, c0, c1, c2, c3, c4);
          break;
        }
        default:
        {
          cout << "Please specify df_mode = (1,2,3) in parameters.dat..." << endl;
          exit(-1);
        }
      }
    }

    printline();


    //write the results to file
    if (OPERATION == 1)
    {
      write_dN_pTdpTdphidy_toFile();
      write_dN_dpTdphidy_toFile();

      // option to do resonance decays option
      if(DO_RESONANCE_DECAYS)
      {
        // call resonance decays routine
        do_resonance_decays(particles);

        // write amended spectra from resonance decays to file
        // currently, unstable particles are still included (should change)
        write_dN_pTdpTdphidy_with_resonance_decays_toFile();
        write_dN_dpTdphidy_with_resonance_decays_toFile();
      }
    }
    else if (OPERATION == 2) write_particle_list_toFile();

    cout << "Freeing freezeout surface memory.." << endl;
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
    cout << "calculate_spectra() took " << sw.takeTime() << " seconds." << endl;
  }

  // void EmissionFunctionArray::do_resonance_decays()
  // {
  //   printf("Starting resonance decays \n");

  //   //declare a particle struct to hold all the particle info
  //   particle particles[10];

  //   //read the in the resonance info
  //   //int max, maxdecay;
  //   //readParticleData("PDG/pdg.dat", &max, &maxdecay, particles)
  //   //calc_reso_decays(max, maxdecay, LIGHTEST_PARTICLE);
  //   printf("Resonance decays finished \n");
  // }
