
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>

#include "main.cuh"
#include "deltafReader.cuh"
#include "ParameterReader.cuh"
#include "gaussThermal.cuh"
#include "readindata.cuh"

using namespace std;


Deltaf_Data::Deltaf_Data(ParameterReader * paraRdr_in)
{
  paraRdr = paraRdr_in;

  hrg_eos = paraRdr->getVal("hrg_eos");
  mode = paraRdr->getVal("mode");
  df_mode = paraRdr->getVal("df_mode");
  include_baryon = paraRdr->getVal("include_baryon");

  if(hrg_eos == 1)
  {
    hrg_eos_path = "deltaf_coefficients/vh/urqmd/";
  }
  else if(hrg_eos == 2)
  {
    hrg_eos_path = "deltaf_coefficients/vh/smash/";
  }
  else if(hrg_eos == 3)
  {
    hrg_eos_path = "deltaf_coefficients/vh/smash_box/";
  }
  else
  {
    printf("Error: please choose hrg_eos = (1,2,3)\n");
    exit(-1);
  }

  jonah_points = 3001;       
  lambda_min = -1.0;       
  lambda_max = 2.0;
  delta_lambda = (lambda_max - lambda_min) / ((double)jonah_points - 1.0);
}

Deltaf_Data::~Deltaf_Data()
{
  // is there any harm in deallocating memory, while it's being used?
}

void Deltaf_Data::load_df_coefficient_data()
{
  printf("Reading in 14 moment and Chapman-Enskog coefficient tables...");

  // now string join
  char c0[100] = "";
  char c1[100] = "";
  char c2[100] = "";
  char c3[100] = "";
  char c4[100] = "";

  char F[100] = "";
  char G[100] = "";
  char betabulk[100] = "";
  char betaV[100] = "";
  char betapi[100] = "";

  sprintf(c0, "%s%s", hrg_eos_path.c_str(), "c0.dat");
  sprintf(c1, "%s%s", hrg_eos_path.c_str(), "c1.dat");
  sprintf(c2, "%s%s", hrg_eos_path.c_str(), "c2.dat");
  sprintf(c3, "%s%s", hrg_eos_path.c_str(), "c3.dat");
  sprintf(c4, "%s%s", hrg_eos_path.c_str(), "c4.dat");

  sprintf(F, "%s%s", hrg_eos_path.c_str(), "F.dat");
  sprintf(G, "%s%s", hrg_eos_path.c_str(), "G.dat");
  sprintf(betabulk, "%s%s", hrg_eos_path.c_str(), "betabulk.dat");
  sprintf(betaV, "%s%s", hrg_eos_path.c_str(), "betaV.dat");
  sprintf(betapi, "%s%s", hrg_eos_path.c_str(), "betapi.dat");


  // coefficient files and names
  FILE * c0_file = fopen(c0, "r"); // 14 moment (coefficients are scaled by some power of temperature)
  FILE * c1_file = fopen(c1, "r");
  FILE * c2_file = fopen(c2, "r");
  FILE * c3_file = fopen(c3, "r");
  FILE * c4_file = fopen(c4, "r");

  FILE * F_file = fopen(F, "r"); // Chapman Enskog (coefficients are scaled by some power of temperature)
  FILE * G_file = fopen(G, "r");
  FILE * betabulk_file = fopen(betabulk, "r");
  FILE * betaV_file = fopen(betaV, "r");
  FILE * betapi_file = fopen(betapi, "r");

  if(c0_file == NULL) printf("Couldn't open c0 coefficient file!\n");
  if(c1_file == NULL) printf("Couldn't open c1 coefficient file!\n");
  if(c2_file == NULL) printf("Couldn't open c2 coefficient file!\n");
  if(c3_file == NULL) printf("Couldn't open c3 coefficient file!\n");
  if(c4_file == NULL) printf("Couldn't open c4 coefficient file!\n");

  if(F_file == NULL) printf("Couldn't open F coefficient file!\n");
  if(G_file == NULL) printf("Couldn't open G coefficient file!\n");
  if(betabulk_file == NULL) printf("Couldn't open betabulk coefficient file!\n");
  if(betaV_file == NULL) printf("Couldn't open betaV coefficient file!\n");
  if(betapi_file == NULL) printf("Couldn't open betapi coefficient file!\n");

  // read 1st line (T dimension) and 2nd line (muB dimension)
  // (c0, ..., c4) should have same (T,muB) dimensions
  fscanf(c0_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(c1_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(c2_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(c3_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(c4_file, "%d\n%d\n", &points_T, &points_muB);

  fscanf(F_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(G_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(betabulk_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(betaV_file, "%d\n%d\n", &points_T, &points_muB);
  fscanf(betapi_file, "%d\n%d\n", &points_T, &points_muB);

  if(!include_baryon) points_muB = 1;

  // skip the header
  char header[300];
  fgets(header, 100, c0_file);
  fgets(header, 100, c1_file);
  fgets(header, 100, c2_file);
  fgets(header, 100, c3_file);
  fgets(header, 100, c4_file);

  fgets(header, 100, F_file);
  fgets(header, 100, G_file);
  fgets(header, 100, betabulk_file);
  fgets(header, 100, betaV_file);
  fgets(header, 100, betapi_file);

  // T and muB arrays
  T_array = (double *)calloc(points_T, sizeof(double));
  muB_array = (double *)calloc(points_muB, sizeof(double));

  // coefficient data
  c0_data = (double **)calloc(points_muB, sizeof(double));
  c1_data = (double **)calloc(points_muB, sizeof(double));
  c2_data = (double **)calloc(points_muB, sizeof(double));
  c3_data = (double **)calloc(points_muB, sizeof(double));
  c4_data = (double **)calloc(points_muB, sizeof(double));

  F_data = (double **)calloc(points_muB, sizeof(double));
  G_data = (double **)calloc(points_muB, sizeof(double));
  betabulk_data = (double **)calloc(points_muB, sizeof(double));
  betaV_data = (double **)calloc(points_muB, sizeof(double));
  betapi_data = (double **)calloc(points_muB, sizeof(double));

  // scan coefficient files
  for(int iB = 0; iB < points_muB; iB++)  // muB
  {
    c0_data[iB] = (double *)calloc(points_T, sizeof(double));
    c1_data[iB] = (double *)calloc(points_T, sizeof(double));
    c2_data[iB] = (double *)calloc(points_T, sizeof(double));
    c3_data[iB] = (double *)calloc(points_T, sizeof(double));
    c4_data[iB] = (double *)calloc(points_T, sizeof(double));

    F_data[iB] = (double *)calloc(points_T, sizeof(double));
    G_data[iB] = (double *)calloc(points_T, sizeof(double));
    betabulk_data[iB] = (double *)calloc(points_T, sizeof(double));
    betaV_data[iB] = (double *)calloc(points_T, sizeof(double));
    betapi_data[iB] = (double *)calloc(points_T, sizeof(double));

    for(int iT = 0; iT < points_T; iT++)  // T
    {
      // set T and muB (fm^-1) arrays from file
      fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c0_data[iB][iT]);
      fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c1_data[iB][iT]);
      fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c2_data[iB][iT]);
      fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c3_data[iB][iT]);
      fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c4_data[iB][iT]);

      fscanf(F_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &F_data[iB][iT]);
      fscanf(G_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &G_data[iB][iT]);
      fscanf(betabulk_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &betabulk_data[iB][iT]);
      fscanf(betaV_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &betaV_data[iB][iT]);
      fscanf(betapi_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &betapi_data[iB][iT]);
    } // iT
  } // iB

  T_min = T_array[0];
  muB_min = muB_array[0];

  // assume uniform grid
  dT = fabs(T_array[1] - T_array[0]);
  dmuB = fabs(muB_array[1] - muB_array[0]);

  fclose(c0_file);
  fclose(c1_file);
  fclose(c2_file);
  fclose(c3_file);
  fclose(c4_file);

  fclose(F_file);
  fclose(G_file);
  fclose(betabulk_file);
  fclose(betaV_file);
  fclose(betapi_file);

  printf("done\n");
}


void Deltaf_Data::compute_jonah_coefficients(particle_info * particle_data, int Nparticle)
{
  // allocate memory for the arrays
  lambda_squared_array = (double *)calloc(jonah_points, sizeof(double));
  z_array = (double *)calloc(jonah_points, sizeof(double));
  bulkPi_over_Peq_array = (double *)calloc(jonah_points, sizeof(double));

  bulkPi_over_Peq_max = -1.0;      // default to lowest value

  // get the average temperature, energy density, pressure
  Plasma QGP;
  QGP.load_thermodynamic_averages();

  const double T = QGP.temperature;    // GeV (assumes freezeout surface of constant temperature)

  // gauss laguerre roots and weights
  Gauss_Laguerre gla;
  gla.load_roots_and_weights("tables/gla_roots_weights_32_points.txt");

  const int pbar_pts = gla.points;

  double * pbar_root1 = gla.root[1];
  double * pbar_root2 = gla.root[2];

  double * pbar_weight1 = gla.weight[1];
  double * pbar_weight2 = gla.weight[2];

  // calculate the interpolation points of z(bulkPi/P), lambda(bulkPi/P)
  for(int i = 0; i < jonah_points; i++)
  {
    double lambda = lambda_min + (double)i * delta_lambda;

    double E = 0.0;                       // energy density (computed with kinetic theory)
    double P = 0.0;                       // pressure
    double E_mod = 0.0;                   // modified energy density
    double P_mod = 0.0;                   // modified pressure

    // calculate modified energy density (sum over hadron resonance contributions)
    for(int n = 0; n < Nparticle; n++)
    {
      double degeneracy = (double)particle_data[n].gspin;
      double mass = particle_data[n].mass;
      double sign = (double)particle_data[n].sign;

      double mbar = mass / T;

      if(mass == 0.0) continue;   // I skip the photon (Gamma) because the calculation breaks down for lambda = -1.0

      // ignore common prefactor = pow(T,4) / two_pi2_hbarC3 (since they will cancel out)
      E += degeneracy * Gauss1D_mod(E_mod_int, pbar_root2, pbar_weight2, pbar_pts, mbar, 0.0, sign);
      P += (1.0 / 3.0) * degeneracy * Gauss1D_mod(P_mod_int, pbar_root2, pbar_weight2, pbar_pts, mbar, 0.0, sign);

      E_mod += degeneracy * Gauss1D_mod(E_mod_int, pbar_root2, pbar_weight2, pbar_pts, mbar, lambda, sign);
      P_mod += (1.0 / 3.0) * degeneracy * Gauss1D_mod(P_mod_int, pbar_root2, pbar_weight2, pbar_pts, mbar, lambda, sign);
    }

    // jonah's formula (ignoring detLambda factor i.e. n_mod / n -> 1)
    double z = E / E_mod;
    double bulkPi_over_Peq = (P_mod / P) * z  -  1.0;

    // set the arrays and update the max bulk pressure
    lambda_squared_array[i] = lambda * lambda;
    z_array[i] = z;
    bulkPi_over_Peq_array[i] = bulkPi_over_Peq;
    bulkPi_over_Peq_max = max(bulkPi_over_Peq_max, bulkPi_over_Peq);

    //cout << lambda_squared_array[i] << "\t" << z_array[i] << "\t" << bulkPi_over_Peq_array[i] << endl;
  }

}

double Deltaf_Data::calculate_linear_temperature(double ** f_data, double T, double TL, double TR, int iTL, int iTR)
{
  // linear interpolation formula f(T)
  //  f_L    f_R

  double f_L = f_data[0][iTL];
  double f_R = f_data[0][iTR];

  return (f_L * (TR - T)  +  f_R * (T - TL)) / dT;
}

double Deltaf_Data::calculate_linear_bulkPi(double *f_data, double bulkPi, double bulkL, double bulkR, int ibulkL, int ibulkR, double dbulk)
{
  double f_L = f_data[ibulkL];
  double f_R = f_data[ibulkR];

  return (f_L * (bulkR - bulkPi)  +  f_R * (bulkPi - bulkL)) / dbulk;
}

deltaf_coefficients Deltaf_Data::linear_interpolation(double T, double E, double P, double bulkPi)
{
  double T4 = T * T * T * T;

  // left and right T, muB indices
  int iTL = (int)floor((T - T_min) / dT);
  int iTR = iTL + 1;

  double TL, TR;

  if(!(iTL >= 0 && iTR < points_T))
  {
    printf("Error: temperature is outside df coefficient table. Exiting...\n");
    exit(-1);
  }
  else
  {
    TL = T_array[iTL];
    TR = T_array[iTR];
  }

  deltaf_coefficients df;


  switch(df_mode)
  {
    case 1: // 14 moment
    {
      // undo the temperature power scaling of coefficients
      df.c0 = calculate_linear_temperature(c0_data, T, TL, TR, iTL, iTR) / T4;
      df.c1 = 0.0;
      df.c2 = calculate_linear_temperature(c2_data, T, TL, TR, iTL, iTR) / T4;
      df.c3 = 0.0;
      df.c4 = 0.0;
      df.shear14_coeff = 2.0 * T * T * (E + P);

      break;
    }
    case 2: // Chapman Enskog
    case 3: // Modified (Mike)
    {
      // undo the temperature power scaling of coefficients
      df.F = calculate_linear_temperature(F_data, T, TL, TR, iTL, iTR) * T;
      df.G = 0.0;
      df.betabulk = calculate_linear_temperature(betabulk_data, T, TL, TR, iTL, iTR) * T4;
      df.betaV = 1.0;
      df.betapi = calculate_linear_temperature(betapi_data, T, TL, TR, iTL, iTR) * T4;

      break;
    }
    case 4: // Modified (Jonah)
    {
      // first regulate the bulk pressure if out of bounds
      if(bulkPi < - P)
      { 
        bulkPi = - (1.0 - 1.e-5) * P;
      }
      else if(bulkPi / P > bulkPi_over_Peq_max) 
      {
        bulkPi = P * (bulkPi_over_Peq_max - 1.e-5);
      }
      
      int ibulkL;
      int ibulkR;

      double bulkL;
      double bulkR;
      double dbulk; 

      double bulkPi_over_Peq = bulkPi / P;

      // search interpolation points by hand
      // since bulkPi_over_Peq_array is not uniform
      bool found_interpolation_points = false;
      for(int i = 0; i < jonah_points; i++)
      {
        if(bulkPi_over_Peq < bulkPi_over_Peq_array[i])
        {
          ibulkL = i - 1;
          ibulkR = i;

          bulkL = bulkPi_over_Peq_array[ibulkL];
          bulkR = bulkPi_over_Peq_array[ibulkR];
          dbulk = fabs(bulkR - bulkL);
          found_interpolation_points = true;

          // cout << ibulkL << "\t" << ibulkR << endl;
          // cout << bulkL << "\t" << bulkR << endl;
          // cout << dbulk << endl;
          break;
        }
      }
      if(!found_interpolation_points)
      {
        printf("Jonah interpolation error: couldn't find interpolation points\n");
      }

      double lambda_squared = calculate_linear_bulkPi(lambda_squared_array, bulkPi_over_Peq, bulkL, bulkR, ibulkL, ibulkR, dbulk);
    
      if(bulkPi < 0.0)
      {
        df.lambda = - sqrt(lambda_squared);
      }
      else if(bulkPi > 0.0)
      {
        df.lambda = sqrt(lambda_squared);
      }
      df.z = calculate_linear_bulkPi(z_array, bulkPi_over_Peq, bulkL, bulkR, ibulkL, ibulkR, dbulk);
      df.betapi = calculate_linear_temperature(betapi_data, T, TL, TR, iTL, iTR) * T4;

      // // linearized correction to lambda, z
      df.delta_lambda = bulkPi / (5.0 * df.betapi -  3.0 * P * (E + P) / E);
      df.delta_z = - 3.0 * df.delta_lambda * P / E;

      break;
    }
    default:
    {
      printf("Error: choose df_mode = (1,2,3,4)\n");
      exit(-1);
    }
  }

  return df;
}



double Deltaf_Data::calculate_bilinear(double ** f_data, double T, double muB, double TL, double TR, double muBL, double muBR, int iTL, int iTR, int imuBL, int imuBR)
{
  // bilinear formula f(T,muB)
  // T = x-axis (cols)
  // muB = y-axis (rows)
  //  f_LR    f_RR
  //
  //  f_LL    f_RL

  double f_LL = f_data[imuBL][iTL];
  double f_LR = f_data[imuBR][iTL];
  double f_RL = f_data[imuBL][iTR];
  double f_RR = f_data[imuBR][iTR];



  return ((f_LL*(TR - T) + f_RL*(T - TL)) * (muBR - muB)  +  (f_LR*(TR - T) + f_RR*(T - TL)) * (muB - muBL)) / (dT * dmuB);
}

deltaf_coefficients Deltaf_Data::bilinear_interpolation(double T, double muB, double E, double P, double bulkPi)
{
  // left and right T, muB indices
  int iTL = (int)floor((T - T_min) / dT);
  int iTR = iTL + 1;

  int imuBL = (int)floor((muB - muB_min) / dmuB);
  int imuBR = imuBL + 1;

  double TL, TR, muBL, muBR;

  if(!(iTL >= 0 && iTR < points_T) || !(imuBL >= 0 && imuBR < points_muB))
  {
    printf("Error: (T,muB) outside df coefficient table. Exiting...\n");
    exit(-1);
  }
  else
  {
    TL = T_array[iTL];
    TR = T_array[iTR];
    muBL = muB_array[imuBL];
    muBR = muB_array[imuBR];
  }

  deltaf_coefficients df;

  switch(df_mode)
  {
    case 1:
    {
      double T3 = T * T * T;
      double T4 = T3 * T;
      double T5 = T4 * T;

      // bilinear interpolated values & undo temperature power scaling
      df.c0 = calculate_bilinear(c0_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) / T4;
      df.c1 = calculate_bilinear(c1_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) / T3;
      df.c2 = calculate_bilinear(c2_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) / T4;
      df.c3 = calculate_bilinear(c3_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) / T4;
      df.c4 = calculate_bilinear(c4_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) / T5;
      df.shear14_coeff = 2.0 * T * T * (E + P);

      break;
    }
    case 2:
    case 3:
    {
      double T3 = T * T * T;
      double T4 = T3 * T;

      // bilinear interpolated values & undo temperature power scaling
      df.F = calculate_bilinear(F_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) * T;
      df.G = calculate_bilinear(G_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR);
      df.betabulk = calculate_bilinear(betabulk_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) * T4;
      df.betaV = calculate_bilinear(betaV_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) * T3;
      df.betapi = calculate_bilinear(betapi_data, T, muB, TL, TR, muBL, muBR, iTL, iTR, imuBL, imuBR) * T4;

      break;
    }
    case 4:
    {
      printf("Bilinear interpolation error: Jonah df doesn't work for nonzero muB. Exiting..\n");
      exit(-1);
    }
    default:
    {
      printf("Bilinear interpolation error: choose df_mode = (1,2,3). Exiting..\n");
      exit(-1);
    }
  }

  return df;
}

deltaf_coefficients Deltaf_Data::evaluate_df_coefficients(double T, double muB, double E, double P, double bulkPi)
{
  // evaluate the df coefficients by interpolating the data

  deltaf_coefficients df;

  if(!include_baryon)
  {
    df = linear_interpolation(T, E, P, bulkPi);         // linear interpolation wrt T (at muB = 0)
  }
  else
  {
    // muB on freezeout surface should be nonzero in general
    // otherwise should set include_baryon = 0
    df = bilinear_interpolation(T, muB, E, P, bulkPi);  // bilinear wrt (T, muB)
  }

  return df;
}


void Deltaf_Data::test_df_coefficients(double bulkPi_over_P)
{
  // test print the output of the df coefficients at average temperature, etc (and a fixed value of bulkPi)

  Plasma QGP;
  QGP.load_thermodynamic_averages();

  double E = QGP.energy_density;
  double T = QGP.temperature;
  double P = QGP.pressure;
  double muB = QGP.baryon_chemical_potential;
  double bulkPi = bulkPi_over_P * P;

  deltaf_coefficients df = evaluate_df_coefficients(T, muB, E, P, bulkPi);

  if(df_mode == 1)
  {
    printf("\n(c0, c1, c2, c3, c4, shear14) = (%lf, %lf, %lf, %lf, %lf, %lf)\n\n", df.c0, df.c1, df.c2, df.c3, df.c4, df.shear14_coeff);
  }
  else if(df_mode == 2 || df_mode == 3)
  {
    printf("\n(F, G, betabulk, betaV, betapi) = (%lf, %lf, %lf, %lf, %lf)\n\n", df.F, df.G, df.betabulk, df.betaV, df.betapi);
  }
  else if(df_mode == 4)
  {
    printf("\n(lambda, z, dlambda, dz, betapi) = (%lf, %lf, %lf, %lf, %lf)\n\n", df.lambda, df.z, df.delta_lambda, df.delta_z, df.betapi);
  }
}










