
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include "iS3D.h"
#include "deltafReader.h"
#include "ParameterReader.h"
#include "readindata.h"

using namespace std;

/*

Deltaf_Reader::Deltaf_Reader(ParameterReader * paraRdr_in)
{
  paraRdr = paraRdr_in;

  mode = paraRdr->getVal("mode");
  df_mode = paraRdr->getVal("df_mode");
  include_baryon = paraRdr->getVal("include_baryon");
}

Deltaf_Reader::~Deltaf_Reader()
{

}

deltaf_coefficients Deltaf_Reader::load_coefficients(FO_surf *surface, long FO_length)
{
  deltaf_coefficients df_data;

  Plasma QGP;
  QGP.load_thermodynamic_averages();

  double T = QGP.temperature;
  double E = QGP.energy_density;
  double P = QGP.pressure;
  double muB = QGP.baryon_chemical_potential;

  df_data.shear14_coeff = 2.0 * T * T * (E + P);

  double T_FO = T / hbarC;
  double muB_FO = muB / hbarC;    // the file units are in fm

  printf("Reading in ");  // or I could compute these guys directly

  if(df_mode == 1)
  {
    printf("14-moment coefficients vhydro...\n");
    // coefficient files and names
    FILE * c0_file = fopen("deltaf_coefficients/vh/old/c0_df14_vh.dat", "r");
    FILE * c1_file = fopen("deltaf_coefficients/vh/old/c1_df14_vh.dat", "r");
    FILE * c2_file = fopen("deltaf_coefficients/vh/old/c2_df14_vh.dat", "r");
    FILE * c3_file = fopen("deltaf_coefficients/vh/old/c3_df14_vh.dat", "r");
    FILE * c4_file = fopen("deltaf_coefficients/vh/old/c4_df14_vh.dat", "r");

    if(c0_file == NULL) printf("Couldn't open c0 coefficient file!\n");
    if(c1_file == NULL) printf("Couldn't open c1 coefficient file!\n");
    if(c2_file == NULL) printf("Couldn't open c2 coefficient file!\n");
    if(c3_file == NULL) printf("Couldn't open c3 coefficient file!\n");
    if(c4_file == NULL) printf("Couldn't open c4 coefficient file!\n");

    int nT = 0;
    int nB = 0;

    // read 1st line (T dimension) and 2nd line (muB dimension)
    fscanf(c0_file, "%d\n%d\n", &nT, &nB);
    fscanf(c1_file, "%d\n%d\n", &nT, &nB);
    fscanf(c2_file, "%d\n%d\n", &nT, &nB);
    fscanf(c3_file, "%d\n%d\n", &nT, &nB);
    fscanf(c4_file, "%d\n%d\n", &nT, &nB);

    if(!include_baryon) nB = 1;

    // skip the header
    char header[300];
    fgets(header, 100, c0_file);
    fgets(header, 100, c1_file);
    fgets(header, 100, c2_file);
    fgets(header, 100, c3_file);
    fgets(header, 100, c4_file);

    // T and muB arrays
    double T_array[nT];
    double muB_array[nB];

    // coefficient tables
    double c0[nT][nB];
    double c1[nT][nB];
    double c2[nT][nB];
    double c3[nT][nB];
    double c4[nT][nB];

    // scan coefficient files
    for(int iB = 0; iB < nB; iB++)  // muB
    {
      bool found = false;   // found interpolation

      for(int iT = 0; iT < nT; iT++)    // T
      {
        // set T and muB (fm^-1) arrays from file
        fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c0[iT][iB]);
        fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c1[iT][iB]);
        fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c2[iT][iB]);
        fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c3[iT][iB]);
        fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &c4[iT][iB]);

        // check if cross freezeout temperature (T_array increasing)
        if(iT > 0 && T_FO < T_array[iT])
        {
          // linear interpolate w.r.t. temperature only
          double T1 = T_array[iT-1];
          double T2 = T_array[iT];

          df_data.c0 = (c0[iT-1][iB] * (T2 - T_FO)  +  c0[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.c1 = (c1[iT-1][iB] * (T2 - T_FO)  +  c1[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.c2 = (c2[iT-1][iB] * (T2 - T_FO)  +  c2[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.c3 = (c3[iT-1][iB] * (T2 - T_FO)  +  c3[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.c4 = (c4[iT-1][iB] * (T2 - T_FO)  +  c4[iT][iB] * (T_FO - T1)) / (T2 - T1);

          // convert 14-momentum coefficients to real-life units
          df_data.c0 /= (hbarC * hbarC * hbarC);
          df_data.c1 /= (hbarC * hbarC);
          df_data.c2 /= (hbarC * hbarC * hbarC);
          df_data.c3 /= hbarC;
          df_data.c4 /= (hbarC * hbarC);

          found = true;
          break;
        }
      } // iT
      if(found) break;
    } // iB

    fclose(c0_file);
    fclose(c1_file);
    fclose(c2_file);
    fclose(c3_file);
    fclose(c4_file);

  } //if(df_mode == 1)
  else if(df_mode == 2 || df_mode == 3 || df_mode == 4)
  {
    printf("Chapman-Enskog coefficients vhydro...\n");

    // coefficient files and names
    FILE * F_file = fopen("deltaf_coefficients/vh/old/F_dfce_vh.dat", "r");
    FILE * G_file = fopen("deltaf_coefficients/vh/old/G_dfce_vh.dat", "r");
    FILE * betabulk_file = fopen("deltaf_coefficients/vh/old/betabulk_dfce_vh.dat", "r");
    FILE * betaV_file = fopen("deltaf_coefficients/vh/old/betaV_dfce_vh.dat", "r");
    FILE * betapi_file = fopen("deltaf_coefficients/vh/old/betapi_dfce_vh.dat", "r");

    if(F_file == NULL) printf("Couldn't open F coefficient file!\n");
    if(G_file == NULL) printf("Couldn't open G coefficient file!\n");
    if(betabulk_file == NULL) printf("Couldn't open betabulk coefficient file!\n");
    if(betaV_file == NULL) printf("Couldn't open betaV coefficient file!\n");
    if(betapi_file == NULL) printf("Couldn't open betapi coefficient file!\n");

    int nT = 0;
    int nB = 0;

    // read 1st line (T dimension) and 2nd line (muB dimension)
    fscanf(F_file, "%d\n%d\n", &nT, &nB);
    fscanf(G_file, "%d\n%d\n", &nT, &nB);
    fscanf(betabulk_file, "%d\n%d\n", &nT, &nB);
    fscanf(betaV_file, "%d\n%d\n", &nT, &nB);
    fscanf(betapi_file, "%d\n%d\n", &nT, &nB);

    if(!include_baryon) nB = 1;

    // skip the headers
    char header[300];

    fgets(header, 100, F_file);
    fgets(header, 100, G_file);
    fgets(header, 100, betabulk_file);
    fgets(header, 100, betaV_file);
    fgets(header, 100, betapi_file);

    // T and muB arrays
    double T_array[nT];
    double muB_array[nB];

    // coefficient tables
    double F[nT][nB];
    double G[nT][nB];
    double betabulk[nT][nB];
    double betaV[nT][nB];
    double betapi[nT][nB];

    // scan coefficient files
    for(int iB = 0; iB < nB; iB++)  // muB
    {
      bool found = false;

      for(int iT = 0; iT < nT; iT++)    // T
      {
        // set T and muB (fm^-1) arrays from file
        fscanf(F_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &F[iT][iB]);
        fscanf(G_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &G[iT][iB]);
        fscanf(betabulk_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &betabulk[iT][iB]);
        fscanf(betaV_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &betaV[iT][iB]);
        fscanf(betapi_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT], &muB_array[iB], &betapi[iT][iB]);

        // check if cross freezeout temperature (T_array increasing)
        if(iT > 0 && T_FO < T_array[iT])
        {
          // linear interpolate w.r.t. temperature only
          double T1 = T_array[iT-1];
          double T2 = T_array[iT];

          df_data.F = (F[iT-1][iB] * (T2 - T_FO)  +  F[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.G = (G[iT-1][iB] * (T2 - T_FO)  +  G[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.betabulk = (betabulk[iT-1][iB] * (T2 - T_FO)  +  betabulk[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.betaV = (betaV[iT-1][iB] * (T2 - T_FO)  +  betaV[iT][iB] * (T_FO - T1)) / (T2 - T1);
          df_data.betapi = (betapi[iT-1][iB] * (T2 - T_FO)  +  betapi[iT][iB] * (T_FO - T1)) / (T2 - T1);

          // convert Chapman-Enskog coefficients to real-life units
          df_data.F *= hbarC;
          df_data.betabulk *= hbarC;
          df_data.betapi *= hbarC;

          found = true;
          break;
        }
      } //iT
      if(found) break;
    } //iB

    fclose(F_file);
    fclose(G_file);
    fclose(betabulk_file);
    fclose(betaV_file);
    fclose(betapi_file);

  } // else if(df_mode == 2 || df_mode == 3 || df_mode == 4)
  else if(df_mode == 5)
  {
    printf("14-moment coefficients vahydro PL...\n");

    // coefficient files and names
    FILE * c0_file = fopen("deltaf_coefficients/vah/c0_vah1.dat", "r");
    FILE * c1_file = fopen("deltaf_coefficients/vah/c1_vah1.dat", "r");
    FILE * c2_file = fopen("deltaf_coefficients/vah/c2_vah1.dat", "r");
    FILE * c3_file = fopen("deltaf_coefficients/vah/c3_vah1.dat", "r");
    FILE * c4_file = fopen("deltaf_coefficients/vah/c4_vah1.dat", "r");

    if(c0_file == NULL) printf("Couldn't open c0 coefficient file!\n");
    if(c1_file == NULL) printf("Couldn't open c1 coefficient file!\n");
    if(c2_file == NULL) printf("Couldn't open c2 coefficient file!\n");
    if(c3_file == NULL) printf("Couldn't open c3 coefficient file!\n");
    if(c4_file == NULL) printf("Couldn't open c4 coefficient file!\n");

    int nL = 0;
    int naL = 0;

    // read 1st line (L dimension) and 2nd line (aL dimension)
    fscanf(c0_file, "%d\n%d\n", &nL, &naL);
    fscanf(c1_file, "%d\n%d\n", &nL, &naL);
    fscanf(c2_file, "%d\n%d\n", &nL, &naL);
    fscanf(c3_file, "%d\n%d\n", &nL, &naL);
    fscanf(c4_file, "%d\n%d\n", &nL, &naL);

    //cout << nL << "\t" << naL << endl;

    //skip the header with labels and units
    char header[300];
    fgets(header, 100, c0_file);
    fgets(header, 100, c1_file);
    fgets(header, 100, c2_file);
    fgets(header, 100, c3_file);
    fgets(header, 100, c4_file);

    // L and aL arrays
    double L_array[nL];
    double aL_array[naL];

    int n1 = nL;
    int n2 = naL;

    double c0[n1][n2];
    double c1[n1][n2];
    double c2[n1][n2];
    double c3[n1][n2];
    double c4[n1][n2];

    double hbarC3 = (hbarC * hbarC * hbarC);

    for (long icell = 0; icell < FO_length; icell++)
    {
      double aL = surface[icell].aL;
      double Lambda = surface[icell].Lambda / hbarC;

      //set the values of delta-f coefficients for every FO cell

      for (int i2 = 0; i2 < n2; i2++)   // aL
      {
        bool found = false;     // found bilinear interpolation

        for (int i1 = 0; i1 < n1; i1++)  // Lambda
        {
          fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1], &aL_array[i2], &c0[i1][i2]);
          fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1], &aL_array[i2], &c1[i1][i2]);
          fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1], &aL_array[i2], &c2[i1][i2]);
          fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1], &aL_array[i2], &c3[i1][i2]);
          fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1], &aL_array[i2], &c4[i1][i2]);

          if( (i1 > 0) && (Lambda < L_array[i1])  && (i2 > 0) && (aL < aL_array[i2]) )
          {
            // bilinear-interpolate w.r.t. Lambda and alpha_L

            double Lambda1 = L_array[i1-1];
            double Lambda2 = L_array[i1];
            double aL1 = aL_array[i2-1];
            double aL2 = aL_array[i2];

            surface[icell].c0 = ((c0[i1-1][i2-1] * (Lambda2 - Lambda)  +  c0[i1][i2-1] * (Lambda - Lambda1)) * (aL2 - aL)
                              + (c0[i1-1][i2] * (Lambda2 - Lambda)  +  c0[i1][i2] * (Lambda - Lambda1)) * (aL - aL1)) / ((aL2 - aL1) * (Lambda2 - Lambda1));

            surface[icell].c1 = ((c1[i1-1][i2-1] * (Lambda2 - Lambda)  +  c1[i1][i2-1] * (Lambda - Lambda1)) * (aL2 - aL)
                              + (c1[i1-1][i2] * (Lambda2 - Lambda)  +  c1[i1][i2] * (Lambda - Lambda1)) * (aL - aL1)) / ((aL2 - aL1) * (Lambda2 - Lambda1));

            surface[icell].c2 = ((c2[i1-1][i2-1] * (Lambda2 - Lambda)  +  c2[i1][i2-1] * (Lambda - Lambda1)) * (aL2 - aL)
                              + (c2[i1-1][i2] * (Lambda2 - Lambda)  +  c2[i1][i2] * (Lambda - Lambda1)) * (aL - aL1)) / ((aL2 - aL1) * (Lambda2 - Lambda1));

            surface[icell].c3 = ((c3[i1-1][i2-1] * (Lambda2 - Lambda)  +  c3[i1][i2-1] * (Lambda - Lambda1)) * (aL2 - aL)
                              + (c3[i1-1][i2] * (Lambda2 - Lambda)  +  c3[i1][i2] * (Lambda - Lambda1)) * (aL - aL1)) / ((aL2 - aL1) * (Lambda2 - Lambda1));

            surface[icell].c4 = ((c4[i1-1][i2-1] * (Lambda2 - Lambda)  +  c4[i1][i2-1] * (Lambda - Lambda1)) * (aL2 - aL)
                              + (c4[i1-1][i2] * (Lambda2 - Lambda)  +  c4[i1][i2] * (Lambda - Lambda1)) * (aL - aL1)) / ((aL2 - aL1) * (Lambda2 - Lambda1));

            // convert to real life units

            surface[icell].c0 /= hbarC3;
            surface[icell].c1 /= hbarC3;
            surface[icell].c2 /= hbarC3;
            surface[icell].c3 /= hbarC3;
            surface[icell].c4 /= hbarC3;

            found = true;  // found
            break;
          }
        } //i1
        if(found) break;
      } //i2
    } // icell

    // close files
    fclose(c0_file);
    fclose(c1_file);
    fclose(c2_file);
    fclose(c3_file);
    fclose(c4_file);
  }
  return df_data;
}
*/

Deltaf_Data::Deltaf_Data(ParameterReader * paraRdr_in)
{
  paraRdr = paraRdr_in;

  hrg_eos = paraRdr->getVal("hrg_eos");
  mode = paraRdr->getVal("mode");
  df_mode = paraRdr->getVal("df_mode");
  include_baryon = paraRdr->getVal("include_baryon");

  if(hrg_eos == 1) 
  {
    hrg_eos_path = urqmd;
  }
  else if(hrg_eos == 2)
  {
    hrg_eos_path = smash;
  }
  else if(hrg_eos == 3)
  {
    hrg_eos_path = smash_box;
  }
  else
  {
    printf("Error: please choose hrg_eos = (1,2,3)\n");
    exit(-1);
  }
}

Deltaf_Data::~Deltaf_Data()
{
  // is there any harm in deallocating memory, while it's being used?
  gsl_spline_free(c0_spline);
  gsl_spline_free(c2_spline);
  gsl_spline_free(c3_spline);

  gsl_spline_free(F_spline);
  gsl_spline_free(betabulk_spline);
  gsl_spline_free(betaV_spline);
  gsl_spline_free(betapi_spline);

  gsl_spline_free(lambda_squared_spline);
  gsl_spline_free(z_spline);
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

  // uniform grid
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

  // now construct cubic splines for lambda(bulkPi/Peq) and z(bulkPi/Peq)
  lambda_squared_spline = gsl_spline_alloc(gsl_interp_cspline, jonah_points);
  z_spline = gsl_spline_alloc(gsl_interp_cspline, jonah_points);

  gsl_spline_init(lambda_squared_spline, bulkPi_over_Peq_array, lambda_squared_array, jonah_points);
  gsl_spline_init(z_spline, bulkPi_over_Peq_array, z_array, jonah_points);
}

void Deltaf_Data::compute_jonah_coefficients(double *Degeneracy, double *Mass, double *Sign, int number_of_chosen_particles)
{
  // allocate memory for the arrays
  lambda_squared_array = (double *)calloc(jonah_points, sizeof(double));
  z_array = (double *)calloc(jonah_points, sizeof(double));
  bulkPi_over_Peq_array = (double *)calloc(jonah_points, sizeof(double));

  bulkPi_over_Peq_max = -1.0;      // default to lowest value

  // get the average temperature, energy density, pressure
  Plasma QGP;
  QGP.load_thermodynamic_averages();

  const double T = QGP.temperature;    // GeV (assume freezeout surface of constant temperature)

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
    for(int n = 0; n < number_of_chosen_particles; n++)
    {
      double degeneracy = Degeneracy[n];
      double mass = Mass[n];
      double sign = Sign[n];

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

  // now construct cubic splines for lambda(bulkPi/Peq) and z(bulkPi/Peq)
  lambda_squared_spline = gsl_spline_alloc(gsl_interp_cspline, jonah_points);
  z_spline = gsl_spline_alloc(gsl_interp_cspline, jonah_points);

  gsl_spline_init(lambda_squared_spline, bulkPi_over_Peq_array, lambda_squared_array, jonah_points);
  gsl_spline_init(z_spline, bulkPi_over_Peq_array, z_array, jonah_points);
}

void Deltaf_Data::construct_cubic_splines()
{
  // Allocate memory for cubic splines
  c0_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);
  c2_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);
  c3_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);

  F_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);
  betabulk_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);
  betapi_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);
  betaV_spline = gsl_spline_alloc(gsl_interp_cspline, points_T);

  // Initialize the cubic splines
  gsl_spline_init(c0_spline, T_array, c0_data[0], points_T);
  gsl_spline_init(c2_spline, T_array, c2_data[0], points_T);
  gsl_spline_init(c3_spline, T_array, c3_data[0], points_T);

  gsl_spline_init(F_spline, T_array, F_data[0], points_T);
  gsl_spline_init(betabulk_spline, T_array, betabulk_data[0], points_T);
  gsl_spline_init(betaV_spline, T_array, betaV_data[0], points_T);
  gsl_spline_init(betapi_spline, T_array, betapi_data[0], points_T);

  // I'm not sure what to do with jonah's coefficients though...(maybe it makes sense to do linear interpolation?)
}


deltaf_coefficients Deltaf_Data::cubic_spline(double T, double E, double P, double bulkPi)
{
  deltaf_coefficients df;

  gsl_interp_accel * accel_T = gsl_interp_accel_alloc();    // for temperature dependent functions
  gsl_interp_accel * accel_bulk = gsl_interp_accel_alloc(); // for bulkPi/Peq dependent functions

  switch(df_mode)
  {
    case 1: // 14 moment
    {
      // undo the temperature power scaling of coefficients
      double T4 = T * T * T * T;

      df.c0 = gsl_spline_eval(c0_spline, T, accel_T) / T4;
      df.c1 = 0.0;
      df.c2 = gsl_spline_eval(c2_spline, T, accel_T) / T4;
      df.c3 = 0.0;
      df.c4 = 0.0;
      df.shear14_coeff = 2.0 * T * T * (E + P);

      break;
    }
    case 2: // Chapman Enskog
    case 3: // Modified (Mike)
    {
      // undo the temperature power scaling of coefficients
      double T4 = T * T * T * T;

      df.F = gsl_spline_eval(F_spline, T, accel_T) * T;
      df.G = 0.0;
      df.betabulk = gsl_spline_eval(betabulk_spline, T, accel_T) * T4; 
      df.betaV = 1.0;
      df.betapi = gsl_spline_eval(betapi_spline, T, accel_T) * T4;

      break;
    }
    case 4: // Modified (Jonah)
    {
      // undo the temperature power scaling of betapi
      double T4 = T * T * T * T;

      double lambda_squared = gsl_spline_eval(lambda_squared_spline, (bulkPi / P), accel_bulk);
      if(bulkPi < 0.0)
      {
        df.lambda = - sqrt(lambda_squared);
      }
      else if(bulkPi > 0.0)
      {
        df.lambda = sqrt(lambda_squared);
      }
      df.z = gsl_spline_eval(z_spline, (bulkPi / P), accel_bulk);
      df.betapi = gsl_spline_eval(betapi_spline, T, accel_T) * T4;
      // linearized correction to lambda, z
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

  gsl_interp_accel_free(accel_T);
  gsl_interp_accel_free(accel_bulk);

  return df;
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
    printf("Error: (T,muB) outside (T,muB)-grid. Exiting...\n");
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

      double c0_LL = c0_data[iTL][imuBL];
      double c0_LR = c0_data[iTL][imuBR];
      double c0_RL = c0_data[iTR][imuBL];
      double c0_RR = c0_data[iTR][imuBR];

      double c1_LL = c1_data[iTL][imuBL];
      double c1_LR = c1_data[iTL][imuBR];
      double c1_RL = c1_data[iTR][imuBL];
      double c1_RR = c1_data[iTR][imuBR];

      double c2_LL = c2_data[iTL][imuBL];
      double c2_LR = c2_data[iTL][imuBR];
      double c2_RL = c2_data[iTR][imuBL];
      double c2_RR = c2_data[iTR][imuBR];

      double c3_LL = c3_data[iTL][imuBL];
      double c3_LR = c3_data[iTL][imuBR];
      double c3_RL = c3_data[iTR][imuBL];
      double c3_RR = c3_data[iTR][imuBR];

      double c4_LL = c4_data[iTL][imuBL];
      double c4_LR = c4_data[iTL][imuBR];
      double c4_RL = c4_data[iTR][imuBL];
      double c4_RR = c4_data[iTR][imuBR];

      // bilinear interpolated values & undo temperature power scaling 
      df.c0 = ((c0_LL*(TR-T) + c0_RL*(T-TL)) * (muBR-muB)  +  (c0_LR*(TR-T) + c0_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) / T4;
      df.c1 = ((c1_LL*(TR-T) + c1_RL*(T-TL)) * (muBR-muB)  +  (c1_LR*(TR-T) + c1_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) / T3;
      df.c2 = ((c2_LL*(TR-T) + c2_RL*(T-TL)) * (muBR-muB)  +  (c2_LR*(TR-T) + c2_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) / T4;
      df.c3 = ((c3_LL*(TR-T) + c3_RL*(T-TL)) * (muBR-muB)  +  (c3_LR*(TR-T) + c3_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) / T4;
      df.c4 = ((c4_LL*(TR-T) + c4_RL*(T-TL)) * (muBR-muB)  +  (c4_LR*(TR-T) + c4_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) / T5;
      df.shear14_coeff = 2.0 * T * T * (E + P);

      break;
    }
    case 2:
    case 3:
    {
      double T3 = T * T * T;
      double T4 = T3 * T;

      double F_LL = F_data[iTL][imuBL];
      double F_LR = F_data[iTL][imuBR];
      double F_RL = F_data[iTR][imuBL];
      double F_RR = F_data[iTR][imuBR];

      double G_LL = G_data[iTL][imuBL];
      double G_LR = G_data[iTL][imuBR];
      double G_RL = G_data[iTR][imuBL];
      double G_RR = G_data[iTR][imuBR];

      double betabulk_LL = betabulk_data[iTL][imuBL];
      double betabulk_LR = betabulk_data[iTL][imuBR];
      double betabulk_RL = betabulk_data[iTR][imuBL];
      double betabulk_RR = betabulk_data[iTR][imuBR];

      double betaV_LL = betaV_data[iTL][imuBL];
      double betaV_LR = betaV_data[iTL][imuBR];
      double betaV_RL = betaV_data[iTR][imuBL];
      double betaV_RR = betaV_data[iTR][imuBR];

      double betapi_LL = betapi_data[iTL][imuBL];
      double betapi_LR = betapi_data[iTL][imuBR];
      double betapi_RL = betapi_data[iTR][imuBL];
      double betapi_RR = betapi_data[iTR][imuBR];

      // bilinear interpolated values & undo temperature power scaling 
      df.F = ((F_LL*(TR-T) + F_RL*(T-TL)) * (muBR-muB)  +  (F_LR*(TR-T) + F_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) * T;
      df.G = ((G_LL*(TR-T) + G_RL*(T-TL)) * (muBR-muB)  +  (G_LR*(TR-T) + G_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB);
      df.betabulk = ((betabulk_LL*(TR-T) + betabulk_RL*(T-TL)) * (muBR-muB)  +  (betabulk_LR*(TR-T) + betabulk_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) * T4;
      df.betaV = ((betaV_LL*(TR-T) + betaV_RL*(T-TL)) * (muBR-muB)  +  (betaV_LR*(TR-T) + betaV_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) * T3;
      df.betapi = ((betapi_LL*(TR-T) + betapi_RL*(T-TL)) * (muBR-muB)  +  (betapi_LR*(TR-T) + betapi_RR*(T-TL)) * (muB-muBL)) / (dT * dmuB) * T4;

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
    df = cubic_spline(T, E, P, bulkPi);   // cubic spline interpolation wrt T (at muB = 0)
  }
  else
  {
    // muB on freezeout surface should be nonzero in general
    // otherwise you best stick to setting include_baryon = 0
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

void Deltaf_Data::compute_particle_densities(particle_info * particle_data, int Nparticle)
{
  // get the average temperature, energy density, pressure
  Plasma QGP;
  QGP.load_thermodynamic_averages();

  const double T = QGP.temperature;    // GeV
  const double E = QGP.energy_density; // GeV / fm^3
  const double P = QGP.pressure;       // GeV / fm^3
  const double muB = QGP.baryon_chemical_potential;
  const double nB = QGP.net_baryon_density;

  deltaf_coefficients df = evaluate_df_coefficients(T, muB, E, P, 0.0);

  double alphaB = muB / T;
  double baryon_enthalpy_ratio = nB / (E + P);

  // gauss laguerre roots and weights
  Gauss_Laguerre gla;
  gla.load_roots_and_weights("tables/gla_roots_weights_32_points.txt");

  const int pbar_pts = gla.points;

  double * pbar_root1 = gla.root[1];
  double * pbar_root2 = gla.root[2];
  double * pbar_root3 = gla.root[3];

  double * pbar_weight1 = gla.weight[1];
  double * pbar_weight2 = gla.weight[2];
  double * pbar_weight3 = gla.weight[3];


  // calculate the equilibrium densities and the
  // bulk / diffusion corrections of each particle
  for(int i = 0; i < Nparticle; i++)
  {
    double mass = particle_data[i].mass;
    double degeneracy = (double)particle_data[i].gspin;
    double baryon = (double)particle_data[i].baryon;
    double sign = (double)particle_data[i].sign;
    double mbar = mass / T;

    // equilibrium density
    double neq_fact = degeneracy * pow(T,3) / two_pi2_hbarC3;
    double neq = neq_fact * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

    // bulk and diffusion density corrections
    double dn_bulk = 0.0;
    double dn_diff = 0.0;

    switch(df_mode)
    {
      case 1: // 14 moment (not sure what the status is)
      {
        double c0 = df.c0;
        double c1 = df.c1;
        double c2 = df.c2;
        double c3 = df.c3;
        double c4 = df.c4;

        double J10_fact = degeneracy * pow(T,3) / two_pi2_hbarC3;
        double J20_fact = degeneracy * pow(T,4) / two_pi2_hbarC3;
        double J30_fact = degeneracy * pow(T,5) / two_pi2_hbarC3;
        double J31_fact = degeneracy * pow(T,5) / two_pi2_hbarC3 / 3.0;

        double J10 =  J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
        double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
        double J30 = J30_fact * GaussThermal(J30_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);
        double J31 = J31_fact * GaussThermal(J31_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);

        dn_bulk = ((c0 - c2) * mass * mass * J10 +  c1 * baryon * J20  +  (4.0 * c2 - c0) * J30);
        // these coefficients need to be loaded.
        // c3 ~ cV / V
        // c4 ~ 2cW / V
        dn_diff = baryon * c3 * neq * T  +  c4 * J31;   // not sure if this is right...
        break;
      }
      case 2: // Chapman-Enskog
      case 3: // Modified (Mike)
      {
        double F = df.F;
        double G = df.G;
        double betabulk = df.betabulk;
        double betaV = df.betaV;

        double J10_fact = degeneracy * pow(T,3) / two_pi2_hbarC3;
        double J11_fact = degeneracy * pow(T,3) / two_pi2_hbarC3 / 3.0;
        double J20_fact = degeneracy * pow(T,4) / two_pi2_hbarC3;

        double J10 = J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
        double J11 = J11_fact * GaussThermal(J11_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
        double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

        dn_bulk = (neq + (baryon * J10 * G) + (J20 * F / pow(T,2))) / betabulk;
        dn_diff = (neq * T * baryon_enthalpy_ratio  -  baryon * J11) / betaV;

        break;
      }
      case 4:
      {
        // bulk/diffusion densities not needed for jonah
        break;
      }
      default:
      {
        cout << "Please choose df_mode = (1,2,3,4) in parameters.dat" << endl;
        exit(-1);
      }
    } // df_mode

    particle_data[i].equilibrium_density = neq;
    particle_data[i].bulk_density = dn_bulk;
    particle_data[i].diff_density = dn_diff;
  }
}








