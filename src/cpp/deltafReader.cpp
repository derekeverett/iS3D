
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>

#include "main.h"
#include "deltafReader.h"
#include "ParameterReader.h"
#include "readindata.h"

using namespace std;

DeltafReader::DeltafReader(ParameterReader * paraRdr_in, string path_in)
{
  paraRdr = paraRdr_in;
  pathTodeltaf = path_in;

  mode = paraRdr->getVal("mode");
  df_mode = paraRdr->getVal("df_mode");
  include_baryon = paraRdr->getVal("include_baryon");

}

DeltafReader::~DeltafReader()
{

}

deltaf_coefficients DeltafReader::load_coefficients(FO_surf surface)
{
  deltaf_coefficients df_data;

  // T and muB (fm^-1)
  double T_FO = surface.T / hbarC;
  double muB_FO = 0.0;
  if(include_baryon) muB_FO = surface.muB / hbarC;


  printf("Reading in Deltaf coefficients...\n");

  // coefficient files and names
  FILE * c0_file;
  FILE * c1_file;
  FILE * c2_file;
  FILE * c3_file;
  FILE * c4_file;

  char c0_name[255];
  char c1_name[255];
  char c2_name[255];
  char c3_name[255];
  char c4_name[255];

  // for skipping header
  char header[300];

  // how to take put pathTodeltaf in here?

  if (mode == 1) //viscous hydro
  {
    sprintf(c0_name, "%s", "deltaf_coefficients/vh/c0_df14_vh.dat");
    sprintf(c1_name, "%s", "deltaf_coefficients/vh/c1_df14_vh.dat");
    sprintf(c2_name, "%s", "deltaf_coefficients/vh/c2_df14_vh.dat");
    sprintf(c3_name, "%s", "deltaf_coefficients/vh/c3_df14_vh.dat");
    sprintf(c4_name, "%s", "deltaf_coefficients/vh/c4_df14_vh.dat");
  }

  else if (mode == 2) //va hydro PL matching
  {
    sprintf(c0_name, "%s", "deltaf_coefficients/vah/c0_vah1.dat");
    sprintf(c1_name, "%s", "deltaf_coefficients/vah/c1_vah1.dat");
    sprintf(c2_name, "%s", "deltaf_coefficients/vah/c2_vah1.dat");
    sprintf(c3_name, "%s", "deltaf_coefficients/vah/c3_vah1.dat");
    sprintf(c4_name, "%s", "deltaf_coefficients/vah/c4_vah1.dat");
  }


  c0_file = fopen(c0_name, "r");
  c1_file = fopen(c1_name, "r");
  c2_file = fopen(c2_name, "r");
  c3_file = fopen(c3_name, "r");
  c4_file = fopen(c4_name, "r");

  if(c0_file == NULL) printf("Couldn't open c0 coefficient file!\n");
  if(c1_file == NULL) printf("Couldn't open c1 coefficient file!\n");
  if(c2_file == NULL) printf("Couldn't open c2 coefficient file!\n");
  if(c3_file == NULL) printf("Couldn't open c3 coefficient file!\n");
  if(c4_file == NULL) printf("Couldn't open c4 coefficient file!\n");

  int nT = 0;
  int nB = 0;
  int nL = 0;
  int naL = 0;

  if (mode == 1)
  {
    // read 1st line (T dimension) and 2nd line (muB dimension)
    fscanf(c0_file, "%d\n%d\n", &nT, &nB);
    fscanf(c1_file, "%d\n%d\n", &nT, &nB);
    fscanf(c2_file, "%d\n%d\n", &nT, &nB);
    fscanf(c3_file, "%d\n%d\n", &nT, &nB);
    fscanf(c4_file, "%d\n%d\n", &nT, &nB);
  }
  else if (mode == 2)
  {
    // read 1st line (L dimension) and 2nd line (aL dimension)
    fscanf(c0_file, "%d\n%d\n", &nL, &naL);
    fscanf(c1_file, "%d\n%d\n", &nL, &naL);
    fscanf(c2_file, "%d\n%d\n", &nL, &naL);
    fscanf(c3_file, "%d\n%d\n", &nL, &naL);
    fscanf(c4_file, "%d\n%d\n", &nL, &naL);
  }


  if (mode == 1) cout << nT << "\t" << nB << endl;
  if (mode == 2) cout << nL << "\t" << naL << endl;

  //skip the header with labels and units
  fgets(header, 100, c0_file);
  fgets(header, 100, c1_file);
  fgets(header, 100, c2_file);
  fgets(header, 100, c3_file);
  fgets(header, 100, c4_file);

  if(!include_baryon) nB = 1;

  // T and muB arrays
  double T_array[nT][nB];
  double muB_array[nT][nB];

  //L and aL arrays
  double L_array[nL][naL];
  double aL_array[nL][naL];

  // coefficient tables

  int n1, n2;

  if (mode == 1) {n1 = nT; n2 = nB;}
  if (mode == 2) {n1 = nL; n2 = naL;}

  double c0[n1][n2];
  double c1[n1][n2];
  double c2[n1][n2];
  double c3[n1][n2];
  double c4[n1][n2];

  cout << "Loading 14-moment coefficients from files in deltaf_coefficients..." << endl;

  // scan c0, c2 files

  if (mode == 1)
  {
    for(int i2 = 0; i2 < n2; i2++)
    {
      for(int i1 = 0; i1 < n1; i1++)
      {
        // set T and muB (fm^-1) arrays from file
        fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &T_array[i1][i2], &muB_array[i1][i2], &c0[i1][i2]);
        fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &T_array[i1][i2], &muB_array[i1][i2], &c1[i1][i2]);
        fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &T_array[i1][i2], &muB_array[i1][i2], &c2[i1][i2]);
        fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &T_array[i1][i2], &muB_array[i1][i2], &c3[i1][i2]);
        fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &T_array[i1][i2], &muB_array[i1][i2], &c4[i1][i2]);

        // check if cross freezeout temperature (T_array increasing)
        if(i1 > 0 && T_FO < T_array[i1][i2])
        {
          // linear-interpolate wrt temperature (c0,c2) at T_invfm
          df_data.c0 = c0[i1-1][i2] + c0[i1][i2] * ((T_FO - T_array[i1-1][i2]) / T_array[i1-1][i2]);
          df_data.c1 = c1[i1-1][i2] + c1[i1][i2] * ((T_FO - T_array[i1-1][i2]) / T_array[i1-1][i2]);
          df_data.c2 = c2[i1-1][i2] + c2[i1][i2] * ((T_FO - T_array[i1-1][i2]) / T_array[i1-1][i2]);
          df_data.c3 = c3[i1-1][i2] + c3[i1][i2] * ((T_FO - T_array[i1-1][i2]) / T_array[i1-1][i2]);
          df_data.c4 = c4[i1-1][i2] + c4[i1][i2] * ((T_FO - T_array[i1-1][i2]) / T_array[i1-1][i2]);
          break;
        }
      } //i1
    } //i2

    // convert 14-momentum coefficients to real-life units
    df_data.c0 /= (hbarC * hbarC * hbarC);
    df_data.c1 /= (hbarC * hbarC);
    df_data.c2 /= (hbarC * hbarC * hbarC);
    df_data.c3 /= (hbarC * hbarC);
    df_data.c4 /= (hbarC * hbarC * hbarC);
  }
  else if (mode == 2)
  {
    for(int i2 = 0; i2 < n2; i2++)
    {
      for(int i1 = 0; i1 < n1; i1++)
      {
        // set L and aL (fm^-1) arrays from file
        fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1][i2], &aL_array[i1][i2], &c0[i1][i2]);
        fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1][i2], &aL_array[i1][i2], &c1[i1][i2]);
        fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1][i2], &aL_array[i1][i2], &c2[i1][i2]);
        fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1][i2], &aL_array[i1][i2], &c3[i1][i2]);
        fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &L_array[i1][i2], &aL_array[i1][i2], &c4[i1][i2]);
      } //i1
    } //i2

    /* NOW WHAT?*/


  } // if (mode == 2)

  return df_data;
}
