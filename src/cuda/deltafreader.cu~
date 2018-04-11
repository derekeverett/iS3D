
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
  
  sprintf(c0_name, "%s", "deltaf_coefficients/c0_df14_vh.dat");
  sprintf(c1_name, "%s", "deltaf_coefficients/c1_df14_vh.dat");
  sprintf(c2_name, "%s", "deltaf_coefficients/c2_df14_vh.dat");
  sprintf(c3_name, "%s", "deltaf_coefficients/c3_df14_vh.dat");
  sprintf(c4_name, "%s", "deltaf_coefficients/c4_df14_vh.dat");

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


  int nT, nB;

  // read 1st line (T dimension) and 2nd line (muB dimension)
  fscanf(c0_file, "%d\n%d\n", &nT, &nB);
  fscanf(c1_file, "%d\n%d\n", &nT, &nB);
  fscanf(c2_file, "%d\n%d\n", &nT, &nB);
  fscanf(c3_file, "%d\n%d\n", &nT, &nB);
  fscanf(c4_file, "%d\n%d\n", &nT, &nB);

  cout << nT << "\t" << nB << endl;

  fgets(header, 100, c0_file);
  fgets(header, 100, c1_file);
  fgets(header, 100, c2_file);
  fgets(header, 100, c3_file);
  fgets(header, 100, c4_file);

  if(!include_baryon) nB = 1;

  // T and muB arrays
  double T_array[nT][nB];    
  double muB_array[nT][nB];

  // coefficient tables
  double c0[nT][nB];          
  double c1[nT][nB];
  double c2[nT][nB];
  double c3[nT][nB];
  double c4[nT][nB];

  cout << "Loading 14-moment coefficients..." << endl;

    // scan c0, c2 files
    for(int iB = 0; iB < nB; iB++)
    {
      for(int iT = 0; iT < nT; iT++)
      {
        // set T and muB (fm^-1) arrays from file
        fscanf(c0_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c0[iT][iB]);
        fscanf(c1_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c1[iT][iB]);
        fscanf(c2_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c2[iT][iB]);
        fscanf(c3_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c3[iT][iB]);
        fscanf(c4_file, "%lf\t\t%lf\t\t%lf\n", &T_array[iT][iB], &muB_array[iT][iB], &c4[iT][iB]);

        // check if cross freezeout temperature (T_array increasing)
        if(iT > 0 && T_FO < T_array[iT][iB])
        {
          // linear-interpolate wrt temperature (c0,c2) at T_invfm
          df_data.c0 = c0[iT-1][iB] + c0[iT][iB] * ((T_FO - T_array[iT-1][iB]) / T_array[iT-1][iB]);
          df_data.c1 = c1[iT-1][iB] + c1[iT][iB] * ((T_FO - T_array[iT-1][iB]) / T_array[iT-1][iB]);
          df_data.c2 = c2[iT-1][iB] + c2[iT][iB] * ((T_FO - T_array[iT-1][iB]) / T_array[iT-1][iB]);
          df_data.c3 = c3[iT-1][iB] + c3[iT][iB] * ((T_FO - T_array[iT-1][iB]) / T_array[iT-1][iB]);
          df_data.c4 = c4[iT-1][iB] + c4[iT][iB] * ((T_FO - T_array[iT-1][iB]) / T_array[iT-1][iB]);
          break;
        }
      } //iT
    } //iB

    // convert 14-momentum coefficients to real-life units 
    df_data.c0 /= (hbarC * hbarC * hbarC);
    df_data.c1 /= (hbarC * hbarC);
    df_data.c2 /= (hbarC * hbarC * hbarC);
    df_data.c3 /= (hbarC * hbarC);
    df_data.c4 /= (hbarC * hbarC * hbarC);




  return df_data;
}







