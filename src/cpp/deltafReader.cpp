
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

deltaf_coefficients DeltafReader::load_coefficients(FO_surf * surface, long FO_length)
{
  deltaf_coefficients df_data;

  // T and muB (fm^-1) (assumed to be ~ same for all cells) 
  double T_FO = surface[0].T / hbarC;
  double muB_FO = 0.0;
  if(include_baryon) muB_FO = surface[0].muB / hbarC;


  printf("Reading in...");

  if(df_mode == 1)
  {
    printf("14-moment coefficients (vhydro)\n");
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

    // how to put pathTodeltaf in here

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


    int nT = 0;
    int nB = 0;

    // read 1st line (T dimension) and 2nd line (muB dimension)
    fscanf(c0_file, "%d\n%d\n", &nT, &nB);
    fscanf(c1_file, "%d\n%d\n", &nT, &nB);
    fscanf(c2_file, "%d\n%d\n", &nT, &nB);
    fscanf(c3_file, "%d\n%d\n", &nT, &nB);
    fscanf(c4_file, "%d\n%d\n", &nT, &nB);

    if(!include_baryon) nB = 1;

    cout << nT << "\t" << nB << endl;

    // skip the headers
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

    // close files
    fclose(c0_file);
    fclose(c1_file);
    fclose(c2_file);
    fclose(c3_file);
    fclose(c4_file);

  }
  else if(df_mode == 2 || df_mode == 3)
  {
    printf("Chapman-Enskog coefficients (vhydro)\n");

    // coefficient files and names
    FILE * F_file;
    FILE * G_file;
    FILE * betabulk_file;
    FILE * betaV_file;
    FILE * betapi_file;

    char F_name[255];
    char G_name[255];
    char betabulk_name[255];
    char betaV_name[255];
    char betapi_name[255];

    // for skipping header
    char header[300];

    // how to take put pathTodeltaf in here?

    sprintf(F_name, "%s", "deltaf_coefficients/F_dfce_vh.dat");
    sprintf(G_name, "%s", "deltaf_coefficients/G_dfce_vh.dat");
    sprintf(betabulk_name, "%s", "deltaf_coefficients/betabulk_dfce_vh.dat");
    sprintf(betaV_name, "%s", "deltaf_coefficients/betaV_dfce_vh.dat");
    sprintf(betapi_name, "%s", "deltaf_coefficients/betapi_dfce_vh.dat");

    F_file = fopen(F_name, "r");
    G_file = fopen(G_name, "r");
    betabulk_file = fopen(betabulk_name, "r");
    betaV_file = fopen(betaV_name, "r");
    betapi_file = fopen(betapi_name, "r");

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

    cout << nT << "\t" << nB << endl;

    // skip the headers
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

    // close files
    fclose(F_file);
    fclose(G_file);
    fclose(betabulk_file);
    fclose(betaV_file);
    fclose(betapi_file);

  }
  else if(df_mode == 4)
  {
    printf("14-moment coefficients (vahydro PL)\n");

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

    sprintf(c0_name, "%s", "deltaf_coefficients/vah/c0_vah1.dat");
    sprintf(c1_name, "%s", "deltaf_coefficients/vah/c1_vah1.dat");
    sprintf(c2_name, "%s", "deltaf_coefficients/vah/c2_vah1.dat");
    sprintf(c3_name, "%s", "deltaf_coefficients/vah/c3_vah1.dat");
    sprintf(c4_name, "%s", "deltaf_coefficients/vah/c4_vah1.dat");

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

    int nL = 0;
    int naL = 0;

    // read 1st line (L dimension) and 2nd line (aL dimension)
    fscanf(c0_file, "%d\n%d\n", &nL, &naL);
    fscanf(c1_file, "%d\n%d\n", &nL, &naL);
    fscanf(c2_file, "%d\n%d\n", &nL, &naL);
    fscanf(c3_file, "%d\n%d\n", &nL, &naL);
    fscanf(c4_file, "%d\n%d\n", &nL, &naL);

    cout << nL << "\t" << naL << endl;

    //skip the header with labels and units
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
