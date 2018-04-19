
#ifndef DELTAFREADER_H
#define DELTAFREADER_H

#include "ParameterReader.cuh"
#include "readindata.cuh"
#include<fstream>

using namespace std;

typedef struct
{
  //  Coefficients of 14 moment approximation (vhydro)
  // df ~ ((c0-c2)m^2 + b.c1(u.p) + (4c2-c0)(u.p)^2).Pi + (b.c3 + c4(u.p))p_u.V^u + c5.p_u.p_v.pi^uv

  double c0;
  double c1;
  double c2;
  double c3;
  double c4;


  //  Coefficients of Chapman Enskog expansion (vhydro)
  // df ~ ((c0-c2)m^2 + b.c1(u.p) + (4c2-c0)(u.p)^2).Pi + (b.c3 + c4(u.p))p_u.V^u + c5.p_u.p_v.pi^uv
  // double F;
  // double G;
  // double betabulk;
  // double betaV;
  // double betapi;


} deltaf_coefficients;


class DeltafReader
{
    private:
        ParameterReader * paraRdr;
        string pathTodeltaf;

        int mode; //type of freezeout surface, VH or VAH
        int df_mode; // type of delta-f correction (e.g. 14-moment, CE, or modified distribution)
        int include_baryon;

    public:
        DeltafReader(ParameterReader * paraRdr_in, string path_in);
        ~DeltafReader();

        deltaf_coefficients load_coefficients(FO_surf *surface, long FO_length_in);
};

#endif
