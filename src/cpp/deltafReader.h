
#ifndef DELTAFREADER_H
#define DELTAFREADER_H

#include "ParameterReader.h"
#include "readindata.h"
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

using namespace std;


class Deltaf_Reader
{
    private:
        ParameterReader * paraRdr;

        int mode; //type of freezeout surface, VH or VAH
        int df_mode; // type of delta-f correction (e.g. 14-moment, CE, or modified distribution)
        int include_baryon;

    public:
        Deltaf_Reader(ParameterReader * paraRdr_in);
        ~Deltaf_Reader();

        deltaf_coefficients load_coefficients(FO_surf *surface, long FO_length_in);
};


class Deltaf_Data
{
    private:
        ParameterReader * paraRdr;

        int mode; //type of freezeout surface, VH or VAH
        int df_mode; // type of delta-f correction (e.g. 14-moment, CE, or modified distribution)
        int include_baryon;

    public:
        int points_T;
        int points_muB;

        double * T_array;
        double * muB_array;
        //  Coefficients of 14 moment approximation (vhydro)
        // df ~ ((c0-c2)m^2 + b.c1(u.p) + (4c2-c0)(u.p)^2).Pi + (b.c3 + c4(u.p))p_u.V^u + c5.p_u.p_v.pi^uv
        double ** c0_data;
        double ** c1_data;
        double ** c2_data;
        double ** c3_data;
        double ** c4_data;

        //  Coefficients of Chapman Enskog expansion (vhydro)
        // df ~ ((c0-c2)m^2 + b.c1(u.p) + (4c2-c0)(u.p)^2).Pi + (b.c3 + c4(u.p))p_u.V^u + c5.p_u.p_v.pi^uv
        double ** F_data;
        double ** G_data;
        double ** betabulk_data;
        double ** betaV_data;
        double ** betapi_data;


        // cubic splines of the coefficients as function of temperature only (neglect muB, nB, Vmu)
        // G = 0 for muB = 0 and (c3, c4, betaV) aren't needed since they couple to baryon diffusion
        // so in the cubic spline evaluation: just set (G, c3, c4) = 0 and betaV = 1 (betaV is in denominator)
        gsl_interp_accel * accelerate;

        gsl_spline * c0_spline;
        gsl_spline * c1_spline;
        gsl_spline * c2_spline;

        gsl_spline * F_spline;
        gsl_spline * betabulk_spline;
        gsl_spline * betapi_spline;

        //gsl_spline_free(y_spline);
        //gsl_interp_accel_free(acc);


        Deltaf_Data(ParameterReader * paraRdr_in);
        ~Deltaf_Data();

        void load_df_coefficient_data();    // read the data files in /deltaf_coefficients/vh

        void construct_cubic_splines();

        deltaf_coefficients evaluate_df_coefficients(double T, double muB, double E, double P);

        deltaf_coefficients cubic_spline(double T, double E, double P);

        deltaf_coefficients bilinear_interpolation(double T, double muB, double E, double P);

        //deltaf_coefficients

};

#endif
