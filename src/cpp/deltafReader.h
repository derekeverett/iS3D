
#ifndef DELTAFREADER_H
#define DELTAFREADER_H

#include "ParameterReader.h"
#include "readindata.h"
#include "gaussThermal.h"
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
        // (c1, G = 0) for muB = 0 and (c3, c4, betaV) aren't needed since they couple to baryon diffusion
        // so in the cubic spline evaluation: just set (G, c1, c3, c4) = 0 and betaV = 1 (betaV is in denominator)

        gsl_spline * c0_spline;
        gsl_spline * c2_spline;

        gsl_spline * F_spline;
        gsl_spline * betabulk_spline;
        gsl_spline * betapi_spline;

        // Jonah coefficients
        const int jonah_points = 301;       // # lambda interpolation points
        const double lambda_min = -1.0;     // lambda min / max values
        const double lambda_max = 2.0;
        const double delta_lambda = (lambda_max - lambda_min) / ((double)jonah_points - 1.0);

        double * lambda_array;              // isotropic momentum scale
        double * z_array;                   // renormalization factor (apart from detLambda)
        double * bulkPi_over_Peq_array;     // bulk pressure output
        double bulkPi_over_Peq_max;         // the maximum bulk pressure in the array

        gsl_spline * lambda_spline;         // cubic splines for lambda(bulkPi/Peq) and z(bulkPi/Peq)
        gsl_spline * z_spline;

        Deltaf_Data(ParameterReader * paraRdr_in);
        ~Deltaf_Data();

        void load_df_coefficient_data();    // read the data files in /deltaf_coefficients/vh

        void construct_cubic_splines();

        // I skip the photon because I think it breaks down for lambda = -1
        void compute_jonah_coefficients(particle_info * particle_data, int Nparticle);

        void compute_jonah_coefficients(double *Degeneracy, double *Mass, double *Sign, int number_of_chosen_particles);

        deltaf_coefficients evaluate_df_coefficients(double T, double muB, double E, double P, double bulkPi);

        deltaf_coefficients cubic_spline(double T, double E, double P, double bulkPi);

        deltaf_coefficients bilinear_interpolation(double T, double muB, double E, double P, double bulkPi);

};

#endif
