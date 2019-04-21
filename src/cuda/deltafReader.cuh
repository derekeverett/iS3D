
#ifndef DELTAFREADER_H
#define DELTAFREADER_H

#include "ParameterReader.cuh"
#include "readindata.cuh"
#include <fstream>

using namespace std;


class Deltaf_Data
{
    private:
        ParameterReader * paraRdr;

        int hrg_eos; // type of pdg file for hadron resonance gas EoS
        int mode; //type of freezeout surface, VH or VAH
        int df_mode; // type of delta-f correction (e.g. 14-moment, CE, or modified distribution)
        int include_baryon;

        string hrg_eos_path;
        string urqmd = "deltaf_coefficients/vh/urqmd/"; // directories of df coefficient tables
        string smash = "deltaf_coefficients/vh/smash/";
        string smash_box = "deltaf_coefficients/vh/smash_box/";
    public:
        int points_T;
        int points_muB;

        double T_min;
        double muB_min;

        double dT;
        double dmuB;

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

        // Jonah coefficients
        const int jonah_points = 301;       // # lambda interpolation points
        const double lambda_min = -1.0;     // lambda min / max values
        const double lambda_max = 2.0;
        const double delta_lambda = (lambda_max - lambda_min) / ((double)jonah_points - 1.0);

        double * lambda_squared_array;      // squared isotropic momentum scale
        double * z_array;                   // renormalization factor (apart from detLambda)
        double * bulkPi_over_Peq_array;     // bulk pressure output
        double bulkPi_over_Peq_max;         // the maximum bulk pressure in the array

        Deltaf_Data(ParameterReader * paraRdr_in);
        ~Deltaf_Data();

        void load_df_coefficient_data();    // read the data files in /deltaf_coefficients/vh

        // I skip the photon because I think it breaks down for lambda = -1
        void compute_jonah_coefficients(particle_info * particle_data, int Nparticle);

        deltaf_coefficients evaluate_df_coefficients(double T, double muB, double E, double P, double bulkPi);

        deltaf_coefficients linear_interpolation(double T, double E, double P, double bulkPi);

        double calculate_linear_temperature(double ** f_data, double T, double TL, double TR, int iTL, int iTR);

        double calculate_bilinear(double ** f_data, double T, double muB, double TL, double TR, double muBL, double muBR, int iTL, int iTR, int imuBL, int imuBR);

        deltaf_coefficients bilinear_interpolation(double T, double muB, double E, double P, double bulkPi);

        void test_df_coefficients(double bulkPi_over_P);

   
};

#endif
