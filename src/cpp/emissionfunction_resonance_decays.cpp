// Author: Mike McNelis
// Date: 8/11/18


// Differences from the old resonance decay code (so far):
// 1) top down approach: loop over parents instead of daughters
// 2) decay grouping: group decay products by type (~10% speedup)
// 3) mT extrapolation instead of high mT cutoff (for modified)
// 4) linear interpolation of log(dN_pTdpTdphidy) instead of dN_pTdpTdphidy


#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <chrono>
#include <limits>
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
//#ifdef _OPENACC
//#include <accelmath.h>
//#endif
using namespace std;

// grouping:
//  - 2-body decays take 62.6s
//  - 3-body decays take 122.9s
//  - 0 decays takes 4ms (good)

// without grouping:
//  - 2-body decays take 63.78s (no difference)
//  - 3-body decays take 141.4s (main benefit)


// the script is laid out now but needs more testing
// - reproduce the plots in the old paper based
// - there's an analytical
// - Chun didn't really have anything helpful to add to understanding the old code



// with grouping it took 185s for the boost invariant calculation
// without grouping it takes 204s (not significanty slower...)
// it's because the majority of decays are two-body (pairs are generally not same type)

// ~ 10% faster...(well some kind of speed up would be appreciated...)
// also the old code takes about 4 minutes, this one takes about 3 minutes (147s without mt extrapolation)

// the other idea is to do parallelization from the top down
// organize 3 body decay and 2 body decay channels

double calculate_Q_factor(double mass_parent, double mass_1, double mass_2, double mass_3)
{
    double Q = 0.0;

    double a = (mass_parent + mass_1) * (mass_parent + mass_1);
    double b = (mass_parent - mass_1) * (mass_parent - mass_1);     // s+
    double c = (mass_2 + mass_3) * (mass_2 + mass_3);               // s-
    double d = (mass_2 - mass_3) * (mass_2 - mass_3);

    /// 24 gauss-legendre points for accurate Q-factor calculation
    const int x_pts = 24;
    double x_root[x_pts] = {-0.99518721999702,-0.97472855597131,-0.93827455200273,-0.8864155270044,-0.8200019859739,-0.74012419157855,-0.64809365193698,-0.54542147138884,-0.43379350762605,-0.31504267969616,-0.19111886747362,-0.064056892862606,0.06405689286261,0.19111886747362,0.31504267969616,0.43379350762605,0.54542147138884,0.64809365193698,0.74012419157855,0.8200019859739,0.8864155270044,0.93827455200273,0.97472855597131,0.99518721999702};

    double x_weight[x_pts] = {0.01234122979999,0.02853138862893,0.0442774388174,0.059298584915437,0.0733464814111,0.08619016153195,0.0976186521041,0.107444270116,0.11550566805373,0.1216704729278,0.12583745634683,0.1279381953468,0.1279381953468,0.1258374563468,0.1216704729278,0.1155056680537,0.107444270116,0.09761865210411,0.08619016153195,0.07334648141108,0.05929858491544,0.04427743881742,0.02853138862893,0.01234122979999};

    for(int i = 0; i < x_pts; i++)
    {
        // coordinate transformation of s = [s-,s+]:  s = s- + (s+ - s-)(1+x)/2, where x = [-1,1]
        // substitution rule prefactor = (s+ - s-) / 2
        double s = c + (b - c) * (1.0 + x_root[i]) / 2.0;
        Q += x_weight[i] * (b - c) * sqrt(fabs((a - s) * (b - s) * (s - c) * (s - d))) / (2.0 * s);
    }
    return Q;
}


int particle_index(particle_info * particle_data, const int number_of_particles, const int entry_mc_id)
{
    // search for index of particle of interest in pdg.dat by matching mc_id
    if(entry_mc_id == 0)
    {
        printf("Error: interested particle's mc_id is 0 (null particle in pdg.dat)\n");
        exit(-1);
    }

    int index;
    bool found = false;

    for(int ipart = 0; ipart < number_of_particles; ipart++)
    {
        if(particle_data[ipart].mc_id == entry_mc_id) // I could also use the class object particles
        {
            index = ipart;
            found = true;
            break;
        }
    }
    if(found)
    {
        return index;
    }
    else
    {
        printf("\nError: couldn't find mc_id in particle data\n");
        exit(-1);
    }
}

int EmissionFunctionArray::particle_chosen_index(int particle_index)
{
    // search for index of particle of interest in chosen_particles.dat by particle's pdg.dat index
    int chosen_index;
    bool found = false;

    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
        if(chosen_particles_sampling_table[ipart] == particle_index)  // I could also use the class object particles
        {
            chosen_index = ipart;
            found = true;
            break;
        }
    }
    if(found)
    {
        return chosen_index;
    }
    else
    {
        printf("\nError: couldn't find particle in chosen_particles.dat\n");
        exit(-1);
    }
}

MT_fit_parameters EmissionFunctionArray::estimate_MT_function_of_dNdypTdpTdphi(int iy, int iphip, double mass_parent)
{
    // get parameters for the fit y = exp(constant + slope * mT) of the distribution dN_dymTdmTdphi at large mT
    // mT_const ~ logy-intercept, mT_slope ~ -effective temperature in GeV
    MT_fit_parameters MT_params;


    // temporary test:
    // MT_params.constant = 4.88706;
    // MT_params.slope = -2.04108;
    // return MT_params;

    // set pT array
    double pTValues[pT_tab_length];
    for(int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

    // get the coordinates (mT_points, log_dNdydpT) when the distribution is dN_pTdpTdphidy positive
    vector<double> mT_points;
    vector<double> logdN_points;

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
        long int iS_parent = ipT + pT_tab_length * (iphip + phi_tab_length * iy);

        double logdN = logdN_PTdPTdPhidY[iS_parent];

        if(std::isfinite(logdN))
        {
            double pT = pTValues[ipT];
            double mT = sqrt(mass_parent * mass_parent + pT * pT);

            if(mT > sqrt(2.73) * mass_parent)   // mT in relativistic region (~ threshold for 2 points for Omega2250)
            {
                mT_points.push_back(mT);
                logdN_points.push_back(logdN);   // called y below
            }
        }
        else
        {
            // dN_PTdPTdPhidY is negative / zero, stop collecting coordinates
            break;
        }
    }

    // need at least two points for linear fit
    if(mT_points.size() < 2)
    {
        printf("\nError: not enough points to construct a least squares fit\n");
        exit(-1);
    }

    // set up least squares matrix equation to solve for
    // the coefficients of a straight line: A^Ty = A^TAx

    // number of points we're fitting a line through
    const int n = mT_points.size();

    // coefficients of a straight line = (constant, mT_slope)
    double x[2] = {0.0, 0.0};

    // matrix A
    double A[n][2];

    for(int i = 0; i < n; i++)
    {
        A[i][0] = 1.0;          // first column
        A[i][1] = mT_points[i]; // second column
        //      [ 1    mT_0   ]
        //      [ 1    mT_1   ]
        // A =  [ ..    ..    ]
        //      [ 1    mT_n-1 ]
    }

    // transpose of A
    double A_transpose[2][n];

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < n; j++)
        {
            A_transpose[i][j] = A[j][i];
        }
    }

    // f = AT * y (y = log_dNdypTdpTdphi)
    double f[2] = {0.0, 0.0};

    for(int i = 0; i < 2; i++)
    {
        double sum = 0.0;
        for(int k = 0; k < n; k++)
        {
            sum += (A_transpose[i][k] * logdN_points[k]);
        }
        f[i] = sum;
    }

    // square matrix M = AT * A
    double ** M = (double**)calloc(2, sizeof(double*));
    for(int i = 0; i < 2; i++) M[i] = (double*)calloc(2, sizeof(double));

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            double sum = 0.0; // default sum zero
            for(int k = 0; k < n; k++)
            {
                // M_ij = A_ik B_kj
                sum += (A_transpose[i][k] * A[k][j]);
            }
            M[i][j] = sum;
        }
    }
    // now ready to solve Mx = f
    int permutation[2];                     // permutation vector
    LUP_decomposition(M, 2, permutation);   // LUP decompose M
    LUP_solve(M, 2, permutation, f);        // solve matrix equation
    for(int i = 0; i < 2; i++) x[i] = f[i]; // solution stored in f
    free_2D(M,2);                           // free memory
    //----------------------------------

    MT_params.constant = x[0];     // get constant and slope of exponential fit
    MT_params.slope = x[1];

    return MT_params;
}

double EmissionFunctionArray::dN_dYMTdMTdPhi_boost_invariant(int parent_chosen_index, double * MTValues, double PhipValues[], double MT, double Phip1, double Phip2, double Phip_min, double Phip_max, double MTmax, MT_fit_parameters ** MT_params)
{
    // linear interpolation boost_invariant of
    // parent log distribution log(dN_dYMTdMTdPhi)

    // two contributions: parent 1 (w/ Phip1) and parent 2 (w/ Phip2)
    double logdN1 = 0.0;
    double logdN2 = 0.0;

    // I think I ironed out the bugs here and the
    // simplified 2-body integration test works

    // I could move the Y search outside the zeta loop later (to save time)

    if(MT <= MTmax)
    {
        // bi-linear interpolation in (Phip, MT)

        // first search for left/right (L/R) interpolation points
        int iPhip1L, iPhip1R;   // Phip1 interpolation indices
        int iPhip2L, iPhip2R;   // Phip2 interpolation indices
        double Phip1R, Phip1L;  // Phip1 interpolation points
        double Phip2R, Phip2L;  // Phip2 interpolation points


        // determine whether Phip1 in phi_gauss_table.dat range:
        //----------------------------------------
        if(Phip1 >= Phip_min && Phip1 <= Phip_max)  // fixed logic statement bug || -> && on 8/7
        {
            iPhip1R = 1;
            while(Phip1 > PhipValues[iPhip1R])
            {
                iPhip1R++;
            }
            iPhip1L = iPhip1R - 1;
            // Phip1 interpolation points
            Phip1L = PhipValues[iPhip1L];
            Phip1R = PhipValues[iPhip1R];
        }
        else
        {
            // settings for outside of range
            iPhip1L = phi_tab_length - 1;
            iPhip1R = 0;
            Phip1L = PhipValues[iPhip1L] - 2.0 * M_PI; // small negative angle
            Phip1R = PhipValues[iPhip1R];              // small positive angle
            // put Phip1 in between interpolation points
            Phip1 -= floor(Phip1 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------

        // repeat for Phip2:
        //----------------------------------------
        if(Phip2 >= Phip_min && Phip2 <= Phip_max)
        {
            iPhip2R = 1;
            while(Phip2 > PhipValues[iPhip2R])
            {
                iPhip2R++;
            }
            iPhip2L = iPhip2R - 1;
            // Phip2 interpolation points
            Phip2L = PhipValues[iPhip2L];
            Phip2R = PhipValues[iPhip2R];
        }
        else
        {
            // settings for outside of range
            iPhip2L = phi_tab_length - 1;
            iPhip2R = 0;
            Phip2L = PhipValues[iPhip2L] - 2.0 * M_PI;
            Phip2R = PhipValues[iPhip2R];
            // put Phip2 in between interpolation points
            Phip2 -= floor(Phip2 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------


        // MT interpolation points:
        //----------------------------------------
        int iMTR = 1;
        // because of MT if statement, loop will terminate:
        while(MT > MTValues[iMTR])
        {
            iMTR++;
        }
        int iMTL = iMTR - 1;
        double MTL = MTValues[iMTL];
        double MTR = MTValues[iMTR];
        //----------------------------------------


        // intervals:
        //----------------------------------------
        double dPhip1 = Phip1R - Phip1L;
        double dPhip2 = Phip2R - Phip2L;
        double dMT = MTR - MTL;
        //----------------------------------------


        // temporary (for precision test)
        // iPhip1L = 0;
        // iPhip1R = 0;
        // iPhip2L = 0;
        // iPhip2R = 0;


        // evaluate interpolation function points for parent 1  (LL, etc ordered in (Phip1, MT))
        //----------------------------------------
        int iS1_LL = iMTL + pT_tab_length * iPhip1L;
        int iS1_RL = iMTL + pT_tab_length * iPhip1R;
        int iS1_LR = iMTR + pT_tab_length * iPhip1L;
        int iS1_RR = iMTR + pT_tab_length * iPhip1R; // fixed bug 8/6

        // log of parent 1 distribution
        double logdN1_LL = logdN_PTdPTdPhidY[iS1_LL];
        double logdN1_RL = logdN_PTdPTdPhidY[iS1_RL];
        double logdN1_LR = logdN_PTdPTdPhidY[iS1_LR];
        double logdN1_RR = logdN_PTdPTdPhidY[iS1_RR];
        //----------------------------------------


        // evaluate interpolation function points for parent 2 (LL, etc ordered in (Phip2, MT))
        //----------------------------------------
        int iS2_LL = iMTL + pT_tab_length * iPhip2L;
        int iS2_RL = iMTL + pT_tab_length * iPhip2R;
        int iS2_LR = iMTR + pT_tab_length * iPhip2L;
        int iS2_RR = iMTR + pT_tab_length * iPhip2R;

        // log of parent 2 distribution
        double logdN2_LL = logdN_PTdPTdPhidY[iS2_LL];
        double logdN2_RL = logdN_PTdPTdPhidY[iS2_RL];
        double logdN2_LR = logdN_PTdPTdPhidY[iS2_LR];
        double logdN2_RR = logdN_PTdPTdPhidY[iS2_RR];
        //----------------------------------------


        // bi-linear interpolation for log parent 1 and 2
        //----------------------------------------
        logdN1 = ((logdN1_LL * (Phip1R - Phip1) + logdN1_RL * (Phip1 - Phip1L)) * (MTR - MT) +
                 (logdN1_LR * (Phip1R - Phip1) + logdN1_RR * (Phip1 - Phip1L)) * (MT - MTL)) / (dPhip1 * dMT);

        logdN2 = ((logdN2_LL * (Phip2R - Phip2) + logdN2_RL * (Phip2 - Phip2L)) * (MTR - MT) +
                 (logdN2_LR * (Phip2R - Phip2) + logdN2_RR * (Phip2 - Phip2L)) * (MT - MTL)) / (dPhip2 * dMT);
        //----------------------------------------
    }
    else
    {
        //return 0.0;
        // temp test
        //logdN1 = 4.88706 - 2.04108 * MT;
        //logdN2 = 4.88706 - 2.04108 * MT;
        //return (exp(logdN1) + exp(logdN2));

        // linear interpolation in Phip using
        // exponential fit in MT direction

        // search for left/right (L/R) interpolation points
        int iPhip1L, iPhip1R ;  // Phip1 interpolation indices
        int iPhip2L, iPhip2R;   // Phip2 interpolation indices
        double Phip1R, Phip1L;  // Phip1 interpolation points
        double Phip2R, Phip2L;  // Phip2 interpolation points

        // determine whether Phip1 in phi_gauss_table.dat range:
        //----------------------------------------
        if(Phip1 >= Phip_min && Phip1 <= Phip_max)
        {
            iPhip1R = 1;
            while(Phip1 > PhipValues[iPhip1R])
            {
                iPhip1R++;
            }
            iPhip1L = iPhip1R - 1;
            // Phip1 interpolation points
            Phip1L = PhipValues[iPhip1L];
            Phip1R = PhipValues[iPhip1R];
        }
        else
        {
            // settings for outside of range
            iPhip1L = phi_tab_length - 1;
            iPhip1R = 0;
            Phip1L = PhipValues[iPhip1L] - 2.0 * M_PI;  // small negative angle
            Phip1R = PhipValues[iPhip1R];               // small positive angle
            // put angle between interpolation points
            Phip1 -= floor(Phip1 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------

        // repeat for Phip2:
        //----------------------------------------
        if(Phip2 >= Phip_min && Phip2 <= Phip_max)
        {
            iPhip2R = 1;
            while(Phip2 > PhipValues[iPhip2R])
            {
                iPhip2R++;
            }
            iPhip2L = iPhip2R - 1;
            // Phip2 interpolation points
            Phip2L = PhipValues[iPhip2L];
            Phip2R = PhipValues[iPhip2R];
        }
        else
        {
            // settings for outside of range
            iPhip2L = phi_tab_length - 1;
            iPhip2R = 0;
            Phip2L = PhipValues[iPhip2L] - 2.0 * M_PI;
            Phip2R = PhipValues[iPhip2R];
            // put angle between interpolation points
            Phip2 -= floor(Phip2 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------

        // intervals
        //----------------------------------------
        double dPhip1 = Phip1R - Phip1L;
        double dPhip2 = Phip2R - Phip2L;
        //----------------------------------------


        // fit parameters for parent 1 (L/R)
        //----------------------------------------
        MT_fit_parameters MT_params1_L = MT_params[0][iPhip1L];
        MT_fit_parameters MT_params1_R = MT_params[0][iPhip1R];

        double const1_L = MT_params1_L.constant;
        double slope1_L = MT_params1_L.slope;
        double const1_R = MT_params1_R.constant;
        double slope1_R = MT_params1_R.slope;
        //----------------------------------------


        // fit parameters for parent 2 (L/R)
        //----------------------------------------
        MT_fit_parameters MT_params2_L = MT_params[0][iPhip2L];
        MT_fit_parameters MT_params2_R = MT_params[0][iPhip2R];

        double const2_L = MT_params2_L.constant;
        double slope2_L = MT_params2_L.slope;
        double const2_R = MT_params2_R.constant;
        double slope2_R = MT_params2_R.slope;
        //----------------------------------------


        // evaluate interpolation points for parents 1 and 2
        double logdN1_L = const1_L + slope1_L * MT;
        double logdN1_R = const1_R + slope1_R * MT;
        double logdN2_L = const2_L + slope2_L * MT;
        double logdN2_R = const2_R + slope2_R * MT;
        //----------------------------------------


        // linear interpolation for log parent 1 and 2
        //----------------------------------------
        logdN1 = (logdN1_L * (Phip1R - Phip1) + logdN1_R * (Phip1 - Phip1L)) / dPhip1;

        logdN2 = (logdN2_L * (Phip2R - Phip2) + logdN2_R * (Phip2 - Phip2L)) / dPhip2;
        //----------------------------------------
    }

    return (exp(logdN1) + exp(logdN2));   // undo the log
}



double EmissionFunctionArray::dN_dYMTdMTdPhi_non_boost_invariant(int parent_chosen_index, double * MTValues, double * PhipValues, int iYL, int iYR, double YL, double YR, double MT, double Phip1, double Phip2, double Y, double Phip_min, double Phip_max, double MTmax, MT_fit_parameters ** MT_params)
{
    // linear interpolation non-boost_invariant of
    // parent distribution log(dN_dYMTdMTdPhi)

    // two contributions: parent 1 (w/ Phip1) and parent 2 (w/ Phip2)
    double logdN1 = 0.0;
    double logdN2 = 0.0;

    // this hasn't been tested fully yet...
    // I could run the 2+1d surface with more y-rapidity points
    // I would have to change the yTable settings though..
    // the midrapidity points should agree with the boost invariant case to some degree


    // need to double check the L,R are corrects
    // double debug this, it's very mesmerising...

    if(MT <= MTmax)
    {
        // tri-linear interpolation in (Y, Phip, MT)

        // first search for left/right (L/R) interpolation points
        int iPhip1L, iPhip1R;   // Phip1 interpolation indices
        int iPhip2L, iPhip2R;   // Phip2 interpolation indices
        double Phip1R, Phip1L;  // Phip1 interpolation points
        double Phip2R, Phip2L;  // Phip2 interpolation points

        // determine whether Phip1 in phi_gauss_table.dat range:
        //----------------------------------------
        if(Phip1 >= Phip_min && Phip1 <= Phip_max)  // fixed logic statement bug || -> && on 8/7
        {
            iPhip1R = 1;
            while(Phip1 > PhipValues[iPhip1R])
            {
                iPhip1R++;
            }
            iPhip1L = iPhip1R - 1;
            // Phip1 interpolation points
            Phip1L = PhipValues[iPhip1L];
            Phip1R = PhipValues[iPhip1R];
        }
        else
        {
            // settings for outside of range
            iPhip1L = phi_tab_length - 1;
            iPhip1R = 0;
            Phip1L = PhipValues[iPhip1L] - 2.0 * M_PI; // small negative angle
            Phip1R = PhipValues[iPhip1R];              // small positive angle
            // put Phip1 in between interpolation points
            Phip1 -= floor(Phip1 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------

        // repeat for Phip2:
        //----------------------------------------
        if(Phip2 >= Phip_min && Phip2 <= Phip_max)
        {
            iPhip2R = 1;
            while(Phip2 > PhipValues[iPhip2R])
            {
                iPhip2R++;
            }
            iPhip2L = iPhip2R - 1;
            // Phip2 interpolation points
            Phip2L = PhipValues[iPhip2L];
            Phip2R = PhipValues[iPhip2R];
        }
        else
        {
            // settings for outside of range
            iPhip2L = phi_tab_length - 1;
            iPhip2R = 0;
            Phip2L = PhipValues[iPhip2L] - 2.0 * M_PI;
            Phip2R = PhipValues[iPhip2R];
            // put Phip2 in between interpolation points
            Phip2 -= floor(Phip2 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------


        // MT interpolation points:
        //----------------------------------------
        int iMTR = 1;
        // because of MT if statement, loop will terminate:
        while(MT > MTValues[iMTR])
        {
            iMTR++;
        }
        int iMTL = iMTR - 1;
        double MTL = MTValues[iMTL];
        double MTR = MTValues[iMTR];
        //----------------------------------------


        // Y interpolation points:
        //----------------------------------------
        // int iYR = 1;
        // // should terminate due to prior break statement
        // while(Y > YValues[iYR])
        // {
        //     iYR++;
        // }
        // int iYL = iYR - 1;
        // double YL = YValues[iYL];
        // double YR = YValues[iYR];
        //----------------------------------------


        // intervals
        //----------------------------------------
        double dY = YR - YL;
        double dPhip1 = Phip1R - Phip1L;
        double dPhip2 = Phip2R - Phip2L;
        double dMT = MTR - MTL;
        //----------------------------------------


        // evaluate interpolation points for parent 1  (LLL, etc ordered in (Y, Phip, MT))
        //----------------------------------------
        long int iS1_LLL = iMTL + pT_tab_length * (iPhip1L + phi_tab_length * iYL);
        long int iS1_RLL = iMTL + pT_tab_length * (iPhip1L + phi_tab_length * iYR);
        long int iS1_LRL = iMTL + pT_tab_length * (iPhip1R + phi_tab_length * iYL);
        long int iS1_RRL = iMTL + pT_tab_length * (iPhip1R + phi_tab_length * iYR);
        long int iS1_LLR = iMTR + pT_tab_length * (iPhip1L + phi_tab_length * iYL);
        long int iS1_RLR = iMTR + pT_tab_length * (iPhip1L + phi_tab_length * iYR);
        long int iS1_LRR = iMTR + pT_tab_length * (iPhip1R + phi_tab_length * iYL);
        long int iS1_RRR = iMTR + pT_tab_length * (iPhip1R + phi_tab_length * iYR);

        // log of parent 1 distribution
        double logdN1_LLL = logdN_PTdPTdPhidY[iS1_LLL];
        double logdN1_RLL = logdN_PTdPTdPhidY[iS1_RLL];
        double logdN1_LRL = logdN_PTdPTdPhidY[iS1_LRL];
        double logdN1_RRL = logdN_PTdPTdPhidY[iS1_RRL];
        double logdN1_LLR = logdN_PTdPTdPhidY[iS1_LLR];
        double logdN1_RLR = logdN_PTdPTdPhidY[iS1_RLR];
        double logdN1_LRR = logdN_PTdPTdPhidY[iS1_LRR];
        double logdN1_RRR = logdN_PTdPTdPhidY[iS1_RRR];
        //----------------------------------------


        // evaluate interpolation points for parent 2
        //----------------------------------------
        long int iS2_LLL = iMTL + pT_tab_length * (iPhip2L + phi_tab_length * iYL);
        long int iS2_RLL = iMTL + pT_tab_length * (iPhip2L + phi_tab_length * iYR);
        long int iS2_LRL = iMTL + pT_tab_length * (iPhip2R + phi_tab_length * iYL);
        long int iS2_RRL = iMTL + pT_tab_length * (iPhip2R + phi_tab_length * iYR);
        long int iS2_LLR = iMTR + pT_tab_length * (iPhip2L + phi_tab_length * iYL);
        long int iS2_RLR = iMTR + pT_tab_length * (iPhip2L + phi_tab_length * iYR);
        long int iS2_LRR = iMTR + pT_tab_length * (iPhip2R + phi_tab_length * iYL);
        long int iS2_RRR = iMTR + pT_tab_length * (iPhip2R + phi_tab_length * iYR);

        // log of parent 2 distribution
        double logdN2_LLL = logdN_PTdPTdPhidY[iS2_LLL];
        double logdN2_RLL = logdN_PTdPTdPhidY[iS2_RLL];
        double logdN2_LRL = logdN_PTdPTdPhidY[iS2_LRL];
        double logdN2_RRL = logdN_PTdPTdPhidY[iS2_RRL];
        double logdN2_LLR = logdN_PTdPTdPhidY[iS2_LLR];
        double logdN2_RLR = logdN_PTdPTdPhidY[iS2_RLR];
        double logdN2_LRR = logdN_PTdPTdPhidY[iS2_LRR];
        double logdN2_RRR = logdN_PTdPTdPhidY[iS2_RRR];
        //----------------------------------------


        // tri-linear interpolation for log parent 1
        //----------------------------------------
        logdN1 = (MTR - MT) * ((logdN1_LLL * (YR - Y) + logdN1_RLL * (Y - YL)) * (Phip1R - Phip1) +
                 (logdN1_LRL * (YR - Y) + logdN1_RRL * (Y - YL)) * (Phip1 - Phip1L))

                                                +

                 (MT - MTL) * ((logdN1_LLR * (YR - Y) + logdN1_RLR * (Y - YL)) * (Phip1R - Phip1) +
                 (logdN1_LRR * (YR - Y) + logdN1_RRR * (Y - YL)) * (Phip1 - Phip1L));

        logdN1 /= (dY * dPhip1 * dMT);
        //----------------------------------------


        // tri-linear interpolation for log parent 2
        //----------------------------------------
        logdN2 = (MTR - MT) * ((logdN2_LLL * (YR - Y) + logdN2_RLL * (Y - YL)) * (Phip2R - Phip2) +
                 (logdN2_LRL * (YR - Y) + logdN2_RRL * (Y - YL)) * (Phip2 - Phip2L))

                                                +

                 (MT - MTL) * ((logdN2_LLR * (YR - Y) + logdN2_RLR * (Y - YL)) * (Phip2R - Phip2) +
                 (logdN2_LRR * (YR - Y) + logdN2_RRR * (Y - YL)) * (Phip2 - Phip2L));

        logdN2 /= (dY * dPhip2 * dMT);
        //----------------------------------------
    }
    else
    {
        // bi-linear interpolation in (Y,Phip)
        // use exponential fit in MT direction
        // first search for left/right (L/R) interpolation points
        int iPhip1L, iPhip1R;   // Phip1 interpolation indices
        int iPhip2L, iPhip2R;   // Phip2 interpolation indices
        double Phip1R, Phip1L;  // Phip1 interpolation points
        double Phip2R, Phip2L;  // Phip2 interpolation points

        // determine whether Phip1 in phi_gauss_table.dat range:
        //----------------------------------------
        if(Phip1 >= Phip_min && Phip1 <= Phip_max)  // fixed logic statement bug || -> && on 8/7
        {
            iPhip1R = 1;
            while(Phip1 > PhipValues[iPhip1R])
            {
                iPhip1R++;
            }
            iPhip1L = iPhip1R - 1;
            // Phip1 interpolation points
            Phip1L = PhipValues[iPhip1L];
            Phip1R = PhipValues[iPhip1R];
        }
        else
        {
            // settings for outside of range
            iPhip1L = phi_tab_length - 1;
            iPhip1R = 0;
            Phip1L = PhipValues[iPhip1L] - 2.0 * M_PI; // small negative angle
            Phip1R = PhipValues[iPhip1R];              // small positive angle
            // put Phip1 in between interpolation points
            Phip1 -= floor(Phip1 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------

        // repeat for Phip2:
        //----------------------------------------
        if(Phip2 >= Phip_min && Phip2 <= Phip_max)
        {
            iPhip2R = 1;
            while(Phip2 > PhipValues[iPhip2R])
            {
                iPhip2R++;
            }
            iPhip2L = iPhip2R - 1;
            // Phip2 interpolation points
            Phip2L = PhipValues[iPhip2L];
            Phip2R = PhipValues[iPhip2R];
        }
        else
        {
            // settings for outside of range
            iPhip2L = phi_tab_length - 1;
            iPhip2R = 0;
            Phip2L = PhipValues[iPhip2L] - 2.0 * M_PI;
            Phip2R = PhipValues[iPhip2R];
            // put Phip2 in between interpolation points
            Phip2 -= floor(Phip2 / M_PI) * (2.0 * M_PI);
        }
        //----------------------------------------


        // Y interpolation points:
        //----------------------------------------
        // int iYR = 1;
        // // should terminate due to prior break statement
        // while(Y > YValues[iYR])
        // {
        //     iYR++;
        // }
        // int iYL = iYR - 1;
        // double YL = YValues[iYL];
        // double YR = YValues[iYR];
        //----------------------------------------


        // intervals
        //----------------------------------------
        double dY = YR - YL;
        double dPhip1 = Phip1R - Phip1L;
        double dPhip2 = Phip2R - Phip2L;
        //----------------------------------------


        // fit parameters for parent 1 (LL, etc ordered in (Y, Phip))
        //----------------------------------------
        MT_fit_parameters MT_params1_LL = MT_params[iYL][iPhip1L];
        MT_fit_parameters MT_params1_RL = MT_params[iYR][iPhip1L];
        MT_fit_parameters MT_params1_LR = MT_params[iYL][iPhip1R];
        MT_fit_parameters MT_params1_RR = MT_params[iYR][iPhip1R];

        double const1_LL = MT_params1_LL.constant;
        double slope1_LL = MT_params1_LL.slope;
        double const1_RL = MT_params1_RL.constant;
        double slope1_RL = MT_params1_RL.slope;
        double const1_LR = MT_params1_LR.constant;
        double slope1_LR = MT_params1_LR.slope;
        double const1_RR = MT_params1_RR.constant;
        double slope1_RR = MT_params1_RR.slope;
        //----------------------------------------


        // fit parameters for parent 2
        //----------------------------------------
        MT_fit_parameters MT_params2_LL = MT_params[iYL][iPhip2L];
        MT_fit_parameters MT_params2_RL = MT_params[iYR][iPhip2L];
        MT_fit_parameters MT_params2_LR = MT_params[iYL][iPhip2R];
        MT_fit_parameters MT_params2_RR = MT_params[iYR][iPhip2R];

        double const2_LL = MT_params2_LL.constant;
        double slope2_LL = MT_params2_LL.slope;
        double const2_RL = MT_params2_RL.constant;
        double slope2_RL = MT_params2_RL.slope;
        double const2_LR = MT_params2_LR.constant;
        double slope2_LR = MT_params2_LR.slope;
        double const2_RR = MT_params2_RR.constant;
        double slope2_RR = MT_params2_RR.slope;
        //----------------------------------------


        // evaluate interpolation points for parent 1
        //----------------------------------------
        double logdN1_LL = const1_LL + slope1_LL * MT;
        double logdN1_LR = const1_LR + slope1_LR * MT;
        double logdN1_RL = const1_RL + slope1_RL * MT;
        double logdN1_RR = const1_RR + slope1_RR * MT;
        //----------------------------------------


        // evaluate interpolation points for parent 2
        //----------------------------------------
        double logdN2_LL = const2_LL + slope2_LL * MT;
        double logdN2_LR = const2_LR + slope2_LR * MT;
        double logdN2_RL = const2_RL + slope2_RL * MT;
        double logdN2_RR = const2_RR + slope2_RR * MT;
        //----------------------------------------


        // bi-linear interpolation for log parent 1
        //----------------------------------------
        logdN1 = (logdN1_LL * (YR - Y) + logdN1_RL * (Y - YL)) * (Phip1R - Phip1) +
                 (logdN1_LR * (YR - Y) + logdN1_RR * (Y - YL)) * (Phip1 - Phip1L);

        logdN1 /= (dY * dPhip1);
        //----------------------------------------


        // bi-linear interpolation for log parent 2
        //----------------------------------------
        logdN2 = (logdN2_LL * (YR - Y) + logdN2_RL * (Y - YL)) * (Phip2R - Phip2) +
                 (logdN2_LR * (YR - Y) + logdN2_RR * (Y - YL)) * (Phip2 - Phip2L);

        logdN2 /= (dY * dPhip2);
        //----------------------------------------
    }
    return (exp(logdN1) + exp(logdN2));    // undo the log
}




void EmissionFunctionArray::do_resonance_decays(particle_info * particle_data)
  {
    printline();
    printf("Starting resonance decays: \n\n");
    printf("I need to change the linear interpolation's MTmax to MTswitch or the last MT point when the distribution is positive!");
    exit(-1);
    Stopwatch sw;
    sw.tic();

    const int number_of_particles = Nparticles; // total number of particles in pdg.dat (includes leptons, photons, antibaryons)

    if(number_of_chosen_particles - 1 <= 0)
    {
        printf("\nError: need at least two chosen particles for resonance decay routine..\n");
        exit(-1);
    }


    // start the resonance decay feed-down, starting with the last chosen resonance particle:
    for(int ichosen = (number_of_chosen_particles - 1); ichosen > 0; ichosen--)
    {
        // chosen particle index in particle_data array of structs
        int ipart = chosen_particles_sampling_table[ichosen];
        int stable = particle_data[ipart].stable;

        // if particle unstable under strong interactions, do resonance decay
        if(!stable)
        {
            // parent resonance info:
            int parent_index = ipart;
            int parent_chosen_index = ichosen;
            int parent_decay_channels = particle_data[ipart].decays;

            // set parent's log distribution for linear interpolation
            int y_pts = y_tab_length;
            if(DIMENSION == 2) y_pts = 1;
            for(int ipT = 0; ipT < pT_tab_length; ipT++)
            {
                for(int iphip = 0; iphip < phi_tab_length; iphip++)
                {
                    for(int iy = 0; iy < y_pts; iy++)
                    {
                        long long int iS3D = parent_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

                        long int iS_parent = ipT + pT_tab_length * (iphip + phi_tab_length * iy); // should be large enough...
                        // don't worry about log nans, skip over them with MT fit
                        logdN_PTdPTdPhidY[iS_parent] = log(dN_pTdpTdphidy[iS3D]);
                        //logdN_PTdPTdPhidY[iS_parent] = 0.0;
                    }
                }
            }

            // go through each decay channel of parent resonance
            for(int ichannel = 0; ichannel < parent_decay_channels; ichannel++)
            {
                // why is this number negative sometimes?
                int decay_products = abs(particle_data[ipart].decays_Npart[ichannel]);

                // set up vector that holds (pdg) particle indices of real daughters
                vector<int> decays_index_vector;

                for(int idaughter = 0; idaughter < decay_products; idaughter++)
                {
                    int daughter_mc_id = particle_data[ipart].decays_part[ichannel][idaughter];
                    int daughter_index = particle_index(particle_data, number_of_particles, daughter_mc_id);
                    decays_index_vector.push_back(daughter_index);

                } // finished setting decays_index_vector

                // only do non-trivial resonance decays
                if(decay_products != 1)
                {
                  resonance_decay_channel(particle_data, ichannel, parent_index, parent_chosen_index, decays_index_vector);
                }
            }
            //printf("\n");
        } //

        cout << "\r" << number_of_chosen_particles - ichosen << " / " << number_of_chosen_particles - 1 << " resonances finished" << flush;
    }
    sw.toc();

    printf("\n\nResonance decays took %f seconds.\n", sw.takeTime());
  }


void EmissionFunctionArray::resonance_decay_channel(particle_info * particle_data, int channel, int parent_index, int parent_chosen_index, vector<int> decays_index_vector)
{
    // decay channel info:
    int parent = parent_index;
    string parent_name = particle_data[parent].name;
    int decay_products = decays_index_vector.size();
    double branch_ratio = particle_data[parent].decays_branchratio[channel];

    // print the decay channel
    // cout << setprecision(3) << branch_ratio * 100.0 << "% of " << parent_name << "s decays to\t";
    // for(int idecay = 0; idecay < decay_products; idecay++)
    // {
    //   cout << particle_data[decays_index_vector[idecay]].mc_id << "\t";
    // }

    switch(decay_products)
    {
        case 2: // 2-body decay
        {
            // for testing 3 - body decays:
            //break;

            // particle index of decay products
            int particle_1 = decays_index_vector[0];
            int particle_2 = decays_index_vector[1];

            // masses involved in 2-body decay
            double mass_parent = particle_data[parent].mass;
            double mass_1 = particle_data[particle_1].mass;
            double mass_2 = particle_data[particle_2].mass;

            // adjust the masses to satisfy energy conservation (I need to move this in two_body_decay)
            bool adjust = false;
            while((mass_1 + mass_2) > mass_parent)
            {
                if(!adjust)
                {
                    //printf("\tAdjusting masses\t");
                    adjust = true;
                }
                mass_parent += 0.25 * particle_data[parent].width;
                mass_1 -= 0.5 * particle_data[particle_1].width;
                mass_2 -= 0.5 * particle_data[particle_2].width;
                if(mass_1 < 0.0 || mass_2 < 0.0)
                {
                    printf("One daughter mass went negative: stop");
                    exit(-1);
                }
            }

            // 2-body decay integration routine
            two_body_decay(particle_data, branch_ratio, parent, parent_chosen_index, particle_1, particle_2, mass_1, mass_2, mass_parent);

            break;
        }
        case 3: // 3-body decay
        {
            // for testing 2-body decay only
            //break;
            // particle index of decay products
            int particle_1 = decays_index_vector[0];
            int particle_2 = decays_index_vector[1];
            int particle_3 = decays_index_vector[2];

            // parent mass 3-body decay
            double mass_parent = particle_data[parent].mass;

            // 3-body decay integration routine
            three_body_decay(particle_data, branch_ratio, parent, parent_chosen_index, particle_1, particle_2, particle_3, mass_parent);

            break;
        }
        case 4:
        {
          break;
        }
        default:
        {
          printf ("Error: number of decay products = 1 or > 4\n");
          exit(-1);
        }
    }
    //printf("\n");
}



void EmissionFunctionArray::two_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, double mass_1, double mass_2, double mass_parent)
{
    double two_Pi = 2.0 * M_PI;

    // original list of decay products
    int number_of_decay_particles = 2;
    int decay_product_list[2] = {particle_1, particle_2};

    // first select decay products that are part of chosen resonance particles
    // TODO: distinguish resonance table from chosen particles table whose final spectra we're interested in

    bool found_particle_1 = false;
    bool found_particle_2 = false;

    vector<int> selected_particles;     // holds indices of selected particles

    // loop through the chosen resonance particles table and check for decay product matches
    for(int ichosen = 0; ichosen < number_of_chosen_particles; ichosen++)
    {
        int chosen_index = chosen_particles_sampling_table[ichosen];

        if((particle_1 == chosen_index) && (!found_particle_1))
        {
            found_particle_1 = true;
        }
        if((particle_2 == chosen_index) && (!found_particle_2))
        {
            found_particle_2 = true;
        }
        if(found_particle_1 && found_particle_2)
        {
            break;
        }
    }

    if(found_particle_1) selected_particles.push_back(particle_1);
    if(found_particle_2) selected_particles.push_back(particle_2);

    int number_of_selected_particles = selected_particles.size();

    if(number_of_selected_particles == 0)
    {
        //printf("Zero decays particles found in resonance table: skip integration routine");
        return;
    }

    // group selected particles by type:
    vector<int> particle_groups;    // group particle indices
    vector<int> group_members;      // group members

    while(selected_particles.size() > 0)
    {
        int particle = selected_particles[0];
        int current_groups = particle_groups.size();
        bool put_particle_in_current_groups = false;

        // loop through current groups
        for(int igroup = 0; igroup < current_groups; igroup++)
        {
            if(particle == particle_groups[igroup])
            {
                // if match, add particle to that group
                group_members[igroup] += 1;
                put_particle_in_current_groups = true;
                break;
            }
        }
        if(!put_particle_in_current_groups)
        {
            // make a new group
            particle_groups.push_back(particle);
            group_members.push_back(1);
        }
        // remove particle from original list and look at the next particle
        selected_particles.erase(selected_particles.begin());
    } // while loop until all particles are grouped

    int groups = particle_groups.size();

    // set particle's chosen particle index, mass, energy_star and momentum_star in each group
    int chosen_index[groups];
    double mass_squared[groups];
    double momentum_star[groups];
    double energy_star[groups];

    for(int igroup = 0; igroup < groups; igroup++)
    {
        // set the chosen particle index (for iS3D index)
        chosen_index[igroup] = particle_chosen_index(particle_groups[igroup]);
        //cout << chosen_index[igroup] << endl;
        // set the mass
        double mass = particle_data[particle_groups[igroup]].mass;

        mass_squared[igroup] = mass * mass;

        int particle_index = particle_groups[igroup];

        // make a vector copy of original decay products list
        vector<int> decay_products_vector;
        for(int k = 0; k < number_of_decay_particles; k++)
        {
            decay_products_vector.push_back(decay_product_list[k]);
        }
        // search for a match and remove the particle from the vector copy
        for(int k = 0; k < number_of_decay_particles; k++)
        {
            int decay_particle = decay_products_vector[k];
            if(decay_particle == particle_index)
            {
                decay_products_vector.erase(decay_products_vector.begin() + k);
                break;
            }
        }
        // get W2 = invariant mass squared of the secondary particle
        int particle_secondary = decay_products_vector[0];
        double mass_secondary = particle_data[particle_2].mass;
        double W2 = mass_secondary * mass_secondary;

        // set the energy_star and momentum_star of particle of interest
        double Estar = (mass_parent * mass_parent + mass * mass - W2) / (2.0 * mass_parent);
        energy_star[igroup] = Estar;
        momentum_star[igroup] = sqrt(Estar * Estar - mass * mass);
    }

    // print the groups (to test grouping)
    // printf("Selected particle groups: ");
    // for(int igroup = 0; igroup < groups; igroup++)
    // {
    //     printf("(%d,%d)\t", particle_data[particle_groups[igroup]].mc_id, group_members[igroup]);
    // }


    // set momentum arrays:
    //---------------------------------------
    int y_pts = y_tab_length;
    if(DIMENSION == 2) y_pts = 1;

    double yValues[y_pts];              // momentum points of the spectra data
    double pTValues[pT_tab_length];
    double phipValues[phi_tab_length];
    double MTValues[pT_tab_length];     // MT points of the parent resonance

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
    }
    else if(DIMENSION == 3)
    {
      for(int iy = 0; iy < y_pts; iy++)
      {
         yValues[iy] = y_tab->get(1, iy + 1);
      }
    }
    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
        double pT = pT_tab->get(1, ipT + 1);
        pTValues[ipT] = pT;
        MTValues[ipT] = sqrt(fabs(pT * pT + mass_parent * mass_parent));
    }
    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
        phipValues[iphip] = phi_tab->get(1, iphip + 1);
    }

    // maximum ranges of parent spectra in Cooper Frye data
    double Phip_min = phipValues[0];
    double Phip_max = phipValues[phi_tab_length - 1];
    double Ymax = fabs(yValues[y_pts - 1]);
    double MTmax = MTValues[pT_tab_length - 1];
    //---------------------------------------
    // finished setting momentum arrays


    // set the roots/weights for the two-body decay double integral

    // Gauss Legendre roots / weights for (v,zeta) integrals:
    const int gauss_pts = 12;
    double gaussLegendre_root[gauss_pts] = {-0.98156063424672, -0.90411725637048, -0.76990267419431, -0.58731795428662, -0.3678314989982, -0.12523340851147,
   0.12523340851147, 0.36783149899818, 0.58731795428662, 0.76990267419431, 0.90411725637048, 0.98156063424672};

    double gaussLegendre_weight[gauss_pts] = {0.04717533638651, 0.1069393259953, 0.16007832854335, 0.20316742672307, 0.23349253653836, 0.2491470458134,
   0.2491470458134, 0.23349253653836, 0.20316742672307, 0.1600783285433, 0.10693932599532, 0.04717533638651};

    double v_root[gauss_pts];
    double v_weight[gauss_pts];
    double coszeta_root[gauss_pts];
    double zeta_weight[gauss_pts];

    for(int i = 0; i < gauss_pts; i++)
    {
        v_root[i] = gaussLegendre_root[i];
        v_weight[i] = gaussLegendre_weight[i];
        // since domain of zeta = [0,pi] use coordinate transform zeta = (1+x)pi/2, where x = [-1,1]
        coszeta_root[i] = cos((M_PI / 2.0) * (1.0 + gaussLegendre_root[i]));    // cos(zeta)
        zeta_weight[i] = gaussLegendre_weight[i];
    }


    // Extrapolate the parent dN_dymTdmTdphi distribution at large mT
    // using the fit function y = exp(constant + slope * mT)
    // The constant and slope parameters are estimated using a least squares fit
    // Alternative #2: extrapolate with last 2 MT points

    MT_fit_parameters ** MT_params = (MT_fit_parameters**)calloc(y_pts,sizeof(MT_fit_parameters*));
    for(int iy = 0; iy < y_pts; iy++) MT_params[iy] = (MT_fit_parameters*)calloc(phi_tab_length,sizeof(MT_fit_parameters));

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
        for(int iy = 0; iy < y_pts; iy++)
        {
            MT_params[iy][iphip] = estimate_MT_function_of_dNdypTdpTdphi(iy, iphip, mass_parent);
        }
    }


    // two-body decay integration:
    //---------------------------------------
    switch(DIMENSION)
    {
        case 2: // boost invariant case
        {
            // loop over particle groups
            for(int igroup = 0; igroup < groups; igroup++)
            {
                // particle index, chosen index, and multiplicity
                int particle_index = particle_groups[igroup];
                int particle_chosen_index = chosen_index[igroup];
                double multiplicity = (double)group_members[igroup];

                double parent_mass2 = mass_parent * mass_parent;

                // particle mass, energy_star and momentum_star in parent rest frame:
                double mass2 = mass_squared[igroup];
                double Estar = energy_star[igroup];
                double Estar2 = Estar * Estar;
                double pstar = momentum_star[igroup];

                // useful expression
                double Estar_M = Estar * mass_parent;

                // prefactor of the integral (factor of pi/2 from the zeta -> x coordinate transformation built in)
                double prefactor = multiplicity * mass_parent * branch_ratio / (8.0 * pstar);

                // loop over momentum
                for(int ipT = 0; ipT < pT_tab_length; ipT++)
                {
                    double pT = pTValues[ipT];  // particle transverse momentum

                    double pT2 = pT * pT;
                    double mT2 = pT2 + mass2;
                    double mT = sqrt(mT2);

                    // useful expressions
                    double M_pT = mass_parent * pT;
                    double Estar_M_mT = Estar_M * mT;
                    double Estar_M_over_pT = Estar_M / pT;
                    double Estar2_plus_pT2 = Estar2 + pT2;

                    // parent rapdity interval
                    double DeltaY = log((pstar + sqrt(Estar2_plus_pT2)) / mT);

                    // useful tables
                    double coshvDeltaY_table[gauss_pts];
                    double coshvDeltaY2_table[gauss_pts];
                    double MTbar_table[gauss_pts];
                    double DeltaMT_table[gauss_pts];
                    double mT_coshvDeltaY_over_pT_table[gauss_pts];
                    double vintegrand_weight_table[gauss_pts];

                    for(int k = 0; k < gauss_pts; k++)
                    {
                        double v = v_root[k];
                        double coshvDeltaY = cosh(v * DeltaY);
                        double mT2_coshvDeltaY2 = mT2 * coshvDeltaY * coshvDeltaY;
                        double mT2_coshvDeltaY2_minus_pT2 = mT2_coshvDeltaY2 - pT2;

                        // set tables:
                        MTbar_table[k] = Estar_M_mT * coshvDeltaY / mT2_coshvDeltaY2_minus_pT2;
                        DeltaMT_table[k] = M_pT * sqrt(fabs(Estar2_plus_pT2 - mT2_coshvDeltaY2)) / mT2_coshvDeltaY2_minus_pT2;
                        mT_coshvDeltaY_over_pT_table[k] = mT * coshvDeltaY / pT;
                        vintegrand_weight_table[k] = DeltaY * v_weight[k] / sqrt(fabs(mT2_coshvDeltaY2_minus_pT2));
                    } // tables

                    //for(int iphip = 0; iphip < 1; iphip++)
                    for(int iphip = 0; iphip < phi_tab_length; iphip++)
                    {
                        double phip = phipValues[iphip];    // particle azimuthal angle

                        double decay2D_integral = 0.0;

                        // do the decay2D_integral over parent rapidity, transverse mass space (v, zeta)
                        for(int iv = 0; iv < gauss_pts; iv++)
                        {
                            // double v = v_root[iv];
                            // double coshvDeltaY = cosh(v * DeltaY);
                            // double coshvDeltaY2 = coshvDeltaY * coshvDeltaY;
                            // double mT2_coshvDeltaY2_minus_pT2 = (mT2 * coshvDeltaY2) - pT2;


                            // double MTbar = Estar_M_mT * coshvDeltaY / mT2_coshvDeltaY2_minus_pT2;
                            // double DeltaMT = M_pT * sqrt(fabs(Estar2_plus_pT2 - mT2 * coshvDeltaY2)) / mT2_coshvDeltaY2_minus_pT2;
                            // double mT_coshvDeltaY_over_pT = mT * coshvDeltaY / pT;
                            // double vintegrand_weight = DeltaY * v_weight[iv] / sqrt(fabs(mT2_coshvDeltaY2_minus_pT2));

                            double MTbar = MTbar_table[iv];
                            double DeltaMT = DeltaMT_table[iv];
                            double mT_coshvDeltaY_over_pT = mT_coshvDeltaY_over_pT_table[iv];
                            double vintegrand_weight = vintegrand_weight_table[iv];

                            double zeta_integral = 0.0;

                            for(int izeta = 0; izeta < gauss_pts; izeta++)
                            {
                                double coszeta = coszeta_root[izeta];
                                double zeta_gauss_weight = zeta_weight[izeta];

                                double MT = MTbar + (DeltaMT * coszeta);    // parent MT
                                double PT = sqrt(MT * MT - parent_mass2);
                                double cosPhip_tilde = (MT * mT_coshvDeltaY_over_pT - Estar_M_over_pT) / PT;

                                // if(fabs(cosPhip_tilde) >= 1.0)
                                // {

                                //     printf("\nError: parent azimuthal angle has no solution\n");
                                //     exit(-1);
                                // }

                                double Phip_tilde = acos(cosPhip_tilde);

                                // force two solutions for the parent azimuthal angle between [0,2pi)
                                double Phip_1 = fmod(Phip_tilde + phip, two_Pi);
                                double Phip_2 = fmod(-Phip_tilde + phip, two_Pi);
                                if(Phip_1 < 0.0) Phip_1 += two_Pi;
                                if(Phip_2 < 0.0) Phip_2 += two_Pi;

                                double integrand = MT * dN_dYMTdMTdPhi_boost_invariant(parent_chosen_index, MTValues, phipValues, MT, Phip_1, Phip_2, Phip_min, Phip_max, MTmax, MT_params);

                                // do something easy:
                                //double integrand = 2.0 * MT * exp(4.88706 - 2.04108 * MT); // this worked
                                zeta_integral += (zeta_gauss_weight * integrand);
                            }

                            decay2D_integral += (vintegrand_weight * zeta_integral);
                        }
                        //cout << setprecision(8) << prefactor * decay2D_integral << endl;
                        // amend the spectra:
                        long long int iS3D = particle_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * iphip);
                        dN_pTdpTdphidy[iS3D] += prefactor * decay2D_integral;
                    } // iphip
                } // ipT
            } // igroups
            break;
        } // case 2
        case 3: // non boost invariant case
        {
            // loop over particle groups
            for(int igroup = 0; igroup < groups; igroup++)
            {
                // particle index, chosen index and multiplicity
                int particle_index = particle_groups[igroup];
                int particle_chosen_index = chosen_index[igroup];
                double multiplicity = (double)group_members[igroup];

                double parent_mass2 = mass_parent * mass_parent;

                // particle mass, energy_star and momentum_star in parent rest frame:
                double mass2 = mass_squared[igroup];
                double Estar = energy_star[igroup];
                double Estar2 = Estar * Estar;
                double pstar = momentum_star[igroup];

                // useful expression
                double Estar_M = Estar * mass_parent;

                // prefactor of the integral (factor of pi/2 from the zeta -> x coordinate transformation built in)
                double prefactor = multiplicity * mass_parent * branch_ratio / (8.0 * pstar);

                // loop over momentum
                for(int ipT = 0; ipT < pT_tab_length; ipT++)
                {
                    double pT = pTValues[ipT];  // particle transverse momentum

                    double pT2 = pT * pT;
                    double mT2 = pT2 + mass2;
                    double mT = sqrt(mT2);

                    // useful expressions
                    double M_pT = mass_parent * pT;
                    double Estar_M_mT = Estar_M * mT;
                    double Estar_M_over_pT = Estar_M / pT;
                    double Estar2_plus_pT2 = Estar2 + pT2;

                    // parent rapdity interval
                    double DeltaY = log((pstar + sqrt(Estar2_plus_pT2)) / mT);

                    // useful tables
                    double coshvDeltaY_table[gauss_pts];
                    double coshvDeltaY2_table[gauss_pts];
                    double MTbar_table[gauss_pts];
                    double DeltaMT_table[gauss_pts];
                    double mT_coshvDeltaY_over_pT_table[gauss_pts];
                    double vintegrand_weight_table[gauss_pts];

                    for(int k = 0; k < gauss_pts; k++)
                    {
                        double v = v_root[k];
                        double coshvDeltaY = cosh(v * DeltaY);
                        double mT2_coshvDeltaY2 = mT2 * coshvDeltaY * coshvDeltaY;
                        double mT2_coshvDeltaY2_minus_pT2 = mT2_coshvDeltaY2 - pT2;

                        // set tables:
                        MTbar_table[k] = Estar_M_mT * coshvDeltaY / mT2_coshvDeltaY2_minus_pT2;
                        DeltaMT_table[k] = M_pT * sqrt(fabs(Estar2_plus_pT2 - mT2_coshvDeltaY2)) / mT2_coshvDeltaY2_minus_pT2;
                        mT_coshvDeltaY_over_pT_table[k] = mT * coshvDeltaY / pT;
                        vintegrand_weight_table[k] = DeltaY * v_weight[k] / sqrt(fabs(mT2_coshvDeltaY2_minus_pT2));
                    } // tables

                    //for(int iphip = 0; iphip < 1; iphip++)
                    for(int iphip = 0; iphip < phi_tab_length; iphip++)
                    {
                        double phip = phipValues[iphip];    // particle azimuthal angle

                        for(int iy = 0; iy < y_pts; iy++)
                        {
                            double y = yValues[iy];         // particle rapidity

                            double decay2D_integral = 0.0;

                            // do the decay2D_integral over parent rapidity, transverse mass space (v, zeta)
                            for(int iv = 0; iv < gauss_pts; iv++)
                            {
                                double v = v_root[iv];
                                double Y = y + v * DeltaY;  // parent rapidity

                                // search for Y interpolation points here
                                int iYR = 1;
                                int iYL;
                                double YL;
                                double YR;

                                bool cutoff_Y = false;
                                if(fabs(Y) <= Ymax)
                                {
                                    while(Y > yValues[iYR])
                                    {
                                        iYR++;
                                    }
                                    iYL = iYR - 1;
                                    YL = yValues[iYL];
                                    YR = yValues[iYR];
                                }
                                else
                                {
                                    cutoff_Y = true;
                                }
                                //-------------------------------

                                double MTbar = MTbar_table[iv];
                                double DeltaMT = DeltaMT_table[iv];
                                double mT_coshvDeltaY_over_pT = mT_coshvDeltaY_over_pT_table[iv];
                                double vintegrand_weight = vintegrand_weight_table[iv];

                                double zeta_integral = 0.0;

                                for(int izeta = 0; izeta < gauss_pts; izeta++)
                                {
                                    if(cutoff_Y) break;   // parent distribution cutoff in Y

                                    double coszeta = coszeta_root[izeta];
                                    double zeta_gauss_weight = zeta_weight[izeta];

                                    double MT = MTbar + (DeltaMT * coszeta);    // parent MT
                                    double PT = sqrt(MT * MT - parent_mass2);
                                    double cosPhip_tilde = (MT * mT_coshvDeltaY_over_pT - Estar_M_over_pT) / PT;

                                    // if(fabs(cosPhip_tilde) >= 1.0)
                                    // {

                                    //     printf("\nError: parent azimuthal angle has no solution\n");
                                    //     exit(-1);
                                    // }

                                    double Phip_tilde = acos(cosPhip_tilde);

                                    // two solutions for the parent azimuthal angle = [0,2pi)
                                    double Phip_1 = fmod(Phip_tilde + phip, two_Pi);
                                    double Phip_2 = fmod(-Phip_tilde + phip, two_Pi);
                                    if(Phip_1 < 0.0) Phip_1 += two_Pi;
                                    if(Phip_2 < 0.0) Phip_2 += two_Pi;

                                    double integrand = MT * dN_dYMTdMTdPhi_non_boost_invariant(parent_chosen_index, MTValues, phipValues, iYL, iYR, YL, YR, MT, Phip_1, Phip_2, Y, Phip_min, Phip_max, MTmax, MT_params);

                                    zeta_integral += (zeta_gauss_weight * integrand);
                                } // izeta

                                decay2D_integral += (vintegrand_weight * zeta_integral);
                            } // iv
                            //cout << setprecision(8) << prefactor * decay2D_integral << endl;
                            long long int iS3D = particle_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

                            dN_pTdpTdphidy[iS3D] += prefactor * decay2D_integral;
                        } // iy
                    } // iphip
                } // ipT
            } // igroups
            break;
        } // case 3
        default:
        {
            printf("\nError: specify boost-invariant 2+1d or 3+1d\n");
            exit(-1);
        }
    }
    //---------------------------------------
    // finished two-body decay routine
    //free_2D(MT_params);
}






void EmissionFunctionArray::three_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, int particle_3, double mass_parent)
{
    // first select decay products that are part of chosen resonance particles
    // TODO: distinguish resonance table from chosen particles table whose final spectra we're interested in
    double two_Pi = 2.0 * M_PI;

     // original list of decay products
    int number_of_decay_particles = 3;
    int decay_product_list[3] = {particle_1, particle_2, particle_3};

    bool found_particle_1 = false;
    bool found_particle_2 = false;
    bool found_particle_3 = false;

    vector<int> selected_particles;     // holds indices of selected particles

    // loop through the chosen resonance particles table and check for decay product matches
    for(int ichosen = 0; ichosen < number_of_chosen_particles; ichosen++)
    {
        int chosen_pdg_index = chosen_particles_sampling_table[ichosen];

        if((particle_1 == chosen_pdg_index) && (!found_particle_1))
        {
            found_particle_1 = true;
        }
        if((particle_2 == chosen_pdg_index) && (!found_particle_2))
        {
            found_particle_2 = true;
        }
        if((particle_3 == chosen_pdg_index) && (!found_particle_3))
        {
            found_particle_3 = true;
        }
        if(found_particle_1 && found_particle_2 && found_particle_3)
        {
            break;
        }
    }

    if(found_particle_1) selected_particles.push_back(particle_1);
    if(found_particle_2) selected_particles.push_back(particle_2);
    if(found_particle_3) selected_particles.push_back(particle_3);

    int number_of_selected_particles = selected_particles.size();

    if(selected_particles.size() == 0)
    {
        //printf("Zero decays particles found in resonance table: skip integration routine");
        return;
    }

     // group selected particles by type
    vector<int> particle_groups;    // group particle indices
    vector<int> group_members;      // group members

    while(selected_particles.size() > 0)
    {
        int particle = selected_particles[0];
        int current_groups = particle_groups.size();
        bool put_particle_in_current_groups = false;

        // loop through current groups
        for(int igroup = 0; igroup < current_groups; igroup++)
        {
            if(particle == particle_groups[igroup])
            {
                // if match, add particle to that group
                group_members[igroup] += 1;
                put_particle_in_current_groups = true;
                break;
            }
        }
        if(!put_particle_in_current_groups)
        {
            // make a new group
            particle_groups.push_back(particle);
            group_members.push_back(1);
        }
        // remove particle from original list and look at the next particle
        selected_particles.erase(selected_particles.begin());
    } // repeat until all particles are grouped

    int groups = particle_groups.size();


    // set particle's chosen particle index and mass_1
    // and masses of remaining particles 2 and 3, and the Q factor
    int chosen_index[groups];
    double mass_1_group[groups];
    double mass_2_group[groups];
    double mass_3_group[groups];
    double Q_group[groups];



    for(int igroup = 0; igroup < groups; igroup++)
    {
        // set the pdg index
        int pdg_index = particle_groups[igroup];

        // set the chosen particle index (for iS3D index)
        chosen_index[igroup] = particle_chosen_index(pdg_index);

        // set the mass of particle type's mass in group
        mass_1_group[igroup] = particle_data[pdg_index].mass;

        // make a vector copy of original decay products list
        vector<int> decay_products_vector;
        for(int k = 0; k < number_of_decay_particles; k++)
        {
            decay_products_vector.push_back(decay_product_list[k]);
        }
        // search for a match and remove the particle from the vector copy
        for(int k = 0; k < number_of_decay_particles; k++)
        {
            int decay_particle = decay_products_vector[k];
            if(decay_particle == pdg_index)
            {
                decay_products_vector.erase(decay_products_vector.begin() + k);
                break;
            }
        }
        // get masses of second and third particle
        int particle_2_pdg_index = decay_products_vector[0];
        int particle_3_pdg_index = decay_products_vector[1];

        mass_2_group[igroup] = particle_data[particle_2_pdg_index].mass;
        mass_3_group[igroup] = particle_data[particle_3_pdg_index].mass;

        double mass_1 = mass_1_group[igroup];
        double mass_2 = mass_2_group[igroup];
        double mass_3 = mass_3_group[igroup];

        Q_group[igroup] = calculate_Q_factor(mass_parent, mass_1, mass_2, mass_3);
    }

    // printf("Selected particle groups: ");
    // for(int igroup = 0; igroup < groups; igroup++)
    // {
    //     //printf("(%d, %d, %f, %f, %f, %f)\t", particle_data[particle_groups[igroup]].mc_id, group_members[igroup], mass_1_group[igroup], mass_2_group[igroup], mass_3_group[igroup], Q_group[igroup]);
    //     printf("(%d,%d)\t", particle_data[particle_groups[igroup]].mc_id, group_members[igroup]);
    // }


    // set momentum arrays:
    //---------------------------------------
    int y_pts = y_tab_length;
    if(DIMENSION == 2) y_pts = 1;

    double yValues[y_pts];              // momentum points of the spectra data
    double pTValues[pT_tab_length];
    double phipValues[phi_tab_length];
    double MTValues[pT_tab_length];     // MT points of the parent resonance

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
    }
    else if(DIMENSION == 3)
    {
      for(int iy = 0; iy < y_pts; iy++)
      {
         yValues[iy] = y_tab->get(1, iy + 1);
      }
    }
    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
        double pT = pT_tab->get(1, ipT + 1);
        pTValues[ipT] = pT;
        MTValues[ipT] = sqrt(fabs(pT * pT + mass_parent * mass_parent));
    }
    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
        phipValues[iphip] = phi_tab->get(1, iphip + 1);
    }

    // maximum ranges of parent spectra in Cooper Frye data
    double Phip_min = phipValues[0];
    double Phip_max = phipValues[phi_tab_length - 1];
    double Ymax = fabs(yValues[y_pts - 1]);
    double MTmax = MTValues[pT_tab_length - 1];
    //---------------------------------------
    // finished setting momentum arrays



    // Gauss Legendre roots / weights for (s,v,zeta) integrals:
    const int gauss_pts = 12;
    double gaussLegendre_root[gauss_pts] = {-0.98156063424672, -0.90411725637048, -0.76990267419431, -0.58731795428662, -0.3678314989982, -0.12523340851147,
   0.12523340851147, 0.36783149899818, 0.58731795428662, 0.76990267419431, 0.90411725637048, 0.98156063424672};

    double gaussLegendre_weight[gauss_pts] = {0.04717533638651, 0.1069393259953, 0.16007832854335, 0.20316742672307, 0.23349253653836, 0.2491470458134,
   0.2491470458134, 0.23349253653836, 0.20316742672307, 0.1600783285433, 0.10693932599532, 0.04717533638651};

    double v_root[gauss_pts];
    double v_weight[gauss_pts];
    double coszeta_root[gauss_pts];
    double zeta_weight[gauss_pts];

    double s_weight[gauss_pts];

    for(int i = 0; i < gauss_pts; i++)
    {
        v_root[i] = gaussLegendre_root[i];
        v_weight[i] = gaussLegendre_weight[i];
        // since domain of zeta = [0,pi] use coordinate transform zeta = (1+x)pi/2, where x = [-1,1]
        coszeta_root[i] = cos((M_PI / 2.0) * (1.0 + gaussLegendre_root[i]));    // cos(zeta)
        zeta_weight[i] = gaussLegendre_weight[i];

        // s weights (s_root done below)
        s_weight[i] = gaussLegendre_weight[i];
    }


    // Extrapolate the parent dN_dymTdmTdphi distribution at large mT
    // using the fit function y = exp(constant + slope * mT)
    // The constant and slope parameters are estimated using a least squares fit
    // Alternative #2: extrapolate with last 2 MT points

    MT_fit_parameters ** MT_params = (MT_fit_parameters**)calloc(y_pts,sizeof(MT_fit_parameters*));
    for(int iy = 0; iy < y_pts; iy++) MT_params[iy] = (MT_fit_parameters*)calloc(phi_tab_length,sizeof(MT_fit_parameters));

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
        for(int iy = 0; iy < y_pts; iy++)
        {
            MT_params[iy][iphip] = estimate_MT_function_of_dNdypTdpTdphi(iy, iphip, mass_parent);
        }
    }


    // three-body decay integration (next work on non-boost invariant case)
    //---------------------------------------
    switch(DIMENSION)
    {
        case 2: // boost invariant case
        {
            // i could make an array that holds constant and slope separately


            // loop over particle groups
            for(int igroup = 0; igroup < groups; igroup++)
            {
                // particle of interest's chosen index and multiplicity
                int particle_chosen_index = chosen_index[igroup];
                double multiplicity = (double)group_members[igroup];

                // parent info
                double parent_mass2 = mass_parent * mass_parent;

                // mass of the particles
                double mass_1 = mass_1_group[igroup];   // particle of interest
                double mass_1_squared = mass_1 * mass_1;
                double mass_2 = mass_2_group[igroup];   // remaining particles
                double mass_3 = mass_3_group[igroup];

                // g(s) and Q_norm info
                double s_plus = (mass_parent - mass_1) * (mass_parent - mass_1);    // b
                double s_minus = (mass_2 + mass_3) * (mass_2 + mass_3);             // c
                double d = (mass_2 - mass_3) * (mass_2 - mass_3);                   // d
                double Q_norm = Q_group[igroup];

                // set up s-roots:
                double s_root[gauss_pts];
                double Estar_table[gauss_pts];
                double pstar_table[gauss_pts];
                double s_integrand_weight_table[gauss_pts];

                for(int k = 0; k < gauss_pts; k++)
                {
                    double s = s_minus + (s_plus - s_minus) * (1.0 + gaussLegendre_root[k]) / 2.0;
                    double weight = s_weight[k];
                    double Estar = (parent_mass2 + mass_1_squared - s) / (2.0 * mass_parent);
                    s_root[k] = s;
                    Estar_table[k] = Estar;
                    s_integrand_weight_table[k] = weight * sqrt(fabs((s - s_minus) * (s - d))) / s;
                    pstar_table[k] = sqrt(Estar * Estar - mass_1_squared);
                }

                // prefactor of the integral:
                // contains factors of (s+ - s-)/2 from s -> x1 and pi/2 from zeta -> x2 transformations
                double prefactor = multiplicity * parent_mass2 * (s_plus - s_minus) * branch_ratio / (8.0 * Q_norm);

                // loop over momentum:
                for(int ipT = 0; ipT < pT_tab_length; ipT++)
                {
                    // particle transverse momentum
                    double pT = pTValues[ipT];

                    double pT2 = pT * pT;
                    double mT2 = pT2 + mass_1_squared;
                    double mT = sqrt(mT2);

                    // useful expressions
                    double M_pT = mass_parent * pT;
                    double M_mT = mass_parent * mT;
                    double M_over_pT = mass_parent / pT;
                    double mT_over_pT = mT / pT;

                    // DeltaY(s) table
                    double DeltaY_table[gauss_pts];
                    for(int k = 0; k < gauss_pts; k++)
                    {
                        double pstar = pstar_table[k];
                        double Estar = Estar_table[k];
                        DeltaY_table[k] = log((pstar + sqrt(Estar * Estar + pT2)) / mT);
                    }

                    for(int iphip = 0; iphip < phi_tab_length; iphip++)
                    //for(int iphip = 0; iphip < 1; iphip++)
                    {
                        // particle azimuthal angle
                        double phip = phipValues[iphip];

                        double decay3D_integral = 0.0;

                        // do the decay3D_integral over (s, v, zeta)
                        for(int is = 0; is < gauss_pts; is++)
                        {
                            double s = s_root[is];

                            // total s weight
                            //double s_integrand_weight = s_weight[is] * sqrt(fabs((s - s_minus) * (s - d))) / s;
                            double s_integrand_weight = s_integrand_weight_table[is];

                            // particle of interest's energy and momentum
                            // magnitude in parent rest frame
                            //double Estar =  (parent_mass2 + mass_1_squared - s) / (2.0 * mass_parent);
                            double Estar = Estar_table[is];
                            double Estar2 = Estar * Estar;
                            //double pstar = sqrt(Estar2 - mass_1_squared);
                            double pstar = pstar_table[is];

                            // useful expressions
                            double Estar_M_mT = Estar * M_mT;
                            double Estar2_plus_pT2 = Estar2 + pT2;
                            double Estar_M_over_pT = Estar * M_over_pT;

                            // parent rapidity interval
                            //double DeltaY = log((pstar + sqrt(Estar2_plus_pT2)) / mT);
                            double DeltaY = DeltaY_table[is];

                            double v_integral = 0.0;

                            for(int iv = 0; iv < gauss_pts; iv++)
                            {
                                double v = v_root[iv];

                                // useful expressions
                                double coshvDeltaY = cosh(v * DeltaY);
                                double mT2_coshvDeltaY2 = mT2 * coshvDeltaY * coshvDeltaY;
                                double mT2_coshvDeltaY2_minus_pT2 = mT2_coshvDeltaY2 - pT2;
                                double mT_coshvDeltaY_over_pT = mT_over_pT * coshvDeltaY;

                                // MT terms
                                double MTbar = Estar_M_mT * coshvDeltaY / mT2_coshvDeltaY2_minus_pT2;;
                                double DeltaMT =  M_pT * sqrt(fabs(Estar2_plus_pT2 - mT2_coshvDeltaY2)) / mT2_coshvDeltaY2_minus_pT2;

                                // total v weight
                                double v_integrand_weight = DeltaY * v_weight[iv] / sqrt(fabs(mT2_coshvDeltaY2_minus_pT2));

                                double zeta_integral = 0.0;

                                for(int izeta = 0; izeta < gauss_pts; izeta++)
                                {
                                    double coszeta = coszeta_root[izeta];
                                    double zeta_gauss_weight = zeta_weight[izeta];

                                    double MT = MTbar + DeltaMT * coszeta;    // parent MT
                                    double PT = sqrt(MT * MT - parent_mass2);
                                    double cosPhip_tilde = (MT * mT_coshvDeltaY_over_pT - Estar_M_over_pT) / PT;

                                    // if(fabs(cosPhip_tilde) >= 1.0)
                                    // {

                                    //     printf("\nError: parent azimuthal angle has no solution\n");
                                    //     exit(-1);
                                    // }

                                    double Phip_tilde = acos(cosPhip_tilde);

                                    // two solutions for the parent azimuthal angle = [0,2pi)
                                    double Phip_1 = fmod(Phip_tilde + phip, two_Pi);
                                    double Phip_2 = fmod(-Phip_tilde + phip, two_Pi);
                                    if(Phip_1 < 0.0) Phip_1 += two_Pi;
                                    if(Phip_2 < 0.0) Phip_2 += two_Pi;

                                    double integrand = MT * dN_dYMTdMTdPhi_boost_invariant(parent_chosen_index, MTValues, phipValues, MT, Phip_1, Phip_2, Phip_min, Phip_max, MTmax, MT_params);

                                    // do something easy:
                                    //double integrand = 2.0 * MT * exp(4.88706 - 2.04108 * MT); // this worked
                                    zeta_integral += zeta_gauss_weight * integrand;
                                }
                                v_integral += v_integrand_weight * zeta_integral;
                            }
                            decay3D_integral += s_integrand_weight * v_integral;
                        }
                        // amend the spectra:
                        //cout << setprecision(8) << prefactor * decay3D_integral << endl;
                        long long int iS3D = particle_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * iphip);
                        dN_pTdpTdphidy[iS3D] += prefactor * decay3D_integral;
                    } // iphip
                } // ipT
            } // igroups
            break;
        } // case 2
        case 3: // non boost invariant case
        {
            // loop over particle groups
            for(int igroup = 0; igroup < groups; igroup++)
            {
                // particle of interest's chosen index and multiplicity
                int particle_chosen_index = chosen_index[igroup];
                double multiplicity = (double)group_members[igroup];

                // parent info
                double parent_mass2 = mass_parent * mass_parent;

                // mass of the particles
                double mass_1 = mass_1_group[igroup];   // particle of interest
                double mass_1_squared = mass_1 * mass_1;
                double mass_2 = mass_2_group[igroup];   // remaining particles
                double mass_3 = mass_3_group[igroup];

                // g(s) and Q_norm info
                double s_plus = (mass_parent - mass_1) * (mass_parent - mass_1);    // b
                double s_minus = (mass_2 + mass_3) * (mass_2 + mass_3);             // c
                double d = (mass_2 - mass_3) * (mass_2 - mass_3);                   // d
                double Q_norm = Q_group[igroup];

                // set up s-roots:
                double s_root[gauss_pts];
                for(int k = 0; k < gauss_pts; k++)
                {
                    s_root[k] = s_minus + (s_plus - s_minus) * (1.0 + gaussLegendre_root[k]) / 2.0;
                }

                // prefactor of the integral:
                // contains factors of (s+ - s-)/2 from s -> x1 and pi/2 from zeta -> x2 transformations
                double prefactor = multiplicity * parent_mass2 * (s_plus - s_minus) * branch_ratio / (8.0 * Q_norm);

                // loop over momentum:
                for(int ipT = 0; ipT < pT_tab_length; ipT++)
                {
                    // particle transverse momentum
                    double pT = pTValues[ipT];

                    double pT2 = pT * pT;
                    double mT2 = pT2 + mass_1_squared;
                    double mT = sqrt(mT2);

                    // useful expressions
                    double M_pT = mass_parent * pT;
                    double M_mT = mass_parent * mT;
                    double M_over_pT = mass_parent / pT;
                    double mT_over_pT = mT / pT;

                    for(int iphip = 0; iphip < phi_tab_length; iphip++)
                    {
                        // particle azimuthal angle
                        double phip = phipValues[iphip];

                        for(int iy = 0; iy < y_pts; iy++)
                        {
                            // particle rapidity
                            double y = yValues[iy];

                            // reset decay integral to 0
                            double decay3D_integral = 0.0;

                            // do the decay3D_integral over (s, v, zeta)
                            for(int is = 0; is < gauss_pts; is++)
                            {
                                double s = s_root[is];

                                // total s weight
                                double s_integrand_weight = s_weight[is] * sqrt(fabs((s - s_minus) * (s - d))) / s;;

                                // particle of interest's energy and momentum
                                // magnitude in parent rest frame
                                double Estar =  (parent_mass2 + mass_1_squared - s) / (2.0 * mass_parent);
                                double Estar2 = Estar * Estar;
                                double pstar = sqrt(Estar2 - mass_1_squared);

                                // useful expressions
                                double Estar_M_mT = Estar * M_mT;
                                double Estar2_plus_pT2 = Estar2 + pT2;
                                double Estar_M_over_pT = Estar * M_over_pT;

                                // parent rapidity interval
                                double DeltaY = log((pstar + sqrt(Estar2_plus_pT2)) / mT);

                                double v_integral = 0.0;

                                for(int iv = 0; iv < gauss_pts; iv++)
                                {
                                    double v = v_root[iv];

                                    // parent rapidity
                                    double Y = y + v * DeltaY;

                                    // search for Y interpolation points here
                                    int iYR = 1;
                                    int iYL;
                                    double YL;
                                    double YR;

                                    bool cutoff_Y = false;
                                    if(fabs(Y) <= Ymax)
                                    {
                                        while(Y > yValues[iYR])
                                        {
                                            iYR++;
                                        }
                                        iYL = iYR - 1;
                                        YL = yValues[iYL];
                                        YR = yValues[iYR];
                                    }
                                    else
                                    {
                                        cutoff_Y = true;
                                    }
                                    //-------------------------------

                                    // useful expressions
                                    double coshvDeltaY = cosh(v * DeltaY);
                                    double coshvDeltaY2 = coshvDeltaY * coshvDeltaY;
                                    double mT2_coshvDeltaY2_minus_pT2 = mT2 * coshvDeltaY2 - pT2;
                                    double mT_coshvDeltaY_over_pT = mT_over_pT * coshvDeltaY;

                                    // MT terms
                                    double MTbar = Estar_M_mT * coshvDeltaY / mT2_coshvDeltaY2_minus_pT2;;
                                    double DeltaMT =  M_pT * sqrt(fabs(Estar2_plus_pT2 - mT2 * coshvDeltaY2)) / mT2_coshvDeltaY2_minus_pT2;

                                    // total v weight
                                    double v_integrand_weight = DeltaY * v_weight[iv] / sqrt(fabs(mT2_coshvDeltaY2_minus_pT2));

                                    double zeta_integral = 0.0;

                                    for(int izeta = 0; izeta < gauss_pts; izeta++)
                                    {
                                        if(cutoff_Y) break;

                                        double coszeta = coszeta_root[izeta];
                                        double zeta_gauss_weight = zeta_weight[izeta];

                                        double MT = MTbar + (DeltaMT * coszeta);    // parent MT
                                        double PT = sqrt(MT * MT - parent_mass2);
                                        double cosPhip_tilde = (MT * mT_coshvDeltaY_over_pT - Estar_M_over_pT) / PT;

                                        // if(fabs(cosPhip_tilde) >= 1.0)
                                        // {

                                        //     printf("\nError: parent azimuthal angle has no solution\n");
                                        //     exit(-1);
                                        // }

                                        double Phip_tilde = acos(cosPhip_tilde);

                                        // two solutions for the parent azimuthal angle = [0,2pi)
                                        double Phip_1 = fmod(Phip_tilde + phip, two_Pi);
                                        double Phip_2 = fmod(-Phip_tilde + phip, two_Pi);
                                        if(Phip_1 < 0.0) Phip_1 += two_Pi;
                                        if(Phip_2 < 0.0) Phip_2 += two_Pi;

                                        double integrand = MT *  dN_dYMTdMTdPhi_non_boost_invariant(parent_chosen_index, MTValues, phipValues, iYL, iYR, YL, YR, MT, Phip_1, Phip_2, Y, Phip_min, Phip_max, MTmax, MT_params);

                                        // do something easy:
                                        //double integrand = 2.0 * MT * exp(4.88706 - 2.04108 * MT); // this worked
                                        zeta_integral += zeta_gauss_weight * integrand;
                                    } // izeta
                                    v_integral += v_integrand_weight * zeta_integral;
                                } // iv
                                decay3D_integral += s_integrand_weight * v_integral;
                            } // is
                            // amend the spectra
                            long long int iS3D = particle_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
                            dN_pTdpTdphidy[iS3D] += prefactor * decay3D_integral;

                        } // iy
                    } // iphip
                } // ipT
            } // igroups
            break;
        } // case 3
        default:
        {
            printf("\nError: specify boost-invariant 2+1d or 3+1d\n");
            exit(-1);
        }
    }
    //---------------------------------------
    // finished three-body decay routine
}




