#include "viscous_correction.h"
#include <math.h>
#include <iostream>
#include <iomanip> 
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
using namespace std;

// should rename file local_rest_frame.cpp

Milne_Basis::Milne_Basis(double ut, double ux, double uy, double un, double uperp, double utperp, double tau)
{
    Ut = ut;    Uy = uy;
    Ux = ux;    Un = un;

    double sinhL = tau * un / utperp;
    double coshL = ut / utperp;

    Xt = uperp * coshL;         Zt = sinhL;
    Xn = uperp * sinhL / tau;   Zn = coshL / tau;

    Xx = 1.0;   Yx = 0.0;
    Xy = 0.0;   Yy = 1.0;

    if(uperp > 1.e-5) // stops (ux=0)/(uperp=0) nans for freezeout cells with no transverse flow
    {
        Xx = utperp * ux / uperp;   Yx = - uy / uperp;
        Xy = utperp * uy / uperp;   Yy = ux / uperp;
    }
}

void Milne_Basis::test_orthonormality(double tau2)
{
    // test whether the basis vectors are orthonormal to numerical precision

    double U_normal = fabs(Ut * Ut  -  Ux * Ux  -  Uy * Uy  -  tau2 * Un * Un  -  1.0); // U.U - 1
    double X_normal = fabs(Xt * Xt  -  Xx * Xx  -  Xy * Xy  -  tau2 * Xn * Xn  +  1.0); // X.X + 1
    double Y_normal = fabs(- Yx * Yx  -  Yy * Yy  +  1.0);                              // Y.Y + 1
    double Z_normal = fabs(Zt * Zt  -  tau2 * Zn * Zn  +  1.0);                         // Z.Z + 1

    double U_dot_X = fabs(Xt * Ut  -  Xx * Ux  -  Xy * Uy  -  tau2 * Xn * Un);          // U.X
    double U_dot_Y = fabs(- Yx * Ux  -  Yy * Uy);                                       // U.Y
    double U_dot_Z = fabs(Zt * Ut  -  tau2 * Zn * Un);                                  // U.Z

    double X_dot_Y = fabs(- Xx * Yx  -  Xy * Yy);                                       // X.Y
    double X_dot_Z = fabs(Xt * Zt  -  tau2 * Xn * Zn);                                  // X.Z
    // Y.Z = 0 automatically (no components to multiply..)                              // Y.Z

    double U_orthogonal = max(U_dot_X, max(U_dot_Y, U_dot_Z));
    double X_orthogonal = max(X_dot_Y, X_dot_Z);

    double epsilon = 1.e-14;    // best order of magnitude I could get, why?...

    if(U_normal > epsilon) printf("U is not normalized to 1 (%.6g)\n", U_normal);
    if(X_normal > epsilon) printf("X is not normalized to -1 (%.6g)\n", X_normal);
    if(Y_normal > epsilon) printf("Y is not normalized to -1 (%.6g)\n", Y_normal);
    if(Z_normal > epsilon) printf("Z is not normalized to -1 (%.6g)\n", Z_normal);
    if(U_orthogonal > epsilon) printf("U is not orthogonal (%.6g)\n", U_orthogonal);
    if(X_orthogonal > epsilon) printf("X is not orthogonal (%.6g)\n", X_orthogonal);
}

Surface_Element_Vector::Surface_Element_Vector(double dsigmat_in, double dsigmax_in, double dsigmay_in, double dsigman_in)
{
    dsigmat = dsigmat_in;
    dsigmax = dsigmax_in;
    dsigmay = dsigmay_in;
    dsigman = dsigman_in;
}

void Surface_Element_Vector::boost_dsigma_to_lrf(Milne_Basis basis_vectors, double ut, double ux, double uy, double un)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    dsigmat_LRF = dsigmat * ut  +  dsigmax * ux  +  dsigmay * uy  +  dsigman * un;      // u.dsigma
    dsigmax_LRF = - (dsigmat * Xt  +  dsigmax * Xx  +  dsigmay * Xy  +  dsigman * Xn);  // -Xi.dsigma
    dsigmay_LRF = - (dsigmax * Yx  +  dsigmay * Yy);
    dsigmaz_LRF = - (dsigmat * Zt  +  dsigman * Zn);
}

void Surface_Element_Vector::compute_dsigma_magnitude()
{
    dsigma_space = sqrt(dsigmax_LRF * dsigmax_LRF  +  dsigmay_LRF * dsigmay_LRF  +  dsigmaz_LRF * dsigmaz_LRF);
    dsigma_magnitude = fabs(dsigmat_LRF) + dsigma_space;
}

void Surface_Element_Vector::compute_dsigma_lrf_polar_angle()
{
    costheta_LRF = dsigmaz_LRF / dsigma_space;
}


Shear_Stress::Shear_Stress(double pitt_in, double pitx_in, double pity_in, double pitn_in, double pixx_in, double pixy_in, double pixn_in, double piyy_in, double piyn_in, double pinn_in)
{
    pitt = pitt_in;
    pitx = pitx_in;
    pity = pity_in;
    pitn = pitn_in;
    pixx = pixx_in;
    pixy = pixy_in;
    pixn = pixn_in;
    piyy = piyy_in;
    piyn = piyn_in;
    pinn = pinn_in;
}

void Shear_Stress::test_pimunu_orthogonality_and_tracelessness(double ut, double ux, double uy, double un, double tau2)
{
    double tau2_un = tau2 * un;

    double pitmu_umu = fabs(pitt * ut  -  pitx * ux  -  pity * uy  -  pitn * tau2_un);    // pi^\tau\mu.u_mu
    double pixmu_umu = fabs(pitx * ut  -  pixx * ux  -  pixy * uy  -  pixn * tau2_un);    // pi^x\mu.u_mu
    double piymu_umu = fabs(pity * ut  -  pixy * ux  -  piyy * uy  -  piyn * tau2_un);    // pi^x\mu.u_mu
    double pinmu_umu = fabs(pitn * ut  -  pixn * ux  -  piyn * uy  -  pinn * tau2_un);    // pi^\eta\mu.u_mu
    double Tr_pimunu = fabs(pitt  -  pixx  -  piyy  -  tau2 * pinn);                      // Tr(pimunu)

    double pimunu_umu = max(pitmu_umu, max(pixmu_umu, max(piymu_umu, pinmu_umu)));        // max pi.u component

    double epsilon = 1.e-15;

    if(pimunu_umu > epsilon) printf("pimunu is not orthogonal (%.6g)\n", pimunu_umu);
    if(Tr_pimunu > epsilon) printf("pimunu is not traceless (%.6g)\n", Tr_pimunu);
}

void Shear_Stress::boost_pimunu_to_lrf(Milne_Basis basis_vectors, double tau2)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    // piij_LRF = Xi.pi.Xj

    pixx_LRF = pitt*Xt*Xt + pixx*Xx*Xx + piyy*Xy*Xy + tau2*tau2*pinn*Xn*Xn
            + 2.0 * (-Xt*(pitx*Xx + pity*Xy) + pixy*Xx*Xy + tau2*Xn*(pixn*Xx + piyn*Xy - pitn*Xt));

    pixy_LRF = Yx*(-pitx*Xt + pixx*Xx + pixy*Xy + tau2*pixn*Xn) + Yy*(-pity*Xt + pixy*Xx + piyy*Xy + tau2*piyn*Xn); 

    pixz_LRF = Zt*(pitt*Xt - pitx*Xx - pity*Xy - tau2*pitn*Xn) - tau2*Zn*(pitn*Xt - pixn*Xx - piyn*Xy - tau2*pinn*Xn);  

    piyy_LRF = pixx*Yx*Yx + 2.0*pixy*Yx*Yy + piyy*Yy*Yy;

    piyz_LRF = -Zt*(pitx*Yx + pity*Yy) + tau2*Zn*(pixn*Yx + piyn*Yy);

    pizz_LRF = - (pixx_LRF + piyy_LRF);
}

void Shear_Stress::diagonalize_pimunu_in_lrf()
{
    double pi_LRF[] = {pixx_LRF, pixy_LRF, pixz_LRF,
                         pixy_LRF, piyy_LRF, piyz_LRF,
                         pixz_LRF, piyz_LRF, pizz_LRF};

    // diagonalize pi_ij (just need the eigenvalues)
    gsl_matrix_view A = gsl_matrix_view_array(pi_LRF, 3, 3);
    //gsl_matrix_view A_copy = gsl_matrix_view_array(pi_LRF, 3, 3);

    /*
    printf("\n");

    for(int i = 0; i < 3; i++)  
    {
        for(int j = 0; j < 3; j++)
        {
            printf("%g\t", gsl_matrix_get(&A.matrix, i, j));
        }
        printf("\n");
    }

    printf("\n");
    */

    gsl_vector * evalues = gsl_vector_alloc(3);
    gsl_eigen_symm_workspace * work = gsl_eigen_symm_alloc(3);

    //gsl_vector * tau = gsl_vector_alloc(3);

    //gsl_linalg_hessenberg_decomp(&A.matrix, tau);
    gsl_eigen_symm(&A.matrix, evalues, work);

    /*
    for(int i = 0; i < 3; i++)  
    {
        for(int j = 0; j < 3; j++)
        {
            printf("%g\t", gsl_matrix_get(&A.matrix, i, j));
        }
        printf("\n");
    }
    
    printf("\n");


    // compute eigenvalues
    gsl_eigen_symm(&A_copy.matrix, evalues, work);

    for(int i = 0; i < 3; i++) 
    {
        printf("%g\n", gsl_vector_get(evalues,i));
    }

    printf("\n");

    exit(-1);

    */

    // gsl manual says the eigenvalues are unordered
    pixx_D = gsl_vector_get(evalues, 0);
    piyy_D = gsl_vector_get(evalues, 1);
    pizz_D = gsl_vector_get(evalues, 2);

    // check that Tr(D) = 0 (okay it seems to work)
    //cout << setprecision(15) << pixx_D << "\t" << piyy_D << "\t" << pizz_D << "\t" << pixx_D + piyy_D + pizz_D << endl;

    // finally extract the transverse asymmetry measure
    if(fabs(pixx_D + piyy_D) < 1.e-2)
    {
        delta_piperp = 0.0;
    }
    else
    {
        delta_piperp = (pixx_D - piyy_D) / (pixx_D + piyy_D);   

        if(isinf(delta_piperp)) printf("Error: |delta_piperp| = %lf\n", delta_piperp);
    }

    azi_plus = 1.0 + 2.0 * delta_piperp / M_PI;  // mean amplitudes of the azimuthal shear factor
    azi_minus = 1.0 - 2.0 * delta_piperp / M_PI; // 1 + delta_perpcos(2.phi) approximated as a square wave
    
    // free memory
    gsl_vector_free(evalues);
    gsl_eigen_symm_free(work);
}

void Shear_Stress::compute_pi_magnitude()
{
    pi_magnitude = sqrt(pixx_LRF * pixx_LRF  +  piyy_LRF * piyy_LRF  +  pizz_LRF * pizz_LRF  +  2.0 * (pixy_LRF * pixy_LRF + pixz_LRF * pixz_LRF + piyz_LRF * piyz_LRF));
}

Baryon_Diffusion::Baryon_Diffusion(double Vt_in, double Vx_in, double Vy_in, double Vn_in)
{
    Vt = Vt_in;
    Vx = Vx_in;
    Vy = Vy_in;
    Vn = Vn_in;
}

void Baryon_Diffusion::test_Vmu_orthogonality(double ut, double ux, double uy, double un, double tau2)
{
    double Vmu_umu = fabs(Vt * ut  -  Vx * ux  -  Vy * uy  -  tau2 * Vn * un);    // V.u

    double epsilon = 1.e-15;

    if(Vmu_umu > epsilon) printf("Vmu is not orthogonal\n");
}

void Baryon_Diffusion::boost_Vmu_to_lrf(Milne_Basis basis_vectors, double tau2)
{
    double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
    double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
    double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
    double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

    // Vi_LRF = -Xi.V

    Vx_LRF = - Vt * Xt  +  Vx * Xx  +  Vy * Xy  +  tau2 * Vn * Xn;
    Vy_LRF = Vx * Yx  +  Vy * Yy;
    Vz_LRF = - Vt * Zt  +  tau2 * Vn * Zn;
}
