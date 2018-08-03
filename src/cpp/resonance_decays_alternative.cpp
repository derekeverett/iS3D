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
#include "deltafReader.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include "gaussThermal.h"
#include "particle.h"
//#ifdef _OPENACC
//#include <accelmath.h>
//#endif
using namespace std;


int particle_index(particle_info * particle_data, const int number_of_particles, const int entry_mc_id)
{
    // search for index of particle_data array struct by matching mc_id
    if(entry_mc_id == 0)
    {
        printf("Error: interested particle's mc_id is 0 (null particle in pdg.dat)\n");
        exit(-1);
    }

    bool found = false;
    int index;
    for(int ipart = 0; ipart < number_of_particles; ipart++)
    {
        if(particle_data[ipart].mc_id == entry_mc_id)  // I could also use the class object particles
        {
            index = ipart;
            found = true;
            break;
        }
    }
    if(found) return index;
    else {printf("Error: couldn't find mc_id in particle data\n"); exit(-1);}
}

MT_fit_parameters EmissionFunctionArray::estimate_MT_function_of_dNdypTdpTdphi(int iy, int iphip, int parent_chosen_index, double mass_parent)
{
    // get parameters for the fit y = exp(constant + slope * mT) of the distribution dN_dymTdmTdphi at large mT 
    // mT_const ~ logy-intercept, mT_slope ~ -effective temperature in GeV
    MT_fit_parameters MT_params; 

    // set pT array
    double pTValues[pT_tab_length];
    for(int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

    // get the coordinates (mT_points, log_dNdydpT) when the distribution is dN_pTdpTdphidy positive
    vector<double> mT_points;
    vector<double> log_dNdypTdpTdphi;
    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
        // I need the parent's chosen index (it's place in the chosen particles list) 
        long long int iS3D = parent_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

        double dN_dymTdmTdphi = dN_pTdpTdphidy[iS3D];

        //cout << pTValues[ipT] << "\t\t" << dN_dymTdmTdphi << endl;

        if(dN_dymTdmTdphi <= 0.0)
        {
            // dN_pTdpTdphidy is negative / zero
            // stop collecting coordinates
            break;
        }
        else
        {
            double pT = pTValues[ipT];
            double mT = sqrt(mass_parent * mass_parent + pT * pT);

            if(mT > sqrt(2.73) * mass_parent)   // mT in relativistic region (~ threshold for 2 points for Omega2250)
            {
                mT_points.push_back(mT);
                log_dNdypTdpTdphi.push_back(log(dN_dymTdmTdphi));   // called y below
            }   
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
            sum += (A_transpose[i][k] * log_dNdypTdpTdphi[k]);
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
    for(int i = 0; i < 2; i++) x[i] = f[i]; // solution x stored in f
    free_2D(M,2);                           // free memory 
    //----------------------------------

    MT_params.constant = x[0];     // get constant and slope of exponential fit
    MT_params.slope = x[1];    

    return MT_params;
}



double EmissionFunctionArray::dN_dYMTdMTdPhi_non_boost_invariant(int parent_chosen_index, double MTValues[], double phipValues[], double yValues[], double MT, double Phip_1, double Phip_2, double Y, double MTmax, MT_fit_parameters ** MT_params)
{
    // do a tri-linear interpolation non-boost_invariant log(dN_dYMTdMTdPhi) of the parent distribution 
    double logdN_1 = 0.0;
    double logdN_2 = 0.0;

    // first search for the interpolation points for the angles 
    if(MT < MTmax)
    {
        int iMT_R = 1;
        while(MT > MTValues[iMT_R]) iMT_R++; 
    }
    else
    {
        // linear interpolation in 
        // call the analytic function for the fit in the MT direction
    }


    // pass the momentum tables, also include 2nd angle



    // undo the log
    return exp(logdN_1) + exp(logdN_2); 
}




void EmissionFunctionArray::do_resonance_decays(particle_info * particle_data)
  {
    printline();
    printf("Starting resonance decays: \n\n");

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
        // index in chosen particles list 
        int parent_chosen_index = ichosen; 

        // chosen particle index in particle_data array of structs
        int ipart = chosen_particles_sampling_table[parent_chosen_index];
        int stable = particle_data[ipart].stable;

        // if particle unstable under strong interactions, do resonance decay
        if(!stable)
        {
            // parent resonance info:
            int parent_index = ipart;
            int parent_decay_channels = particle_data[ipart].decays;

            // go through each decay channel of parent resonance
            for(int ichannel = 0; ichannel < parent_decay_channels; ichannel++)
            {
                // why is this number negative sometimes?
                int decay_products = abs(particle_data[ipart].decays_Npart[ichannel]);

                // set up vector that holds particle indices of real daughters
                vector<int> decays_index_vector;

                for(int idaughter = 0; idaughter < decay_products; idaughter++)
                {
                    int daughter_mc_id = particle_data[ipart].decays_part[ichannel][idaughter];
                    int daughter_index = particle_index(particle_data, number_of_particles, daughter_mc_id);
                    decays_index_vector.push_back(daughter_index);

                } // finished setting decays_index_vector

                // int real_daughters = 0;
                // for(int idaughter = 0; idaughter < Maxdecaypart; idaughter++)
                // {
                //     // caution: some daughter entries have a default mc_id = 0 (fake placeholders)
                //     int daughter_mc_id = particle_data[ipart].decays_part[ichannel][idaughter];
                //     if(daughter_mc_id != 0)
                //     {
                //       int daughter_index = particle_index(particle_data, number_of_particles, daughter_mc_id);
                //       decays_index_vector.push_back(daughter_index);
                //       real_daughters++;
                //     }

                // } // finished setting decays_index_vector

                // only do non-trivial resonance decays
                if(decay_products != 1)
                {
                  resonance_decay_channel(particle_data, ichannel, parent_index, parent_chosen_index, decays_index_vector);
                }
            }
            printf("\n");
        } //

        //cout << "\r" << number_of_particles - ipart - 1 << " / " << number_of_particles - ilight - 1 << " resonances finished" << flush;
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
    cout << setprecision(3) << branch_ratio * 100.0 << "% of " << parent_name << "s decays to\t";
    for(int idecay = 0; idecay < decay_products; idecay++)
    {
      cout << particle_data[decays_index_vector[idecay]].mc_id << "\t";
    }

    switch(decay_products)
    {
        case 2: // 2-body decay
        {
            // particle index of decay products
            int particle_1 = decays_index_vector[0];
            int particle_2 = decays_index_vector[1];

            // masses involved in 2-body decay
            double mass_parent = particle_data[parent].mass;
            double mass_1 = particle_data[particle_1].mass;
            double mass_2 = particle_data[particle_2].mass;

            // adjust the masses to satisfy energy conservation
            while((mass_1 + mass_2) > mass_parent)
            {
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
            // particle index of decay products
            int particle_1 = decays_index_vector[0];
            int particle_2 = decays_index_vector[1];
            int particle_3 = decays_index_vector[2];

            // masses involved in 3-body decay
            double mass_parent = particle_data[parent].mass;
            double mass_1 = particle_data[particle_1].mass;
            double mass_2 = particle_data[particle_2].mass;
            double mass_3 = particle_data[particle_3].mass;       // what do they do with the masses here?

            // 3-body decay integration routine
            three_body_decay(particle_data, branch_ratio, parent, parent_chosen_index, particle_1, particle_2, particle_3, mass_1, mass_2, mass_3, mass_parent);

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
    printf("\n");
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
        printf("Zero decays particles found in resonance table: skip integration routine");
        return;
    }

    // group selected particles by type
    vector<int> particle_groups;    // group particle indices
    vector<int> group_members;      // group members

    // group first particle and remove it from original list
    particle_groups.push_back(selected_particles[0]);
    group_members.push_back(1);
    selected_particles.erase(selected_particles.begin());

    // group next remaining particles if any:
    while(selected_particles.size() > 0)
    {
        int next_particle = selected_particles[0];
        int current_groups = particle_groups.size();
        bool put_next_particle_in_current_groups = false;

        // loop through current groups
        for(int igroup = 0; igroup < current_groups; igroup++)
        {
            if(next_particle == particle_groups[igroup])
            {
                // if match, add particle to that group
                group_members[igroup] += 1;
                put_next_particle_in_current_groups = true;
                break;
            }
        }
        if(!put_next_particle_in_current_groups)
        {
            // make a new group
            particle_groups.push_back(next_particle);
            group_members.push_back(1);
        }
        // remove particle from original list and look at the next particle
        selected_particles.erase(selected_particles.begin());
    } // while loop until all particles are grouped

    int groups = particle_groups.size();

    // set particle's mass, energy_star and momentum_star in each group
    double mass_squared[groups];
    double momentum_star[groups];
    double energy_star[groups];

    for(int igroup = 0; igroup < groups; igroup++)
    {
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
    printf("Particle groups: ");
    for(int igroup = 0; igroup < groups; igroup++)
    {
        printf("(%d,%d)\t", particle_data[particle_groups[igroup]].mc_id, group_members[igroup]);
    }


    // set momentum arrays: (yValues, pTValues, cosphiValues, sinphiValues)
    //---------------------------------------
    int y_pts = y_tab_length;
    if(DIMENSION == 2) y_pts = 1;

    double yValues[y_pts];
    double pTValues[pT_tab_length];
    double MTValues[pT_tab_length];     // MT points of the parent resonance 
    double phipValues[phi_tab_length];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
    }
    else if(DIMENSION == 3)
    {
      for(int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);
    }

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
        double pT = pT_tab->get(1, ipT + 1);
        pTValues[ipT] = pT; 
        MTValues[ipT] = sqrt(fabs(pT * pT + mass_parent * mass_parent));
    }

    for(int iphip = 0; iphip < phi_tab_length; iphip++) phipValues[iphip] = phi_tab->get(1, iphip + 1);
    

    // maximum ranges of parent spectra in Cooper Frye data
    double Ymax = fabs(yValues[y_pts - 1]);
    double MTmax = MTValues[pT_tab_length - 1];


    // finished setting momentum arrays
    //---------------------------------------


    // fix the roots/weights for the decay integral

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
    double ** M = (double**)calloc(2, sizeof(double*));
    for(int i = 0; i < 2; i++) M[i] = (double*)calloc(2, sizeof(double));


    MT_fit_parameters ** MT_params = (MT_fit_parameters**)calloc(y_pts,sizeof(MT_fit_parameters*));
    for(int iy = 0; iy < y_pts; iy++) MT_params[iy] = (MT_fit_parameters*)calloc(phi_tab_length,sizeof(MT_fit_parameters));

    for(int iphi = 0; iphi < phi_tab_length; iphi++)
    {
        for(int iy = 0; iy < y_pts; iy++)
        {
            //MT_params[iy][iphi] = estimate_MT_function_of_dNdypTdpTdphi(iy, iphi, parent_chosen_index, mass_parent);
        }
    }


    // two-body decay integration:
    //---------------------------------------

    // I should probably distinguish between boost-invariant and 3+1d case
    // in the 3+1d case I have a finite rapidity window, and the integration
    // over the parent rapidity may require extrapolation?
    // it looks like in the old code: they truncated rapidity regions out of usual window

    // loop over particle groups
    for(int igroup = 0; igroup < groups; igroup++)
    {
        // particle index and multiplicity
        int particle_index = particle_groups[igroup];
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
                double coshvDeltaY2 = coshvDeltaY * coshvDeltaY;
                double mT2_coshvDeltaY2_minus_pT2 = (mT2 * coshvDeltaY2) - pT2; 

                // set tables:
                MTbar_table[k] = Estar_M_mT * coshvDeltaY / mT2_coshvDeltaY2_minus_pT2;
                DeltaMT_table[k] = M_pT * sqrt(fabs(Estar2_plus_pT2 - mT2 * coshvDeltaY2)) / mT2_coshvDeltaY2_minus_pT2;
                mT_coshvDeltaY_over_pT_table[k] = mT * coshvDeltaY / pT;
                vintegrand_weight_table[k] = prefactor * DeltaY * v_weight[k] / sqrt(fabs(mT2_coshvDeltaY2_minus_pT2)); 
            }

            for(int iphip = 0; iphip < phi_tab_length; iphip++)
            {
                double phip = phipValues[iphip];    // particle azimuthal angle 

                for(int iy = 0; iy < y_pts; iy++)
                {
                    double y = yValues[iy];     // particle rapidity 

                    double decay2D_integral = 0.0;

                    // do the decay2D_integral over parent rapidity, transverse mass space (v, zeta)
                    for(int iv = 0; iv < gauss_pts; iv++)
                    {
                        double v = v_root[iv];
                        double Y = y + v * DeltaY;  // parent rapidity 

                        double MTbar = MTbar_table[iv];
                        double DeltaMT = DeltaMT_table[iv];
                        double mT_coshvDeltaY_over_pT = mT_coshvDeltaY_over_pT_table[iv];
                        double vintegrand_weight = vintegrand_weight_table[iv];

                        //cout << "\n" << setprecision(5) << pT << "\t" << DeltaY << "\t" << mT2_coshvDeltaY2_minus_pT2 << "\t" << MT_bar << "\t" << DeltaMT << endl;
                        //exit(-1);

                        double zeta_integral = 0.0;

                        for(int izeta = 0; izeta < gauss_pts; izeta++)
                        {
                            if(fabs(Y) > Ymax) break;   // cutoff of parent distribution in Y

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

                            double integrand = MT * dN_dYMTdMTdPhi_non_boost_invariant(parent_chosen_index, MTValues, phipValues, yValues, MT, Phip_1, Phip_2, Y, MTmax, MT_params); 

                            zeta_integral += (zeta_gauss_weight * integrand);
                        }

                        decay2D_integral += (vintegrand_weight * zeta_integral);
                    }


                    // since I've already calculated a lot of things, I might as well put the parent momentum integration directly


                    // integration over parent momentum space goes here:
                    //double spectrum = EdNdp3_2bodyN(pT, iphip, y, mass, W2, mass_parent, parent);

                    //spectrum = 0.0;

                    int particle_chosen_index = 0;  // fix this... (this is a bug..) 

                    long long int iS3D = particle_chosen_index + number_of_chosen_particles * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

                    dN_pTdpTdphidy[iS3D] += decay2D_integral;
                }
            }
        }
    }
    //---------------------------------------
    // finished two-body decay routine
}






void EmissionFunctionArray::three_body_decay(particle_info * particle_data, double branch_ratio, int parent, int parent_chosen_index, int particle_1, int particle_2, int particle_3, double mass_1, double mass_2, double mass_3, double mass_parent)
{
    // first select decay products that are part of chosen resonance particles
    // TODO: distinguish resonance table from chosen particles table whose final spectra we're interested in

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
        int chosen_index = chosen_particles_sampling_table[ichosen];

        if((particle_1 == chosen_index) && (!found_particle_1))
        {
            found_particle_1 = true;
        }
        if((particle_2 == chosen_index) && (!found_particle_2))
        {
            found_particle_2 = true;
        }
        if((particle_3 == chosen_index) && (!found_particle_3))
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
        printf("Zero decays particles found in resonance table: skip integration routine");
        return;
    }

     // group selected particles by type
    vector<int> particle_groups;    // group particle indices
    vector<int> group_members;      // group members

    // group first particle and remove it from original list
    particle_groups.push_back(selected_particles[0]);
    group_members.push_back(1);
    selected_particles.erase(selected_particles.begin());

    // group remaining particles if any:
    while(selected_particles.size() > 0)
    {
        int next_particle = selected_particles[0];
        bool put_next_particle_in_current_groups = false;
        int current_groups = particle_groups.size();

        // loop through current groups
        for(int igroup = 0; igroup < current_groups; igroup++)
        {
            if(next_particle == particle_groups[igroup])
            {
                // if match, add particle to that group
                group_members[igroup] += 1;
                put_next_particle_in_current_groups = true;
                break;
            }
        }
        if(!put_next_particle_in_current_groups)
        {
            // make a new group
            particle_groups.push_back(next_particle);
            group_members.push_back(1);
        }
        // remove particle from original list and look at the next particle
        selected_particles.erase(selected_particles.begin());
    }

    int groups = particle_groups.size();


    // set the mass of particle type's mass in group
    double mass_groups[groups];
    for(int igroup = 0; igroup < groups; igroup++)
    {
        mass_groups[igroup] = particle_data[particle_groups[igroup]].mass;
    }

    // set invariant mass of second object for each group
    double invariant_mass_squared[groups];

    for(int igroup = 0; igroup < groups; igroup++)
    {
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


        // get mass of the second and third particle (should double check this...)
        int particle_2 = decay_products_vector[0];
        int particle_3 = decay_products_vector[1];
        double mass_2 = particle_data[particle_2].mass;
        double mass_3 = particle_data[particle_3].mass;

        // I don't what to do yet..

        //invariant_mass_squared[igroup] = mass_2 * mass_2;


    }

    printf("Particle groups: ");
    for(int igroup = 0; igroup < groups; igroup++)
    {
        printf("(%d,%d)\t", particle_data[particle_groups[igroup]].mc_id, group_members[igroup]);
    }






    // set momentum arrays: (yValues, pTValues, cosphiValues, sinphiValues)
    //---------------------------------------
    int y_pts = y_tab_length;
    if(DIMENSION == 2) y_pts = 1;

    double yValues[y_pts];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;
    }
    else if(DIMENSION == 3)
    {
      for(int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);
    }

    double pTValues[pT_tab_length];

    for(int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

    double phipValues[phi_tab_length];
    double cosphipValues[phi_tab_length];
    double sinphipValues[phi_tab_length];

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phip = phi_tab->get(1, iphip + 1);
      phipValues[iphip] = phip;
      cosphipValues[iphip] = cos(phip);
      sinphipValues[iphip] = sin(phip);
    }
    // finished setting momentum arrays
    //---------------------------------------



    // loop over momentum
    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
        double pT = pTValues[ipT];

        for(int iphip = 0; iphip < phi_tab_length; iphip++)
        {
            double phip = phipValues[iphip];
            // phi = iphi * deltaphi; // I think this is a bug b/c phi table isn't uniform

            for(int iy = 0; iy < y_pts; iy++)
            {
                double y = yValues[iy];

                // call the 2-body decay integral and add its contribution to particle1
                //double spectrum = Edndp3_2bodyN(y, pT, phip, mass1, mass2, mass_parent, parent_mc_id);

                //double branch_ratio = particleDecay[decay_channel].branch;

                // iS3D index of particle1 with momentum (pT,phip,y)
                // note: chosen_particles = npart should be same as number of resonances involved

                //long long int is3D = particle1 + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

                //dN_pTdpTdphidy[iS3D] += (branch_ratio * spectrum);

            }
        }
    }
    // finished three-body decay routine
}




