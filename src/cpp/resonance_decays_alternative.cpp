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
        printf("Error: mc_id of interest is 0 (null particle)\n");
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

// double dn2ptN(double w2, void *para1)
// {
//   pblockN *para = (pblockN *) para1;
//   para->e0 = (para->mr * para->mr + para->m1 * para->m1 - w2) / (2 * para->mr);
//   para->p0 = sqrt (para->e0 * para->e0 - para->m1 * para->m1);
//   //Integrate the "dnpir1N" kernal over cos(theta) using gaussian integration
//   double r = gauss (PTN1, *dnpir1N, -1.0, 1.0, para);
//   return r;
// }

// typedef struct pblockN
// {
//   double y, pT, phip, mT, E, pz;    // momentum of decay product 1
//   double mass_1, mass_2, mass_3;    // masses of decay products
//   double mass_parent;               // parent resonance mass
//   double costh, sinth;              //
//   double e0, p0;                    //
//   int parent_mc_id;                 // parent mc_id
// } pblockN;


// double Edndp3_2bodyN(double y, double pT, double phip, double mass_1, double mass_2, double mass_parent, int parent_mc_id)
// // in units of GeV^-2, includes phasespace and volume, does not include degeneracy factors
// // double y;  /* rapidity of particle 1 */
// // double pt; /* transverse momentum of particle 1 */
// // double phi; /* phi angle of particle 1 */
// // double m1, m2; /* restmasses of decay particles in MeV */
// // double mr;     /* restmass of resonance MeV            */
// // int res_num;   /* Montecarlo number of the Resonance   */
// {
//   double mT = sqrt(pT * pT + mass_1 * mass_1);

//   pblockN parameter;

//   parameter.pT = pT;                // basically fix outgoing momentum of particle 1
//   parameter.mT = mT;
//   parameter.E = mT * cosh(y);
//   parameter.pz = mT * sinh(y);
//   parameter.y = y;
//   parameter.phi = phip;
//   parameter.mass_1 = mass_1;
//   parameter.mass_2 = mass_2;
//   para.mass_parent = mass_parent;
//   para.res_num = parent_mc_id;

//   double norm2 = 1.0 / (2.0 * PI);      // 2-body normalization (must be a formula)

//   // calls the integration routines for 2-body
//   double res2 = norm2 * dn2ptN(mass_2 * mass_2, &para);     // I'm having a hard time following the nest of integrations

//   if (res2 < 0.0) res2 = 0.0;
//   return res2;          /* like Ed3ndp3_2body() */
// }


void EmissionFunctionArray::do_resonance_decays(particle_info * particle_data)
  {
    printline();
    printf("Starting resonance decays: \n\n");

    const int number_of_particles = Nparticles;         // total number of particles in pdg.dat (includes leptons, photons, antibaryons)
    //const int light_particle_id = LIGHTEST_PARTICLE;  // lightest particle mc_id

    // note: I moved pion-0 as the lightest hadron in pdg.dat

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
            int parent_decay_channels = particle_data[ipart].decays;

            // go through each decay channel of the parent resonance
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
                  resonance_decay_channel(particle_data, ichannel, parent_index, decays_index_vector);
                }
            }
            printf("\n");
        } //

        //cout << "\r" << number_of_particles - ipart - 1 << " / " << number_of_particles - ilight - 1 << " resonances finished" << flush;
    }

    printf("\n\nResonance decays finished \n");
  }


void EmissionFunctionArray::resonance_decay_channel(particle_info * particle_data, int channel, int parent_index, vector<int> decays_index_vector)
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
            two_body_decay(particle_data, branch_ratio, parent, particle_1, particle_2, mass_1, mass_2, mass_parent);

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
            three_body_decay(particle_data, branch_ratio, parent, particle_1, particle_2, particle_3, mass_1, mass_2, mass_3, mass_parent);

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


void EmissionFunctionArray::two_body_decay(particle_info * particle_data, double branch_ratio, int parent, int particle_1, int particle_2, double mass_1, double mass_2, double mass_parent)
{
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
        bool next_particle_in_current_groups = false;

        // loop through current groups
        for(int igroup = 0; igroup < current_groups; igroup++)
        {
            if(next_particle == particle_groups[igroup])
            {
                // if match, add particle to that group
                group_members[igroup] += 1;
                next_particle_in_current_groups = true;
                break;
            }
        }
        if(!next_particle_in_current_groups)
        {
            // make a new group
            particle_groups.push_back(next_particle);
            group_members.push_back(1);
        }
        // remove particle from original list and look at the next particle
        selected_particles.erase(selected_particles.begin());
    } // while loop until all particles are grouped

    int groups = particle_groups.size();

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
    // finished two-body decay routine
}






void EmissionFunctionArray::three_body_decay(particle_info * particle_data, double branch_ratio, int parent, int particle_1, int particle_2, int particle_3, double mass_1, double mass_2, double mass_3, double mass_parent)
{
    // first select decay products that are part of chosen resonance particles
    // TODO: distinguish resonance table from chosen particles table whose final spectra we're interested in

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
        bool next_particle_in_current_groups = false;
        int current_groups = particle_groups.size();

        // loop through current groups
        for(int igroup = 0; igroup < current_groups; igroup++)
        {
            if(next_particle == particle_groups[igroup])
            {
                // if match, add particle to that group
                group_members[igroup] += 1;
                next_particle_in_current_groups = true;
                break;
            }
        }
        if(!next_particle_in_current_groups)
        {
            // make a new group
            particle_groups.push_back(next_particle);
            group_members.push_back(1);
        }
        // remove particle from original list and look at the next particle
        selected_particles.erase(selected_particles.begin());
    }

    int groups = particle_groups.size();

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




