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
    int index = 0;
    for(int ipart = 0; ipart < number_of_particles; ipart++)
    {
        if(particle_data[ipart].mc_id == entry_mc_id)  // I could also use the class object particles
        {
            index = ipart;
            break;
        }
    }
    return index;
}

double dn2ptN(double w2, void *para1)
{
  pblockN *para = (pblockN *) para1;
  para->e0 = (para->mr * para->mr + para->m1 * para->m1 - w2) / (2 * para->mr);
  para->p0 = sqrt (para->e0 * para->e0 - para->m1 * para->m1);
  //Integrate the "dnpir1N" kernal over cos(theta) using gaussian integration
  double r = gauss (PTN1, *dnpir1N, -1.0, 1.0, para);
  return r;
}

typedef struct pblockN
{
  double y, pT, phip, mT, E, pz;    // momentum of decay product 1
  double mass_1, mass_2, mass_3;    // masses of decay products
  double mass_parent;               // parent resonance mass
  double costh, sinth;              //
  double e0, p0;                    //
  int parent_mc_id;                 // parent mc_id
} pblockN;


double Edndp3_2bodyN(double y, double pT, double phip, double mass_1, double mass_2, double mass_parent, int parent_mc_id)
// in units of GeV^-2, includes phasespace and volume, does not include degeneracy factors
// double y;  /* rapidity of particle 1 */
// double pt; /* transverse momentum of particle 1 */
// double phi; /* phi angle of particle 1 */
// double m1, m2; /* restmasses of decay particles in MeV */
// double mr;     /* restmass of resonance MeV            */
// int res_num;   /* Montecarlo number of the Resonance   */
{
  double mT = sqrt(pT * pT + mass_1 * mass_1);

  pblockN parameter;

  parameter.pT = pT;                // basically fix outgoing momentum of particle 1
  parameter.mT = mT;
  parameter.E = mT * cosh(y);
  parameter.pz = mT * sinh(y);
  parameter.y = y;
  parameter.phi = phip;
  parameter.mass_1 = mass_1;
  parameter.mass_2 = mass_2;
  para.mass_parent = mass_parent;
  para.res_num = parent_mc_id;

  double norm2 = 1.0 / (2.0 * PI);      // 2-body normalization (must be a formula)

  // calls the integration routines for 2-body
  double res2 = norm2 * dn2ptN(mass_2 * mass_2, &para);     // I'm having a hard time following the nest of integrations

  if (res2 < 0.0) res2 = 0.0;
  return res2;          /* like Ed3ndp3_2body() */
}


void EmissionFunctionArray::do_resonance_decays(particle_info * particle_data)
  {
    printf("Starting resonance decays: \n");

    const int number_of_particles = Nparticles;           // total number of particles in pdg.dat (includes leptons, photons, antibaryons)
    const int lightest_particle_id = LIGHTEST_PARTICLE;   // lightest particle mc_id

    // particle index of lightest particle included in decay routine (e.g. pi+'s' is # 9)
    const int ilight = particle_index(particle_data, number_of_particles, lightest_particle_id);

    // number of resonances included in the decay routine
    const int included_resonances = number_of_particles - (ilight + 1);


    if(included_resonances != chosen_particles)
    {
        // this is a very crude match. one should check the order of the structs as well
        // everything from mass to charge,etc should be identical
        printf("Error: resonances included in decay routine does not match chosen_particles");
        exit(-1);
    }


    // start the resonance decay feed-down, starting with the heaviest resonance:
    for(int ipart = number_of_particles - 1; ipart > ilight - 1; ipart--)
    {
        int stable = particle_data[ipart].stable;
        // if particle unstable in strong interactions, do resonance decay
        if(!stable)
        {
            // parent resonance info:
            int parent_index = ipart;
            int parent_decay_channels = particle_data[ipart].decays;
            if(parent_decay_channels == 1)
            {
              printf("Error: unstable resonance only has trivial decay: exiting..");
              exit(-1);
            }
            int parent_baryon = particle_data[ipart].baryon;  // is this already taken care of in readin or not? if not, adjust it

            // go through each decay channel of the parent resonance
            for(int ichannel = 0; ichannel < parent_decay_channels; ichannel++)
            {
                int decay_products = particle_data[ipart].decays_Npart[ichannel];

                // make an array holding particle indices of the daughters
                int decay_index[decay_products];
                for(int idaughter = 0; idaughter < decay_products; idaughter++)
                {
                    int daughter_mc_id = particle_data[ipart].decays_part[][];
                    decay_index[idaughter] = particle_index(particle_data, number_of_particles, daughter_mc_id);
                }
                // amend decay product dN produced from decay channel to the thermal spectra
                resonance_decay(particle_data, parent_index, decay_index, included_resonances);
            }
        } //

        cout << "\r" << number_of_resonances - ipart - 1 << " / " << number_of_resonances - ilight - 1 << " resonances finished" << flush;
    }

    printf("\n\nResonance decays finished \n");
  }


void EmissionFunctionArray::resonace_decay(particle_info * resonance, int particle_index, int parent_index, int daughter_match_index, int decay_channel, const int total_resonances_included)
{

    nblock paranorm;      /* for 3body normalization integral */  // I don't know what this is...
    double norm3;         /* normalisation of 3-body integral */

    double y;
    double m1, m2, m3, mr;
    int pn2, pn3, pn4;        /* internal numbers for resonances */


    // set momentum arrays: (yValues, pTValues, cosphiValues, sinphiValues)
    //---------------------------------------
    int y_pts = y_tab_length;

    if(DIMENSION == 2) y_pts = 1;
    else {printf("Resonance decay routine is currently boost invariant, can't couple with 3+1d spectra. Exiting..."); exit(-1);}

    double yValues[y_pts];

    if(DIMENSION == 2) yValues[0] = 0.0;
    else {printf("Fill in now"); exit(-1);}

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

    //double deltaphi = 2*PI/nphi;  // seems out of place b/c phi table is NOT uniform

    // parent info:
    int parent = parent_index;                   // particle index of parent, mc_id
    int parent_mc_id = resonance[parent].mc_id;  // mc_id of parent

    int npart = total_resonances_included;

    // number of decay products from decay channel
    int decay_products = particleDecay[decay_channel].numpart;

    switch(decay_products)
    {
        case 1: // stable
        {
            printf("Skipping trivial decay channel (stable)");
            break;
        }
        case 2: // 2-body decay
        {
            int particle1 = particle_index;     // particle index of resonance of interest
            int particle2;                      // particle index of other decay product

            if(daughter_match_index == 0)
            {
                particle2 = particle_index(resonance, Nparticles, particleDecay[decay_channel].part[1]);
            }
            else
            {
                particle2 = particle_index(resonance, Nparticles, particleDecay[decay_channel].part[0]);
            }

            // masses involved in 2-body decay
            double mass_parent = resonance[parent].mass;
            double mass1 = resonance[particle1].mass;
            double mass2 = resonance[particle2].mass;

            // it wants the parent mass to be heavier obviously
            // but why these fractions?
            while((mass1 + mass2) > mass_parent)
            {
                mass_parent += 0.25 * resonance[parent].width;
                mass1 -= 0.5 * resonance[particle1].width;
                mass2 -= 0.5 * resonance[particle2].width;
                if(mass1 < 0.0 || mass2 < 0.0)
                {
                    printf("One daughter mass went negative: stop");
                    exit(-1);
                }
            }

            if(DIMENSION == 2)
            {
                for(int ipT = 0; ipT < pT_tab_length; ipT++)
                {
                    double pT = pTValues[ipT];

                    for(int iphip = 0; iphip < phi_tab_length; iphip++)
                    {
                        double phip = phiValues[iphip];
                        // phi = iphi * deltaphi; // I think this is a bug b/c phi table isn't uniform

                        for(int iy = 0; iy < y_pts; iy++)
                        {
                            double y = yValues[iy];

                            // call the 2-body decay integral and add its contribution to particle1
                            double spectrum = Edndp3_2bodyN(y, pT, phip, mass1, mass2, mass_parent, parent_mc_id);

                            double branch_ratio = particleDecay[decay_channel].branch;

                            // iS3D index of particle1 with momentum (pT,phip,y)
                            // note: chosen_particles = npart should be same as number of resonances involved

                            long long int is3D = particle1 + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));

                            dN_pTdpTdphidy[iS3D] += (branch_ratio * spectrum);

                            //particle[pn].dNdypTdpTdphi[iy][ipT][iphi] += (particleDecay[decay_channel].branch * spectrum);
                        }
                    }
                }
            }
          else
          {
            for (int iy = 0; iy < ny; iy++)
            {
              y = particle[pn].y[iy];
              for (int ipT = 0; ipT < npt; ipT++)
              {
                / if (pseudofreeze) y = Rap(particle[pn].y[n], particle[pn].pt[l], m1);

                for (int iphi = 0; iphi < nphi; iphi++)
                {
                  double phi = 0.0;
                  //if (pseudofreeze) phi = i * deltaphi;
                  //else phi = phiArray[i];
                  phi = iphi * deltaphi;
                  double spectrum = Edndp3_2bodyN(y, particle[pn].pt[ipT], phi, m1, m2, mr, particle[pnR].mcID);
                  // Call the 2-body decay integral and add its contribution to the daughter
                  // particle of interest
                  particle[pn].dNdypTdpTdphi[iy][ipT][iphi] += (particleDecay[j].branch*spectrum);
                  //if (iy == ny/2 && iphi == 0) particle[pn].resCont[iy][ipT][iphi] += (particleDecay[j].branch*spectrum);
                }
              }
            }
          }
          break;
        }

    default:
    {
      printf ("ERROR in add_reso! \n");
      printf ("%i decay not implemented ! \n", abs (particleDecay[j].numpart));
      exit (0);
    }
  }
}





