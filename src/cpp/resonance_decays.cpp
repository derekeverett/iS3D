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


int particle_index(particle_info * resonance, const int number_of_resonances, const int entry_mc_id)
{
    // search for index of particle_data array struct by matching mc_id
    int index = 0;
    for(int ipart = 0; ipart < number_of_resonances; ipart++)
    {
        if(resonance[ipart].mc_id == entry_mc_id)  // I could also use the class object particles
        {
            index = ipart;
            break;
        }
    }
    return index;
}


void EmissionFunctionArray::do_resonance_decays(particle_info * resonance, const int Maxdecaypart)
  {
    printf("Starting resonance decays: \n");

    // first pass particle_info struct that was read in earlier from the pdg.dat file
    // I have the number of resonances and light particle id from the class, also passed Maxdecaypart
    const int number_of_resonances = Nparticles;
    const int lightest_particle_id = LIGHTEST_PARTICLE;


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

    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    for(int iphi = 0; iphi < phi_tab_length; iphi++)
    {
      double phi = phi_tab->get(1, iphi + 1);
      cosphiValues[iphi] = cos(phi);
      sinphiValues[iphi] = sin(phi);
    }
    // finished setting momentum arrays
    //---------------------------------------

    // I should refresh how the decay info is stored in the pdg.dat file


    // More from the readspec, readin functions....
    // the old particle_info struct holds other data:
    //  - slope: asymptotic slope of mt-spectrum (not sure what it's used for; seems like there is a bug in struct definition)
    //  - dNdptdphi: holds this distribution for each particle type (I can account for this later)
    // the resonace decay code also has a separate particle_info struct = particleDecay that holds res. decay info separately
    //  - I think I can readjust the code by using the full particle_info struct instead
    //  - or maybe not and I would need to load the decay struct in readindata.cpp
    //  - it essentially contains the same kind of information so it should be possible
    //
    // there's also this regulation thing by Chun that I don't understand so skip for now...


    // particleDecay[Ndecays] struct {reso, numpart, branch, part[5]}
    // decays != 0 is same thing as saying stable = 0 (unstable)
    // does stable = 1 mean decays = 1? i.e. the daughter is itself?
    //    - Ndecays = sum_i decays_i
    //    - decays_i = # decay channels of resonance i
    //    - reso = MC_ID of decaying resonance i
    //    - numpart = number of daughter particles in that decay
    //    - branch = branching ratio of that decay
    //    - part[5] = MC_ID list of daughter particles in that decay


    // the next thing the code does is read in a spectra file containing dNdptdphi
    // we just need the spectra array from the class: and readapt it


    // maxdecay = Ndecays = total # of decay channels
    //calc_reso_decays(max, maxdecay, LIGHTEST_PARTICLE);

    // index of lightest particle (e.g. pi+'s' is # 9)
    int ilight = particle_index(resonance, number_of_resonances, lightest_particle_id);

    // start the resonance decay feed-down, starting with the heaviest resonance:
    //---------------------------------------
    for(int ipart = number_of_resonances - 1; ipart > ilight - 1; ipart--)
    {
        int resonance_mc_id = resonance[ipart].mc_id;

        // couldn't you start with the branching ratio and decay products of each channel for starting resonance?
        // -- for one thing, I don't have the decay channels of the antibaryon on hand
        // -- well I mean, I'm sure that could be adjusted (you just need to take the anti of everything)
        // -- or I could construct a particleDecay struct

        // cycle through every decay channel known to see if
        // particle was a daughter particle in a decay channel
        for(int ichannel = 0; ichannel < maxdecay; ichannel++)
        {
            int parent_mc_id = particleDecay[ichannel].reso;
            int number_of_daughters = particleDecay[ichannel].numpart;
            int parent_index = particle_index(resonance, number_of_resonances, parent_mc_id);

            for(int idaughter = 0; idaughter < number_of_daughters; idaughter++)
            {
                int daughter_mc_id = particleDecay[ichannel].part[idaughter];

                // make sure decay channel isn't trivial and contains the daughter particle
                // I need to be careful about the anti-baryon...
                // because the decay products are only baryons and mesons
                // note: it's probably the reason why it was set up in cases of baryon, antibaryon, meson
                if((resonance_mc_id == daughter_mc_id) && (number_of_daughters != 1))
                {
                    add_resonance(ipart, parent_index, idaughter, ichannel);  // alright so what is this?
                }

            }
        }




        cout << "\r" << number_of_resonances - ipart - 1 << " / " << number_of_resonances - ilight - 1 << " resonances finished" << flush;
    }
    //---------------------------------------



    printf("\n\nResonance decays finished \n");
  }









