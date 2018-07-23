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


int particle_entry(particle_info * particle_data, int number_of_resonances, int entry_mc_id)
{
    // searches for the array argument of particle_data array struct with matching mc_id 
    int entry = 0; 
    for(int ipart = 0; ipart < number_of_resonances; ipart++)
    {
        if(particle_data[ipart].mc_id == entry_mc_id)
        {
            entry = ipart; 
            break;
        }
    }
    return entry;
}


void EmissionFunctionArray::do_resonance_decays(particle_info * particle_data, const int Maxdecaypart)
  {
    printf("Starting resonance decays \n");

    // first pass particle_info struct that was read in earlier from the pdg.dat file 
    // I have the number of resonances and light particle id from the class, also passed Maxdecaypart 
    const int number_of_resonances = Nparticles;
    const int light_particle_id = LIGHTEST_PARTICLE;


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

    // search for entry of lightest particle 
    int ilight = particle_entry(particle_data, number_of_resonances, LIGHTEST_PARTICLE);
    
    // start the resonance decay feeddown, starting with the heaviest resonance at the end of the pdg.dat list 
    for(int ipart = number_of_resonances - 1; ipart > ilight - 1; ipart--)
    {
        cout << ipart << endl;
    }



    printf("Resonance decays finished \n");
  }









