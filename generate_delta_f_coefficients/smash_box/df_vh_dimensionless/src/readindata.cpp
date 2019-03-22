
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>

#include "../include/readindata.hpp"

using namespace std;

Gauss_Laguerre::Gauss_Laguerre()
{

}

void Gauss_Laguerre::load_roots_and_weights(string file_name)
{
  stringstream file;
  file << file_name;

  FILE * gauss_file = fopen(file.str().c_str(), "r");
  if(gauss_file == NULL) printf("Error: couldn't open gauss laguerre file\n");

  // get the powers and number of gauss points
  fscanf(gauss_file, "%d\t%d", &alpha, &points);

  // allocate memory for the roots and weights
  root = (double **)calloc(alpha, sizeof(double));
  weight = (double **)calloc(alpha, sizeof(double));

  for(int i = 0; i < alpha; i++)
  {
    root[i] = (double *)calloc(points, sizeof(double));
    weight[i] = (double *)calloc(points, sizeof(double));
  }

  int dummy;  // dummy alpha index

  // load the arrays
  for(int i = 0; i < alpha; i++)
  {
    for(int j = 0; j < points; j++)
    {
      fscanf(gauss_file, "%d\t%lf\t%lf", &dummy, &root[i][j], &weight[i][j]);
    }
  }
  fclose(gauss_file);
}



read_mcid::read_mcid(long int mcid_in)
{
  mcid = mcid_in;

  char sign = '\0';

  if(mcid < 0)
  {
    sign = '-';
    printf("Error: should only be particles (not antiparticles) in pdg_test.dat\n");
  }

  int * mcid_holder = (int *)calloc(max_digits, sizeof(int));

  long int x = abs(mcid_in);
  int digits = 1;

  // get individual digits from right to left
  for(int i = 0; i < max_digits; i++)
  {
    mcid_holder[i] = (int)(x % (long int)10);

    x /= (long int)10;

    if(x > 0)
    {
      digits++;
      if(digits > max_digits) printf("Error: mcid %ld is > %d digits\n", mcid_in, max_digits);
    }
  }

  // set the quantum numbers
  nJ = mcid_holder[0];      // hadrons have 7-digit codes
  nq3 = mcid_holder[1];
  nq2 = mcid_holder[2];
  nq1 = mcid_holder[3];
  nL = mcid_holder[4];
  nR = mcid_holder[5];
  n = mcid_holder[6];
  n8 = mcid_holder[7];      // there are some hadrons with 8-digit codes
  n9 = mcid_holder[8];
  n10 = mcid_holder[9];     // nuclei have 10-digit codes

  nJ += n8;                 // I think n8 adds to nJ if spin > 9

  // test print the mcid
  //cout << sign << n << nR << nL << nq1 << nq2 << nq3 << nJ << endl;

  free(mcid_holder);  // free memory

  // get relevant particle properties
  is_particle_a_deuteron();
  is_particle_a_hadron();
  is_particle_a_meson();
  is_particle_a_baryon();
  get_spin();
  get_gspin();
  get_baryon();
  get_sign();
  does_particle_have_distinct_antiparticle();
}


void read_mcid::is_particle_a_deuteron()
{
  // check if particle is a deuteron (easy way)
  is_deuteron = (mcid == 1000010020);

  if(is_deuteron) printf("Error: there is a deuteron in HRG\n");
}

void read_mcid::is_particle_a_hadron()
{
  // check if particle is a hadron
  is_hadron = (!is_deuteron && nq3 != 0 && nq2 != 0);
}

void read_mcid::is_particle_a_meson()
{
  // check if particle is a meson
  is_meson = (is_hadron && nq1 == 0);
}

void read_mcid::is_particle_a_baryon()
{
  // check if particle is a baryon
  is_baryon = (is_hadron && nq1 != 0);
}

void read_mcid::get_spin()
{
  // get the spin x 2 of the particle
  if(is_hadron)
  {
    if(nJ == 0)
    {
      spin = 0;  // special cases: K0_L=0x130 & K0_S=0x310
      return;
    }
    else
    {
      spin = nJ - 1;
      return;
    }
  }
  else if(is_deuteron)
  {
    spin = 2;
    return;
  }
  else
  {
    printf("Error: particle is not a deuteron or hadron\n");
    // this assumes that we only have white particles (no single
    // quarks): Electroweak fermions have 11-17, so the
    // second-to-last-digit is the spin. The same for the Bosons: they
    // have 21-29 and 2spin = 2 (this fails for the Higgs).
    spin = nq3;
    return;
  }

}


void read_mcid::get_gspin()
{
  // get the spin degeneracy of the particle
  if(is_hadron && nJ > 0)
  {
    gspin = nJ;
    return;
  }
  else if(is_deuteron)
  {
    gspin = 3;  // isospin is 0 -> spin = 1 (generalize for any I later)
    return;
  }
  else
  {
    printf("Error: particle is not a deuteron or hadron\n");
    gspin = spin + 1; // lepton
    return;
  }

}

void read_mcid::get_baryon()
{
  // get the baryon number of the particle
  if(is_deuteron)
  {
    //baryon = nq3  +  10 * nq2  +  100 * nq1; // nucleus
    baryon = 2;
    return;
  }
  else if(is_hadron)
  {
    if(is_meson)
    {
      baryon = 0;
      return;
    }
    else if(is_baryon)
    {
      baryon = 1;
      return;
    }
  }
  else
  {
    printf("Error: particle is not a deuteron or hadron\n");
    baryon = 0;
    return;
  }
}

void read_mcid::get_sign()
{
  // get the quantum statistics sign of the particle
  if(is_deuteron)
  {
    sign = -1;   // deuteron is boson
    return;
  }
  else if(is_hadron)
  {
    if(is_meson)
    {
      sign = -1;
      return;
    }
    else if(is_baryon)
    {
      sign = 1;
      return;
    }
  }
  else
  {
    printf("Error: particle is not a deuteron or hadron\n");
    sign = spin % 2;
    return;
  }
}

void read_mcid::does_particle_have_distinct_antiparticle()
{
  // check if the particle has distinct antiparticle
  if(is_hadron) // hadron
  {
    has_antiparticle = ((baryon != 0) || (nq2 != nq3));
    return;
  }
  else if(is_deuteron)
  {
    has_antiparticle = true;
  }
  else
  {
    printf("Error: particle is not a deuteron or hadron\n");
    has_antiparticle = (nq3 == 1); // lepton
    return;
  }
}



int read_resonances_list_smash_box(particle_info * particle)
{
  printf("\nReading in resonances smash box pdg.dat (check if mcid_entries large enough)\n");

  //******************************|
  //******************************|
  const int mcid_entries = 4;   //|   (increase if > 4 mcid entries per line)
  //******************************|
  //******************************|

  long int * mc_id = (long int*)calloc(mcid_entries, sizeof(long int));

  string name;
  double mass;
  double width;
  char parity;

  string line;

  ifstream pdg_smash_box("pdg.dat");

  int i = 0;  // particle index

  while(getline(pdg_smash_box, line))
  {
      istringstream current_line(line);

      current_line >> name >> mass >> width >> parity >> mc_id[0] >> mc_id[1] >> mc_id[2] >> mc_id[3];

      if(line.empty() || line.at(0) == '#')
      {
        // skip lines that are blank or start with comments
        continue;
      }

      for(int k = 0; k < mcid_entries; k++)
      {
        // add particles with nonzero mc_id's to particle struct
        if(mc_id[k] != 0)
        {
          // get remaining particle info from the mcid
          read_mcid mcid_info(mc_id[k]);

          particle[i].name = name;
          particle[i].mass = mass;
          particle[i].width = width;
          particle[i].mc_id = mc_id[k];
          particle[i].gspin = mcid_info.gspin;
          particle[i].baryon = mcid_info.baryon;
          particle[i].sign = mcid_info.sign;

          i++;

          if(mcid_info.has_antiparticle)
          {
            // create antiparticle next to hadron
            ostringstream antiname;
              antiname << "Anti-" << name;

            particle[i].name = antiname.str();
            particle[i].mass = mass;
            particle[i].width = width;
            particle[i].mc_id = -mc_id[k];
            particle[i].gspin = mcid_info.gspin;
            particle[i].baryon = -mcid_info.baryon;
            particle[i].sign = mcid_info.sign;

            i++;

          } // add antiparticle

        } // check if mc_id if nonzero

        // reset mc_id's to zero
        mc_id[k] = 0;

      } // mcid index

      if(i > (Maxparticle -1))
      {
        printf("\nError: number of particles in file exceeds Maxparticle = %d. Exiting...\n\n", Maxparticle);
        exit(-1);
      }

  } // scanning file

  free(mc_id);

  int Nparticle = i;  // total number of resonances

  printf("There are %d resonances\n", Nparticle);

  // count the number of mesons, baryons and
  // antibaryons to make sure it makes sense
  int meson = 0;
  int baryon = 0;
  int antibaryon = 0;

  for(int i = 0; i < Nparticle; i++)
  {
    if(particle[i].baryon == 0) meson++;
    else if(particle[i].baryon > 0) baryon++;
    else if(particle[i].baryon < 0) antibaryon++;
  }
  if(baryon != antibaryon) printf("Error: (anti)baryons not paired correctly\n");

  printf("\nThere are %d mesons, %d baryons and %d antibaryons\n\n", meson, baryon, antibaryon);

  particle_info last_particle = particle[Nparticle - 1];

  printf("Last particle: mcid = %ld, %s, m = %lf GeV (please check this) \n\n", last_particle.mc_id, last_particle.name.c_str(), last_particle.mass);

  return Nparticle;
}


