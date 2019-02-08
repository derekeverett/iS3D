
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


int read_resonances_list(particle_info * particle)
{
  double eps = 1e-15;

  ifstream pdg("pdg.dat");

  int local_i = 0;

  int dummy_int;

  while(!pdg.eof())
  {
    pdg >> particle[local_i].mc_id;     // monte carlo id
    pdg >> particle[local_i].name;      // name (now it's in strange characters is that okay?)
    pdg >> particle[local_i].mass;      // mass (GeV)
    pdg >> particle[local_i].width;     // resonance width (GeV)
    pdg >> particle[local_i].gspin;     // spin degeneracy
    pdg >> particle[local_i].baryon;    // baryon number
    pdg >> particle[local_i].strange;   // strangeness
    pdg >> particle[local_i].charm;     // charmness
    pdg >> particle[local_i].bottom;    // bottomness
    pdg >> particle[local_i].gisospin;  // isospin degeneracy
    pdg >> particle[local_i].charge;    // electric charge
    pdg >> particle[local_i].decays;    // decay channels

     for (int j = 0; j < particle[local_i].decays; j++)
      {
        pdg >> dummy_int;
        pdg >> particle[local_i].decays_Npart[j];
        pdg >> particle[local_i].decays_branchratio[j];
        pdg >> particle[local_i].decays_part[j][0];
        pdg >> particle[local_i].decays_part[j][1];
        pdg >> particle[local_i].decays_part[j][2];
        pdg >> particle[local_i].decays_part[j][3];
        pdg >> particle[local_i].decays_part[j][4];
      }

      //decide whether particle is stable under strong interactions
      if (particle[local_i].decays_Npart[0] == 1) particle[local_i].stable = 1;
      else particle[local_i].stable = 0;

      // add anti-particle entry
      //if (particle[local_i].baryon == 1)
      if (particle[local_i].baryon != 0)
      {
        local_i++;
        particle[local_i].mc_id = -particle[local_i-1].mc_id;
        ostringstream antiname;
        antiname << "Anti-baryon-" << particle[local_i-1].name;
        particle[local_i].name = antiname.str();
        particle[local_i].mass = particle[local_i-1].mass;
        particle[local_i].width = particle[local_i-1].width;
        particle[local_i].gspin = particle[local_i-1].gspin;
        particle[local_i].baryon = -particle[local_i-1].baryon;
        particle[local_i].strange = -particle[local_i-1].strange;
        particle[local_i].charm = -particle[local_i-1].charm;
        particle[local_i].bottom = -particle[local_i-1].bottom;
        particle[local_i].gisospin = particle[local_i-1].gisospin;
        particle[local_i].charge = -particle[local_i-1].charge;
        particle[local_i].decays = particle[local_i-1].decays;
        particle[local_i].stable = particle[local_i-1].stable;

        for (int j = 0; j < particle[local_i].decays; j++)
        {
          particle[local_i].decays_Npart[j]=particle[local_i-1].decays_Npart[j];
          particle[local_i].decays_branchratio[j]=particle[local_i-1].decays_branchratio[j];

          for (int k=0; k< Maxdecaypart; k++)
          {
            if(particle[local_i-1].decays_part[j][k] == 0) particle[local_i].decays_part[j][k] = (particle[local_i-1].decays_part[j][k]);
            else
            {
              int idx;
              // find the index for decay particle
              for(idx = 0; idx < local_i; idx++)
              if (particle[idx].mc_id == particle[local_i-1].decays_part[j][k]) break;
              if(idx == local_i && particle[local_i-1].stable == 0 && particle[local_i-1].decays_branchratio[j] > eps)
              {
                cout << "Error: can not find decay particle index for anti-baryon!" << endl;
                cout << "particle mc_id : " << particle[local_i-1].decays_part[j][k] << endl;
                exit(1);
              }
              if (particle[idx].baryon == 0 && particle[idx].charge == 0 && particle[idx].strange == 0) particle[local_i].decays_part[j][k] = (particle[local_i-1].decays_part[j][k]);
              else particle[local_i].decays_part[j][k] = (- particle[local_i-1].decays_part[j][k]);
            }
          }
        }
      }
      local_i++;  // add one to the counting variable "i" for the meson/baryon
  }

  pdg.close();


  const int Nparticle = local_i;             // take account the final fake one (not yet resolved)
  //Nparticle = local_i - 1;         // take account the final fake one (why were there two blanks?)


  for(int i = 0; i < Nparticle; i++)
  {
    if(particle[i].baryon == 0)       // set the quantum statistics sign
    {
      particle[i].sign = -1;
    }
    else particle[i].sign = 1;
  }


  // count the number of mesons, baryons and antibaryons
  // to make sure it makes sense
  int meson = 0;
  int baryon = 0;
  int antibaryon = 0;

  for(int i = 0; i < Nparticle; i++)
  {
    if(particle[i].baryon == 0) meson++;
    else if(particle[i].baryon == 1) baryon++;
    else if(particle[i].baryon == -1) antibaryon++;
    else printf("Error: particle (%d) has baryon number = %d\n", particle[i].mc_id, particle[i].baryon);
  }
  if(baryon != antibaryon) printf("Error: (anti)baryons not paired correctly\n");

  printf("\nThere are %d mesons, %d baryons and %d antibaryons\n\n", meson, baryon, antibaryon);

  return Nparticle;
}
