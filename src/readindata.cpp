
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>

#include "main.h"
#include "readindata.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "Table.h"

using namespace std;

read_FOdata::read_FOdata(ParameterReader* paraRdr_in, string path_in)
{
  paraRdr = paraRdr_in;
  pathToInput = path_in;
  mode = paraRdr->getVal("hydro_mode");
  turn_on_bulk = paraRdr->getVal("turn_on_bulk");
  turn_on_muB = paraRdr->getVal("turn_on_muB");
}

read_FOdata::~read_FOdata()
{

}

int read_FOdata::get_number_of_freezeout_cells()
{
  int number_of_cells = 0;
  //add more options for types of FO file
  else if (mode == 2) // outputs from MUSIC full (3+1)-d
  {
    ostringstream surface_file;
    surface_file << pathToInput << "/surface.dat";
    Table block_file(surface_file.str().c_str());
    number_of_cells = block_file.getNumberOfRows();
  }
  return(number_of_cells);
}

void read_FOdata::read_in_freeze_out_data(int length, FO_surf* surf_ptr)
{
  if (mode == 2)   // MUSIC full (3+1)-d outputs
  read_FOsurfdat_MUSIC(length, surf_ptr);
  //put more options here
  return;
}

void read_FOdata::read_FOsurfdat_MUSIC(long length, FO_surf* surf_ptr)
{
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());
  for (long i = 0; i < length; i++)
  {
    // freeze out position
    surfdat >> surf_ptr[i].tau;
    surfdat >> surf_ptr[i].x;
    surfdat >> surf_ptr[i].y;
    surfdat >> surf_ptr[i].eta;

    // freeze out normal vectors
    surfdat >> surf_ptr[i].dat;
    surfdat >> surf_ptr[i].dax;
    surfdat >> surf_ptr[i].day;
    surfdat >> surf_ptr[i].dan;

    // flow velocity
    surfdat >> surf_ptr[i].ut;
    surfdat >> surf_ptr[i].ux;
    surfdat >> surf_ptr[i].uy;
    surfdat >> surf_ptr[i].un;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;
    surf_ptr[i].E = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].T = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].muB = dummy * hbarC;
    surfdat >> dummy;                    //(e+p)/T
    surf_ptr[i].P = dummy * surf_ptr[i].T - surf_ptr[i].E;
    surf_ptr[i].Bn = 0.0; //what is this?
    surf_ptr[i].muS = 0.0;

    // dissipative quantities at freeze out
    surfdat >> dummy;
    surf_ptr[i].pitt = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pitx = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pity = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pitn = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pixx = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pixy = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pixn = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].piyy = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].piyn = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pinn = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].bulkPi = dummy * hbarC;
    if(turn_on_muB == 1)
    {
      surfdat >> dummy;
      surf_ptr[i].muB = dummy * hbarC;
    }
    else
    surf_ptr[i].muB = 0.0;
  }
  surfdat.close();
  return;
}

int read_FOdata::read_resonances_list(particle_info* particle)
{
  double eps = 1e-15;
  int Nparticle=0;
  cout << " -- Read in particle resonance decay table...";
  ifstream resofile("EOS/pdg.dat");
  int local_i = 0;
  int dummy_int;
  while (!resofile.eof())
  {
    resofile >> particle[local_i].monval;
    resofile >> particle[local_i].name;
    resofile >> particle[local_i].mass;
    resofile >> particle[local_i].width;
    resofile >> particle[local_i].gspin;	      //spin degeneracy
    resofile >> particle[local_i].baryon;
    resofile >> particle[local_i].strange;
    resofile >> particle[local_i].charm;
    resofile >> particle[local_i].bottom;
    resofile >> particle[local_i].gisospin;     //isospin degeneracy
    resofile >> particle[local_i].charge;
    resofile >> particle[local_i].decays;

    for (int j = 0; j < particle[local_i].decays; j++)
    {
      resofile >> dummy_int;
      resofile >> particle[local_i].decays_Npart[j];
      resofile >> particle[local_i].decays_branchratio[j];
      resofile >> particle[local_i].decays_part[j][0];
      resofile >> particle[local_i].decays_part[j][1];
      resofile >> particle[local_i].decays_part[j][2];
      resofile >> particle[local_i].decays_part[j][3];
      resofile >> particle[local_i].decays_part[j][4];
    }

    //decide whether particle is stable under strong interactions
    if (particle[local_i].decays_Npart[0] == 1) particle[local_i].stable = 1;
    else particle[local_i].stable = 0;

    //add anti-particle entry
    if (particle[local_i].baryon == 1)
    {
      local_i++;
      particle[local_i].monval = -particle[local_i-1].monval;
      ostringstream antiname;
      antiname << "Anti-" << particle[local_i-1].name;
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
                  cout << "particle monval : " << particle[local_i-1].decays_part[j][k] << endl;
                  exit(1);
                }
              if (particle[idx].baryon == 0 && particle[idx].charge == 0 && particle[idx].strange == 0) particle[local_i].decays_part[j][k] = (particle[local_i-1].decays_part[j][k]);
              else particle[local_i].decays_part[j][k] = (- particle[local_i-1].decays_part[j][k]);
            }
          }
        }
      }
    local_i++;	// Add one to the counting variable "i" for the meson/baryon
  }
resofile.close();
Nparticle = local_i - 1; //take account the final fake one
for (int i = 0; i < Nparticle; i++)
  {
    if (particle[i].baryon == 0) particle[i].sign = -1;
    else particle[i].sign = 1;
  }
return(Nparticle);
}
