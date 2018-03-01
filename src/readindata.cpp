
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

FO_data_reader::FO_data_reader(ParameterReader* paraRdr_in, string path_in)
{
  paraRdr = paraRdr_in;
  pathToInput = path_in;
  mode = paraRdr->getVal("mode"); //this determines whether the file read in has viscous hydro dissipative currents or viscous anisotropic dissipative currents
  include_baryon = paraRdr->getVal("include_baryon");
  include_bulk_deltaf = paraRdr->getVal("include_bulk_deltaf");
  include_shear_deltaf = paraRdr->getVal("include_shear_deltaf");
  include_baryondiff_deltaf = paraRdr->getVal("include_baryondiff_deltaf");
}

FO_data_reader::~FO_data_reader()
{

}

int FO_data_reader::get_number_cells()
{
  int number_of_cells = 0;
  //add more options for types of FO file
  ostringstream surface_file;
  surface_file << pathToInput << "/surface.dat";
  Table block_file(surface_file.str().c_str());
  number_of_cells = block_file.getNumberOfRows();
  return(number_of_cells);
}

void FO_data_reader::read_surf_switch(long length, FO_surf* surf_ptr)
{
  if (mode == 1) read_surf_VH(length, surf_ptr); //surface file containing viscous hydro dissipative currents
  else if (mode == 2) read_surf_VAH(length, surf_ptr); //surface file containing anisotropic viscous hydro dissipative currents
  return;
}

//THIS FORMAT IS DIFFERENT THAN MUSIC 3+1D FORMAT ! baryon number, baryon chemical potential at the end...
void FO_data_reader::read_surf_VH(long length, FO_surf* surf_ptr)
{
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());
  for (long i = 0; i < length; i++)
  {
    // contravariant spacetime position
    surfdat >> surf_ptr[i].tau;
    surfdat >> surf_ptr[i].x;
    surfdat >> surf_ptr[i].y;
    surfdat >> surf_ptr[i].eta;

    // contravariant surface normal vector
    surfdat >> surf_ptr[i].dat;
    surfdat >> surf_ptr[i].dax;
    surfdat >> surf_ptr[i].day;
    surfdat >> surf_ptr[i].dan;

    // contravariant flow velocity
    surfdat >> surf_ptr[i].ut;
    surfdat >> surf_ptr[i].ux;
    surfdat >> surf_ptr[i].uy;
    surfdat >> surf_ptr[i].un;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;
    surf_ptr[i].E = dummy * hbarC; //energy density
    surfdat >> dummy;
    surf_ptr[i].T = dummy * hbarC; //Temperature
    surfdat >> dummy;
    surf_ptr[i].P = dummy * hbarC; //pressure

    //file formatting may be easier if we force user to leave shear and bulk stresses in freezeout file
    // dissipative quantities at freeze out
    //if (include_shear_deltaf)
    //{
      surfdat >> dummy;
      surf_ptr[i].pitt = dummy * hbarC; //ten contravariant components of shear stress tensor
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
    //}
    //if (include_bulk_deltaf)
    //{
      surfdat >> dummy;
      surf_ptr[i].bulkPi = dummy * hbarC; //bulk pressure
    //}
    if (include_baryon)
    {
      surfdat >> dummy;
      surf_ptr[i].muB = dummy * hbarC; //baryon chemical potential
    }
    if (include_baryondiff_deltaf)
    {
      surfdat >> dummy;
      surf_ptr[i].nB = dummy * hbarC; //baryon density
      surfdat >> dummy;
      surf_ptr[i].Vt = dummy * hbarC; //four contravariant components of baryon diffusion vector
      surfdat >> dummy;
      surf_ptr[i].Vx = dummy * hbarC;
      surfdat >> dummy;
      surf_ptr[i].Vy = dummy * hbarC;
      surfdat >> dummy;
      surf_ptr[i].Vn = dummy * hbarC;
    }

  }
  surfdat.close();
  return;
}

void FO_data_reader::read_surf_VAH(long length, FO_surf* surf_ptr)
{
}

int FO_data_reader::read_resonances_list(particle_info* particle)
{
  double eps = 1e-15;
  int Nparticle=0;
  cout << " -- Read in particle resonance decay table...";
  ifstream resofile("PDG/pdg.dat");
  int local_i = 0;
  int dummy_int;
  while (!resofile.eof())
  {
    resofile >> particle[local_i].mc_id;
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
      particle[local_i].mc_id = -particle[local_i-1].mc_id;
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
                  cout << "particle mc_id : " << particle[local_i-1].decays_part[j][k] << endl;
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
