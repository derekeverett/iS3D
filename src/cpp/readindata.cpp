
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
  dimension = paraRdr->getVal("dimension");
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
  else if (mode == 2) read_surf_VAH_PLMatch(length, surf_ptr); //surface file containing anisotropic viscous hydro dissipative currents for use with CPU-VAH
  else if (mode == 3) read_surf_VAH_PLPTMatch(length, surf_ptr); //usrface file containing anisotropic viscous hydro dissipative currents for use with Mike's Hydro
  else if (mode == 4) read_surf_VH_MUSIC(length, surf_ptr);
  else if (mode == 5) read_surf_VH_Vorticity(length, surf_ptr); //surface file containing viscous hydro dissipative currents and thermal vorticity tensor
  return;
}

//THIS FORMAT IS DIFFERENT THAN MUSIC 3+1D FORMAT ! baryon number, baryon chemical potential at the end...
void FO_data_reader::read_surf_VH(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface vhydro..." << endl;
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

    // COVARIANT surface normal vector
    //note cornelius writes covariant normal vector (used in cpu-ch, gpu-vh,...)
    surfdat >> surf_ptr[i].dat;
    surfdat >> surf_ptr[i].dax;
    surfdat >> surf_ptr[i].day;
    surfdat >> surf_ptr[i].dan;
    // added safeguard for boost invariant read-ins on 7/16
    if(dimension == 2 && surf_ptr[i].dan != 0)
    {
      cout << "2+1d boost invariant surface read-in error at cell # " << i << ": dsigma_eta is not zero. Please fix it to zero." << endl;
      exit(-1);
    }

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

void FO_data_reader::read_surf_VH_Vorticity(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface vhydro with thermal vorticity terms..." << endl;
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

    // COVARIANT surface normal vector
    //note cornelius writes covariant normal vector (used in cpu-ch, gpu-vh,...)
    surfdat >> surf_ptr[i].dat;
    surfdat >> surf_ptr[i].dax;
    surfdat >> surf_ptr[i].day;
    surfdat >> surf_ptr[i].dan;
    // added safeguard for boost invariant read-ins on 7/16
    if(dimension == 2 && surf_ptr[i].dan != 0)
    {
      cout << "2+1d boost invariant surface read-in error at cell # " << i << ": dsigma_eta is not zero. Please fix it to zero." << endl;
      exit(-1);
    }

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

    surfdat >> dummy;
    surf_ptr[i].bulkPi = dummy * hbarC; //bulk pressure

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

    //thermal vorticity w^\mu\nu , should be dimensionless...
    surfdat >> surf_ptr[i].wtx;
    surfdat >> surf_ptr[i].wty;
    surfdat >> surf_ptr[i].wtn;
    surfdat >> surf_ptr[i].wxy;
    surfdat >> surf_ptr[i].wxn;
    surfdat >> surf_ptr[i].wyn;
  }
  surfdat.close();
  return;
}

// 3+1D MUSIC boost invariant format
void FO_data_reader::read_surf_VH_MUSIC(long length, FO_surf* surf_ptr)
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
    surf_ptr[i].eta = 0.0;

    // COVARIANT surface normal vector
    //note cornelius writes covariant normal vector (used in cpu-vh, gpu-vh,...)
    surfdat >> dummy;
    surf_ptr[i].dat = dummy * surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].dax = dummy * surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].day = dummy * surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].dan = dummy * surf_ptr[i].tau;
    if(dimension == 2 && surf_ptr[i].dan != 0)
    {
      cout << "2+1d boost invariant surface read-in error at cell # " << i << ": dsigma_eta is not zero. Please fix it to zero." << endl;
      exit(-1);
    }

    // contravariant flow velocity
    surfdat >> surf_ptr[i].ut;
    surfdat >> surf_ptr[i].ux;
    surfdat >> surf_ptr[i].uy;
    surfdat >> dummy;
    surf_ptr[i].un = dummy / surf_ptr[i].tau;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;
    surf_ptr[i].E = dummy * hbarC;                         // energy density
    surfdat >> dummy;
    surf_ptr[i].T = dummy * hbarC;                         // temperature
    surfdat >> dummy;
    surf_ptr[i].muB = dummy * hbarC;                       // baryon chemical potential
    surfdat >> dummy;                                      // entropy density
    surf_ptr[i].P = dummy * surf_ptr[i].T - surf_ptr[i].E; // p = T*s - e

    // ten contravariant components of shear stress tensor
    surfdat >> dummy;
    surf_ptr[i].pitt = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pitx = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pity = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pitn = dummy * hbarC / surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].pixx = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pixy = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].pixn = dummy * hbarC / surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].piyy = dummy * hbarC;
    surfdat >> dummy;
    surf_ptr[i].piyn = dummy * hbarC / surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].pinn = dummy * hbarC / surf_ptr[i].tau / surf_ptr[i].tau;

    //bulk pressure
    surfdat >> dummy;
    surf_ptr[i].bulkPi = dummy * hbarC;


  }
  surfdat.close();
  return;
}

void FO_data_reader::read_surf_VAH_PLMatch(long FO_length, FO_surf * surface)
{
  // vahydro: Dennis' version
  cout << "Reading in freezeout surface vahydro PL..." << endl;

  ostringstream surface_file;
  surface_file << pathToInput << "/surface.dat";      // stream "input/surface.dat" to surface_file
  ifstream surface_data(surface_file.str().c_str());  // open surface.dat
  // intermediate value which is read from file; need to convert units
  double data;

  // intermediate temperature, equilibrium pressure and longitudinal
  // pressure (fm^-4) used for extracting variables lambda and aL
  double T, P, PL;
  double aL, Lambda;

  // load surface struct
  for(long i = 0; i < FO_length; i++)
  {
    // file format: (x^mu, da_mu, u^mu, E, T, P, pl, pi^munu, W^mu, bulkPi)


    // need to add (pl, Wmu, aL, Lambda) to the struct

    // Make sure all the units are correct

    // contravariant Milne spacetime position
    surface_data >> surface[i].tau; // (fm)
    surface_data >> surface[i].x;   // (fm)
    surface_data >> surface[i].y;   // (fm)
    surface_data >> surface[i].eta; // (1)

    // covariant surface normal vector
    //  * note: cornelius writes covariant normal vector (used in cpu-vh, gpu-vh)
    surface_data >> surface[i].dat; // (fm^3)
    surface_data >> surface[i].dax; // (fm^3)
    surface_data >> surface[i].day; // (fm^3)
    surface_data >> surface[i].dan; // (fm^4)
    if(dimension == 2 && surface[i].dan != 0)
    {
      cout << "2+1d boost invariant surface read-in error at cell # " << i << ": dsigma_eta is not zero. Please fix it to zero." << endl;
      exit(-1);
    }

    // contravariant fluid velocity
    surface_data >> surface[i].ut;  // (1)
    surface_data >> surface[i].ux;  // (1)
    surface_data >> surface[i].uy;  // (1)
    surface_data >> surface[i].un;  // (fm^-1)

    // energy density
    surface_data >> data;
    surface[i].E = data * hbarC;     // (fm^-4 -> GeV/fm^3)

    // temperature
    surface_data >> T;               // store T for (aL,Lambda) calculation
    surface[i].T = T * hbarC;        // (fm^-1 -> GeV)

    // equilibrium pressure
    surface_data >> P;               // store P for (aL,Lambda) calculation
    surface[i].P = P * hbarC;        // (fm^-4 -> GeV/fm^3)

    // longitudinal pressure
    surface_data >> PL;              // store pl for (aL,Lambda) calculation
    surface[i].PL = PL * hbarC;      // (fm^-4 -> GeV/fm^3)

    // contravariant transverse shear stress (pi^munu == pi_perp^munu)
    surface_data >> data;
    surface[i].pitt = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pitx = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pity = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pitn = data * hbarC;  // (fm^-5 -> GeV/fm^4)
    surface_data >> data;
    surface[i].pixx = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pixy = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pixn = data * hbarC;  // (fm^-5 -> GeV/fm^4)
    surface_data >> data;
    surface[i].piyy = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].piyn = data * hbarC;  // (fm^-5 -> GeV/fm^4)
    surface_data >> data;
    surface[i].pinn = data * hbarC;  // (fm^-6 -> GeV/fm^5)

    // contravariant longitudinal momentum diffusion current (W^mu == W_perpz^mu)
    surface_data >> data;
    surface[i].Wt = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].Wx = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].Wy = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].Wn = data * hbarC;  // (fm^-5 -> GeV/fm^4)

    surface_data >> data;
    surface[i].bulkPi = data * hbarC;   // (fm^-4 -> GeV/fm^3)


    // infer anisotropic variables: conformal factorization approximation
    if((PL / P) < 3.0)
    {
      aL = aL_fit(PL / P);
      Lambda = (T / pow(0.5 * aL * R200(aL), 0.25));

      surface[i].aL = aL;
      surface[i].Lambda = Lambda * hbarC;    // (fm^-1 -> GeV)
    }
    else
    {
      cout << "pl is too large, stopping anisotropic variables..." << endl;
      exit(-1);
    }


  } // i
  // close file
  surface_data.close();
  return;
}

void FO_data_reader::read_surf_VAH_PLPTMatch(long FO_length, FO_surf * surface)
{
  // vahydro: mike's version (which is better ;)
  ostringstream surface_file;
  surface_file << pathToInput << "/surface.dat";      // stream "input/surface.dat" to surface_file
  ifstream surface_data(surface_file.str().c_str());  // open surface.dat

  // intermediate data value (for converting to correct units)
  double data;

  // load surface struct
  for(long i = 0; i < FO_length; i++)
  {
    // surface.dat file format (columns):
    // (x^mu, da_mu, u^mu, e, T, pl, pt, pi^munu, W^mu, Lambda, at, al, muB, upsilonB, nB, nBl, V^mu)

    // need to make mode option for FO_surf struct

    // Make sure all the units are correct

    // contravariant Milne spacetime position
    surface_data >> surface[i].tau; // (fm)
    surface_data >> surface[i].x;   // (fm)
    surface_data >> surface[i].y;   // (fm)
    surface_data >> surface[i].eta; // (1)

    // covariant surface normal vector
    //  * note: cornelius writes covariant normal vector (used in cpu-vh, gpu-vh)
    surface_data >> surface[i].dat; // (fm^3)
    surface_data >> surface[i].dax; // (fm^3)
    surface_data >> surface[i].day; // (fm^3)
    surface_data >> surface[i].dan; // (fm^4)
    if(dimension == 2 && surface[i].dan != 0)
    {
      cout << "2+1d boost invariant surface read-in error at cell # " << i << ": dsigma_eta is not zero. Please fix it to zero." << endl;
      exit(-1);
    }

    // contravariant fluid velocity
    surface_data >> surface[i].ut;  // (1)
    surface_data >> surface[i].ux;  // (1)
    surface_data >> surface[i].uy;  // (1)
    surface_data >> surface[i].un;  // (fm^-1)

    // energy density and temperature
    surface_data >> data;
    surface[i].E = data * hbarC; // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].T = data * hbarC; // (fm^-1 -> GeV)

    // longitudinal and transverse pressures
    surface_data >> data;
    surface[i].PL = data * hbarC; // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].PT = data * hbarC; // (fm^-4 -> GeV/fm^3)

    // contravariant transverse shear stress (pi^munu == pi_perp^munu)
    surface_data >> data;
    surface[i].pitt = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pitx = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pity = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pitn = data * hbarC;  // (fm^-5 -> GeV/fm^4)
    surface_data >> data;
    surface[i].pixx = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pixy = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].pixn = data * hbarC;  // (fm^-5 -> GeV/fm^4)
    surface_data >> data;
    surface[i].piyy = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].piyn = data * hbarC;  // (fm^-5 -> GeV/fm^4)
    surface_data >> data;
    surface[i].pinn = data * hbarC;  // (fm^-6 -> GeV/fm^5)

    // contravariant longitudinal momentum diffusion current (W^mu == W_perpz^mu)
    surface_data >> data;
    surface[i].Wt = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].Wx = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].Wy = data * hbarC;  // (fm^-4 -> GeV/fm^3)
    surface_data >> data;
    surface[i].Wn = data * hbarC;  // (fm^-5 -> GeV/fm^4)

    // effective temperature
    surface_data >> data;
    surface[i].Lambda = data * hbarC;   // (fm^-1 -> GeV)

    // transverse and longitudinal momentum deformation scales
    surface_data >> surface[i].aT;      // (1)
    surface_data >> surface[i].aL;      // (1)

    if(include_baryon)
    {
      // baryon chemical potential
      surface_data >> data;
      surface[i].muB = data * hbarC;      // (fm^-1 -> GeV)

      // effective baryon chemical potential
      surface_data >> data;
      surface[i].upsilonB = data * hbarC; // (fm^-1 -> GeV)
    }

    if(include_baryondiff_deltaf)
    {
      // net baryon density
      surface_data >> data;
      surface[i].nB = data * hbarC;   // (fm^-3 -> GeV/fm^2)

      // LRF longitudinal baryon diffusion
      surface_data >> data;
      surface[i].nBL = data * hbarC;  // (fm^-3 -> GeV/fm^2)

      // contravariant transverse baryon diffusion (V^mu == V_perp^mu)
      surface_data >> data;
      surface[i].Vt = data * hbarC;   // (fm^-3 -> GeV/fm^2)
      surface_data >> data;
      surface[i].Vx = data * hbarC;   // (fm^-3 -> GeV/fm^2)
      surface_data >> data;
      surface[i].Vy = data * hbarC;   // (fm^-3 -> GeV/fm^2)
    }
  } // i
  // close file
  surface_data.close();
  return;
}

int FO_data_reader::read_resonances_list(particle_info* particle)
{
  double eps = 1e-15;
  int Nparticle=0;
  cout << "Reading in particle data group file..." << endl;
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
