
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>

#include "iS3D.h"
#include "readindata.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "Table.h"
#include "gaussThermal.h"

using namespace std;

Gauss_Laguerre::Gauss_Laguerre()
{
  /////////////////////////
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

Gauss_Legendre::Gauss_Legendre()
{
  /////////////////////////
}

void Gauss_Legendre::load_roots_and_weights(string file_name)
{
  stringstream file;
  file << file_name;

  FILE * gauss_file = fopen(file.str().c_str(), "r");

  if(gauss_file == NULL) printf("Error: couldn't open gauss legendre file\n");

  fscanf(gauss_file, "%d", &points);

  // allocate memory for the roots and weights
  root = (double *)calloc(points, sizeof(double));
  weight = (double *)calloc(points, sizeof(double));

  // load the arrays
  for(int i = 0; i < points; i++)
  {
    fscanf(gauss_file, "%lf\t%lf", &root[i], &weight[i]);
  }

  fclose(gauss_file);
}

Plasma::Plasma()
{
  /////////////////////////
}

void Plasma::load_thermodynamic_averages()
{
  // reads the averaged thermodynamic quantities of the freezeout surface
  // ** assumes the file has already been created in read_surf_switch

  FILE * thermodynamic_file = fopen("average_thermodynamic_quantities.dat", "r");

  if(thermodynamic_file == NULL) printf("Error opening average thermodynamic file\n");
  fscanf(thermodynamic_file, "%lf\n%lf\n%lf\n%lf\n%lf", &temperature, &energy_density, &pressure, &baryon_chemical_potential, &net_baryon_density);
  fclose(thermodynamic_file);
}


FO_data_reader::FO_data_reader(ParameterReader* paraRdr_in, string path_in)
{
  paraRdr = paraRdr_in;
  pathToInput = path_in;
  mode = paraRdr->getVal("mode"); //this determines whether the file read in has viscous hydro dissipative currents or viscous anisotropic dissipative currents
  dimension = paraRdr->getVal("dimension");
  df_mode = paraRdr->getVal("df_mode");
  include_baryon = paraRdr->getVal("include_baryon");
  include_bulk_deltaf = paraRdr->getVal("include_bulk_deltaf");
  include_shear_deltaf = paraRdr->getVal("include_shear_deltaf");
  include_baryondiff_deltaf = paraRdr->getVal("include_baryondiff_deltaf");
}


FO_data_reader::~FO_data_reader()
{
  //////////////////////////////
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
  if      (mode == 0) read_surf_VH_old(length, surf_ptr); //old GPU-VH surface file containing viscous hydro dissipative currents
  else if (mode == 1) read_surf_VH(length, surf_ptr); //GPU-VH surface file containing viscous hydro dissipative currents
  else if (mode == 2) read_surf_VAH_PLMatch(length, surf_ptr); //CPU-VAH surface file containing anisotropic viscous hydro dissipative currents for use with CPU-VAH
  else if (mode == 3) read_surf_VAH_PLPTMatch(length, surf_ptr); //surface file containing anisotropic viscous hydro dissipative currents for use with Mike's Hydro
  else if (mode == 4) read_surf_VH_MUSIC(length, surf_ptr); //boost invariant surface from old (private) version of MUSIC
  else if (mode == 5) read_surf_VH_Vorticity(length, surf_ptr); //GPU-VH surface file containing viscous hydro dissipative currents and thermal vorticity tensor
  else if (mode == 6) read_surf_VH_MUSIC_New(length, surf_ptr); //boost invariant surface from new (public) version of MUSIC
  else if (mode == 7) read_surf_VH_hiceventgen(length, surf_ptr); //boost invariant surface produced by the hydro module in https://github.com/Duke-QCD/hic-eventgen
  return;
}

//THIS FORMAT IS DIFFERENT THAN MUSIC 3+1D FORMAT ! baryon number, baryon chemical potential at the end...
// THIS IS THE OLD FORMAT
void FO_data_reader::read_surf_VH_old(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface in old CPU-VH format" << endl;
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());

  // average thermodynamic quantities on surface
  double Tavg = 0.0;
  double Eavg = 0.0;
  double Pavg = 0.0;
  double muBavg = 0.0;
  double nBavg = 0.0;
  double total_surface_volume = 0.0;

  // average viscous quantities
  double bulkPiavg = 0.0;
  double pimunuavg = 0.0;

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
    double E = dummy * hbarC; //energy density
    surf_ptr[i].E = E;

    surfdat >> dummy;
    double T = dummy * hbarC; //temperature
    surf_ptr[i].T = T;

    surfdat >> dummy;
    double P = dummy * hbarC; //pressure
    surf_ptr[i].P = P;

    surfdat >> dummy;
    double pitt = dummy * hbarC;
    surf_ptr[i].pitt = pitt;

    surfdat >> dummy;
    double pitx = dummy * hbarC;
    surf_ptr[i].pitx = pitx;

    surfdat >> dummy;
    double pity = dummy * hbarC;
    surf_ptr[i].pity = pity;

    surfdat >> dummy;
    double pitn = dummy * hbarC;
    surf_ptr[i].pitn = pitn;

    surfdat >> dummy;
    double pixx = dummy * hbarC;
    surf_ptr[i].pixx = pixx;

    surfdat >> dummy;
    double pixy = dummy * hbarC;
    surf_ptr[i].pixy = pixy;

    surfdat >> dummy;
    double pixn = dummy * hbarC;
    surf_ptr[i].pixn = pixn;

    surfdat >> dummy;
    double piyy = dummy * hbarC;
    surf_ptr[i].piyy = piyy;

    surfdat >> dummy;
    double piyn = dummy * hbarC;
    surf_ptr[i].piyn = piyn;

    surfdat >> dummy;
    double pinn = dummy * hbarC;
    surf_ptr[i].pinn = pinn;

    surfdat >> dummy;
    double bulkPi = dummy * hbarC;
    surf_ptr[i].bulkPi = bulkPi; // bulk pressure

    double muB = 0.0; // nonzero if include_baryon
    double nB = 0.0;  // effectively defaulted to zero if no diffusion correction needed  (even if include_baryon)

    if (include_baryon)
    {
      surfdat >> dummy;
      muB = dummy * hbarC;              // baryon chemical potential
      surf_ptr[i].muB = muB;
    }
    if (include_baryondiff_deltaf)
    {
      surfdat >> nB;                    // baryon density
      surf_ptr[i].nB = nB;
      surfdat >> surf_ptr[i].Vt;        // four contravariant components of baryon diffusion vector
      surfdat >> surf_ptr[i].Vx;
      surfdat >> surf_ptr[i].Vy;
      surfdat >> surf_ptr[i].Vn;        // fixed units on 10/8 (overlooked)
    }

    // getting average thermodynamic quantities
    double tau = surf_ptr[i].tau;
    double tau2 = tau * tau;
    double ux = surf_ptr[i].ux;
    double uy = surf_ptr[i].uy;
    double un = surf_ptr[i].un;
    double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau * tau * un * un);  // enforce normalization
    double dat = surf_ptr[i].dat;
    double dax = surf_ptr[i].dax;
    double day = surf_ptr[i].day;
    double dan = surf_ptr[i].dan;

    double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;
    double dsigma_dsigma = dat * dat  -  dax * dax  -  day * day  -  dan * dan / (tau * tau);

    // maybe this isn't the right formula
    double dsigma_magnitude = fabs(udsigma) + sqrt(fabs(udsigma * udsigma  -  dsigma_dsigma));

    total_surface_volume += dsigma_magnitude;

    Eavg += (E * dsigma_magnitude);
    Tavg += (T * dsigma_magnitude);
    Pavg += (P * dsigma_magnitude);
    muBavg += (muB * dsigma_magnitude);
    nBavg += (nB * dsigma_magnitude);

    pimunuavg += (sqrt(fabs(pitt*pitt - 2.0*pitx*pitx - 2.0*pity*pity - 2.0*tau2*pitn*pitn + pixx*pixx + 2.0*pixy*pixy + 2.0*tau2*pixn*pixn + piyy*piyy + 2.0*tau2*piyn*piyn + tau2*tau2*pinn*pinn)) * dsigma_magnitude);
    bulkPiavg += (bulkPi * dsigma_magnitude);

  }
  surfdat.close();

  Tavg /= total_surface_volume;
  Eavg /= total_surface_volume;
  Pavg /= total_surface_volume;
  muBavg /= total_surface_volume;
  nBavg /= total_surface_volume;

  pimunuavg /= total_surface_volume;
  bulkPiavg /= total_surface_volume;

  // write averaged thermodynamic quantities to file
  ofstream thermal_average("average_thermodynamic_quantities.dat", ios_base::out);
  thermal_average << setprecision(15) << Tavg << "\n" << Eavg << "\n" << Pavg << "\n" << muBavg << "\n" << nBavg;
  thermal_average.close();
  return;
}

void FO_data_reader::read_surf_VH(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface in CPU-VH format" << endl;
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());

  // average thermodynamic quantities on surface
  double Tavg = 0.0;
  double Eavg = 0.0;
  double Pavg = 0.0;
  double muBavg = 0.0;
  double nBavg = 0.0;
  double total_surface_volume = 0.0;

  // average viscous quantities
  double bulkPiavg = 0.0;
  double pimunuavg = 0.0;

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
      //exit(-1);
    }

    // contravariant flow velocity
    surfdat >> surf_ptr[i].ux;
    surfdat >> surf_ptr[i].uy;
    surfdat >> surf_ptr[i].un;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;
    double E = dummy * hbarC; //energy density
    surf_ptr[i].E = E;

    surfdat >> dummy;
    double T = dummy * hbarC; //temperature
    surf_ptr[i].T = T;

    surfdat >> dummy;
    double P = dummy * hbarC; //pressure
    surf_ptr[i].P = P;

    surfdat >> dummy;
    double pixx = dummy * hbarC;
    surf_ptr[i].pixx = pixx;

    surfdat >> dummy;
    double pixy = dummy * hbarC;
    surf_ptr[i].pixy = pixy;

    surfdat >> dummy;
    double pixn = dummy * hbarC;
    surf_ptr[i].pixn = pixn;

    surfdat >> dummy;
    double piyy = dummy * hbarC;
    surf_ptr[i].piyy = piyy;

    surfdat >> dummy;
    double piyn = dummy * hbarC;
    surf_ptr[i].piyn = piyn;

    surfdat >> dummy;
    double bulkPi = dummy * hbarC;
    surf_ptr[i].bulkPi = bulkPi; // bulk pressure

    double muB = 0.0; // nonzero if include_baryon
    double nB = 0.0;  // effectively defaulted to zero if no diffusion correction needed  (even if include_baryon)

    if (include_baryon)
    {
      surfdat >> dummy;
      muB = dummy * hbarC;              // baryon chemical potential
      surf_ptr[i].muB = muB;
    }
    if (include_baryondiff_deltaf)
    {
      surfdat >> nB;                    // baryon density
      surf_ptr[i].nB = nB;
      //surfdat >> surf_ptr[i].Vt;        // four contravariant components of baryon diffusion vector
      surfdat >> surf_ptr[i].Vx;
      surfdat >> surf_ptr[i].Vy;
      surfdat >> surf_ptr[i].Vn;        // fixed units on 10/8 (overlooked)
    }

    // getting average thermodynamic quantities
    double tau = surf_ptr[i].tau;
    double tau2 = tau * tau;
    double ux = surf_ptr[i].ux;
    double uy = surf_ptr[i].uy;
    double un = surf_ptr[i].un;
    double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau * tau * un * un);  // enforce normalization
    double dat = surf_ptr[i].dat;
    double dax = surf_ptr[i].dax;
    double day = surf_ptr[i].day;
    double dan = surf_ptr[i].dan;

    double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;
    double dsigma_dsigma = dat * dat  -  dax * dax  -  day * day  -  dan * dan / (tau * tau);

    // maybe this isn't the right formula
    double dsigma_magnitude = fabs(udsigma) + sqrt(fabs(udsigma * udsigma  -  dsigma_dsigma));

    total_surface_volume += dsigma_magnitude;

    Eavg += (E * dsigma_magnitude);
    Tavg += (T * dsigma_magnitude);
    Pavg += (P * dsigma_magnitude);
    muBavg += (muB * dsigma_magnitude);
    nBavg += (nB * dsigma_magnitude);

    //pimunuavg += (sqrt(fabs(pitt*pitt - 2.0*pitx*pitx - 2.0*pity*pity - 2.0*tau2*pitn*pitn + pixx*pixx + 2.0*pixy*pixy + 2.0*tau2*pixn*pixn + piyy*piyy + 2.0*tau2*piyn*piyn + tau2*tau2*pinn*pinn)) * dsigma_magnitude);
    //bulkPiavg += (bulkPi * dsigma_magnitude);

  }
  surfdat.close();

  Tavg /= total_surface_volume;
  Eavg /= total_surface_volume;
  Pavg /= total_surface_volume;
  muBavg /= total_surface_volume;
  nBavg /= total_surface_volume;

  pimunuavg /= total_surface_volume;
  bulkPiavg /= total_surface_volume;

  // write averaged thermodynamic quantities to file
  ofstream thermal_average("average_thermodynamic_quantities.dat", ios_base::out);
  thermal_average << setprecision(15) << Tavg << "\n" << Eavg << "\n" << Pavg << "\n" << muBavg << "\n" << nBavg;
  thermal_average.close();
  return;
}

void FO_data_reader::read_surf_VH_Vorticity(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface in CPU-VH format with thermal vorticity terms" << endl;
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
      //exit(-1);
    }

    // contravariant flow velocity
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
    surf_ptr[i].bulkPi = dummy * hbarC; //bulk pressure

    if (include_baryon)
    {
      surfdat >> dummy;
      surf_ptr[i].muB = dummy * hbarC; //baryon chemical potential
    }
    if (include_baryondiff_deltaf)
    {
      surfdat >> surf_ptr[i].nB;       //baryon density
      surfdat >> surf_ptr[i].Vt;       //four contravariant components of baryon diffusion vector
      surfdat >> surf_ptr[i].Vx;
      surfdat >> surf_ptr[i].Vy;
      surfdat >> surf_ptr[i].Vn;       // fixed units on 10/8 (overlooked)
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

// old MUSIC boost invariant format
void FO_data_reader::read_surf_VH_MUSIC(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface in (old) MUSIC boost invariant format" << endl;
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());

  // average thermodynamic quantities on surface
  double Tavg = 0.0;
  double Eavg = 0.0;
  double Pavg = 0.0;
  double muBavg = 0.0;
  double nBavg = 0.0;
  double total_surface_volume = 0.0;

  double Tmin = 1.0;

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
    if (dimension == 2 && i == 0 && surf_ptr[i].dan != 0)
    {
      cout << "dsigma_eta is not zero in first cell. Setting all dsigma_eta to zero!" << endl;
      surf_ptr[i].dan = 0.0;
    }
    if (dimension == 2 && surf_ptr[i].dan != 0) surf_ptr[i].dan = 0.0;

    // contravariant flow velocity
    surfdat >> surf_ptr[i].ut;
    surfdat >> surf_ptr[i].ux;
    surfdat >> surf_ptr[i].uy;
    surfdat >> dummy;
    surf_ptr[i].un = dummy / surf_ptr[i].tau;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;
    double E = dummy * hbarC;
    surf_ptr[i].E = E;                         // energy density
    surfdat >> dummy;
    double T = dummy * hbarC;
    surf_ptr[i].T = T;                         // temperature

    if(Tmin > T) Tmin = T;

    surfdat >> dummy;
    double muB = dummy * hbarC;
    surf_ptr[i].muB = muB;                       // baryon chemical potential
    surfdat >> dummy;                            // entropy density
    double P = dummy * T - E;
    surf_ptr[i].P = P; // p = T*s - e

    double nB = 0.0;

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

    // getting average thermodynamic quantities
    double tau = surf_ptr[i].tau;
    double ux = surf_ptr[i].ux;
    double uy = surf_ptr[i].uy;
    double un = surf_ptr[i].un;
    double ut = sqrt(1.0 + ux * ux + uy * uy + tau * tau * un * un);  // enforce normalization
    double dat = surf_ptr[i].dat;
    double dax = surf_ptr[i].dax;
    double day = surf_ptr[i].day;
    double dan = surf_ptr[i].dan;

    double udsigma = ut * dat + ux * dax + uy * day + un * dan;
    double dsigma_dsigma = dat * dat - dax * dax - day * day - dan * dan / (tau * tau);
    double dsigma_magnitude = fabs(udsigma) + sqrt(fabs(udsigma * udsigma - dsigma_dsigma));

    total_surface_volume += dsigma_magnitude;

    Eavg += (E * dsigma_magnitude);
    Tavg += (T * dsigma_magnitude);
    Pavg += (P * dsigma_magnitude);
    muBavg += (muB * dsigma_magnitude);
    nBavg += (nB * dsigma_magnitude);
  }
  surfdat.close();

  Tavg /= total_surface_volume;
  Eavg /= total_surface_volume;
  Pavg /= total_surface_volume;
  muBavg /= total_surface_volume;
  nBavg /= total_surface_volume;

  // write averaged thermodynamic quantities to file
  ofstream thermal_average("average_thermodynamic_quantities.dat", ios_base::out);
  thermal_average << setprecision(15) << Tavg << "\n" << Eavg << "\n" << Pavg << "\n" << muBavg << "\n" << nBavg;
  thermal_average.close();

  return;
}

// New public MUSIC version boost invariant format
void FO_data_reader::read_surf_VH_MUSIC_New(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface in (new) public MUSIC boost invariant format" << endl;
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());

  // average thermodynamic quantities on surface
  double Tavg = 0.0;
  double Eavg = 0.0;
  double Pavg = 0.0;
  double muBavg = 0.0;
  double nBavg = 0.0;
  double total_surface_volume = 0.0;

  for (long i = 0; i < length; i++)
  {
    // contravariant spacetime position
    surfdat >> surf_ptr[i].tau;
    surfdat >> surf_ptr[i].x;
    surfdat >> surf_ptr[i].y;
    surfdat >> dummy;
    surf_ptr[i].eta = 0.0;

    surfdat >> dummy;
    surf_ptr[i].dat = dummy * surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].dax = dummy * surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].day = dummy * surf_ptr[i].tau;
    surfdat >> dummy;
    surf_ptr[i].dan = 0.0;

    // contravariant flow velocity
    surfdat >> surf_ptr[i].ut;
    surfdat >> surf_ptr[i].ux;
    surfdat >> surf_ptr[i].uy;
    surfdat >> dummy;
    surf_ptr[i].un = dummy / surf_ptr[i].tau;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;
    double E = dummy * hbarC;
    surf_ptr[i].E = E;                          // energy density
    surfdat >> dummy;
    double T = dummy * hbarC;
    surf_ptr[i].T = T;                          // temperature
    surfdat >> dummy;
    surf_ptr[i].muB = dummy * hbarC;;           // baryon chemical potential
    surfdat >> dummy;                           // strangeness chemical potential
    surfdat >> dummy;                           // charm chemical potential
    surfdat >> dummy;                           // entropy density (e + P) / T
    double P = dummy * T - E;
    surf_ptr[i].P = P; // p = T*s - e

    double nB = 0.0;

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

    // getting average thermodynamic quantities
    double tau = surf_ptr[i].tau;
    double ux = surf_ptr[i].ux;
    double uy = surf_ptr[i].uy;
    double un = surf_ptr[i].un;
    double ut = sqrt(1.0 + ux * ux + uy * uy + tau * tau * un * un);  // enforce normalization
    double dat = surf_ptr[i].dat;
    double dax = surf_ptr[i].dax;
    double day = surf_ptr[i].day;
    double dan = surf_ptr[i].dan;
    double muB = surf_ptr[i].muB;

    double udsigma = ut * dat + ux * dax + uy * day + un * dan;
    double dsigma_dsigma = dat * dat - dax * dax - day * day - dan * dan / (tau * tau);
    double dsigma_magnitude = fabs(udsigma) + sqrt(fabs(udsigma * udsigma - dsigma_dsigma));

    total_surface_volume += dsigma_magnitude;

    Eavg += (E * dsigma_magnitude);
    Tavg += (T * dsigma_magnitude);
    Pavg += (P * dsigma_magnitude);
    muBavg += (muB * dsigma_magnitude);
    nBavg += (nB * dsigma_magnitude);
  }
  surfdat.close();

  Tavg /= total_surface_volume;
  Eavg /= total_surface_volume;
  Pavg /= total_surface_volume;
  muBavg /= total_surface_volume;
  nBavg /= total_surface_volume;

  // write averaged thermodynamic quantities to file
  ofstream thermal_average("average_thermodynamic_quantities.dat", ios_base::out);
  thermal_average << setprecision(15) << Tavg << "\n" << Eavg << "\n" << Pavg << "\n" << muBavg << "\n" << nBavg;
  thermal_average.close();

  return;
}


void FO_data_reader::read_surf_VAH_PLMatch(long FO_length, FO_surf * surface)
{
  // vahydro: Dennis' version
  cout << "Reading in freezeout surface in VAH P_L matching format" << endl;

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
      //exit(-1);
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
  cout << "Reading in freezeout surface in VAH P_L, P_T matching format" << endl;
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

// hiceventgen boost invariant format
void FO_data_reader::read_surf_VH_hiceventgen(long length, FO_surf* surf_ptr)
{
  cout << "Reading in freezeout surface in hiceventgen boost invariant format" << endl;
  ostringstream surfdat_stream;
  double dummy;
  surfdat_stream << pathToInput << "/surface.dat";
  ifstream surfdat(surfdat_stream.str().c_str());

  //skip the one line header

  // average thermodynamic quantities on surface
  double Tavg = 0.0;
  double Eavg = 0.0;
  double Pavg = 0.0;
  double muBavg = 0.0;
  double nBavg = 0.0;
  double total_surface_volume = 0.0;

  for (long i = 0; i < length; i++)
  {
    // contravariant spacetime position
    surfdat >> surf_ptr[i].tau;
    surfdat >> surf_ptr[i].x;
    surfdat >> surf_ptr[i].y;
    surfdat >> dummy; //eta
    surf_ptr[i].eta = 0.0;

    double tau = surf_ptr[i].tau;

    //covariant surface normal (Missing jacobian factor of tau in surface file!)
    surfdat >> dummy; // da_tau / tau
    surf_ptr[i].dat = dummy * tau;
    surfdat >> dummy; // da_x / tau
    surf_ptr[i].dax = dummy * tau;
    surfdat >> dummy; // da_y / tau
    surf_ptr[i].day = dummy * tau;
    surfdat >> dummy;  // da_eta / tau
    surf_ptr[i].dan = 0.0;

    // covariant flow velocity
    double vx, vy, vn;
    double ut, ux, uy, un;

    surfdat >> vx;
    surfdat >> vy;
    surfdat >> vn;

    vn = 0.0;

    double denom =  1.0 - (vx * vx) - (vy * vy);
    if (denom < 0.0) cout << "1.0 - vx*vx - vy*vy - vn*vn/tau/tau < 0 !" << endl;

    ut = sqrt( 1.0 / denom );
    surf_ptr[i].ux = ut * vx;
    surf_ptr[i].uy = ut * vy;
    surf_ptr[i].un = 0.0;

     // ten contravariant components of shear stress tensor
    //units are GeV/fm^3
    surfdat >> dummy; // pi^tt
    //surf_ptr[i].pitt = dummy;
    surfdat >> dummy; // pi^tx
    //surf_ptr[i].pitx = dummy;
    surfdat >> dummy; // pi^ty
    //surf_ptr[i].pity = dummy;
    surfdat >> dummy; // pi^tz
    //surf_ptr[i].pitn = dummy;

    // at eta_s = 0, pi^xn = pi^xz / tau, same for pi^yn ...
    surfdat >> dummy; // pi^xx
    surf_ptr[i].pixx = dummy;
    surfdat >> dummy; // pi^xy
    surf_ptr[i].pixy = dummy;
    surfdat >> dummy; // pi^xz
    surf_ptr[i].pixn = dummy / tau;
    surfdat >> dummy; // pi^yy
    surf_ptr[i].piyy = dummy;
    surfdat >> dummy; // pi^yz
    surf_ptr[i].piyn = dummy / tau;

    surfdat >> dummy; // pi^zz
    //surf_ptr[i].pinn = dummy;

    //bulk pressure
    surfdat >> dummy;
    surf_ptr[i].bulkPi = dummy;

    // thermodynamic quantities at freeze out
    surfdat >> dummy;   // temperature [GeV]
    double T = dummy;
    surf_ptr[i].T = T;

    surfdat >> dummy;   // energy density [GeV/fm^3]
    double E = dummy;
    surf_ptr[i].E = E;

    surfdat >> dummy;   // pressure [GeV/fm^3]
    double P = dummy;
    surf_ptr[i].P = P;

    surfdat >> dummy;    // baryon chemical potential [GeV/fm^33]
    double muB = dummy;
    surf_ptr[i].muB = muB;

    double nB = 0.0;

    double dat = surf_ptr[i].dat;
    double dax = surf_ptr[i].dax;
    double day = surf_ptr[i].day;
    double dan = surf_ptr[i].dan;

    double udsigma = ut * dat + ux * dax + uy * day + un * dan;
    double dsigma_dsigma = dat * dat - dax * dax - day * day - dan * dan / (tau * tau);
    double dsigma_magnitude = fabs(udsigma) + sqrt(fabs(udsigma * udsigma - dsigma_dsigma));

    total_surface_volume += dsigma_magnitude;

    Eavg += (E * dsigma_magnitude);
    Tavg += (T * dsigma_magnitude);
    Pavg += (P * dsigma_magnitude);
    muBavg += (muB * dsigma_magnitude);
    nBavg += (nB * dsigma_magnitude);
  }
  surfdat.close();

  Tavg /= total_surface_volume;
  Eavg /= total_surface_volume;
  Pavg /= total_surface_volume;
  muBavg /= total_surface_volume;
  nBavg /= total_surface_volume;

  // write averaged thermodynamic quantities to file
  ofstream thermal_average("average_thermodynamic_quantities.dat", ios_base::out);
  thermal_average << setprecision(15) << Tavg << "\n" << Eavg << "\n" << Pavg << "\n" << muBavg << "\n" << nBavg;
  thermal_average.close();

  return;
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


PDG_Data::PDG_Data(ParameterReader * paraRdr_in)
{
  paraRdr = paraRdr_in;
  hrg_eos = paraRdr->getVal("hrg_eos"); 
}


PDG_Data::~PDG_Data()
{
  //////////////////////////////
}


int PDG_Data::read_resonances_conventional(particle_info * particle, string pdg_filename)
{
  double eps = 1e-15;
  int Nparticle = 0;
  printf("Loading particle info from urqmd/smash pdg file (check that only 1 blank line at end of file)\n");
  printf("...using read_resonances_conventional() \n");
  ifstream resofile(pdg_filename);
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

    if (particle[local_i].decays > Maxdecaychannel)
      {
	cout << "Error : particle[local_i].decays = "<< particle[local_i].decays << "; local_i = " << local_i << endl;
	exit(1);
      }

    for (int j = 0; j < particle[local_i].decays; j++)
    {
      resofile >> dummy_int;
      resofile >> particle[local_i].decays_Npart[j];
      if (particle[local_i].decays_Npart[j] > Maxdecaypart)
	{
	  cout << "Error : particle[local_i].decays_Npart[j] = "<< particle[local_i].decays_Npart[j] << "; local_i = " << local_i << "; j = " << j << endl;
	} 
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
    if (particle[local_i].baryon > 0) // changed on Feb. 2019
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
	      cout << "local_i = " << local_i << endl;
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
    //if (particle[i].baryon == 0) particle[i].sign = -1; // this made deuteron a fermion
    if (particle[i].baryon % 2 == 0) particle[i].sign = -1;
    else particle[i].sign = 1;
  }

  // count the number of mesons, baryons and antibaryons
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

  printf("\nThere are %d resonances: %d mesons, %d baryons and %d antibaryons\n\n", Nparticle, meson, baryon, antibaryon);

  particle_info last_particle = particle[Nparticle - 1];

  printf("Last particle: mcid = %ld, %s, m = %lf GeV (please check this) \n\n", last_particle.mc_id, last_particle.name.c_str(), last_particle.mass);

  return Nparticle;
}


int PDG_Data::read_resonances_smash_box(particle_info * particle, string pdg_filename)
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

  ifstream pdg_smash_box(pdg_filename);

  int i = 0;  // particle index

  while(getline(pdg_smash_box, line))
  {
      istringstream current_line(line);

      // add mc_id[..] if increase mcid_entries
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

  pdg_smash_box.close();

  free(mc_id);

  int Nparticle = i;  // total number of resonances

  // count the number of mesons, baryons and antibaryons
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

  printf("\nThere are %d resonances: %d mesons, %d baryons and %d antibaryons\n\n", Nparticle, meson, baryon, antibaryon);

  particle_info last_particle = particle[Nparticle - 1];

  printf("Last particle: mcid = %ld, %s, m = %lf GeV (please check this) \n\n", last_particle.mc_id, last_particle.name.c_str(), last_particle.mass);

  return Nparticle;
}


int PDG_Data::read_resonances(particle_info * particle)
{
  int Nparticle;

  switch(hrg_eos)
  {
    case 1:
    {
      Nparticle = read_resonances_conventional(particle, urqmd);
      break;
    }
    case 2:
    {
      Nparticle = read_resonances_conventional(particle, smash);
      break;
    }
    case 3:
    {
      Nparticle = read_resonances_smash_box(particle, smash_box);
      break;
    }
    default:
    {
      printf("Error: please choose hrg_eos = (1,2,3)\n");
      exit(-1);
    }
  }

  return Nparticle;
}

