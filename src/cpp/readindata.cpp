
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


jonah_coefficients compute_jonah_bulk_coefficients(particle_info * particle_data, int Nparticle)
{
  // computes the z and Lambda coefficients as a function of bulk pressure (assumes muB = 0)
  jonah_coefficients jonah;

  // get the average temperature, energy density, pressure
  Plasma QGP;
  QGP.load_thermodynamic_averages();

  const double T = QGP.temperature;    // GeV
  const double E = QGP.energy_density; // GeV / fm^3
  const double P = QGP.pressure;       // GeV / fm^3

  const double T3 = pow(T,3);
  const double T4 = pow(T,4);


  // gauss laguerre roots and weights
  Gauss_Laguerre gla;
  gla.load_roots_and_weights("tables/gla_roots_weights_32_points.txt");

  const int pbar_pts = gla.points;

  double * pbar_root1 = gla.root[1];
  double * pbar_root2 = gla.root[2];

  double * pbar_weight1 = gla.weight[1];
  double * pbar_weight2 = gla.weight[2];


  // compute the coefficients
  const double lambda_min = -1.0;
  const double lambda_max = 2.0;
  const double delta_lambda = (lambda_max - lambda_min) / ((double)jonah.points - 1.0);

  const double prefactor = T4 / two_pi2_hbarC3;

  double bulkPi_over_Peq_max = -1.0;      // default value

  // calculate the interpolation points of z(bulkPi/P), lambda(bulkPi/P)
  for(int i = 0; i < jonah.points; i++)
  {
    double lambda = lambda_min + (double)i * delta_lambda;

    double E_mod = 0.0;                   // modified energy density
    double P_mod = 0.0;                   // modified pressure

    // calculate modified energy density (sum over hadron resonance contributions)
    //for(int n = 0; i < Nparticle; i++)  // n = 0 is Gamma
    for(int n = 1; n < Nparticle; n++)    // should I skip the photon (Gamma)?
    {
      double degeneracy = (double)particle_data[n].gspin;     // I need to check this at some point
      double mass = particle_data[n].mass;
      double sign = (double)particle_data[n].sign;

      double mbar = mass / T;

      E_mod += prefactor * degeneracy * Gauss1D_mod(E_mod_int, pbar_root2, pbar_weight2, pbar_pts, mbar, lambda, sign);
      P_mod += (prefactor / 3.0) * degeneracy * Gauss1D_mod(P_mod_int, pbar_root2, pbar_weight2, pbar_pts, mbar, lambda, sign);
    }

    // jonah's formula (ignoring detLambda factor i.e. n_mod / n => 1)
    double z = E / E_mod;
    double bulkPi_over_Peq = (P_mod / P) * z  -  1.0;

    // update the max bulk pressure
    bulkPi_over_Peq_max = max(bulkPi_over_Peq_max, bulkPi_over_Peq);

    // load the arrays
    jonah.lambda[i] = lambda;
    jonah.z[i] = z;
    jonah.bulkPi_over_Peq[i] = bulkPi_over_Peq;
  }

  jonah.bulkPi_over_Peq_max = bulkPi_over_Peq_max;

  // now get the cubic spline interpolation coefficients (does it require even spacing?...)
  return jonah;

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
  if (mode == 1) read_surf_VH(length, surf_ptr); //surface file containing viscous hydro dissipative currents
  else if (mode == 2) read_surf_VAH_PLMatch(length, surf_ptr); //surface file containing anisotropic viscous hydro dissipative currents for use with CPU-VAH
  else if (mode == 3) read_surf_VAH_PLPTMatch(length, surf_ptr); //surface file containing anisotropic viscous hydro dissipative currents for use with Mike's Hydro
  else if (mode == 4) read_surf_VH_MUSIC(length, surf_ptr); //boost invariant surface from old (private) version of MUSIC
  else if (mode == 5) read_surf_VH_Vorticity(length, surf_ptr); //surface file containing viscous hydro dissipative currents and thermal vorticity tensor
  else if (mode == 6) read_surf_VH_MUSIC_New(length, surf_ptr); //boost invariant surface from new (public) version of MUSIC
  return;
}

//THIS FORMAT IS DIFFERENT THAN MUSIC 3+1D FORMAT ! baryon number, baryon chemical potential at the end...
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

int FO_data_reader::read_resonances_list(particle_info* particle, FO_surf* surf_ptr, deltaf_coefficients * df)
{
  double eps = 1e-15;
  int Nparticle=0;
  cout << "Reading in particle data group file..." << endl;   // is SMASH format same as the old one?
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

  // get the average temperature, energy density, pressure
  Plasma QGP;
  QGP.load_thermodynamic_averages();

  const double T = QGP.temperature;    // GeV
  const double E = QGP.energy_density; // GeV / fm^3
  const double P = QGP.pressure;       // GeV / fm^3
  const double muB = QGP.baryon_chemical_potential;
  const double nB = QGP.net_baryon_density;

  double alphaB = muB / T;
  double baryon_enthalpy_ratio = nB / (E + P);


  // gauss laguerre roots and weights
  Gauss_Laguerre gla;
  gla.load_roots_and_weights("tables/gla_roots_weights_32_points.txt");

  const int pbar_pts = gla.points;

  double * pbar_root1 = gla.root[1];
  double * pbar_root2 = gla.root[2];
  double * pbar_root3 = gla.root[3];

  double * pbar_weight1 = gla.weight[1];
  double * pbar_weight2 = gla.weight[2];
  double * pbar_weight3 = gla.weight[3];


  // calculate the equilibrium densities and the
  // bulk / diffusion corrections of each particle
  for(int i = 0; i < Nparticle; i++)
  {
    double mass = particle[i].mass;
    double degeneracy = (double)particle[i].gspin;
    double baryon = (double)particle[i].baryon;
    double sign = (double)particle[i].sign;
    double mbar = mass / T;

    // equilibrium density
    double neq_fact = degeneracy * pow(T,3) / two_pi2_hbarC3;
    double neq = neq_fact * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

    // bulk and diffusion density corrections
    double dn_bulk = 0.0;
    double dn_diff = 0.0;

    switch(df_mode)
    {
      case 1: // 14 moment (not sure what the status is)
      {
        double c0 = df->c0;
        double c1 = df->c1;
        double c2 = df->c2;
        double c3 = df->c3;
        double c4 = df->c4;

        double J10_fact = degeneracy * pow(T,3) / two_pi2_hbarC3;
        double J20_fact = degeneracy * pow(T,4) / two_pi2_hbarC3;
        double J30_fact = degeneracy * pow(T,5) / two_pi2_hbarC3;
        double J31_fact = degeneracy * pow(T,5) / two_pi2_hbarC3 / 3.0;

        double J10 =  J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
        double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);
        double J30 = J30_fact * GaussThermal(J30_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);
        double J31 = J31_fact * GaussThermal(J31_int, pbar_root3, pbar_weight3, pbar_pts, mbar, alphaB, baryon, sign);

        dn_bulk = ((c0 - c2) * mass * mass * J10 +  c1 * baryon * J20  +  (4.0 * c2 - c0) * J30);
        // these coefficients need to be loaded.
        // c3 ~ cV / V
        // c4 ~ 2cW / V
        dn_diff = baryon * c3 * neq * T  +  c4 * J31;   // not sure if this is right...
        break;
      }
      case 2: // Chapman-Enskog
      case 3: // Modified
      {
        double F = df->F;
        double G = df->G;
        double betabulk = df->betabulk;
        double betaV = df->betaV;

        double J10_fact = degeneracy * pow(T,3) / two_pi2_hbarC3;
        double J11_fact = degeneracy * pow(T,3) / two_pi2_hbarC3 / 3.0;
        double J20_fact = degeneracy * pow(T,4) / two_pi2_hbarC3;

        double J10 = J10_fact * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
        double J11 = J11_fact * GaussThermal(J11_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);
        double J20 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

        dn_bulk = (neq + (baryon * J10 * G) + (J20 * F / pow(T,2))) / betabulk;
        dn_diff = (neq * T * baryon_enthalpy_ratio  -  baryon * J11) / betaV;

        break;
      }
      default:
      {
        cout << "Please choose df_mode = 1,2,3 in parameters.dat" << endl;
        exit(-1);
      }
    } // df_mode

    particle[i].equilibrium_density = neq;
    particle[i].bulk_density = dn_bulk;
    particle[i].diff_density = dn_diff;
  }

  return Nparticle;
}
