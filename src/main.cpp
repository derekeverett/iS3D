
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<math.h>
#include<vector>
#include<sys/time.h>

#include "main.h"
#include "Table.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "arsenal.h"
#include "ParameterReader.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "Welcome to iS3D, a program to compute particle spectra from 3+1D Hydro Freezeout Surfaces!" << endl;
  cout << "Based on iSpectra v1.2 : Chun Shen and Zhi Qiu" << endl;

  // Read-in parameters
  cout << "Reading in Parameters from parameters.dat" << endl;
  ParameterReader *paraRdr = new ParameterReader;
  paraRdr->readFromFile("parameters.dat");
  paraRdr->readFromArguments(argc, argv);
  paraRdr->echo();

  string pathToInput = "input";
  string pathToOutput = "results";

  //load freeze out information
  read_FOdata freeze_out_data(paraRdr, pathToInput);

  long FO_length = 0;
  FO_length = freeze_out_data.get_number_cells();
  cout << "Total number of freezeout cells: " <<  FO_length << endl;

  FO_surf* surf_ptr = new FO_surf[FO_length];

  freeze_out_data.read_surf_switch(FO_length, surf_ptr);

  particle_info *particle = new particle_info [Maxparticle];

  cout << "Finished reading freezeout surface!" << endl;

  Table chosen_particles("PDG/chosen_particles.dat"); // skip others except for these particle
  Table pT_tab("tables/pT_gauss_table.dat"); // pt value and weight table
  Table phi_tab("tables/phi_gauss_table.dat"); // phi value and weight table
  Table y_tab("tables/y_riemann_table_11pt.dat"); //y values and weights, here just a riemann sum!

  EmissionFunctionArray efa(paraRdr, &chosen_particles, &pT_tab, &phi_tab, &y_tab, particle, surf_ptr, FO_length);

  efa.calculate_spectra();

  delete [] FOsurf_ptr;
  delete paraRdr;

  cout << "Done. Goodbye!" << endl;

}
