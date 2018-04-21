
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
#include "deltafReader.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "Welcome to iS3D, a program to accelerate particle spectra computation from 3+1D Hydro Freezeout Surfaces!" << endl;
  cout << "Derek Everett, Sameed Pervaiz, Mike McNelis and Lipei Du (2018)" << endl;
  cout << "Based on iSpectra v1.2 : Chun Shen and Zhi Qiu" << endl;

  // read in parameters
  cout << "Reading in Parameters from parameters.dat" << endl;
  ParameterReader * paraRdr = new ParameterReader;
  paraRdr->readFromFile("parameters.dat");
  paraRdr->readFromArguments(argc, argv);
  paraRdr->echo();

  // load freeze out information:
  string pathToInput = "input";
  //string pathToOutput = "results";
  FO_data_reader freeze_out_data(paraRdr, pathToInput);

  long FO_length = 0;
  FO_length = freeze_out_data.get_number_cells();
  cout << "Total number of freezeout cells: " <<  FO_length << endl;

  FO_surf * surf_ptr = new FO_surf[FO_length];
  freeze_out_data.read_surf_switch(FO_length, surf_ptr);

  // load delta-f coefficients:
  deltaf_coefficients df;
  string pathTodeltaf = "deltaf_coefficients"; // not functional
  DeltafReader deltaf(paraRdr, pathTodeltaf);
  df = deltaf.load_coefficients(surf_ptr, FO_length);


  particle_info *particle = new particle_info [Maxparticle];
  int Nparticle = freeze_out_data.read_resonances_list(particle, surf_ptr, df); // number of resonances in pdf file (this changes...)

  cout << "Finished reading freezeout surface!" << endl;



  // FOR THIS READ IN TO WORK PROPERLY, chosen_particles.dat MUST HAVE AN EMPTY ROW AT THE END!
  // perhaps switch to a different method of reading in the chosen_particles.dat file that doesn't
  // have this undesirable feature
  Table chosen_particles("PDG/chosen_particles.dat"); // skip others except for these particles

  cout << "Number of chosen particles : " << chosen_particles.getNumberOfRows() << endl;
  Table pT_tab("tables/pT_gauss_table.dat"); // pT value and weight table
  Table phi_tab("tables/phi_gauss_table.dat"); // phi value and weight table
  Table y_tab("tables/y_riemann_table_11pt.dat"); //y values and weights, here just a riemann sum!
  Table eta_tab("tables/eta_trapezoid_table_41pt.dat"); //y values and weights, here just a riemann sum!

  EmissionFunctionArray efa(paraRdr, &chosen_particles, &pT_tab, &phi_tab, &y_tab, &eta_tab, particle, Nparticle, surf_ptr, FO_length, df);

  efa.calculate_spectra();

  delete [] surf_ptr;
  delete [] particle;
  delete paraRdr;

  cout << "Done Calculating particle spectra. Output stored in results folder. Goodbye!" << endl;

}
