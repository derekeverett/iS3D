
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<math.h>
#include<vector>
#include<sys/time.h>

#include "main.cuh"
#include "Table.cuh"
#include "readindata.cuh"
#include "emissionfunction.cuh"
#include "arsenal.cuh"
#include "ParameterReader.cuh"
#include "deltafReader.cuh"

using namespace std;

int main()
{
  cout << "Welcome to iS3D, a program to accelerate particle spectra computation from 3+1D Hydro Freezeout Surfaces!" << endl;
  cout << "Derek Everett, Sameed Pervaiz, Mike McNelis and Lipei Du (2018)" << endl;
  cout << "Based on iSpectra v1.2 : Chun Shen and Zhi Qiu" << endl;

  // Read-in parameters:
  cout << "Reading in parameters:\n" << endl;
  ParameterReader * paraRdr = new ParameterReader;
  paraRdr->readFromFile("iS3D_parameters.dat");
  paraRdr->echo();


  // Load freeze out information:
  FO_data_reader freeze_out_data(paraRdr, "input");
  long FO_length = freeze_out_data.get_number_cells();
  FO_surf * surf_ptr = new FO_surf[FO_length];
  freeze_out_data.read_surf_switch(FO_length, surf_ptr);


  // Load pdg particle info and chosen particle ids:
  particle_info * particle_data = new particle_info[Maxparticle];
  PDG_Data pdg(paraRdr);
  int Nparticle = pdg.read_resonances(particle_data);
  Table chosen_particles("PDG/chosen_particles.dat");


  // Load df coefficient data:
  Deltaf_Data * df_data = new Deltaf_Data(paraRdr);
  df_data->load_df_coefficient_data();
  df_data->construct_cubic_splines();
  df_data->compute_jonah_coefficients(particle_data, Nparticle);
  df_data->test_df_coefficients(-0.1);


  cout << "Total number of freezeout cells: " <<  FO_length << endl;
  cout << "Number of chosen particles: " << chosen_particles.getNumberOfRows() << endl;


  // Load momentum and eta tables
  Table pT_tab("tables/pT_uniform_table.dat");
  Table phi_tab("tables/phi_gauss_legenddre_table.dat");
  Table y_tab("tables/y_trapezoid_table_21pt.dat");
  Table eta_tab("tables/eta/eta_gauss_table_48pt_range_|6.70095|.dat");


  // Calculate particle spectra:
  EmissionFunctionArray efa(paraRdr, &chosen_particles, &pT_tab, &phi_tab, &y_tab, &eta_tab, particle_data, Nparticle, surf_ptr, FO_length, df_data);

  efa.calculate_spectra();

  delete [] surf_ptr;
  delete [] particle_data;
  delete paraRdr;
  delete df_data;

  cout << "Done calculating particle spectra. Output stored in results folder. Goodbye!" << endl;

}



