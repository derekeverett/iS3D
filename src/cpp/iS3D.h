#ifndef _IS3D_SRC
#define _IS3D_SRC

#include <string>
#include <vector>
#include "particle.h"
#include <cmath>

const double hbarC = 0.197327053;  //GeV*fm
const double two_pi = 2.0 * M_PI;
const double two_pi2_hbarC3 = 2.0 * pow(M_PI, 2) * pow(hbarC, 3);
const double four_pi2_hbarC3 = 4.0 * pow(M_PI, 2) * pow(hbarC, 3);

const int Maxparticle = 400; //size of array for storage of the particles
const int Maxdecaychannel = 17;
const int Maxdecaypart = 5;

class IS3D {
private:

public:
  IS3D();
  ~IS3D();

  //the freezeout surface info

  //contravariant position
  std::vector<double> tau;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> eta;

  //covariant surface normal vector
  std::vector<double> dsigma_tau;
  std::vector<double> dsigma_x;
  std::vector<double> dsigma_y;
  std::vector<double> dsigma_eta;

  std::vector<double> E; //energy density
  std::vector<double> T; //Temperature
  std::vector<double> P; //Thermal Pressure

  //contravriant flow velocity
  std::vector<double> ux;
  std::vector<double> uy;
  std::vector<double> un;

  //five indep components of contravariant shear stress pi^{\mu\nu}
  std::vector<double> pixx;
  std::vector<double> pixy;
  std::vector<double> pixn;
  std::vector<double> piyy;
  std::vector<double> piyn;
  std::vector<double> pinn;

  std::vector<double> Pi; //bulk pressure

  //vector to store final particle list for all events
  std::vector<Sampled_Particle> final_particles_;

  //this calls the particlization routine
  //depending on parameters, will either do smooth cooper
  //frye integral (w or w/o res decays)
  // or sampler
  void run_particlization(int fo_from_file);

  //read the freezeout surface from disk 'input/surface.dat'
  void read_fo_surf_from_file();

  //read the freezeout surface file from a c++ pointer
  void read_fo_surf_from_memory(
                                std::vector<double> tau_in,
                                std::vector<double> x_in,
                                std::vector<double> y_in,
                                std::vector<double> eta_in,
                                std::vector<double> dsigma_tau_in,
                                std::vector<double> dsigma_x_in,
                                std::vector<double> dsigma_y_in,
                                std::vector<double> dsigma_eta_in,
                                std::vector<double> E_in,
                                std::vector<double> T_in,
                                std::vector<double> P_in,
                                std::vector<double> ux_in,
                                std::vector<double> uy_in,
                                std::vector<double> un_in,
                                std::vector<double> pixx_in,
                                std::vector<double> pixy_in,
                                std::vector<double> pixn_in,
                                std::vector<double> piyy_in,
                                std::vector<double> piyn_in,
                                std::vector<double> pinn_in,
                                std::vector<double> Pi_in
                                );
  //void set_particle_list(std::vector< std::vector<Sampled_Particle> > list_in);
};

#endif
