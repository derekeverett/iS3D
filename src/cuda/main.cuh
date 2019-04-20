#ifndef MAIN_H
#define MAIN_H

#include <string>
#include <cmath>
using namespace std;

const double hbarC = 0.197327053;  // GeV.fm
const double two_pi = 2.0 * M_PI;
const double two_pi2_hbarC3 = 2.0 * pow(M_PI, 2) * pow(hbarC, 3);
const double four_pi2_hbarC3 = 4.0 * pow(M_PI, 2) * pow(hbarC, 3);

//TO DO remove these hardcoded entries and declare a std::vector<particle_info> which can be appended
const int Maxparticle = 600; 	// size of array for storage of the particles
const int Maxdecaychannel = 50;
const int Maxdecaypart = 5;

#endif
