#include "iS3D.h"

//using namespace std;

int main(int argc, char *argv[])
{
  //create an instance of IS3D class
  IS3D particlization;

  //run iS3D
  //if argument == 1, freeeout surface is read from file
  //otherwise freezeout surface is read from memory 
  particlization.run_particlization(1);

}
