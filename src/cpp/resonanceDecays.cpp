#include "particle.h"

void readParticleData(char filename[FILEDIM], int* particlemax, int* decaymax, particle *particles)
{
    FILE *dat;
    double dummy1;

    for (int k = 0; k < MAXINTV; k++) partid[k] = -1;
    dat = fopen(filename, "r");
    if (dat == NULL)
    {
        printf(" NO file: %s  available ! \n", filename);
        printf(" GOOD BYE AND HAVE A NICE DAY! \n");
        exit(0);
    }
    // Read in the particle data from the specified resonance table
    // Save the data is the particle array

    int i = 0;
    int j = 0;
    while(fscanf(dat,"%li%s%lf%lf%i%i%i%i%i%i%i%i", &particles[i].mcID,
          &particles[i].name, &particles[i].mass, &particles[i].width,
          &particles[i].gspin, &particles[i].baryon,
          &particles[i].strange, &particles[i].charm,
          &particles[i].bottom, &particles[i].gisospin,
          &particles[i].charge, &particles[i].decays) == 12)
    {
        partid[MHALF + particle[i].mcID] = i;
        particles[i].stable = 0;

        /* read in the decays */
        // These decays are saved in a seperate data set, decay[i].
        for (int k = 0; k < particles[i].decays; k++)
        {
            int h = fscanf(dat,"%i%i%lf%i%i%i%i%i",
             &particleDecay[j].reso, &particleDecay[j].numpart, &particleDecay[j].branch,
             &particleDecay[j].part[0], &particleDecay[j].part[1], &particleDecay[j].part[2],
             &particleDecay[j].part[3], &particleDecay[j].part[4]);
            if (h != 8)
            {
                printf("Error in scanf decay \n");
                printf(" GOOD BYE AND HAVE A NICE DAY! \n");
                exit(0);
            }
            if (particleDecay[j].numpart == 1) particle[i].stable = 1;
            j++; // Add one to the decay counting variable "j"
        }

        /* setting of additional parameters */
        if (particle[i].baryon == 1)
        {
            i++;// If the particle is a baryon, add a particle for the anti-baryon
            // Add one to the counting variable "i" for the number of particles for the anti-baryon
            particle[i].mcID = -particle[i-1].mcID;
            strcpy(particle[i].name, "  Anti-");
            strncat(particle[i].name, particle[i-1].name, 18);
            particle[i].mass     =  particle[i-1].mass;
            particle[i].width    =  particle[i-1].width;
            particle[i].gspin    =  particle[i-1].gspin;
            particle[i].baryon   = -particle[i-1].baryon;
            particle[i].strange  = -particle[i-1].strange;
            particle[i].charm    = -particle[i-1].charm;
            particle[i].bottom   = -particle[i-1].bottom;
            particle[i].gisospin =  particle[i-1].gisospin;
            particle[i].charge   = -particle[i-1].charge;
            particle[i].decays   = particle[i-1].decays;
            partid[MHALF + particle[i].mcID] = i;
            particle[i].stable =  particle[i-1].stable;
        }
        i++; // Add one to the counting variable "i" for the meson/baryon
    }
    fclose(dat);

    *particlemax = i;   // Set the maxparticle variable to be the value of "i"
    *decaymax  = j;     // Set the maxdecays variable to be the value of "j"

    if( (*particlemax) > NUMPARTICLE )
    {
        printf("Array for particles too small! \n");
        printf("GOOD BYE AND HAVE A NICE DAY! \n");
        exit(0);
    }
}
