/* functions.c            functions for reso.c main program calculations
*   Josef Sollfrank                  Nov. .98             */

// Commented and adjusted by Evan Frodermann, Aug 2005

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "Table.h"

#include "functions.h"
#include "tools.h"

// SB code !!!
#define NUMDECAY 1700 /* size of array for storage of the decays */
#define NUMPARTICLE 400 /*  size of array for storage of the particles */
#define MAXINTV 20000000 /* size of arry for Montecarlo numbers */
#define MHALF (MAXINTV/2)
#define NY 200
#define NPT 50 /* size of arry for storage of the pt-spectrum */
#define NPHI 120 /* size of arry for storage of the y-spectrum */
#define NPHI1 NPHI + 1
#define PI 3.14159265358979323 /* any question? */
#define ALPHA 0.00729735308 /* fine structure constant 1/137.035...*/
#define HBARC 0.197327054 /* = plank constant times speed of light */
#define HBARC3 (HBARC*HBARC*HBARC)
#define FILEDIM 140

extern struct par
{
  long int 	mcID;     /* Montecarlo number according PDG */
  char	name[26];
  double	mass;
  double	width;
  int	    gspin;      /* spin degeneracy */
  int	    baryon;
  int	    strange;
  int	    charm;
  int	    bottom;
  int	    gisospin;  /* isospin degeneracy */
  int	    charge;
  int	    decays;    /* amount of decays listed for this resonance */
  int	    stable;     /* defines whether this particle is considered as stable */
  int     npT;
  int     nphi;
  int     ny;
  double  phimin;
  double  phimax;
  double  ymax;
  double  deltaY;
  double  dNdypTdpTdphi[NY][NPT][NPHI];
  double	mt[NPT];	/* mt values for spectrum */
  double	pt[NPT];	/* pt values for spectrum */
  double  y[NY];    // y values for spectrum
  double  phi[NPHI];
  double  slope[NPHI1];		/* assymtotic slope of mt-spectrum */
}particle[NUMPARTICLE];

extern double PHI[NPHI];    /* Array for phi-storage */

/* decay array for each decay listed */
extern struct de
{
    int reso;       /* Montecarlo number of decaying resonance */
    int numpart;    /* number of daughter particles after decay */
    double  branch;     /* branching ratio */
    int part[5];    /* array of daughter particels Montecarlo number */
}particleDecay[NUMDECAY];

/* array for converting Montecarlo numbers in internal numbering of the resonances */
extern int partid[MAXINTV];


//*******************************************************************************
// The following arrays are set for specific integration routines and point with
// weights for those routines

static  double  gaus2x[] = { 0.577350269189626 };
static  double  gaus4x[] = {    0.8611363115,   0.3399810435    };
static  double  gaus8x[] = {    0.9602898564,   0.7966664774,
                0.3137066458,   0.3626837833    };
static  double  gaus10x[] = {   0.1488743389,   0.4333953941,
                0.6794095682,   0.8650633666,
                        0.97390652  };
static  double  gaus12x[] = {   0.9815606342,   0.9041172563,
                0.7699026741,   0.5873179542,
                0.3678314989,   0.1252334085    };
static  double  gaus16x[] = {
        0.989400934991650,  0.944575023073233,
        0.865631202387832,  0.755404408355003,
        0.617876244402644,  0.458016777657227,
        0.281603550779259,  0.095012509837637   };
static  double  gaus20x[] = {
        0.993128599185094,  0.963971927277913,
        0.912234428251325,  0.839116971822218,
        0.746331906460150,  0.636053680726515,
        0.510867001950827,  0.373706088715419,
        0.227785851141645,  0.076526521133497   };
static  double  gaus48x[] = {
        0.998771007252426118601,    0.993530172266350757548,
        0.984124583722826857745,    0.970591592546247250461,
        0.952987703160430860723,    0.931386690706554333114,
        0.905879136715569672822,    0.876572020274247885906,
        0.843588261624393530711,    0.807066204029442627083,
        0.767159032515740339254,    0.724034130923814654674,
        0.677872379632663905212,    0.628867396776513623995,
        0.577224726083972703818,    0.523160974722233033678,
        0.466902904750958404545,    0.408686481990716729916,
        0.348755886292160738160,    0.287362487355455576736,
        0.224763790394689061225,    0.161222356068891718056,
        0.097004699209462698930,    0.032380170962869362033 };

static  double
gala4x[] = {    0.322547689619,     1.745761101158,
        4.536620296921,     9.395070912301  };

static  double
gala8x[] = {    0.170279632305,     0.903701776799,
        2.251086629866,     4.266700170288,
        7.045905402393,     10.758516010181,
        15.740678641278,    22.863131736889 };
static  double
gala12x[] = {   0.115722117358,     0.611757484515,
        1.512610269776,     2.833751337744,
        4.599227639418,     6.844525453115,
        9.621316842457,     13.006054993306,
        17.116855187462,    22.151090379397,
        28.487967250984,    37.099121044467 };

static  double
gala15x[] = {   0.093307812017,         0.492691740302,
        1.215595412071,         2.269949526204,
        3.667622721751,         5.425336627414,
        7.565916226613,        10.120228568019,
           13.130282482176,        16.654407708330,
               20.776478899449,        25.623894226729,
               31.407519169754,        38.530683306486,
           48.026085572686  };

static  double
gala48x[] = { 2.9811235829960e-02,   0.15710799061788,
    0.38626503757646,   0.71757469411697,
     1.1513938340264,    1.6881858234190,
     2.3285270066532,    3.0731108616526,
     3.9227524130465,    4.8783933559213,
     5.9411080546246,    7.1121105358907,
     8.3927625990912,    9.7845831846873,
     11.289259168010,    12.908657778286,
     14.644840883210,    16.500081428965,
     18.476882386874,    20.577998634022,
     22.806462290521,    25.165612156439,
     27.659128044481,    30.291071001009,
     33.065930662499,    35.988681327479,
     39.064848764198,    42.300590362903,
     45.702792038511,    49.279186382837,
     53.038498087817,    56.990624814804,
     61.146864786140,    65.520206929019,
     70.125706236113,    74.980977518911,
     80.106857350324,    85.528311116034,
     91.275707993668,    97.386667713582,
    103.908833357176,    110.90422088498,
     118.45642504628,    126.68342576889,
     135.76258957786,    145.98643270946,
     157.91561202298,    172.99632814856 };
//*********************************************************************************


/**************************************************************************
*                                     *
*   readParticleData()                          *
*                                     *
*   reads in the particle data file and stores the datas in the arrays    *
*   particle.* and decay.* and fills up the rest of data (antibaryons)    *
***********************************************************************   *
**************************************************************************/

void readParticleData(char filename[FILEDIM], int* particlemax, int* decaymax)
{
    int i = 0, j = 0, k, h;
    FILE *dat;
    double dummy1;

    for (k = 0; k < MAXINTV; k++) partid[k] = -1;
    dat = fopen(filename,"r");
    if (dat == NULL)
    {
        printf(" NO file: %s  available ! \n", filename);
        printf(" GOOD BYE AND HAVE A NICE DAY! \n");
        exit(0);
    }
    // Read in the particle data from the specified resonance table
    // Save the data is the structure particle[pn]

    while(fscanf(dat,"%li%s%lf%lf%i%i%i%i%i%i%i%i", &particle[i].mcID,
          particle[i].name, &particle[i].mass, &particle[i].width,
          &particle[i].gspin, &particle[i].baryon,
          &particle[i].strange, &particle[i].charm,
          &particle[i].bottom, &particle[i].gisospin,
          &particle[i].charge, &particle[i].decays)==12)
    {
        partid[MHALF + particle[i].mcID] = i;
        particle[i].stable = 0;

        /* read in the decays */
        // These decays are saved in a seperate data set, decay[i].
        for (k = 0; k < particle[i].decays; k++)
        {
            h = fscanf(dat,"%i%i%lf%i%i%i%i%i",
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

//*********************************************************************************
// This function reads in the EdN/d3p spectra calculated by spectra via the HYDRO output.
// Set the pt points with the gaussian values given above.  These are the same values
// given to the spectra in the azspectra0p0 program. The default filename is phipspectra.dat
void readSpectra(char specfile[FILEDIM], int *particlemax, int *decaymax)
{
    char resofile[FILEDIM] = "EOS/pdg.dat";
    int npa = 0;
    double dum, dum1, dum2;
    FILE *spec = fopen(specfile,"r");
    if(spec == NULL)
    {
        printf(" NO file: %s  available ! \n", specfile);
        printf(" GOOD BYE AND HAVE A NICE DAY! \n");
        exit(0);
    }
    readParticleData(resofile, particlemax, decaymax);

    Table pT_tab("tables/pT_gauss_table.dat");
    Table phi_tab("tables/phi_gauss_table.dat");
    Table y_tab("tables/y_riemann_table_11pt.dat");
    int npT = pT_tab.getNumberOfRows();
    int nphi = phi_tab.getNumberOfRows();
    int ny = y_tab.getNumberOfRows();

    int WarningCounter = 0;
    for (int n = 0; n < *particlemax; n++)
    {
        npa++;
        int pn = n;
        if (pn == -1)
        {
            printf(" particle %i not in reso table ! \n");
            printf(" GOOD BYE AND HAVE A NICE DAY! \n");
            exit(0);
        }
        if (npT > NPT)
        {
            printf(" NPT = %i array to small !\n", npT);
            printf(" GOOD BYE AND HAVE A NICE DAY! \n");
            exit(0);
        }
        if (nphi > NPHI)
        {
            printf(" NPHI = %i array to small !\n", nphi);
            printf(" GOOD BYE AND HAVE A NICE DAY! \n");
            exit(0);
        }

        for (int ipT = 0; ipT < npT; ipT++) particle[pn].pt[ipT] = pT_tab.get(1,ipT + 1);
        for (int iy = 0; iy < ny; iy++) particle[pn].y[iy]       =  y_tab.get(1,iy + 1);
        for (int iphi = 0; iphi < nphi; iphi++) particle[pn].phi[iphi]        = phi_tab.get(1,iphi + 1);

        particle[pn].slope[npT] = 1;
        particle[pn].nphi       = nphi;
        particle[pn].npT        = npT;

        for (int iy = 0; iy < ny; iy++)
        {
            for (int ipT = 0; ipT < npT; ipT++)
            {
              for (int iphi = 0; iphi < nphi; iphi++)
              {
                fscanf(spec,"%lf",&dum);
                particle[pn].dNdypTdpTdphi[iy][ipT][iphi] = dum;
              }
            }
        }
    }
    fclose(spec);
    if (npa > 0) printf(" Successful read in of %5i spectra !\n",npa);
}

//*********************************************************************************************
// After calculating the spectra in the decay routines, this routine writes that data to file.
// The spectra is written to spec_###.dat in block format for the pt/phi dependence.
// The pt values for each point are saved to specPT_###.dat. The ### corresponds to the
// monte carlo value number assigned the resonance table.  The phi data was already stored in
// angle.dat. (by default)
void writeSpectra(int particlemax, char outdir[FILEDIM])
{
    FILE *out, *out2;
    char filename[FILEDIM];
    char filename2[FILEDIM];

    char p[SSL];
    int i, j, k;

    for(i=1;i<particlemax;i++)// Cycle through the particles
    {
        if(particle[i].stable == 1) //Only print out the stable particles
        {
            strcpy(filename,outdir);
            strcpy(filename2,outdir);
            strcat(filename, "/spec_");
            strcat(filename2, "/PT_");
            if(particle[i].mcID < 0)
            {
                strcat(filename,"A");
                strcat(filename2, "A");
            }
            convei(abs(particle[i].mcID),p);

            strcat(filename,p);
            strcat(filename2,p);
            strcat(filename,".dat");
            strcat(filename2, ".dat");

            printf(" Produce %s \n", filename);
            printf(" Produce %s \n", filename2);
            out = fopen(filename,"w");
            out2 = fopen(filename2, "w");
            for(k=0;k<particle[i].nphi;k++) //Print out the desired data.
            {
                for(j=0;j<particle[i].npT;j++)
                {
                    fprintf(out," %11.4lE", particle[i].dNdypTdpTdphi[j][k]);
                }
                fprintf(out,"\n");
            }
            for(j=0;j<particle[i].npT;j++)
            {
                fprintf(out2," %11.4lE", particle[i].pt[j]);
            }
            fprintf(out2,"\n");

            fclose(out);
            fclose(out2);
        }
    }
}


/*************************************************
*
*   Edndp3
*
**************************************************/
// This function interpolates the needed spectra for a given y, pt and phi.

double Edndp3(double yr, double ptr, double phirin, int res_num)
// double   yr;     /* y  of resonance */
// double   ptr;        /* pt of resonance */
// double   phirin;     /* phi angle  of resonance */
// int  res_num;    /* Montecarlo number of resonance   */
{
  // if pseudofreeze flag is set, yr is the *pseudorapidity* of the resonance
  if (phirin < 0.0) { printf("ERROR: phir %15.8le < 0 !!! \n", phirin); exit(0); }
  if (phirin > 2.0*PI) { printf("ERROR: phir %15.8le > 2PI !!! \n", phirin); exit(0); }
  double phir = phirin;
  int pn = partid[MHALF + res_num];

  if (ptr > particle[pn].pt[particle[pn].npT - 1]) return 0.;
  if (fabs(yr) > particle[pn].ymax && !BOOST_INV) return 0.;

  int nphi = 1;
  while ( (phir > PHI[nphi]) && ( nphi < (particle[pn].nphi-1) ) ) nphi++;
  int npT = 1;
  while (ptr > particle[pn].pt[npT] && npT<(particle[pn].npT - 1)) npT++;
  int ny = 1;
  while ( yr > particle[pn].y[ny] && ny < (particle[pn].ny - 1) ) ny++;

  /* phi interpolation */
  double f1 = util->lin_int(PHI[nphi-1], PHI[nphi], particle[pn].dNdypTdpTdphi[ny-1][npT-1][nphi-1], particle[pn].dNdypTdpTdphi[ny-1][npT-1][nphi], phir);
  double f2 = util->lin_int(PHI[nphi-1], PHI[nphi], particle[pn].dNdypTdpTdphi[ny-1][npT][nphi-1], particle[pn].dNdypTdpTdphi[ny-1][npT][nphi], phir);

  // security: if for some reason we got a negative number of particles
  // (happened in the viscous code at large eta sometimes)
  if (f1 < 0.) f1 = 1e-30;
  if (f2 < 0.) f2 = 1e-30;
  double f1s = f1;
  double f2s = f2;
  /*
  if (ptr > PTCHANGE && f1s > 0 && f2s > 0) {
    f1 = log(f1);
    f2 = log(f2);
  }
  */
  double val1 = util->lin_int(particle[pn].pt[npT-1], particle[pn].pt[npT], f1, f2, ptr);
  //if (ptr > PTCHANGE && f1s > 0 && f2s > 0) val1 = exp(val1);
  f1 = util->lin_int(PHI[nphi-1], PHI[nphi], particle[pn].dNdypTdpTdphi[ny][npT-1][nphi-1], particle[pn].dNdypTdpTdphi[ny][npT-1][nphi], phir);
  f2 = util->lin_int(PHI[nphi-1], PHI[nphi], particle[pn].dNdypTdpTdphi[ny][npT][nphi-1], particle[pn].dNdypTdpTdphi[ny][npT][nphi], phir);

  // security: if for some reason we got a negative number of particles
  // (happened in the viscous code at large eta sometimes)
  if (f1 < 0.) f1 = 1e-30;
  if (f2 < 0.) f2 = 1e-30;
  f1s = f1;
  f2s = f2;
  /*
  if (ptr > PTCHANGE && f1s > 0 && f2s > 0)
  {
    f1 = log(f1);
    f2 = log(f2);
  }
  */
  double val2 = util->lin_int(particle[pn].pt[npT-1], particle[pn].pt[npT], f1, f2, ptr);
  //if (ptr > PTCHANGE && f1s > 0 && f2s > 0) val2 = exp(val2);
  double val = util->lin_int(particle[pn].y[ny-1], particle[pn].y[ny], val1, val2, yr);
  return val;
}
