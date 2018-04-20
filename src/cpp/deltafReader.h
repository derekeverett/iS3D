
#ifndef DELTAFREADER_H
#define DELTAFREADER_H

#include "ParameterReader.h"
#include "readindata.h"
#include<fstream>

using namespace std;

class DeltafReader
{
    private:
        ParameterReader * paraRdr;
        string pathTodeltaf;
        
        int df_mode; // type of delta-f correction (e.g. 14-moment, CE, or modified distribution)
        int include_baryon;

    public:
        DeltafReader(ParameterReader * paraRdr_in, string path_in);
        ~DeltafReader();

        deltaf_coefficients load_coefficients(FO_surf * surface, long FO_length_in);
};

#endif
