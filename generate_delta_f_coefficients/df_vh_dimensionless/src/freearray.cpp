
#include <stdlib.h>
#include "../include/freearray.hpp"


void free_2D(double ** M, int n)
{
	for (int i = 0; i < n; i++) free(M[i]);
    free(M);
}


void free_3D(double *** M, int n, int m)
{
	for (int i = 0; i < n; i++)
    {
    	for(int j = 0; j < m; j++)
    	{
    		free(M[i][j]);
    	}
        free(M[i]);
    }
    free(M);
}

