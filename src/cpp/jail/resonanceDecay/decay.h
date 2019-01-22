/* decay.h        header file for decay.c
*  Sollfrank Josef           Nov .98                      */

#ifndef decay_h
#define decay_h

#define BOOST_INV 0 //this needs to be changed from a preprocessor variable to a parameter that user can set at runtime

void calc_reso_decays(int maxpart, int maxdecay, int bound);

#endif
