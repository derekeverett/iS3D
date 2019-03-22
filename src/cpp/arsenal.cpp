
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cmath>
#include <iomanip>
#include <cstdarg>
#include <stdio.h> //for printf
#include "arsenal.h"

using namespace std;

void printline()
{
  cout << "______________________________\n" << endl;
}

//**********************************************************************
double sixPoint2dInterp(double x, double y,
    double v00, double v01, double v02, double v10, double v11, double v20)
{
  /* Assume a 2d function f(x,y) has a quadratic form:
    f(x,y) = axx*x^2 + axy*x*y + ayy*y^2 + bx*x + by*y + c
    Also assume that:
    f(0,0)=v00, f(0,1)=v01, f(0,2)=v02, f(1,0)=v10, f(1,1)=v11, f(2,0)=v20
    Then its value at (x,y) can be solved.
    The best result is produced if x and y are between 0 and 1.
  */

  // Calculate coefficients:
  double axx = 1.0/2.0*(v00 - 2*v10 + v20);
  double axy = v00 - v01 - v10 + v11;
  double ayy = 1.0/2.0*(v00 - 2*v01 + v02);
  double bx = 1.0/2.0*(-3.0*v00 + 4*v10 - v20);
  double by = 1.0/2.0*(-3.0*v00 + 4*v01 - v02);
  double c = v00;

  // Calcualte f(x,y):
  return axx*x*x + axy*x*y + ayy*y*y + bx*x + by*y + c;
}


//**********************************************************************
double interpCubicDirect(vector<double>* x, vector<double>* y, double x0)
// Returns the interpreted value of y=y(x) at x=x0 using cubic polynomial interpolation method.
// -- x,y: the independent and dependent double x0ables; x is assumed to be equal spaced and increasing
// -- x0: where the interpolation should be performed
{
  long size = x->size();
  if (size==1) {cout<<"interpCubicDirect warning: table size = 1"; return (*y)[0];}
  double dx = (*x)[1]-(*x)[0]; // increment in x

  // if close to left end:
  if (abs(x0-(*x)[0])<dx*1e-30) return (*y)[0];

  // find x's integer index
  long idx = floor((x0-(*x)[0])/dx);

  if (idx<0 || idx>=size-1)
  {
    cout    << "interpCubicDirect: x0 out of bounds." << endl
            << "x ranges from " << (*x)[0] << " to " << (*x)[size-1] << ", "
            << "x0=" << x0 << ", " << "dx=" << dx << ", " << "idx=" << idx << endl;
    exit(1);
  }

  if (idx==0)
  {
    // use quadratic interpolation at left end
    double A0 = (*y)[0], A1 = (*y)[1], A2 = (*y)[2], deltaX = x0 - (*x)[0]; // deltaX is the increment of x0 compared to the closest lattice point
    return (A0-2.0*A1+A2)/(2.0*dx*dx)*deltaX*deltaX - (3.0*A0-4.0*A1+A2)/(2.0*dx)*deltaX + A0;
  }
  else if (idx==size-2)
  {
    // use quadratic interpolation at right end
    double A0 = (*y)[size-3], A1 = (*y)[size-2], A2 = (*y)[size-1], deltaX = x0 - ((*x)[0] + (idx-1)*dx);
    return (A0-2.0*A1+A2)/(2.0*dx*dx)*deltaX*deltaX - (3.0*A0-4.0*A1+A2)/(2.0*dx)*deltaX + A0;
  }
  else
  {
    // use cubic interpolation
    double A0 = (*y)[idx-1], A1 = (*y)[idx], A2 = (*y)[idx+1], A3 = (*y)[idx+2], deltaX = x0 - ((*x)[0] + idx*dx);
    //cout << A0 << "  " << A1 << "  " << A2 << "  " << A3 << endl;
    return (-A0+3.0*A1-3.0*A2+A3)/(6.0*dx*dx*dx)*deltaX*deltaX*deltaX
            + (A0-2.0*A1+A2)/(2.0*dx*dx)*deltaX*deltaX
            - (2.0*A0+3.0*A1-6.0*A2+A3)/(6.0*dx)*deltaX
            + A1;
  }

}




//**********************************************************************
double interpLinearDirect(vector<double>* x, vector<double>* y, double x0)
// Returns the interpreted value of y=y(x) at x=x0 using linear interpolation method.
// -- x,y: the independent and dependent double x0ables; x is assumed to be equal spaced and increasing
// -- x0: where the interpolation should be performed
{
  long size = x->size();
  if (size==1) {cout<<"interpLinearDirect warning: table size = 1"<<endl; return (*y)[0];}
  double dx = (*x)[1]-(*x)[0]; // increment in x

  // if close to left end:
  if (abs(x0-(*x)[0])<dx*1e-30) return (*y)[0];

  // find x's integer index
  long idx = floor((x0-(*x)[0])/dx);

  if (idx<0 || idx>=size-1)
  {
    cout    << "interpLinearDirect: x0 out of bounds." << endl
            << "x ranges from " << (*x)[0] << " to " << (*x)[size-1] << ", "
            << "x0=" << x0 << ", " << "dx=" << dx << ", " << "idx=" << idx << endl;
    exit(1);
  }

  return (*y)[idx] + ((*y)[idx+1]-(*y)[idx])/dx*(x0-(*x)[idx]);

}


//**********************************************************************
double interpNearestDirect(vector<double>* x, vector<double>* y, double x0)
// Returns the interpreted value of y=y(x) at x=x0 using nearest interpolation method.
// -- x,y: the independent and dependent double x0ables; x is assumed to be equal spaced and increasing
// -- x0: where the interpolation should be performed
{
  long size = x->size();
  if (size==1) {cout<<"interpNearestDirect warning: table size = 1"<<endl; return (*y)[0];}
  double dx = (*x)[1]-(*x)[0]; // increment in x

  // if close to left end:
  if (abs(x0-(*x)[0])<dx*1e-30) return (*y)[0];

  // find x's integer index
  long idx = floor((x0-(*x)[0])/dx);

  if (idx<0 || idx>=size-1)
  {
    cout    << "interpNearestDirect: x0 out of bounds." << endl
            << "x ranges from " << (*x)[0] << " to " << (*x)[size-1] << ", "
            << "x0=" << x0 << ", " << "dx=" << dx << ", " << "idx=" << idx << endl;
    exit(1);
  }

  return x0-(*x)[idx]>dx/2 ? (*y)[idx+1] : (*y)[idx];

}


//**********************************************************************
double interpCubicMono(vector<double>* x, vector<double>* y, double xx)
// Returns the interpreted value of y=y(x) at x=x0 using cubic polynomial interpolation method.
// -- x,y: the independent and dependent double x0ables; x is *NOT* assumed to be equal spaced but it has to be increasing
// -- xx: where the interpolation should be performed
{
  long size = x->size();
  if (size==1) {cout<<"interpCubicMono warning: table size = 1"<<endl; return (*y)[0];}

  // if close to left end:
  if (abs(xx-(*x)[0])<((*x)[1]-(*x)[0])*1e-30) return (*y)[0];

  // find x's integer index
  long idx = binarySearch(x, xx);

  if (idx<0 || idx>=size-1)
  {
    cout    << "interpCubicMono: x0 out of bounds." << endl
            << "x ranges from " << (*x)[0] << " to " << (*x)[size-1] << ", "
            << "xx=" << xx << ", " << "idx=" << idx << endl;
    exit(1);
  }

  if (idx==0)
  {
    // use linear interpolation at the left end
    return (*y)[0] + ( (*y)[1]-(*y)[0] )/( (*x)[1]-(*x)[0] )*( xx-(*x)[0] );
  }
  else if (idx==size-2)
  {
    // use linear interpolation at the right end
    return (*y)[size-2] + ( (*y)[size-1]-(*y)[size-2] )/( (*x)[size-1]-(*x)[size-2] )*( xx-(*x)[size-2] );
  }
  else
  {
    // use cubic interpolation
    long double y0 = (*y)[idx-1], y1 = (*y)[idx], y2 = (*y)[idx+1], y3 = (*y)[idx+2];
    long double y01=y0-y1, y02=y0-y2, y03=y0-y3, y12=y1-y2, y13=y1-y3, y23=y2-y3;
    long double x0 = (*x)[idx-1], x1 = (*x)[idx], x2 = (*x)[idx+1], x3 = (*x)[idx+2];
    long double x01=x0-x1, x02=x0-x2, x03=x0-x3, x12=x1-x2, x13=x1-x3, x23=x2-x3;
    long double x0s=x0*x0, x1s=x1*x1, x2s=x2*x2, x3s=x3*x3;
    long double denominator = x01*x02*x12*x03*x13*x23;
    long double C0, C1, C2, C3;
    C0 = (x0*x02*x2*x03*x23*x3*y1
          + x1*x1s*(x0*x03*x3*y2+x2s*(-x3*y0+x0*y3)+x2*(x3s*y0-x0s*y3))
          + x1*(x0s*x03*x3s*y2+x2*x2s*(-x3s*y0+x0s*y3)+x2s*(x3*x3s*y0-x0*x0s*y3))
          + x1s*(x0*x3*(-x0s+x3s)*y2+x2*x2s*(x3*y0-x0*y3)+x2*(-x3*x3s*y0+x0*x0s*y3))
          )/denominator;
    C1 = (x0s*x03*x3s*y12
          + x2*x2s*(x3s*y01+x0s*y13)
          + x1s*(x3*x3s*y02+x0*x0s*y23-x2*x2s*y03)
          + x2s*(-x3*x3s*y01-x0*x0s*y13)
          + x1*x1s*(-x3s*y02+x2s*y03-x0s*y23)
          )/denominator;
    C2 = (-x0*x3*(x0s-x3s)*y12
          + x2*(x3*x3s*y01+x0*x0s*y13)
          + x1*x1s*(x3*y02+x0*y23-x2*y03)
          + x2*x2s*(-x3*y01-x0*y13)
          + x1*(-x3*x3s*y02+x2*x2s*y03-x0*x0s*y23)
          )/denominator;
    C3 = (x0*x03*x3*y12
          + x2s*(x3*y01+x0*y13)
          + x1*(x3s*y02+x0s*y23-x2s*y03)
          + x2*(-x3s*y01-x0s*y13)
          + x1s*(-x3*y02+x2*y03-x0*y23)
          )/denominator;
/*    cout  << x0s*x03*x3s*y12 << "  "
          <<  x2*x2s*(x3s*y01+x0s*y13) << "   "
          <<  x1s*(x3*x3s*y02+x0*x0s*y23-x2*x2s*y03) << "  "
          <<  x2s*(-x3*x3s*y01-x0*x0s*y13) << "  "
          <<  x1*x1s*(-x3s*y02+x2s*y03-x0s*y23) << endl;
    cout << denominator << endl;

    cout << x0 << " " << x1 << "  " << x2 << "  " << x3 << endl;
    cout << y0 << " " << y1 << "  " << y2 << "  " << y3 << endl;
    cout << C0 << "  " << C1 << "  " << C2 << "  " << C3 << endl;*/
    return C0 + C1*xx + C2*xx*xx + C3*xx*xx*xx;
  }

}

//**********************************************************************
double interpLinearMono(vector<double>* x, vector<double>* y, double xx)
// Returns the interpreted value of y=y(x) at x=x0 using linear interpolation method.
// -- x,y: the independent and dependent double x0ables; x is *NOT* assumed to be equal spaced but it has to be increasing
// -- xx: where the interpolation should be performed
{
  long size = x->size();
  if (size==1) {cout<<"interpLinearMono warning: table size = 1"<<endl; return (*y)[0];}

  // if close to left end:
  if (abs(xx-(*x)[0])<((*x)[1]-(*x)[0])*1e-30) return (*y)[0];

  // find x's integer index
  long idx = binarySearch(x, xx);

  if (idx<0 || idx>=size-1)
  {
    cout    << "interpLinearMono: x0 out of bounds." << endl
            << "x ranges from " << (*x)[0] << " to " << (*x)[size-1] << ", "
            << "xx=" << xx << ", " << "idx=" << idx << endl;
    exit(1);
  }

  return (*y)[idx] + ( (*y)[idx+1]-(*y)[idx] )/( (*x)[idx+1]-(*x)[idx] )*( xx-(*x)[idx] );

}

//**********************************************************************
double interpNearestMono(vector<double>* x, vector<double>* y, double xx)
// Returns the interpreted value of y=y(x) at x=x0 using nearest interpolation method.
// -- x,y: the independent and dependent double x0ables; x is *NOT* assumed to be equal spaced but it has to be increasing
// -- xx: where the interpolation should be performed
{
  long size = x->size();
  if (size==1) {cout<<"interpNearestMono warning: table size = 1"<<endl; return (*y)[0];}

  // if close to left end:
  if (abs(xx-(*x)[0])<((*x)[1]-(*x)[0])*1e-30) return (*y)[0];

  // find x's integer index
  long idx = binarySearch(x, xx);

  if (idx<0 || idx>=size-1)
  {
    cout    << "interpNearestMono: x0 out of bounds." << endl
            << "x ranges from " << (*x)[0] << " to " << (*x)[size-1] << ", "
            << "xx=" << xx << ", " << "idx=" << idx << endl;
    exit(1);
  }

  return xx-(*x)[idx] > (*x)[idx+1]-xx ? (*y)[idx+1] : (*y)[idx];

}

//**********************************************************************
double invertFunc(double (*func)(double), double y, double xL, double xR, double dx, double x0, double relative_accuracy)
//Purpose:
//  Return x=func^(-1)(y) using Newton method.
//  -- func: double 1-argument function to be inverted
//  -- xL: left boundary (for numeric derivative)
//  -- xR: right boundary (for numeric derivative)
//  -- dx: step (for numeric derivative)
//  -- x0: initial value
//  -- y: the value to be inverted
//  -- Returns inverted value
//Solve: f(x)=0 with f(x)=table(x)-y => f'(x)=table'(x)
{
  double accuracy;
  int tolerance;

  double XX1, XX2; // used in iterations
  double F0, F1, F2, F3, X1, X2; // intermedia variables
  int impatience; // number of iterations


  // initialize parameters
  accuracy = dx*relative_accuracy;

  tolerance = 60;
  impatience = 0;

  // initial value, left point and midxle point
  XX2 = x0;
  XX1 = XX2-10*accuracy; // this value 10*accuracy is meanless, just to make sure the check in the while statement goes through

  while (abs(XX2-XX1)>accuracy)
  {
    XX1 = XX2; // copy values

    // value of function at XX
    F0 = (*func)(XX1) - y; // the value of the function at this point

    // decide X1 and X2 for differentiation
    if (XX1>xL+dx)
      X1 = XX1 - dx;
    else
      X1 = xL;

    if (XX1<xR-dx)
      X2 = XX1 + dx;
    else
      X2 = xR;

    // get values at X1 and X2
    F1 = (*func)(X1);
    F2 = (*func)(X2);
    F3 = (F1-F2)/(X1-X2); // derivative at XX1

    XX2 = XX1 - F0/F3; // Newton's mysterious method

    impatience = impatience + 1;
    //cout << "impatience=" << impatience << endl;
    if (impatience>tolerance)
    {
      cout << "invertFunc: " << "max number of iterations reached." << endl;
      exit(-1);
    }

  } // <=> abs(XX2-XX1)>accuracy

  return XX2;
}



//**********************************************************************
vector<double> *zq_x_global, *zq_y_global;
double invertTableDirect_hook(double xx) {return interpCubicDirect(zq_x_global,zq_y_global,xx);}
double invertTableDirect(vector<double>* x, vector<double>* y, double y0, double x0, double relative_accuracy)
// Return x0=y^(-1)(y0) for y=y(x); use interpCubic and invertFunc.
//  -- x,y: the independent and dependent variables. x is assumed to be equal-spaced.
//  -- y0: where the inversion should be performed.
//  -- x0: initial guess
{
  long size = x->size();
  if (size==1) return (*y)[0];
  zq_x_global = x; zq_y_global = y;
  return invertFunc(&invertTableDirect_hook, y0, (*x)[0], (*x)[size-1], (*x)[1]-(*x)[0], x0, relative_accuracy);
}


//**********************************************************************
vector<double> stringToDoubles(string str)
// Return a vector of doubles from the string "str". "str" should
// be a string containing a line of data.
{
  stringstream sst(str+" "); // add a blank at the end so the last data will be read
  vector<double> valueList;
  double val;
  sst >> val;
  while (sst.eof()==false)
  {
    valueList.push_back(val);
    sst >> val;
  }
  return valueList;
}

//**********************************************************************
double stringToDouble(string str)
// Return the 1st doubles number read from the string "str". "str" should be a string containing a line of data.
{
  stringstream sst(str+" "); // add a blank at the end so the last data will be read
  double val;
  sst >> val;
  return val;
}


//**********************************************************************
vector< vector<double>* >* readBlockData(istream &stream_in)
// Return a nested vector of vector<double>* object. Each column of data
// is stored in a vector<double> array and the collection is the returned
// object. Data are read from the input stream "stream_in". Each line
// of data is processed by the stringToDoubles function. Note that the
// data block is dynamicall allocated and is not release within the
// function.
// Note that all "vectors" are "new" so don't forget to delete them.
// Warning that also check if the last line is read correctly. Some files
// are not endded properly and the last line is not read.
{
  vector< vector<double>* >* data;
  vector<double> valuesInEachLine;
  long lineSize;
  long i; // temp variable
  char buffer[99999]; // each line should be shorter than this

  // first line:
  stream_in.getline(buffer,99999);
  valuesInEachLine = stringToDoubles(buffer);
  // see if it is empty:
  lineSize = valuesInEachLine.size();
  if (lineSize==0)
  {
    // empty:
    cout << "readBlockData warning: input stream has empty first row; no data read" << endl;
    return NULL;
  }
  else
  {
    // not empty; allocate memory:
    data = new vector< vector<double>* >(lineSize);
    for (i=0; i<lineSize; i++) (*data)[i] = new vector<double>;
  }

  // rest of the lines:
  while (stream_in.eof()==false)
  {
    // set values:
    for (i=0; i<lineSize; i++) (*(*data)[i]).push_back(valuesInEachLine[i]);
    // next line:
    stream_in.getline(buffer,99999);
    valuesInEachLine = stringToDoubles(buffer);
  }

  return data;
}


//**********************************************************************
void releaseBlockData(vector< vector<double>* >* data)
// Use to delete the data block allocated by readBlockData function.
{
  if (data)
  {
    for (unsigned long i=0; i<data->size(); i++) delete (*data)[i];
    delete data;
  }
}


//**********************************************************************
// From Wikipedia --- the free encyclopeida
//
// Recursive auxiliary function for adaptiveSimpsons() function below
//
double adaptiveSimpsonsAux(double (*f)(double), double a, double b, double epsilon,
                           double S, double fa, double fb, double fc, int bottom) {
  double c = (a + b)/2, h = b - a;
  double d = (a + c)/2, e = (c + b)/2;
  double fd = f(d), fe = f(e);
  double Sleft = (h/12)*(fa + 4*fd + fc);
  double Sright = (h/12)*(fc + 4*fe + fb);
  double S2 = Sleft + Sright;
  if (bottom <= 0 || fabs(S2 - S) <= 15*epsilon)
    return S2 + (S2 - S)/15;
  return adaptiveSimpsonsAux(f, a, c, epsilon/2, Sleft,  fa, fc, fd, bottom-1) +
         adaptiveSimpsonsAux(f, c, b, epsilon/2, Sright, fc, fb, fe, bottom-1);
}
//
// Adaptive Simpson's Rule
//
double adaptiveSimpsons(double (*f)(double),   // ptr to function
                        double a, double b,  // interval [a,b]
                        double epsilon,  // error tolerance
                        int maxRecursionDepth) {   // recursion cap
  double c = (a + b)/2, h = b - a;
  double fa = f(a), fb = f(b), fc = f(c);
  double S = (h/6)*(fa + 4*fc + fb);
  return adaptiveSimpsonsAux(f, a, b, epsilon, S, fa, fb, fc, maxRecursionDepth);
}


//**********************************************************************
double qiu_simpsons(double (*f)(double), // ptr to function
                    double a, double b, // interval [a,b]
                    double epsilon, int maxRecursionDepth) // recursion maximum
// My version of the adaptive simpsons integration method.
{
  double f_1=f(a)+f(b), f_2=0., f_4=0.; // sum of values of f(x) that will be weighted by 1, 2, 4 respectively, depending on where x is
  double sum_previous=0., sum_current=0.; // previous and current sum (intgrated value)

  long count = 1; // how many new mid-points are there
  double length = (b-a), // helf length of the interval
         step = length/count; // mid-points are located at a+(i+0.5)*step, i=0..count-1

  int currentRecursionDepth = 1;

  f_4 = f(a+0.5*step); // mid point of [a,b]
  sum_current = (length/6)*(f_1 + f_2*2. + f_4*4.); // get the current sum

  do
  {
    sum_previous = sum_current; // record the old sum
    f_2 += f_4; // old mid-points with weight 4 will be new mid-points with weight 2

    count*=2; // increase number of mid-points
    step/=2.0; // decrease jumping step by half
    f_4 = 0.; // prepare to sum up f_4
    for (int i=0; i<count; i++) f_4 += f(a+step*(i+0.5)); // sum up f_4

    sum_current = (length/6/count)*(f_1 + f_2*2. + f_4*4.); // calculate current sum
    //cout << sum_current << endl;

    if (currentRecursionDepth>maxRecursionDepth)
    {
      cout << endl << "Warning qiu_simpsons: maximum recursion depth reached!" << endl << endl;
      break; // safety treatment
    }
    else currentRecursionDepth++;

  } while (abs(sum_current-sum_previous)>epsilon);

  return sum_current;
}

//**********************************************************************
string toLower(string str)
// Convert all character in string to lower case
{
  string tmp = str;
  for (string::iterator it=tmp.begin(); it<=tmp.end(); it++) *it = tolower(*it);
  return tmp;
}

//**********************************************************************
string trim(string str)
// Convert all character in string to lower case
{
  string tmp = str;
  long number_of_char = 0;
  for (size_t ii=0; ii<str.size(); ii++)
    if (str[ii]!=' ' && str[ii]!='\t')
    {
      tmp[number_of_char]=str[ii];
      number_of_char++;
    }
  tmp.resize(number_of_char);
  return tmp;
}


//**********************************************************************
long binarySearch(vector<double>* A, double value, bool skip_out_of_range)
// Return the index of the largest number less than value in the list A
// using binary search. Index starts with 0.
// If skip_out_of_range is set to true, then it will return -1 for those
// samples that are out of the table range.
{
   int length = A->size();
   int idx_i, idx_f, idx;
   idx_i = 0;
   idx_f = length-1;

   if(value > (*A)[idx_f])
   {
      if (skip_out_of_range) return -1;
      cout << "binarySearch: desired value is too large, exceeding the end of the table." << endl;
      exit(1);
   }
   if(value < (*A)[idx_i])
   {
      if (skip_out_of_range) return -1;
      cout << "binarySearch: desired value is too small, exceeding the beginning of table." << endl;
      exit(1);
   }
   idx = (int) floor((idx_f+idx_i)/2.);
   while((idx_f-idx_i) > 1)
   {
     if((*A)[idx] < value)
        idx_i = idx;
     else
        idx_f = idx;
     idx = (int) floor((idx_f+idx_i)/2.);
   }
   return(idx_i);
}


//**********************************************************************
double gamma_function(double x)
// gamma.cpp -- computation of gamma function.
//      Algorithms and coefficient values from "Computation of Special
//      Functions", Zhang and Jin, John Wiley and Sons, 1996.
// Returns gamma function of argument 'x'.
//
// NOTE: Returns 1e308 if argument is a negative integer or 0,
//      or if argument exceeds 171.
{
  int i,k,m;
  double ga,gr,r=0,z;

  static double g[] = {
    1.0,
    0.5772156649015329,
   -0.6558780715202538,
   -0.420026350340952e-1,
    0.1665386113822915,
   -0.421977345555443e-1,
   -0.9621971527877e-2,
    0.7218943246663e-2,
   -0.11651675918591e-2,
   -0.2152416741149e-3,
    0.1280502823882e-3,
   -0.201348547807e-4,
   -0.12504934821e-5,
    0.1133027232e-5,
   -0.2056338417e-6,
    0.6116095e-8,
    0.50020075e-8,
   -0.11812746e-8,
    0.1043427e-9,
    0.77823e-11,
   -0.36968e-11,
    0.51e-12,
   -0.206e-13,
   -0.54e-14,
    0.14e-14};

  if (x > 171.0) return 1e308;    // This value is an overflow flag.
  if (x == (int)x) {
    if (x > 0.0) {
      ga = 1.0;               // use factorial
      for (i=2;i<x;i++) {
        ga *= i;
      }
     }
     else
      ga = 1e308;
   }
   else {
    if (fabs(x) > 1.0) {
      z = fabs(x);
      m = (int)z;
      r = 1.0;
      for (k=1;k<=m;k++) {
        r *= (z-k);
      }
      z -= m;
    }
    else
      z = x;
    gr = g[24];
    for (k=23;k>=0;k--) {
      gr = gr*z+g[k];
    }
    ga = 1.0/(gr*z);
    if (fabs(x) > 1.0) {
      ga *= r;
      if (x < 0.0) {
        ga = -M_PI/(x*ga*sin(M_PI*x));
      }
    }
  }
  return ga;
}


//**********************************************************************
void print_progressbar(double percentage, int length, string symbol)
// Print out a progress bar with the given percentage. Use a negative value to reset the progress bar.
{
  static int status=0;
  static int previous_stop=0;

  if (percentage<0)
  {
    // reset
    status = 0;
    previous_stop = 0;
  }

  // initializing progressbar
  if (status==0)
  {
    cout << "\r";
    cout << "[";
    for (int i=1; i<=length; i++) cout << " ";
    cout << "]";
    cout << "\r";
    cout << "[";
  }

  // plot status
  int stop;
  if (percentage>=0.99) stop=0.99*length;
  else stop = percentage*length;
  for (int i=previous_stop; i<stop; i++) cout << symbol;
  if (previous_stop<stop) previous_stop=stop;

  // simulate a rotating bar
  if (status==0) cout << "-";
  switch (status)
  {
    case 1: cout << "\\"; break;
    case 2: cout << "|"; break;
    case 3: cout << "/"; break;
    case 4: cout << "-"; break;
  }
  cout << "\b";
  status++;
  if (status==5) status=1;
  cout.flush();
}


//**********************************************************************
void formatedPrint(ostream& os, int count, ...)
// For easier scientific data outputing.
{
  va_list ap;
  va_start(ap, count); //Requires the last fixed parameter (to get the address)
  for(int j=0; j<count; j++)
      os << scientific << setprecision(10) << "  " << va_arg(ap, double); //Requires the type to cast to. Increments ap to the next argument.
  va_end(ap);
  os << endl;
}



//**********************************************************************
void display_logo(int which)
// Personal amusement.
{
  switch (which)
  {
    case 1:
    cout << " ____  ____            _                    " << endl;
    cout << "|_   ||   _|          (_)                   " << endl;
    cout << "  | |__| |    .---.   __    _ .--.    ____  " << endl;
    cout << "  |  __  |   / /__\\\\ [  |  [ `.-. |  [_   ] " << endl;
    cout << " _| |  | |_  | \\__.,  | |   | | | |   .' /_ " << endl;
    cout << "|____||____|  '.__.' [___] [___||__] [_____]" << endl;
    cout << "                                            " << endl;
    break;

    case 2:
    cout << ":::    ::: :::::::::: ::::::::::: ::::    ::: :::::::::" << endl;
    cout << ":+:    :+: :+:            :+:     :+:+:   :+:      :+: " << endl;
    cout << "+:+    +:+ +:+            +:+     :+:+:+  +:+     +:+  " << endl;
    cout << "+#++:++#++ +#++:++#       +#+     +#+ +:+ +#+    +#+   " << endl;
    cout << "+#+    +#+ +#+            +#+     +#+  +#+#+#   +#+    " << endl;
    cout << "#+#    #+# #+#            #+#     #+#   #+#+#  #+#     " << endl;
    cout << "###    ### ########## ########### ###    #### #########" << endl;
    break;

    case 3:
    cout << " __  __     ______     __     __   __     _____    " << endl;
    cout << "/\\ \\_\\ \\   /\\  ___\\   /\\ \\   /\\ '-.\\ \\   /\\___  \\  " << endl;
    cout << "\\ \\  __ \\  \\ \\  __\\   \\ \\ \\  \\ \\ \\-.  \\  \\/_/  /__ " << endl;
    cout << " \\ \\_\\ \\_\\  \\ \\_____\\  \\ \\_\\  \\ \\_\\\\'\\_\\   /\\_____\\" << endl;
    cout << "  \\/_/\\/_/   \\/_____/   \\/_/   \\/_/ \\/_/   \\/_____/" << endl;
    break;

  }

}



//**********************************************************************
void GaussLegendre_getWeight(int npts,double* xg,double* wg, double A, double B, int iop)
// Calculate the sampling location and weight for Gauss-Legendre quadrature
// -- From Hirano and Nara's MC-KLN code.
{
//ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//  gauss.f: Points and weights for Gaussian quadrature                 c
//                       c
//  taken from: "Projects in Computational Physics" by Landau and Paez  c
//         copyrighted by John Wiley and Sons, New York            c
//                                                                      c
//  written by: Oregon State University Nuclear Theory Group            c
//         Guangliang He & Rubin H. Landau                         c
//  supported by: US National Science Foundation, Northwest Alliance    c
//                for Computational Science and Engineering (NACSE),    c
//                US Department of Energy                          c
//                       c
//  comment: error message occurs if subroutine called without a main   c
//  comment: this file has to reside in the same directory as integ.c   c
//ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  static const double EPS = 3.0e-14;
  int m=(npts+1)/2;
  for(int i=0;i<m;i++) {
    double  t=cos(M_PI*(i+1-0.25)/(npts+0.5));
    double t1=t;
    double pp;
    do {
        double p1=1.0;
        double p2=0.0;
        double aj=0.0;
        for(int j=0;j<npts;j++) {
            double p3=p2;
            p2=p1;
            aj += 1.0;
            p1=((2.0*aj-1.0)*t*p2-(aj-1.0)*p3)/aj;
        }
        pp=npts*(t*p1-p2)/(t*t-1.0);
        t1=t;
        t=t1-p1/pp;
    } while(abs(t-t1)>EPS);
    xg[i]=-t;
    xg[npts-1-i]=t;
    wg[i]=2.0/((1.0-t*t)*pp*pp);
    wg[npts-1-i]=wg[i];
    }

//GaussLegendre::GaussRange(int N,int iop,double A,double B,
//  double* xg1,double* wg1)
//{
//     transform gausspoints to other range than [-1;1]
//     iop = 1  [A,B]       uniform
//     iop = 2  [0,inf]     A is midpoint
//     opt = 3  [-inf,inf]  scale is A
//     opt = 4  [B,inf]     A+2B is midoint
//     opt = 5  [0,B]     AB/(A+B)+ is midoint

  int N=npts;
  double xp, wp;
  for(int i=0; i<N; i++) {
      if(iop == 1) {
          //...... A to B
          xp=(B+A)/2+(B-A)/2*xg[i];
          wp=(B-A)/2*wg[i];
      } else if(iop == -1) {
          //......   A to B
          xp=(B+A)/2+(B-A)/2*xg[i];
          if(i <= N/2)
            xp=(A+B)/2-(xp-A);
          else
            xp=(A+B)/2+(B-xp);
          wp=(B-A)/2*wg[i];
      } else if(iop == 2) {
          //...... zero to infinity
          xp=A*(1+xg[i])/(1-xg[i]);
          double tmp=(1-xg[i]);
          wp=2.*A/(tmp*tmp)*wg[i];
      } else if(iop ==  3) {
          //...... -inf to inf scale A
          xp=A*(xg[i])/(1-xg[i]*xg[i]);
          double tmp=1-xg[i]*xg[i];
          wp=A*(1+xg[i]*xg[i])/(tmp*tmp)*wg[i];
      } else if(iop == 4) {
          //......  B to inf,  A+2B is midoint
          xp=(A+2*B+A*xg[i])/(1-xg[i]);
          double tmp=1-xg[i];
          wp=2.*(B+A)/(tmp*tmp)*wg[i];
      } else if(iop == 5) {
          //...... -A to A , scale B
          //xp=A*pow(abs(xg[i]),B) *sign(1.0,xg(i));
          double tmp = xg[i] >= 0 ? 1.0 : -1.0;
          xp=A*pow(abs(xg[i]),B) * tmp;
          //xp=A*pow(abs(xg[i]),B) *sign(1.0,xg(i));
          wp=A*B*pow(abs(xg[i]),(B-1))*wg[i];
      } else if(iop ==  6) {
          //...... 0 to B , AB/(A+B) is midpoint
          xp=A*B*(1+xg[i])/(B+A-(B-A)*xg[i]);
          double tmp = B+A-(B-A)*xg[i];
          wp=2*A*B*B/(tmp*tmp)*wg[i];
      } else {
          cerr << " invalid option iop = " << iop << endl;
          exit(1);
      }
      xg[i]=xp;
      wg[i]=wp;
  }
}





//**********************************************************************
void get_bin_average_and_count(istream& is, ostream& os, vector<double>* bins, long col_to_bin, void (*func)(vector<double>*), long wanted_data_columns, bool silence)
// Group data into bins by set by the "bins". The data in the column
// "col_to_bin" read from "is" are the ones used to determine the binning.
// Once the binning is decided, the averages of all data are calculated.
// The result, which has the same number of rows and the number of bins,
// will be outputted to "os". The i-th line of the output data contains the
// average of data from each column, for the i-th bin. The output will
// have 2 more columns; the 1st being the number count N and the 2nd being
// dN/dx where dx is the bin width.
// The values given in "bins" define the boundaries of bins and is assumed
// to be non-decreasing.
// After each line is decided to go into which bin, the function specified
// by "func" will be called to transform data. It is the transformed data
// that will be averaged. The transformed data can have different number of
// columns than the data passed in, in which case its number of columns
// is specified by "wanted_data_columns". The counting info will still
// be always written in the last two columns.
// The function specified by "func" should accepts a vector of doubles
// which is one line of data, and then modify it as returned result. The
// data vector passed in already has the desired length so it can be modified
// directly.
// The argument "col_to_bin" starts with 1.
// Refer also to getBinnedAverageAndCount MATLAB program.
{
  // initialization
  char* buffer = new char[99999]; // each line should be shorter than this
  // get first line and continue initialization
  is.getline(buffer, 99999);
  vector<double> line_data = stringToDoubles(buffer);
  long number_of_cols = line_data.size();
  long number_of_bins = bins->size()-1;

  if (number_of_cols==0)
  {
    cout << "get_bin_average_and_count error: the input data is empty!" << endl;
    exit(-1);
  }

  // create the counting array
  if (wanted_data_columns>0) number_of_cols = wanted_data_columns;
  double bin_total_and_count[number_of_bins][number_of_cols+2];
  for (long i=0; i<number_of_bins; i++)
  for (long j=0; j<number_of_cols+2; j++)
  {
    bin_total_and_count[i][j] = 0;
  }

  // add up all data
  long number_of_lines=1;
  while (is.eof()==false)
  {
    // determine which bin
    long bin_idx = binarySearch(bins, line_data[col_to_bin-1], true);
    if (bin_idx!=-1)
    {
      // transform data
      line_data.resize(number_of_cols, 0);
      if (func) (*func) (&line_data);
      // add to the counting matrix
      for (long j=0; j<number_of_cols; j++) bin_total_and_count[bin_idx][j] += line_data[j];
      // also the counting column
      bin_total_and_count[bin_idx][number_of_cols] ++;
    }
    // next iteration
    is.getline(buffer, 99999);
    line_data = stringToDoubles(buffer);
    if (number_of_lines % 100000 == 0 && !silence) cout << "Line " << number_of_lines << " reached." << endl;
    number_of_lines++;
  }

  // find the averages
  for (long i=0; i<number_of_bins; i++)
  for (long j=0; j<number_of_cols; j++)
  {
    if (bin_total_and_count[i][number_of_cols]<1e-15) continue;
    bin_total_and_count[i][j] /= bin_total_and_count[i][number_of_cols];
  }


  // get dN/(d bin_width)
  for (long i=0; i<number_of_bins; i++)
  {
    if (bin_total_and_count[i][number_of_cols]<1e-15) continue;
    bin_total_and_count[i][number_of_cols+1] = bin_total_and_count[i][number_of_cols]/((*bins)[i+1]-(*bins)[i]);
  }

  // output
  for (long i=0; i<number_of_bins; i++)
  {
    for (long j=0; j<number_of_cols+2; j++)
    {
      os << scientific << scientific << setprecision(10) << bin_total_and_count[i][j] << "  ";
    }
    os << endl;
  }

}



double aL_fit(double pl_peq_ratio)
{
  // calculates the anistropic parameter alphaL as a function of PL/Peq
  // using the conformal factorization approximation

  double x = pl_peq_ratio;  // longitudinal pressure / equilibrium pressure

  double x2 = x * x;
  double x3 = x2 * x;
  double x4 = x3 * x;
  double x5 = x4 * x;
  double x6 = x5 * x;
  double x7 = x6 * x;
  double x8 = x7 * x;
  double x9 = x8 * x;
  double x10 = x9 * x;
  double x11 = x10 * x;
  double x12 = x11 * x;
  double x13 = x12 * x;
  double x14 = x13 * x;

  double result = (2.307660683188896e-22 + 1.7179667824677117e-16*x + 7.2725449826862375e-12*x2 + 4.2846163672079405e-8*x3 + 0.00004757224421671691*x4 +
     0.011776118846199547*x5 + 0.7235583305942909*x6 + 11.582755440134724*x7 + 44.45243622597357*x8 + 12.673594148032494*x9 -
     33.75866652773691*x10 + 8.04299287188939*x11 + 1.462901772148128*x12 - 0.6320131889637761*x13 + 0.048528166213735346*x14)/
   (5.595674409987461e-19 + 8.059757191879689e-14*x + 1.2033043382301483e-9*x2 + 2.9819348588423508e-6*x3 + 0.0015212379997299082*x4 +
     0.18185453852532632*x5 + 5.466199358534425*x6 + 40.1581708710626*x7 + 44.38310108782752*x8 - 55.213789667214364*x9 +
     1.5449108423263358*x10 + 11.636087951096759*x11 - 4.005934533735304*x12 + 0.4703844693488544*x13 - 0.014599143701745957*x14);

  return result;
}


double R200(double aL)
{
  // calculates the R200 function associated with kinetic energy density

  double result;

  double x = (1.0 / (aL * aL)) - 1.0;  // same as xi in conformal ahydro
  double t200;
  double delta = 0.01;

  if(x > delta)
  {
    t200 = 1.0 + (1.0 + x) * atan(sqrt(x))/sqrt(x);
  }
  else if(x < -delta && x > -1.0)
  {
    t200 = 1.0 + (1.0 + x) * atanh(sqrt(-x))/sqrt(-x);
  }
  else if(x >= -delta && x <= delta)
  {
    t200 = 2.0 + x*(0.6666666666666667 + x*(-0.1333333333333333 +
      x*(0.05714285714285716 + x*(-0.031746031746031744 + x*(0.020202020202020193 +
      x*(-0.013986013986013984 + (0.010256410256410262 - 0.00784313725490196*x)*x))))));
  }
  else if(x <= -1.0)
  {
    cout << "x is out of bounds!" << endl;
    exit(-1);
  }

  result = (aL * t200);

  return result;
}





// For solving matrix equation Mij.pmodj = pi

void LUP_decomposition(double ** A, int n, int * pvector)
{
  // takes A and decomposes it into LU (with row permutations P)
  // A = n x n matrix; function does A -> PA = LU; (L,U) of PA stored in same ** array
  // n = size of A
  // pvector = permutation vector; set initial pvector[i] = i (to track implicit partial pivoting)
  //       function updates pvector if there are any rows exchanges in A (to be used on b in LUP_solve)

  double EPS_MIN = 1.0e-16;

  int i;     // rows
  int j;     // columns
  int k;     // dummy matrix index
  int imax;  // pivot row index
  double big;
  double sum;
  double temp;
  double implicit_scale[n];

  // Initialize permutation vector
  // to default no-pivot values
  for(i = 0; i < n; i++)
  {
    pvector[i] = i;
  }
  // Implicit scaling info. for A
  for(i = 0; i < n; i++)
  {
    big = 0.0;
    for(j = 0; j < n; j++)
    {
      temp = fabs(A[i][j]);
      if(temp > big)
      {
        big = temp;  // update biggest element in the ith row
      }
    }
    if(big == 0.0)
    {
      printf("Singular matrix in the routine");
      break;
    }
    implicit_scale[i] = 1.0 / big;  // store implicit scale of row i (will be used later)
  }
  // LU Decomposition
  for(j = 0; j < n; j++)
  {
    // loop rows i = 1 to j - 1
    for(i = 0; i < j; i++)
    {
      sum = A[i][j];
      for(k = 0; k < i; k++)
      {
        sum -= A[i][k] * A[k][j];
      }
      A[i][j] = sum;  // update U[i][j] elements
    }

    big = 0.0;          // initialize search for the largest normalized pivot in j column

    // loop through rows i = j to n-1
    for(i = j; i < n; i++)
    {
      sum = A[i][j];
      for(k = 0; k < j; k++)
      {
        sum -= A[i][k] * A[k][j];
      }
      A[i][j] = sum;   // update U[j][j] and L[i][j] elements (no division in L yet until the pivot determined)

      temp = implicit_scale[i] * fabs(sum);
      if(temp >= big)
      {
        big = temp;  // searchs for the biggest normalized member (imax) in column j
        imax = i;  // implicit scale * A[i][j] normalizes each row entry i in column j before comparing
      }          // implicit scale different for each i, that's why it's important
    }
    if(j != imax)
    {
      // then exchange rows j and imax of A
      for(k = 0; k < n; k++)
        {
          temp = A[imax][k];
          A[imax][k] = A[j][k];
          A[j][k] = temp;
        }
      implicit_scale[imax] = implicit_scale[j];   // interchange scaling
    }

    pvector[j] = imax;        // update permutation vector keeps track of pivot indices

    if(A[j][j] == 0.0)
    {
      A[j][j] = EPS_MIN;        // matrix is singular
    }
    if(j != n-1)                  // there is no L[n,n] element
    {
      temp = 1.0 / A[j][j];     // divide L[i,j] elements by the pivot
      for(i = j+1; i < n; i++)
      {
        A[i][j] *= temp;
      }
    }
  }
}

void LUP_solve(double ** PA, int n, int pvector[], double b[])
{
  // input vector b is transformed to the solution x  of Ax = b
  // PA is the input permutated matrix from LUP_decomposition (will not be updated here)
  // input pvector comes from LUP_decomposition (generally not default); used to switch rows of b

  // Forward substitution routine for Ly = b
  for(int i = 0; i < n; i++)
  {
    int ip = pvector[i];     // permute b accordingly
    double sum = b[ip];          // starting value given right b[ip]
    b[ip] = b[i];                // switch value of b[ip] to b[i]
    for(int j = 0; j < i; j++)
    {
      sum -= PA[i][j] * b[j];    // forward iteration
    }
    b[i] = sum;                  // update y and store in b
  }

  // Backward substitution routine for Ux = y
  for(int i = n-1; i >= 0; i--)
  {
    double sum = b[i];
    for(int j = i+1; j < n; j++)
    {
      sum -= PA[i][j] * b[j];       // backward iteration
    }
    b[i] = sum / PA[i][i];          // solution
  }
}


void matrix_multiplication(double ** A, const double x[], double y[], int n, int m)
{
  // multiplies y_i = A_ij * x_j (stores result in y)

  for(int i = 0; i < n; i++)    // rows
  {
    // initialize to zero
    y[i] = 0.0; 

    for(int j = 0; j < m; j++)  // columns
    {
      y[i] += A[i][j] * x[j];
    }
  }
}

void vector_copy(const double a[], double c[], int n)
{
  // copy a to c
  for(int i = 0; i < n; i++)
  {
    c[i] = a[i];
  } 
}

void vector_addition(const double a[], const double b[], double c[], int n)
{
  // adds c_i = a_i + b_i (stores result in c)
  for(int i = 0; i < n; i++)
  {
    c[i] = a[i] + b[i];
  } 
}

void vector_subtraction(const double a[], const double b[], double c[], int n)
{
  // adds c_i = a_i - b_i (stores result in c)
  for(int i = 0; i < n; i++)
  {
    c[i] = a[i] - b[i];
  } 
}

// for deallocating 2D matrices

void free_2D(double ** M, int n)
{
  for(int i = 0; i < n; i++) free(M[i]);
  free(M);
}

void free_3D(double *** M, int n, int m)
{
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      free(M[i][j]);
    }
    free(M[i]);
  }
  free(M);
}



