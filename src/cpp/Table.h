
#include <string>
#include <ostream>
#include <iostream>
#include <vector>

#ifndef Table_h
#define Table_h

using namespace std;

class Table
{
  private:
    std::vector< std::vector<double>* >* data;
    long numberOfCols, numberOfRows;
  public:
    Table();
    Table(string);
    Table(Table&);
    Table(long,long,double defaultValue=0.0);
    Table(double**, long, long);
    ~Table();
    void deleteTable();
    void loadTableFromFile(string);
    void loadTableFromDoubleArray(double**, long, long);
    void extendTable(long, long, double defaultValue=0);
    double get(long, long);
    void set(long, long, double, double defaultValue=0);
    std::vector<double>* getColumn(long);
    void setAll(double defaultValue=0.0);
    long getNumberOfRows();
    long getNumberOfCols();
    long getSizeDim1() {return getNumberOfCols();};
    long getSizeDim2() {return getNumberOfRows();};
    void printTable(ostream& os=std::cout);
    double getFirst(long);
    double getLast(long);
    double interp(long, long, double, int mode=6);
    double invert(long, long, double, int mode=6);
};

#endif
