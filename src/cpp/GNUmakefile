# ===========================================================================
#  Makefile iS3D
# ===========================================================================
##
##  Environments :	MAIN	= 	main sourcefile
##
##  Usage : 	(g)make	[all]		compile the whole project
##			install	make all and copy binary to $INSTPATH
##			clean		remove objectfiles in obj_$TYPE
##			distclean	remove all objectsfiles and binaries
##

#CC := $(shell ../../crank/configure_compiler_C++.sh)
#CC = g++-7 #Mac OSX high sierra
#CC = pgc++
CC = g++
#CFLAGS= -std=c++11 -march=native -O3 -lgsl -lgslcblas -lm
CFLAGS= -std=c++11 -fopenmp -march=native -O3 -lgsl -lgslcblas -lm
#CFLAGS= -acc -fast -Minfo -ta=nvidia,time
#CFLAGS= -acc -fast -ta=tesla:managed #uses unified memory - simplifies copying of dynamic arrays host <-> device

RM		=	rm -f
O               =       .o
LDFLAGS         =       $(CFLAGS)
SYSTEMFILES     =       $(SRCGNU)

# --------------- Files involved ------------------

ifeq "$(MAIN)" ""
MAIN		=	iS3D.e
endif

SRC   =	main.cpp arsenal.cpp emissionfunction.cpp emissionfunction_smooth_kernels.cpp emissionfunction_sampling_kernels.cpp Table.cpp readindata.cpp ParameterReader.cpp deltafReader.cpp gaussThermal.cpp
INC   = main.h arsenal.h emissionfunction.h Table.h readindata.h Stopwatch.h ParameterReader.h deltafReader.h gaussThermal.h


# -------------------------------------------------

OBJDIR		=	obj
SRCFILES 	= 	$(SRC) $(INC) GNUmakefile
OBJECTS		=	$(addprefix $(OBJDIR)/, $(addsuffix $O, \
			$(basename $(SRC))))
TARGET		=	$(MAIN)
INSTPATH	=	../

# --------------- Pattern rules -------------------

$(OBJDIR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.cpp:
	if [ -f $@ ] ; then touch $@ ; else false ; fi

# -------------------------------------------------

.PHONY:		all mkobjdir clean distclean install

all:		mkobjdir $(TARGET)

help:
		@grep '^##' GNUmakefile

mkobjdir:
		-@mkdir -p $(OBJDIR)

$(TARGET):	$(OBJECTS)
		$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
#		strip $(TARGET)

clean:
		-rm $(OBJECTS)

distclean:
		-rm $(TARGET)
		-rm -r obj

install:	$(TARGET)
		cp $(TARGET) $(INSTPATH)

# --------------- Dependencies -------------------
./main.cpp: main.h Table.h readindata.h emissionfunction.h arsenal.h ParameterReader.h
./arsenal.cpp: arsenal.h
./emissionfunction.cpp: main.h readindata.h emissionfunction.h Stopwatch.h arsenal.h Table.h gaussThermal.h
./emissionfunction_smooth_kernels.cpp: main.h readindata.h emissionfunction.h Stopwatch.h arsenal.h Table.h gaussThermal.h
./emissionfunction_sampling_kernels.cpp: main.h readindata.h emissionfunction.h Stopwatch.h arsenal.h Table.h gaussThermal.h
./Table.cpp: Table.h arsenal.h
./readindata.cpp: main.h readindata.h arsenal.h Table.h ParameterReader.h
./ParameterReader.cpp: arsenal.h ParameterReader.h
./gaussThermal.cpp: gaussThermal.h
