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

#choose a compiler.
#CC = pgc++ # pgi compiler for openacc support  
#CC = g++
CC = icpc # intel compiler tends to perform much better than gcc

#choose flags corresponding to compiler  
#CFLAGS= -std=c++11 -fopenmp -march=native -O3 -lgsl -lgslcblas -lm # for g++
CFLAGS= -std=c++11 -fopenmp -march=native -O3 -lgsl -lgslcblas # for icpc
#CFLAGS= -acc -fast -Minfo -ta=nvidia,time # for pgi 
#CFLAGS= -acc -fast -ta=tesla:managed # for pgi, also uses unified memory

RM		=	rm -f
O               =       .o
LDFLAGS         =       $(CFLAGS)
SYSTEMFILES     =       $(SRCGNU)

# --------------- Files involved ------------------

ifeq "$(MAIN)" ""
MAIN		=	iS3D.e
endif

SRC   =	main.cpp arsenal.cpp emissionfunction.cpp emissionfunction_smooth_kernels.cpp emissionfunction_sampling_kernels.cpp emissionfunction_polzn_kernels.cpp Table.cpp readindata.cpp ParameterReader.cpp deltafReader.cpp gaussThermal.cpp viscous_correction.cpp
INC   = main.h arsenal.h emissionfunction.h Table.h readindata.h Stopwatch.h ParameterReader.h deltafReader.h gaussThermal.h viscous_correction.h


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
./emissionfunction_polzn_kernels.cpp: main.h readindata.h emissionfunction.h Stopwatch.h arsenal.h Table.h gaussThermal.h
./emissionfunction_sampling_kernels.cpp: main.h readindata.h emissionfunction.h Stopwatch.h arsenal.h Table.h gaussThermal.h viscous_correction.h
./Table.cpp: Table.h arsenal.h
./readindata.cpp: main.h readindata.h arsenal.h Table.h ParameterReader.h
./ParameterReader.cpp: arsenal.h ParameterReader.h
./gaussThermal.cpp: gaussThermal.h
