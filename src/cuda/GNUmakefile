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
#CC = g++
#CC = pgc++
CC = nvcc
#CFLAGS= -O3 -Wall -march=native
#CFLAGS= -O3 -Wall -fopenmp -march=native
#CFLAGS= -acc -fast -Minfo -ta=nvidia,time
#CFLAGS= -acc -fast -ta=tesla:managed #uses unified memory - simplifies copying of dynamic arrays host <-> device

RM		=	rm -f
O               =       .o
LDFLAGS         =       $(CFLAGS)
SYSTEMFILES     =       $(SRCGNU)

# --------------- Files involved ------------------

ifeq "$(MAIN)" ""
MAIN		=	iS3D_GPU.e
endif

SRC   = main.cu arsenal.cu emissionfunction.cu Table.cu readindata.cu ParameterReader.cu gaussThermal.cu
INC   = main.cuh arsenal.cuh emissionfunction.cuh Table.cuh readindata.cuh Stopwatch.cuh ParameterReader.cuh gaussThermal.cuh

# -------------------------------------------------

OBJDIR		=	obj
SRCFILES 	= 	$(SRC) $(INC) GNUmakefile
OBJECTS		=	$(addprefix $(OBJDIR)/, $(addsuffix $O, \
			$(basename $(SRC))))
TARGET		=	$(MAIN)
INSTPATH	=	../

# --------------- Pattern rules -------------------

$(OBJDIR)/%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.cu:
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
./main.cu: main.cuh Table.cuh readindata.cuh emissionfunction.cuh arsenal.cuh ParameterReader.cuh
./arsenal.cu: arsenal.cuh
./emissionfunction.cu: main.cuh readindata.cuh emissionfunction.cuh Stopwatch.cuh arsenal.cuh Table.cuh
./Table.cu: Table.cuh arsenal.cuh
./readindata.cu: main.cuh readindata.cuh arsenal.cuh Table.cuh ParameterReader.cuh
./ParameterReader.cu: arsenal.cuh ParameterReader.cuh
#./deltafReader.cu: main.cuh deltafReader.cuh ParameterReader.cuh readindata.cuh gaussThermal.cuh
./gaussThermal.cu: gaussThermal.cuh

