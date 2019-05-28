
UNAME := $(shell uname)
NVCC := $(shell command -v nvcc 2> /dev/null)

DIR_MAIN = ./
DIR_SRC = ./src
DIR_OBJ = ./src

CFLAGS= $(OPTIMIZATION) $(OPTIONS)

# Compiler options
COMPILER = icpc
OPTIMIZATION = -O3
CFLAGS := $(CFLAG) -Wno-comment

# Library options
ifeq ($(UNAME), Linux)
LIBS := -lm -lgsl 
endif
ifeq ($(UNAME), Darwin)
LIBS := -lc++ -lm 
#LIBS := -lm -lgsl
endif

INCLUDES = -I include

CPP := $(shell find $(DIR_SRC) -name '*.cpp')
OBJ =$(CPP:$(DIR_SRC)%.cpp=$(DIR_OBJ)%.o)

EXE = deltaf_table

# $@ = target
# $< = dependency
# $^ = + everything else

$(EXE): $(OBJ)
	echo "Linking   $@ ($(COMPILER))"
	$(COMPILER) -o $@ $^ $(LIBS) $(INCLUDES)
#	echo "\nRunning...\n"
#	$(DIR_MAIN)$(EXE)
#	rm *.o

$(DIR_OBJ)%.o: $(DIR_SRC)%.cpp
	echo "Compiling $< ($(COMPILER))"
	$(COMPILER) $(CFLAGS) $(INCLUDES) -c -o $@ $<

run:
	echo "Running..."
	$(DIR_MAIN)$(EXE)

clean:
	echo "Deleting executable and object files"
	rm -rf $(EXE)
	rm *.o

.SILENT:
