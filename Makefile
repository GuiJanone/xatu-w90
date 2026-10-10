# Compiler & compiler flags
CC = g++
FC = gfortran
CFLAGS = -O2 -Wall -lm
FFLAGS = -O2 -Wall -Wno-tabs -lm

TEST_FP_FLAGS = \
    -fno-fast-math \
    -fno-unsafe-math-optimizations \
    -ffp-contract=off \
    -frounding-math \
    -fno-associative-math \
    -fno-reciprocal-math

# Include folders
INCLUDE = -I$(PWD)/include

# Libraries
LIBS = -DARMA_DONT_USE_WRAPPER -L$(PWD) -lxatu -larmadillo -lopenblas -llapack -larpack -fopenmp -lgfortran
#LIBS = -DARMA_DONT_USE_WRAPPER -DARMA_BLAS_64BIT_INT -L$(PWD) -lxatu -larmadillo -lopenblas -llapack -larpack -fopenmp -lgfortran

# Conditional flags for compilation
ifeq ($(DEBUG), 1)
	CFLAGS = -Wall -lm -g
	FFLAGS = -Wall -Wno-tabs -lm -g
endif
ifeq ($(TEST), 1)
	CFLAGS = -Wall -lm $(TEST_FP_FLAGS)
	FFLAGS = -Wall -Wno-tabs -lm $(TEST_FP_FLAGS)
endif
ifeq ($(HDF5), 1)
	CFLAGS += -DARMA_USE_HDF5
	LIBS += -lhdf5
endif
# LAPACK/BLAS and ARPACK can be replaced on the command line (cluster builds, MKL, ILP64), e.g.
#   make build xatu LAPACK_LIBS="-L/path/to/openblas/lib -lopenblas"
LAPACK_LIBS ?= -lopenblas -llapack
ARPACK_LIBS ?= -larpack
LIBS := $(filter-out -lopenblas -llapack -larpack,$(LIBS)) $(ARPACK_LIBS) $(LAPACK_LIBS) -lgfortran
# 64-bit-integer LAPACK (ILP64): lifts the 32-bit workspace limit of 'diag' (zheevd, dimension <= 32766).
# LAPACK_LIBS must then name an ILP64 library with the standard symbol names, e.g.
#   make build xatu ILP64=1 LAPACK_LIBS="-L/opt/openblas-ilp64/lib -lopenblas" ARPACK_LIBS="-L/opt/arpack-ilp64/lib -larpack"
# ARPACK_LIBS= (empty) builds without ARPACK (method 'sparse' unavailable). Run 'make clean' when switching.
ifeq ($(ILP64), 1)
    ifeq ($(origin LAPACK_LIBS), file)
        $(error ILP64=1 needs LAPACK_LIBS pointing to a 64-bit-integer LAPACK/BLAS)
    endif
    LIBS := $(filter-out -larmadillo,$(LIBS)) -DARMA_BLAS_64BIT_INT -DXATU_ILP64
    ifeq ($(strip $(ARPACK_LIBS)),)
        LIBS += -DARMA_DONT_USE_ARPACK
    endif
endif

# Compilation targets
CC_SRC_FILES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst src/%.cpp, build/%.o, $(CC_SRC_FILES))
FC_SRC_FILES := $(wildcard src/*.f90)
OBJECTS_FC := $(patsubst src/%.f90, build/%.o, $(FC_SRC_FILES))
OBJECTS += $(OBJECTS_FC)

# Create folders
dummy_build_folder := $(shell mkdir -p build)
dummy_bin_folder := $(shell mkdir -p bin)

build:	$(OBJECTS)
	ar rcs libxatu.a $(OBJECTS) 
	
xatu: main/xatu.cpp $(OBJECTS) 
	$(CC) -o bin/$@ $< $(CFLAGS) $(INCLUDE) $(LIBS)

%: main/%.cpp $(OBJECTS)
	$(CC) -o bin/$@ $< $(CFLAGS) $(INCLUDE) $(LIBS)

# Compilation steps
# $< refers to first prerequisite and $@ to the target
build/%.o: src/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS) $(INCLUDE) $(LIBS) 

build/%.o: src/%.f90
	$(FC) -c $< -o $@ $(FFLAGS) $(LIBS) $(INCLUDE)

clean:
	rm -rf build/*.o bin/* libxatu.a
