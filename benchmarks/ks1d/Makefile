
PB = $(abspath ../..)

BENCHMARK = ks1d
DRIVERS   = pfasst ark4 xbraid

include $(PB)/mk/Makefile

CXXSRC   += ks1d.cpp
CXXSRC   += fftw++.cc convolution.cc
CXXFLAGS += -Ifftw++
CXXFLAGS += -g
#CXXFLAGS += -fopenmp -Wall -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse

vpath %.cc fftw++
vpath %.cpp ks1d

OBJ = $(addprefix build/, $(addsuffix .o,$(basename $(CXXSRC))))

ks1d-pfasst: pfasst/main.cpp $(OBJ)
	$(CXX) -std=c++11 -o $@ $(CXXFLAGS) $^ -lfftw3_omp -lfftw3

ks1d-ark4: ark4/main.cpp $(OBJ)
	$(CXX) -std=c++11 -o $@ $(CXXFLAGS) $^ -lfftw3_omp -lfftw3

ks1d-xbraid: xbraid/main.cpp xbraid/braid_vector.hpp $(OBJ)
	$(CXX) -fopenmp -std=c++11 -o $@ $(CXXFLAGS) $^ -lfftw3_omp -lfftw3 -L$(PB)/pb/lib -lbraid

ks1d-ridc: ridc/main.cpp ridc/ridc.cpp ridc/ridc_utils.cpp $(OBJ)
	$(CXX) -fopenmp -std=c++11 -o $@ $(CXXFLAGS) $^ -lfftw3_omp -lfftw3
