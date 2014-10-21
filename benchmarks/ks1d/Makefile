
PB = $(abspath ../..)

BENCHMARK = ks1d
DRIVERS   = pfasst

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
