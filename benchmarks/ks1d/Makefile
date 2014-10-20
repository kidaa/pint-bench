
PB = $(abspath ../..)

BENCHMARK = ks1d
DRIVERS   = pfasst
PACKAGES  = fftw++

include $(PB)/mk/Makefile

CXXFLAGS += -Icommon -Ifftw++
CXXFLAGS += -fopenmp -Wall -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse

vpath %.cc fftw++
CXXSRC += fftw++.cc convolution.cc

vpath %.cpp common
CXXSRC += ks1d.cpp



OBJ = $(addprefix build/, $(addsuffix .o,$(basename $(CXXSRC))))

ks1d-pfasst: pfasst/main.cpp $(OBJ)
	$(CXX) -std=c++11 -o $@ $(CXXFLAGS) $^ -lfftw3_omp -lfftw3

build/%.o: %.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $^

build/%.o: %.cc
	@mkdir -p build
	$(CXX) -c -o $@ $(CXXFLAGS) $^
