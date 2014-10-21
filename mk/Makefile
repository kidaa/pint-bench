#
# Generic makefile rules for pint-bench benchmarks
#

CXX = mpic++
CXXFLAGS += -I$(PB)/pb/include -I$(PB)/pb/include/eigen3

CC = mpicc
CFLAGS += -I$(PB)/pb/include

CXXFLAGS += -I$(BENCHMARK)
CFLAGS   += -I$(BENCHMARK)

EXEC = $(foreach DRIVER,$(DRIVERS),$(BENCHMARK)-$(DRIVER))

all: $(EXEC)

build/%.o: %.cpp
	@mkdir -p build
	$(CXX) -c -o $@ $(CXXFLAGS) $^

build/%.o: %.cc
	@mkdir -p build
	$(CXX) -c -o $@ $(CXXFLAGS) $^
