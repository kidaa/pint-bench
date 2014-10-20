#
# Generic makefile rules for pint-bench benchmarks
#

CXX = mpic++
CXXFLAGS += -I$(PB)/opt/include

CC = mpicc
CFLAGS += -I$(PB)/opt/include

EXEC = $(foreach DRIVER,$(DRIVERS),$(BENCHMARK)-$(DRIVER))

all: $(EXEC)
