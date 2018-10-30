
GTEST = /home/tlj/Programs/gtest-1.7.0

# This part is if MKL is available. It is possible to switch using the
# -DMKL for compiling and adding the necessary linking arguments. 
# Below is a suggestion.
#MKLROOT = /opt/intel/composer_xe_2011_sp1.10.319
MKLROOT = /opt/intel/composer_xe_2011_sp1.11.339

#Dynamic linking
MKL = -m64 -I$(MKLROOT)/mkl/include -I$(MKLROOT)/mkl/include/fftw -L$(MKLROOT)/mkl/lib/intel64 -L$(MKLROOT)/compiler/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

DIR = ../src
LINK = -DMKL $(MKL) -DALIGNMENT32

#Include both single and double precision fftw libraries for convenience
#LINK = -lblas -llapack -lfftw3 -lfftw3f

all:
	g++ -Wall -g -Wno-write-strings -DDOUBLE -isystem $(GTEST)/include -I$(DIR) -pthread  $(DIR)/slp_joint.cpp $(DIR)/tools.cpp $(DIR)/vector.cpp $(DIR)/projection_l1.cpp $(DIR)/filter.cpp test.cpp libgtest.a  -o test_double $(LINK) 

	g++ -Wall -g -Wno-write-strings -DSINGLE -isystem $(GTEST)/include -I$(DIR) -pthread   $(DIR)/slp_joint.cpp $(DIR)/tools.cpp $(DIR)/vector.cpp $(DIR)/projection_l1.cpp $(DIR)/filter.cpp test.cpp libgtest.a -o test_single $(LINK) 
