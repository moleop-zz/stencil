all: heat_conduction.cu
	nvcc -arch=sm_35 heat_conduction.cu -o heat2_cond

debug: heat_conduction.cu
	nvcc -g -G -arch=sm_35 heat_conduction.cu -o heat2_cond
