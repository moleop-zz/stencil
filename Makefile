all: heat2.cu
	nvcc -arch=sm_35 heat_conduction.cu -o heat_cond

debug: heat2.cu
	nvcc -g -G -arch=sm_35 heat_conduction.cu -o heat_cond
