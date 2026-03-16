all:
	g++ -std=c++11 -Wall -O3 -march=native -ffast-math -fopenmp -DUSE_BLAS main.cpp -o main_cblas -lopenblas

	g++ -std=c++11 -Wall -O3 -march=native -ffast-math -fopenmp main.cpp -o main_native  -lopenblas


	nvcc -O3 -arch=sm_70 main.cu -o main_gpu

	nvcc -O3 -arch=sm_70 -DUSE_CUBLAS main.cu -o main_gpu_cublas -lcublas
clean:
	rm -rf main_native main_cblas main_gpu main_gpu_cublas