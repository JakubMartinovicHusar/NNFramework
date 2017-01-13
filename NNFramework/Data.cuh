#ifdef HAS_CUDA

#include <host_defines.h> /* the cuda specific header */

#define __cuda_align__(n) __align__((n)) /* the cuda alignment keyword */

#else

#define __cuda_align__(n) /* empty */

#endif

class __cuda_align__(16)  Data {

private:
    
public:
	float inputs;
	int count
	__host__ __device__ Data() 
	{
	}
	__host__ __device__ Data(float values, int count) 
	{
		inputs = values;
	}
    __host__ __device__  ~Data() {}

	Data* toCuda(){
		cudaError_t cudaStatus;
		int* d_inputs;
		cudaStatus = cudaMalloc((void**)&d_inputs, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed(population_size)!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(d_inputs, inputs, population_size *  sizeof(Network), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed(network)!");
			goto Error;
	}

	}
};

class  __cuda_align__(16) DataList {

public:
	int _input_size;
	int _datalist_size;
	Data* data;
	__host__ __device__ DataList(const int datalist_size, const int input_size) 
	{
		_datalist_size = datalist_size;
		_input_size = input_size;
		data = new Data[datalist_size];
	}
    __host__ __device__  ~DataList() {
		delete[] data;
	}

	  __host__ __device__  void AddData(float* values, int position) 
	{
		Data dt (values);
		data[position] = dt;
	}
};
