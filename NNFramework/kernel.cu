#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "Data.cuh"
#include "Network.cuh"



cudaError_t trainCycleWithCuda(Network *network);//, DataList *data, int population_size);

__global__ void trainCycleKernel(Network *network)//, DataList *data, int *population_size)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	/*for(int m = 0; m<data->_datalist_size; m++){
		Network n = network[i];//.Calculate(&data[m]);
		float* results = n.Calculate(&(data->data[m]), data->_input_size,1);
		delete[] results;
	}*/
}

int main()
{

	int population_size = 100;
	int data_size = 1000;
	Network* network = new Network[population_size];
	const int input_size = 3;
	DataList* data = new DataList(data_size, input_size);
	for(int i = 0; i<data_size; i++) {
		float inputs[input_size];
		for(int j = 0; j<input_size; j++) {
			inputs[j] = static_cast<float>(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
		}
		data->AddData(inputs, i);
	}
	// Add vectors in parallel.
	cudaError_t cudaStatus = trainCycleWithCuda(network);//, data, population_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	system("pause");

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t trainCycleWithCuda(Network *network)//, DataList *data, int population_size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	Network *network_dev;
	//DataList *data_dev;
	//Data* datavalues_dev;
	/*  int *dev_a = 0;*/
	int *population_size_dev = 0;
	network[0].Initialize();

	network_dev = network[0].GetCudaCopy();
	/*int *data_size_dev = 0;
	


	// Choose which GPU to run on, change this on a multi-GPU system.


	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}



	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&population_size_dev, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(population_size)!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&data_size_dev, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(data_size)!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **) &network_dev, population_size * sizeof(Network));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(network)");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **) &data_dev, sizeof(DataList)/*+sizeof(Data)*data->_datalist_size*/
	/*);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(data)!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(network_dev, network, population_size *  sizeof(Network), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(network)!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(data_dev, data, sizeof(DataList)/*+sizeof(Data)*data->_datalist_size*//*, cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(network)!");
		goto Error;
	}
	*/
	/*
	//inputs = new float[data->_input_size];
	Data * dl = new Data[data->_datalist_size];
	for (size_t i = 0; i < data->_datalist_size; ++i)
	{
		float * ild;
		/*float * il = (float *)malloc(data->_input_size*sizeof(float));
		for(size_t j = 0; j < data->_input_size; ++j){
			il[j] = data->data[i].inputs[j];
		}
		cudaMalloc((void**) &ild, data->_input_size*sizeof(float));
		cudaMemcpy(ild, data->data[i].inputs, data->_input_size*sizeof(float), cudaMemcpyHostToDevice);

		Data * d;
		cudaStatus = cudaMalloc((void **) &d, sizeof(Data));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed(data)!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(d, &(data->data)[i], sizeof(Data), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed(network)!");
			goto Error;
		}
		d->inputs = ild;
		dl[i] = *d; 
		delete[] ild;
	}
	data_dev->data = dl;



//data_dev->data = &datavalues_dev;


cudaStatus = cudaMemcpy(population_size_dev, &population_size,  sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed(population_size)!");
	goto Error;
}
	*/

// Launch a kernel on the GPU with one thread for each element.
int numBlocks = 1;
dim3 threadsPerBlock((unsigned int)1,1);
trainCycleKernel<<<numBlocks, threadsPerBlock>>>(network_dev);//, data_dev, population_size_dev);
/*
// Check for any errors launching the kernel
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
}

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	goto Error;
}
*/
// Copy output vector from GPU buffer to host memory.
/* cudaStatus = cudaMemcpy(network_dev, network, sizeof(network), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}*/
/*
Error:
cudaFree(network_dev);
*/

return cudaStatus;
}
