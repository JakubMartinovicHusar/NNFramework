#pragma once
#include "CudaConvertible.cu"
#include "Randomizers.cu"
#include <host_defines.h> /* the cuda specific header */

class Network {

private:
    float* weight_array;
	int weight_size;
	float* synapse_array_from;
	int synapse_array_from_size;
	float* synapse_array_to;
	int synapse_array_to_size;
	char transformation_function;
	__device__ void transform(char* transformation_function, float* value){
		if(transformation_function == 0)
		{		
			//value = __expf(2*value);
			//value =  (value - 1) / (value + 1);
		}
	}
public:
	__host__ __device__ Network(): transformation_function(0) {	}
    __host__ __device__  ~Network() {}
    __host__ __device__  float* Calculate(Data* data, const int input_size, const int output_size) 
	{
		
		float* result = new float[output_size];
		for(int i = 0; i<output_size; i++)
		{
			result[i] = 0.0f;
		}

		for(int j = 0; j<output_size; j++)
			for(int k = 0; k<input_size; k++)
			{
				result[j] +=  data->inputs[k];
		}
			return result;
	}

	void Initialize(){
		weight_array = new float[2];
		weight_array[0] = Randomize::sampleNormal();
		weight_array[1] = Randomize::sampleNormal();
	}

		Network * GetCudaCopy()
		{
			Network * t_dev = NULL;
			t_dev = CudaConvertible::ConvertPtr(this, t_dev);
			float* weight_array_tmp = NULL;
			weight_array_tmp = CudaConvertible::ConvertPtr(weight_array, weight_array_tmp);
			cudaMemcpy(&(t_dev->weight_array), &weight_array_tmp, sizeof(float*), cudaMemcpyHostToDevice);
			return t_dev;
		}

};