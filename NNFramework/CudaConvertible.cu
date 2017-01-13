

class CudaConvertible
{
public:
	template<typename T>
	static T* ConvertPtr(T* value, T* destination){
		  cudaError_t cudaStatus;
	
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		cudaStatus = 
			cudaMalloc((void**)&destination, sizeof(T));
		if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }

	  cudaStatus = cudaMemcpy(destination, value, sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
	}	
	return destination;
	};

	
};