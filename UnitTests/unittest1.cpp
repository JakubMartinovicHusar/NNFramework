#include "stdafx.h"
#include "CppUnitTest.h"
#include "../NNFramework/Data.cuh"  

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{		
	TEST_CLASS(UnitTest_Data)
	{
	public:
		
		TEST_METHOD(TestMethod_Data_toCuda)
		{
			float inpt[5] = {0,1,2,5,10};
			Data* data = new Data(inpt);
			Data* data->toCuda();

		}

	};
}