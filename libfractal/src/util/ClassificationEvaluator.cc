/*
   Copyright 2015 Kyuyeon Hwang (kyuyeon.hwang@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#include "ClassificationEvaluator.h"

#include <cmath>


namespace fractal
{

const double ClassificationEvaluator::GetLoss(const unsigned long outputIdx) const
{
	return GetAverageCrossEntropy(outputIdx);
}


const double ClassificationEvaluator::GetAverageCrossEntropy(const unsigned long outputIdx) const
{
	verify(outputIdx < nOutput);

	return ceSum[outputIdx] / nSample[outputIdx];
}


const double ClassificationEvaluator::GetFrameErrorRate(const unsigned long outputIdx) const
{
	verify(outputIdx < nOutput);

	return (double) nError[outputIdx] / nSample[outputIdx];
}


const unsigned long ClassificationEvaluator::GetFrameErrorCount(const unsigned long outputIdx) const
{
	verify(outputIdx < nOutput);

	return nError[outputIdx];
}


void ClassificationEvaluator::Reset()
{
	unsigned long i;

	RegressionEvaluator::Reset();

	for(i = 0; i < nOutput; i++)
	{
		nError[i] = 0;
		ceSum[i] = 0.0;
	}
}


void ClassificationEvaluator::EvaluateFrames(const unsigned long outputIdx, Matrix<FLOAT> &target,
		Matrix<FLOAT> &output, const unsigned long nStream, PStream &stream)
{
	unsigned long i, j, dim, nFrame;
	unsigned long idx, tMaxIdx, oMaxIdx, nPartialError;
	double sePartialSum, cePartialSum;
	FLOAT *t, *o, tCur, oCur;

	FLOAT err;
	FLOAT tMax, oMax;

	dim = target.GetNumRows();
	nFrame = target.GetNumCols();

	verify(dim == output.GetNumRows());
	verify(nFrame == output.GetNumCols());

	target.HostPull(stream);
	output.HostPull(stream);

	stream.engine->StreamSynchronize(stream);

	t = target.GetHostData();
	o = output.GetHostData();

	nSample[outputIdx] += nFrame;

	sePartialSum = 0.0;
	cePartialSum = 0.0;
	nPartialError = 0;

#ifdef FRACTAL_USE_OMP
	#pragma omp parallel for private(tCur, oCur, tMaxIdx, oMaxIdx, tMax, oMax, idx, err , j) \
		reduction(+:sePartialSum, cePartialSum, nPartialError)
#endif
	for(i = 0; i < nFrame; i++)
	{
		tMaxIdx = 0;
		oMaxIdx = 0;
		tMax = t[i * dim];
		oMax = o[i * dim];

		for(j = 0; j < dim; j++)
		{
			idx = i * dim + j;

			tCur = t[idx];
			oCur = o[idx];

			err = oCur - tCur;
			sePartialSum += err * err;

			if(tCur > (FLOAT) 0)
				cePartialSum -= tCur * std::log(oCur + (double) 1e-300);

			if(oCur > oMax)
			{
				oMax = oCur;
				oMaxIdx = j;
			}

			if(tCur > tMax)
			{
				tMax = tCur;
				tMaxIdx = j;
			}
		}

		if(tMaxIdx != oMaxIdx) nPartialError++;
	}


	seSum[outputIdx] += sePartialSum;
	ceSum[outputIdx] += cePartialSum;
	nError[outputIdx] += nPartialError;
}


void ClassificationEvaluator::MemAlloc()
{
	RegressionEvaluator::MemAlloc();

	nError.clear();
	ceSum.clear();

	nError.shrink_to_fit();
	ceSum.shrink_to_fit();

	nError.resize(nOutput);
	ceSum.resize(nOutput);
}

}

