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


#ifndef FRACTAL_CLASSIFICATIONEVALUATOR_H_
#define FRACTAL_CLASSIFICATIONEVALUATOR_H_

#include "RegressionEvaluator.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class ClassificationEvaluator : public RegressionEvaluator
{
public:
	ClassificationEvaluator() : RegressionEvaluator() {};

	virtual const double GetLoss(const unsigned long outputIdx) const;

	const double GetAverageCrossEntropy(const unsigned long outputIdx) const;
	const double GetFrameErrorRate(const unsigned long outputIdx) const;
	const unsigned long GetFrameErrorCount(const unsigned long outputIdx) const;

protected:
	virtual void Reset();
	virtual void EvaluateFrames(const unsigned long outputIdx, Matrix<FLOAT> &target,
			Matrix<FLOAT> &output, const unsigned long nStream, PStream &stream);
	virtual void MemAlloc();

	std::vector<unsigned long> nError;
	std::vector<double> ceSum;
};

}

#endif /* FRACTAL_CLASSIFICATIONEVALUATOR_H_ */

