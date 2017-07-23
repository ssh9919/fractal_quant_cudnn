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


#ifndef FRACTAL_AUTOOPTIMIZER_H_
#define FRACTAL_AUTOOPTIMIZER_H_


#include <string>

#include "Optimizer.h"
#include "Evaluator.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class AutoOptimizer
{
public:
	AutoOptimizer();
	virtual ~AutoOptimizer() {}

	void Optimize(Rnn &rnn, Stream &trainStream, Stream &evalStream, Evaluator &evaluator,
			const PortMapList &inputPorts, const PortMapList &outputPorts,
			const unsigned long nTrainFramePerEpoch, const unsigned long nEvalFramePerEpoch,
			const unsigned long windowSize, const unsigned long stepSize);

	void SetWorkspacePath(const std::string &path) { workspacePath = path; }
	void SetInitLearningRate(const FLOAT val) { initLearningRate = val; }
	void SetMinLearningRate(const FLOAT val) { minLearningRate = val; }
	void SetMomentum(const FLOAT val) { momentum = val; }
	void SetRmsprop(const bool val) { rmsprop = val; }
	void SetAdadelta(const bool val) { adadelta = val; }
	void SetRmsDecayRate(const FLOAT val) { rmsDecayRate = val; }
	void SetMaxRetryCount(const unsigned long val) { maxRetryCount = val; }
	void SetLearningRateDecayRate(const FLOAT val) { learningRateDecayRate = val; }

	void SetLambdaLoss(std::function<double (Evaluator &)> lambda) { lambdaLoss = lambda; }
	void SetLambdaPostEval(std::function<void (Evaluator &)> lambda) {lambdaPostEval = lambda; }

	const std::string &GetWorkspacePath() { return workspacePath; }
	const FLOAT GetInitLearningRate() { return initLearningRate; }
	const FLOAT GetMinLearningRate() { return minLearningRate; }
	const FLOAT GetMomentum() { return momentum; }
	const bool GetRmsprop() { return rmsprop; }
	const bool GetAdadelta() { return adadelta; }
	const FLOAT GetRmsDecayRate() { return rmsDecayRate; }
	const unsigned long GetMaxRetryCount() { return maxRetryCount; }
	const FLOAT GetLearningRateDecayRate() { return learningRateDecayRate; }


protected:
	std::function<double (Evaluator &)> lambdaLoss;
	std::function<void (Evaluator &)> lambdaPostEval;
	
	std::string workspacePath;

	FLOAT initLearningRate;
	FLOAT minLearningRate;
	FLOAT momentum;

	bool rmsprop;
	bool adadelta;
	FLOAT rmsDecayRate;

	FLOAT learningRateDecayRate;

	unsigned long maxRetryCount;
};

}

#endif /* FRACTAL_AUTOOPTIMIZER_H_ */

