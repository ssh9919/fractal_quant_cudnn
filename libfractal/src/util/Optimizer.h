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


#ifndef FRACTAL_OPTIMIZER_H_
#define FRACTAL_OPTIMIZER_H_

#include "Pipe.h"
#include "PortMap.h"
#include "Stream.h"
#include "../core/Rnn.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class BackpropArgs
{
public:
	Rnn *rnn;
	Stream *stream;

	unsigned long numFrame;
	unsigned long nStream;
	unsigned long batchSize;
	unsigned long frameStep;
	unsigned long nInput;
	unsigned long nOutput;

	Probe *inputProbe;
	Probe *outputProbe;

	unsigned long *inputChannel;
	unsigned long *outputChannel;

	Matrix<FLOAT> *input;
	Matrix<FLOAT> *target;
};


class Optimizer
{
public:
	Optimizer();
	virtual ~Optimizer();


	void Backprop(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long numFrame, const unsigned long windowSize, const unsigned long stepSize);

	inline void SetLearningRate(const FLOAT val) { learningRate = val; }
	inline void SetMomentum(const FLOAT val) { momentum = val; }
	inline void SetAdadelta(const bool val) { adadelta = val; }
	inline void SetRmsprop(const bool val) { rmsprop = val; }

	inline const FLOAT GetLearningRate() { return learningRate; }
	inline const FLOAT GetMomentum() { return momentum; }
	inline const bool GetAdadelta() { return adadelta; }
	inline const bool GetRmsprop() { return rmsprop; }

protected:
	static void BackpropPipe0(Optimizer *optimizer, BackpropArgs &args);
	static void BackpropPipe1(Optimizer *optimizer, BackpropArgs &args);
	static void BackpropPipe2(Optimizer *optimizer, BackpropArgs &args);
	static void BackpropPipe3(Optimizer *optimizer, BackpropArgs &args);

	FLOAT learningRate;
	FLOAT momentum;

	Pipe pipe[4];
	PStream pStreamDataTransferToBuf;
	PStream pStreamDataTransferToRnn;
	PEvent pEventDataTransferToBuf;
	PEvent pEventDataTransferToRnn;

	bool adadelta;
	bool rmsprop;
};

}

#endif /* FRACTAL_OPTIMIZER_H_ */

