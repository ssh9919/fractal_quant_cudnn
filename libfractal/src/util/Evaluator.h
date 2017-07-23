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


#ifndef FRACTAL_EVALUATOR_H_
#define FRACTAL_EVALUATOR_H_

#include "Pipe.h"
#include "PortMap.h"
#include "Stream.h"
#include "../core/Rnn.h"
#include "../core/FractalCommon.h"


namespace fractal
{
class Rnn;
class EvaluateArgs
{
public:
	Rnn *rnn;
	Stream *stream;

	unsigned long numFrame;
	unsigned long nStream;
	unsigned long frameStep;
	unsigned long nInput;
	unsigned long nOutput;

	Probe *inputProbe;
	Probe *outputProbe;

	unsigned long *inputChannel;
	unsigned long *outputChannel;

	Matrix<FLOAT> *input;
	Matrix<FLOAT> *output, *outputBuf;
	Matrix<FLOAT> *target;
	Matrix<FLOAT> *targetPipe1, *targetPipe2, *targetPipe3, *targetPipe4, *targetPipe5;
};


class Evaluator
{
public:
	Evaluator() : nOutput(0) {};

	void Evaluate(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long numFrame, const unsigned long stepSize);

	virtual const double GetLoss(const unsigned long outputIdx) const = 0;

	const unsigned long GetNumOutput() { return nOutput; }


protected:
	virtual void Reset() = 0;

	virtual void EvaluateFrames(const unsigned long outputIdx, Matrix<FLOAT> &target,
			Matrix<FLOAT> &output, const unsigned long nStream, PStream &stream) = 0;

	virtual void MemAlloc() = 0;

	void SetNumOutput(const unsigned long nOutput);

	static void EvaluatePipe0(Evaluator *evaluator, EvaluateArgs &args);
	static void EvaluatePipe1(Evaluator *evaluator, EvaluateArgs &args);
	static void EvaluatePipe2(Evaluator *evaluator, EvaluateArgs &args);
	static void EvaluatePipe3(Evaluator *evaluator, EvaluateArgs &args);
	static void EvaluatePipe4(Evaluator *evaluator, EvaluateArgs &args);
	static void EvaluatePipe5(Evaluator *evaluator, EvaluateArgs &args);
	static void EvaluatePipe6(Evaluator *evaluator, EvaluateArgs &args);

	Pipe pipe[7];
	PStream pStreamDataTransferToBuf;
	PStream pStreamDataTransferToRnn;
	PStream pStreamDataTransferFromRnn;
	PStream pStreamDataTransferFromBuf;
	PStream pStreamEvaluateFrames;
	PEvent pEventDataTransferToBuf;
	PEvent pEventDataTransferToRnn;
	PEvent pEventDataTransferFromRnn;
	PEvent pEventDataTransferFromBuf;

	unsigned long nOutput;
};

}

#endif /* FRACTAL_EVALUATOR_H_ */

