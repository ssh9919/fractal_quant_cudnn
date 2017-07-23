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


#include "Evaluator.h"

#include <thread>


#ifdef FRACTAL_PIPELINE
#define PIPELINE_WIDTH 256
#endif /* FRACTAL_PIPELINE */

#define LOC 1


namespace fractal
{

void Evaluator::SetNumOutput(const unsigned long nOutput)
{
	if(this->nOutput == nOutput) return;

	this->nOutput = nOutput;
	MemAlloc();
}


void Evaluator::Evaluate(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long numFrame, const unsigned long stepSize)
{
	verify(numFrame > 0 && stepSize > 0);

	EvaluateArgs args;
	Engine *engine;
	unsigned long i;

	PortMapList::const_iterator portIter, portIter_end;

	args.nInput = inputPorts.size();
	args.nOutput = outputPorts.size();

	std::vector<Probe> inputProbe(args.nInput);
	std::vector<Probe> outputProbe(args.nOutput);

	std::vector<Matrix<FLOAT>> input(args.nInput);
	std::vector<Matrix<FLOAT>> output(args.nOutput);
	std::vector<Matrix<FLOAT>> outputBuf(args.nOutput);

	std::vector<Matrix<FLOAT>> target(args.nOutput);
	std::vector<Matrix<FLOAT>> targetPipe1(args.nOutput);
	std::vector<Matrix<FLOAT>> targetPipe2(args.nOutput);
	std::vector<Matrix<FLOAT>> targetPipe3(args.nOutput);
	std::vector<Matrix<FLOAT>> targetPipe4(args.nOutput);
	std::vector<Matrix<FLOAT>> targetPipe5(args.nOutput);

	std::vector<unsigned long> inputChannel(args.nInput);
	std::vector<unsigned long> outputChannel(args.nOutput);

	engine = rnn.GetEngine();
	verify(engine != NULL);

	args.rnn = &rnn;
	args.stream = &stream;
	args.numFrame = numFrame;
	args.nStream = stream.GetNumStream();
	args.frameStep = stepSize * args.nStream;
	args.inputProbe = inputProbe.data();
	args.outputProbe = outputProbe.data();
	args.inputChannel = inputChannel.data();
	args.outputChannel = outputChannel.data();
	args.input = input.data();
	args.output = output.data();
	args.outputBuf = outputBuf.data();
	args.target = target.data();
	args.targetPipe1 = targetPipe1.data();
	args.targetPipe2 = targetPipe2.data();
	args.targetPipe3 = targetPipe3.data();
	args.targetPipe4 = targetPipe4.data();
	args.targetPipe5 = targetPipe5.data();

	verify(args.numFrame % args.nStream == 0);
	//verify(args.numFrame % (args.nStream * stepSize) == 0);


	/* Init evaluator */
	Reset();
	SetNumOutput(args.nOutput);


	/* Link probes and resize sequence buffers */
	portIter_end = inputPorts.end();
	for(portIter = inputPorts.begin(), i = 0; portIter != portIter_end; portIter++, i++)
	{
		inputProbe[i].SetInput(true);
		inputProbe[i].SetEngine(engine);

		rnn.LinkProbe(inputProbe[i], std::get<0>(*portIter));

		inputChannel[i] = std::get<1>(*portIter);

		unsigned long dim = stream.GetDimension(inputChannel[i]);
		verify(inputProbe[i].GetLayerSize() == dim);

		input[i].Resize(dim, args.frameStep);
		input[i].SetEngine(engine);
	}

	portIter_end = outputPorts.end();
	for(portIter = outputPorts.begin(), i = 0; portIter != portIter_end; portIter++, i++)
	{
		outputProbe[i].SetOutput(true);
		outputProbe[i].SetEngine(engine);

		rnn.LinkProbe(outputProbe[i], std::get<0>(*portIter));

		outputChannel[i] = std::get<1>(*portIter);

		unsigned long dim = stream.GetDimension(outputChannel[i]);
		verify(outputProbe[i].GetLayerSize() == dim);

		target[i].Resize(dim, args.frameStep);
		target[i].SetEngine(engine);
		targetPipe1[i].Resize(dim, args.frameStep);
		targetPipe1[i].SetEngine(engine);
		targetPipe2[i].Resize(dim, args.frameStep);
		targetPipe2[i].SetEngine(engine);
		targetPipe3[i].Resize(dim, args.frameStep);
		targetPipe3[i].SetEngine(engine);
		targetPipe4[i].Resize(dim, args.frameStep);
		targetPipe4[i].SetEngine(engine);
		targetPipe5[i].Resize(dim, args.frameStep);
		targetPipe5[i].SetEngine(engine);

		output[i].Resize(dim, args.frameStep);
		output[i].SetEngine(engine);
		outputBuf[i].Resize(dim, args.frameStep);
		outputBuf[i].SetEngine(engine);
	}


	/* Initialize the RNN */
	rnn.SetBatchSize(args.frameStep);
	//rnn.InitForward(args.frameStep - args.nStream, args.frameStep - 1);
	rnn.InitForward(0, args.frameStep - 1);


	/* Main loop */
	engine->StreamCreate(pStreamDataTransferToBuf, LOC);
	engine->StreamCreate(pStreamDataTransferToRnn, LOC);
	engine->StreamCreate(pStreamDataTransferFromRnn, LOC);
	engine->StreamCreate(pStreamDataTransferFromBuf, LOC);
	engine->StreamCreate(pStreamEvaluateFrames, LOC);
	engine->EventCreate(pEventDataTransferToBuf, LOC);
	engine->EventCreate(pEventDataTransferToRnn, LOC);
	engine->EventCreate(pEventDataTransferFromRnn, LOC);
	engine->EventCreate(pEventDataTransferFromBuf, LOC);

	pipe[0].Init();
	pipe[1].Init();
	pipe[2].Init();
	pipe[3].Init();
	pipe[4].Init();
	pipe[5].Init();
	pipe[6].Init();

	std::thread thdPipe0(EvaluatePipe0, this, std::ref(args));
	std::thread thdPipe1(EvaluatePipe1, this, std::ref(args));
	std::thread thdPipe2(EvaluatePipe2, this, std::ref(args));
	std::thread thdPipe3(EvaluatePipe3, this, std::ref(args));
	std::thread thdPipe4(EvaluatePipe4, this, std::ref(args));
	std::thread thdPipe5(EvaluatePipe5, this, std::ref(args));
	std::thread thdPipe6(EvaluatePipe6, this, std::ref(args));

	pipe[0].SendSignal();
	pipe[1].SendSignal();
	pipe[2].SendSignal();
	pipe[4].SendSignal();
	pipe[5].SendSignal();

	thdPipe0.join();
	thdPipe1.join();
	thdPipe2.join();
	thdPipe3.join();
	thdPipe4.join();
	thdPipe5.join();
	thdPipe6.join();

	engine->StreamDestroy(pStreamDataTransferToBuf);
	engine->StreamDestroy(pStreamDataTransferToRnn);
	engine->StreamDestroy(pStreamDataTransferFromRnn);
	engine->StreamDestroy(pStreamDataTransferFromBuf);
	engine->StreamDestroy(pStreamEvaluateFrames);
	engine->EventDestroy(pEventDataTransferToBuf);
	engine->EventDestroy(pEventDataTransferToRnn);
	engine->EventDestroy(pEventDataTransferFromRnn);
	engine->EventDestroy(pEventDataTransferFromBuf);

	rnn.Synchronize();
}


void Evaluator::EvaluatePipe0(Evaluator *evaluator, EvaluateArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = 0;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;
		unsigned long nForwardFrame = batchTo - batchFrom + 1;

		evaluator->pipe[0].Wait(1);


		/* Wait until the memory transfer to the engine finishes */

		engine->StreamSynchronize(evaluator->pStreamDataTransferToBuf);


		/* Generate sequences from the streams */

		for(unsigned long streamIdx = 0; streamIdx < args.nStream; streamIdx++)
		{
			for(unsigned long i = 0; i < nForwardFrame / args.nStream; i++)
			{
				for(unsigned long j = 0; j < args.nInput; j++)
				{
					unsigned long dim = args.stream->GetDimension(args.inputChannel[j]);
					args.stream->GenerateFrame(streamIdx, args.inputChannel[j],
							args.input[j].GetHostData() + (i * args.nStream + streamIdx) * dim);
				}

				for(unsigned long j = 0; j < args.nOutput; j++)
				{
					unsigned long dim = args.stream->GetDimension(args.outputChannel[j]);
					args.stream->GenerateFrame(streamIdx, args.outputChannel[j],
							args.target[j].GetHostData() + (i * args.nStream + streamIdx) * dim);
				}

				args.stream->Next(streamIdx);
			}
		}

		evaluator->pipe[1].SendSignal();
	}
}


void Evaluator::EvaluatePipe1(Evaluator *evaluator, EvaluateArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		evaluator->pipe[1].Wait(2);

		engine->StreamWaitEvent(evaluator->pStreamDataTransferToBuf, evaluator->pEventDataTransferToRnn);

		/* Copy the input sequences to the buffers */
		for(unsigned long i = 0; i < args.nInput; i++)
		{
			args.input[i].HostPush();
			engine->MemPull(args.input[i].GetMem(), LOC, evaluator->pStreamDataTransferToBuf);
			//args.input[i].Pull(LOC, evaluator->pStreamDataTransferToBuf);
		}

		/* Propagate the target sequences */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.targetPipe1[i].Swap(args.target[i]);
		}

		engine->EventRecord(evaluator->pEventDataTransferToBuf, evaluator->pStreamDataTransferToBuf);

		evaluator->pipe[0].SendSignal();
		evaluator->pipe[2].SendSignal();
	}
}


void Evaluator::EvaluatePipe2(Evaluator *evaluator, EvaluateArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = 0;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

		evaluator->pipe[2].Wait(2);

		engine->StreamWaitEvent(evaluator->pStreamDataTransferToRnn, evaluator->pEventDataTransferToBuf);
		engine->StreamWaitEvent(evaluator->pStreamDataTransferToRnn, evaluator->pEventDataTransferFromRnn);

		/* Copy the input sequences to the RNN */
		for(unsigned long i = 0; i < args.nInput; i++)
		{
			Matrix<FLOAT> stateSub(args.inputProbe[i].GetState(), batchFrom, batchTo);
			Matrix<FLOAT> inputSub(args.input[i], 0, batchTo - batchFrom);

			engine->MatCopy(inputSub, stateSub, evaluator->pStreamDataTransferToRnn);

			engine->EventRecord(evaluator->pEventDataTransferToRnn, evaluator->pStreamDataTransferToRnn);
			engine->StreamWaitEvent(args.inputProbe[i].GetPStream(), evaluator->pEventDataTransferToRnn);

			args.inputProbe[i].EventRecord();
		}

		/* Propagate the target sequences */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.targetPipe2[i].Swap(args.targetPipe1[i]);
		}

		evaluator->pipe[1].SendSignal();
		evaluator->pipe[3].SendSignal();
	}
}


void Evaluator::EvaluatePipe3(Evaluator *evaluator, EvaluateArgs &args)
{
	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = 0;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

		evaluator->pipe[3].Wait(1);

                /* Forward pass */
#ifdef FRACTAL_PIPELINE
		long pStepSize = (PIPELINE_WIDTH + args.nStream - 1) / args.nStream * args.nStream;

		for(long i = (long) batchFrom; i <= (long) batchTo; i += pStepSize)
			args.rnn->Forward(i, std::min(i + pStepSize - 1, (long) batchTo), args.nStream);
#else
		args.rnn->Forward(batchFrom, batchTo, args.nStream);
#endif /* FRACTAL_PIPELINE */


		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.outputProbe[i].Wait();
			args.outputProbe[i].EventRecord();
		}

		/* Propagate the target sequences */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.targetPipe3[i].Swap(args.targetPipe2[i]);
		}

		evaluator->pipe[4].SendSignal();
	}
}


void Evaluator::EvaluatePipe4(Evaluator *evaluator, EvaluateArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = 0;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

		evaluator->pipe[4].Wait(2);

		engine->StreamWaitEvent(evaluator->pStreamDataTransferFromRnn, evaluator->pEventDataTransferFromBuf);

		/* Copy the output from the RNN */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			Matrix<FLOAT> actSub(args.outputProbe[i].GetActivation(), batchFrom, batchTo);
			Matrix<FLOAT> outputBufSub(args.outputBuf[i], 0, batchTo - batchFrom);

			args.outputProbe[i].StreamWaitEvent(evaluator->pStreamDataTransferFromRnn);
			engine->MatCopy(actSub, outputBufSub, evaluator->pStreamDataTransferFromRnn);
		}

		/* Propagate the target sequences */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.targetPipe4[i].Swap(args.targetPipe3[i]);
		}

		engine->EventRecord(evaluator->pEventDataTransferFromRnn, evaluator->pStreamDataTransferFromRnn);

		evaluator->pipe[2].SendSignal();
		evaluator->pipe[5].SendSignal();
	}
}


void Evaluator::EvaluatePipe5(Evaluator *evaluator, EvaluateArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		evaluator->pipe[5].Wait(2);

		engine->StreamWaitEvent(evaluator->pStreamDataTransferFromBuf, evaluator->pEventDataTransferFromRnn);

		/* Copy the output from the buffer */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.output[i].Swap(args.outputBuf[i]);
			args.output[i].HostPull(evaluator->pStreamDataTransferFromBuf);
			//args.outputBuf[i].Export(args.output[i], evaluator->pStreamDataTransferFromBuf);
		}

		/* Propagate the target sequences */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.targetPipe5[i].Swap(args.targetPipe4[i]);
		}

		engine->EventRecord(evaluator->pEventDataTransferFromBuf, evaluator->pStreamDataTransferFromBuf);

		evaluator->pipe[4].SendSignal();
		evaluator->pipe[6].SendSignal();
	}
}


void Evaluator::EvaluatePipe6(Evaluator *evaluator, EvaluateArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = 0;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

		evaluator->pipe[6].Wait(1);

		engine->StreamSynchronize(evaluator->pStreamDataTransferFromBuf);
		//args.rnn->Synchronize();

		/* Evaluate frames */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			Matrix<FLOAT> outputSub(args.output[i], 0, batchTo - batchFrom);
			Matrix<FLOAT> targetSub(args.targetPipe5[i], 0, batchTo - batchFrom);

			evaluator->EvaluateFrames(i, targetSub, outputSub, args.nStream, evaluator->pStreamEvaluateFrames);
		}

		evaluator->pipe[5].SendSignal();
	}
}


#if 0
void Evaluator::Evaluate(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long numFrame, const unsigned long stepSize)
{
	verify(numFrame > 0 && stepSize > 0);

	unsigned long frameStep, batchFrom, batchTo;
	unsigned long dim, streamIdx, nStream;
	unsigned long i, j;
	unsigned long nInput, nOutput;
	unsigned long prevBatchFrom, prevBatchTo;
	unsigned long nextBatchFrom, nextBatchTo;
	long frameIdx;

	Engine *engine;

	PortMapList::const_iterator portIter, portIter_end;

	std::vector<Probe> inputProbe(inputPorts.size());
	std::vector<Probe> outputProbe(outputPorts.size());

	nStream = stream.GetNumStream();
	nInput = inputPorts.size();
	nOutput = outputPorts.size();

	verify(numFrame % nStream == 0);

	std::vector<Matrix<FLOAT>> input(nInput);
	std::vector<Matrix<FLOAT>> target(nOutput);
	std::vector<Matrix<FLOAT>> target2(nOutput);
	std::vector<Matrix<FLOAT>> output(nOutput);
	std::vector<unsigned long> inputChannel(nInput);
	std::vector<unsigned long> outputChannel(nOutput);

	frameStep = stepSize * nStream;

	engine = rnn.GetEngine();
	verify(engine != NULL);

	/* Init evaluator */
	Reset();
	SetNumOutput(nOutput);


	/* Link probes and resize sequence buffers */
	portIter_end = inputPorts.end();
	for(portIter = inputPorts.begin(), i = 0; portIter != portIter_end; portIter++, i++)
	{
		inputProbe[i].SetInput(true);
		inputProbe[i].SetEngine(engine);

		rnn.LinkProbe(inputProbe[i], std::get<0>(*portIter));

		inputChannel[i] = std::get<1>(*portIter);

		dim = stream.GetDimension(inputChannel[i]);
		verify(inputProbe[i].GetLayerSize() == dim);

		input[i].Resize(dim, frameStep);
		input[i].SetEngine(engine);
	}

	portIter_end = outputPorts.end();
	for(portIter = outputPorts.begin(), i = 0; portIter != portIter_end; portIter++, i++)
	{
		outputProbe[i].SetOutput(true);
		outputProbe[i].SetEngine(engine);

		rnn.LinkProbe(outputProbe[i], std::get<0>(*portIter));

		outputChannel[i] = std::get<1>(*portIter);

		dim = stream.GetDimension(outputChannel[i]);
		verify(outputProbe[i].GetLayerSize() == dim);

		target[i].Resize(dim, frameStep);
		target2[i].Resize(dim, frameStep);
		output[i].Resize(dim, frameStep);
		target[i].SetEngine(engine);
		target2[i].SetEngine(engine);
		output[i].SetEngine(engine);
	}


	/* Initialize the RNN */
	rnn.SetBatchSize(frameStep);
	rnn.InitForward(frameStep - nStream, frameStep - 1);


	prevBatchFrom = prevBatchTo = 0;
	batchFrom = batchTo = 0;
	nextBatchFrom = nextBatchTo = 0;


	/* Main loop */
	for(frameIdx = -frameStep; frameIdx < (long)(numFrame + frameStep); frameIdx += frameStep)
	{
		prevBatchFrom = batchFrom;
		prevBatchTo = batchTo;

		batchFrom = nextBatchFrom;
		batchTo = nextBatchTo;


		/* Swap the target buffers */
		for(i = 0; i < nOutput; i++)
		{
			target[i].Swap(target2[i]);
		}


		if(frameIdx > 0)
		{
			/* Export outputs */
			for(i = 0; i < nOutput; i++)
			{
				Matrix<FLOAT> actSub(outputProbe[i].GetActivation(), prevBatchFrom, prevBatchTo);
				Matrix<FLOAT> outputSub(output[i], 0, prevBatchTo - prevBatchFrom);

				outputProbe[i].Wait();
				actSub.Export(outputSub, outputProbe[i].GetPStream());
				outputProbe[i].GetEngine()->StreamSynchronize(outputProbe[i].GetPStream());
			}
		}


		if(frameIdx >= 0 && frameIdx < (long) numFrame)
		{
			/* Forward pass */
			rnn.Forward(batchFrom, batchTo, nStream);
		}


		if(frameIdx > 0)
		{
			/* Evaluate frames */
			for(i = 0; i < nOutput; i++)
			{
				Matrix<FLOAT> outputSub(output[i], 0, prevBatchTo - prevBatchFrom);
				Matrix<FLOAT> targetSub(target[i], 0, prevBatchTo - prevBatchFrom);

				EvaluateFrames(i, targetSub, outputSub, nStream, outputProbe[i].GetPStream());
			}
		}


		if(frameIdx + (long) frameStep < (long) numFrame)
		{
			nextBatchFrom = 0;
			nextBatchTo = std::min(numFrame - (frameIdx + frameStep), frameStep) - 1;


			/* Generate next sequences from the streams */
			for(streamIdx = 0; streamIdx < nStream; streamIdx++)
			{
				for(i = 0; i < (nextBatchTo - nextBatchFrom + 1) / nStream; i++)
				{
					for(j = 0; j < nInput; j++)
					{
						dim = stream.GetDimension(inputChannel[j]);
						stream.GenerateFrame(streamIdx, inputChannel[j],
								input[j].GetHostData() + (i * nStream + streamIdx) * dim);
					}

					for(j = 0; j < nOutput; j++)
					{
						dim = stream.GetDimension(outputChannel[j]);
						stream.GenerateFrame(streamIdx, outputChannel[j],
								target[j].GetHostData() + (i * nStream + streamIdx) * dim);
					}

					stream.Next(streamIdx);
				}
			}

			for(i = 0; i < nInput; i++)
			{
				input[i].HostPush();
			}
			for(i = 0; i < nOutput; i++)
			{
				target[i].HostPush();
			}


			/* Copy the next sequences to the engine */
			for(i = 0; i < nInput; i++)
			{
				Matrix<FLOAT> stateSub(inputProbe[i].GetState(), nextBatchFrom, nextBatchTo);
				Matrix<FLOAT> inputSub(input[i], 0, nextBatchTo - nextBatchFrom);

				stateSub.Import(inputSub, inputProbe[i].GetPStream());

				inputProbe[i].EventRecord();
			}
		}
	}

	rnn.Synchronize();
}
#endif

}

