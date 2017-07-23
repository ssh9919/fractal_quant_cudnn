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


#include "Optimizer.h"

#include <thread>
#include <vector>


#ifdef FRACTAL_PIPELINE
#define PIPELINE_WIDTH 256
#endif /* FRACTAL_PIPELINE */


#define LOC 1

namespace fractal
{


Optimizer::Optimizer()
{
	learningRate = (FLOAT) 1e-5;
	momentum = (FLOAT) 0.9;

	adadelta = false;

	rmsprop = false;
}


Optimizer::~Optimizer()
{
}

#if 0
void Optimizer::Backprop(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long numFrame, const unsigned long windowSize, const unsigned long stepSize)
{
	verify(numFrame > 0 && stepSize > 0);

	unsigned long batchSize, frameStep, batchFrom, batchTo;
	unsigned long dim, frameIdx, nStream;
	unsigned long i;
	unsigned long nInput, nOutput;
	unsigned long nForwardFrame;

	Engine *engine;

	PortMapList::const_iterator portIter, portIter_end;

	std::vector<Probe> inputProbe(inputPorts.size());
	std::vector<Probe> outputProbe(outputPorts.size());

	nStream = stream.GetNumStream();
	nInput = inputPorts.size();
	nOutput = outputPorts.size();

	//verify(numFrame % nStream == 0);
	verify(numFrame % (nStream * stepSize) == 0);

	std::vector<Matrix<FLOAT>> input(nInput);
	std::vector<Matrix<FLOAT>> target(nOutput);
	std::vector<unsigned long> inputChannel(nInput);
	std::vector<unsigned long> outputChannel(nOutput);

	frameStep = stepSize * nStream;
	batchSize = windowSize * nStream;

	engine = rnn.GetEngine();
	verify(engine != NULL);


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
		target[i].SetEngine(engine);
	}


	/* Initialize the RNN */
	rnn.SetBatchSize(batchSize);
	rnn.InitForward(batchSize - nStream, batchSize - 1);


	/* Main loop */
	for(frameIdx = 0; frameIdx < numFrame; frameIdx += frameStep)
	{
		batchFrom = frameIdx % batchSize;
		batchTo = batchFrom + std::min(numFrame - frameIdx, frameStep) - 1;
		nForwardFrame = batchTo - batchFrom + 1;


		/* Generate sequences from the streams */
		PrepareData(stream, input, target, inputChannel, outputChannel, (batchTo - batchFrom + 1) / nStream);

		rnn.Synchronize();

		/* Copy the sequences to the engine */
		DataTransferToEngine(input, target, inputProbe, outputProbe, batchFrom, batchTo);


		/* Forward pass */
#ifdef FRACTAL_SEQUENTIAL
		for(long i = (long) batchFrom; i <= (long) batchTo; i += nStream)
			rnn.Forward(i, i + nStream - 1, nStream);
#else
//		if(frameIdx + frameStep > batchSize && batchFrom + frameStep < batchSize)
//			rnn.Forward(batchFrom + frameStep, batchSize - 1, nStream);
//		rnn.Forward(0, batchTo, nStream);
#ifdef FRACTAL_PIPELINE
		long pStepSize = (PIPELINE_WIDTH + nStream - 1) / nStream * nStream;

		for(long i = (long) batchFrom; i <= (long) batchTo; i += pStepSize)
			rnn.Forward(i, std::min(i + pStepSize - 1, (long) batchTo), nStream);
#else
		rnn.Forward(batchFrom, batchTo, nStream);
#endif /* FRACTAL_PIPELINE */
#endif /* FRACTAL_SEQUENTIAL */


		/* Compute output errors */
		for(i = 0; i < nOutput; i++)
		{
			Matrix<FLOAT> actSub(outputProbe[i].GetActivation(), batchFrom, batchTo);
			Matrix<FLOAT> errSub(outputProbe[i].GetError(), batchFrom, batchTo);

			outputProbe[i].Wait();

			engine->MatAdd(actSub, errSub, (FLOAT) -1, outputProbe[i].GetPStream());

			outputProbe[i].EventRecord();
		}


		/* Compute derivatives of the activation functions */
		rnn.CalcActDeriv(0, std::min(frameIdx + frameStep, batchSize) - 1);


		/* Backward pass */
		batchTo = batchFrom + frameStep - 1;
		rnn.InitBackward(batchTo - nStream + 1, batchTo);

#ifdef FRACTAL_SEQUENTIAL
		for(long i = (long) batchTo; i >= 0; i -= nStream)
			rnn.Backward(i - nStream + 1, i, nStream);
		for(long i = (long) batchSize - 1; i > (long) batchTo; i -= nStream)
			rnn.Backward(i - nStream + 1, i, nStream);
#else
#ifdef FRACTAL_PIPELINE
		for(long i = (long) batchTo; i >= 0; i -= pStepSize)
			rnn.Backward(std::max(i - pStepSize + 1, (long) 0), i, nStream);
		for(long i = (long) batchSize - 1; i > (long) batchTo; i -= pStepSize)
			rnn.Backward(std::max(i - pStepSize, (long) batchTo) + 1, i, nStream);
#else
		rnn.Backward(0, batchTo, nStream);
		if(frameIdx + frameStep > batchSize && batchFrom + frameStep < batchSize)
			rnn.Backward(batchFrom + frameStep, batchSize - 1, nStream);
#endif /* FRACTAL_PIPELINE */
#endif /* FRACTAL_SEQUENTIAL */


		/* Update weights */
		rnn.UpdateWeights(0, std::min(frameIdx + frameStep, batchSize) - 1, nForwardFrame,
				learningRate, momentum, adadelta, rmsprop);
	}

	rnn.Synchronize();
}


void Optimizer::PrepareData(Stream &stream, std::vector<Matrix<FLOAT>> &input, std::vector<Matrix<FLOAT>> &target,
	const std::vector<unsigned long> &inputChannel, const std::vector<unsigned long> &outputChannel,
	const unsigned long nFramePerStream)
{
	unsigned long streamIdx, i, j;
	unsigned long nStream, nInput, nOutput;
	unsigned long dim;

	nStream = stream.GetNumStream();
	nInput = inputChannel.size();
	nOutput = outputChannel.size();

	for(streamIdx = 0; streamIdx < nStream; streamIdx++)
	{
		for(i = 0; i < nFramePerStream; i++)
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
}


void Optimizer::DataTransferToEngine(std::vector<Matrix<FLOAT>> &input, std::vector<Matrix<FLOAT>> &target,
		std::vector<Probe> &inputProbe, std::vector<Probe> &outputProbe,
		const unsigned long batchFrom, const unsigned long batchTo)
{
	unsigned long i;
	unsigned long nInput, nOutput;

	nInput = inputProbe.size();
	nOutput = outputProbe.size();

	for(i = 0; i < nInput; i++)
	{
		Matrix<FLOAT> stateSub(inputProbe[i].GetState(), batchFrom, batchTo);
		Matrix<FLOAT> inputSub(input[i], 0, batchTo - batchFrom);

		stateSub.Import(inputSub, inputProbe[i].GetPStream());

		inputProbe[i].EventRecord();
	}

	for(i = 0; i < nOutput; i++)
	{
		Matrix<FLOAT> errSub(outputProbe[i].GetError(), batchFrom, batchTo);
		Matrix<FLOAT> targetSub(target[i], 0, batchTo - batchFrom);

		outputProbe[i].GetEngine()->MatSet(outputProbe[i].GetError(), (FLOAT) 0, outputProbe[i].GetPStream());

		errSub.Import(targetSub, outputProbe[i].GetPStream());

		outputProbe[i].EventRecord();
	}
}


#else

void Optimizer::Backprop(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long numFrame, const unsigned long windowSize, const unsigned long stepSize)
{
	verify(numFrame > 0 && stepSize > 0);

	BackpropArgs args;
	Engine *engine;
	unsigned long i;

	PortMapList::const_iterator portIter, portIter_end;

	args.nInput = inputPorts.size();
	args.nOutput = outputPorts.size();

	std::vector<Probe> inputProbe(args.nInput);
	std::vector<Probe> outputProbe(args.nOutput);

	std::vector<Matrix<FLOAT>> input(args.nInput);
	std::vector<Matrix<FLOAT>> target(args.nOutput);
	std::vector<unsigned long> inputChannel(args.nInput);
	std::vector<unsigned long> outputChannel(args.nOutput);

	engine = rnn.GetEngine();
	verify(engine != NULL);

	args.rnn = &rnn;
	args.stream = &stream;
	args.numFrame = numFrame;
	args.nStream = stream.GetNumStream();
	args.batchSize = windowSize * args.nStream;
	args.frameStep = stepSize * args.nStream;
	args.inputProbe = inputProbe.data();
	args.outputProbe = outputProbe.data();
	args.inputChannel = inputChannel.data();
	args.outputChannel = outputChannel.data();
	args.input = input.data();
	args.target = target.data();

	//verify(numFrame % nStream == 0);
	verify(args.numFrame % (args.nStream * stepSize) == 0);

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
	}


	/* Initialize the RNN */
	rnn.SetBatchSize(args.batchSize);
	//rnn.InitForward(args.batchSize - args.nStream, args.batchSize - 1);
	rnn.InitForward(0, args.batchSize - 1);
        rnn.EnableDropout(true);

	/* Main loop */
	engine->StreamCreate(pStreamDataTransferToBuf, LOC);
	engine->StreamCreate(pStreamDataTransferToRnn, LOC);
	engine->EventCreate(pEventDataTransferToBuf, LOC);
	engine->EventCreate(pEventDataTransferToRnn, LOC);

	pipe[0].Init();
	pipe[1].Init();
	pipe[2].Init();
	pipe[3].Init();

	std::thread thdPipe0(BackpropPipe0, this, std::ref(args));
	std::thread thdPipe1(BackpropPipe1, this, std::ref(args));
	std::thread thdPipe2(BackpropPipe2, this, std::ref(args));
	std::thread thdPipe3(BackpropPipe3, this, std::ref(args));

	pipe[0].SendSignal();
	pipe[1].SendSignal();
	pipe[2].SendSignal();

	thdPipe0.join();
	thdPipe1.join();
	thdPipe2.join();
	thdPipe3.join();

	engine->StreamDestroy(pStreamDataTransferToBuf);
	engine->StreamDestroy(pStreamDataTransferToRnn);
	engine->EventDestroy(pEventDataTransferToBuf);
	engine->EventDestroy(pEventDataTransferToRnn);

        rnn.EnableDropout(false);
	rnn.Synchronize();
}


void Optimizer::BackpropPipe0(Optimizer *optimizer, BackpropArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = frameIdx % args.batchSize;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;
		unsigned long nForwardFrame = batchTo - batchFrom + 1;

		optimizer->pipe[0].Wait(1);


		/* Wait until the memory transfer to the engine finishes */

		engine->StreamSynchronize(optimizer->pStreamDataTransferToBuf);


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

		optimizer->pipe[1].SendSignal();
	}
}


void Optimizer::BackpropPipe1(Optimizer *optimizer, BackpropArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		optimizer->pipe[1].Wait(2);

		engine->StreamWaitEvent(optimizer->pStreamDataTransferToBuf, optimizer->pEventDataTransferToRnn);

		/* Copy the sequences to the buffers */

		for(unsigned long i = 0; i < args.nInput; i++)
		{
			args.input[i].HostPush();
			engine->MemPull(args.input[i].GetMem(), LOC, optimizer->pStreamDataTransferToBuf);
			//args.input[i].Pull(LOC, optimizer->pStreamDataTransferToBuf);
		}

		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			args.target[i].HostPush();
			engine->MemPull(args.target[i].GetMem(), LOC, optimizer->pStreamDataTransferToBuf);
			//args.target[i].Pull(LOC, optimizer->pStreamDataTransferToBuf);
		}

		engine->EventRecord(optimizer->pEventDataTransferToBuf, optimizer->pStreamDataTransferToBuf);

		optimizer->pipe[0].SendSignal();
		optimizer->pipe[2].SendSignal();
	}
}


void Optimizer::BackpropPipe2(Optimizer *optimizer, BackpropArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		unsigned long batchFrom = frameIdx % args.batchSize;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

		optimizer->pipe[2].Wait(2);

		//args.rnn->Synchronize();
		engine->StreamWaitEvent(optimizer->pStreamDataTransferToRnn, optimizer->pEventDataTransferToBuf);
		args.rnn->StreamWait(optimizer->pStreamDataTransferToRnn);


		/* Copy the sequences to the RNN */

		for(unsigned long i = 0; i < args.nInput; i++)
		{
			Matrix<FLOAT> stateSub(args.inputProbe[i].GetState(), batchFrom, batchTo);
			Matrix<FLOAT> inputSub(args.input[i], 0, batchTo - batchFrom);

			engine->MatCopy(inputSub, stateSub, optimizer->pStreamDataTransferToRnn);

			engine->EventRecord(optimizer->pEventDataTransferToRnn, optimizer->pStreamDataTransferToRnn);
			engine->StreamWaitEvent(args.inputProbe[i].GetPStream(), optimizer->pEventDataTransferToRnn);

			args.inputProbe[i].EventRecord();
		}

		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			Matrix<FLOAT> errSub(args.outputProbe[i].GetError(), batchFrom, batchTo);
			Matrix<FLOAT> targetSub(args.target[i], 0, batchTo - batchFrom);

			engine->MatSet(args.outputProbe[i].GetError(), (FLOAT) 0, optimizer->pStreamDataTransferToRnn);
			engine->MatCopy(targetSub, errSub, optimizer->pStreamDataTransferToRnn);

			engine->EventRecord(optimizer->pEventDataTransferToRnn, optimizer->pStreamDataTransferToRnn);
			engine->StreamWaitEvent(args.outputProbe[i].GetPStream(), optimizer->pEventDataTransferToRnn);

			args.outputProbe[i].EventRecord();
		}

		optimizer->pipe[1].SendSignal();
		optimizer->pipe[3].SendSignal();
	}
}


void Optimizer::BackpropPipe3(Optimizer *optimizer, BackpropArgs &args)
{
	Engine *engine = args.rnn->GetEngine();

	for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
	{
		optimizer->pipe[3].Wait(1);

		unsigned long batchFrom = frameIdx % args.batchSize;
		unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;
		unsigned long nForwardFrame = batchTo - batchFrom + 1;

                /* Generate dropout mask */

                args.rnn->GenerateDropoutMask(batchFrom, batchTo);

		/* Forward pass */
#ifdef FRACTAL_SEQUENTIAL
		for(long i = (long) batchFrom; i <= (long) batchTo; i += args.nStream)
			args.rnn->Forward(i, i + args.nStream - 1, args.nStream);
#else
//		if(frameIdx + args.frameStep > args.batchSize && batchFrom + args.frameStep < args.batchSize)
//			rnn.Forward(batchFrom + args.frameStep, batchSize - 1, args.nStream);
//		rnn.Forward(0, batchTo, nStream);
#ifdef FRACTAL_PIPELINE
		long pStepSize = (PIPELINE_WIDTH + args.nStream - 1) / args.nStream * args.nStream;

		for(long i = (long) batchFrom; i <= (long) batchTo; i += pStepSize)
			args.rnn->Forward(i, std::min(i + pStepSize - 1, (long) batchTo), args.nStream);
#else
		args.rnn->Forward(batchFrom, batchTo, args.nStream);
#endif /* FRACTAL_PIPELINE */
#endif /* FRACTAL_SEQUENTIAL */


		/* Compute output errors */
		for(unsigned long i = 0; i < args.nOutput; i++)
		{
			Matrix<FLOAT> actSub(args.outputProbe[i].GetActivation(), batchFrom, batchTo);
			Matrix<FLOAT> errSub(args.outputProbe[i].GetError(), batchFrom, batchTo);

			args.outputProbe[i].Wait();

			/* errSub initially stores target data */

			/* err = err(target) - act */
			engine->MatAdd(actSub, errSub, (FLOAT) -1, args.outputProbe[i].GetPStream());

			args.outputProbe[i].EventRecord();
		}


		/* Compute derivatives of the activation functions */
		args.rnn->CalcActDeriv(0, std::min(frameIdx + args.frameStep, args.batchSize) - 1);


		/* Backward pass */
		batchTo = batchFrom + args.frameStep - 1;
		//args.rnn->InitBackward(batchTo - args.nStream + 1, batchTo);
		args.rnn->InitBackward(0, args.batchSize - 1);

#ifdef FRACTAL_SEQUENTIAL
		for(long i = (long) batchTo; i >= 0; i -= args.nStream)
			args.rnn->Backward(i - args.nStream + 1, i, args.nStream);
		for(long i = (long) args.batchSize - 1; i > (long) batchTo; i -= args.nStream)
			args.rnn->Backward(i - args.nStream + 1, i, args.nStream);
#else
#ifdef FRACTAL_PIPELINE
		for(long i = (long) batchTo; i >= 0; i -= pStepSize)
			args.rnn->Backward(std::max(i - pStepSize + 1, (long) 0), i, args.nStream);
		for(long i = (long) args.batchSize - 1; i > (long) batchTo; i -= pStepSize)
			args.rnn->Backward(std::max(i - pStepSize, (long) batchTo) + 1, i, args.nStream);
#else
		args.rnn->Backward(0, batchTo, args.nStream);
		if(frameIdx + args.frameStep > args.batchSize && batchFrom + args.frameStep < args.batchSize)
			args.rnn->Backward(batchFrom + args.frameStep, args.batchSize - 1, args.nStream);
#endif /* FRACTAL_PIPELINE */
#endif /* FRACTAL_SEQUENTIAL */


		/* Update weights */
		args.rnn->UpdateWeights(0, std::min(frameIdx + args.frameStep, args.batchSize) - 1, nForwardFrame,
				optimizer->learningRate, optimizer->momentum, optimizer->adadelta, optimizer->rmsprop);

		optimizer->pipe[2].SendSignal();
	}
}
#endif


}

