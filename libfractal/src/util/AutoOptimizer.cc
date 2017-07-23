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


#include "AutoOptimizer.h"

#include <chrono>
#include <iostream>
#include <cmath>


namespace fractal
{

AutoOptimizer::AutoOptimizer()
{
	workspacePath = "workspace";
	initLearningRate = 1e-5;
	minLearningRate = 1e-7;
	momentum = 0.9;

	adadelta = false;
	rmsprop = false;
	rmsDecayRate = 0.95;

	maxRetryCount = 4;

	learningRateDecayRate = 0.5;

	lambdaLoss = [] (Evaluator &evaluator) -> double
	{
		double loss = 0.0;

		for(unsigned long i = 0; i < evaluator.GetNumOutput(); i++)
		{
			loss += evaluator.GetLoss(i);
		}

		return loss;
	};

	lambdaPostEval = [] (Evaluator &evaluator) -> void
	{
		for(unsigned long i = 0; i < evaluator.GetNumOutput(); i++)
		{
			printf("Loss (%ld): %f\n", i, evaluator.GetLoss(i));
		}
	};
}


void AutoOptimizer::Optimize(Rnn &rnn, Stream &trainStream, Stream &evalStream, Evaluator &evaluator,
		const PortMapList &inputPorts, const PortMapList &outputPorts,
		const unsigned long nTrainFramePerEpoch, const unsigned long nEvalFramePerEpoch,
		const unsigned long windowSize, const unsigned long stepSize)
{
	std::string pivotPath = workspacePath + "/net/0/";
	std::string prevPath = workspacePath + "/net/1/";
	std::string bestPath = workspacePath + "/net/best/";

	FLOAT learningRate;
	unsigned long evalStepSize;
	unsigned long retryCount;
	unsigned long totalTrainedFrame, totalTrainedFrameAtPivot, totalTrainedFrameAtBest, totalDiscardedFrame;
	double prevLoss, pivotLoss, bestLoss;

	Optimizer optimizer;

	optimizer.SetLearningRate(initLearningRate);
	optimizer.SetMomentum(momentum);
	optimizer.SetRmsprop(rmsprop);
	optimizer.SetAdadelta(adadelta);

	rnn.InitNesterov();

	if(adadelta == true)
	{
		rnn.InitAdadelta(rmsDecayRate);

		verify(rmsprop == false);
	}
	else
	{
		if(rmsprop == true)
			rnn.InitRmsprop(rmsDecayRate);
	}


	rnn.Ready();

	evalStepSize = windowSize * trainStream.GetNumStream() / evalStream.GetNumStream();
	learningRate = initLearningRate;
	totalTrainedFrame = 0;
	totalTrainedFrameAtPivot = 0;
	totalDiscardedFrame = 0;
	bestLoss = 0.0;
	pivotLoss = 0.0;
	prevLoss = 0.0;
	retryCount = 0;
	rnn.SaveState(prevPath);
	rnn.SaveState(bestPath);

	std::cout << "======================================================================" << std::endl;
	std::cout << "                            Auto Optimizer                            " << std::endl;
	std::cout << "======================================================================" << std::endl;
	std::cout << std::endl;
	std::cout << "-------------------------- General Settings --------------------------" << std::endl;
	std::cout << "Workspace path: " << workspacePath << std::endl;
	std::cout << "Maximum retry count: " << maxRetryCount << std::endl;
	std::cout << "Decay rate of the learning rate: " << learningRateDecayRate << std::endl;
	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;
	std::cout << "------------------------- Training Parameters ------------------------" << std::endl;
	std::cout << "Initial learning rate: " << initLearningRate << std::endl;
	std::cout << "Minimum learning rate: " << minLearningRate << std::endl;
	std::cout << "Momentum: " << momentum << std::endl;

	std::cout << "AdaDelta: " << (adadelta == true ? "enabled" : "disabled") << std::endl;
	std::cout << "RMSProp: " << (rmsprop == true ? "enabled" : "disabled") << std::endl;
	if(rmsprop == true || adadelta == true) std::cout << "RMS decay rate: " << rmsDecayRate << std::endl;

	std::cout << "Number of frames per epoch: " << nTrainFramePerEpoch << std::endl;
	std::cout << "Number of streams: " << trainStream.GetNumStream() << std::endl;
	std::cout << "Forward step size: " << stepSize << std::endl;
	std::cout << "Backward window size: " << windowSize << std::endl;
	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;
	std::cout << "------------------------ Evaluation Parameters -----------------------" << std::endl;
	std::cout << "Number of frames per epoch: " << nEvalFramePerEpoch << std::endl;
	std::cout << "Number of streams: " << evalStream.GetNumStream() << std::endl;
	std::cout << "Forward step size: " << evalStepSize << std::endl;
	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	while(true)
	{
		std::cout << "----------------------------------------------------------------------" << std::endl;

		/* Training */
		std::cout << "Training ...  ";
		std::cout.flush();
		{
			trainStream.Reset();

			auto t1 = std::chrono::steady_clock::now();
			optimizer.Backprop(rnn, trainStream, inputPorts, outputPorts, nTrainFramePerEpoch, windowSize, stepSize);
			auto t2 = std::chrono::steady_clock::now();

			std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

			std::cout << "(" << time_span.count() << " sec)" << std::endl;
		}

		totalTrainedFrame += nTrainFramePerEpoch;


		/* Evaluation */
		std::cout << "Evaluating ...  ";
		std::cout.flush();
		{
			evalStream.Reset();

			auto t1 = std::chrono::steady_clock::now();
			evaluator.Evaluate(rnn, evalStream, inputPorts, outputPorts, nEvalFramePerEpoch, evalStepSize);
			auto t2 = std::chrono::steady_clock::now();

			std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
			std::cout << "(" << time_span.count() << " sec)" << std::endl;
		}

		std::cout << "Total trained frames: " << totalTrainedFrame << std::endl;
		std::cout << "Total discarded frames: " << totalDiscardedFrame << std::endl;

		lambdaPostEval(evaluator);


		double curLoss = lambdaLoss(evaluator);
#if 0
		if(totalTrainedFrame > nTrainFramePerEpoch && bestLoss < curLoss)
		{
			rnn.LoadState(rnnSavePath);

			totalTrainedFrame -= nTrainFramePerEpoch;
			learningRate *= 0.5;
			if(learningRate < minLearningRate) break;
			optimizer.SetLearningRate(learningRate);

			printf("New learning rate: %g\n\n", learningRate);
		}
		else
		{
			bestLoss = curLoss;
			rnn.SaveState(rnnSavePath);
		}
#endif

		if(std::isnan(curLoss) == true)
		{
			curLoss = (FLOAT)1.0 / (FLOAT)+0.0;
		}

		if(totalTrainedFrame == nTrainFramePerEpoch || curLoss < bestLoss)
		{
			bestLoss = curLoss;
			totalTrainedFrameAtBest = totalTrainedFrame;
			rnn.SaveState(bestPath);
		}


		if(totalTrainedFrame > nTrainFramePerEpoch && prevLoss < curLoss)
		{
			std::cout << "----------------------------------------------------------------------" << std::endl;

			retryCount++;
			if(retryCount > maxRetryCount)
			{
				retryCount = 0;

				learningRate *= learningRateDecayRate;
				optimizer.SetLearningRate(learningRate);
				//optimizer.SetMetaLearningRate(learningRate);

				if(learningRate < minLearningRate) break;

				rnn.LoadState(pivotPath);
				rnn.SaveState(prevPath);

				std::cout << "Discard the recently trained " << totalTrainedFrame - totalTrainedFrameAtPivot << " frames" << std::endl;
				std::cout << "New learning rate: " << learningRate << std::endl;

				totalDiscardedFrame += totalTrainedFrame - totalTrainedFrameAtPivot;
				totalTrainedFrame = totalTrainedFrameAtPivot;
				prevLoss = pivotLoss;
			}
			else
			{
				std::cout << "Retry count: " << retryCount << " / " << maxRetryCount << std::endl;
			}
		}
		else
		{
			retryCount = 0;

			pivotLoss = prevLoss;
			prevLoss = curLoss;
			pivotPath.swap(prevPath);
			rnn.SaveState(prevPath);

			totalTrainedFrameAtPivot = totalTrainedFrame - nTrainFramePerEpoch;
		}
	}

	rnn.LoadState(bestPath);


	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	std::cout << "Done." << std::endl;
	std::cout << "Total trained frames: " << totalTrainedFrameAtBest << std::endl;
	std::cout << "Total discarded frames: " << totalDiscardedFrame << std::endl;

}

}

