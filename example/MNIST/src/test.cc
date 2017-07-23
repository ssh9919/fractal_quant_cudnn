#include <ctime>
#include <cstdio>
#include <string>
#include <sstream>
#include <iostream>
#include <fractal/fractal.h>
#include "MNISTDataSet.h"
#include <stdlib.h>
using namespace fractal;

int main(int argc, char *argv[])
{
	Rnn rnn;
	Engine engine;
	LayerParam statePenaltyParam;
	InitWeightParam initWeightParam;
	unsigned long long randomSeed;

	struct timespec ts;
	unsigned long inputChannel, outputChannel, inputDim, outputDim;
	//verify(argc == 2);

	

	std::string workspacePath = argv[1];
	//statePenaltyParam.statePenalty = NO_STATE_PENALTY;

	//initWeightParam.stdev = 1e-2;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	randomSeed = ts.tv_nsec;
	printf("Random seed: %lld\n\n", randomSeed);


	MNISTDataSet MNISTTrainData,MNISTDevData,MNISTTestData;
	DataStream MNISTTrainDataStream,MNISTDevDataStream,MNISTTestDataStream;
	
    MNISTTrainData.Resize(55000, 784, 10, 1); 
	MNISTDevData.Resize(5000, 784, 10, 1); 
	MNISTTestData.Resize(10000, 784, 10, 1); 
	
	readMNISTDB(MNISTTrainData,MNISTDevData,MNISTTestData,RELU);

	MNISTTrainDataStream.LinkDataSet(&MNISTTrainData);
	MNISTDevDataStream.LinkDataSet(&MNISTDevData);
	MNISTTestDataStream.LinkDataSet(&MNISTTestData);


	inputChannel = MNISTDataSet::CHANNEL_FEATURE;
	outputChannel = MNISTDataSet::CHANNEL_LABEL;

	inputDim = MNISTTrainData.GetDimension(inputChannel);
	outputDim = MNISTTrainData.GetDimension(outputChannel);

	printf("Train: %ld sequences\n", MNISTTrainData.GetNumSeq());
	printf("  Dev: %ld sequences\n", MNISTDevData.GetNumSeq());
	printf(" Test: %ld sequences\n", MNISTTestData.GetNumSeq());

	printf("\n");

	printf(" Input dim: %ld\n", inputDim);
	printf("Output dim: %ld\n", outputDim);

	printf("\n");

	/* Setting random seeds */
	engine.SetRandomSeed(randomSeed);
	MNISTTrainDataStream.SetRandomSeed(randomSeed);
	MNISTDevDataStream.SetRandomSeed(randomSeed);
	MNISTTestDataStream.SetRandomSeed(randomSeed);
	
	rnn.SetEngine(&engine);

	/* Construct the neural network */

        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 1;                                              
            layerSpec.dimY = 1; 
            layerSpec.numMaps = 1;
            int size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
			rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, size, layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 28;                                              
            layerSpec.dimY = 28; 
            layerSpec.numMaps = 1;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim,layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 24;                                             
            layerSpec.dimY = 24; 
            layerSpec.numMaps = 6;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("CONV1", ACT_RECTLINEAR, AGG_SUM, size,layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 12;                                              
            layerSpec.dimY = 12; 
            layerSpec.numMaps = 6;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("POOL1", ACT_RECTLINEAR, AGG_SUM, size,layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 8;                                              
            layerSpec.dimY = 8; 
            layerSpec.numMaps = 16;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("CONV2", ACT_RECTLINEAR, AGG_SUM, size,layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 4;                                              
            layerSpec.dimY = 4; 
            layerSpec.numMaps = 16;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("POOL2", ACT_RECTLINEAR, AGG_SUM, size,layerSpec);
        }
		{       
            LayerSpec layerSpec;
            layerSpec.dimX = 256;                                            
            layerSpec.dimY = 1; 
            layerSpec.numMaps = 1;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("HIDDEN1",ACT_RECTLINEAR,AGG_SUM,size,layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 256;                                            
            layerSpec.dimY = 1; 
            layerSpec.numMaps = 1;
            unsigned long size =  layerSpec.dimX*layerSpec.dimY*layerSpec.numMaps;
            rnn.AddLayer("HIDDEN2",ACT_RECTLINEAR,AGG_SUM,size,layerSpec);
        }
        {       
            LayerSpec layerSpec;
            layerSpec.dimX = 10;                                              
            layerSpec.dimY = 1; 
            layerSpec.numMaps = 1;
            rnn.AddLayer("OUTPUT", ACT_SOFTMAX, AGG_SUM,outputDim,layerSpec);
        }
        {
            ConnSpec connSpec;
            connSpec.kernelDimX = 5;
            connSpec.kernelDimY = 5;
            connSpec.connType = CONN_CONV;
            
            rnn.AddConnection("INPUT", "CONV1", 0, false,connSpec);
        }
        {
            ConnSpec connSpec;
            connSpec.connType = CONN_POOL;
            rnn.AddConnection("CONV1", "POOL1", 0, false,connSpec);
        }
        {
            ConnSpec connSpec;
            connSpec.kernelDimX = 5;
            connSpec.kernelDimY = 5;
            connSpec.connType = CONN_CONV;
            rnn.AddConnection("POOL1", "CONV2", 0, false,connSpec);
        }
        {
            ConnSpec connSpec;
            connSpec.connType = CONN_POOL;
            rnn.AddConnection("CONV2", "POOL2", 0, false,connSpec);
        }

		{
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("POOL2", "HIDDEN1", 0, false,connSpec);
        }
        
        {
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("HIDDEN1", "HIDDEN2", 0, false,connSpec);
        }
        {
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("HIDDEN2", "OUTPUT", 0, false,connSpec);
        }
		{
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("BIAS", "OUTPUT", 0, false,connSpec);
        }
        { 
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("BIAS", "HIDDEN1", 0, false,connSpec);
        }
        { 
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("BIAS", "HIDDEN2", 0, false,connSpec);
        }
        { 
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            rnn.AddConnection("BIAS", "HIDDEN3", 0, false,connSpec);
        }
	rnn.Ready();

	/* Set ports */
	PortMapList inputPorts, outputPorts;

	inputPorts.push_back(PortMap("INPUT", inputChannel));
	outputPorts.push_back(PortMap("OUTPUT", outputChannel));
	
	ClassificationEvaluator classificationEvaluator;
	AutoOptimizer autoOptimizer;
	
	MNISTTrainDataStream.SetNumStream(128);//64);//128);
	MNISTDevDataStream.SetNumStream(128);//64);//128);
	MNISTTestDataStream.SetNumStream(128);//64);//128);
	//mini batch size = nstream * forward step size;
	
	
	MNISTTrainDataStream.Reset();
	MNISTDevDataStream.Reset();
	MNISTTrainDataStream.Reset();
	/* Initialize weights */
	rnn.InitWeights(1e-2);
	
	printf("Ready\n\n");
	printf("Number of weights: %ld\n\n", rnn.GetNumWeights());


	/* Training */
#if 1
	{
		auto lambdaPostEval = [] (Evaluator &evaluator) -> void
		{
			printf("   Dev:    MSE: %f   ACE: %f   FER: %f\n",
					dynamic_cast<ClassificationEvaluator &>(evaluator).GetMeanSquaredError(0),
					dynamic_cast<ClassificationEvaluator &>(evaluator).GetAverageCrossEntropy(0),
					dynamic_cast<ClassificationEvaluator &>(evaluator).GetFrameErrorRate(0));

			fflush(stdout);
		};

		auto lambdaLoss = [] (Evaluator &evaluator) -> double
		{
			double loss = 0.0;

			loss += dynamic_cast<ClassificationEvaluator &>(evaluator).GetFrameErrorRate(0);
			//loss += dynamic_cast<ClassificationEvaluator &>(evaluator).GetAverageCrossEntropy(0);

			return loss;
		};

		autoOptimizer.SetWorkspacePath(workspacePath);
		autoOptimizer.SetInitLearningRate(1e-4);
		autoOptimizer.SetMinLearningRate(1e-8);
		autoOptimizer.SetLearningRateDecayRate(0.1);
		autoOptimizer.SetMaxRetryCount(5);
		autoOptimizer.SetMomentum(0.9);
		autoOptimizer.SetRmsprop(true);
		//autoOptimizer.SetAdadelta(true);
		autoOptimizer.SetRmsDecayRate(0.95);
		autoOptimizer.SetLambdaPostEval(lambdaPostEval);
		autoOptimizer.SetLambdaLoss(lambdaLoss);

		autoOptimizer.Optimize(rnn, MNISTTrainDataStream, MNISTDevDataStream,
				classificationEvaluator, inputPorts, outputPorts,
				//256 * 1024, 1024 * 1024,64,32);	
				55040, 5120, 1, 1);  //64 * 1024, 32, 16);
	}
#endif
	/* Evaluate the best network */

	MNISTTrainDataStream.Reset();
	MNISTDevDataStream.Reset();
	MNISTTestDataStream.Reset();
	
	printf("Best network:\n");
	fflush(stdout);

	classificationEvaluator.Evaluate(rnn, MNISTTrainDataStream, inputPorts, outputPorts,   55040 ,1);//8 * 1024 * 1024,32);//4 * 1024 * 1024, 32);
	printf("Train :   MSE: %f  ACE: %f  FER: %f\n", classificationEvaluator.GetMeanSquaredError(0),
			classificationEvaluator.GetAverageCrossEntropy(0),
			classificationEvaluator.GetFrameErrorRate(0));
	fflush(stdout);

	classificationEvaluator.Evaluate(rnn, MNISTDevDataStream, inputPorts, outputPorts,  5120 ,1);//8 * 1024 * 1024 ,32);//4 * 1024 * 1024, 32);
	printf("  Dev :   MSE: %f  ACE: %f  FER: %f\n", classificationEvaluator.GetMeanSquaredError(0),
			classificationEvaluator.GetAverageCrossEntropy(0),
			classificationEvaluator.GetFrameErrorRate(0));
	fflush(stdout);

	classificationEvaluator.Evaluate(rnn, MNISTTestDataStream, inputPorts, outputPorts, 10112 ,1);// 8 * 1024 * 1024 ,32);//4 * 1024 * 1024, 32);
	printf("  Test :   MSE: %f  ACE: %f  FER: %f\n", classificationEvaluator.GetMeanSquaredError(0),
			classificationEvaluator.GetAverageCrossEntropy(0),
			classificationEvaluator.GetFrameErrorRate(0));
	fflush(stdout);

	rnn.SetEngine(NULL);

	return 0;
}

