#include "MNISTDataSet.h"

#include <cstring>
#include <fstream>
#include <cstdint>
#include <cmath>

//include for sajid mnist parsing lib
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const unsigned long MNISTDataSet::CHANNEL_FEATURE = 0;
const unsigned long MNISTDataSet::CHANNEL_LABEL = 1;
const unsigned long MNISTDataSet::CHANNEL_SIG_NEWSEQ = 2;

int inputVectorSize = 784;//1024;

const char *datadir = "";
char format = 0;
long firstindex = 10000;
long lastindex = 69999;


MNISTDataSet::MNISTDataSet()
{
	nSeq = 0;

	featDim = 0;
	
	labelDim = 0;
	
	p = infimnist_create(datadir);
}
MNISTDataSet::~MNISTDataSet()
{
  infimnist_destroy(p);
}
const unsigned long MNISTDataSet::GetNumChannel() const
{
	return 3;
}


const unsigned long MNISTDataSet::GetDimension(const unsigned long channelIdx) const
{
	switch(channelIdx)
	{
		case CHANNEL_FEATURE:
			return featDim;

		case CHANNEL_LABEL:
			return labelDim;

		case CHANNEL_SIG_NEWSEQ:
			return 1000;

		default:
			verify(false);
	}

	return 0;
}


const unsigned long MNISTDataSet::GetNumSeq() const
{
	return nSeq;
}


const unsigned long MNISTDataSet::GetNumFrame(const unsigned long seqIdx) const
{
	verify(seqIdx < nSeq);

	return nFrame[seqIdx];
}

struct infimnist_s 
{
  float (*x)[EXSIZE];               /* x[0]...x[l-1] */
  float (*fields)[EXSIZE];          /* F[0]...F[nb_fields-1] */
  float (*tangent)[NTAN][EXSIZE];   /* T[0]...T[2*nb_train-1] */
  float *y; 			    /* category */
  float alpha;
  long  count;
  long (*cachekeys)[CACHECOLS];
  unsigned char* (*cacheptr)[CACHECOLS];
};

void MNISTDataSet::GetFrameData(const unsigned long seqIdx, const unsigned long channelIdx,
		const unsigned long frameIdx, FLOAT *const frame)
{
	unsigned long i;

 	 float alpha;
  	int j,k1,k2;
	verify(seqIdx < nSeq);
	verify(frameIdx < nFrame[seqIdx]);
	switch(channelIdx)
	{
		case CHANNEL_FEATURE:
		
			memcpy(frame, feature[seqIdx].data(), sizeof(FLOAT) * featDim);
			break;
		case CHANNEL_LABEL:
			for(i = 0; i < labelDim; i++)
			{
#if SIPS
				frame[i] = (FLOAT) (label[seqIdx][frameIdx] == i);
#else
				frame[i] = (FLOAT) (label[seqIdx] == i);
#endif
			}
			break;

		case CHANNEL_SIG_NEWSEQ:
			for(i = 0; i < GetDimension(CHANNEL_SIG_NEWSEQ); i++)
			{
				frame[i] = (FLOAT) (frameIdx == 0);
			}
			break;

		default:
			verify(false);
	}
}

void MNISTDataSet::Resize(unsigned long numSamples,unsigned long dimInput,unsigned long dimTarget,unsigned long numFrames)
{
	unsigned long i;
	this->nSeq = numSamples;
	this->featDim = dimInput;
	this->labelDim = dimTarget;
	
	this->nFrame.resize(numSamples);
	this->nFrame.shrink_to_fit();
	
	for(i=0;i<numSamples;i++)
	{
		this->nFrame[i] = numFrames;
	}

	feature.resize(numSamples);
	feature.shrink_to_fit();
	label.resize(numSamples);
	label.shrink_to_fit();

	for(i=0;i<numSamples;i++)
	{
		feature[i].resize(dimInput);
		feature[i].shrink_to_fit();
	}
}

int MNISTDataSet::readTestFiles(MNISTDataSet& test_samples)
{
	//infimnist_t *p;
	int i,j;
	
	for(i=0;i<10000;i++)
	{
		const unsigned char *s = infimnist_get_pattern(p,i);
		test_samples.label[i] = infimnist_get_label(p,i);
		for(j=0;j<inputVectorSize;j++)
		{
			test_samples.feature[i][j] = s[j]/255.0;
		}
	}


  return 0;
}


int MNISTDataSet::readTrainingDevFiles(MNISTDataSet &train_samples, MNISTDataSet &dev_samples)
{
	int i,j;
		
	for(i=10000;i<65000;i++)
	{
		const unsigned char *s = infimnist_get_pattern(p,i);
		train_samples.label[i-10000] = infimnist_get_label(p,i);
		for(j=0;j<inputVectorSize;j++)
		{
			train_samples.feature[i-10000][j] = s[j]/255.0;
		}
	}

	for(i=65000;i<70000;i++)
	{
		const unsigned char *s = infimnist_get_pattern(p,i);
		dev_samples.label[i-65000] = infimnist_get_label(p,i);
		for(j=0;j<inputVectorSize;j++)
		{
			dev_samples.feature[i-65000][j] = s[j]/255.0;
		}
	}



	return 0;
}


int readMNISTDB(MNISTDataSet &train_samples, MNISTDataSet &dev_samples, MNISTDataSet &test_samples, afunc_t inpLT)
{
	train_samples.LT = inpLT;
	test_samples.LT = inpLT;
	dev_samples.LT = inpLT;
	int rTrain = train_samples.readTrainingDevFiles(train_samples,dev_samples);
	if(rTrain == -1)
		verify(false);
	int rTest = test_samples.readTestFiles(test_samples);
	if(rTest == -1)
		verify(false);

	return 0;
}
