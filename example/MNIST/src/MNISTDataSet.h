#ifndef __MNISTDATASET_H__
#define __MNISTDATASET_H__

#include <unordered_map>
#include <vector>
#include <string>
#include <list>

#include <fractal/fractal.h>
#include "infimnist.h"

typedef enum  {InvTF = -1, SIGMOID = 1, TANH = 2, RELU = 3} afunc_t;
typedef float FLOAT;
#define SIPS 0

class MNISTDataSet : public fractal::DataSet
{
public:
	
	static const unsigned long CHANNEL_FEATURE;
	static const unsigned long CHANNEL_LABEL;
	static const unsigned long CHANNEL_SIG_NEWSEQ;

	MNISTDataSet();
	~MNISTDataSet();
	

	const unsigned long GetNumChannel() const;
	const unsigned long GetDimension(const unsigned long channelIdx) const;
	const unsigned long GetNumSeq() const;
	const unsigned long GetNumFrame(const unsigned long seqIdx) const;

	void GetFrameData(const unsigned long seqIdx, const unsigned long channelIdx, const unsigned long frameIdx, FLOAT *const frame);

	int readTestFiles(MNISTDataSet &test_samples);
	int readTrainingDevFiles(MNISTDataSet &train_samples, MNISTDataSet &dev_samples);
	
	void Resize(unsigned long numSamples,unsigned long dimInput,unsigned  long dimTarget,unsigned long numFrames);

	unsigned long LT;
	infimnist_t *p;
protected:
	//typedef std::unordered_map<std::string, unsigned long> LabelTable;

	
	unsigned long nSeq;
	unsigned long featDim, labelDim;
	//unsigned long featPeriod, featWindowSize, labelPeriod;

	//std::vector<std::string> featFile, alignFil;
	
	std::vector<unsigned long> nFrame;

	std::vector<std::vector<FLOAT>> feature;
	//std::vector<std::vector<unsigned char>> feature;
#if SIPS
	std::vector<std::vector<unsigned long>> label;
#else
	std::vector<unsigned long> label;
#endif
	//LabelTable labelTable;
};


int readMNISTDB(MNISTDataSet &train_samples, MNISTDataSet &dev_samples, MNISTDataSet &test_samples, afunc_t inpLT );
#endif /* __MNISTDATASET_H__ */

