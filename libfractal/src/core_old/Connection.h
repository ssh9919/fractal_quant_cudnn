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


#ifndef FRACTAL_CONNECTION_H_
#define FRACTAL_CONNECTION_H_


#include <string>

#include "Engine.h"
#include "Matrix.h"
#include "FractalCommon.h"
//#include <fractal/fractal.h>

namespace fractal
{

enum ConnType {CONN_FULL, CONN_POOL, CONN_CONV, CONN_CONVBIAS};
class ConnSpec
{
    public:
        ConnSpec() : connType(CONN_CONV), kernelDimX(0), kernelDimY(0) {}
        ConnType connType;
        
        long kernelDimX;
        long kernelDimY;
        int M;
};

class Layer;
class InitWeightParam;


class Connection
{
public:
	Connection(Layer *const from, Layer *const to, const unsigned long delayAmount,  const ConnSpec &connSpecconst, bool isIdentity);
	virtual ~Connection();

	void SetEngine(Engine *const engine, PStream *const stream);
	void SetBatchSize(const unsigned long batchSize);

	void UnlinkMatrices();

	void InitWeights(const InitWeightParam &param);

	void InitAdadelta(const FLOAT decayRate);
	void InitNesterov();
	void InitRmsprop(const FLOAT decayRate);

	void InitErr(const unsigned long batchFrom, const unsigned long batchTo);

	void Forward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream);
	void UpdateDstErr(const unsigned long batchFrom, const unsigned long batchTo);
	void Backward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream);

	void UpdateWeights(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nFrame,
			const FLOAT rate, const FLOAT momentum, const bool adaptiveRates, const bool rmsprop);

	inline const bool IsDelayed() const { return delayAmount > 0; }
	inline const bool IsIdentity() const { return _identity; }
	inline Layer *const GetSrcLayer() const { return srcLayer; }
	inline Layer *const GetDstLayer() const { return dstLayer; }

	void SetPStream(PStream *const stream);
	PStream &GetPStream();

	void EventRecord();
	void StreamWaitEvent(PStream &stream);

	void ForwardWait();
	void BackwardWait();

	void SaveState(const std::string &filename);
	void LoadState(const std::string &filename);

	int sgn(FLOAT val);
	int quant_done;
	int quant_cnt;
	void WeightQuant_ex1(const std::string &filename,int in_M);
	void WeightQuant_ex2(const std::string &filename,int in_M,FLOAT best_result);
	void WeightQuant2();
	void QuantFinetune(const std::string &filename,double finetune_cnt);
	void QuantFinetune(const std::string &filename, double finetune_cnt, double best_finetune_cnt);
	const unsigned long GetNumWeights();
        bool no_weight;
	FLOAT delta;
	FLOAT delta_temp;
	FLOAT delta_pre;
	int M;
        ConnSpec spec;
	Matrix<FLOAT> weights, weightsTrans;
	Layer *srcLayer, *dstLayer;
protected:
	void TransposeWeightMatrix();

	Engine *engine;
        


	bool _identity;
	unsigned long delayAmount;


	unsigned long batchSize;
	FLOAT rmsDecayRate;
	//for fixed point optimization mode
	Matrix<FLOAT> weights_fixed, weightsTrans_fixed;
	Matrix<FLOAT> weights_fixed_temp, weightsTrans_fixed_temp;
	///////////////////////////////////
	bool weightsTransValid;

	Matrix<FLOAT> vels; /* momentum */
	Matrix<FLOAT> derivs, msDeriv; /* Rmsprop, Adadelta */
	Matrix<FLOAT> msDelta; /* Adadelta */
	Matrix<FLOAT> dstAct, srcAct;
	Matrix<FLOAT> dstErr, srcErr;

	//for fixed point optimization mode
	Matrix<FLOAT> dstAct_fixed, srcAct_fixed;

	PStream *stream;
	PStream stream_host;
	PEvent event;

	friend Layer;
};

}

#endif /* FRACTAL_CONNECTION_H_ */

