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


#ifndef FRACTAL_RNN_H_
#define FRACTAL_RNN_H_

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <stack>
#include <string>

#include "InitWeightParam.h"
#include "Engine.h"
#include "Layer.h"
#include "Probe.h"
#include "Connection.h"
#include "FractalCommon.h"

namespace fractal
{
class Rnn
{
public:
	Rnn();
	virtual ~Rnn();

	void SetEngine(Engine *engine);
	Engine *GetEngine() const;

	void AddLayer(const std::string &name, ActType actType, StateType stateType, const unsigned long size, const LayerSpec &spec ,const LayerParam &param = LayerParam());
	void AddConnection(const std::string &from, const std::string &to, const unsigned long delayAmount, const bool isIdentity,const ConnSpec &connspec, const InitWeightParam &initWeightParam = InitWeightParam());

	void DeleteLayer(const std::string &name);
	void DeleteConnection(const std::string &from, const std::string &to);

	void LinkProbe(Probe &probe, const std::string &layerName);

	void SetStatePenalty(const FLOAT statePenalty);
	void SetBatchSize(const unsigned long batchSize);
	const unsigned long GetBatchSize() const;
	void InitForward(const unsigned long batchFrom, const unsigned long batchTo);
	void InitBackward(const unsigned long batchFrom, const unsigned long batchTo);
	void InitWeights(const InitWeightParam &param);
	void InitAdadelta(const FLOAT decayRate);
	void InitNesterov();
	void InitRmsprop(const FLOAT decayRate);

	void Forward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream);
	void Backward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream);

	void CalcActDeriv(const unsigned long batchFrom, const unsigned long batchTo);
	void UpdateWeights(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nFrame,
			const FLOAT rate, const FLOAT momentum, const bool adaptiveRates, const bool rmsprop);

	void Synchronize();
	void StreamWait(PStream &stream);

	void Ready();

	void Clear();

	void SaveState(const std::string &path);
	void LoadState(const std::string &path);

	void ReluQuant();
	int WeightQuant_ex1(const std::string &path,int in_M_R,int in_M_F);
	int WeightQuant_ex2(const std::string &path,int in_M_R,int in_M_F,FLOAT best_result);
	int QuantFinetune(const std::string &path);

	const unsigned long GetNumWeights();

	typedef std::list<Layer *> Scc;
	typedef std::list<Layer *> LayerList;
	typedef std::unordered_map<std::string, Layer *> LayerMap;
	typedef std::unordered_set<Connection *> ConnSet;
	typedef std::list<Scc *> SccList;
	typedef std::list<PStream *> PStreamList;
	SccList sccList;
protected:

	Layer *FindLayer(const std::string &layerName);

	void AddConnection(Layer *const from, Layer *const to, const unsigned long delayAmount, const bool isIdentity,const ConnSpec &connSpec ,const InitWeightParam &initWeightParam);

	void DeleteConnection(Layer *const from, Layer *const to);

	void Tarjan();
	void LinkProbe(Probe &probe, Layer *const layer);

	void ClearLayers();
	void ClearConnections();
	void ClearSccList();
	void ClearPStreams();

	Scc *const CreateScc(std::stack<Layer *> &sccStack, const Layer *const root, const long group);

	void CreatePStreams(const unsigned long loc);
	void CreateDefaultPStream(const unsigned long loc);
	void DestroyDefaultPStream();

	Engine *engine;

	LayerMap layerMap;
	ConnSet connSet;
	PStreamList pStreamList;
	PStream *defaultPStream;

	unsigned long batchSize;

	bool isReady;
};

}

#endif /* FRACTAL_RNN_H_ */

