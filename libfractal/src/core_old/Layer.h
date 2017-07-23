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


#ifndef FRACTAL_LAYER_H_
#define FRACTAL_LAYER_H_


#include <string>
#include <list>

#include "Engine.h"
#include "Matrix.h"
#include "FractalCommon.h"



namespace fractal
{

class LayerSpec
{
    public:
        LayerSpec() : dimX(1), dimY(1), numMaps(1){}
		int quant_bit;
        long dimX, dimY;
        long numMaps;
};

enum ActType {ACT_BIAS, ACT_SIGMOID, ACT_TANH, ACT_SOFTPLUS, ACT_RECTLINEAR, ACT_LINEAR, ACT_ONE_MINUS_LINEAR, ACT_INVERSE, ACT_SOFTMAX};
enum StateType {AGG_DONTCARE, AGG_SUM, AGG_MULT};

extern const FLOAT NO_STATE_PENALTY;

class Connection;
class Probe;


class LayerParam
{
    public:
        LayerParam() : initVal((FLOAT) 0), statePenalty(NO_STATE_PENALTY) {}

        FLOAT initVal;
        FLOAT statePenalty;
};


class Layer
{
public:
	typedef std::list<Connection *> ConnList;

	Layer(const std::string &name, ActType actType, StateType stateType, const unsigned long size, const LayerSpec &spec, const LayerParam &param);
	virtual ~Layer();

	void SetEngine(Engine *const engine, PStream *const stream);

	void AddSrcConnection(Connection *const conn);
	void AddDstConnection(Connection *const conn);
	void RemoveSrcConnection(Connection *const conn);
	void RemoveDstConnection(Connection *const conn);

	inline const std::string &GetName() const { return name; }
	inline const unsigned long GetSize() const { return size; }
	inline const unsigned long GetBatchSize() const { return batchSize; }

	void SetBatchSize(const unsigned long batchSize);
	void SetInitVal(const FLOAT val);
	void SetStatePenalty(const FLOAT val);

	void UnlinkMatrices();

	void InitAct(const unsigned long batchFrom, const unsigned long batchTo);
	void InitErr(const unsigned long batchFrom, const unsigned long batchTo);

	void Forward(const unsigned long batchFrom, const unsigned long batchTo);
	void Backward(const unsigned long batchFrom, const unsigned long batchTo);
	
	void CalcActDeriv(const unsigned long batchFrom, const unsigned long batchTo);

	void LinkProbe(Probe *const probe);
	void UnlinkProbe();
	const bool IsLinked() const;

	inline const ConnList &GetSrcConnections() const { return srcList; }
	inline const ConnList &GetDstConnections() const { return dstList; }

	/* For graph algorithms */
	inline void SetVisited(const bool isVisited) { this->isVisited = isVisited; }
	inline const bool GetVisited() const { return isVisited; }
	inline void SetIndex(const long index) { this->index = index; }
	inline const long GetIndex() const { return index; }
	inline void SetGroup(const long group) { this->group = group; }
	inline const long GetGroup() const { return group; }

	void SetPStream(PStream *const stream);
	PStream &GetPStream();

	void EventRecord();
	void StreamWaitEvent(PStream &stream);

	void ForwardWait();
	void BackwardWait();

	
	std::vector<FLOAT> z;
	
	int sgn(FLOAT val);
	
	int quant_bit;
	FLOAT sig_delta;
	FLOAT tanh_delta;
	int M_relu;
	FLOAT relu_delta;
	int relu_delta_decision;
	int relu_delta_final_decision; 
	FLOAT max_act;
	FLOAT min_act;
	void ReluQuant();
	
	PStream stream_host;
	unsigned long size, batchSize;
protected:
	Layer(const Layer &obj);

	void Activation(const unsigned long batchFrom, const unsigned long batchTo);
	void UpdateState(const unsigned long batchFrom, const unsigned long batchTo);

	void UpdateDstErr(const unsigned long batchFrom, const unsigned long batchTo);
	void UpdateSrcErr(const unsigned long batchFrom, const unsigned long batchTo);
	void DistributeErr(Connection *conn, const unsigned long batchFrom, const unsigned long batchTo);

	Engine *engine;

        LayerSpec spec;

	ActType actType;
	StateType stateType;

	std::string name;

	ConnList srcList;
	ConnList dstList;

	Matrix<FLOAT> act, state, srcErr, dstErr;

	Probe *linkedProbe;

	FLOAT initVal, statePenalty;

	/* For graph algorithms */
	bool isVisited;
	long index, group;

	PStream *stream;
	PEvent event;

	friend Probe;
	friend Connection;
};

}

#endif /* FRACTAL_LAYER_H_ */

