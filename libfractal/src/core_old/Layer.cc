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


#include "Layer.h"

#include "Connection.h"
#include "Probe.h"
#include <iostream>

namespace fractal
{

const FLOAT NO_STATE_PENALTY = (FLOAT) -1;

Layer::Layer(const std::string &name, ActType actType, StateType stateType, const unsigned long size, const LayerSpec &spec ,const LayerParam &param)
{
	verify(size > 0);

	this->name = name;
	this->actType = actType;
	this->stateType = stateType;
	this->size = size;
    this->spec = spec;
	this->quant_bit =this->spec.quant_bit;	
	batchSize = 0;
	initVal = (FLOAT) 0;
	 max_act= 0.f;
	 min_act = 10000.f;
	relu_delta_decision = 0;
	relu_delta_final_decision = 0; 
			if(quant_bit != 0)
				sig_delta = 1/(pow(2,quant_bit)-1);
			else
				sig_delta = 100.f;
			if(quant_bit != 0)
				tanh_delta = 1/(pow(2,quant_bit-1)-1);
			else
				tanh_delta = 100.f;
			if(quant_bit != 0)
				M_relu = pow(2,quant_bit)-1;
			else
				M_relu = 100;
	statePenalty = NO_STATE_PENALTY;

	linkedProbe = NULL;

	engine = NULL;
	stream = NULL;

	SetInitVal(param.initVal);
	SetStatePenalty(param.statePenalty);
}


Layer::~Layer()
{
	this->engine->StreamDestroy(stream_host);
	SetEngine(NULL, NULL);

	UnlinkProbe();
}


void Layer::SetEngine(Engine *const engine, PStream *const stream)
{
	if(this->engine == engine) return;

	act.SetEngine(engine);
	state.SetEngine(engine);
	srcErr.SetEngine(engine);
	dstErr.SetEngine(engine);

	if(this->engine != NULL)
	{
		this->engine->EventDestroy(event);
		this->stream = NULL;
	}

	this->engine = engine;

	if(engine != NULL)
	{
	    engine->StreamCreate(stream_host,0);
		engine->EventCreate(event, 1);
		SetPStream(stream);
	}
}


void Layer::SetBatchSize(const unsigned long batchSize)
{
	if(this->batchSize == batchSize) return;

	verify(batchSize >= 0);

	this->batchSize = batchSize;

	act.Resize(size, batchSize);
	state.Resize(size, batchSize);
	srcErr.Resize(size, batchSize);
	dstErr.Resize(size, batchSize);
}


void Layer::SetStatePenalty(const FLOAT val)
{
	statePenalty = val;
}


void Layer::UnlinkMatrices()
{
	act.Unlink();
	state.Unlink();
	srcErr.Unlink();
	dstErr.Unlink();
}


void Layer::SetInitVal(const FLOAT val)
{
	initVal = val;
}


void Layer::InitAct(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(engine != NULL);

	Matrix<FLOAT> actSub(act, batchFrom, batchTo);

	engine->MatSet(actSub, initVal, *stream);

	//EventRecord();
}


void Layer::InitErr(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(engine != NULL);

	Matrix<FLOAT> dstErrSub(dstErr, batchFrom, batchTo);

	engine->MatSet(dstErrSub, (FLOAT) 0, *stream);

	//EventRecord();
}

void Layer::Forward(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);

#ifdef FRACTAL_VERBOSE
	printf("Layer::Forward: %s (%ld, %ld)\n",
			GetName().c_str(), batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */

	if(IsLinked() == true && linkedProbe->IsInput() == true)
	{
		//linkedProbe->StreamWaitEvent(*stream);
	}
	else
	{
		UpdateState(batchFrom, batchTo);
	}
	Activation(batchFrom, batchTo);

	//EventRecord();
}


void Layer::Backward(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);

#ifdef FRACTAL_VERBOSE
	printf("Layer::Backward: %s (%ld, %ld)\n",
			GetName().c_str(), batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */

	if(IsLinked() == true && linkedProbe->IsOutput() == true)
	{
		//linkedProbe->StreamWaitEvent(*stream);
	}
	else
	{
		UpdateDstErr(batchFrom, batchTo);
		UpdateSrcErr(batchFrom, batchTo);
	}

	//EventRecord();
}


void Layer::CalcActDeriv(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(engine != NULL);

#ifdef FRACTAL_VERBOSE
	printf("Layer::CalcActDeriv: %s (%ld, %ld)\n",
			GetName().c_str(), batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */

	Matrix<FLOAT> srcErrSub(srcErr, batchFrom, batchTo);
	Matrix<FLOAT> actSub(act, batchFrom, batchTo);

	switch(actType)
	{
		case ACT_SIGMOID:
			engine->FuncSigmoidDeriv(actSub, srcErrSub, *stream);
			break;

		case ACT_TANH:
			engine->FuncTanhDeriv(actSub, srcErrSub, *stream);
			break;

		case ACT_SOFTPLUS:
			engine->FuncSoftplusDeriv(actSub, srcErrSub, *stream);
			break;

		case ACT_RECTLINEAR:
			engine->FuncRectLinearDeriv(actSub, srcErrSub, *stream);
			break;

		case ACT_LINEAR:
			break;

		case ACT_ONE_MINUS_LINEAR:
			engine->MatSet(srcErrSub, (FLOAT) -1, *stream);
			break;

		case ACT_INVERSE:
			engine->MatSet(srcErrSub, (FLOAT) -1, *stream);
			break;

		case ACT_SOFTMAX:
			/* Not supported */
			verify(false);
			break;

		default:
			verify(false);
	}
}


void Layer::AddSrcConnection(Connection *const conn)
{
	srcList.push_back(conn);
}


void Layer::AddDstConnection(Connection *const conn)
{
	dstList.push_back(conn);
}


void Layer::RemoveSrcConnection(Connection *const conn)
{
	srcList.remove(conn);
}


void Layer::RemoveDstConnection(Connection *const conn)
{
	dstList.remove(conn);
}


void Layer::LinkProbe(Probe *const probe)
{
	if(linkedProbe != probe)
	{
		UnlinkProbe();
		linkedProbe = probe;
		probe->LinkLayer(this);
	}
}


void Layer::UnlinkProbe()
{
	Probe *tmp;

	if(linkedProbe != NULL)
	{
		tmp = linkedProbe;
		linkedProbe = NULL;
		tmp->UnlinkLayer();
	}
}

const bool Layer::IsLinked() const
{
	return (linkedProbe != NULL);
}

int Layer::sgn(FLOAT val){
	return(FLOAT(0) <val)-(val < FLOAT(0));
}
void Layer::ReluQuant()
{
	
	long total_data_num = z.size(); 
	std::vector<FLOAT> temp;
	temp.resize(z.size());
	temp.shrink_to_fit();
	int	converge_cnt=0; 
	FLOAT relu_delta_pre;
	FLOAT temp1 = 0.f;
	FLOAT temp2 = 0.f;
	for(int i = 0 ; i < total_data_num ; i++)
	{
		if(max_act < z[i]) max_act = z[i];
		if(min_act > z[i]) min_act = z[i];
		temp[i] = 0.f;
	}
	std::cout<<"relu max : "<<max_act<<"   relu min : "<<min_act<<std::endl;
	relu_delta = 0.00091f;
	relu_delta_pre = 0.00091f;
	while(1)
	{
		for(int i = 0; i< total_data_num ; i++)
		{
			temp[i] = std::min(static_cast<FLOAT>(floor((fabs(z[i])/relu_delta_pre)+0.5)),static_cast<FLOAT>(M_relu));
			//std::cout<<"z["<<i<<"] : "<<z[i]<<std::endl;
			//z[i] = sgn(ptractSub[i])*std::min(static_cast<FLOAT>(floor((fabs(ptractSub[i])/delta_pre)+0.5)),static_cast<FLOAT>((M_relu-1)/2));
			//printf("sgn z[i]: %f\n",z[i]);
			//z[i] *= std::min(static_cast<FLOAT>(floor((abs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2));
			//printf("after front: %2.10f  after : %2.10f\n",static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2) );
		}
		for(int i = 0; i< total_data_num ; i++)
		{
			temp1+= temp[i]*z[i];	
			temp2+= temp[i]*temp[i];
		}
		//std::cout<<"temp1 : "<<temp1<<" temp2 : "<<temp2<<std::endl;
		relu_delta_pre = relu_delta;
		relu_delta  = temp1/temp2;
		//std::cout<<std::endl;
		//std::cout<<"delta : "<<delta<<std::endl;
		//std::cout<<"delta_pre : "<<delta_pre<<std::endl;
		//std::cout<<std::endl;
		temp1 = 0.f;
		temp2 = 0.f;
		if(fabs(relu_delta) -fabs(relu_delta_pre) < 0.00001)
		{
			if(converge_cnt >=5) break;
			converge_cnt++; 
		}
		else
		{
			converge_cnt = 0;
		}
	}

}
void Layer::Activation(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(engine != NULL);

#if QUANT_RELU
	FLOAT *ptractSub;
	int NUM_SIGNAL = batchTo - batchFrom + 1;
#endif
	Matrix<FLOAT> stateSub(state, batchFrom, batchTo);
	Matrix<FLOAT> actSub(act, batchFrom, batchTo);

	switch(actType)
	{
		case ACT_BIAS:
			engine->MatSet(actSub, (FLOAT) 1, *stream);
			break;

		case ACT_SIGMOID:
			engine->FuncSigmoid(stateSub, actSub, *stream,sig_delta);
			break;

		case ACT_TANH:
			engine->FuncTanh(stateSub, actSub, *stream, tanh_delta);
			break;

		case ACT_SOFTPLUS:
			engine->FuncSoftplus(stateSub, actSub, *stream);
			break;

		case ACT_RECTLINEAR:
			if(relu_delta_final_decision == 1)
			{
				engine->FuncRectLinear(stateSub, actSub, *stream,relu_delta,M_relu,relu_delta_final_decision);
			}
			else
			{
				engine->FuncRectLinear(stateSub, actSub, *stream,100.0, M_relu,relu_delta_final_decision);
					
			}
#if QUANT_RELU
			if(relu_delta_decision == 0)
			{
				ptractSub = actSub.GetPtrForReadWrite(stream_host);
				for(int i = 0 ; i < NUM_SIGNAL ; i++)
					z.push_back(ptractSub[i]);	

				if(z.size()>= ( 240000-1 ) )
				{
					 relu_delta_decision = 1;
					 std::cout<<"relu_delta_decision flag is on !!!"<<std::endl;
					 ReluQuant();	
					 relu_delta_final_decision = 1;
					 std::cout<<"relu_delta : "<<relu_delta<<std::endl;
				}
			}
#endif

#if 0
			if(relu_delta_decision == 0)
			{
				delta_cnt++;
				ptractSub = actSub.GetPtrForReadWrite(stream_host);
				
				for(int i = 0 ; i < NUM_SIGNAL ; i++)
				{
					if(max_actSub < ptractSub[i]) max_actSub = ptractSub[i];
					if(min_actSub > ptractSub[i]) min_actSub = ptractSub[i];
					z[i] = 0.f;
				}
				std::cout<<"max : "<<max_actSub<<" min : "<<min_actSub<<std::endl;
				//delta = 0.0091f;
				//delta_pre = 0.0091f;
				while(1)
				{
					for(int i = 0; i< NUM_SIGNAL ; i++)
					{
						z[i] = std::min(static_cast<FLOAT>(floor((fabs(ptractSub[i])/delta_pre)+0.5)),static_cast<FLOAT>((M_relu-1)/2));//(ddd));
				//		std::cout<<"z["<<i<<"] : "<<z[i]<<std::endl;
						//z[i] = sgn(ptractSub[i])*std::min(static_cast<FLOAT>(floor((fabs(ptractSub[i])/delta_pre)+0.5)),static_cast<FLOAT>((M_relu-1)/2));
						//printf("sgn z[i]: %f\n",z[i]);
						//z[i] *= std::min(static_cast<FLOAT>(floor((abs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2));
						//printf("after front: %2.10f  after : %2.10f\n",static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2) );
					}
					for(int i = 0; i< NUM_SIGNAL ; i++)
					{
						temp1+= z[i]*ptractSub[i];	
						temp2+= z[i]*z[i];
					}
					//std::cout<<"temp1 : "<<temp1<<" temp2 : "<<temp2<<std::endl;
					delta_pre = delta;
					delta  = temp1/temp2;
					//std::cout<<std::endl;
					//std::cout<<"delta : "<<delta<<std::endl;
					//std::cout<<"delta_pre : "<<delta_pre<<std::endl;
					//std::cout<<std::endl;
					temp1 = 0.f;
					temp2 = 0.f;
					if(fabs(delta) -fabs(delta_pre) < 0.0001)
					{
						if(converge_cnt >=5) break;
						converge_cnt++; 
					}
				}
				delta_final += delta;
			//	std::cout<<"delta : "<<delta<<std::endl;
			//	std::cout<<"delta_cnt : "<<delta_cnt<<std::endl;
			//	std::cout<<name<<std::endl;
			}
			else
			{
				ptractSub = actSub.GetPtrForReadWrite(stream_host);
				for(int i = 0; i < NUM_SIGNAL ; i++)
				{
				
//					ptractSub[i] = std::min(static_cast<FLOAT>(floor((fabs(ptractSub[i])/delta_final)+0.5)),static_cast<FLOAT>(ddd));
//					ptractSub[i] = delta_final*ptractSub[i];
					//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
					ptractSub[i] = std::min(static_cast<FLOAT>(floor((fabs(ptractSub[i])/delta)+0.5)),static_cast<FLOAT>(ddd));
					ptractSub[i] = delta*ptractSub[i];
				}

				actSub.FinishWrite(stream_host);
			}
#endif 
			break;

		case ACT_LINEAR:
			//engine->MatCopy(stateSub, actSub, *stream);
			act.Link(state);
			break;

		case ACT_ONE_MINUS_LINEAR:
			engine->MatSet(actSub, (FLOAT) 1, *stream);
			engine->MatAdd(stateSub, actSub, (FLOAT) -1, *stream);
			break;

		case ACT_INVERSE:
			engine->MatSet(actSub, (FLOAT) 0, *stream);
			engine->MatAdd(stateSub, actSub, (FLOAT) -1, *stream);
			break;

		case ACT_SOFTMAX:
			engine->FuncSoftmax(stateSub, actSub, *stream);
			break;

		default:
			verify(false);
	}
}


void Layer::UpdateState(const unsigned long batchFrom, const unsigned long batchTo)
{
	ConnList::const_iterator iter, iter_end;
	Matrix<FLOAT> stateSub(state, batchFrom, batchTo);
	Connection *firstConn = NULL;
	bool isFirst = true;

	verify((stateType == AGG_DONTCARE) == srcList.empty());
	verify(engine != NULL);

	iter_end = srcList.end();
	for(iter = srcList.begin(); iter != iter_end; ++iter)
	{
		//(*iter)->StreamWaitEvent(*stream);

		if(isFirst == true)
		{
			firstConn = (*iter);
			isFirst = false;
		}
		else
		{
			Matrix<FLOAT> srcSub((*iter)->dstAct, batchFrom, batchTo);

			switch(stateType)
			{
				case AGG_SUM:
					if(firstConn == NULL)
					{
						engine->MatAdd(srcSub, stateSub, stateSub, *stream);
					}
					else
					{
						Matrix<FLOAT> firstSrcSub(firstConn->dstAct, batchFrom, batchTo);
						engine->MatAdd(srcSub, firstSrcSub, stateSub, *stream);
						firstConn = NULL;
					}
					break;

				case AGG_MULT:
					if(firstConn == NULL)
					{
						engine->MatElemMult(srcSub, stateSub, stateSub, *stream);
					}
					else
					{
						Matrix<FLOAT> firstSrcSub(firstConn->dstAct, batchFrom, batchTo);
						engine->MatElemMult(srcSub, firstSrcSub, stateSub, *stream);
						firstConn = NULL;
					}
					break;

				default:
					verify(false);
			}
		}
	}

	if(firstConn != NULL)
	{
		//Matrix<FLOAT> firstSrcSub(firstConn->dstAct, batchFrom, batchTo);
		//engine->MatCopy(firstSrcSub, stateSub, *stream);
		state.Link(firstConn->dstAct);
	}
}


void Layer::UpdateDstErr(const unsigned long batchFrom, const unsigned long batchTo)
{
	ConnList::const_iterator iter, iter_end;
	Matrix<FLOAT> dstErrSub(dstErr, batchFrom, batchTo);
	Connection *firstConn = NULL;
	bool isFirst = true;


	verify(engine != NULL);
	//verify(dstList.empty() == false);

	if(dstList.empty() == true)
	{
		engine->MatSet(dstErrSub, (FLOAT) 0, *stream);
		return;
	}

	iter_end = dstList.end();
	for(iter = dstList.begin(); iter != iter_end; ++iter)
	{
		//(*iter)->StreamWaitEvent(*stream);

		if(isFirst == true)
		{
			firstConn = (*iter);
			isFirst = false;
		}
		else
		{
			Matrix<FLOAT> dstSub((*iter)->srcErr, batchFrom, batchTo);

			if(firstConn == NULL)
			{
				engine->MatAdd(dstSub, dstErrSub, dstErrSub, *stream);
			}
			else
			{
				Matrix<FLOAT> firstDstSub(firstConn->srcErr, batchFrom, batchTo);
				engine->MatAdd(dstSub, firstDstSub, dstErrSub, *stream);
				firstConn = NULL;
			}
		}
	}

	if(firstConn != NULL)
	{
		dstErr.Link(firstConn->srcErr);
	}
}


void Layer::UpdateSrcErr(const unsigned long batchFrom, const unsigned long batchTo)
{
	Matrix<FLOAT> dstErrSub(dstErr, batchFrom, batchTo);
	Matrix<FLOAT> srcErrSub(srcErr, batchFrom, batchTo);
	Matrix<FLOAT> stateSub(state, batchFrom, batchTo);

	verify(engine != NULL);

	if(actType != ACT_LINEAR)
	{
		engine->MatElemMult(dstErrSub, srcErrSub, srcErrSub, *stream);
		if(statePenalty > (FLOAT) 0)
		{
			engine->MatAdd(stateSub, srcErrSub, -statePenalty, *stream);
		}
	}
	else
	{
		if(statePenalty > (FLOAT) 0)
		{
			engine->MatCopy(dstErrSub, srcErrSub, *stream);
			engine->MatAdd(stateSub, srcErrSub, -statePenalty, *stream);
		}
		else
		{
			srcErr.Link(dstErr);
		}
	}

/*
	if(actType != ACT_LINEAR)
	{
		Matrix<FLOAT> dstErrSub(dstErr, batchFrom, batchTo);
		Matrix<FLOAT> srcErrSub(srcErr, batchFrom, batchTo);
		engine->MatElemMult(dstErrSub, srcErrSub, srcErrSub);
	}
	else
	{
		srcErr.Link(dstErr);
	}
*/
}


void Layer::DistributeErr(Connection *conn, const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(engine != NULL);

	switch(stateType)
	{
		case AGG_SUM:
			conn->dstErr.Link(srcErr);
			break;

		case AGG_MULT:
			{
				ConnList::const_iterator iter, iter_end;

				Matrix<FLOAT> srcErrSub(srcErr, batchFrom, batchTo);
				Matrix<FLOAT> connErrSub(conn->dstErr, batchFrom, batchTo);

				bool isFirst = true;

				iter_end = srcList.end();
				for(iter = srcList.begin(); iter != iter_end; ++iter)
				{
					if(*iter == conn) continue;

					Matrix<FLOAT> srcActSub((*iter)->dstAct, batchFrom, batchTo);

					if(isFirst == true)
					{
						engine->MatElemMult(srcErrSub, srcActSub, connErrSub, *conn->stream);
						isFirst = false;
					}
					else
					{
						engine->MatElemMult(connErrSub, srcActSub, connErrSub, *conn->stream);
					}
				}

				if(isFirst == true)
				{
					conn->dstErr.Link(srcErr);
				}
			}

			break;

		default:
			verify(false);
	}
}


void Layer::SetPStream(PStream *const stream)
{
	verify(stream->engine == engine);
	this->stream = stream;
}


PStream &Layer::GetPStream()
{
	verify(engine != NULL);
	return *stream;
}


void Layer::EventRecord()
{
	verify(engine != NULL);
	engine->EventRecord(event, *stream);
}


void Layer::StreamWaitEvent(PStream &stream)
{
	verify(engine != NULL);
	engine->StreamWaitEvent(stream, event);
}


void Layer::ForwardWait()
{
	ConnList::const_iterator iter, iter_end;

	if(IsLinked() == true && linkedProbe->IsInput() == true)
	{
		linkedProbe->StreamWaitEvent(*stream);
	}

	iter_end = srcList.end();
	for(iter = srcList.begin(); iter != iter_end; ++iter)
	{
		(*iter)->StreamWaitEvent(*stream);
	}
}


void Layer::BackwardWait()
{
	ConnList::const_iterator iter, iter_end;

	if(IsLinked() == true && linkedProbe->IsOutput() == true)
	{
		linkedProbe->StreamWaitEvent(*stream);
	}

	iter_end = dstList.end();
	for(iter = dstList.begin(); iter != iter_end; ++iter)
	{
		(*iter)->StreamWaitEvent(*stream);
	}
}

}

