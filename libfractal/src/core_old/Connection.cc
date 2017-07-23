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


#include "Connection.h"

#include <iostream>
#include <fstream>
#include <sstream>
//#include <algorithm>
#include <math.h>

#include "Layer.h"
#include "InitWeightParam.h"

//#define FRACTAL_VERBOSE

namespace fractal{

Connection::Connection(Layer *const from, Layer *const to, const unsigned long delayAmount, const ConnSpec &connSpec, const bool isIdentity)
{
	srcLayer = from;
	dstLayer = to;
	this->delayAmount = delayAmount;
	this->_identity = isIdentity;
        this->spec = connSpec;
	this->quant_done = 0;
	this-> quant_cnt = 0;        
        M = this->spec.M;
	batchSize = 0;
	
	
	if(IsIdentity() == false)
	{
            unsigned long a;
            unsigned long b;
                switch(spec.connType)
                {

                    case CONN_FULL:
                        weights.Resize(dstLayer->GetSize(), srcLayer->GetSize());
                        weightsTrans.Resize(srcLayer->GetSize(), dstLayer->GetSize());
                        vels.Resize(dstLayer->GetSize(), srcLayer->GetSize());
						weights_fixed.Resize(dstLayer->GetSize(), srcLayer->GetSize());
						weightsTrans_fixed.Resize(srcLayer->GetSize(), dstLayer->GetSize());
						weights_fixed_temp.Resize(dstLayer->GetSize(), srcLayer->GetSize());
						weightsTrans_fixed_temp.Resize(srcLayer->GetSize(),dstLayer->GetSize());
                        this->no_weight = false; 
                        break;
                    case CONN_POOL:
                        weights.Resize(dstLayer->GetSize(), srcLayer->GetSize());
                        weightsTrans.Resize(srcLayer->GetSize(), dstLayer->GetSize());
                        vels.Resize(dstLayer->GetSize(), srcLayer->GetSize());
						weights_fixed.Resize(dstLayer->GetSize(), srcLayer->GetSize());
						weightsTrans_fixed.Resize(srcLayer->GetSize(), dstLayer->GetSize());
						weights_fixed_temp.Resize(dstLayer->GetSize(), srcLayer->GetSize());
						weightsTrans_fixed_temp.Resize(srcLayer->GetSize(),dstLayer->GetSize());
                       
                        this->no_weight = true; 

                        break;

                    case CONN_CONV:
                        a = srcLayer->spec.numMaps * dstLayer->spec.numMaps * this->spec.kernelDimX* this->spec.kernelDimY;
                        b = 1;
                        weights.Resize(a,b);
                        weightsTrans.Resize(b,a);
						weights_fixed.Resize(a, b);
						weightsTrans_fixed.Resize(b, a);
						weights_fixed_temp.Resize(a, b);
						weightsTrans_fixed_temp.Resize(b,a);
                        vels.Resize(a,b);
                        this->no_weight = false; 

                        break;
                    
                    case CONN_CONVBIAS:
                        a = dstLayer->spec.numMaps;
                        b = 1;
                        weights.Resize(a,b);
                        weightsTrans.Resize(b,a);
						weights_fixed.Resize(a,b);
						weightsTrans_fixed.Resize(b,a);
						weights_fixed_temp.Resize(a,b);
						weightsTrans_fixed_temp.Resize(b,a);
                        vels.Resize(a,b);
                        this->no_weight = false; 
                                             
                        break;
                }
        }

        engine = NULL;
	stream = NULL;

	rmsDecayRate = (FLOAT) 0.9;

	weightsTransValid = false;
}


Connection::~Connection()
{
	//const unsigned long loc = 0;
	//engine->StreamDestroy(stream_host);
	SetEngine(NULL, NULL);
}


void Connection::SetEngine(Engine *const engine, PStream *const stream)
{
	if(this->engine == engine) return;

	weights.SetEngine(engine);
	weightsTrans.SetEngine(engine);
	weights_fixed.SetEngine(engine);
	weightsTrans_fixed.SetEngine(engine);
	weights_fixed_temp.SetEngine(engine);
	weightsTrans_fixed_temp.SetEngine(engine);
	vels.SetEngine(engine);
	derivs.SetEngine(engine);
	msDeriv.SetEngine(engine);
	msDelta.SetEngine(engine);
	dstAct.SetEngine(engine);
	srcAct.SetEngine(engine);
	dstErr.SetEngine(engine);
	srcErr.SetEngine(engine);

	weightsTransValid = false;

	if(this->engine != NULL)
	{
		this->engine->EventDestroy(event);
		this->stream = NULL;
	}

	this->engine = engine;

	if(engine != NULL)
	{
		engine->EventCreate(event, 1);
		SetPStream(stream);
	}
}


void Connection::SetBatchSize(const unsigned long batchSize)
{
	if(this->batchSize == batchSize) return;

	verify(batchSize >= 0);

	this->batchSize = batchSize;
            srcAct.Resize(srcLayer->GetSize(), batchSize);
            dstAct.Resize(dstLayer->GetSize(), batchSize);
            srcErr.Resize(srcLayer->GetSize(), batchSize);
            dstErr.Resize(dstLayer->GetSize(), batchSize);
}

void Connection::UnlinkMatrices()
{
	weights.Unlink();
	weightsTrans.Unlink();
	weights_fixed.Unlink();
	weightsTrans_fixed.Unlink();
	weights_fixed_temp.Unlink();
	weightsTrans_fixed_temp.Unlink();
	vels.Unlink();
	derivs.Unlink();
	msDeriv.Unlink();
	msDelta.Unlink();
	dstAct.Unlink();
	srcAct.Unlink();
	dstErr.Unlink();
	srcErr.Unlink();

	weightsTransValid = false;
}


void Connection::InitWeights(const InitWeightParam &param)
{
	verify(engine != NULL);

	if(IsIdentity() == false)
	{
                engine->MatRandN(weights, param.mean, param.stdev, *stream);
                engine->MatRandN(weights_fixed, param.mean, param.stdev, *stream);

		weightsTransValid = false;
	}
}


void Connection::InitAdadelta(const FLOAT decayRate)
{
	verify(engine != NULL);

	if(IsIdentity() == false)
	{
		derivs.Resize(dstLayer->GetSize(), srcLayer->GetSize());
		msDeriv.Resize(dstLayer->GetSize(), srcLayer->GetSize());
		msDelta.Resize(dstLayer->GetSize(), srcLayer->GetSize());

		rmsDecayRate = decayRate;

		engine->MatSet(msDeriv, (FLOAT) 1, *stream);
		engine->MatSet(msDelta, (FLOAT) 0, *stream);
	}
}


void Connection::InitNesterov()
{
	verify(engine != NULL);

	if(IsIdentity() == false)
	{
		engine->MatSet(vels, (FLOAT) 0, *stream);
	}
}


void Connection::InitRmsprop(const FLOAT decayRate)
{
	verify(engine != NULL);

	if(IsIdentity() == false)
	{
            if(this->spec.connType == CONN_FULL)
            {
                derivs.Resize(dstLayer->GetSize(), srcLayer->GetSize());
                msDeriv.Resize(dstLayer->GetSize(), srcLayer->GetSize());
            }
            else if(this->spec.connType == CONN_CONV)
            {
                unsigned long a;
                unsigned long b;
                a = srcLayer->spec.numMaps * dstLayer->spec.numMaps * this->spec.kernelDimX* this->spec.kernelDimY;
                b = 1;
                derivs.Resize(a,b);
                msDeriv.Resize(a,b);

            }
            else if(this->spec.connType == CONN_POOL)
            {
                derivs.Resize(dstLayer->GetSize(), srcLayer->GetSize());
                msDeriv.Resize(dstLayer->GetSize(), srcLayer->GetSize());
            }

            else if(this->spec.connType == CONN_CONVBIAS)
            {
                unsigned long a;
                unsigned long b;
                a = dstLayer->spec.numMaps;
                b = 1;
                derivs.Resize(a,b);
                msDeriv.Resize(a,b);
            }

            rmsDecayRate = decayRate;

            engine->MatSet(msDeriv, (FLOAT) 1, *stream);
        }
}


void Connection::InitErr(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(engine != NULL);

	Matrix<FLOAT> srcErrSub(srcErr, batchFrom, batchTo);

	engine->MatSet(srcErrSub, (FLOAT) 0, *stream);

	//EventRecord();
}


void Connection::Forward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream)
{
	unsigned long actFrom, actTo, delay;
#if QUANT_DIRECT
	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();	
#endif
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(engine != NULL);

#ifdef FRACTAL_VERBOSE
	printf("Connection::Forward: %s -> %s %s (%ld, %ld)\n",
			srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
			IsDelayed() == true ? "(DELAYED)" : "",
			batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */

	//srcLayer->StreamWaitEvent(*stream);

	if(IsDelayed() == true)
	{
		delay = IsDelayed() == true ? nStream * delayAmount : 0;

		actFrom = (batchFrom + batchSize - delay) % batchSize;
		actTo = (batchTo + batchSize - delay) % batchSize;

		verify(actFrom >= 0 && actTo < batchSize && actFrom <= actTo);

		Matrix<FLOAT> actSub(srcLayer->act, actFrom, actTo);
		Matrix<FLOAT> srcActSub(srcAct, batchFrom, batchTo);

		engine->MatCopy(actSub, srcActSub, *stream);
	}
	else
	{
		srcAct.Link(srcLayer->act);
	}

	if(IsIdentity() == true)
	{
		dstAct.Link(srcAct);
	}
	else
	{
                Matrix<FLOAT> srcActSub(srcAct, batchFrom, batchTo);
                Matrix<FLOAT> dstActSub(dstAct, batchFrom, batchTo);

            //if(IsDelayed() == true)
            //	engine->FuncBoundRange(dstActSub, dstActSub, (FLOAT) -10, (FLOAT) 10, *stream);
                switch(spec.connType)
                {
                    case CONN_CONVBIAS:
                        
						engine->ConvBiasForward(dstActSub, weights, dstLayer->spec.dimY, dstLayer->spec.dimX, dstLayer->spec.numMaps, batchTo-batchFrom+1,*stream);
                        
                        break;

                    case CONN_FULL:
#if QUANT_DIRECT
						if((NUM_WEIGHTS == dstLayer->size) || M == 100 || quant_done == 0 )// forward step for bias 
						{
							engine->MatMult(weights, false, srcActSub, false, dstActSub, (FLOAT) 1, (FLOAT) 0, *stream);
						}
						else //forward step for general weights
						{
							engine->MatMult(weights_fixed, false, srcActSub, false, dstActSub, (FLOAT) 1, (FLOAT) 0, *stream);
						}
#else
                        engine->MatMult(weights, false, srcActSub, false, dstActSub, (FLOAT) 1, (FLOAT) 0, *stream);
#endif
						break;
                        
                    case CONN_POOL:
                        engine->MaxPoolForward(srcActSub,dstActSub,
                                srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
                                dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
                                batchTo-batchFrom+1,*stream
                                );
//void Engine::MaxPoolForward(FLOAT *_prevLayerAct, FLOAT *_nextLayerState,long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps,      long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps,      long curBatchSize)
                        break;
                    
					case CONN_CONV:
#if QUANT_DIRECT
						if(M==100|| quant_done == 0 )
						{
							engine->ConvForward(srcActSub,dstActSub,weights,
									spec.kernelDimX,spec.kernelDimY,
									srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
									dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
									batchTo-batchFrom+1,*stream); 
						}
						else
						{
							engine->ConvForward(srcActSub,dstActSub,weights_fixed,
									spec.kernelDimX,spec.kernelDimY,
									srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
									dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
									batchTo-batchFrom+1,*stream); 

						}
#else
					    engine->ConvForward(srcActSub,dstActSub,weights,
                                spec.kernelDimX,spec.kernelDimY,
                                srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
                                dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
                                batchTo-batchFrom+1,*stream); 

#endif
//void Engine::ConvForward(FLOAT *_prevLayerAct, FLOAT *_nextLayerState, FLOAT *_weight, long kernelDimX, long kernelDimY, long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps, long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps, long curBatchSize)
                        break;
                    
                    
                    default:
                        verify(false);
                }
        }
        //EventRecord();
}


void Connection::UpdateDstErr(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(engine != NULL);

#ifdef FRACTAL_VERBOSE
	printf("Connection::UpdateDstErr: %s <- %s (%ld, %ld)\n",
			srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
			batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */

	//dstLayer->StreamWaitEvent(*stream);
	dstLayer->DistributeErr(this, batchFrom, batchTo);
}


void Connection::Backward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream)
{
	unsigned long srcErrFrom, srcErrTo, delay;
#if QUANT_RETRAIN
	const int NUM_WEIGHTS = weightsTrans.GetNumRows() * weightsTrans.GetNumCols();	
#endif

	verify(engine != NULL);

#ifdef FRACTAL_VERBOSE
	printf("Connection::Backward: %s <- %s %s (%ld, %ld)\n",
			srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
			IsDelayed() == true ? "(DELAYED)" : "",
			batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */


	delay = IsDelayed() == true ? nStream * delayAmount : 0;
	srcErrFrom = (batchFrom + batchSize - delay) % batchSize;
	srcErrTo = (batchTo + batchSize - delay) % batchSize;
	verify(srcErrFrom >= 0 && srcErrTo < batchSize && srcErrFrom <= srcErrTo);

	Matrix<FLOAT> srcErrSub(srcErr, srcErrFrom, srcErrTo);
	Matrix<FLOAT> dstErrSub(dstErr, batchFrom, batchTo);
        
	Matrix<FLOAT> srcActSub(srcAct, batchFrom, batchTo);//srcErrFrom, srcErrTo);
	Matrix<FLOAT> dstActSub(dstAct, batchFrom, batchTo);
	if(IsIdentity() == true)
	{
		if(IsDelayed() == true)
		{
			engine->MatCopy(dstErrSub, srcErrSub, *stream);
		}
		else
		{
			srcErr.Link(dstErr);
		}
	}
        else
        {
            //engine->MatMult(weights, true, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);
            if(weightsTransValid == false)
            {
                TransposeWeightMatrix();
            }
            switch(spec.connType)
            {
                case CONN_FULL:
#if QUANT_RETRAIN
					if(NUM_WEIGHTS==dstLayer->size || M == 100 )
					{
						engine->MatMult(weightsTrans, false, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);
					}
					else
					{
						engine->MatMult(weightsTrans_fixed, false, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);
					}
#else
					engine->MatMult(weightsTrans, false, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);
				
#endif
					break;

                case CONN_POOL:
                    engine->MaxPoolBackward(srcActSub,dstActSub,
                            srcErrSub,dstErrSub,
                            srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
                            dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
                            batchTo-batchFrom+1,*stream

                            );      
                    //void Engine::MaxPoolBackward(FLOAT *_prevLayerAct, FLOAT *_nextLayerState,FLOAT *_prevLayerErr, FLOAT *_nextLayerErr,        long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps,       long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps, long curBatchSize)
                    break;

                case CONN_CONV:
#if QUANT_RETRAIN
                
					if(M == 100)
					{
				    engine->ConvBackward(srcActSub,srcErrSub,
                            dstErrSub,
                            weights,derivs,
                            spec.kernelDimX,spec.kernelDimY,
                            srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
                            dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
                            true,batchTo-batchFrom+1,*stream

                            );
					}
					else
					{
				    engine->ConvBackward(srcActSub,srcErrSub,
                            dstErrSub,
                            weights_fixed,derivs,
                            spec.kernelDimX,spec.kernelDimY,
                            srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
                            dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
                            true,batchTo-batchFrom+1,*stream

                            );
					
					}
#else
                    engine->ConvBackward(srcActSub,srcErrSub,
                            dstErrSub,
                            weights,derivs,
                            spec.kernelDimX,spec.kernelDimY,
                            srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
                            dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
                            true,batchTo-batchFrom+1,*stream

                            );
#endif

                    //void Engine::ConvBackward(FLOAT *_prevLayerAct, FLOAT *_prevLayerErr, FLOAT *_nextLayerErr, FLOAT *_weight, FLOAT *_deriv, long kernelDimX, long      kernelDimY, long prevLayerDimX, long prevLayerDimY, long        prevLayerNumMaps, long nextLayerDimX, long nextLayerDimY, long        nextLayerNumMaps, bool performBackwardProp, long curBatchSize  )
                    break;
                //case CONN_CONVBIAS:
                    
                    //engine->ConvBiasBackward(dstLayer->_err, _deriv, nextLayer->spec.dimX, nextLayer->spec.dimY, nextLayer->spec.numMaps, curBatchSize);
                    
                  //  break;
                default : printf("sipal default \n");break;
            }

                /* Clip the gradients */
                //if(IsDelayed() == true)
                //	engine->FuncBoundRange(srcErrSub, srcErrSub, (FLOAT) -1, (FLOAT) 1, *stream);
        }

	//EventRecord();
}


void Connection::UpdateWeights(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nFrame,
		const FLOAT rate, const FLOAT momentum, const bool adadelta, const bool rmsprop)
{
        if(this->no_weight != true)
        {
#if QUANT_RETRAIN
	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();// weightsTrans.GetNumRows() * weightsTrans.GetNumCols();	
#endif
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(engine != NULL);

	if(IsIdentity() == true) return;


#ifdef FRACTAL_VERBOSE
	printf("Connection::UpdateWeights: %s -> %s (%ld, %ld)\n",
			srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
			batchFrom, batchTo);
#endif /* FRACTAL_VERBOSE */

	Matrix<FLOAT> srcActSub(srcAct, batchFrom, batchTo);
	Matrix<FLOAT> dstErrSub(dstErr, batchFrom, batchTo);

	Matrix<FLOAT> srcErrSub_dummy(srcErr, batchFrom, batchTo);
	Matrix<FLOAT> dstActSub_dummy(dstErr, batchFrom, batchTo);
	/* Simplified Nesterov momentum (Yoshua Bengio) */
	engine->MatAdd(vels, weights, -momentum, *stream);

	if(adadelta == true)
	{
		verify(rmsprop == false);

		engine->MatMult(dstErrSub, false, srcActSub, true, derivs, (FLOAT) 1, (FLOAT) 0, *stream);
		engine->Adadelta(derivs, derivs, msDeriv, msDelta, rate, rmsDecayRate, *stream);
		engine->MatAdd(vels, vels, momentum - (FLOAT) 1, *stream); // vels *= momentum
		engine->MatAdd(derivs, vels, (FLOAT) 1, *stream); // vels += rate * derivs
	}
	else if(rmsprop == true)
	{
		//engine->MatMult(dstErrSub, false, srcActSub, true, derivs, (FLOAT) 1 / (FLOAT) nFrame, (FLOAT) 0, *stream);
		if(this->spec.connType == CONN_FULL)
		{
#if QUANT_RETRAIN//maybe no need to change..? 
			if(NUM_WEIGHTS==dstLayer->size)
			{
				engine->MatMult(dstErrSub, false, srcActSub, true, derivs, (FLOAT) 1, (FLOAT) 0, *stream);
			}
			else
			{
				engine->MatMult(dstErrSub, false, srcActSub, true, derivs, (FLOAT) 1, (FLOAT) 0, *stream);
			
			}
#else
				engine->MatMult(dstErrSub, false, srcActSub, true, derivs, (FLOAT) 1, (FLOAT) 0, *stream);

#endif
		}
		else if(this->spec.connType == CONN_CONV)
		{
#if QUANT_RETRAIN
if(M==100)
{
			engine->ConvBackward(srcActSub,srcErrSub_dummy,
					dstErrSub,
					weights,derivs,
					spec.kernelDimX,spec.kernelDimY,
					srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
					dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
					false,batchTo-batchFrom+1,*stream

					);
}
else
{
			engine->ConvBackward(srcActSub,srcErrSub_dummy,
					dstErrSub,
					weights_fixed,derivs,
					spec.kernelDimX,spec.kernelDimY,
					srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
					dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
					false,batchTo-batchFrom+1,*stream

					);

}
#else
			engine->ConvBackward(srcActSub,srcErrSub_dummy,
					dstErrSub,
					weights,derivs,
					spec.kernelDimX,spec.kernelDimY,
					srcLayer->spec.dimX,srcLayer->spec.dimY,srcLayer->spec.numMaps,
					dstLayer->spec.dimX,dstLayer->spec.dimY,dstLayer->spec.numMaps,
					false,batchTo-batchFrom+1,*stream

					);

#endif

		}

		else if(this->spec.connType == CONN_CONVBIAS)
		{
			engine->ConvBiasBackward(dstErrSub, derivs, dstLayer->spec.dimX, dstLayer->spec.dimY, dstLayer->spec.numMaps,batchTo-batchFrom+1,*stream);

		}

		engine->Rmsprop(derivs, derivs, msDeriv, rmsDecayRate, *stream);
		engine->MatAdd(vels, vels, momentum - (FLOAT) 1, *stream); // vels *= momentum
		engine->MatAdd(derivs, vels, rate, *stream); // vels += rate * derivs
	}
	else
	{
		/* vels = momentum * vels + rate * derivs */
		//engine->MatMult(dstErrSub, false, srcActSub, true, vels, rate / (FLOAT) nFrame, momentum, *stream);
		engine->MatMult(dstErrSub, false, srcActSub, true, vels, rate, momentum, *stream);
	}
	engine->MatAdd(vels, weights, (FLOAT) 1 + momentum, *stream);
#if QUANT_RETRAIN
	WeightQuant2();
#endif

	weightsTransValid = false;
		}
}


void Connection::SetPStream(PStream *const stream)
{
	verify(stream->engine == engine);
	this->stream = stream;
}


PStream &Connection::GetPStream()
{
	verify(engine != NULL);
	return *stream;
}


void Connection::EventRecord()
{
	verify(engine != NULL);
	engine->EventRecord(event, *stream);
}


void Connection::StreamWaitEvent(PStream &stream)
{
	verify(engine != NULL);
	engine->StreamWaitEvent(stream, event);
}


void Connection::ForwardWait()
{
	srcLayer->StreamWaitEvent(*stream);
}


void Connection::BackwardWait()
{
	dstLayer->StreamWaitEvent(*stream);
}


void Connection::TransposeWeightMatrix()
{
	verify(engine != NULL);

	if(IsIdentity() == false)
	{
		engine->MatTranspose(weights, weightsTrans, *stream);
		engine->MatTranspose(weights_fixed, weightsTrans_fixed, *stream);
		engine->MatTranspose(weights_fixed_temp, weightsTrans_fixed_temp, *stream);

		weightsTransValid = true;
	}
}


void Connection::SaveState(const std::string &filename)
{
	/* Save weights, vels, msDeriv */

	weights.Save(filename + ".weights");
	//weights_fixed.Save(filename + ".weights_fixed");
	vels.Save(filename + ".vels");
	msDeriv.Save(filename + ".msDeriv");
	msDelta.Save(filename + ".msDelta");


	/* Save rmsDecayRate */

	std::string paramFileName = filename + ".param";
        std::ofstream paramFile;

        paramFile.open(paramFileName, std::ios_base::out);

	verify(paramFile.is_open() == true);

	paramFile << "rmsDecayRate = " << rmsDecayRate << std::endl;

	verify(paramFile.good() == true);

        paramFile.close();
}


void Connection::LoadState(const std::string &filename)
{
	/* Load weights, vels, msDeriv */

	weights.Load(filename + ".weights");
	//weights_fixed.Load(filename + ".weights_fixed");
	vels.Load(filename + ".vels");
	msDeriv.Load(filename + ".msDeriv");
	msDelta.Load(filename + ".msDelta");

	weightsTransValid = false;

	/* Load rmsDecayRate */

	std::string paramFileName = filename + ".param";
	std::ifstream paramFile;
	std::string buf, bufLHS, bufRHS;
	size_t pos1, pos2, pos3, pos4, pos5;

	paramFile.open(paramFileName, std::ios_base::in);

	verify(paramFile.is_open() == true);

	while(paramFile.eof() == false)
	{
		std::getline(paramFile, buf);
		verify(paramFile.bad() == false);

		pos1 = buf.find_first_not_of(" \n\r\t");
		if(pos1 == std::string::npos) continue;

		pos5 = buf.find_last_not_of(" \n\r\t");

		pos3 = buf.find_first_of('=');
		verify(pos3 != std::string::npos);
		verify(pos3 > pos1 && pos3 < pos5);

		pos2 = buf.find_last_not_of(" \n\r\t", pos3 - 1);
		pos4 = buf.find_first_not_of(" \n\r\t", pos3 + 1);

		bufLHS = buf.substr(pos1, pos2 - pos1 + 1);
		bufRHS = buf.substr(pos4, pos5 - pos4 + 1);

		//std::cout << bufLHS << "=" << bufRHS << std::endl;

		if(bufLHS == "rmsDecayRate")
		{
			std::istringstream(bufRHS) >> rmsDecayRate;
		}
	}

	paramFile.close();
}

int Connection::sgn(FLOAT val){
	return(FLOAT(0) <val)-(val < FLOAT(0));
}

void Connection::WeightQuant_ex1(const std::string &filename, int in_M)
{
	//printf("point 0\n");
	//std::cout<<"SrcLayer : "<<GetSrcLayer()->GetName()<<std::endl;
	//std::cout<<"DstLayer : "<<GetDstLayer()->GetName()<<std::endl;

	//printf("point 1\n");  
	/* Load weights, vels, msDeriv */

	if(this->no_weight == true)return;
	const unsigned long loc = 0;
	engine->StreamCreate(stream_host,loc);
	//weights.Load(filename + ".weights");

	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols(); 
	printf("NUM_WEIGHTS : %d\n",NUM_WEIGHTS);
	printf("srclayer size  : %u\n",srcLayer->size);
	printf("dstlayer size  : %u\n",dstLayer->size);
	if(NUM_WEIGHTS == dstLayer->size) return;//for bias or peephole
	if(NUM_WEIGHTS == 0) return;//no weights
	if(M == 100) return;
	//printf("point 2\n");  
	FLOAT *ptrweights;
	FLOAT *ptrweights_fixed;
	ptrweights = weights.GetPtrForReadWrite(stream_host);
	ptrweights_fixed = weights_fixed.GetPtrForReadWrite(stream_host);
	//printf("point 3\n");  
	//vels.Load(filename + ".vels");
	//msDeriv.Load(filename + ".msDeriv");
	//msDelta.Load(filename + ".msDelta");
	//const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();  
	//printf("point 4 NUM_WEIGHTS : %d\n",NUM_WEIGHTS);    
	FLOAT max_weights = 0.f;
	FLOAT min_weights = 10000.f;
	for(int i = 0; i< NUM_WEIGHTS;i++)
	{   
		if(fabs(max_weights)< fabs(ptrweights[i])) max_weights = ptrweights[i];
		if(fabs(min_weights) > fabs(ptrweights[i]))min_weights = ptrweights[i];
	}   
	printf("max : %f   min : %f\n",max_weights,min_weights);
	weightsTransValid = false;
	//FLOAT delta = 0.f; //delta t
	//FLOAT delta_pre = 0.0091f;//delta t-1 
	delta = 0.00091f;
	delta_pre=0.00091f;
	//printf("avg : %f\n",delta_pre);
	//printf("avg2 : %f\n",(FLOAT)(max_weights+ min_weights)/(FLOAT)srcLayer->size);
	//delta_pre =fabs((FLOAT)(max_weights+ min_weights)/(FLOAT)srcLayer->size);
	FLOAT z[NUM_WEIGHTS];  //z t
	if(M == 0) M = in_M; //doesn't need anymortkdtn
	//else if((M <= 99.0) && (M >= 101.0))
	printf("%d\n",M);
	FLOAT temp1 = 0.f;
	FLOAT temp2 = 0.f;
	int converge_cnt = 0;
	for(int i = 0; i< NUM_WEIGHTS;i++)
	{
		z[i] = 0.f;
		//printf("%d : %f\n",i,ptrweights[i]);
	}

	//printf("point 5\n");
	while(1)
	{
		for(int i = 0; i< NUM_WEIGHTS ; i++)
		{
			z[i] = sgn(ptrweights[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2));
			//printf("sgn z[i]: %f\n",z[i]);
			//z[i] *= std::min(static_cast<FLOAT>(floor((abs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2));
			//printf("after front: %2.10f  after : %2.10f\n",static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2) );
		}
		for(int i = 0; i< NUM_WEIGHTS ; i++)
		{
			temp1+= z[i]*ptrweights[i];
			temp2+= z[i]*z[i];
		}
		//printf("temp1 : %f temp2 : %f \n",temp1,temp2);
		delta_pre = delta;
		delta  = temp1/temp2;
		//printf("delta : %f\n",delta);
		temp1 = 0.f;
		temp2 = 0.f;
		//std::cout<<std::endl;
		//std::cout<<"delta : "<<delta<<std::endl;
		//std::cout<<"delta_pre : "<<delta_pre<<std::endl;
		//std::cout<<std::endl;
		if((fabs(delta) - fabs(delta_pre))<0.00001 )
		{
			if(converge_cnt >=5) break;
			converge_cnt++;
		}
		else
		{
			converge_cnt = 0;
		}
	}
	printf("delta : %f\n",delta);
	//      delta = delta*1.2;
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = sgn(ptrweights[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta)+0.5)),static_cast<FLOAT>((M-1)/2));
		ptrweights_fixed[i] = delta*ptrweights_fixed[i];
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	weights_fixed.FinishWrite(stream_host);
	quant_done = 1;
}


void Connection::WeightQuant_ex2(const std::string &filename, int in_M, FLOAT best_result)
{
	//printf("point 0\n");
	//std::cout<<"SrcLayer : "<<GetSrcLayer()->GetName()<<std::endl;
	//std::cout<<"DstLayer : "<<GetDstLayer()->GetName()<<std::endl;

	if(this->no_weight == true)return;
	/* Load weights, vels, msDeriv */

	const unsigned long loc = 0;
	engine->StreamCreate(stream_host,loc);
	//weights.Load(filename + ".weights");
	
	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();	
	printf("NUM_WEIGHTS : %d\n",NUM_WEIGHTS);
	printf("srclayer size  : %u\n",srcLayer->size);
	printf("dstlayer size  : %u\n",dstLayer->size);
	if(NUM_WEIGHTS == dstLayer->size) return;//for bias or peephole
	if(NUM_WEIGHTS == 0) return;//no weights
	if(M == 100) return;
	FLOAT *ptrweights;
	FLOAT *ptrweights_fixed;
	ptrweights = weights.GetPtrForReadWrite(stream_host);
	ptrweights_fixed = weights_fixed.GetPtrForReadWrite(stream_host);

if(quant_cnt == 0)
{
	FLOAT max_weights = 0.f;
	FLOAT min_weights = 10000.f;
	for(int i = 0; i< NUM_WEIGHTS;i++)
	{
		if(fabs(max_weights)< fabs(ptrweights[i])) max_weights = ptrweights[i];
		if(fabs(min_weights) > fabs(ptrweights[i]))min_weights = ptrweights[i];
	}
	printf("max : %f   min : %f\n",max_weights,min_weights);
	weightsTransValid = false;
	//FLOAT delta = 0.f; //delta t
	//FLOAT delta_pre = 0.0091f;//delta t-1
	delta = 0.00091f;
	delta_pre=0.00091f;
	//printf("avg : %f\n",delta_pre);
	//printf("avg2 : %f\n",(FLOAT)(max_weights+ min_weights)/(FLOAT)srcLayer->size);
	//delta_pre =fabs((FLOAT)(max_weights+ min_weights)/(FLOAT)srcLayer->size); 
	FLOAT z[NUM_WEIGHTS];  //z t
        if(M == 0)
		{
			 M = in_M; //doesn't need anymortkdtn 
			 printf("M is change 0 to %d\n",M);
        }
		//else if((M <= 99.0) && (M >= 101.0))      
	printf("%d\n",M);
	FLOAT temp1 = 0.f;
	FLOAT temp2 = 0.f;
	int converge_cnt = 0;
	for(int i = 0; i< NUM_WEIGHTS;i++)
	{
		z[i] = 0.f;
		//printf("%d : %f\n",i,ptrweights[i]);
	}

	//printf("point 5\n");	
	while(1)
	{
		for(int i = 0; i< NUM_WEIGHTS ; i++)
		{
			z[i] = sgn(ptrweights[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2));
			//printf("sgn z[i]: %f\n",z[i]);
			//z[i] *= std::min(static_cast<FLOAT>(floor((abs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2));
			//printf("after front: %2.10f  after : %2.10f\n",static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_pre)+0.5)),static_cast<FLOAT>((M-1)/2) );
		}
		for(int i = 0; i< NUM_WEIGHTS ; i++)
		{
			temp1+= z[i]*ptrweights[i];	
			temp2+= z[i]*z[i];
		}
		//printf("temp1 : %f temp2 : %f \n",temp1,temp2);
		delta_pre = delta;
		delta  = temp1/temp2;
		//printf("delta : %f\n",delta);
		temp1 = 0.f;
		temp2 = 0.f;
		//std::cout<<std::endl;
		//std::cout<<"delta : "<<delta<<std::endl;
		//std::cout<<"delta_pre : "<<delta_pre<<std::endl;
		//std::cout<<std::endl;
		if((fabs(delta) - fabs(delta_pre))<0.00001 )
		{
			if(converge_cnt >=5) break;
			converge_cnt++; 
		}
		else
		{
			converge_cnt = 0;
		}
	}
	printf("delta : %f\n",delta);
}
if(quant_cnt < 16)
{
	delta_temp = delta *(((FLOAT)quant_cnt/10.f)+0.5);
	printf("delta_temp : %f, quant_cnt : %d\n",delta_temp,quant_cnt);
	fflush(stdout);
	
//	delta = delta*1.2;
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = sgn(ptrweights[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta_temp)+0.5)),static_cast<FLOAT>((M-1)/2));
		ptrweights_fixed[i] = delta_temp*ptrweights_fixed[i];
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	weights_fixed.FinishWrite(stream_host);
	quant_cnt += 1;
}
if(quant_cnt >= 16)
{
	printf("before : %f\n",delta);
	delta = delta*best_result;
	printf("after : %f\n",delta);
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = sgn(ptrweights[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta)+0.5)),static_cast<FLOAT>((M-1)/2));
		ptrweights_fixed[i] = delta*ptrweights_fixed[i];
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	weights_fixed.FinishWrite(stream_host);
	quant_done = 1;
}
}

void Connection::WeightQuant2()
{
	//printf("point 0\n");
	  //const unsigned long loc = 0;
	  //engine->StreamCreate(stream_host,loc);
	//printf("point 1\n");	
	/* Load weights, vels, msDeriv */

	//weights.Load(filename + ".weights");
	
	if(this->no_weight == true)return;
	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();	
	//printf("NUM_WEIGHTS : %d\n",NUM_WEIGHTS);
	//printf("srclayer size  : %u\n",srcLayer->size);
	//printf("dstlayer size  : %u\n",dstLayer->size);
	if(NUM_WEIGHTS == dstLayer->size) return;
	if(NUM_WEIGHTS == 0) return;
	if(M == 100) return;
	//printf("point 2\n");	
	FLOAT *ptrweights;
	FLOAT *ptrweights_fixed;
	ptrweights = weights.GetPtrForReadWrite(stream_host);
	ptrweights_fixed = weights_fixed.GetPtrForReadWrite(stream_host);
	//printf("%d\n",M);
	//FLOAT delta = 0.f; //delta t
	//delta = delta*1.2;
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = sgn(ptrweights[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights[i])/delta)+0.5)),static_cast<FLOAT>((M-1)/2));
		ptrweights_fixed[i] = delta*ptrweights_fixed[i];
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	//for(int i = 0; i < NUM_WEIGHTS ; i++)
	//{
	//	ptrweights[i] = ptrweights_fixed[i];
	//}
	weights_fixed.FinishWrite(stream_host);
	//weights.FinishWrite(stream_host);
}

void Connection::QuantFinetune(const std::string &filename, double finetune_cnt,double best_finetune_cnt)
{
	//const unsigned long loc = 0;
	//engine->StreamCreate(stream_host,loc);
	/* Load weights, vels, msDeriv */

	//weights.Load(filename + ".weights");
	if(this->no_weight == true)return;
	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();	
	if(NUM_WEIGHTS == dstLayer->size) return;
	if(NUM_WEIGHTS == 0) return;
        if(M == 100) return;
            
	//printf("point 2\n");	
	//FLOAT *ptrweights;
	FLOAT *ptrweights_fixed;
	FLOAT *ptrweights_fixed_temp;
	//ptrweights = weights.GetPtrForReadWrite(stream_host);
	ptrweights_fixed = weights_fixed.GetPtrForReadWrite(stream_host);
	ptrweights_fixed_temp = weights_fixed_temp.GetPtrForReadWrite(stream_host);
        printf("before : %f\n",delta);
	delta = delta * static_cast<FLOAT>(best_finetune_cnt);
        printf("after : %f\n",delta);
        
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = sgn(ptrweights_fixed_temp[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights_fixed_temp[i])/delta)+0.5)),static_cast<FLOAT>((M-1)/2));
		ptrweights_fixed[i] = delta*ptrweights_fixed[i];
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	/*
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = ptrweights_fixed_temp[i]*static_cast<FLOAT>(best_finetune_cnt);
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	*/
	weights_fixed.FinishWrite(stream_host);
}
void Connection::QuantFinetune(const std::string &filename, double finetune_cnt)
{
	//printf("point 0\n");
	//const unsigned long loc = 0;
	//engine->StreamCreate(stream_host,loc);
	//printf("point 1\n");	
	/* Load weights, vels, msDeriv */

	//weights.Load(filename + ".weights");
	if(this->no_weight == true)return;
	const int NUM_WEIGHTS = weights.GetNumRows() * weights.GetNumCols();	
	if(NUM_WEIGHTS == dstLayer->size) return;
	if(NUM_WEIGHTS == 0) return;
        if(M == 100) return;
	//printf("point 2\n");	
	FLOAT *ptrweights;
	FLOAT *ptrweights_fixed;
	FLOAT *ptrweights_fixed_temp;
	ptrweights = weights.GetPtrForReadWrite(stream_host);
	ptrweights_fixed = weights_fixed.GetPtrForReadWrite(stream_host);
	ptrweights_fixed_temp = weights_fixed_temp.GetPtrForReadWrite(stream_host);
	FLOAT delta_temp;
	delta_temp = delta * static_cast<FLOAT>(finetune_cnt); 
        printf("before : %f    after : %f\n",delta,delta_temp);
	if((finetune_cnt > 0.48) && (finetune_cnt < 0.52))
	{
		for(int i = 0; i < NUM_WEIGHTS ; i++)
			ptrweights_fixed_temp[i] = ptrweights[i];

		weights_fixed_temp.FinishWrite(stream_host);
	}
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = sgn(ptrweights_fixed_temp[i])*std::min(static_cast<FLOAT>(floor((fabs(ptrweights_fixed_temp[i])/delta_temp)+0.5)),static_cast<FLOAT>((M-1)/2));
		ptrweights_fixed[i] = delta_temp*ptrweights_fixed[i];
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	/*
	for(int i = 0; i < NUM_WEIGHTS ; i++)
	{
		ptrweights_fixed[i] = ptrweights_fixed_temp[i]*static_cast<FLOAT>(finetune_cnt);
		//printf("i[%d] = %f : %f\n",i,ptrweights[i],ptrweights_fixed[i]);
	}
	*/
	weights_fixed.FinishWrite(stream_host);
}
const unsigned long Connection::GetNumWeights()
{
	if(IsIdentity() == true) return 0;
	else return srcLayer->GetSize() * dstLayer->GetSize();
}

}

