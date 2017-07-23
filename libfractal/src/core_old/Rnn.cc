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



/* Project Minju */


#include "Rnn.h"
#include <sys/stat.h>
#include <sstream>
#include <iostream>

#define MAX_NUM_PSTREAM 4

namespace fractal
{

/* Special indices for graph algorithms */
static const long UNTOUCHED = -1;
static const long TOUCHED = -2;
static const long SCC_DETERMINED = -3;


Rnn::Rnn()
{
	batchSize = 0;
	isReady = false;
	engine = NULL;
	defaultPStream = NULL;
}


Rnn::~Rnn()
{
	Clear();
}


void Rnn::SetEngine(Engine *engine)
{
	if(this->engine == engine) return;
	LayerMap::const_iterator layerIter, layerIter_end;
	ConnSet::const_iterator connIter, connIter_end;

	ClearPStreams();
	DestroyDefaultPStream();

	this->engine = engine;

	if(engine != NULL) CreateDefaultPStream(1);

	layerIter_end = layerMap.end();
	for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
	{
		layerIter->second->SetEngine(engine, defaultPStream);
	}

	connIter_end = connSet.end();
	for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
	{
		(*connIter)->SetEngine(engine, defaultPStream);
	}

	isReady = false;
}


Engine *Rnn::GetEngine() const
{
	return engine;
}


void Rnn::AddLayer(const std::string &name, ActType actType, StateType stateType, const unsigned long size, const LayerSpec &spec, const LayerParam &param)
{
	Layer *layer;

	verify(FindLayer(name) == NULL);

	layer = new Layer(name, actType, stateType, size, spec, param);
	layer->SetBatchSize(batchSize);
	layer->SetEngine(engine, defaultPStream);
	layerMap.insert(std::pair<std::string, Layer *>(name, layer));

	isReady = false;
}


void Rnn::AddConnection(const std::string &from, const std::string &to, const unsigned long delayAmount, const bool isIdentity, const ConnSpec &connSpec, const InitWeightParam &initWeightParam)
{
	Layer *layerFrom, *layerTo;

	layerFrom = FindLayer(from);
	layerTo = FindLayer(to);

	verify(layerFrom != NULL);
	verify(layerTo != NULL);

	AddConnection(layerFrom, layerTo, delayAmount, isIdentity, connSpec, initWeightParam);

	isReady = false;
}


void Rnn::DeleteLayer(const std::string &name)
{
	Layer *layer;
	LayerMap::const_iterator layerIter;
	Layer::ConnList::const_iterator connIter, connIter_end;

	layerIter = layerMap.find(name);
	verify(layerIter != layerMap.end());

	layer = layerIter->second;

	connIter_end = layer->GetSrcConnections().end();
	for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
	{
		(*connIter)->GetSrcLayer()->UnlinkMatrices();
		(*connIter)->GetSrcLayer()->RemoveDstConnection(*connIter);

		verify(connSet.erase(*connIter) == 1);
		delete *connIter;
	}

	connIter_end = layer->GetDstConnections().end();
	for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
	{
		(*connIter)->GetDstLayer()->UnlinkMatrices();
		(*connIter)->GetDstLayer()->RemoveSrcConnection(*connIter);

		verify(connSet.erase(*connIter) == 1);
		delete *connIter;
	}

	layerMap.erase(layerIter);
	delete layer;

	isReady = false;
}


void Rnn::DeleteConnection(const std::string &from, const std::string &to)
{
	Layer *layerFrom, *layerTo;

	layerFrom = FindLayer(from);
	layerTo = FindLayer(to);

	verify(layerFrom != NULL);
	verify(layerTo != NULL);

	DeleteConnection(layerFrom, layerTo);

	isReady = false;
}


void Rnn::LinkProbe(Probe &probe, const std::string &layerName)
{
	Layer *layer;

	layer = FindLayer(layerName);

	verify(layer != NULL);

	LinkProbe(probe, layer);
}


void Rnn::SetBatchSize(const unsigned long batchSize)
{
	verify(batchSize >= 0);

	LayerMap::const_iterator layerIter, layerIter_end;
	ConnSet::const_iterator connIter, connIter_end;


	this->batchSize = batchSize;

	layerIter_end = layerMap.end();
	for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
	{
		layerIter->second->SetBatchSize(batchSize);
	}

	connIter_end = connSet.end();
	for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
	{
		(*connIter)->SetBatchSize(batchSize);
	}
}


const unsigned long Rnn::GetBatchSize() const
{
	return batchSize;
}


void Rnn::InitForward(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);

	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		if((*iter)->IsDelayed() == true)
		{
			(*iter)->GetSrcLayer()->InitAct(batchFrom, batchTo);
			(*iter)->GetSrcLayer()->EventRecord();
		}
	}
}


void Rnn::InitBackward(const unsigned long batchFrom, const unsigned long batchTo)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);

	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		if((*iter)->IsDelayed() == true)
		{
			(*iter)->InitErr(batchFrom, batchTo);
			(*iter)->EventRecord();
		}
	}
}


void Rnn::InitWeights(const InitWeightParam &param)
{
	verify(param.IsValid() == true);
	verify(engine != NULL);

	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
                (*iter)->InitWeights(param);
		engine->StreamSynchronize((*iter)->GetPStream());
	}
}


void Rnn::InitNesterov()
{
	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		(*iter)->InitNesterov();
	}
}


void Rnn::InitAdadelta(const FLOAT decayRate)
{
	verify(decayRate > (FLOAT) 0);

	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		(*iter)->InitAdadelta(decayRate);
	}
}


void Rnn::InitRmsprop(const FLOAT decayRate)
{
	verify(decayRate > (FLOAT) 0);

	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		(*iter)->InitRmsprop(decayRate);
	}
}


void Rnn::Forward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(nStream > 0 && nStream <= batchTo - batchFrom + 1);
	verify((batchTo - batchFrom + 1) % nStream == 0);

	unsigned long i;
	long group;
	bool loopDetected;

	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;

	verify(engine != NULL);

	Ready();

	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		/* Inter-SCC parallel propagation */

		scc = *sccIter;

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group) continue;

				verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

				/* Inter-SCC connection */
				conn->ForwardWait();
				conn->Forward(batchFrom, batchTo, nStream);
				conn->EventRecord();
			}
		}

		/* Intra-SCC propagation */

		/* Check if there are loops inside the scc */
		if(scc->size() == 1)
		{
			loopDetected = false;
			layer = scc->front();
			group = layer->GetGroup();

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group)
				{
					/* Intra-SCC connection */
					loopDetected = true;
					break;
				}
			}
		}
		else
		{
			loopDetected = true;
		}

		if(loopDetected == true)
		{
			/* Sequential propagation */
			for(i = batchFrom; i <= batchTo; i += nStream)
			{
				/* Delayed connection */
				layerIter_end = scc->end();
				for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
				{
					layer = *layerIter;
					group = layer->GetGroup();

					connIter_end = layer->GetSrcConnections().end();
					for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
					{
						conn = *connIter;
						if(conn->GetSrcLayer()->GetGroup() != group) continue;
						if(conn->IsDelayed() == false) continue;

						/* Intra-SCC connection */
						if(i == batchFrom) conn->ForwardWait();
						conn->Forward(i, i + nStream - 1, nStream);
						if(i + nStream > batchTo) conn->EventRecord();
					}
				}

				/* Non-delayed connection */
				layerIter_end = scc->end();
				for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
				{
					layer = *layerIter;
					group = layer->GetGroup();

					connIter_end = layer->GetSrcConnections().end();
					for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
					{
						conn = *connIter;
						if(conn->GetSrcLayer()->GetGroup() != group) continue;
						if(conn->IsDelayed() == true) continue;

						/* Intra-SCC connection */
						//if(i == batchFrom) conn->ForwardWait();
						conn->Forward(i, i + nStream - 1, nStream);
						//if(i + nStream > batchTo) conn->EventRecord();
					}

					/* Intra-SCC layer activation */
					if(i == batchFrom) layer->ForwardWait();
					layer->Forward(i, i + nStream - 1);
					if(i + nStream > batchTo) layer->EventRecord();
				}
			}
		}
		else /* Loop not detected */
		{
			/* Parallel activation */
			layer->ForwardWait();
			layer->Forward(batchFrom, batchTo);
			layer->EventRecord();
		}
	}
}


void Rnn::Backward(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nStream)
{
	verify(batchFrom >= 0 && batchTo < batchSize && batchFrom <= batchTo);
	verify(nStream > 0 && nStream <= batchTo - batchFrom + 1);
	verify((batchTo - batchFrom + 1) % nStream == 0);

	long i;
	long group;
	bool loopDetected;

	SccList::const_reverse_iterator sccIter, sccIter_end;
	Scc::const_reverse_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
        
	verify(engine != NULL);

	Ready();

	sccIter_end = sccList.rend();
	for(sccIter = sccList.rbegin(); sccIter != sccIter_end; ++sccIter)
	{
		/* Inter-SCC parallel propagation */

		scc = *sccIter;

		layerIter_end = scc->rend();
		for(layerIter = scc->rbegin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();

			connIter_end = layer->GetDstConnections().end();
			for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetDstLayer()->GetGroup() == group) continue;

				verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

				/* Inter-SCC connection */
				conn->BackwardWait();
				conn->UpdateDstErr(batchFrom, batchTo);
                                //if(conn->spec.connType == 2)
                                  //  printf("conv : %d\n",layer->GetSrcConnections().empty());
                                //else 
                                 //   printf("full : %d\n",layer->GetSrcConnections().empty());
                                    //if(layer->GetSrcConnections().empty() == false && layer->IsLinked() == false)
				if((layer->GetSrcConnections().empty() == false))
					conn->Backward(batchFrom, batchTo, nStream);
				conn->EventRecord();
			}
		}

		/* Intra-SCC propagation */

		/* Check if there are loops inside the scc */
		if(scc->size() == 1)
		{
			loopDetected = false;
			layer = scc->front();
			group = layer->GetGroup();

			connIter_end = layer->GetDstConnections().end();
			for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetDstLayer()->GetGroup() == group)
				{
					/* Intra-SCC connection */
					loopDetected = true;
					break;
				}
			}
		}
		else
		{
			loopDetected = true;
		}

		if(loopDetected == true)
		{
			/* Sequential propagation */
			for(i = batchTo; i >= (long) batchFrom; i -= nStream)
			{
				/* Non-delayed connection */
				layerIter_end = scc->rend();
				for(layerIter = scc->rbegin(); layerIter != layerIter_end; ++layerIter)
				{
					layer = *layerIter;
					if(layer->GetSrcConnections().empty() == true) continue;
					group = layer->GetGroup();

					/* Intra-SCC layer activation */
					//if(layer->IsLinked() == false)
					if(i == (long) batchTo) layer->BackwardWait();
					layer->Backward(i - nStream + 1, i);
					if(i < (long) (batchFrom + nStream)) layer->EventRecord();

					connIter_end = layer->GetSrcConnections().end();
					for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
					{
						conn = *connIter;
						if(conn->GetSrcLayer()->GetGroup() != group) continue;
						if(conn->IsDelayed() == true) continue;

						/* Intra-SCC connection */
						//if(i == batchTo) conn->BackwardWait();
						conn->UpdateDstErr(i - nStream + 1, i);
						//if(conn->GetSrcLayer()->GetSrcConnections().empty() == false && conn->GetSrcLayer()->IsLinked() == false)
						if(conn->GetSrcLayer()->GetSrcConnections().empty() == false)
							conn->Backward(i - nStream + 1, i, nStream);
						//if(i < batchFrom + nStream) conn->EventRecord();
					}
				}

				/* Delayed connection */
				layerIter_end = scc->rend();
				for(layerIter = scc->rbegin(); layerIter != layerIter_end; ++layerIter)
				{
					layer = *layerIter;
					group = layer->GetGroup();

					connIter_end = layer->GetSrcConnections().end();
					for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
					{
						conn = *connIter;
						if(conn->GetSrcLayer()->GetGroup() != group) continue;
						if(conn->IsDelayed() == false) continue;

						/* Intra-SCC connection */
						//if(i == batchTo) conn->BackwardWait();
						conn->UpdateDstErr(i - nStream + 1, i);
						//if(conn->GetSrcLayer()->GetSrcConnections().empty() == false && conn->GetSrcLayer()->IsLinked() == false)
						if(conn->GetSrcLayer()->GetSrcConnections().empty() == false)
							conn->Backward(i - nStream + 1, i, nStream);
						if(i < (long) (batchFrom + nStream)) conn->EventRecord();
					}
				}
			}
		}
		else /* Loop not detected */
		{
			/* Parallel activation */
			//if(layer->GetSrcConnections().empty() == false && layer->IsLinked() == false)
			if(layer->GetSrcConnections().empty() == false)
			{
				layer->BackwardWait();
				layer->Backward(batchFrom, batchTo);
				layer->EventRecord();
			}
		}
	}
}


void Rnn::CalcActDeriv(const unsigned long batchFrom, const unsigned long batchTo)
{
	LayerMap::const_iterator iter, iter_end;

	iter_end = layerMap.end();
	for(iter = layerMap.begin(); iter != iter_end; ++iter)
	{
		if(iter->second->GetSrcConnections().empty() == false && iter->second->IsLinked() == false)
			iter->second->CalcActDeriv(batchFrom, batchTo);
	}
}


void Rnn::UpdateWeights(const unsigned long batchFrom, const unsigned long batchTo, const unsigned long nFrame,
		const FLOAT rate, const FLOAT momentum, const bool adaptiveRates, const bool rmsprop)
{
	ConnSet::const_iterator iter, iter_end;

	verify(isReady == true);
	verify(engine != NULL);

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		(*iter)->UpdateWeights(batchFrom, batchTo, nFrame, rate, momentum, adaptiveRates, rmsprop);
	}
}


void Rnn::Ready()
{
	if(isReady == true) return;

	verify(layerMap.empty() == false);
	verify(engine != NULL);


	LayerMap::const_iterator layerIter, layerIter_end;
	ConnSet::const_iterator connIter, connIter_end;


	layerIter_end = layerMap.end();
	for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
	{
		layerIter->second->UnlinkMatrices();
	}

	connIter_end = connSet.end();
	for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
	{
		(*connIter)->UnlinkMatrices();
	}


	Tarjan();
	ClearPStreams();
	CreatePStreams(1);

	isReady = true;
}


void Rnn::Tarjan()
{
	/* Tarjan's Algorithm (non-recursive) */
	/* Find strongly connected components and perform topological sort */

	long index, group;
	Layer *v, *w;
	std::stack<Layer *> dfsStack, sccStack;
	LayerMap::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;


	ClearSccList();

	/* Initialize flags */

	layerIter_end = layerMap.end();
	for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
	{
		layerIter->second->SetVisited(false);
		layerIter->second->SetIndex(UNTOUCHED);
	}


	index = 0;
	group = 0;

	layerIter_end = layerMap.end();
	for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
	{
		if(layerIter->second->GetVisited() == true) continue;

		/* Strong conenct: depth first search */

		dfsStack.push(layerIter->second);

		while(dfsStack.empty() == false)
		{
			v = dfsStack.top();

			if(v->GetIndex() == UNTOUCHED)
			{
				sccStack.push(v);
				v->SetIndex(index);
				v->SetGroup(index);
				index++;

				const Layer::ConnList srcConnList = v->GetSrcConnections();

				connIter_end = srcConnList.end();
				for(connIter = srcConnList.begin(); connIter != connIter_end; ++connIter)
				{
					w = (*connIter)->GetSrcLayer();

					if(w->GetIndex() == UNTOUCHED)
						dfsStack.push(w);
				}

				continue;
			}

			dfsStack.pop();
			if(v->GetVisited() == true) continue;

			/* Post visit */

			const Layer::ConnList srcConnList = v->GetSrcConnections();

			connIter_end = srcConnList.end();
			for(connIter = srcConnList.begin(); connIter != connIter_end; ++connIter)
			{
				w = (*connIter)->GetSrcLayer();
				if(w->GetIndex() == SCC_DETERMINED) continue;

				v->SetGroup(std::min(v->GetGroup(), w->GetGroup()));
			}

			/* Is v root? Create new SCC */
			if(v->GetIndex() == v->GetGroup())
			{
				sccList.push_back(CreateScc(sccStack, v, group));
				group++;
			}

			v->SetVisited(true);
		}
	}
}


Rnn::Scc *const Rnn::CreateScc(std::stack<Layer *> &sccStack, const Layer *const root, const long group)
{
	/* Create SCC and perform topological sort */
	/* Memory allocation (scc) */

	Scc *scc;
	Layer *v, *w;
	std::stack<Layer *> dfsStack;
	LayerList sccLayerList;
	LayerList::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;


	printf("%ld: ", group);

	scc = new Scc;

	do
	{
		v = sccStack.top();
		sccStack.pop();

		v->SetIndex(UNTOUCHED);
		v->SetGroup(group);
		v->SetVisited(false);

		sccLayerList.push_front(v);
	} while(v != root);


	layerIter_end = sccLayerList.end();
	for(layerIter = sccLayerList.begin(); layerIter != layerIter_end; ++layerIter)
	{
		if((*layerIter)->GetVisited() == true) continue;
		

		/* Depth first search */

		dfsStack.push(*layerIter);

		while(dfsStack.empty() == false)
		{
			v = dfsStack.top();

			if(v->GetIndex() == UNTOUCHED)
			{
				v->SetIndex(TOUCHED);

				const Layer::ConnList srcConnList = v->GetSrcConnections();

				connIter_end = srcConnList.end();
				for(connIter = srcConnList.begin(); connIter != connIter_end; ++connIter)
				{
					if((*connIter)->IsDelayed() == true) continue;

					w = (*connIter)->GetSrcLayer();

					if(w->GetIndex() == UNTOUCHED)
						dfsStack.push(w);
					else if(w->GetIndex() == TOUCHED)
					{  /* Loop detected */
						fprintf(stderr, "\nLoop detected !!\n");
						verify(false);
					}
				}

				continue;
			}

			dfsStack.pop();
			if(v->GetVisited() == true) continue;

			v->SetIndex(SCC_DETERMINED);

			/* Post visit */

			scc->push_back(v);
			printf("%s ", v->GetName().c_str());

			v->SetVisited(true);
		}
	}

	printf("\n");

	return scc;
}


#if 0
void Rnn::CreatePStreams(unsigned long loc)
{
	unsigned long i, j;
	long group;

	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
	PStream *pStream[16];

	for(i = 0; i < 16; i++)
	{
		pStream[i] = new PStream();
		engine->StreamCreate(*pStream[i], loc);
		pStreamList.push_back(pStream[i]);
	}

	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		scc = *sccIter;

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();

			i = j = 0;
			layer->SetPStream(pStream[i]);
			printf("%s : %ld\n", layer->GetName().c_str(), i);

			connIter_end = layer->GetDstConnections().end();
			for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetDstLayer()->GetGroup() == group)
				{
					/* Intra-scc connection */

					conn->SetPStream(pStream[i]);
					printf("%s -> %s : %ld\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str(), i);

					i = (i + 1) % 16;
				}
				else
				{
					/* Inter-scc connection */

					conn->SetPStream(pStream[j]);
					printf("%s -> %s : %ld\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str(), j);

					j = (j + 1) % 16;
				}
				conn = *connIter;

			}
		}
	}
}
#endif


#if 0
void Rnn::CreatePStreams(unsigned long loc)
{
	long group;

	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
	PStream *interPStream, *intraPStream;


	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		scc = *sccIter;

		intraPStream = new PStream();
		engine->StreamCreate(*intraPStream, loc);
		pStreamList.push_back(intraPStream);

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();

			layer->SetPStream(intraPStream);
			//printf("\n%p: %s\n", intraPStream->cudaStream, layer->GetName().c_str());

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group)
				{
					/* Intra-scc connection */

					conn->SetPStream(intraPStream);
					//printf("%s -> %s\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str());
				}
				else
				{
					/* Inter-scc connection */

					interPStream = new PStream();
					engine->StreamCreate(*interPStream, loc);
					pStreamList.push_back(interPStream);

					conn->SetPStream(interPStream);
					//printf("%s -> %s\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str());
				}
			}
		}
	}
}
#endif


#if 1
void Rnn::CreatePStreams(unsigned long loc)
{
	bool loopDetected;
	long group;
	long nLoop, loopIdx;

	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
	PStream *pStream;

	/* Count the number of loops */
	nLoop = 0;
	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		scc = *sccIter;
		/* Check if there are loops inside the scc */
		if(scc->size() == 1)
		{
			loopDetected = false;
			layer = scc->front();
			group = layer->GetGroup();

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group)
				{
					loopDetected = true;
					break;
				}
			}
		}
		else
		{
			loopDetected = true;
		}

		if(loopDetected == true)
		{
			nLoop++;
		}
	}

	/* Assign PStreams */
	loopIdx = 0;
	pStream = new PStream();
	engine->StreamCreate(*pStream, loc);
	pStreamList.push_back(pStream);

	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		scc = *sccIter;
		/* Check if there are loops inside the scc */
		if(scc->size() == 1)
		{
			loopDetected = false;
			layer = scc->front();
			group = layer->GetGroup();

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group)
				{
					loopDetected = true;
					break;
				}
			}
		}
		else
		{
			loopDetected = true;
		}


		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();

			layer->SetPStream(pStream);

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				conn->SetPStream(pStream);
			}
		}

		if(loopDetected == true)
		{
			if(loopIdx < nLoop - 1 && loopIdx * MAX_NUM_PSTREAM / nLoop != (loopIdx + 1) * MAX_NUM_PSTREAM / nLoop)
			{
				pStream = new PStream();
				engine->StreamCreate(*pStream, loc);
				pStreamList.push_back(pStream);
			}
			loopIdx++;
		}
	}
}
#endif


void Rnn::CreateDefaultPStream(const unsigned long loc)
{
	verify(defaultPStream == NULL);

	defaultPStream = new PStream();
	engine->StreamCreate(*defaultPStream, loc);
}


void Rnn::DestroyDefaultPStream()
{
	if(defaultPStream == NULL) return;

	engine->StreamDestroy(*defaultPStream);

	delete defaultPStream;
	defaultPStream = NULL;
}


void Rnn::Synchronize()
{
	verify(engine != NULL);

	PStreamList::const_iterator iter, iter_end;

	iter_end = pStreamList.end();
	for(iter = pStreamList.begin(); iter != iter_end; ++iter)
	{
		engine->StreamSynchronize(**iter);
	}
}


void Rnn::StreamWait(PStream &stream)
{
	verify(engine != NULL);

	LayerMap::const_iterator layerIter, layerIter_end;
	ConnSet::const_iterator connIter, connIter_end;


	layerIter_end = layerMap.end();
	for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
	{
		layerIter->second->EventRecord();
		layerIter->second->StreamWaitEvent(stream);
	}

	connIter_end = connSet.end();
	for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
	{
		(*connIter)->EventRecord();
		(*connIter)->StreamWaitEvent(stream);
	}
}


void Rnn::Clear()
{
	ClearSccList();
	ClearConnections();
	ClearLayers();
	ClearPStreams();
	DestroyDefaultPStream();

	SetEngine(NULL);

	isReady = false;
}


Layer *Rnn::FindLayer(const std::string &layerName)
{
	LayerMap::const_iterator iter;

	iter = layerMap.find(layerName);

	if(iter == layerMap.end()) return NULL;
	else return iter->second;
}


void Rnn::AddConnection(Layer *const from, Layer *const to, const unsigned long delayAmount, const bool isIdentity, const ConnSpec &connSpec, const InitWeightParam &initWeightParam)
{
	Connection *conn = new Connection(from, to, delayAmount,connSpec, isIdentity );

	conn->SetBatchSize(batchSize);
	conn->SetEngine(engine, defaultPStream);

	if(initWeightParam.IsValid() == true)
	{
		conn->InitWeights(initWeightParam);
	}

	from->AddDstConnection(conn);
	to->AddSrcConnection(conn);

	connSet.insert(conn);
}


void Rnn::DeleteConnection(Layer *const from, Layer *const to)
{
	Connection *conn;
	Layer::ConnList::const_iterator connIter, connIter_end;

	from->UnlinkMatrices();
	to->UnlinkMatrices();

	connIter_end = to->GetSrcConnections().end();
	for(connIter = to->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
	{
		if(from == (*connIter)->GetSrcLayer()) break;;
	}

	verify(connIter != connIter_end);

	conn = *connIter;

	from->RemoveDstConnection(conn);
	to->RemoveSrcConnection(conn);

	verify(connSet.erase(conn) == 1);
	delete conn;
}


void Rnn::LinkProbe(Probe &probe, Layer *const layer)
{
	probe.LinkLayer(layer);
}


void Rnn::ClearLayers()
{
	LayerMap::const_iterator iter, iter_end;

	iter_end = layerMap.end();
	for(iter = layerMap.begin(); iter != iter_end; ++iter)
	{
		delete iter->second;
	}

	layerMap.clear();
}


void Rnn::ClearConnections()
{
	ConnSet::const_iterator iter, iter_end;

	iter_end = connSet.end();
	for(iter = connSet.begin(); iter != iter_end; ++iter)
	{
		delete *iter;
	}

	connSet.clear();
}


void Rnn::ClearSccList()
{
	SccList::const_iterator iter, iter_end;

	iter_end = sccList.end();
	for(iter = sccList.begin(); iter != iter_end; ++iter)
	{
		delete *iter;
	}

	sccList.clear();
}


void Rnn::ClearPStreams()
{
	PStreamList::const_iterator iter, iter_end;

	iter_end = pStreamList.end();
	for(iter = pStreamList.begin(); iter != iter_end; ++iter)
	{
		engine->StreamDestroy(**iter);
		delete *iter;
	}

	pStreamList.clear();
}


void Rnn::SaveState(const std::string &path)
{
	/* For Linux */

	verify(path != "");

	system(std::string("mkdir -p " + path).c_str());

	//mkdir(path.c_str(), 0755);
	for(auto &layer : layerMap)
	{
		std::string dstLayerName = layer.second->GetName();
		std::string dirPath = path + "/" + dstLayerName;
		mkdir(dirPath.c_str(), 0755);

		//unsigned long i = 0;

		for(auto &conn : layer.second->GetSrcConnections())
		{
			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;

			conn->SaveState(filename.str());

			//i++;
		}
	}
}


void Rnn::LoadState(const std::string &path)
{
	/* For Linux */

	verify(path != "");

	for(auto &layer : layerMap)
	{
		std::string dstLayerName = layer.second->GetName();
		std::string dirPath = path + "/" + dstLayerName;

		//unsigned long i = 0;

		for(auto &conn : layer.second->GetSrcConnections())
		{
			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;

			conn->LoadState(filename.str());

			//i++;
		}
	}
}
void Rnn::ReluQuant()
{

	for(auto &layer : layerMap)
	{
		layer.second->ReluQuant();
		std::cout<<"verify layer : "<<layer.second->GetName()<<std::endl;
	}
}
int Rnn::WeightQuant_ex2(const std::string &path,int in_M_R,int in_M_F,FLOAT best_result)
{
	/* For Linux */
	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
	long group;

	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		/*Inter-SCC parallel propagation*/
		scc = *sccIter;

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();
			std::string dstLayerName = layer->GetName();
			std::string dirPath = path + "/" + dstLayerName;

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group) continue;
				const int NUM_WEIGHTS = conn->weights.GetNumRows()*conn->weights.GetNumCols();
				if(NUM_WEIGHTS == conn->dstLayer->size) continue;
				if(NUM_WEIGHTS == 0) continue;
				if(conn->M == 100) continue;


			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;
			printf("\n");
			std::cout<<filename.str()<<std::endl;
				/* Inter-SCC connection */
			if(conn->quant_done == 0)
			{
				conn->WeightQuant_ex2(filename.str(),in_M_F,best_result);
				return 0;
			}
			}
		}

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();
			std::string dstLayerName = layer->GetName();
			std::string dirPath = path + "/" + dstLayerName;

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() != group) continue;
				const int NUM_WEIGHTS = conn->weights.GetNumRows()*conn->weights.GetNumCols();
				if(NUM_WEIGHTS == conn->dstLayer->size) continue;
				if(NUM_WEIGHTS == 0) continue;
				if(conn->M == 100) continue;

				//verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;
			printf("\n");
			std::cout<<filename.str()<<std::endl;
				/* Inter-SCC connection */
			if(conn->quant_done == 0)	
			{
				conn->WeightQuant_ex2(filename.str(),in_M_R,best_result);
				return 0;
			}
			}
		}

	}	
		//printf("rnn point 1\n");	
	verify(path != "");
	return 1;
#if 0
	for(auto &layer : layerMap)
	{
		std::string dstLayerName = layer.second->GetName();
		std::string dirPath = path + "/" + dstLayerName;

	//printf("rnn point 2\n");	
		//unsigned long i = 0;

		for(auto &conn : layer.second->GetSrcConnections())
		{
	//printf("rnn point 3\n");	
			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;

	//printf("rnn point 4\n");
			printf("\n");
			std::cout<<filename.str()<<std::endl;
			conn->WeightQuant(filename.str());
	//printf("rnn point 5\n");	
			//i++;
		}
	}
#endif
}
int Rnn::WeightQuant_ex1(const std::string &path,int in_M_R,int in_M_F)
{
	/* For Linux */
	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
	long group;

	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{    
		/*Inter-SCC parallel propagation*/
		scc = *sccIter;
		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{    
			layer = *layerIter;
			group = layer->GetGroup();
			std::string dstLayerName = layer->GetName();
			std::string dirPath = path + "/" + dstLayerName;

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{    
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() == group) continue;

				//verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

				std::string srcLayerName = conn->GetSrcLayer()->GetName();
				std::stringstream filename;
				//filename << dirPath << "/" << i << "." << srcLayerName;
				filename << dirPath << "/" << srcLayerName;
				printf("\n");
				std::cout<<filename.str()<<std::endl;
				/* Inter-SCC connection */
				conn->WeightQuant_ex1(filename.str(),in_M_F);
			}    
		}    

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{    
			layer = *layerIter;
			group = layer->GetGroup();
			std::string dstLayerName = layer->GetName();
			std::string dirPath = path + "/" + dstLayerName;

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				if(conn->GetSrcLayer()->GetGroup() != group) continue;

				//verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

				std::string srcLayerName = conn->GetSrcLayer()->GetName();
				std::stringstream filename;
				//filename << dirPath << "/" << i << "." << srcLayerName;
				filename << dirPath << "/" << srcLayerName;
				printf("\n");
				std::cout<<filename.str()<<std::endl;
				/* Inter-SCC connection */
				conn->WeightQuant_ex1(filename.str(),in_M_R);
			}
		}

	}
	//printf("rnn point 1\n");
	verify(path != "");
#if 0
	for(auto &layer : layerMap)
	{
		std::string dstLayerName = layer.second->GetName();
		std::string dirPath = path + "/" + dstLayerName;

		//printf("rnn point 2\n");
		//unsigned long i = 0;

		for(auto &conn : layer.second->GetSrcConnections())
		{
			//printf("rnn point 3\n");
			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;

			//printf("rnn point 4\n");
			printf("\n");
			std::cout<<filename.str()<<std::endl;
			conn->WeightQuant(filename.str());
			//printf("rnn point 5\n");
			//i++;
		}
	}
#endif
}

#if 0
int Rnn::QuantFinetune(const std::string &path )
{
	/* For Linux */
	SccList::const_iterator sccIter, sccIter_end;
	Scc::const_iterator layerIter, layerIter_end;
	Layer::ConnList::const_iterator connIter, connIter_end;

	Scc *scc;
	Layer *layer;
	Connection *conn;
	long group;

	sccIter_end = sccList.end();
	for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
	{
		scc = *sccIter;

		layerIter_end = scc->end();
		for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
		{
			layer = *layerIter;
			group = layer->GetGroup();
			std::string dstLayerName = layer->GetName();
			std::string dirPath = path + "/" + dstLayerName;

			connIter_end = layer->GetSrcConnections().end();
			for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
			{
				conn = *connIter;
				//if(conn->GetSrcLayer()->GetGroup() == group) continue;

				//verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

				std::string srcLayerName = conn->GetSrcLayer()->GetName();
				std::stringstream filename;
				//filename << dirPath << "/" << i << "." << srcLayerName;
				filename << dirPath << "/" << srcLayerName;
				printf("\n");
				std::cout<<filename.str()<<std::endl;
				/* Inter-SCC connection */
				//conn->QuantFinetune(filename.str());
			}
		}
	}	
		//printf("rnn point 1\n");	
	verify(path != "");
#if 0
	for(auto &layer : layerMap)
	{
		std::string dstLayerName = layer.second->GetName();
		std::string dirPath = path + "/" + dstLayerName;

	//printf("rnn point 2\n");	
		//unsigned long i = 0;

		for(auto &conn : layer.second->GetSrcConnections())
		{
	//printf("rnn point 3\n");	
			std::string srcLayerName = conn->GetSrcLayer()->GetName();
			std::stringstream filename;
			//filename << dirPath << "/" << i << "." << srcLayerName;
			filename << dirPath << "/" << srcLayerName;

	//printf("rnn point 4\n");
			printf("\n");
			std::cout<<filename.str()<<std::endl;
			conn->WeightQuant(filename.str());
	//printf("rnn point 5\n");	
			//i++;
		}
	}
#endif
}
#endif
const unsigned long Rnn::GetNumWeights()
{
	unsigned long numWeights = 0;

	for(auto &conn : connSet)
	{
		numWeights += conn->GetNumWeights();
	}

	return numWeights;
}

}

