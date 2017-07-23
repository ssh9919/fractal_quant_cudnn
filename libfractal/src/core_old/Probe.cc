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


#include "Probe.h"

#include "Layer.h"


namespace fractal
{

Probe::Probe()
{
	linkedLayer = NULL;
	engine = NULL;
	_input = false;
	_output = false;
}


Probe::~Probe()
{
	SetEngine(NULL);
	UnlinkLayer();
}


void Probe::SetEngine(Engine *engine)
{
	if(this->engine != NULL)
	{
		this->engine->EventDestroy(event);
		this->engine->StreamDestroy(stream);
	}

	if(engine != NULL)
	{
		engine->EventCreate(event, 1);
		engine->StreamCreate(stream, 1);
	}

	this->engine = engine;
}


Engine *Probe::GetEngine() const
{
	return engine;
}


void Probe::SetInput(const bool val)
{
	_input = val;
}


void Probe::SetOutput(const bool val)
{
	_output = val;
}


void Probe::LinkLayer(Layer *const layer)
{
	if(linkedLayer != layer)
	{
		UnlinkLayer();
		linkedLayer = layer;
		layer->LinkProbe(this);
	}
}


void Probe::UnlinkLayer()
{
	Layer *tmp;

	if(linkedLayer != NULL)
	{
		tmp = linkedLayer;
		linkedLayer = NULL;
		tmp->UnlinkProbe();
	}
}


const bool Probe::IsLinked() const
{
	return (linkedLayer != NULL);
}


const unsigned long Probe::GetLayerSize() const
{
	verify(linkedLayer != NULL);
	return linkedLayer->GetSize();
}


Matrix<FLOAT> &Probe::GetActivation()
{
	verify(linkedLayer != NULL);
	return linkedLayer->act;
}


Matrix<FLOAT> &Probe::GetState()
{
	verify(linkedLayer != NULL);
	return linkedLayer->state;
}


Matrix<FLOAT> &Probe::GetError()
{
	verify(linkedLayer != NULL);
	return linkedLayer->srcErr;
}


PStream &Probe::GetPStream()
{
	verify(engine != NULL);
	return stream;
}


void Probe::EventRecord()
{
	verify(engine != NULL);
	engine->EventRecord(event, stream);
}


void Probe::EventSynchronize()
{
	verify(engine != NULL);
	engine->EventSynchronize(event);
}


void Probe::StreamWaitEvent(PStream &stream)
{
	verify(engine != NULL);
	engine->StreamWaitEvent(stream, event);
}


void Probe::Wait()
{
	verify(engine != NULL);
	verify(linkedLayer != NULL);

	engine->StreamWaitEvent(stream, linkedLayer->event);
}

}

