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


#include "DataStream.h"

#include <cstring>

#include "DataSet.h"


namespace fractal
{

DataStream::DataStream()
{
    nStream = 1;
    nChannel = 0;
    dataSet = NULL;
    dataOrder = ORDER_SHUFFLE;

    Alloc();
}


void DataStream::LinkDataSet(DataSet *dataSet)
{
    unsigned long channelIdx;

    this->dataSet = dataSet;

    nChannel = dataSet->GetNumChannel();

    dim.clear();
    delay.clear();

    dim.shrink_to_fit();
    delay.shrink_to_fit();

    dim.resize(nChannel);
    delay.resize(nChannel);

    for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
    {
        dim[channelIdx] = dataSet->GetDimension(channelIdx);
        delay[channelIdx] = 0;
    }

    Alloc();
    Reset();
}


void DataStream::UnlinkDataSet()
{
    nChannel = 0;
    dataSet = NULL;

    Alloc();
}


void DataStream::Alloc()
{
    unsigned long streamIdx, channelIdx;

    seqIdx.clear();
    frameIdx.clear();
    bufIdx.clear();
    buf.clear();

    seqIdx.shrink_to_fit();
    frameIdx.shrink_to_fit();
    bufIdx.shrink_to_fit();
    buf.shrink_to_fit();

    seqIdx.resize(nStream);
    frameIdx.resize(nStream);
    bufIdx.resize(nStream);
    buf.resize(nStream);

    for(streamIdx = 0; streamIdx < nStream; streamIdx++)
    {
        bufIdx[streamIdx].resize(nChannel);
        buf[streamIdx].resize(nChannel);

        for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
        {
            buf[streamIdx][channelIdx].resize(delay[channelIdx] * dim[channelIdx]);
        }
    }
}


void DataStream::SetNumStream(const unsigned long nStream)
{
    verify(nStream > 0);

    this->nStream = nStream;

    Alloc();
    Reset();
}


const unsigned long DataStream::GetNumStream() const
{
    return nStream;
}


const unsigned long DataStream::GetNumChannel() const
{
    return nChannel;
}


const unsigned long DataStream::GetDimension(const unsigned long channelIdx) const
{
    verify(channelIdx < nChannel);

    return dim[channelIdx];
}


void DataStream::Reset()
{
    unsigned long streamIdx, channelIdx;
    unsigned long i, maxDelay;

    for(streamIdx = 0; streamIdx < nStream; streamIdx++)
    {
        for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
        {
            bufIdx[streamIdx][channelIdx] = 0;
        }
    }

    maxDelay = 0;
    for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
    {
        maxDelay = maxDelay >= delay[channelIdx] ? maxDelay : delay[channelIdx];
    }

    switch(dataOrder)
    {
        case(ORDER_SHUFFLE):
            Shuffle();
            nextSeqIdx = 0;
            break;

        case(ORDER_RANDOM):
            break;

        case(ORDER_SEQUENTIAL):
            nextSeqIdx = 0;
            break;

        default:
            verify(false);
    }

    for(streamIdx = 0; streamIdx < nStream; streamIdx++)
    {
        NewSeq(streamIdx);

        for(i = 0; i < maxDelay; i++)
        {
            Next(streamIdx);
        }
    }
}


void DataStream::Next(const unsigned long streamIdx)
{
    unsigned long channelIdx, curSeqIdx, curFrameIdx, curBufIdx, curDim;
    FLOAT *curBuf;

    verify(dataSet != NULL);

    curSeqIdx = seqIdx[streamIdx];
    curFrameIdx = frameIdx[streamIdx];

    /* Copy and store delayed frames */
    for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
    {
        if(delay[channelIdx] == 0) continue;

        curBufIdx = bufIdx[streamIdx][channelIdx];
        curDim = dim[channelIdx];
        curBuf = buf[streamIdx][channelIdx].data() + curBufIdx * curDim;

        dataSet->GetFrameData(curSeqIdx, channelIdx, curFrameIdx, curBuf);

        bufIdx[streamIdx][channelIdx] = (curBufIdx + 1) % delay[channelIdx];
    }


    /* Increase the indices */
    frameIdx[streamIdx]++;

    if(frameIdx[streamIdx] == dataSet->GetNumFrame(seqIdx[streamIdx]))
    {
        NewSeq(streamIdx);
    }
}


void DataStream::GenerateFrame(const unsigned long streamIdx, const unsigned long channelIdx, FLOAT *const frame)
{
    unsigned long curSeqIdx, curFrameIdx, curDim, curBufIdx;
    FLOAT *curBuf;

    verify(dataSet != NULL);

    curDim = dim[channelIdx];

    if(delay[channelIdx] > 0)
    {
        curBufIdx = bufIdx[streamIdx][channelIdx];
        curBuf = buf[streamIdx][channelIdx].data() + curBufIdx * curDim;

        memcpy(frame, curBuf, sizeof(FLOAT) * curDim);
    }
    else
    {
        curSeqIdx = seqIdx[streamIdx];
        curFrameIdx = frameIdx[streamIdx];

        dataSet->GetFrameData(curSeqIdx, channelIdx, curFrameIdx, frame);
    }
}


void DataStream::SetDelay(const unsigned long channelIdx, const unsigned long delay)
{
    verify(channelIdx < nChannel);

    this->delay[channelIdx] = delay;

    Alloc();
    Reset();
}


void DataStream::NewSeq(const unsigned long streamIdx)
{
    unsigned long nSeq, newSeqIdx;

    verify(dataSet != NULL);

    nSeq = dataSet->GetNumSeq();
    verify(nSeq > 0);

    switch(dataOrder)
    {
        case(ORDER_SHUFFLE):

            verify(shuffledSeqIdx.size() == nSeq);

            newSeqIdx = shuffledSeqIdx[nextSeqIdx];
            nextSeqIdx++;
            if(nextSeqIdx >= nSeq)
            {
                Shuffle();
                nextSeqIdx = 0;
            }

            break;

        case(ORDER_RANDOM):
            {
                std::uniform_int_distribution<unsigned long> randDist(0, nSeq - 1);
                newSeqIdx = randDist(randGen);
            }

            break;

        case(ORDER_SEQUENTIAL):

            newSeqIdx = nextSeqIdx;
            nextSeqIdx = (nextSeqIdx + 1) % nSeq;

            break;

        default:
            verify(false);
    }

    frameIdx[streamIdx] = 0;
    seqIdx[streamIdx] = newSeqIdx;
    verify(dataSet->GetNumFrame(newSeqIdx) > 0);
}


void DataStream::Shuffle()
{
    unsigned long nSeq, tmp;

    verify(dataSet != NULL);

    nSeq = dataSet->GetNumSeq();

    if(nSeq != shuffledSeqIdx.size())
    {
        shuffledSeqIdx.resize(nSeq);
        shuffledSeqIdx.shrink_to_fit();
        for(unsigned long i = 0; i < nSeq; i++)
        {
            shuffledSeqIdx[i] = i;
        }
    }

    for(unsigned long i = 0; i < nSeq - 1; i++)
    {
        std::uniform_int_distribution<unsigned long> randDist(i, nSeq - 1);
        unsigned long j = randDist(randGen);

        /* Swap */
        tmp = shuffledSeqIdx[i];
        shuffledSeqIdx[i] = shuffledSeqIdx[j];
        shuffledSeqIdx[j] = tmp;
    }
}


void DataStream::SetRandomSeed(const unsigned long long seed)
{
    randGen.seed(seed);
}


void DataStream::SetDataOrder(const DataOrder order)
{
    dataOrder = order;
}


}

