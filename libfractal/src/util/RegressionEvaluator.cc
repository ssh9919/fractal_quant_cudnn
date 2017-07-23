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


#include "RegressionEvaluator.h"

namespace fractal
{

const double RegressionEvaluator::GetLoss(const unsigned long outputIdx) const
{
    return GetMeanSquaredError(outputIdx);
}

const double RegressionEvaluator::GetMeanSquaredError(const unsigned long outputIdx) const
{
    verify(outputIdx < nOutput);

    return seSum[outputIdx] / nSample[outputIdx];
}


void RegressionEvaluator::Reset()
{
    unsigned long i;

    for(i = 0; i < nOutput; i++)
    {
        seSum[i] = 0.0;
        nSample[i] = 0;
    }
}


void RegressionEvaluator::EvaluateFrames(const unsigned long outputIdx, Matrix<FLOAT> &target,
        Matrix<FLOAT> &output, const unsigned long nStream, PStream &stream)
{
    unsigned long i, n, dim, nFrame;
    double err, sePartialSum;
    FLOAT *t, *o;

    dim = target.GetNumRows();
    nFrame = target.GetNumCols();

    verify(dim == output.GetNumRows());
    verify(nFrame == output.GetNumCols());

    target.HostPull(stream);
    output.HostPull(stream);

    stream.engine->StreamSynchronize(stream);

    t = target.GetHostData();
    o = output.GetHostData();

    n = dim * nFrame;

    nSample[outputIdx] += nFrame;

    sePartialSum = 0.0;

#ifdef FRACTAL_USE_OMP
    #pragma omp parallel for private(err) \
        reduction(+:sePartialSum)
#endif
    for(i = 0; i < n; i++)
    {
        err = o[i] - t[i];
        sePartialSum += err * err;
    }

    seSum[outputIdx] += sePartialSum;
}


void RegressionEvaluator::MemAlloc()
{
    nSample.clear();
    seSum.clear();

    nSample.shrink_to_fit();
    seSum.shrink_to_fit();

    nSample.resize(nOutput);
    seSum.resize(nOutput);
}

}

