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


#ifndef FRACTAL_CUDAKERNELS_H_
#define FRACTAL_CUDAKERNELS_H_


#include <cuda_runtime.h>
#include "FractalCommon.h"

namespace fractal
{

namespace cudaKernels
{
    template<class T>
    void MemSet(T *_x, const T val, const unsigned long n, const cudaStream_t stream);

    /* _z = _x + _y */
    template<class T>
    void ElemMult(const T *_x, const T *_y, T *_z, const unsigned long n, const cudaStream_t stream);

    /* _z = _x + _y */
    template<class T>
    void Add(const T *_x, const T *_y, T *_z, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void FuncSigmoid(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream, FLOAT delta);

    template<class T>
    void FuncTanh(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream, FLOAT delta);

    template<class T>
    void FuncSoftplus(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void FuncRectLinear(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream, FLOAT delta, int M,int relu_delta_final_decision);

    template<class T>
    void FuncSoftmax(const T *_x, T *_y, const unsigned long layerSize, const unsigned long batchSize, const cudaStream_t stream);

    template<class T>
    void FuncBoundRange(const T *_x, T *_y, const T min, const T max, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void FuncSigmoidDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void FuncTanhDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void FuncSoftplusDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void FuncRectLinearDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void Rmsprop(T *_newDerivs, const T *_derivs, T *_msDeriv, const T decayRate, const unsigned long n, const cudaStream_t stream);

    template<class T>
    void Adadelta(T *_deltas, const T *_derivs, T *_msDeriv, T *_msDelta, const T learningRate, const T decayRate, const unsigned long n, const cudaStream_t stream);
}

}

#endif /* FRACTAL_CUDAKERNELS_H_ */

