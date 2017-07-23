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


#include "CudaKernels.h"
#include <stdio.h>
#define THREAD_PER_BLOCK 512


namespace fractal
{

namespace cudaKernels
{


template<class T>
inline __device__ T _exp(const T x);

template<class T>
inline __device__ T _log(const T x);

template<class T>
inline __device__ T _sqrt(const T x);

template<class T>
static __global__ void MemSetKernel(T *x, const T val, const unsigned long n);

template<class T>
static __global__ void ElemMultKernel(const T *x, const T *y, T *z, const unsigned long n);

template<class T>
static __global__ void AddKernel(const T *x, const T *y, T *z, const unsigned long n);

template<class T>
static __global__ void FuncSigmoidKernel(const T *x, T *y, const unsigned long n, FLOAT delta);

/* IBM check start */
/* template signal quantization for tanh */
template<class T>
static __global__ void FuncTanhKernel(const T *x, T *y, const unsigned long n,FLOAT delta);
/* template signal quantization for tanh */
/* IBM check end */

/* IBM check start */
/* template weight quantization */
template<class T> 
static __global__ void WeightQuantKernel(const T *x, T *y, const unsigned long n,FLOAT delta, int M);
/* template weight quantization */
/* IBM check end */

template<class T>
static __global__ void FuncSoftplusKernel(const T *x, T *y, const unsigned long n);

/* IBM check start */
/* template signal quantization for rectlinear */
template<class T>
static __global__ void FuncRectLinearKernel(const T *x, T *y,T *y_fixed , const unsigned long n,FLOAT delta, int M,int relu_delta_final_decision);
/* template signal quantization for rectlinear */
/* IBM check end */

template<class T>
static __global__ void FuncSoftmaxKernel(const T *x, T *y, const unsigned long n);

template<class T>
static __global__ void FuncBoundRangeKernel(const T *x, T *y, const T _min, const T _max, const unsigned long n);

template<class T>
static __global__ void FuncSigmoidDerivKernel(const T *x, T *y, const unsigned long n);

template<class T>
static __global__ void FuncTanhDerivKernel(const T *x, T *y, const unsigned long n);

template<class T>
static __global__ void FuncSoftplusDerivKernel(const T *x, T *y, const unsigned long n);

template<class T>
static __global__ void FuncRectLinearDerivKernel(const T *x, T *y, const unsigned long n);

template<class T>
    static __global__ void GenerateDropoutMaskKernel(T *mask, const T *uniformDist, const unsigned long n, const T dropoutRate);

template<class T>
static __global__ void RmspropKernel(T *newDerivs, const T *derivs, T *msDeriv, const T decayRate, const unsigned long n);

template<class T>
static __global__ void AdadeltaKernel(T *deltas, const T *derivs, T *msDeriv, T *msDelta, const T learningRate, const T decayRate, const unsigned long n);


template<>
inline __device__ float _exp<float>(const float x)
{
    return min(__expf(x), 1e30);
}


template<>
inline __device__ double _exp<double>(const double x)
{
    return min(exp(x), 1e300);
}


template<>
inline __device__ float _log<float>(const float x)
{
    return __logf(x);
}


template<>
inline __device__ double _log<double>(const double x)
{
    return log(x);
}


template<>
inline __device__ float _sqrt<float>(const float x)
{
    return __fsqrt_rn(x);
}


template<>
inline __device__ double _sqrt<double>(const double x)
{
    return __dsqrt_rn(x);
}



template<class T>
static __global__ void MemSetKernel(T *x, const T val, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    x[idx] = val;
}


template<class T>
static __global__ void ElemMultKernel(const T *x, const T *y, T *z, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    z[idx] = x[idx] * y[idx];
}


template<class T>
static __global__ void AddKernel(const T *x, const T *y, T *z, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    z[idx] = x[idx] + y[idx];
}


/* IBM check start */
/* Signal quantization kernel for Sigmoid */
/* If the QUANT_RELU flag is on Quantization Model */
template<class T>
static __global__ void FuncSigmoidKernel(const T *x, T *y,T *y_fixed, const unsigned long n, FLOAT delta)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    y_fixed[idx] = (T)1 / ((T)1 + _exp<T>(-x[idx]));
    y[idx] = (T)1 / ((T)1 + _exp<T>(-x[idx]));
#if QUANT_RELU // condition for signal quantization 
if((delta <101.0 && delta > 99.0) == 0)
{
	y[idx] = floor((fabs(y_fixed[idx])/delta)+(T)0.5);
	y[idx] = y[idx]*delta;
}
#endif
}
/* Signal quantization kernel for Sigmoid */
/* IBM check end */


/* IBM check start */
/* Signal quantization kernel for Tanh */
/* If the QUANT_RELU flag is on Quantization Model */
template<class T>
static __global__ void FuncTanhKernel(const T *x, T *y, const unsigned long n, FLOAT delta)
{
    unsigned long idx;
    T v;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    v = _exp<T>((T)(-2) * x[idx]);

    y[idx] = (T)2 / ((T)1 + v) - (T)1;
#if QUANT_RELU
if((delta <101.0 && delta > 99.0) == 0)
{
	T sign_;
	sign_ = signbit(y[idx]); 
	
	if(sign_ != 0)	
		y[idx] = -1 * min(floor((fabs(y[idx])/delta)+(T)0.5),(1/delta));
	else 
		y[idx] = min(floor((fabs(y[idx])/delta)+(T)0.5),(1/delta));
	
	y[idx] = y[idx]*delta;
}
#endif
}
/* Signal quantization kernel for Tanh */
/* IBM check end */



/* IBM check start */
/* Weight quantization kernel */
template<class T>
static __global__ void WeightQuantKernel(const T *x, T *y, const unsigned long n, FLOAT delta,int M)
{
    unsigned long idx;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
	int sign_;
	sign_ = signbit(x[idx]); 
	
	if(sign_ != 0)	
		y[idx] = (T)-1 * min((T)floor((fabs(x[idx])/delta)+(T)0.5),(T)(M-1)/2);
	else 
		y[idx] = min((T)floor((fabs(x[idx])/delta)+(T)0.5),(T)((M-1)/2));
	
	y[idx] = y[idx]*delta;
}
/* Weight quantization kernel */
/* IBM check end */

template<class T>
static __global__ void FuncSoftplusKernel(const T *x, T *y, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    y[idx] = _log<T>((T)1 + _exp<T>(x[idx]));
}


/* IBM check start */
/* Signal quantization kernel for Rectlinear */
/* If the QUANT_RELU flag is on Quantization Model */
template<class T>
static __global__ void FuncRectLinearKernel(const T *x, T *y, T *y_fixed, const unsigned long n,FLOAT delta, int M,int relu_delta_final_decision)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    //y[idx] = max((T)0, x[idx]);
    /* Leaky */
    y[idx] = max((T)0.01 * x[idx], x[idx]);
    y_fixed[idx] = max((T)0.01 * x[idx], x[idx]);
#if QUANT_RELU
if(relu_delta_final_decision == 1)
{
	//if(threadIdx.x == 1 )printf("pre : %f\n",y[idx]);
	
	y[idx] =  min((T)floor((y[idx]/delta)+(T)0.5),(T)(M-1));

        y[idx] = y[idx]*delta;
	//if(threadIdx.x == 1) printf("after : %f\n",y[idx]);
}
	
#endif
}
/* Signal quantization kernel for Rectlinear */
/* IBM check end */


template<class T>
static __global__ void FuncSoftmaxKernel(const T *x, T *y, const unsigned long n)
{
        __shared__ T _v[THREAD_PER_BLOCK];
        T v_tmp, v_max;
        unsigned long i;

        x += blockIdx.x * n;
        y += blockIdx.x * n;


        /* Sequential reduction(max) */
        v_tmp = threadIdx.x < n ? x[threadIdx.x] : (FLOAT) 0;

        #pragma unroll
        for(i = threadIdx.x + blockDim.x; i < n; i += blockDim.x)
        {
                v_tmp = max(v_tmp, x[i]);
        }

    _v[threadIdx.x] = v_tmp;

        __syncthreads();

        /* Parallel reduction(max) */
        i = (blockDim.x >> 1);
        if(threadIdx.x < i)
                v_tmp = _v[threadIdx.x];

        for(; i > 0; i >>= 1)
        {
                if(threadIdx.x < i && threadIdx.x + i < n)
                {
                        v_tmp = max(v_tmp, _v[threadIdx.x + i]);
                        _v[threadIdx.x] = v_tmp;
                }

                __syncthreads();
        }

    v_max = _v[0];

    __syncthreads();

        /* Sequential reduction(+) */
        v_tmp = (T) 0;

        #pragma unroll
        for(i = threadIdx.x; i < n; i += blockDim.x)
        {
                v_tmp += _exp<T>(x[i] - v_max);
        }

    _v[threadIdx.x] = v_tmp;

        __syncthreads();

        /* Parallel reduction(+) */
        i = (blockDim.x >> 1);
        if(threadIdx.x < i)
                v_tmp = _v[threadIdx.x];

        for(; i > 0; i >>= 1)
        {
                if(threadIdx.x < i)
                {
                        v_tmp += _v[threadIdx.x + i];
                        _v[threadIdx.x] = v_tmp;
                }

                __syncthreads();
        }


    /* Update */
        v_tmp = _v[0];

        #pragma unroll
        for(i = threadIdx.x; i < n; i += blockDim.x)
        {
                y[i] = _exp<T>(x[i] - v_max) / v_tmp;
        }
}


template<class T>
static __global__ void FuncBoundRangeKernel(const T *x, T *y, const T _min, const T _max, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    y[idx] = min(_max, max(_min, x[idx]));
}


template<class T>
static __global__ void FuncSigmoidDerivKernel(const T *x, T *y, const unsigned long n)
{
    unsigned long idx;
    T v;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    v = x[idx];
    y[idx] = v * ((T)1 - v);
}


template<class T>
static __global__ void FuncTanhDerivKernel(const T *x, T *y, const unsigned long n)
{
    unsigned long idx;
    T v;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    v = x[idx];
    y[idx] = ((T)1 - v) * ((T)1 + v);
}


template<class T>
static __global__ void FuncSoftplusDerivKernel(const T *x, T *y, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    y[idx] = (T)1 - _exp<T>(-x[idx]);
}


template<class T>
static __global__ void FuncRectLinearDerivKernel(const T *x, T *y, const unsigned long n)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    //y[idx] = (T)(x[idx] > (T)0);
    /* Leaky */
    y[idx] = (T)0.01 + (T)0.99 * (T)(x[idx] > (T)0);
}

    template<class T>
static __global__ void GenerateDropoutMaskKernel(T *mask, const T *uniformDist, const unsigned long n, const T dropoutRate)
{
    unsigned long idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    mask[idx] = (T)(uniformDist[idx] >= dropoutRate) / ((T)1 - dropoutRate);
}

template<class T>
static __global__ void RmspropKernel(T *newDerivs, const T *derivs, T *msDeriv, const T decayRate, const unsigned long n)
{
    unsigned long idx;
    T ms, rms, deriv;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    ms = msDeriv[idx];
    deriv = derivs[idx];

    T bound = _sqrt<T>((T)1 / ((T)1 - decayRate));

    ms = decayRate * ms + ((T)1 - decayRate) * deriv * deriv;
    rms = _sqrt<T>(ms) + (T)1e-20;

    newDerivs[idx] = min(bound, max(-bound, deriv / rms));
    msDeriv[idx] = ms;
}


template<class T>
static __global__ void AdadeltaKernel(T *deltas, const T *derivs, T *msDeriv, T *msDelta, const T learningRate, const T decayRate, const unsigned long n)
{
    unsigned long idx;
    T _msDelta, rmsDelta;
    T _msDeriv, rmsDeriv;
    T deriv, delta;

    const T bound = (T)10;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    _msDeriv = msDeriv[idx];
    _msDelta = msDelta[idx];
    deriv = derivs[idx];

    _msDeriv = decayRate * _msDeriv + ((T)1 - decayRate) * deriv * deriv;
    rmsDeriv = _sqrt<T>(_msDeriv) + (T)1e-20;

    rmsDelta = _sqrt<T>(_msDelta + learningRate * learningRate);

    delta = rmsDelta * min(bound, max(-bound, deriv / rmsDeriv));

    _msDelta = decayRate * _msDelta + ((T)1 - decayRate) * delta * delta;

    deltas[idx] = delta;
    msDeriv[idx] = _msDeriv;
    msDelta[idx] = _msDelta;
}


template<class T>
void MemSet(T *_x, const T val, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MemSetKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, val, n);
}


template<class T>
void ElemMult(const T *_x, const T *_y, T *_z, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    ElemMultKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, _z, n);
}


template<class T>
void Add(const T *_x, const T *_y, T *_z, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    AddKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, _z, n);
}


/* IBM check start */
/* Signal quantization kernel call for Sigmoid */
template<class T>
void FuncSigmoid(const T *_x, T *_y, T *_y_fixed, const unsigned long n, const cudaStream_t stream,FLOAT delta)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSigmoidKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, _y_fixed, n, delta);
}
/* Signal quantization kernel for Sigmoid */
/* IBM check end */

/* IBM check start */
/* Signal quantization kernel call for Tanh */
template<class T>
void FuncTanh(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream,FLOAT delta)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncTanhKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n, delta);
}
/* Signal quantization kernel for Tanh */
/* IBM check end */

/* IBM check start */
/* Weight quantization kernel call */
template<class T>
void WeightQuant(const T *_x, T *_y, const unsigned long n,const cudaStream_t stream, FLOAT delta, int M)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    WeightQuantKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n, delta, M);
}
/* Weight quantization kernel call */
/* IBM check end */

template<class T>
void FuncSoftplus(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSoftplusKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n);
}


/* IBM check start */
/* Signal quantization kernel call for Rectlinear*/
template<class T>
void FuncRectLinear(const T *_x, T *_y, T *_y_fixed,const unsigned long n, const cudaStream_t stream, FLOAT delta, int M,int relu_delta_final_decision)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncRectLinearKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, _y_fixed,n,delta,M,relu_delta_final_decision);
}
/* Signal quantization kernel call for rectlinear*/
/* IBM check end */


template<class T>
void FuncSoftmax(const T *_x, T *_y, const unsigned long layerSize, const unsigned long batchSize, const cudaStream_t stream)
{
    dim3 dimGrid(batchSize);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSoftmaxKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, layerSize);
}


template<class T>
void FuncBoundRange(const T *_x, T *_y, const T min, const T max, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncBoundRangeKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, min, max, n);
}


template<class T>
void FuncSigmoidDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSigmoidDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n);
}


template<class T>
void FuncTanhDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncTanhDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n);
}


template<class T>
void FuncSoftplusDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSoftplusDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n);
}


template<class T>
void FuncRectLinearDeriv(const T *_x, T *_y, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncRectLinearDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, _y, n);
}

    template<class T>
void GenerateDropoutMask(T *_mask, const T *_uniformDist, const unsigned long n, const T dropoutRate, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    GenerateDropoutMaskKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_mask, _uniformDist, n, dropoutRate);
}


template<class T>
void Rmsprop(T *_newDerivs, const T *_derivs, T *_msDeriv, const T decayRate, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    RmspropKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_newDerivs, _derivs, _msDeriv, decayRate, n);
}


template<class T>
void Adadelta(T *_deltas, const T *_derivs, T *_msDeriv, T *_msDelta, const T learningRate, const T decayRate, const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    AdadeltaKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_deltas, _derivs, _msDeriv, _msDelta, learningRate, decayRate, n);
}


template void MemSet<float>(float *_x, const float val, const unsigned long n, const cudaStream_t stream);
template void MemSet<double>(double *_x, const double val, const unsigned long n, const cudaStream_t stream);

template void ElemMult<float>(const float *_x, const float *_y, float *_z, const unsigned long n, const cudaStream_t stream);
template void ElemMult<double>(const double *_x, const double *_y, double *_z, const unsigned long n, const cudaStream_t stream);

template void Add<float>(const float *_x, const float *_y, float *_z, const unsigned long n, const cudaStream_t stream);
template void Add<double>(const double *_x, const double *_y, double *_z, const unsigned long n, const cudaStream_t stream);

/* IBM check start */
template void FuncSigmoid<float>(const float *_x, float *_y, float *_y_fixed,const unsigned long n, const cudaStream_t stream, FLOAT delta);
template void FuncSigmoid<double>(const double *_x, double *_y, double *_y_fixed, const unsigned long n, const cudaStream_t stream, FLOAT delta);

template void FuncTanh<float>(const float *_x, float *_y, const unsigned long n, const cudaStream_t stream,FLOAT delta);
template void FuncTanh<double>(const double *_x, double *_y, const unsigned long n, const cudaStream_t stream,FLOAT delta);

template void WeightQuant<float>(const float *_x, float *_y, const unsigned long n,const cudaStream_t stream, FLOAT delta, int M);
template void WeightQuant<double>(const double *_x, double *_y, const unsigned long n,const cudaStream_t stream, FLOAT delta, int M);

template void FuncRectLinear<float>(const float *_x, float *_y, float *_y_fixed,const unsigned long n, const cudaStream_t stream, FLOAT delta,int M,int relu_delta_final_decision);
template void FuncRectLinear<double>(const double *_x, double *_y, double *_y_fixed, const unsigned long n, const cudaStream_t stream, FLOAT delta, int M,int relu_delta_final_decision);
/* IBM check end */

template void FuncSoftplus<float>(const float *_x, float *_y, const unsigned long n, const cudaStream_t stream);
template void FuncSoftplus<double>(const double *_x, double *_y, const unsigned long n, const cudaStream_t stream);


template void FuncSoftmax<float>(const float *_x, float *_y, const unsigned long layerSize, const unsigned long batchSize, const cudaStream_t stream);
template void FuncSoftmax<double>(const double *_x, double *_y, const unsigned long layerSize, const unsigned long batchSize, const cudaStream_t stream);

template void FuncBoundRange<float>(const float *_x, float *_y, const float min, const float max, const unsigned long n, const cudaStream_t stream);
template void FuncBoundRange<double>(const double *_x, double *_y, const double min, const double max, const unsigned long n, const cudaStream_t stream);

template void FuncSigmoidDeriv<float>(const float *_x, float *_y, const unsigned long n, const cudaStream_t stream);
template void FuncSigmoidDeriv<double>(const double *_x, double *_y, const unsigned long n, const cudaStream_t stream);

template void FuncTanhDeriv<float>(const float *_x, float *_y, const unsigned long n, const cudaStream_t stream);
template void FuncTanhDeriv<double>(const double *_x, double *_y, const unsigned long n, const cudaStream_t stream);

template void FuncSoftplusDeriv<float>(const float *_x, float *_y, const unsigned long n, const cudaStream_t stream);
template void FuncSoftplusDeriv<double>(const double *_x, double *_y, const unsigned long n, const cudaStream_t stream);

template void FuncRectLinearDeriv<float>(const float *_x, float *_y, const unsigned long n, const cudaStream_t stream);
template void FuncRectLinearDeriv<double>(const double *_x, double *_y, const unsigned long n, const cudaStream_t stream);

template void GenerateDropoutMask<float>(float *_mask, const float *_uniformDist, const unsigned long n, const float dropoutRate, const cudaStream_t stream);
template void GenerateDropoutMask<double>(double *_mask, const double *_uniformDist, const unsigned long n, const double dropoutRate, const cudaStream_t stream);

template void Rmsprop<float>(float *_newDerivs, const float *_derivs, float *_msDeriv, const float decayRate, const unsigned long n, const cudaStream_t stream);
template void Rmsprop<double>(double *_newDerivs, const double *_derivs, double *_msDeriv, const double decayRate, const unsigned long n, const cudaStream_t stream);

template void Adadelta<float>(float *_deltas, const float *_derivs, float *_msDeriv, float *_msDelta, const float learningRate, const float decayRate, const unsigned long n, const cudaStream_t stream);
template void Adadelta<double>(double *_deltas, const double *_derivs, double *_msDeriv, double *_msDelta, const double learningRate, const double decayRate, const unsigned long n, const cudaStream_t stream);

}

}

