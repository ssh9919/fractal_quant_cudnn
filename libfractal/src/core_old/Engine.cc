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


#include "Engine.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef FRACTAL_USE_CUDA

#include "CudaKernels.h"
#include <iostream>
#include <fstream>
#include <sstream>

#define FatalError(s) {                                                \
    std::stringstream _message;                                \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cout << _message.str() << "\nAborting...\n";                  \
}   

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if(status != CUDNN_STATUS_SUCCESS) {                              \
        _error << "CUDNN failure: " << status;\
        FatalError(_error.str());\
    }\
}

#define CUDA_CHUNK_SIZE (2 * sizeof(FLOAT)) /* In bytes. curandGenerateNormal requires even number of elements */

#ifdef FRACTAL_DOUBLE_PRECISION
    #define GEAM cublasDgeam
    #define GEMV cublasDgemv
    #define GEMM cublasDgemm
    #define AXPY cublasDaxpy
    #define COPY cublasDcopy
    #define RANDN curandGenerateNormalDouble
#elif defined(FRACTAL_SINGLE_PRECISION)
    #define GEAM cublasSgeam
    #define GEMV cublasSgemv
    #define GEMM cublasSgemm
    #define AXPY cublasSaxpy
    #define COPY cublasScopy
    #define RANDN curandGenerateNormal
#endif /* FRACTAL_DOUBLE_PRECISION */

#else /* FRACTAL_USE_CUDA */

#include <cstdlib>
#include <cstring>

#endif /* FRACTAL_USE_CUDA */


#ifdef FRACTAL_USE_ATLAS
extern "C"
{
#include <cblas.h>
}
#endif /* FRACTAL_USE_ATLAS */


namespace fractal
{

Engine::Engine()
{
    memCount = 0;
    memAllocCount = 0;
    hostLoc = 0;
    eventCount = 0;
    streamCount = 0;

#ifdef FRACTAL_USE_CUDA
    numLoc = 2;

    /* Initialize CUBLAS and CURAND */
    verify(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);

    verify(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS);
    verify(curandSetPseudoRandomGeneratorSeed(curandGen, 0) == CURAND_STATUS_SUCCESS);
        dataType = CUDNN_DATA_FLOAT;
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();
#else
    numLoc = 1;
#endif /* FRACTAL_USE_CUDA */
}


Engine::~Engine()
{
    verify(memCount == 0);
    verify(memAllocCount == 0);
    verify(eventCount == 0);
    verify(streamCount == 0);

#ifdef FRACTAL_USE_CUDA
    cudaThreadSynchronize();

    verify(curandDestroyGenerator(curandGen) == CURAND_STATUS_SUCCESS);
    verify(cublasDestroy(cublasHandle) == CUBLAS_STATUS_SUCCESS);
    verify(cudaGetLastError() == cudaSuccess);
    destroyHandles();
#endif /* FRACTAL_USE_CUDA */
}


void Engine::createHandles()
{
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&srcpoolTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstpoolTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
	StreamCreate(stream_host,0);
}
void Engine::destroyHandles()
{
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(srcpoolTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstpoolTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    checkCUDNN(cudnnDestroy(cudnnHandle));
	StreamDestroy(stream_host);
}

void Engine::ConvForward(Matrix<FLOAT> &_prevLayerAct, Matrix<FLOAT> &_nextLayerState, Matrix<FLOAT> &_weight, long kernelDimX, long kernelDimY, long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps, long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps, long curBatchSize,PStream &stream )
{
    verify(kernelDimX == kernelDimY);
    verify(_prevLayerAct.GetEngine() == this);
    verify(_nextLayerState.GetEngine() == this);
    verify(_weight.GetEngine() == this);

    Mem *memprevAct, *memnextState, *memweight;
    FLOAT *ptrprevAct, *ptrnextState, *ptrweight;
    unsigned long loc;
    float alpha = 1.f;
    float beta = 0.f;

    verify((memprevAct = _prevLayerAct.GetMem()) != NULL);
    verify((memnextState = _nextLayerState.GetMem()) != NULL);
    verify((memweight = _weight.GetMem()) != NULL);

    loc = stream.loc;

    memprevAct->Pull(loc, stream);
    memweight->Pull(loc, stream);

    if(_nextLayerState.GetNumRows()*_nextLayerState.GetNumCols() * sizeof(FLOAT)<memnextState->GetSize())
        memnextState->Pull(loc,stream);
    else
        MemAlloc(memnextState, loc);

    ptrprevAct = (FLOAT *)memprevAct->GetPtr(loc) + _prevLayerAct.GetOffset();
    ptrweight = (FLOAT *)memweight->GetPtr(loc) + _weight.GetOffset();
    ptrnextState = (FLOAT *)memnextState->GetPtr(loc) + _nextLayerState.GetOffset();

    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,curBatchSize,prevLayerNumMaps,prevLayerDimY,prevLayerDimX));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,curBatchSize,nextLayerNumMaps,nextLayerDimY,nextLayerDimX));


    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,dataType,nextLayerNumMaps,prevLayerNumMaps,kernelDimY,kernelDimX));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,CUDNN_CONVOLUTION));

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,0,&algo));
    size_t sizeInBytes = 0;
    void* workSpace = NULL;
    
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,algo,&sizeInBytes));

         // if(sizeInBytes!=0)
         // {
         //         checkCudaErrors(cudaMalloc(&workSpace,sizeInBytes));
         // }
/*FLOAT result_backward[28*28];// new FLOAT* (100);//(FLOAT*)malloc(100*sizeof(FLOAT));
cudaMemcpy(result_backward,ptrprevAct,28*28*sizeof(FLOAT),cudaMemcpyDeviceToHost);
cudaThreadSynchronize();
static int first_data = 1;
if(first_data==1){
    printf("\n");
    for(int i = 0 ; i< 28*28;i++)
    {
        printf(" %.4f",result_backward[i]);
        if(i%28 == 0)printf("\n");
    }
}*/
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,&alpha,srcTensorDesc,ptrprevAct,filterDesc,ptrweight,convDesc,algo,workSpace,sizeInBytes,&beta,dstTensorDesc,ptrnextState));
/*cudaThreadSynchronize();
cudaMemcpy(result_backward,ptrnextState,28*28*sizeof(FLOAT),cudaMemcpyDeviceToHost);
cudaThreadSynchronize();
if(first_data == 1)
{
        printf("\n\n\n");
    for(int i = 0 ; i< 28*28;i++){
    printf(" %.4f",result_backward[i]);
        if(i%28==0)printf("\n");
}
}

cudaMemcpy(result_backward,ptrweight,1*1*sizeof(FLOAT),cudaMemcpyDeviceToHost);
cudaThreadSynchronize();
if(first_data == 1)
{
        printf("kernel\n\n\n");
    for(int i = 0 ; i< 1*1;i++){
    printf(" %.4f",result_backward[i]);
        if(i%1==0)printf("\n");
}
    first_data = 0;
}
//if(sizeInBytes!=0)
    //{
    //      checkCudaErrors(cudaFree(workSpace));
    //}
*/
    memnextState->Push(loc);

}

void Engine::ConvBackward(Matrix<FLOAT> &_prevLayerAct, Matrix<FLOAT> &_prevLayerErr, Matrix<FLOAT> &_nextLayerErr, Matrix<FLOAT> &_weight, Matrix<FLOAT> &_deriv, long kernelDimX, long
        kernelDimY, long prevLayerDimX, long prevLayerDimY, long
        prevLayerNumMaps, long nextLayerDimX, long nextLayerDimY, long
        nextLayerNumMaps, bool performBackwardProp, long curBatchSize,PStream &stream 
        )
{

    verify(_prevLayerAct.GetEngine() == this);
    verify(_prevLayerErr.GetEngine() == this);
    verify(_nextLayerErr.GetEngine() == this);
    verify(_weight.GetEngine() == this);
    verify(_deriv.GetEngine() == this);

    //verify((transA == false ? A.GetNumCols() : A.GetNumRows()) == (transB == false ? B.GetNumRows() : B.GetNumCols()));
    //verify(C.GetNumRows() == (transA == false ? A.GetNumRows() : A.GetNumCols()));
    //verify(C.GetNumCols() == (transB == false ? B.GetNumCols() : B.GetNumRows()));

    Mem *memprevAct, *memprevErr, *memnextErr,*memweight,*memderiv;
    FLOAT *ptrprevAct, *ptrprevErr, *ptrnextErr, *ptrweight, *ptrderiv;
    unsigned long loc;

    //MatSet(_deriv, (FLOAT) 0, stream);
    verify((memprevAct = _prevLayerAct.GetMem()) != NULL);
    verify((memprevErr = _prevLayerErr.GetMem()) != NULL);
    verify((memnextErr = _nextLayerErr.GetMem()) != NULL);
    verify((memderiv = _deriv.GetMem()) != NULL);
    verify((memweight = _weight.GetMem()) != NULL);

    loc = stream.loc;

    if(performBackwardProp == true)
    {
        memnextErr->Pull(loc, stream);
        memweight->Pull(loc, stream);
        if(_prevLayerErr.GetNumRows()*_prevLayerErr.GetNumCols() * sizeof(FLOAT)<memprevErr->GetSize())
            memprevErr->Pull(loc,stream);
        else
            MemAlloc(memprevErr, loc);
    }
    else
    {
        memnextErr->Pull(loc, stream);
        memprevAct->Pull(loc, stream);
        if(_deriv.GetNumRows()*_deriv.GetNumCols() * sizeof(FLOAT)<memderiv->GetSize())
            memderiv->Pull(loc,stream);
        else
            MemAlloc(memderiv, loc);
    }

    

    ptrprevAct = (FLOAT *)memprevAct->GetPtr(loc) + _prevLayerAct.GetOffset();
    ptrprevErr = (FLOAT *)memprevErr->GetPtr(loc) + _prevLayerErr.GetOffset();
    ptrnextErr = (FLOAT *)memnextErr->GetPtr(loc) + _nextLayerErr.GetOffset();
    ptrweight = (FLOAT *)memweight->GetPtr(loc) + _weight.GetOffset();
    ptrderiv = (FLOAT *)memderiv->GetPtr(loc) + _deriv.GetOffset();


    //const FLOAT* _prevLayerAct_d = _prevLayerAct;
    //const FLOAT* _nextLayerErr_d = _nextLayerErr;
    //const FLOAT* _weight_d = _weight;

    //cudnnTensorDescriptor_t srcTensorDesc_d = srcTensorDesc;
    //cudnnTensorDescriptor_t dstTensorDesc_d = dstTensorDesc;

    //cudnnFilterDescriptor_t filterDesc_d = filterDesc;
    //cudnnConvolutionDescriptor_t convDesc_d = convDesc;

    float alpha = 1.f;
    float beta = 0.f;

    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,curBatchSize,nextLayerNumMaps,nextLayerDimY,nextLayerDimX));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,curBatchSize,prevLayerNumMaps,prevLayerDimY,prevLayerDimX));

    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,dataType,nextLayerNumMaps,prevLayerNumMaps,kernelDimY,kernelDimX));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,CUDNN_CONVOLUTION));
     //   FLOAT result_backward[100];
     //   cudaMemcpy(result_backward,ptrprevErr,100*sizeof(FLOAT),cudaMemcpyDeviceToHost);
     //   for(int i = 0 ; i< 100;i++) printf("pre : %d:%f\n",i,result_backward[i]);

    if(performBackwardProp == true)
    {
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,&alpha,filterDesc,ptrweight,srcTensorDesc,ptrnextErr,convDesc,&beta,dstTensorDesc,ptrprevErr));
    memprevErr->Push(loc);

    }
       // cudaMemcpy(result_backward,ptrprevErr,100*sizeof(FLOAT),cudaMemcpyDeviceToHost);
       // for(int i = 0 ; i< 100;i++) printf("after : %d:%f\n",i,result_backward[i]);

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,curBatchSize,prevLayerNumMaps,prevLayerDimY,prevLayerDimX));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,curBatchSize,nextLayerNumMaps,nextLayerDimY,nextLayerDimX));

    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,dataType,nextLayerNumMaps,prevLayerNumMaps,kernelDimY,kernelDimX));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,CUDNN_CONVOLUTION));

/*float result_backward[28*28];// new FLOAT* (100);//(FLOAT*)malloc(100*sizeof(FLOAT));
cudaMemcpy(result_backward,ptrnextErr,24*24*sizeof(FLOAT),cudaMemcpyDeviceToHost);

cudaThreadSynchronize();
FILE * fp_a,*fp_b,*fp_c;
if(first_data==0){

fp_a = fopen("./next_err.dat","w");
    //printf("\n");
    for(int i = 0 ; i< 24*24;i++)
    {
        //if(i!=0 && i%24 == 0)printf("\n");
        fprintf(fp_a,"%f ",result_backward[i]);
    }
fclose(fp_a);
}*/
    if(performBackwardProp == false)
    {
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,&alpha,srcTensorDesc,ptrprevAct,dstTensorDesc,ptrnextErr,convDesc,&beta,filterDesc,ptrderiv));
    memderiv->Push(loc);
    }
/*cudaThreadSynchronize();
cudaMemcpy(result_backward,ptrprevAct,28*28*sizeof(FLOAT),cudaMemcpyDeviceToHost);
cudaThreadSynchronize();

if(first_data == 0)
{
fp_b = fopen("./prev_act.dat","w");
 //       printf("\n\n\n");
    for(int i = 0 ; i< 28*28;i++){
        //if( i!=0 && i%28==0)printf("\n");
    fprintf(fp_b,"%f ",result_backward[i]);
}
fclose(fp_b);
}
cudaMemcpy(result_backward,ptrderiv,5*5*sizeof(FLOAT),cudaMemcpyDeviceToHost);
cudaThreadSynchronize();

if(first_data == 0)
{
fp_c = fopen("./kernel.dat","w");
       // printf("kernel\n\n\n");
    for(int i = 0 ; i< 5*5;i++){
        //if(i!=0 && i%5==0)printf("\n");
        fprintf(fp_c,"%f ",result_backward[i]);
    }
    first_data = 1;
fclose(fp_c);
}*/
}

void Engine::ConvBiasForward(Matrix<FLOAT> &_nextLayerState, Matrix<FLOAT> &_biases, long nextLayerDimY, long nextLayerDimX, long nextLayerNumMaps, long curBatchSize, PStream &stream)
{
    verify(_biases.GetEngine() == this);
    verify(_nextLayerState.GetEngine() == this);

    Mem *memnextState, *membiases;
    FLOAT *ptrnextState, *ptrbiases;
    unsigned long loc;

    float alpha = 1.f;
    float beta = 0.f;
    
    verify((memnextState = _nextLayerState.GetMem()) != NULL);
    verify((membiases = _biases.GetMem()) != NULL);

    loc = stream.loc;

    //memnextState->Pull(loc, stream);
    membiases->Pull(loc, stream);

    if(_nextLayerState.GetNumRows()*_nextLayerState.GetNumCols() * sizeof(FLOAT)<memnextState->GetSize())
      memnextState->Pull(loc,stream);
    //else
    MemAlloc(memnextState, loc);

    ptrnextState = (FLOAT *)memnextState->GetPtr(loc) + _nextLayerState.GetOffset();
    ptrbiases = (FLOAT *)membiases->GetPtr(loc) + _biases.GetOffset();
    //add cudnn code   
     
    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    //checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,tensorFormat,dataType,curBatchSize,prevLayerNumMaps,prevLayerDimY,prevLayerDimX));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,tensorFormat,dataType,1,nextLayerNumMaps,1,1));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,curBatchSize,nextLayerNumMaps,nextLayerDimY,nextLayerDimX));

    checkCUDNN(cudnnAddTensor(cudnnHandle,CUDNN_ADD_SAME_C,&alpha,biasTensorDesc,ptrbiases,&beta,dstTensorDesc, ptrnextState));


    memnextState->Push(loc);
}


void Engine::ConvBiasBackward(Matrix<FLOAT> &_nextLayerErr, Matrix<FLOAT> &_deriv, long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps,long curBatchSize, PStream &stream)    
{                       
    verify(_deriv.GetEngine() == this);
    verify(_nextLayerErr.GetEngine() == this);

    Mem *memnextErr, *memderiv;
    FLOAT *ptrnextErr, *ptrderiv;
    unsigned long loc;
    float alpha = 1.f;
    float beta = 0.f;

    verify((memnextErr = _nextLayerErr.GetMem()) != NULL);
    verify((memderiv = _deriv.GetMem()) != NULL);

    loc = stream.loc;

    memnextErr->Pull(loc, stream);
    //memderiv->Pull(loc, stream);

    if(_deriv.GetNumRows()*_deriv.GetNumCols() * sizeof(FLOAT)<memderiv->GetSize())
        memderiv->Pull(loc,stream);
    else
        MemAlloc(memderiv, loc);

    ptrnextErr = (FLOAT *)memnextErr->GetPtr(loc) + _nextLayerErr.GetOffset();
    ptrderiv = (FLOAT *)memderiv->GetPtr(loc) + _deriv.GetOffset();
    
    //MatSet(_deriv, (FLOAT) 0, stream);
    
    
    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,curBatchSize,nextLayerNumMaps,nextLayerDimY,nextLayerDimX));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,1,nextLayerNumMaps,1,1));

    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle,&alpha,srcTensorDesc,ptrnextErr,&beta,dstTensorDesc, ptrderiv));

    memderiv->Push(loc);
}
#if 0 //sajid mode
void Engine::ConvBiasForward(Matrix<FLOAT> &_nextLayerState, Matrix<FLOAT> &_biases, long nextLayerDimY, long nextLayerDimX, long nextLayerNumMaps, long curBatchSize, PStream &stream)
{
    verify(_biases.GetEngine() == this);
    verify(_nextLayerState.GetEngine() == this);

    Mem *memnextState, *membiases;
    FLOAT *ptrnextState, *ptrbiases;
    unsigned long loc;

    verify((memnextState = _nextLayerState.GetMem()) != NULL);
    verify((membiases = _biases.GetMem()) != NULL);

    //memnextState->Pull(loc, stream);
    membiases->Pull(loc, stream);

    //if(_nextLayerState.GetNumRows()*_nextLayerState.GetNumCols() * sizeof(FLOAT)<memnextState->GetSize())
    //  memnextState->Pull(loc,stream);
    //else
    MemAlloc(memnextState, loc);

    ptrnextState = (FLOAT *)memnextState->GetPtr(loc) + _nextLayerState.GetOffset();
    ptrbiases = (FLOAT *)membiases->GetPtr(loc) + _biases.GetOffset();
       
#ifdef FRACTAL_USE_CUDA
        cudaKernels::conv_bias_forward<FLOAT>(ptrnextState, ptrbiases, nextLayerDimY, nextLayerDimX,nextLayerNumMaps,curBatchSize, stream.cudaStream);
#else
	verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */
    
    memnextState->Push(loc);
}


void Engine::ConvBiasBackward(Matrix<FLOAT> &_nextLayerErr, Matrix<FLOAT> &_deriv, long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps, long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps,long curBatchSize, PStream &stream)    
{                       
    verify(_deriv.GetEngine() == this);
    verify(_nextLayerErr.GetEngine() == this);

    Mem *memnextErr, *memderiv;
    FLOAT *ptrnextErr, *ptrderiv;
    unsigned long loc;
    float alpha = 1.f;
    float beta = 0.f;

    verify((memnextErr = _nextLayerErr.GetMem()) != NULL);
    verify((memderiv = _deriv.GetMem()) != NULL);

    loc = stream.loc;

    memnextErr->Pull(loc, stream);
    //memderiv->Pull(loc, stream);

    if(_deriv.GetNumRows()*_deriv.GetNumCols() * sizeof(FLOAT)<memderiv->GetSize())
        memderiv->Pull(loc,stream);
    else
        MemAlloc(memderiv, loc);

    ptrnextErr = (FLOAT *)memnextErr->GetPtr(loc) + _nextLayerErr.GetOffset();
    ptrderiv = (FLOAT *)memderiv->GetPtr(loc) + _deriv.GetOffset();
    
    MatSet(_deriv, (FLOAT) 0, stream);
    
/*    
    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,curBatchSize,prevLayerNumMaps,prevLayerDimY,prevLayerDimX));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,curBatchSize,nextLayerNumMaps,nextLayerDimY,nextLayerDimX));

    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle,&alpha,srcTensorDesc,ptrnextErr,&beta,dstTensorDesc, ptrderiv));
*/
#ifdef FRACTAL_USE_CUDA
        cudaKernels::conv_bias_backward_compute_deriv<FLOAT>(ptrnextErr, ptrderiv, nextLayerDimY, nextLayerDimX, nextLayerNumMaps,curBatchSize, stream.cudaStream);
#else
	verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */
    memderiv->Push(loc);
}
#endif


void Engine::MaxPoolForward(Matrix<FLOAT> &_prevLayerAct, Matrix<FLOAT> &_nextLayerState,
        long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps,
        long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps,
        long curBatchSize, PStream &stream )
{
    verify(_prevLayerAct.GetEngine() == this);
    verify(_nextLayerState.GetEngine() == this);

    Mem *memprevAct, *memnextState;
    FLOAT *ptrprevAct, *ptrnextState;
    unsigned long loc;
    float alpha = 1.f;
    float beta = 0.f;

    verify((memprevAct = _prevLayerAct.GetMem()) != NULL);
    verify((memnextState = _nextLayerState.GetMem()) != NULL);

    loc = stream.loc;

    memprevAct->Pull(loc, stream);

    MemAlloc(memnextState, loc);

    ptrprevAct = (FLOAT *)memprevAct->GetPtr(loc) + _prevLayerAct.GetOffset();
    ptrnextState = (FLOAT *)memnextState->GetPtr(loc) + _nextLayerState.GetOffset();

    
    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc,
                CUDNN_POOLING_MAX,
                2, 2, // window
                0, 0, // padding
                2, 2  // stride
                ) );
    
    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                tensorFormat,
                dataType,
                curBatchSize, prevLayerNumMaps,
                prevLayerDimY,
                prevLayerDimX));
    
    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                tensorFormat,
                dataType,
                curBatchSize, nextLayerNumMaps,
                nextLayerDimY,
                nextLayerDimX));
    
    checkCUDNN( cudnnPoolingForward(cudnnHandle,
                poolingDesc,
                &alpha,
                srcTensorDesc,
                ptrprevAct,
                &beta,
                dstTensorDesc,
                ptrnextState));

        
        memnextState->Push(loc);


}


void Engine::MaxPoolBackward(Matrix<FLOAT> &_prevLayerAct, Matrix<FLOAT> &_nextLayerState,Matrix<FLOAT> &_prevLayerErr, Matrix<FLOAT> &_nextLayerErr,
        long prevLayerDimX, long prevLayerDimY, long prevLayerNumMaps,
        long nextLayerDimX, long nextLayerDimY, long nextLayerNumMaps,
        long curBatchSize, PStream &stream )
{
    verify(_prevLayerAct.GetEngine() == this);
    verify(_prevLayerErr.GetEngine() == this);
    verify(_nextLayerErr.GetEngine() == this);
    verify(_nextLayerState.GetEngine() == this);

    //verify((transA == false ? A.GetNumCols() : A.GetNumRows()) == (transB == false ? B.GetNumRows() : B.GetNumCols()));
    //verify(C.GetNumRows() == (transA == false ? A.GetNumRows() : A.GetNumCols()));
    //verify(C.GetNumCols() == (transB == false ? B.GetNumCols() : B.GetNumRows()));

    Mem *memprevAct, *memprevErr, *memnextErr,*memnextState;
    FLOAT *ptrprevAct, *ptrprevErr, *ptrnextErr,  *ptrnextState;
    unsigned long loc;

    verify((memprevAct = _prevLayerAct.GetMem()) != NULL);
    verify((memprevErr = _prevLayerErr.GetMem()) != NULL);
    verify((memnextErr = _nextLayerErr.GetMem()) != NULL);
    verify((memnextState = _nextLayerState.GetMem()) != NULL);

    loc = stream.loc;

    memprevAct->Pull(loc, stream);
    memnextState->Pull(loc, stream);
    memnextErr->Pull(loc, stream);

    MemAlloc(memprevErr, loc);

    ptrprevAct = (FLOAT *)memprevAct->GetPtr(loc) + _prevLayerAct.GetOffset();
    ptrnextErr = (FLOAT *)memnextErr->GetPtr(loc) + _nextLayerErr.GetOffset();
    ptrnextState = (FLOAT *)memnextState->GetPtr(loc) + _nextLayerState.GetOffset();
    ptrprevErr = (FLOAT *)memprevErr->GetPtr(loc) + _prevLayerErr.GetOffset();

    float alpha = 1.f;
    float beta = 0.f;
   
    checkCUDNN(cudnnSetStream(cudnnHandle,stream.cudaStream));
    checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc,
                CUDNN_POOLING_MAX,
                2, 2, // window
                0, 0, // padding
                2, 2  // stride
                ) );
    
    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                tensorFormat,
                dataType,
                curBatchSize, nextLayerNumMaps,
                nextLayerDimY,
                nextLayerDimX));
    
    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                tensorFormat,
                dataType,
                curBatchSize, prevLayerNumMaps,
                prevLayerDimY,
                prevLayerDimX));
    
    checkCUDNN( cudnnSetTensor4dDescriptor(srcpoolTensorDesc,
                tensorFormat,
                dataType,
                curBatchSize, nextLayerNumMaps,
                nextLayerDimY,
                nextLayerDimX));
    
    checkCUDNN( cudnnSetTensor4dDescriptor(dstpoolTensorDesc,
                tensorFormat,
                dataType,
                curBatchSize, prevLayerNumMaps,
                prevLayerDimY,
                prevLayerDimX));
    
    checkCUDNN( cudnnPoolingBackward(cudnnHandle,
                poolingDesc,
                &alpha,
                srcTensorDesc,
                ptrnextState,
                srcpoolTensorDesc,
                ptrnextErr,
                dstTensorDesc,
                ptrprevAct,
                &beta,
                dstpoolTensorDesc,
                ptrprevErr));
    
    memprevErr->Push(loc);
}
void Engine::MemAdd(Mem *mem)
{
    mtxMem.lock();

    verify(mem->GetEngine() == this);

    memCount++;

    mtxMem.unlock();
}


void Engine::MemDel(Mem *mem)
{
    mtxMem.lock();

    verify(mem->GetEngine() == this);

    memCount--;

    mtxMem.unlock();
}


void Engine::MemAlloc(Mem *mem, unsigned long loc)
{
    mtxMem.lock();
    mem->Lock();

    verify(mem->GetEngine() == this);

    size_t size;
    void **ptr;

    ptr = mem->GetPtrs();
    if(ptr[loc] != NULL)
    {
        mtxMem.unlock();
        mem->Unlock();
        return;
    }

    size = mem->GetSize();
    verify(size > 0);


#ifdef FRACTAL_USE_CUDA
    if(loc == hostLoc)
    {
        verify(cudaMallocHost(ptr + loc, size) == cudaSuccess);
    }
    else
    {
        size = (size + CUDA_CHUNK_SIZE - 1) / CUDA_CHUNK_SIZE * CUDA_CHUNK_SIZE;
        verify(cudaMalloc(ptr + loc, size) == cudaSuccess);
    }
#else
    verify((ptr[loc] = malloc(size)) != NULL);
#endif /* FRACTAL_USE_CUDA */

    memAllocCount++;

    mtxMem.unlock();
    mem->Unlock();
}


void Engine::MemDealloc(Mem *mem)
{
    mtxMem.lock();
    mem->Lock();

    verify(mem->GetEngine() == this);

    void **ptr;
    unsigned long i;

    ptr = mem->GetPtrs();

    for(i = 0; i < numLoc; i++)
    {
        if(ptr[i] != NULL)
        {
#ifdef FRACTAL_USE_CUDA
            if(i == hostLoc)
            {
                verify(cudaFreeHost(ptr[i]) == cudaSuccess);
            }
            else
            {

                verify(cudaFree(ptr[i]) == cudaSuccess);
            }
#else
            free(mem->GetPtr(0));
#endif /* FRACTAL_USE_CUDA */

            ptr[i] = NULL;
            memAllocCount--;
        }
    }

    mem->Invalidate();

    mtxMem.unlock();
    mem->Unlock();
}


void Engine::MemPull(Mem *mem, const unsigned long loc, PStream &stream)
{
    mtxMem.lock();
    mem->Lock();

    verify(mem->GetEngine() == this);

    unsigned long recent;

    if(mem->IsValid(loc) == true)
    {
        mtxMem.unlock();
        mem->Unlock();
        return;
    }

    recent = mem->GetRecentLoc();

    //verify(mem->IsValid(recent) == true);

    MemAlloc(mem, loc);
    if(mem->IsValid(recent) == true)
        MemCopy(mem, 0, recent, mem, 0, loc, mem->GetSize(), stream);

    mem->Validate(loc);

    mtxMem.unlock();
    mem->Unlock();
}


void Engine::MemCopy(const Mem *memSrc, const size_t offsetSrc, Mem *memDst, const size_t offsetDst, const size_t size, PStream &stream)
{
    verify(memSrc != memDst);
    verify(memSrc->GetEngine() == this);
    verify(memDst->GetEngine() == this);

    unsigned long locSrc, locDst;

    locSrc = memSrc->GetRecentLoc();
    locDst = memDst->GetRecentLoc();
    verify(memSrc->IsValid(locSrc) == true);

    //if(memDst->GetSize() != size) /* Partial copy */
    //		memDst->Pull(locDst);
    //	else
    MemAlloc(memDst, locDst);

    MemCopy(memSrc, offsetSrc, locSrc, memDst, offsetDst, locDst, size, stream);

    memDst->Push(locDst);
}


void Engine::MemCopyFromHost(Mem *memDst, const size_t offsetDst, const void *ptrSrc, const size_t size, PStream &stream)
{
    verify(memDst->GetEngine() == this);

    unsigned long locSrc, locDst;
    void *ptrDst;

    locSrc = GetHostLoc();

    locDst = memDst->GetRecentLoc();
    MemAlloc(memDst, locDst);

    ptrDst = (unsigned char *)memDst->GetPtr(locDst) + offsetDst;
    MemCopy(ptrSrc, locSrc, ptrDst, locDst, size, stream);

    memDst->Push(locDst);
}


void Engine::MemCopyToHost(const Mem *memSrc, const size_t offsetSrc, void *ptrDst, const size_t size, PStream &stream)
{
    verify(memSrc->GetEngine() == this);

    unsigned long locSrc, locDst;
    void *ptrSrc;

    locDst = GetHostLoc();

    locSrc = memSrc->GetRecentLoc();
    verify(memSrc->IsValid(locSrc) == true);

    ptrSrc = (unsigned char *)memSrc->GetPtr(locSrc) + offsetSrc;
    MemCopy(ptrSrc, locSrc, ptrDst, locDst, size, stream);
}


void Engine::MemCopy(const Mem *memSrc, const size_t offsetSrc, const unsigned long locSrc,
        Mem *memDst, const size_t offsetDst, const unsigned long locDst, const size_t size, PStream &stream)
{
    if(memSrc == memDst && locSrc == locDst)
    {
        verify(offsetSrc == offsetDst);
        return;
    }

    void *ptrSrc, *ptrDst;

    ptrSrc = (unsigned char *)memSrc->GetPtr(locSrc) + offsetSrc;
    ptrDst = (unsigned char *)memDst->GetPtr(locDst) + offsetDst;

    MemCopy(ptrSrc, locSrc, ptrDst, locDst, size, stream);
}


void Engine::MemCopy(const void *ptrSrc, const unsigned long locSrc,
        void *ptrDst, const unsigned long locDst, const size_t size, PStream &stream)
{
#ifdef FRACTAL_USE_CUDA
    if(stream.loc == hostLoc)
    {
        verify(cudaMemcpy(ptrDst, ptrSrc, size, cudaMemcpyDefault) == cudaSuccess);
    }
    else
    {
        verify(cudaMemcpyAsync(ptrDst, ptrSrc, size, cudaMemcpyDefault, stream.cudaStream) == cudaSuccess);
    }
#else
    verify(stream.loc == hostLoc);
    memcpy(ptrDst, ptrSrc, size);
#endif /* FRACTAL_USE_CUDA */
}


void Engine::MatMult(Matrix<FLOAT> &A, const bool transA, Matrix<FLOAT> &B, const bool transB, Matrix<FLOAT> &C, const FLOAT alpha, const FLOAT beta, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);

    verify((transA == false ? A.GetNumCols() : A.GetNumRows()) == (transB == false ? B.GetNumRows() : B.GetNumCols()));
    verify(C.GetNumRows() == (transA == false ? A.GetNumRows() : A.GetNumCols()));
    verify(C.GetNumCols() == (transB == false ? B.GetNumCols() : B.GetNumRows()));

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);

    if(beta == (FLOAT) 0)
        ptrC = C.GetPtrForWrite(stream);
    else
        ptrC = C.GetPtrForReadWrite(stream);

    /* TODO: check if loc is host or not */

#ifdef FRACTAL_USE_CUDA
    verify(cublasSetStream(cublasHandle, stream.cudaStream) == CUBLAS_STATUS_SUCCESS);

    if(C.GetNumCols() == 1) /* If C is a vector */
    {
        verify(GEMV(cublasHandle,
                    transA == true ? CUBLAS_OP_T : CUBLAS_OP_N,
                    A.GetNumRows(),
                    A.GetNumCols(),
                    &alpha,
                    ptrA,
                    A.GetNumRows(),
                    ptrB,
                    1,
                    &beta,
                    ptrC,
                    1)
                == CUBLAS_STATUS_SUCCESS);
    }	
    else
    {
        verify(GEMM(cublasHandle,
                    transA == true ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transB == true ? CUBLAS_OP_T : CUBLAS_OP_N,
                    C.GetNumRows(),
                    C.GetNumCols(),
                    transA == true ? A.GetNumRows() : A.GetNumCols(),
                    &alpha,
                    ptrA,
                    A.GetNumRows(),
                    ptrB,
                    B.GetNumRows(),
                    &beta,
                    ptrC,
                    C.GetNumRows())
                == CUBLAS_STATUS_SUCCESS);
    }
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatElemMult(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());
    verify(A.GetNumRows() == C.GetNumRows());
    verify(A.GetNumCols() == C.GetNumCols());

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);
    ptrC = C.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::ElemMult<FLOAT>(ptrA, ptrB, ptrC, A.GetNumRows() * B.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatAdd(Matrix<FLOAT> &A, Matrix<FLOAT> &B, const FLOAT alpha, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());

    FLOAT *ptrA, *ptrB;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    verify(cublasSetStream(cublasHandle, stream.cudaStream) == CUBLAS_STATUS_SUCCESS);

    verify(AXPY(cublasHandle,
                A.GetNumRows() * A.GetNumCols(),
                &alpha,
                ptrA,
                1,
                ptrB,
                1)
            == CUBLAS_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    B.FinishWrite(stream);
}


void Engine::MatAdd(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());
    verify(A.GetNumRows() == C.GetNumRows());
    verify(A.GetNumCols() == C.GetNumCols());

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);
    ptrC = C.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::Add<FLOAT>(ptrA, ptrB, ptrC, A.GetNumRows() * B.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatSet(Matrix<FLOAT> &mat, const FLOAT val, PStream &stream)
{
    verify(mat.GetEngine() == this);

    FLOAT *ptr;

    ptr = mat.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::MemSet<FLOAT>(ptr, val, mat.GetNumRows() * mat.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatRandN(Matrix<FLOAT> &mat, const FLOAT mean, const FLOAT stdev, PStream &stream)
{
    verify(mat.GetEngine() == this);
    verify(stdev >= (FLOAT) 0);

    FLOAT *ptr;

    ptr = mat.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    /* curandGenerateNormal requires even number of elements */
    unsigned long n;
    verify(CUDA_CHUNK_SIZE >= 2 * sizeof(FLOAT));
    n = mat.GetNumRows() * mat.GetNumCols();
    n = (n + 1) / 2 * 2;

    verify(curandSetStream(curandGen, stream.cudaStream) == CURAND_STATUS_SUCCESS);
    verify(RANDN(curandGen, ptr, n, mean, stdev) == CURAND_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatCopy(Matrix<FLOAT> &A, Matrix<FLOAT> &B, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());

    FLOAT *ptrA, *ptrB;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cublasSetStream(cublasHandle, stream.cudaStream);
    verify(COPY(cublasHandle,
                A.GetNumRows() * A.GetNumCols(),
                ptrA,
                1,
                ptrB,
                1)
            == CUBLAS_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    B.FinishWrite(stream);
}


void Engine::MatTranspose(Matrix<FLOAT> &A, Matrix<FLOAT> &B, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumCols());
    verify(A.GetNumCols() == B.GetNumRows());

    FLOAT *ptrA, *ptrB;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    FLOAT alpha = (FLOAT) 1;
    FLOAT beta = (FLOAT) 0;

    cublasSetStream(cublasHandle, stream.cudaStream);
    verify(GEAM(cublasHandle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                B.GetNumRows(),
                B.GetNumCols(),
                &alpha,
                ptrA,
                A.GetNumRows(),
                &beta,
                ptrB,
                B.GetNumRows(),
                ptrB,
                B.GetNumRows())
            == CUBLAS_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    B.FinishWrite(stream);
}


void Engine::FuncSigmoid(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream, FLOAT delta)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForReadWrite(stream);
#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncSigmoid(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream, delta);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */
	Y.FinishWrite(stream);
#if 0//QUANT_DIRECT	
	ptrY = Y.GetPtrForReadWrite(stream_host);
	int bit = 8;// change the stream to steam_host because we have to use cpu address.
	FLOAT delta = 1/(pow(2,bit)-1);
	//printf("size : %d \n",Y.GetNumRows()*Y.GetNumCols());
	//printf("delta : %f \n",delta);
	for(int i = 0;i < Y.GetNumRows()*Y.GetNumCols();i++)
	{
		//printf("before %d : %f ",i,ptrY[i]);
		ptrY[i] = static_cast<FLOAT>(floor((fabs(ptrY[i])/delta)+0.5));	
		ptrY[i] = ptrY[i]*delta;
		//printf("after : %f\n",i,ptrY[i]);
		//static_cast<FLOAT>(floor((fabs(ptrY[i])/delta)+0.5));	
	}
	Y.FinishWrite(stream_host);
#endif
}


void Engine::FuncTanh(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream,FLOAT delta)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncTanh(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream, delta);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncSoftplus(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncSoftplus(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncRectLinear(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream, FLOAT delta, int M,int relu_delta_final_decision)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncRectLinear(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream, delta, M,relu_delta_final_decision);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
#if 0	
	ptrY = Y.GetPtrForReadWrite(stream_host);
	//int bit = 2;// change the stream to steam_host because we have to use cpu address.
	//FLOAT delta_relu = 1/(pow(2,bit)-1);
	//printf("size : %d \n",Y.GetNumRows()*Y.GetNumCols());
	//printf("delta : %f \n",delta);
	for(int i = 0;i < Y.GetNumRows()*Y.GetNumCols();i++)
	{
		//printf("before %d : %f ",i,ptrY[i]);
		ptrY[i] = std::min(static_cast<FLOAT>(floor((fabs(ptrY[i])/delta_relu)+0.5)),static_cast<FLOAT>((M_relu-1)/2));	
		ptrY[i] = ptrY[i]*delta_relu;
		//printf("after : %f\n",i,ptrY[i]);
		//static_cast<FLOAT>(floor((fabs(ptrY[i])/delta)+0.5));	
	}
	Y.FinishWrite(stream);
#endif
}


void Engine::FuncSoftmax(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncSoftmax<FLOAT>(ptrX, ptrY, Y.GetNumRows(), Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncBoundRange(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, const FLOAT min, const FLOAT max, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncBoundRange(ptrX, ptrY, min, max, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif

    Y.FinishWrite(stream);
}


void Engine::FuncSigmoidDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncSigmoidDeriv<FLOAT>(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncTanhDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncTanhDeriv<FLOAT>(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncSoftplusDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncSoftplusDeriv<FLOAT>(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncRectLinearDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::FuncRectLinearDeriv<FLOAT>(ptrX, ptrY, Y.GetNumRows() * Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::Adadelta(Matrix<FLOAT> &deltas, Matrix<FLOAT> &derivs, Matrix<FLOAT> &msDeriv, Matrix<FLOAT> &msDelta, const FLOAT learningRate, const FLOAT decayRate, PStream &stream)
{
    verify(deltas.GetEngine() == this);
    verify(derivs.GetEngine() == this);
    verify(msDeriv.GetEngine() == this);

    verify(derivs.GetNumRows() == deltas.GetNumRows());
    verify(derivs.GetNumCols() == deltas.GetNumCols());
    verify(derivs.GetNumRows() == msDeriv.GetNumRows());
    verify(derivs.GetNumCols() == msDeriv.GetNumCols());
    verify(derivs.GetNumRows() == msDelta.GetNumRows());
    verify(derivs.GetNumCols() == msDelta.GetNumCols());

    FLOAT *ptrDerivs, *ptrDeltas, *ptrMsDeriv, *ptrMsDelta;

    ptrDeltas = deltas.GetPtrForWrite(stream);
    ptrDerivs = derivs.GetPtrForReadWrite(stream);
    ptrMsDeriv = msDeriv.GetPtrForReadWrite(stream);
    ptrMsDelta = msDelta.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::Adadelta<FLOAT>(ptrDeltas, ptrDerivs, ptrMsDeriv, ptrMsDelta, learningRate, decayRate, derivs.GetNumRows() * derivs.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    deltas.FinishWrite(stream);
    msDeriv.FinishWrite(stream);
    msDelta.FinishWrite(stream);
}


void Engine::Rmsprop(Matrix<FLOAT> &newDerivs, Matrix<FLOAT> &derivs, Matrix<FLOAT> &msDeriv, const FLOAT decayRate, PStream &stream)
{
    verify(newDerivs.GetEngine() == this);
    verify(derivs.GetEngine() == this);
    verify(msDeriv.GetEngine() == this);

    verify(derivs.GetNumRows() == newDerivs.GetNumRows());
    verify(derivs.GetNumCols() == newDerivs.GetNumCols());
    verify(derivs.GetNumRows() == msDeriv.GetNumRows());
    verify(derivs.GetNumCols() == msDeriv.GetNumCols());

    FLOAT *ptrDerivs, *ptrNewDerivs, *ptrMsDeriv;

    ptrNewDerivs = derivs.GetPtrForWrite(stream);
    ptrDerivs = derivs.GetPtrForReadWrite(stream);
    ptrMsDeriv = msDeriv.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    cudaKernels::Rmsprop<FLOAT>(ptrNewDerivs, ptrDerivs, ptrMsDeriv, decayRate, derivs.GetNumRows() * derivs.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */


    newDerivs.FinishWrite(stream);
    msDeriv.FinishWrite(stream);
}


void Engine::EventCreate(PEvent &event, const unsigned long loc)
{
    mtxEvent.lock();

    event.engine = this;
    event.loc = loc;

#ifdef FRACTAL_USE_CUDA
    event.cudaStream = NULL;
    verify(cudaEventCreateWithFlags(&event.cudaEvent, cudaEventDisableTiming) == cudaSuccess);
#else
    event.streamId = 0;
#endif /* FRACTAL_USE_CUDA */

    eventCount++;

    mtxEvent.unlock();
}


void Engine::EventDestroy(PEvent &event)
{
    mtxEvent.lock();

    verify(event.engine == this);

#ifdef FRACTAL_USE_CUDA
    verify(cudaEventDestroy(event.cudaEvent) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */

    eventCount--;
    event.engine = NULL;

    mtxEvent.unlock();
}


void Engine::EventRecord(PEvent &event, PStream &stream)
{
    mtxEvent.lock();
    mtxStream.lock();

    verify(event.engine == this);
    verify(stream.engine == this);
    verify(event.loc == stream.loc);

#ifdef FRACTAL_USE_CUDA
    event.cudaStream = stream.cudaStream;
    verify(cudaEventRecord(event.cudaEvent, stream.cudaStream) == cudaSuccess);
#else
    event.streamId = stream.streamId;
#endif /* FRACTAL_USE_CUDA */

    mtxEvent.unlock();
    mtxStream.unlock();
}


void Engine::EventSynchronize(PEvent &event)
{
    mtxEvent.lock();

    verify(event.engine == this);

#ifdef FRACTAL_USE_CUDA
    verify(cudaEventSynchronize(event.cudaEvent) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */

    mtxEvent.unlock();
}


void Engine::StreamCreate(PStream &stream, const unsigned long loc)
{
    mtxStream.lock();
    stream.engine = this;
    stream.loc = loc;

#ifdef FRACTAL_USE_CUDA
#ifdef FRACTAL_CUDA_MULTISTREAM
    if(stream.loc == hostLoc) stream.cudaStream = 0;
    else verify(cudaStreamCreate(&stream.cudaStream) == cudaSuccess);
#else
    stream.cudaStream = 0;
#endif /* FRACTAL_CUDA_MULTISTREAM */
#endif /* FRACTAL_USE_CUDA */

    streamCount++;

    mtxStream.unlock();
}


void Engine::StreamDestroy(PStream &stream)
{
    mtxStream.lock();

    verify(stream.engine == this);

#ifdef FRACTAL_USE_CUDA
#ifdef FRACTAL_CUDA_MULTISTREAM
    if(stream.loc != hostLoc)
        verify(cudaStreamDestroy(stream.cudaStream) == cudaSuccess);
#endif /* FRACTAL_CUDA_MULTISTREAM */
#endif /* FRACTAL_USE_CUDA */

    streamCount--;
    stream.engine = NULL;

    mtxStream.unlock();
}


void Engine::StreamWaitEvent(PStream &stream, PEvent &event)
{
    mtxEvent.lock();
    mtxStream.lock();

    verify(event.engine == this);
    verify(stream.engine == this);

#ifdef FRACTAL_USE_CUDA
    verify(cudaStreamWaitEvent(stream.cudaStream, event.cudaEvent, 0) == cudaSuccess);
#else

#endif /* FRACTAL_USE_CUDA */

    mtxEvent.unlock();
    mtxStream.unlock();
}


void Engine::StreamSynchronize(PStream &stream)
{
#ifdef FRACTAL_USE_CUDA
    mtxStream.lock();

    verify(stream.engine == this);

    cudaStream_t cudaStreamCopy = stream.cudaStream;

    mtxStream.unlock();

    verify(cudaStreamSynchronize(cudaStreamCopy) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */
}


void Engine::SetRandomSeed(unsigned long long seed)
{
#ifdef FRACTAL_USE_CUDA
    verify(curandSetPseudoRandomGeneratorSeed(curandGen, seed) == CURAND_STATUS_SUCCESS);
#else
    verify(false);
#endif /* FRACTAL_USE_CUDA */
}

}

