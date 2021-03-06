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


#include "Matrix.h"

#include <fstream>
#include <iostream>
#include "Mem.h"
#include "Engine.h"


namespace fractal{

template<class T>
Matrix<T>::Matrix(const unsigned long nRows, const unsigned long nCols)
{
    engine = NULL;
    mem = NULL;

    isSub = false;

    offset = 0;

    Resize(nRows, nCols);
}


template<class T>
Matrix<T>::Matrix(Matrix<T> &A, const unsigned long a1, const unsigned long a2)
{
    A.Lock();

    //*this = A;
    mem = A.mem;
    offset = A.offset;
    nRows = A.nRows;
    nCols = A.nCols;
    engine = A.engine;

    isSub = true;

    verify(a2 >= a1);
    verify(a2 < A.nCols);

    nCols = a2 - a1 + 1;
    offset = A.offset + a1 * nRows;

    A.Unlock();
}


template<class T>
Matrix<T>::~Matrix()
{
    Clear();
}


template<class T>
void Matrix<T>::SetEngine(Engine *engine)
{
    Lock();

    Unlink();

    this->engine = engine;

    Malloc();

    Unlock();
}


template<class T>
void Matrix<T>::Resize(const unsigned long nRows, const unsigned long nCols)
{
    Lock();

    verify(nRows >= 0 && nCols >= 0);

    Unlink();

    this->nRows = nRows;
    this->nCols = nCols;

    Malloc();

    Unlock();
}


template<class T>
void Matrix<T>::Malloc()
{
    verify(isSub == false);

    Clear();

    if(engine != NULL && nRows * nCols > 0)
    {
        mem = new Mem(engine, sizeof(T) * nRows * nCols);
    }
}


template<class T>
void Matrix<T>::Clear()
{
    if(isSub == false && mem != NULL)
    {
        delete mem;
    }

    mem = NULL;
}


template<class T>
void Matrix<T>::Link(Matrix<T> &src)
{
    Lock();
    src.Lock();

    if(mem == src.mem)
    {
        Unlock();
        src.Unlock();
        return;
    }

    verify(nRows == src.nRows);
    verify(nCols == src.nCols);

    Clear();

    mem = src.mem;
    offset = src.offset;
    nRows = src.nRows;
    nCols = src.nCols;
    engine = src.engine;

    isSub = true;

    Unlock();
    src.Unlock();
}


template<class T>
void Matrix<T>::Unlink()
{
    Lock();

    if(isSub == false)
    {
        Unlock();
        return;
    }

    Clear();

    isSub = false;

    offset = 0;

    Malloc();

    Unlock();
}


template<class T>
void Matrix<T>::Import(const std::vector<T> &vec, PStream &stream)
{
    verify(vec.size() == nRows * nCols);

    mem->CopyFromHost(offset * sizeof(T), vec.data(), nRows * nCols * sizeof(T), stream);
}


template<class T>
void Matrix<T>::Import(const Matrix<T> &mat, PStream &stream)
{
    verify(mat.nRows == nRows);
    verify(mat.nCols == nCols);
    verify(mat.mem != NULL);
    verify(mem != NULL);
    verify(engine != NULL);

    engine->MemCopy(mat.mem, mat.offset * sizeof(T), mem, offset * sizeof(T), nRows * nCols * sizeof(T), stream);
}


template<class T>
void Matrix<T>::Export(std::vector<T> &vec, PStream &stream) const
{
    verify(vec.size() == nRows * nCols);

    mem->CopyToHost(offset * sizeof(T), vec.data(), nRows * nCols * sizeof(T), stream);
}


template<class T>
void Matrix<T>::Export(Matrix<T> &mat, PStream &stream) const
{
    verify(mat.nRows == nRows);
    verify(mat.nCols == nCols);
    verify(mat.mem != NULL);
    verify(mem != NULL);
    verify(engine != NULL);

    engine->MemCopy(mem, offset * sizeof(T), mat.mem, mat.offset * sizeof(T), nRows * nCols * sizeof(T), stream);
}


template<class T>
T *Matrix<T>::GetPtrForReadWrite(PStream &stream)
{
    T *ptr;

    verify(mem != NULL);

    mem->Pull(stream.loc, stream);

    ptr = static_cast<T *>(mem->GetPtr(stream.loc)) + offset;

    return ptr;
}


template<class T>
T *Matrix<T>::GetPtrForWrite(PStream &stream)
{
    T *ptr;

    verify(mem != NULL);
    verify(engine != NULL);

    if(isSub == true)
        mem->Pull(stream.loc, stream);
    else
        engine->MemAlloc(mem, stream.loc);

    ptr = static_cast<T *>(mem->GetPtr(stream.loc)) + offset;

    return ptr;
}


template<class T>
void Matrix<T>::FinishWrite(PStream &stream)
{
    verify(mem != NULL);

    mem->Push(stream.loc);
}


template<class T>
T *const Matrix<T>::GetHostData()
{
    T *ptr;
    unsigned long hostLoc;

    verify(engine != NULL);

    hostLoc = engine->GetHostLoc();
    engine->MemAlloc(mem, hostLoc);

    verify(mem != NULL);

    ptr = static_cast<T *>(mem->GetPtr(hostLoc)) + offset;

    return ptr;
}


template<class T>
void Matrix<T>::HostPush()
{
    verify(engine != NULL);
    verify(mem != NULL);

    mem->Push(engine->GetHostLoc());
}


template<class T>
void Matrix<T>::HostPull(PStream &stream)
{
    verify(engine != NULL);
    verify(mem != NULL);

    mem->Pull(engine->GetHostLoc(), stream);
}


template<class T>
void Matrix<T>::Swap(Matrix<T> &mat)
{
    Lock();
    mat.Lock();

    verify(mat.nRows == nRows);
    verify(mat.nCols == nCols);
    verify(isSub == false);
    verify(mat.isSub == false);
    verify(mem != NULL);
    verify(mat.mem != NULL);

    Mem *tmp;

    tmp = mem;
    mem = mat.mem;
    mat.mem = tmp;

    Unlock();
    mat.Unlock();
}


template<class T>
void Matrix<T>::Save(const std::string &filename)
{
    Lock();

    verify(engine != NULL);

    std::ofstream fileStream;

    fileStream.open(filename, std::ofstream::binary |std::ofstream::out|std::ofstream::trunc );
	if(fileStream.fail())std::cout<<filename<<" : fail"<<std::endl;
    verify(fileStream.is_open() == true);

    unsigned int tmp;

    tmp = static_cast<unsigned int>(nRows);
    fileStream.write(reinterpret_cast<char *>(&tmp), sizeof(unsigned int));

    tmp = static_cast<unsigned int>(nCols);
    fileStream.write(reinterpret_cast<char *>(&tmp), sizeof(unsigned int));

    tmp = static_cast<unsigned int>(sizeof(T));
    fileStream.write(reinterpret_cast<char *>(&tmp), sizeof(unsigned int));

    if(nRows * nCols > 0)
    {
        PStream stream;

        engine->StreamCreate(stream, engine->GetHostLoc());
        HostPull(stream);
        fileStream.write(reinterpret_cast<char *>(GetHostData()), sizeof(T) * nRows * nCols);
        engine->StreamDestroy(stream);
    }

    verify(fileStream.good() == true);
    fileStream.close();

    Unlock();
}


template<class T>
void Matrix<T>::Load(const std::string &filename)
{
    Lock();

    std::ifstream fileStream;

    fileStream.open(filename, std::ifstream::binary);
    verify(fileStream.is_open() == true);

    unsigned int _nRows, _nCols, tmp;

    fileStream.read(reinterpret_cast<char *>(&_nRows), sizeof(unsigned int));
    if(isSub == true) verify(_nRows == nRows);

    fileStream.read(reinterpret_cast<char *>(&_nCols), sizeof(unsigned int));
    if(isSub == true) verify(_nCols == nCols);

    Resize(_nRows, _nCols);

    fileStream.read(reinterpret_cast<char *>(&tmp), sizeof(unsigned int));
    verify(tmp == sizeof(T));

    if(nRows * nCols > 0)
    {
        fileStream.read(reinterpret_cast<char *>(GetHostData()), sizeof(T) * nRows * nCols);
        HostPush();
    }

    verify(fileStream.fail() == false);
    fileStream.close();

    Unlock();
}



template class Matrix<float>;
template class Matrix<double>;
template class Matrix<char>;
template class Matrix<unsigned char>;
template class Matrix<short>;
template class Matrix<unsigned short>;
template class Matrix<int>;
template class Matrix<unsigned int>;
template class Matrix<long>;
template class Matrix<unsigned long>;
template class Matrix<long long>;
template class Matrix<unsigned long long>;


}
