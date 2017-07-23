/* ======================================================
 * The Infinite MNIST dataset
 *
 * The files named "data/t10k-*" and "data/train-*" are the original MNIST
 * files (http://yann.lecun.com/exdb/mnist/).  The other files were initially
 * written for the experiments reported in paper "Training Invariant Support
 * Vector Machines using Selective Sampling" by Loosli, Canu, and Bottou
 * (http://leon.bottou.org/papers/loosli-canu-bottou-2006)
 *
 * Copyright (C) 2006- Leon Bottou and Gaelle Loosli                            
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details. You should have received a copy of the GNU General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 * ====================================================== */


#include "infimnist.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define ASSERTFAIL(f,l) do {                         \
    fprintf(stderr,"Assertion failed: %s:%d\n",f,l); \
    abort(); } while(0)

#define ASSERT(x) do {                       \
    if (!(x)) ASSERTFAIL(__FILE__,__LINE__); \
  } while(0)

#define ASSERTX(x,exec) do {                            \
    if (!(x)) {exec; ASSERTFAIL(__FILE__,__LINE__);}    \
  } while(0)

#define T10K_IMAGES     "/export/home1/shshin/DataSet/infimnist/data/t10k-images-idx3-ubyte"
#define T10K_LABELS     "/export/home1/shshin/DataSet/infimnist/data/t10k-labels-idx1-ubyte"
//#define TRAIN_IMAGES    "/home/shshin/data/mnist8m-patterns-idx3-ubyte" 
//#define TRAIN_LABELS    "/home/shshin/data/mnist8m-labels-idx1-ubyte"  
#define TRAIN_IMAGES    "/export/home1/shshin/DataSet/infimnist/data/train-images-idx3-ubyte"
#define TRAIN_LABELS    "/export/home1/shshin/DataSet/infimnist/data/train-labels-idx1-ubyte" 
#define TRAIN_TANGENTS  "/export/home1/shshin/DataSet/infimnist/data/tangVec_float_60000x28x28.bin"
#define DEFORM_FIELDS   "/export/home1/shshin/DataSet/infimnist/data/fields_float_1522x28x28.bin"



struct infimnist_s 
{
  float (*x)[EXSIZE];               /* x[0]...x[l-1] */
  float (*fields)[EXSIZE];          /* F[0]...F[nb_fields-1] */
  float (*tangent)[NTAN][EXSIZE];   /* T[0]...T[2*nb_train-1] */
  float *y; 			    /* category */
  float alpha;
  long  count;
  long (*cachekeys)[CACHECOLS];
  unsigned char* (*cacheptr)[CACHECOLS];
};

static void
load_float_dataset(const char *dir, const char *file, int nbp, int s, int bla, float *into)
{
  int i;
  FILE *fid;
  char *filename = (char*)malloc(strlen(dir)+strlen(file)+4);
  ASSERT(filename);
  strcpy(filename, dir);
  strcat(filename, "/");
  strcat(filename, file);
  fid = fopen(filename, "rb");
  ASSERTX(fid, fprintf(stderr,"Cannot open file \"%s\"\n", filename));
  fseek(fid, bla, SEEK_SET);
  for (i=0; i<nbp; i++)
    {
      int r = fread(into, sizeof(float), s, fid);
      ASSERT(r == s);
      into += s;
    }
  fclose(fid);
  free(filename);
}

static void
load_ubyte_dataset(const char *dir, const char *file, int nbp, int s, int bla, float *into)
{
  int i,j;
  FILE *fid;
  char *filename = (char*)malloc(strlen(dir)+strlen(file)+4);
  unsigned char *temp = (unsigned char*)malloc(s);
  ASSERT(filename);
  ASSERT(temp);
  strcpy(filename, dir);
  strcat(filename, "");
  strcat(filename, file);
  fid = fopen(filename, "rb");
  ASSERTX(fid, fprintf(stderr,"Cannot open file \"%s\"\n", filename));
  fseek(fid, bla, SEEK_SET);
  for (i=0; i<nbp; i++)
    {
      int r = fread(temp, 1, s, fid);
      ASSERT(r == s);
      for (j=0; j<s; j++)
        *into++ = (float)temp[j];
    }
  fclose(fid);
  free(filename);
  free(temp);
}

infimnist_t *
infimnist_create(const char *dirname)
{
  int i,j;
  infimnist_t *p;
  const char *dir = (dirname) ? dirname : "";
  p = (infimnist_t*)malloc(sizeof(infimnist_t));
  ASSERT(p);
  memset(p, 0, sizeof(infimnist_t));
  p->x = (void*)malloc(EXSIZE*(TESTNUM+TRAINNUM)*sizeof(float));
  //p->fields = (void*)malloc(EXSIZE*FIELDNUM*sizeof(float));
  //p->tangent = (void*)malloc(EXSIZE*NTAN*TRAINNUM*sizeof(float));
  p->y = (float*)malloc((TESTNUM+TRAINNUM)*sizeof(float));
  //ASSERT(p->x && p->fields && p->tangent && p->y);
  p->cachekeys = (void*)malloc((CACHEROWS*CACHECOLS)*sizeof(long));
  p->cacheptr = (void*)malloc((CACHEROWS*CACHECOLS)*sizeof(void*));
  ASSERT(p->cachekeys && p->cacheptr);

  load_ubyte_dataset(dir, T10K_IMAGES, TESTNUM, EXSIZE, 16, &p->x[0][0]);
  load_ubyte_dataset(dir, T10K_LABELS, TESTNUM, 1, 8, &p->y[0]);
  load_ubyte_dataset(dir, TRAIN_IMAGES, TRAINNUM, EXSIZE, 16, &p->x[TESTNUM][0]);
  load_ubyte_dataset(dir, TRAIN_LABELS, TRAINNUM, 1, 8, &p->y[TESTNUM]);
  //load_float_dataset(dir, DEFORM_FIELDS, FIELDNUM, EXSIZE, 0, &p->fields[0][0]);
  //load_float_dataset(dir, TRAIN_TANGENTS, TRAINNUM*NTAN, EXSIZE, 0, &p->tangent[0][0][0]);
  p->alpha = 1.00;
  p->count = 0;
  for (i=0; i<CACHEROWS; i++)
    for (j=0; j<CACHECOLS; j++)
      {
        p->cachekeys[i][j] = -1;
        p->cacheptr[i][j] = NULL;
      }
  return p;
}

void 
infimnist_cache_clear(infimnist_t *p)
{
  int i,j;
  for (i=0; i<CACHEROWS; i++)
    for (j=0; j<CACHECOLS; j++)
      {
        p->cachekeys[i][j] = -1;
        if (p->cacheptr[i][j])
          free(p->cacheptr[i][j]);
        p->cacheptr[i][j] = NULL;
      }
}

void 
infimnist_destroy(infimnist_t *p)
{
  if (p) 
    {
      if (p->cachekeys && p->cacheptr)
        infimnist_cache_clear(p);
      if (p->x) 
        free(p->x);
      if (p->y) 
        free(p->y);
      if (p->fields) 
        free(p->fields);
      if (p->tangent) 
        free(p->tangent);
      if (p->cachekeys) 
        free(p->cachekeys);
      if (p->cacheptr)
        free(p->cacheptr);
      free(p);
    }
}


int 
infimnist_get_label(infimnist_t *p, long i)
{
  ASSERTX(i >= 0, fprintf(stderr,"Invalid infimnist index\n"));
  if (i < TESTNUM + TRAINNUM) 
    return p->y[i];
  else
    return p->y[TESTNUM + ((i-TESTNUM) % TRAINNUM)];
}


unsigned char *
translation(unsigned char *c, int t)
{
  int i;
  switch (t)
    {
    case 0: // tr v1
      for (i=0;i<EXSIZE-COTE;i++)
        c[i] = c[i+COTE];
      for (i=EXSIZE-COTE;i<EXSIZE;i++)
        c[i] = 0;
      break;
    case 1: // tr v-1
      for (i=EXSIZE-1;i>=COTE;i--)
        c[i] = c[i-COTE];
      for (i=COTE-1;i>=0;i--)
        c[i] = 0;
      break;
    case 2: // tr h1
      for (i=0;i<EXSIZE-1;i++)
        if ( i%COTE == 0)
          c[i] = 0;
        else
          c[i] = c[i+1];
      c[EXSIZE-1] = 0;
      break;
    case 3: // tr h1;
      for (i=EXSIZE-1;i>0;i--)
        if ((i-1)%COTE == 0)
          c[i] = 0;
        else
          c[i] = c[i-1];
      c[0] = 0;
      break;
    case 4: // 0 & 2
      c = translation(c, 0);
      c = translation(c, 2);
      break;
    case 5: // 0 & 3
      c = translation(c, 0);
      c = translation(c, 3);
      break;
    case 6: // 1 & 2
      c = translation(c, 1);
      c = translation(c, 2);
      break;
    case 7: // 1 & 3
      c = translation(c, 1);
      c = translation(c, 3);
      break;
    case 8: // nothing
      break;
    }
  return c;
}

unsigned char *
compute_transformed_vector(infimnist_t *p, long i)
{
  float alpha;
  int j,a,k1,k2;
  unsigned char *s;
  ASSERT(i >= 0);
  s = (unsigned char*)malloc(EXSIZE);
  ASSERT(s);
  if (i < TESTNUM+TRAINNUM)
    {
      for (j=0; j<EXSIZE; j++)
        s[j] = p->x[i][j];
      return s;
    }
  k1 = (int)(((unsigned long)(i*131071L))%(FIELDNUM-1));
  k2 = (int)(((unsigned long)(k1+1+2*i))%FIELDNUM);
  a = (int)((unsigned long)(i-TESTNUM)%TRAINNUM);
  alpha = p->alpha;
  if ((k1 + k2) & 1)
    alpha = -alpha;
  for (j=0; j<EXSIZE; j++)
    {
      float x = p->x[a+TESTNUM][j] +
        alpha * (p->fields[k1][j] * p->tangent[a][0][j] -
                 p->fields[k2][j] * p->tangent[a][1][j] );
      if (x < 0) 
        s[j] = 0;
      else if (x > 255) 
        s[j] = 255;
      else
        s[j] = (unsigned char)(int)x;
    }
  return translation(s, k1 % 9);
}


const unsigned char *
infimnist_get_pattern(infimnist_t *p, long i)
{
  int j, b;
  unsigned char *s;
  ASSERTX(i >= 0, fprintf(stderr,"Invalid infimnist index\n"));
  b = (int)(i % CACHEROWS);
  for (j=0; j<CACHECOLS; j++)
    if (i == p->cachekeys[b][j])
      return p->cacheptr[b][j];
  s = compute_transformed_vector(p, i);
  for (j=0; j<CACHECOLS; j++)
    if (p->cachekeys[b][j] < 0)
      {
        p->cachekeys[b][j] = i;
        p->cacheptr[b][j] = s;
        return s;
      }
  j = (p->count++) % CACHECOLS;
  free (p->cacheptr[b][j]);
  p->cacheptr[b][j] = s;
  p->cachekeys[b][j] = i;
  return s;
}

