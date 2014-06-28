/* Parallel Kernel Independent Fast Multipole Method
   Copyright (C) 2010 George Biros, Harper Langston, Ilya Lashuk
   Copyright (C) 2010, Aparna Chandramowlishwaran, Aashay Shingrapure, Rich Vuduc

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */

#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <cuda.h>
#include <cutil.h>

#include "../p3d/point3d.h"
#include "gpu_setup.h"
#include "kernel3d_mpi.hpp"

#define MPI_ASSERT(c)  mpi_assert__ (((long)c), #c, __FILE__, __LINE__)

#define PI_4I 0.079577471F
#define PI_8I 0.0397887358F

#define BLOCK_HEIGHT 64
#define BLOCK_WIDTH 1

static
void
mpi_assert__ (long cond, const char* str_cond, const char* file, size_t line)
{
  if (!cond) {
    int rank;
    char procname[MPI_MAX_PROCESSOR_NAME+1];
    int procnamelen;
    memset (procname, 0, sizeof (procname));
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name (procname, &procnamelen);
    fprintf (stderr, "*** [%s:%lu--p%d(%s)] ASSERTION FAILURE: %s ***\n",
	     file, (unsigned long)line, rank, procname, str_cond);
    fflush (stderr);
    MPI_Abort (MPI_COMM_WORLD, 1);
  }
}

#if defined (GPU_CERR)
void
gpu_checkerr__stdout (const char* filename, size_t line)
{
  FILE* fp = stdout;
  cudaError_t C_E = cudaGetLastError ();
  if (C_E) {
    int rank;
    char procname[MPI_MAX_PROCESSOR_NAME+1];
    int procnamelen;
    memset (procname, 0, sizeof (procname));
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name (procname, &procnamelen);
    fprintf ((fp), "*** [%s:%lu--p%d(%s)] CUDA ERROR: %s ***\n", filename, line, rank, procname, cudaGetErrorString (C_E));
    fflush (fp);
  }
}
#endif

void
gpu_msg__stdout (const char* msg, const char* filename, size_t lineno)
{
  FILE* fp = stdout;
  int rank;
  char procname[MPI_MAX_PROCESSOR_NAME+1];
  int procnamelen;
  memset (procname, 0, sizeof (procname));
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name (procname, &procnamelen);
  fprintf (fp, "===> [%s:%lu--p%d(%s)] %s\n", filename, lineno, rank, procname, msg);
}

void
gpu_check_pointer (const void* p, const char* fn, size_t l)
{
  if (!p) {
    gpu_msg__stdout ("NULL pointer", fn, l);
    MPI_Abort (MPI_COMM_WORLD, -1);
    MPI_ASSERT (p);
  }
}

size_t
gpu_count (void)
{
 int dev_count;
 CUDA_SAFE_CALL (cudaGetDeviceCount (&dev_count)); GPU_CE;
  if (dev_count > 0) {
    fprintf (stderr, "==> Found %d GPU device%s.\n",
   dev_count,
   dev_count == 1 ? "" : "s");
    return (size_t)dev_count;
  }
  return 0; /* no devices found */
}

static
const char *
get_log_dir_ (void)
{
  static const char* log_dir_ = NULL;
  if (!log_dir_) {
    const char* s = getenv ("LOG_DIR");
    if (s && strlen (s) > 0)
      log_dir_ = s;
    else
      log_dir_ = ".";
  }
  MPI_ASSERT (log_dir_);
  return log_dir_;
}

void
gpu_dumpinfo (FILE* fp, size_t dev_id)
{
  FILE* fp_out = fp;
  if (!fp) {
    /* Open 'default' file based on node name */
    int rank = -1;
    char procname[MPI_MAX_PROCESSOR_NAME+1];
    int procnamelen;
    memset (procname, 0, sizeof (procname));
    MPI_Get_processor_name (procname, &procnamelen);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    const char* log_dir = get_log_dir_ ();
    int pathlen = strlen (log_dir) + 1 + MPI_MAX_PROCESSOR_NAME + 15 + 1;
    char* log_file = new char[pathlen];
    MPI_ASSERT (log_file);
    memset (log_file, 0, pathlen);
    sprintf (log_file, "%s/%s--p%d.log", log_dir, procname, rank);
    fp_out = fopen (log_file, "wt");
    delete[] log_file;
    MPI_ASSERT (fp_out);
  }
  cudaDeviceProp p;
  MPI_ASSERT (dev_id < gpu_count ());
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&p, (int)dev_id)); GPU_CE;
  fprintf (fp_out, "==> Device %lu: \"%s\"\n", (unsigned long)dev_id, p.name);
  fprintf (fp_out, " Major revision number: %d\n", p.major);
  fprintf (fp_out, " Minor revision number: %d\n", p.minor);
  fprintf (fp_out, " Total amount of global memory: %u MB\n", p.totalGlobalMem >> 20);
#if CUDART_VERSION >= 2000
  fprintf (fp_out, " Number of multiprocessors: %d\n", p.multiProcessorCount);
  fprintf (fp_out, " Number of cores: %d\n", 8 * p.multiProcessorCount);
#endif
  fprintf (fp_out, " Total amount of constant memory: %u MB\n", p.totalConstMem >> 20);
  fprintf (fp_out, " Total amount of shared memory per block: %u KB\n", p.sharedMemPerBlock >> 10);
  fprintf (fp_out, " Total number of registers available per block: %d\n", p.regsPerBlock);
  fprintf (fp_out, " Warp size: %d\n", p.warpSize);
  fprintf (fp_out, " Maximum number of threads per block: %d\n", p.maxThreadsPerBlock);
  fprintf (fp_out, " Maximum sizes of each dimension of a block: %d x %d x %d\n",
   p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
  fprintf (fp_out, " Maximum sizes of each dimension of a grid: %d x %d x %d\n",
   p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
  fprintf (fp_out, " Maximum memory pitch: %u bytes\n", p.memPitch);
  fprintf (fp_out, " Texture alignment: %u bytes\n", p.textureAlignment);
  fprintf (fp_out, " Clock rate: %.2f GHz\n", p.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
  fprintf (fp_out, " Concurrent copy and execution: %s\n", p.deviceOverlap ? "Yes" : "No");
#endif
  if (!fp && fp_out)
    fclose (fp_out);
}

void
gpu_select (size_t dev_id)
{
  fprintf (stderr, "==> Selecting GPU device: %lu\n", (unsigned long)dev_id);
  CUDA_SAFE_CALL (cudaSetDevice ((int)dev_id)); GPU_CE;
  gpu_dumpinfo (NULL, dev_id);
}

/** Allocates 'n' bytes, initialized to zero */
void *
gpu_calloc (size_t n)
{
  void* p = NULL;
  if (n) {
    cudaMalloc(&p, n); GPU_CE;
    if (!p) {
      int mpirank;
      MPI_Comm_rank (MPI_COMM_WORLD, &mpirank);
      fprintf (stderr, "[%s:%lu::p%d] Can't allocate %lu bytes!\n",
	       __FILE__, __LINE__, mpirank, (unsigned long)n);
    }
    MPI_ASSERT (p);
    cudaMemset (p, 0, n); GPU_CE;
  }
  return p;
}

double *
gpu_calloc_double (size_t n)
{
  return (double *)gpu_calloc (n * sizeof (double));
}

float *
gpu_calloc_float (size_t n)
{
  return (float *)gpu_calloc (n * sizeof (float));
}

int *
gpu_calloc_int (size_t n)
{
  return (int *)gpu_calloc (n * sizeof (int));
}


void
gpu_copy_cpu2gpu (void* d, const void* s, size_t n_bytes)
{
  if (n_bytes) {
    cudaMemcpy (d, s, n_bytes, cudaMemcpyHostToDevice);
    GPU_CE;
  }
}

void
gpu_copy_cpu2gpu_float (float* d, const float* s, size_t n)
{
  gpu_copy_cpu2gpu (d, s, n * sizeof (float));
}

void
gpu_copy_cpu2gpu_double (double* d, const double* s, size_t n)
{
  gpu_copy_cpu2gpu (d, s, n * sizeof (double));
}


void
gpu_copy_cpu2gpu_int (int* d, const int* s, size_t n)
{
  gpu_copy_cpu2gpu (d, s, n * sizeof (int));
}

void
gpu_copy_gpu2cpu (void* d, const void* s, size_t n_bytes)
{
  if (n_bytes) {
    cudaMemcpy (d, s, n_bytes, cudaMemcpyDeviceToHost);
    GPU_CE;
  }
}

void
gpu_copy_gpu2cpu_float (float* d, const float* s, size_t n)
{
  gpu_copy_gpu2cpu (d, s, n * sizeof (float));
}

void
gpu_copy_gpu2cpu_double (double* d, const double* s, size_t n)
{
  gpu_copy_gpu2cpu (d, s, n * sizeof (double));
}

////////////////////////////////////////BEGIN KERNEL///////////////////////////////////////////////

//#define GRID_WIDTH 1

using namespace std;

#ifdef USE_DOUBLE
__global__ void ulist_kernel(double *t_dp,double *trgVal_dp,
          double *s_dp,
          int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
          int numAugTrg,double kernel_coef) {
  __shared__ double4 s_sh[BLOCK_HEIGHT];
  double3 t_reg;


  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
  if(uniqueBlockId<numAugTrg) {


    double tv_reg=0.0F;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

      t_reg=((double3*)t_dp)[trgIdx];


      double dX_reg;
      double dY_reg;
      double dZ_reg;



    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    //fetching cs and cp into shared mem
  //    for(int i=0;i<ceilf((double)numSrcBox_reg/(double)BLOCK_HEIGHT);i++)
  //      if(threadIdx.x<numSrcBox_reg-i*BLOCK_HEIGHT) {
  //        cs_sh[i*BLOCK_HEIGHT+threadIdx.x]=cs_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
  //        cp_sh[i*BLOCK_HEIGHT+threadIdx.x]=cp_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
  //      }


    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {


      if(num_thread_reg>=lastsum) {
        while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
        loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
        lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
  #ifdef DS_ORG
      s_sh[threadIdx.x]=((double4*)s_dp)[loc_reg];
  #else
      sx_sh[threadIdx.x]=sx_dp[loc_reg];
      sy_sh[threadIdx.x]=sy_dp[loc_reg];
      sz_sh[threadIdx.x]=sz_dp[loc_reg];
      sd_sh[threadIdx.x]=srcDen_dp[loc_reg];
  #endif

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 64
      for(int src=0;src<BLOCK_HEIGHT;src++) {
  #ifdef DS_ORG
        dX_reg=s_sh[src].x-t_reg.x;
        dY_reg=s_sh[src].y-t_reg.y;
        dZ_reg=s_sh[src].z-t_reg.z;

        dX_reg*=dX_reg;
        dY_reg*=dY_reg;
        dZ_reg*=dZ_reg;

        dX_reg += dY_reg+dZ_reg;

        dX_reg = rsqrtf(dX_reg);

        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

        tv_reg+=dX_reg*s_sh[src].w;
  #else
        dX_reg=sx_sh[src]-tx_reg;
        dY_reg=sy_sh[src]-ty_reg;
        dZ_reg=sz_sh[src]-tz_reg;

        dX_reg*=dX_reg;
        dY_reg*=dY_reg;
        dZ_reg*=dZ_reg;

        dX_reg += dY_reg+dZ_reg;

        dX_reg = rsqrtf(dX_reg);

        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

        tv_reg+=dX_reg*sd_sh[src] ;
  #endif

        }
    } // chunk
    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
        while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
        loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
  //      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();
  #ifdef DS_ORG
      s_sh[threadIdx.x]=((double4*)s_dp)[loc_reg];
  #else
      sx_sh[threadIdx.x]=sx_dp[loc_reg];
      sy_sh[threadIdx.x]=sy_dp[loc_reg];
      sz_sh[threadIdx.x]=sz_dp[loc_reg];
      sd_sh[threadIdx.x]=srcDen_dp[loc_reg];
  #endif

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
  #ifdef DS_ORG
      dX_reg=s_sh[src].x-t_reg.x;
      dY_reg=s_sh[src].y-t_reg.y;
      dZ_reg=s_sh[src].z-t_reg.z;

      dX_reg*=dX_reg;
      dY_reg*=dY_reg;
      dZ_reg*=dZ_reg;

      dX_reg += dY_reg+dZ_reg;

      dX_reg = rsqrtf(dX_reg);
        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

      tv_reg+=dX_reg*s_sh[src].w;
  #else
      dX_reg=sx_sh[src]-tx_reg;
      dY_reg=sy_sh[src]-ty_reg;
      dZ_reg=sz_sh[src]-tz_reg;

      dX_reg*=dX_reg;
      dY_reg*=dY_reg;
      dZ_reg*=dZ_reg;

      dX_reg += dY_reg+dZ_reg;

      dX_reg = rsqrtf(dX_reg);

      dX_reg = dX_reg + (dX_reg-dX_reg);
      dX_reg = fmaxf(dX_reg,0.0F);

      tv_reg+=dX_reg*sd_sh[src] ;
  #endif

    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[trgIdx]=tv_reg*PI_4I*kernel_coef;    //div by pi here not inside loop
    }

  }    //extra invalid padding block
}

#else

__global__ void ulist_kernel(float *t_dp,float *trgVal_dp,
          float *s_dp,
          int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
          int numAugTrg,float kernel_coef) {
  __shared__ float4 s_sh[BLOCK_HEIGHT];
  float3 t_reg;


  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
  if(uniqueBlockId<numAugTrg) {


    float tv_reg=0.0F;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

      t_reg=((float3*)t_dp)[trgIdx];


      float dX_reg;
      float dY_reg;
      float dZ_reg;



    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    //fetching cs and cp into shared mem
  //    for(int i=0;i<ceilf((float)numSrcBox_reg/(float)BLOCK_HEIGHT);i++)
  //      if(threadIdx.x<numSrcBox_reg-i*BLOCK_HEIGHT) {
  //        cs_sh[i*BLOCK_HEIGHT+threadIdx.x]=cs_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
  //        cp_sh[i*BLOCK_HEIGHT+threadIdx.x]=cp_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
  //      }


    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {


      if(num_thread_reg>=lastsum) {
        while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
        loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
        lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
  #ifdef DS_ORG
      s_sh[threadIdx.x]=((float4*)s_dp)[loc_reg];
  #else
      sx_sh[threadIdx.x]=sx_dp[loc_reg];
      sy_sh[threadIdx.x]=sy_dp[loc_reg];
      sz_sh[threadIdx.x]=sz_dp[loc_reg];
      sd_sh[threadIdx.x]=srcDen_dp[loc_reg];
  #endif

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 64
      for(int src=0;src<BLOCK_HEIGHT;src++) {
  #ifdef DS_ORG
        dX_reg=s_sh[src].x-t_reg.x;
        dY_reg=s_sh[src].y-t_reg.y;
        dZ_reg=s_sh[src].z-t_reg.z;

        dX_reg*=dX_reg;
        dY_reg*=dY_reg;
        dZ_reg*=dZ_reg;

        dX_reg += dY_reg+dZ_reg;

        dX_reg = rsqrtf(dX_reg);

        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

        tv_reg+=dX_reg*s_sh[src].w;
  #else
        dX_reg=sx_sh[src]-tx_reg;
        dY_reg=sy_sh[src]-ty_reg;
        dZ_reg=sz_sh[src]-tz_reg;

        dX_reg*=dX_reg;
        dY_reg*=dY_reg;
        dZ_reg*=dZ_reg;

        dX_reg += dY_reg+dZ_reg;

        dX_reg = rsqrtf(dX_reg);

        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

        tv_reg+=dX_reg*sd_sh[src] ;
  #endif

        }
    } // chunk
    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
        while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
        loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
  //      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();
  #ifdef DS_ORG
      s_sh[threadIdx.x]=((float4*)s_dp)[loc_reg];
  #else
      sx_sh[threadIdx.x]=sx_dp[loc_reg];
      sy_sh[threadIdx.x]=sy_dp[loc_reg];
      sz_sh[threadIdx.x]=sz_dp[loc_reg];
      sd_sh[threadIdx.x]=srcDen_dp[loc_reg];
  #endif

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
  #ifdef DS_ORG
      dX_reg=s_sh[src].x-t_reg.x;
      dY_reg=s_sh[src].y-t_reg.y;
      dZ_reg=s_sh[src].z-t_reg.z;

      dX_reg*=dX_reg;
      dY_reg*=dY_reg;
      dZ_reg*=dZ_reg;

      dX_reg += dY_reg+dZ_reg;

      dX_reg = rsqrtf(dX_reg);
        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

      tv_reg+=dX_reg*s_sh[src].w;
  #else
      dX_reg=sx_sh[src]-tx_reg;
      dY_reg=sy_sh[src]-ty_reg;
      dZ_reg=sz_sh[src]-tz_reg;

      dX_reg*=dX_reg;
      dY_reg*=dY_reg;
      dZ_reg*=dZ_reg;

      dX_reg += dY_reg+dZ_reg;

      dX_reg = rsqrtf(dX_reg);

      dX_reg = dX_reg + (dX_reg-dX_reg);
      dX_reg = fmaxf(dX_reg,0.0F);

      tv_reg+=dX_reg*sd_sh[src] ;
  #endif

    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[trgIdx]=tv_reg*PI_4I*kernel_coef;    //div by pi here not inside loop
    }

  }    //extra invalid padding block
}

#endif

// kernel for Stokes
#ifdef USE_DOUBLE
__global__ void ulist_kernel_stokes_velocity(double *t_dp,double *trgVal_dp,
    double *s_dp,
    int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
    int numAugTrg, double kernel_coef)
{
  __shared__ double3 sC_sh[BLOCK_HEIGHT]; // sC_sh[i] will contain coordinates of source "i"
  __shared__ double3 sD_sh[BLOCK_HEIGHT]; // sD_sh[i] will contain source density  of source "i" (which has 3 components for Stokes) 

  double3 t_reg;  // position of current target

  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;

  if(uniqueBlockId<numAugTrg) {


    double3 tv_reg={0.0F,0.0F,0.0F} ;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

    t_reg=((double3*)t_dp)[trgIdx];

    double dX_reg;
    double dY_reg;
    double dZ_reg;


    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    //fetching cs and cp into shared mem
    //    for(int i=0;i<ceilf((double)numSrcBox_reg/(double)BLOCK_HEIGHT);i++)
    //      if(threadIdx.x<numSrcBox_reg-i*BLOCK_HEIGHT) {
    //        cs_sh[i*BLOCK_HEIGHT+threadIdx.x]=cs_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
    //        cp_sh[i*BLOCK_HEIGHT+threadIdx.x]=cp_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
    //      }


    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
      sC_sh[threadIdx.x]=((double3*)s_dp)[2*loc_reg];
      sD_sh[threadIdx.x]=((double3*)s_dp)[2*loc_reg+1];

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 32
      for(int src=0;src<BLOCK_HEIGHT;src++) {
	dX_reg=sC_sh[src].x-t_reg.x;
	dY_reg=sC_sh[src].y-t_reg.y;
	dZ_reg=sC_sh[src].z-t_reg.z;

	double invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	double3 cur_pot = sD_sh[src];
	double tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	tv_reg.x += cur_pot.x*invR;
	tv_reg.y += cur_pot.y*invR;
	tv_reg.z += cur_pot.z*invR;
      }
    } // chunk

    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	//      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();

    sC_sh[threadIdx.x]=((double3*)s_dp)[2*loc_reg];
    sD_sh[threadIdx.x]=((double3*)s_dp)[2*loc_reg+1];

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
      dX_reg=sC_sh[src].x-t_reg.x;
      dY_reg=sC_sh[src].y-t_reg.y;
      dZ_reg=sC_sh[src].z-t_reg.z;

      double invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

      // following two lines set invR to zero if invR is infinity
      invR = invR + (invR-invR);
      invR = fmaxf(invR, 0.0F);

      double3 cur_pot = sD_sh[src];
      double tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
      cur_pot.x += tmp_scalar*dX_reg;
      cur_pot.y += tmp_scalar*dY_reg;
      cur_pot.z += tmp_scalar*dZ_reg;

      tv_reg.x += cur_pot.x*invR;
      tv_reg.y += cur_pot.y*invR;
      tv_reg.z += cur_pot.z*invR;
    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[3*trgIdx]   = tv_reg.x*PI_8I*kernel_coef;    //div by pi here not inside loop
      trgVal_dp[3*trgIdx+1] = tv_reg.y*PI_8I*kernel_coef;
      trgVal_dp[3*trgIdx+2] = tv_reg.z*PI_8I*kernel_coef;
    }
  }    //extra invalid padding block -- what  ?????????
}

#else

__global__ void ulist_kernel_stokes_velocity(float *t_dp,float *trgVal_dp,
    float *s_dp,
    int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
    int numAugTrg, float kernel_coef)
{
  __shared__ float3 sC_sh[BLOCK_HEIGHT]; // sC_sh[i] will contain coordinates of source "i"
  __shared__ float3 sD_sh[BLOCK_HEIGHT]; // sD_sh[i] will contain source density  of source "i" (which has 3 components for Stokes) 

  float3 t_reg;  // position of current target

  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;

  if(uniqueBlockId<numAugTrg) {


    float3 tv_reg={0.0F,0.0F,0.0F} ;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

    t_reg=((float3*)t_dp)[trgIdx];

    float dX_reg;
    float dY_reg;
    float dZ_reg;


    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    //fetching cs and cp into shared mem
    //    for(int i=0;i<ceilf((float)numSrcBox_reg/(float)BLOCK_HEIGHT);i++)
    //      if(threadIdx.x<numSrcBox_reg-i*BLOCK_HEIGHT) {
    //        cs_sh[i*BLOCK_HEIGHT+threadIdx.x]=cs_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
    //        cp_sh[i*BLOCK_HEIGHT+threadIdx.x]=cp_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
    //      }


    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
      sC_sh[threadIdx.x]=((float3*)s_dp)[2*loc_reg];
      sD_sh[threadIdx.x]=((float3*)s_dp)[2*loc_reg+1];

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 32
      for(int src=0;src<BLOCK_HEIGHT;src++) {
	dX_reg=sC_sh[src].x-t_reg.x;
	dY_reg=sC_sh[src].y-t_reg.y;
	dZ_reg=sC_sh[src].z-t_reg.z;

	float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	float3 cur_pot = sD_sh[src];
	float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	tv_reg.x += cur_pot.x*invR;
	tv_reg.y += cur_pot.y*invR;
	tv_reg.z += cur_pot.z*invR;
      }
    } // chunk

    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	//      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();

    sC_sh[threadIdx.x]=((float3*)s_dp)[2*loc_reg];
    sD_sh[threadIdx.x]=((float3*)s_dp)[2*loc_reg+1];

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
      dX_reg=sC_sh[src].x-t_reg.x;
      dY_reg=sC_sh[src].y-t_reg.y;
      dZ_reg=sC_sh[src].z-t_reg.z;

      float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

      // following two lines set invR to zero if invR is infinity
      invR = invR + (invR-invR);
      invR = fmaxf(invR, 0.0F);

      float3 cur_pot = sD_sh[src];
      float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
      cur_pot.x += tmp_scalar*dX_reg;
      cur_pot.y += tmp_scalar*dY_reg;
      cur_pot.z += tmp_scalar*dZ_reg;

      tv_reg.x += cur_pot.x*invR;
      tv_reg.y += cur_pot.y*invR;
      tv_reg.z += cur_pot.z*invR;
    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[3*trgIdx]   = tv_reg.x*PI_8I*kernel_coef;    //div by pi here not inside loop
      trgVal_dp[3*trgIdx+1] = tv_reg.y*PI_8I*kernel_coef;
      trgVal_dp[3*trgIdx+2] = tv_reg.z*PI_8I*kernel_coef;
    }
  }    //extra invalid padding block -- what  ?????????
}

#endif





// special "Stokes FMM" kernel -- used for equivalent densities for Stokes-velocity
#ifdef USE_DOUBLE
__global__ void ulist_kernel_stokes_fmm(double *t_dp,double *trgVal_dp,
    double *s_dp,
    int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
    int numAugTrg, double kernel_coef)
{
  __shared__ double3 sC_sh[BLOCK_HEIGHT]; // sC_sh[i] will contain coordinates of source "i"
  __shared__ double4 sD_sh[BLOCK_HEIGHT]; // sD_sh[i] will contain source density  of source "i" (which has 4 components for Stokes-fmm ) 

  double3 t_reg;  // position of current target

  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;

  if(uniqueBlockId<numAugTrg) {


    double3 tv_reg={0.0F,0.0F,0.0F} ;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

    t_reg=((double3*)t_dp)[trgIdx];

    double dX_reg;
    double dY_reg;
    double dZ_reg;


    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
      sC_sh[threadIdx.x].x=s_dp[7*loc_reg];
      sC_sh[threadIdx.x].y=s_dp[7*loc_reg+1];
      sC_sh[threadIdx.x].z=s_dp[7*loc_reg+2];

      sD_sh[threadIdx.x].x=s_dp[7*loc_reg+3];
      sD_sh[threadIdx.x].y=s_dp[7*loc_reg+4];
      sD_sh[threadIdx.x].z=s_dp[7*loc_reg+5];
      sD_sh[threadIdx.x].w=s_dp[7*loc_reg+6];

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 32
      for(int src=0;src<BLOCK_HEIGHT;src++) {
	dX_reg=sC_sh[src].x-t_reg.x;
	dY_reg=sC_sh[src].y-t_reg.y;
	dZ_reg=sC_sh[src].z-t_reg.z;

	double invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	double4 cur_pot = sD_sh[src];
	double tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z - 2/kernel_coef*cur_pot.w)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	tv_reg.x += cur_pot.x*invR;
	tv_reg.y += cur_pot.y*invR;
	tv_reg.z += cur_pot.z*invR;
      }
    } // chunk

    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	//      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();

    sC_sh[threadIdx.x].x=s_dp[7*loc_reg];
    sC_sh[threadIdx.x].y=s_dp[7*loc_reg+1];
    sC_sh[threadIdx.x].z=s_dp[7*loc_reg+2];

    sD_sh[threadIdx.x].x=s_dp[7*loc_reg+3];
    sD_sh[threadIdx.x].y=s_dp[7*loc_reg+4];
    sD_sh[threadIdx.x].z=s_dp[7*loc_reg+5];
    sD_sh[threadIdx.x].w=s_dp[7*loc_reg+6];

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
      dX_reg=sC_sh[src].x-t_reg.x;
      dY_reg=sC_sh[src].y-t_reg.y;
      dZ_reg=sC_sh[src].z-t_reg.z;

      double invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

      // following two lines set invR to zero if invR is infinity
      invR = invR + (invR-invR);
      invR = fmaxf(invR, 0.0F);

      double4 cur_pot = sD_sh[src];
      double tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z - 2/kernel_coef*cur_pot.w)*invR*invR;
      cur_pot.x += tmp_scalar*dX_reg;
      cur_pot.y += tmp_scalar*dY_reg;
      cur_pot.z += tmp_scalar*dZ_reg;

      tv_reg.x += cur_pot.x*invR;
      tv_reg.y += cur_pot.y*invR;
      tv_reg.z += cur_pot.z*invR;
    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[3*trgIdx]   = tv_reg.x*PI_8I*kernel_coef;    //div by pi here not inside loop
      trgVal_dp[3*trgIdx+1] = tv_reg.y*PI_8I*kernel_coef;
      trgVal_dp[3*trgIdx+2] = tv_reg.z*PI_8I*kernel_coef;
    }
  }    //extra invalid padding block -- what  ?????????
}

#else

__global__ void ulist_kernel_stokes_fmm(float *t_dp,float *trgVal_dp,
    float *s_dp,
    int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
    int numAugTrg, float kernel_coef)
{
  __shared__ float3 sC_sh[BLOCK_HEIGHT]; // sC_sh[i] will contain coordinates of source "i"
  __shared__ float4 sD_sh[BLOCK_HEIGHT]; // sD_sh[i] will contain source density  of source "i" (which has 4 components for Stokes-fmm ) 

  float3 t_reg;  // position of current target

  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;

  if(uniqueBlockId<numAugTrg) {


    float3 tv_reg={0.0F,0.0F,0.0F} ;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

    t_reg=((float3*)t_dp)[trgIdx];

    float dX_reg;
    float dY_reg;
    float dZ_reg;


    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
      sC_sh[threadIdx.x].x=s_dp[7*loc_reg];
      sC_sh[threadIdx.x].y=s_dp[7*loc_reg+1];
      sC_sh[threadIdx.x].z=s_dp[7*loc_reg+2];

      sD_sh[threadIdx.x].x=s_dp[7*loc_reg+3];
      sD_sh[threadIdx.x].y=s_dp[7*loc_reg+4];
      sD_sh[threadIdx.x].z=s_dp[7*loc_reg+5];
      sD_sh[threadIdx.x].w=s_dp[7*loc_reg+6];

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 32
      for(int src=0;src<BLOCK_HEIGHT;src++) {
	dX_reg=sC_sh[src].x-t_reg.x;
	dY_reg=sC_sh[src].y-t_reg.y;
	dZ_reg=sC_sh[src].z-t_reg.z;

	float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	float4 cur_pot = sD_sh[src];
	float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z - 2/kernel_coef*cur_pot.w)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	tv_reg.x += cur_pot.x*invR;
	tv_reg.y += cur_pot.y*invR;
	tv_reg.z += cur_pot.z*invR;
      }
    } // chunk

    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
	while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
	loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
	//      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();

    sC_sh[threadIdx.x].x=s_dp[7*loc_reg];
    sC_sh[threadIdx.x].y=s_dp[7*loc_reg+1];
    sC_sh[threadIdx.x].z=s_dp[7*loc_reg+2];

    sD_sh[threadIdx.x].x=s_dp[7*loc_reg+3];
    sD_sh[threadIdx.x].y=s_dp[7*loc_reg+4];
    sD_sh[threadIdx.x].z=s_dp[7*loc_reg+5];
    sD_sh[threadIdx.x].w=s_dp[7*loc_reg+6];

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
      dX_reg=sC_sh[src].x-t_reg.x;
      dY_reg=sC_sh[src].y-t_reg.y;
      dZ_reg=sC_sh[src].z-t_reg.z;

      float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

      // following two lines set invR to zero if invR is infinity
      invR = invR + (invR-invR);
      invR = fmaxf(invR, 0.0F);

      float4 cur_pot = sD_sh[src];
      float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z - 2/kernel_coef*cur_pot.w)*invR*invR;
      cur_pot.x += tmp_scalar*dX_reg;
      cur_pot.y += tmp_scalar*dY_reg;
      cur_pot.z += tmp_scalar*dZ_reg;

      tv_reg.x += cur_pot.x*invR;
      tv_reg.y += cur_pot.y*invR;
      tv_reg.z += cur_pot.z*invR;
    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[3*trgIdx]   = tv_reg.x*PI_8I*kernel_coef;    //div by pi here not inside loop
      trgVal_dp[3*trgIdx+1] = tv_reg.y*PI_8I*kernel_coef;
      trgVal_dp[3*trgIdx+2] = tv_reg.z*PI_8I*kernel_coef;
    }
  }    //extra invalid padding block -- what  ?????????
}

#endif

void make_ds(int **tbdsf, int **tbdsr, int **cs, int **cp, point3d_t* P,int *numAugTrg,int *numSrcBoxTot) {
  for(int i=0;i<P->numTrgBox;i++) {
    *numAugTrg+=(P->trgBoxSize[i]/BLOCK_HEIGHT+((P->trgBoxSize[i]%BLOCK_HEIGHT)?1:0));
    *numSrcBoxTot+=P->uListLen[i];

  }
  int srcidx[P->numSrcBox];
  int srcsum=0;
  for(int i=0;i<P->numSrcBox;i++) {
    srcidx[i]=srcsum;
    srcsum+=P->srcBoxSize[i];
  }
//  cout<<"Split "<<P->numTrgBox<<" targets boxes into "<<*numAugTrg<<endl;
//  cout<<"Total source boxes: "<<*numSrcBoxTot<<endl;

  *tbdsf=(int*)malloc(sizeof(int)*2*P->numTrgBox); MPI_ASSERT (*tbdsf || !P->numTrgBox);
  *tbdsr=(int*)malloc(sizeof(int)*3**numAugTrg); MPI_ASSERT (*tbdsr || !numAugTrg);
  *cs=(int*)malloc(sizeof(int)**numSrcBoxTot); MPI_ASSERT (*cs || !numSrcBoxTot);
  *cp=(int*)malloc(sizeof(int)**numSrcBoxTot); MPI_ASSERT (*cp || !numSrcBoxTot);

  int cc=0;
  int tt=0;
  int tbi=0;

  for(int i=0;i<P->numTrgBox;i++) {
    (*tbdsf)[i*2]=cc;
    int cumulSum=0;
    for(int k=0;k<P->uListLen[i];k++) {
      int srcbox=P->uList[i][k];
      cumulSum+=P->srcBoxSize[srcbox];
      (*cs)[cc]=cumulSum;
      (*cp)[cc]=srcidx[srcbox];
      cc++;
    }
    (*tbdsf)[i*2+1]=cumulSum;
    int remtrg=P->trgBoxSize[i];
    while(remtrg>0) {
      (*tbdsr)[3*tbi]=i;
      (*tbdsr)[3*tbi+1]=(remtrg<BLOCK_HEIGHT)?remtrg:BLOCK_HEIGHT;
      (*tbdsr)[3*tbi+2]=tt;
      tt+=(*tbdsr)[3*tbi+1];
      tbi++;    //tbi corresponds to gpu block id
      remtrg-=BLOCK_HEIGHT;
    }
  }
}

//extern "C"
//{

#ifdef USE_DOUBLE
void dense_inter_gpu(point3d_t *P) {
  double *s_dp,*t_dp;
  double *trgVal_dp;
  int *tbdsf_dp, *tbdsr_dp;
  int *tbdsf,*tbdsr,*cs,*cp,numAugTrg=0,numSrcBoxTot=0;
  int *cs_dp,*cp_dp;
  int srcDOF, trgDOF;

  GPU_MSG ("GPU U-list");

  make_ds (&tbdsf, &tbdsr, &cs, &cp, P, &numAugTrg, &numSrcBoxTot);

  switch(P->kernel_type)
  {
    case KNL_LAP_S_U:
      srcDOF=trgDOF=1;
      break;
    case KNL_STK_S_U:
      srcDOF=trgDOF=3;
      break;
    case KNL_STK_F_U:
      srcDOF=4;
      trgDOF=3;
      break;
    default:
      MPI_ASSERT(false);
  }

  cudaMalloc((void**)&s_dp,(P->numSrc + BLOCK_HEIGHT) * (3+srcDOF)*sizeof(double));
  // s_dp = gpu_calloc_double ((P->numSrc + BLOCK_HEIGHT) * (3+srcDOF)); /* Padded by BLOCK_HEIGHT */

  cudaMalloc((void**)&t_dp,(P->numTrg + BLOCK_HEIGHT) * 3*sizeof(double));
  // t_dp = gpu_calloc_double ((P->numTrg + BLOCK_HEIGHT) * 3);

  // trgVal_dp = gpu_calloc_double (P->numTrg*trgDOF);
  cudaMalloc( (void**)&trgVal_dp, P->numTrg*trgDOF*sizeof(double));

  tbdsf_dp = gpu_calloc_int (P->numTrgBox * 2);
  tbdsr_dp = gpu_calloc_int (numAugTrg * 3);
  cs_dp = gpu_calloc_int (numSrcBoxTot);
  cp_dp = gpu_calloc_int (numSrcBoxTot);

 
  //Put data into the device
  gpu_copy_cpu2gpu_double (s_dp, P->src_, P->numSrc * (3+srcDOF));
  gpu_copy_cpu2gpu_double (t_dp, P->trg_, P->numTrg * 3);

  gpu_copy_cpu2gpu_int (tbdsf_dp, tbdsf, 2 * P->numTrgBox);
  gpu_copy_cpu2gpu_int (tbdsr_dp, tbdsr, 3 * numAugTrg);
  gpu_copy_cpu2gpu_int (cs_dp, cs, numSrcBoxTot);
  gpu_copy_cpu2gpu_int (cp_dp, cp, numSrcBoxTot);

  //kernel call
  int GRID_WIDTH=(int)ceil((double)numAugTrg/65535.0F);
  int GRID_HEIGHT=(int)ceil((double)numAugTrg/(double)GRID_WIDTH);
  dim3 BlockDim (BLOCK_HEIGHT,BLOCK_WIDTH);  //Block width will be 1
  dim3 GridDim (GRID_HEIGHT, GRID_WIDTH);    //Grid width should be 1
  //fprintf (stdout, "@@ [%s:%lu::p%d] numAugTrg=%d; BlockDim x GridDim = [%d x %d] x [%d x %d]\n", __FILE__, (unsigned long)__LINE__, mpirank, numAugTrg, BLOCK_HEIGHT, BLOCK_WIDTH, GRID_HEIGHT, GRID_WIDTH);

#if defined (__DEVICE_EMULATION__)
  GPU_MSG (">>> Device emulation mode <<<\n");
#endif
  if (numAugTrg) // No need to call kernel if numAugTrg == 0
  switch(P->kernel_type)
  {
    case KNL_LAP_S_U:
      ulist_kernel<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg,1/P->kernel_coef[0]); GPU_CE;
      break;
    case KNL_STK_S_U:
      ulist_kernel_stokes_velocity<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg,1/P->kernel_coef[0]); GPU_CE;
      break;
    case KNL_STK_F_U:
      ulist_kernel_stokes_fmm<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg,1/P->kernel_coef[0]); GPU_CE;
      break;
    default:
      MPI_ASSERT(false);
  }

  gpu_copy_gpu2cpu_double (P->trgVal, trgVal_dp, P->numTrg*trgDOF);

  cudaFree(s_dp); GPU_CE;
  cudaFree(t_dp); GPU_CE;

  cudaFree(trgVal_dp); GPU_CE;
  cudaFree(tbdsf_dp); GPU_CE;
  cudaFree(tbdsr_dp); GPU_CE;
  cudaFree(cs_dp); GPU_CE;
  cudaFree(cp_dp); GPU_CE;

  free(cs);
  free(cp);
  free(tbdsf);
  free(tbdsr);
}

#else

void dense_inter_gpu(point3d_t *P) {
  float *s_dp,*t_dp;
  float *trgVal_dp;
  int *tbdsf_dp, *tbdsr_dp;
  int *tbdsf,*tbdsr,*cs,*cp,numAugTrg=0,numSrcBoxTot=0;
  int *cs_dp,*cp_dp;
  int srcDOF, trgDOF;

  GPU_MSG ("GPU U-list");

  make_ds (&tbdsf, &tbdsr, &cs, &cp, P, &numAugTrg, &numSrcBoxTot);

  switch(P->kernel_type)
  {
    case KNL_LAP_S_U:
      srcDOF=trgDOF=1;
      break;
    case KNL_STK_S_U:
      srcDOF=trgDOF=3;
      break;
    case KNL_STK_F_U:
      srcDOF=4;
      trgDOF=3;
      break;
    default:
      MPI_ASSERT(false);
  }

  cudaMalloc((void**)&s_dp,(P->numSrc + BLOCK_HEIGHT) * (3+srcDOF)*sizeof(float));
  // s_dp = gpu_calloc_float ((P->numSrc + BLOCK_HEIGHT) * (3+srcDOF)); /* Padded by BLOCK_HEIGHT */

  cudaMalloc((void**)&t_dp,(P->numTrg + BLOCK_HEIGHT) * 3*sizeof(float));
  // t_dp = gpu_calloc_float ((P->numTrg + BLOCK_HEIGHT) * 3);

  // trgVal_dp = gpu_calloc_float (P->numTrg*trgDOF);
  cudaMalloc( (void**)&trgVal_dp, P->numTrg*trgDOF*sizeof(float));

  tbdsf_dp = gpu_calloc_int (P->numTrgBox * 2);
  tbdsr_dp = gpu_calloc_int (numAugTrg * 3);
  cs_dp = gpu_calloc_int (numSrcBoxTot);
  cp_dp = gpu_calloc_int (numSrcBoxTot);

 
  //Put data into the device
  gpu_copy_cpu2gpu_float (s_dp, P->src_, P->numSrc * (3+srcDOF));
  gpu_copy_cpu2gpu_float (t_dp, P->trg_, P->numTrg * 3);

  gpu_copy_cpu2gpu_int (tbdsf_dp, tbdsf, 2 * P->numTrgBox);
  gpu_copy_cpu2gpu_int (tbdsr_dp, tbdsr, 3 * numAugTrg);
  gpu_copy_cpu2gpu_int (cs_dp, cs, numSrcBoxTot);
  gpu_copy_cpu2gpu_int (cp_dp, cp, numSrcBoxTot);

  //kernel call
  int GRID_WIDTH=(int)ceil((float)numAugTrg/65535.0F);
  int GRID_HEIGHT=(int)ceil((float)numAugTrg/(float)GRID_WIDTH);
  dim3 BlockDim (BLOCK_HEIGHT,BLOCK_WIDTH);  //Block width will be 1
  dim3 GridDim (GRID_HEIGHT, GRID_WIDTH);    //Grid width should be 1
  //fprintf (stdout, "@@ [%s:%lu::p%d] numAugTrg=%d; BlockDim x GridDim = [%d x %d] x [%d x %d]\n", __FILE__, (unsigned long)__LINE__, mpirank, numAugTrg, BLOCK_HEIGHT, BLOCK_WIDTH, GRID_HEIGHT, GRID_WIDTH);

#if defined (__DEVICE_EMULATION__)
  GPU_MSG (">>> Device emulation mode <<<\n");
#endif
  if (numAugTrg) // No need to call kernel if numAugTrg == 0
  switch(P->kernel_type)
  {
    case KNL_LAP_S_U:
      ulist_kernel<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg,1/P->kernel_coef[0]); GPU_CE;
      break;
    case KNL_STK_S_U:
      ulist_kernel_stokes_velocity<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg,1/P->kernel_coef[0]); GPU_CE;
      break;
    case KNL_STK_F_U:
      ulist_kernel_stokes_fmm<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg,1/P->kernel_coef[0]); GPU_CE;
      break;
    default:
      MPI_ASSERT(false);
  }

  gpu_copy_gpu2cpu_float (P->trgVal, trgVal_dp, P->numTrg*trgDOF);

  cudaFree(s_dp); GPU_CE;
  cudaFree(t_dp); GPU_CE;

  cudaFree(trgVal_dp); GPU_CE;
  cudaFree(tbdsf_dp); GPU_CE;
  cudaFree(tbdsr_dp); GPU_CE;
  cudaFree(cs_dp); GPU_CE;
  cudaFree(cp_dp); GPU_CE;

  free(cs);
  free(cp);
  free(tbdsf);
  free(tbdsr);
}
#endif
//}//end extern
