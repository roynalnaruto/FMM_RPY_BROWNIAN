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

using namespace std;
#include <mpi.h>

#define PI_4I 0.079577471F
#define PI_8I 0.0397887358F

//#define PI3_4I 0.238732413F
#include <cutil.h>
//#include <cutil_inline.h>
#include "../p3d/upComp.h"
#include "../p3d/dnComp.h"
#include "gpu_setup.h"
#include "kernel3d_mpi.hpp"

#include <cstdio>
#define MPI_ASSERT(c)  mpi_assert__ (((long)c), #c, __FILE__, __LINE__)

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

#define BLOCK_HEIGHT 64

__constant__ float3 sampos[320];	//undefined for everything greater than 295 for 6, greater than 191 for 4

__constant__ float3 samposDn[152];	//undefined for everything greater than 151 for 6 and 55 for 4

__global__ void up_kernel(float *src_dp,float *trgVal_dp,float *trgCtr_dp,float *trgRad_dp,int *srcBox_dp,int numSrcBox) {
	__shared__ float4 s_sh[BLOCK_HEIGHT];

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numSrcBox) {
		float3 trgCtr;
		float trgRad;
	//	float3 samp[5];
		float3 trg[5];
		float dX_reg;
		float dY_reg;
		float dZ_reg;
		int2 src=((int2*)srcBox_dp)[uniqueBlockId];	//x has start, y has size
		src.x+=threadIdx.x;

		trgCtr=((float3*)trgCtr_dp)[uniqueBlockId];
		trgRad=trgRad_dp[uniqueBlockId];

		//construct the trg

		trg[0].x=trgCtr.x+trgRad*sampos[4*threadIdx.x].x;
		trg[0].y=trgCtr.y+trgRad*sampos[4*threadIdx.x].y;
		trg[0].z=trgCtr.z+trgRad*sampos[4*threadIdx.x].z;
		trg[1].x=trgCtr.x+trgRad*sampos[4*threadIdx.x+1].x;
		trg[1].y=trgCtr.y+trgRad*sampos[4*threadIdx.x+1].y;
		trg[1].z=trgCtr.z+trgRad*sampos[4*threadIdx.x+1].z;
		trg[2].x=trgCtr.x+trgRad*sampos[4*threadIdx.x+2].x;
		trg[2].y=trgCtr.y+trgRad*sampos[4*threadIdx.x+2].y;
		trg[2].z=trgCtr.z+trgRad*sampos[4*threadIdx.x+2].z;
		trg[3].x=trgCtr.x+trgRad*sampos[4*threadIdx.x+3].x;
		trg[3].y=trgCtr.y+trgRad*sampos[4*threadIdx.x+3].y;
		trg[3].z=trgCtr.z+trgRad*sampos[4*threadIdx.x+3].z;
		trg[4].x=trgCtr.x+trgRad*sampos[256+threadIdx.x].x;
		trg[4].y=trgCtr.y+trgRad*sampos[256+threadIdx.x].y;
		trg[4].z=trgCtr.z+trgRad*sampos[256+threadIdx.x].z;

	//	int numSrc=srcBoxSize[uniqueBlockId];

		float4 tv=make_float4(0.0F,0.0F,0.0F,0.0F);
		float tve=0.0F;






		int num_chunk_loop=src.y/BLOCK_HEIGHT;
		for(int chunk=0;chunk<num_chunk_loop;chunk++) {
			__syncthreads();
			s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
			__syncthreads();

			src.x+=BLOCK_HEIGHT;

			for(int s=0;s<BLOCK_HEIGHT;s++) {
				dX_reg=s_sh[s].x-trg[0].x;
				dY_reg=s_sh[s].y-trg[0].y;
				dZ_reg=s_sh[s].z-trg[0].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tv.x+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[1].x;
				dY_reg=s_sh[s].y-trg[1].y;
				dZ_reg=s_sh[s].z-trg[1].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tv.y+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[2].x;
				dY_reg=s_sh[s].y-trg[2].y;
				dZ_reg=s_sh[s].z-trg[2].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tv.z+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[3].x;
				dY_reg=s_sh[s].y-trg[3].y;
				dZ_reg=s_sh[s].z-trg[3].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tv.w+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[4].x;
				dY_reg=s_sh[s].y-trg[4].y;
				dZ_reg=s_sh[s].z-trg[4].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tve+=dX_reg*s_sh[s].w;
				///////////////////////////////
			}

		}	//end num chunk loop
		__syncthreads();
		s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
		__syncthreads();
		for(int s=0;s<src.y%BLOCK_HEIGHT;s++) {
			dX_reg=s_sh[s].x-trg[0].x;
			dY_reg=s_sh[s].y-trg[0].y;
			dZ_reg=s_sh[s].z-trg[0].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tv.x+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[1].x;
			dY_reg=s_sh[s].y-trg[1].y;
			dZ_reg=s_sh[s].z-trg[1].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tv.y+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[2].x;
			dY_reg=s_sh[s].y-trg[2].y;
			dZ_reg=s_sh[s].z-trg[2].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tv.z+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[3].x;
			dY_reg=s_sh[s].y-trg[3].y;
			dZ_reg=s_sh[s].z-trg[3].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tv.w+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[4].x;
			dY_reg=s_sh[s].y-trg[4].y;
			dZ_reg=s_sh[s].z-trg[4].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tve+=dX_reg*s_sh[s].w;
			///////////////////////////////
		}	//end residual loop

		//write back
		tv.x*=PI_4I;
		tv.y*=PI_4I;
		tv.z*=PI_4I;
		tv.w*=PI_4I;
	//	tv.x=(float)trgCtr;
	//	tv.y=tv.z=tv.w=0.0F;
		((float4*)trgVal_dp)[uniqueBlockId*74+threadIdx.x]=tv;
		if(threadIdx.x<40)
			trgVal_dp[uniqueBlockId*296+256+threadIdx.x]=tve*PI_4I;
	}

}

__global__ void up_kernel_4(float *src_dp,float *trgVal_dp,float *trgCtr_dp,float *trgRad_dp,int *srcBox_dp,int numSrcBox) {
	__shared__ float4 s_sh[BLOCK_HEIGHT];

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numSrcBox) {
		float3 trgCtr;
		float trgRad;
	//	float3 samp[5];
		float3 trg[3];
		float dX_reg;
		float dY_reg;
		float dZ_reg;
		int2 src=((int2*)srcBox_dp)[uniqueBlockId];	//x has start, y has size
		src.x+=threadIdx.x;

		trgCtr=((float3*)trgCtr_dp)[uniqueBlockId];
		trgRad=trgRad_dp[uniqueBlockId];

		//construct the trg

		trg[0].x=trgCtr.x+trgRad*sampos[2*threadIdx.x].x;
		trg[0].y=trgCtr.y+trgRad*sampos[2*threadIdx.x].y;
		trg[0].z=trgCtr.z+trgRad*sampos[2*threadIdx.x].z;
		trg[1].x=trgCtr.x+trgRad*sampos[2*threadIdx.x+1].x;
		trg[1].y=trgCtr.y+trgRad*sampos[2*threadIdx.x+1].y;
		trg[1].z=trgCtr.z+trgRad*sampos[2*threadIdx.x+1].z;
		trg[2].x=trgCtr.x+trgRad*sampos[128+threadIdx.x].x;		//128 is blockheight*(trg2fetch-1)
		trg[2].y=trgCtr.y+trgRad*sampos[128+threadIdx.x].y;
		trg[2].z=trgCtr.z+trgRad*sampos[128+threadIdx.x].z;

	//	int numSrc=srcBoxSize[uniqueBlockId];

		float2 tv=make_float2(0.0F,0.0F);					//can be converted into a generic array.. not too big
		float tve=0.0F;






		int num_chunk_loop=src.y/BLOCK_HEIGHT;
		for(int chunk=0;chunk<num_chunk_loop;chunk++) {
			__syncthreads();
			s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
			__syncthreads();

			src.x+=BLOCK_HEIGHT;

			for(int s=0;s<BLOCK_HEIGHT;s++) {
				dX_reg=s_sh[s].x-trg[0].x;
				dY_reg=s_sh[s].y-trg[0].y;
				dZ_reg=s_sh[s].z-trg[0].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tv.x+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[1].x;
				dY_reg=s_sh[s].y-trg[1].y;
				dZ_reg=s_sh[s].z-trg[1].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tv.y+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[2].x;
				dY_reg=s_sh[s].y-trg[2].y;
				dZ_reg=s_sh[s].z-trg[2].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
				tve+=dX_reg*s_sh[s].w;
			}
		}	//end num chunk loop
		__syncthreads();
		s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
		__syncthreads();
		for(int s=0;s<src.y%BLOCK_HEIGHT;s++) {
			dX_reg=s_sh[s].x-trg[0].x;
			dY_reg=s_sh[s].y-trg[0].y;
			dZ_reg=s_sh[s].z-trg[0].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tv.x+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[1].x;
			dY_reg=s_sh[s].y-trg[1].y;
			dZ_reg=s_sh[s].z-trg[1].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tv.y+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[2].x;
			dY_reg=s_sh[s].y-trg[2].y;
			dZ_reg=s_sh[s].z-trg[2].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);
			tve+=dX_reg*s_sh[s].w;
			///////////////////////////////
		}	//end residual loop

		//write back
		tv.x*=PI_4I;
		tv.y*=PI_4I;
	//	tv.x=(float)trgCtr;
	//	tv.y=tv.z=tv.w=0.0F;
		((float2*)(trgVal_dp+uniqueBlockId*152))[threadIdx.x]=tv;	//in generic, float3 writes will be unrolled into multiple writes
		if(threadIdx.x<24)
			trgVal_dp[uniqueBlockId*152+128+threadIdx.x]=tve*PI_4I;
	}

}

__global__ void up_kernel_stokes_velocity_4(float *src_dp,float *trgVal_dp,float *trgCtr_dp,float *trgRad_dp,int *srcBox_dp,int numSrcBox)
{
  __shared__ float3 sc_sh[BLOCK_HEIGHT]; // source coordinates
  __shared__ float3 sd_sh[BLOCK_HEIGHT]; // source densities

  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
  if(uniqueBlockId<numSrcBox) {
    float3 trgCtr;
    float trgRad;
    //	float3 samp[5];
    float3 trg[3];
    float dX_reg;
    float dY_reg;
    float dZ_reg;
    int2 src=((int2*)srcBox_dp)[uniqueBlockId];	//x has start, y has size
    src.x+=threadIdx.x;

    trgCtr=((float3*)trgCtr_dp)[uniqueBlockId];
    trgRad=trgRad_dp[uniqueBlockId];

    //construct the trg

    trg[0].x=trgCtr.x+trgRad*sampos[2*threadIdx.x].x;
    trg[0].y=trgCtr.y+trgRad*sampos[2*threadIdx.x].y;
    trg[0].z=trgCtr.z+trgRad*sampos[2*threadIdx.x].z;
    trg[1].x=trgCtr.x+trgRad*sampos[2*threadIdx.x+1].x;
    trg[1].y=trgCtr.y+trgRad*sampos[2*threadIdx.x+1].y;
    trg[1].z=trgCtr.z+trgRad*sampos[2*threadIdx.x+1].z;
    trg[2].x=trgCtr.x+trgRad*sampos[128+threadIdx.x].x;		//128 is blockheight*(trg2fetch-1)
    trg[2].y=trgCtr.y+trgRad*sampos[128+threadIdx.x].y;
    trg[2].z=trgCtr.z+trgRad*sampos[128+threadIdx.x].z;

    //	int numSrc=srcBoxSize[uniqueBlockId];

    float3 pot0=make_float3(0.0F,0.0F,0.0F);
    float3 pot1=make_float3(0.0F,0.0F,0.0F);
    float3 pot2=make_float3(0.0F,0.0F,0.0F);

    int num_chunk_loop=src.y/BLOCK_HEIGHT;
    for(int chunk=0;chunk<num_chunk_loop;chunk++) {
      __syncthreads();
      sc_sh[threadIdx.x]=((float3*)src_dp)[2*src.x];
      sd_sh[threadIdx.x]=((float3*)src_dp)[2*src.x+1];
      __syncthreads();

      src.x+=BLOCK_HEIGHT;

      for(int s=0;s<BLOCK_HEIGHT;s++) {
	dX_reg=sc_sh[s].x-trg[0].x;
	dY_reg=sc_sh[s].y-trg[0].y;
	dZ_reg=sc_sh[s].z-trg[0].z;

	float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	float3 cur_pot = sd_sh[s];
	float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	pot0.x += cur_pot.x*invR;
	pot0.y += cur_pot.y*invR;
	pot0.z += cur_pot.z*invR;

	///////////////////////////////
	
	dX_reg=sc_sh[s].x-trg[1].x;
	dY_reg=sc_sh[s].y-trg[1].y;
	dZ_reg=sc_sh[s].z-trg[1].z;

	invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	cur_pot = sd_sh[s];
	tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	pot1.x += cur_pot.x*invR;
	pot1.y += cur_pot.y*invR;
	pot1.z += cur_pot.z*invR;

	///////////////////////////////
	
	dX_reg=sc_sh[s].x-trg[2].x;
	dY_reg=sc_sh[s].y-trg[2].y;
	dZ_reg=sc_sh[s].z-trg[2].z;

	invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	cur_pot = sd_sh[s];
	tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	pot2.x += cur_pot.x*invR;
	pot2.y += cur_pot.y*invR;
	pot2.z += cur_pot.z*invR;
      }
    }	//end num chunk loop
    __syncthreads();
      sc_sh[threadIdx.x]=((float3*)src_dp)[2*src.x];
      sd_sh[threadIdx.x]=((float3*)src_dp)[2*src.x+1];
    __syncthreads();
    for(int s=0;s<src.y%BLOCK_HEIGHT;s++) {
	dX_reg=sc_sh[s].x-trg[0].x;
	dY_reg=sc_sh[s].y-trg[0].y;
	dZ_reg=sc_sh[s].z-trg[0].z;

	float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	float3 cur_pot = sd_sh[s];
	float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	pot0.x += cur_pot.x*invR;
	pot0.y += cur_pot.y*invR;
	pot0.z += cur_pot.z*invR;

	///////////////////////////////
	
	dX_reg=sc_sh[s].x-trg[1].x;
	dY_reg=sc_sh[s].y-trg[1].y;
	dZ_reg=sc_sh[s].z-trg[1].z;

	invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	cur_pot = sd_sh[s];
	tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	pot1.x += cur_pot.x*invR;
	pot1.y += cur_pot.y*invR;
	pot1.z += cur_pot.z*invR;

	///////////////////////////////
	
	dX_reg=sc_sh[s].x-trg[2].x;
	dY_reg=sc_sh[s].y-trg[2].y;
	dZ_reg=sc_sh[s].z-trg[2].z;

	invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

	// following two lines set invR to zero if invR is infinity
	invR = invR + (invR-invR);
	invR = fmaxf(invR, 0.0F);

	cur_pot = sd_sh[s];
	tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z)*invR*invR;
	cur_pot.x += tmp_scalar*dX_reg;
	cur_pot.y += tmp_scalar*dY_reg;
	cur_pot.z += tmp_scalar*dZ_reg;

	pot2.x += cur_pot.x*invR;
	pot2.y += cur_pot.y*invR;
	pot2.z += cur_pot.z*invR;
    }	//end residual loop

    //write back
    pot0.x *= PI_8I;
    pot0.y *= PI_8I;
    pot0.z *= PI_8I;
    pot1.x *= PI_8I;
    pot1.y *= PI_8I;
    pot1.z *= PI_8I;
    pot2.x *= PI_8I;
    pot2.y *= PI_8I;
    pot2.z *= PI_8I;

    ((float3*)trgVal_dp)[uniqueBlockId*152+2*threadIdx.x]=pot0;
    ((float3*)trgVal_dp)[uniqueBlockId*152+2*threadIdx.x+1]=pot1;

    if(threadIdx.x<24)
      ((float3*)trgVal_dp)[uniqueBlockId*152+128+threadIdx.x]=pot2;
  }
}

// void unmake_ds_up(float *trgValE,upComp_t *UpC) {
// 	int t=0;
// 	for(int i=0;i<UpC->numSrcBox;i++) {
// 		for(int j=0;j<UpC->trgDim;j++) {
// //			assert(UpC->trgVal[i]!=NULL);
// 
// 			if(UpC->trgVal[i]!=NULL)
// 				UpC->trgVal[i][j]=trgValE[t];
// 			t++;
// //			cout<<i<<","<<j<<endl;
// //			cout<<trgValE[t-1]<<endl;
// 		}
// 	}
// }


void make_ds_up(int *srcBox,upComp_t *UpC) {	//TODO
	int start=0;
	int t=0;
	int size;
	for(int i=0;i<UpC->numSrcBox;i++) {
		srcBox[t++]=start;
		size=UpC->srcBoxSize[i];
		srcBox[t++]=size;
		start+=size;
	}
}

void gpu_up(upComp_t *UpC) {
  int srcDOF, trgDOF;
  GPU_MSG ("Upward computation");
  if (!UpC || !UpC->numSrcBox) { GPU_MSG ("==> No source boxes; skipping..."); return; }
  //	cudaSetDevice(0);
//	unsigned int timer;
//	float ms;
//	cutCreateTimer(&timer);

	float *src_dp,*trgVal_dp,*trgCtr_dp,*trgRad_dp;
	int *srcBox_dp;

	// float trgValE[UpC->trgDim*UpC->numSrcBox];
	int srcBox[2*UpC->numSrcBox];

	make_ds_up(srcBox,UpC);

	switch(UpC->kernel_type)
	{
	  case KNL_LAP_S_U:
	    srcDOF=trgDOF=1;
	    break;
	  case KNL_STK_S_U:
	    srcDOF=trgDOF=3;
	    break;
	  default:
	    MPI_ASSERT(false);
	}

	src_dp = gpu_calloc_float ((UpC->numSrc + BLOCK_HEIGHT) * (UpC->dim+srcDOF));
	trgCtr_dp = gpu_calloc_float (UpC->numSrcBox*3);
	trgRad_dp = gpu_calloc_float (UpC->numSrcBox);
	srcBox_dp = gpu_calloc_int (UpC->numSrcBox*2);
	trgVal_dp = gpu_calloc_float (UpC->trgDim*UpC->numSrcBox*trgDOF);

	gpu_copy_cpu2gpu_float (src_dp, UpC->src_, UpC->numSrc * (UpC->dim+srcDOF));
	gpu_copy_cpu2gpu_float (trgCtr_dp, UpC->trgCtr, UpC->numSrcBox*3);
	gpu_copy_cpu2gpu_float (trgRad_dp, UpC->trgRad, UpC->numSrcBox);
	gpu_copy_cpu2gpu_int (srcBox_dp, srcBox, UpC->numSrcBox*2);

	cudaMemcpyToSymbol(sampos,UpC->samPosF/*samp*/,sizeof(float)*UpC->trgDim*3); GPU_CE;
	int GRID_WIDTH=(int)ceil((float)UpC->numSrcBox/65535.0F);
	int GRID_HEIGHT=(int)ceil((float)UpC->numSrcBox/(float)GRID_WIDTH);
	dim3 GridDim(GRID_HEIGHT, GRID_WIDTH);
//	cout<<"Width: "<<GRID_WIDTH<<" HEIGHT: "<<GRID_HEIGHT<<endl;

	switch(UpC->kernel_type)
	{
	  case KNL_LAP_S_U:
	    if(UpC->trgDim==296) {
	      up_kernel<<<GridDim,BLOCK_HEIGHT>>>(src_dp,trgVal_dp,trgCtr_dp,trgRad_dp,srcBox_dp,UpC->numSrcBox);
	    }
	    else if(UpC->trgDim==152) {
	      up_kernel_4<<<GridDim,BLOCK_HEIGHT>>>(src_dp,trgVal_dp,trgCtr_dp,trgRad_dp,srcBox_dp,UpC->numSrcBox);
	    }
	    else
	    {
	      GPU_MSG ("Upward computations not implemented for this kernel and this accuracy"); //Exit the process?
	      MPI_ASSERT(false);
	    }
	    //also, a generic call can be put here
	    break;
	  case KNL_STK_S_U:
	    if(UpC->trgDim==152)
	      up_kernel_stokes_velocity_4<<<GridDim,BLOCK_HEIGHT>>>(src_dp,trgVal_dp,trgCtr_dp,trgRad_dp,srcBox_dp,UpC->numSrcBox);
	    else
	    {
	      GPU_MSG ("Upward computations not implemented for this kernel and this accuracy"); //Exit the process?
	      MPI_ASSERT(false);
	    }
	    break;
	  default:
	    MPI_ASSERT(false);
	}
	GPU_CE;

	gpu_copy_gpu2cpu_float (UpC->trgVal, trgVal_dp, UpC->trgDim*UpC->numSrcBox*trgDOF);
//	CUT_SAFE_CALL(cutStopTimer(timer));
//	ms = cutGetTimerValue(timer);
//	cout<<"Up kernel: "<<ms<<"ms"<<endl;
	// unmake_ds_up(trgValE,UpC);	//FIXME: copies the gpu output into the 2d array used by the interface... make the interface use a 1d array

	cudaFree(src_dp); GPU_CE;
	cudaFree(trgCtr_dp); GPU_CE;
	cudaFree(trgRad_dp); GPU_CE;
	cudaFree(srcBox_dp); GPU_CE;
	cudaFree(trgVal_dp); GPU_CE;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void make_ds_down(int *trgBox,dnComp_t *DnC) {
	int tt=0;
	int tot=0;
	for(int i=0;i<DnC->numTrgBox;i++) {
		int rem=DnC->trgBoxSize[i];
		while(rem>0) {
			trgBox[tt++]=tot;		//start
			int size=(rem<BLOCK_HEIGHT)?rem:BLOCK_HEIGHT;
			trgBox[tt++]=size;		//size
			trgBox[tt++]=i;			//box
			tot+=size;
			rem-=size;
		}
	}
}

// void unmake_ds_down(float *trgValE,dnComp_t *DnC) {
// 	int t=0;
// 	for(int i=0;i<DnC->numTrgBox;i++) {
// 		for(int j=0;j<DnC->trgBoxSize[i];j++) {
// 			if(DnC->trgVal[i]!=NULL) {
// 				DnC->trgVal[i][j]=trgValE[t++];
// //				cout<<DnC->trgVal[i][j]<<endl;
// 			}
// 		}
// 	}
// }

__global__ void dn_kernel(float *trg_dp,float *trgVal_dp,float *srcCtr_dp,float *srcRad_dp,int *trgBox_dp,float *srcDen_dp,int numAugTrg) {
	__shared__ float4 s_sh[64];
	int3 trgBox;

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numAugTrg) {
		trgBox=((int3*)trgBox_dp)[uniqueBlockId];		//start,size,box

		float3 t_reg=((float3*)trg_dp)[trgBox.x+threadIdx.x];

		float3 srcCtr=((float3*)srcCtr_dp)[trgBox.z];
		float srcRad=srcRad_dp[trgBox.z];

		float dX_reg,dY_reg,dZ_reg;
		float tv_reg=0.0;

		//every thread computes a single src body


		s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[threadIdx.x].x;
		s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[threadIdx.x].y;
		s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[threadIdx.x].z;

		s_sh[threadIdx.x].w=srcDen_dp[152*trgBox.z+threadIdx.x];

		__syncthreads();
		for(int src=0;src<64;src++) {
			dX_reg=s_sh[src].x-t_reg.x;
			dY_reg=s_sh[src].y-t_reg.y;
			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);

			tv_reg+=dX_reg*s_sh[src].w;
		}
		__syncthreads();
		s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[64+threadIdx.x].x;
		s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[64+threadIdx.x].y;
		s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[64+threadIdx.x].z;

		s_sh[threadIdx.x].w=srcDen_dp[152*trgBox.z+threadIdx.x+64];

		__syncthreads();
		for(int src=0;src<64;src++) {
			dX_reg=s_sh[src].x-t_reg.x;
			dY_reg=s_sh[src].y-t_reg.y;
			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);

			tv_reg+=dX_reg*s_sh[src].w;
		}
		__syncthreads();
		if(threadIdx.x<24) {
			s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[128+threadIdx.x].x;
			s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[128+threadIdx.x].y;
			s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[128+threadIdx.x].z;

			s_sh[threadIdx.x].w=srcDen_dp[152*trgBox.z+threadIdx.x+128];
		}

		__syncthreads();
		for(int src=0;src<24;src++) {
			dX_reg=s_sh[src].x-t_reg.x;
			dY_reg=s_sh[src].y-t_reg.y;
			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);

			tv_reg+=dX_reg*s_sh[src].w;
		}

		if(threadIdx.x<trgBox.y)
			trgVal_dp[trgBox.x+threadIdx.x]=tv_reg*PI_4I;
//			trgVal_dp[trgBox.x+threadIdx.x]=trgBox.z;
	}//extra padding block

}

__global__ void dn_kernel_4(float *trg_dp,float *trgVal_dp,float *srcCtr_dp,float *srcRad_dp,int *trgBox_dp,float* srcDen_dp,int numAugTrg) {

	__shared__ float4 s_sh[56];
	int3 trgBox;

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numAugTrg) {
		trgBox=((int3*)trgBox_dp)[uniqueBlockId];		//start,size,box

		float3 t_reg=((float3*)trg_dp)[trgBox.x+threadIdx.x];

		float3 srcCtr=((float3*)srcCtr_dp)[trgBox.z];
		float srcRad=srcRad_dp[trgBox.z];

		float dX_reg,dY_reg,dZ_reg;
		float tv_reg=0.0;

		//every thread computes a single src body

		if(threadIdx.x<56) {	//no segfaults here

			s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[threadIdx.x].x;
			s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[threadIdx.x].y;
			s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[threadIdx.x].z;

			s_sh[threadIdx.x].w=srcDen_dp[56*trgBox.z+threadIdx.x];
		}
		__syncthreads();
		for(int src=0;src<56;src++) {
			dX_reg=s_sh[src].x-t_reg.x;

			dY_reg=s_sh[src].y-t_reg.y;

			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
//@@
dX_reg = dX_reg + (dX_reg-dX_reg);
dX_reg = fmaxf(dX_reg,0.0F);

			tv_reg+=dX_reg*s_sh[src].w;
		}

		if(threadIdx.x<trgBox.y)
			trgVal_dp[trgBox.x+threadIdx.x]=tv_reg*PI_4I;
	}//extra padding block

}

__global__ void dn_kernel_stokes_fmm_4(float *trg_dp,float *trgVal_dp,float *srcCtr_dp,float *srcRad_dp,int *trgBox_dp,float* srcDen_dp,int numAugTrg)
{

  __shared__ float3 sc_sh[56];
  __shared__ float4 sd_sh[56];
  int3 trgBox;

  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
  if(uniqueBlockId<numAugTrg) {
    trgBox=((int3*)trgBox_dp)[uniqueBlockId];		//start,size,box

    float3 t_reg=((float3*)trg_dp)[trgBox.x+threadIdx.x];

    float3 srcCtr=((float3*)srcCtr_dp)[trgBox.z];
    float srcRad=srcRad_dp[trgBox.z];

    float dX_reg,dY_reg,dZ_reg;
    float3 tv_reg={0.0F,0.0F,0.0F} ;

    //every thread computes a single src body

    if(threadIdx.x<56) {	//no segfaults here

      sc_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[threadIdx.x].x;
      sc_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[threadIdx.x].y;
      sc_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[threadIdx.x].z;
      sd_sh[threadIdx.x]=((float4*)srcDen_dp)[56*trgBox.z+threadIdx.x];
    }
    __syncthreads();
    for(int src=0;src<56;src++) {
      dX_reg=sc_sh[src].x-t_reg.x;
      dY_reg=sc_sh[src].y-t_reg.y;
      dZ_reg=sc_sh[src].z-t_reg.z;

      float invR = rsqrtf(dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);

      // following two lines set invR to zero if invR is infinity
      invR = invR + (invR-invR);
      invR = fmaxf(invR, 0.0F);

      float4 cur_pot = sd_sh[src];
      float tmp_scalar = (dX_reg*cur_pot.x + dY_reg*cur_pot.y + dZ_reg*cur_pot.z - 2*cur_pot.w)*invR*invR;
      cur_pot.x += tmp_scalar*dX_reg;
      cur_pot.y += tmp_scalar*dY_reg;
      cur_pot.z += tmp_scalar*dZ_reg;

      tv_reg.x += cur_pot.x*invR;
      tv_reg.y += cur_pot.y*invR;
      tv_reg.z += cur_pot.z*invR;
    }

    if(threadIdx.x<trgBox.y)
    {
      trgVal_dp[3*(trgBox.x+threadIdx.x)]  = tv_reg.x*PI_8I;
      trgVal_dp[3*(trgBox.x+threadIdx.x)+1]= tv_reg.y*PI_8I;
      trgVal_dp[3*(trgBox.x+threadIdx.x)+2]= tv_reg.z*PI_8I;
    }
  }//extra padding block
}


int getnumAugTrg(dnComp_t *DnC) {
	int numAugTrg=0;
	for(int i=0;i<DnC->numTrgBox;i++) {
		numAugTrg+=(int)ceil((float)DnC->trgBoxSize[i]/(float)BLOCK_HEIGHT);
	}
	return numAugTrg;
}

void gpu_down(dnComp_t *DnC) {
  GPU_MSG ("Downward (combine) pass");
	int numAugTrg = getnumAugTrg(DnC);
	if (!numAugTrg) { GPU_MSG ("==> numAugTrg == 0; skipping..."); return; }
	float *trg_dp,*trgVal_dp,*srcCtr_dp,*srcRad_dp,*srcDen_dp;
	int *trgBox_dp;	//has start and size and block
	// int trgBox[3*numAugTrg];
	vector<int> trgBox(3*numAugTrg);
	int srcDOF, trgDOF;

	make_ds_down(&trgBox[0],DnC);

	switch(DnC->kernel_type)
	{
	  case KNL_LAP_S_U:
	    srcDOF=trgDOF=1;
	    break;
	  case KNL_STK_F_U:
	    srcDOF=4;
	    trgDOF=3;
	    break;
	  default:
	    MPI_ASSERT(false);
	}


	trg_dp = gpu_calloc_float ((DnC->numTrg+BLOCK_HEIGHT) * (DnC->dim));
	srcCtr_dp = gpu_calloc_float (DnC->numTrgBox*3);
	srcRad_dp = gpu_calloc_float (DnC->numTrgBox);
	trgBox_dp = gpu_calloc_int (numAugTrg*3);
	trgVal_dp = gpu_calloc_float (DnC->numTrg*trgDOF);
	srcDen_dp = gpu_calloc_float (DnC->numTrgBox*DnC->srcDim*srcDOF);

	gpu_copy_cpu2gpu_float (trg_dp, DnC->trg_, DnC->numTrg * DnC->dim);
	gpu_copy_cpu2gpu_float (srcCtr_dp, DnC->srcCtr, DnC->numTrgBox*3);
	gpu_copy_cpu2gpu_float (srcRad_dp, DnC->srcRad, DnC->numTrgBox);
	gpu_copy_cpu2gpu_int (trgBox_dp, &trgBox[0], numAugTrg*3);
	gpu_copy_cpu2gpu_float (srcDen_dp, DnC->srcDen, DnC->numTrgBox*DnC->srcDim*srcDOF);
	cudaMemcpyToSymbol(samposDn, DnC->samPosF, sizeof(float)*DnC->srcDim*3); GPU_CE;
//	int GRID_HEIGHT=UpC->numSrcBox;
	int GRID_WIDTH=(int)ceil((float)numAugTrg/65535.0F);
	int GRID_HEIGHT=(int)ceil((float)numAugTrg/(float)GRID_WIDTH);
	dim3 GridDim(GRID_HEIGHT, GRID_WIDTH);
//	cout<<"Width: "<<GRID_WIDTH<<" HEIGHT: "<<GRID_HEIGHT<<endl;
	switch (DnC->kernel_type)
	{
	  case KNL_LAP_S_U:
	    if(DnC->srcDim==152) {
	      dn_kernel<<<GridDim,BLOCK_HEIGHT>>>(trg_dp,trgVal_dp,srcCtr_dp,srcRad_dp,trgBox_dp,srcDen_dp,numAugTrg);
	    }
	    else if(DnC->srcDim==56) {
	      dn_kernel_4<<<GridDim,BLOCK_HEIGHT>>>(trg_dp,trgVal_dp,srcCtr_dp,srcRad_dp,trgBox_dp,srcDen_dp,numAugTrg);
	    }
	    else
	    {
	      GPU_MSG ("Downward computations not implemented for this accuracy");	//Exit the process?
	      MPI_ASSERT(false);
	    }
	    GPU_CE;
	    break;
	  case KNL_STK_F_U:
	    if(DnC->srcDim==56) 
	      dn_kernel_stokes_fmm_4<<<GridDim,BLOCK_HEIGHT>>>(trg_dp,trgVal_dp,srcCtr_dp,srcRad_dp,trgBox_dp,srcDen_dp,numAugTrg);
	    else
	    {
	      GPU_MSG ("Downward computations not implemented for this accuracy");
	      MPI_ASSERT(false);
	    }
	    GPU_CE;
	    break;
	  default:
	    MPI_ASSERT(false);
	}


	gpu_copy_gpu2cpu_float (DnC->trgVal, trgVal_dp, DnC->numTrg*trgDOF);

	cudaFree(trg_dp); GPU_CE;
	cudaFree(srcCtr_dp); GPU_CE;
	cudaFree(srcRad_dp); GPU_CE;
	cudaFree(trgBox_dp); GPU_CE;
	cudaFree(trgVal_dp); GPU_CE;
	cudaFree(srcDen_dp); GPU_CE;
}
