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

/**
 * \file src/upComp.h
 * \brief Implements a structure for upward computation
 */

#if !defined (INC_UPCOMP_H)
#define INC_UPCOMP_H /*!< upComp.h included */

/* ============================================================================
 */
/* \brief upComp structure */


typedef struct upComp {

/*	P->src_ has the form x1 y1 z1 d1 x2 y2 z2 d2 ......
	P->trgVal has the form t1 t2 t3.....*/
  /* For Stokes velocity kernel P->src_ has the form 
   * x1 y1 z1 dx1 dy1 dz1 x2 y2 z2 dx2 dy2 dz2 ...
   * P->trgVal has the form tx1 ty1 tz1 tx2 ty2 tz2 ..... 
   */ 

  int tag;
  int numSrc;     /* number of source points */
  int numSrcBox;  /* number of source boxes */
  int dim;	  /* dimension */
  int kernel_type;
  float kernel_coef[2];   // kernel coefficients;

#ifdef DS_ORG
  float* src_;    /* source coordinates */
#endif
  float* srcDen; /* source density values of size numSrc */
  float* trgVal; /* target potentials */
  int trgDim;	//296 for 6, 152 for 4, forget about 8

  int* srcBoxSize; /* number of points in source boxes */
  float* trgCtr;   /* center of the target box */
  float* trgRad;   /* radius of the target box */	//merge with ctr? float4
//#ifndef SAVE_ME_FROM_FLTNUMMAT
//  FltNumMat samPos; /* sample position */
//#endif
  float *samPosF;	//pointer to sampos array


} upComp_t;

#endif

/* eof */
