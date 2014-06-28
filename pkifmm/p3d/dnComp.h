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
 * \file src/dnComp.h
 * \brief Implements a structure for downward computation
 */

#if !defined (INC_DNCOMP_H)
#define INC_DNCOMP_H /*!< dnComp.h included */

/* ============================================================================
 */
/* \brief dnComp structure */
typedef struct dnComp {

/*	DnC->trg_ has the form x1 y1 z1 x2 y2 z2 ......
	DnC->trgVal has the form t1 t2 t3.....*/

  int tag;
  int numTrg;     /* number of target points */
  int numTrgBox;  /* number of target boxes */
  int dim;	  /* dimension */
  int kernel_type;
  float kernel_coef[2];   // kernel coefficients;

#ifdef DS_ORG
  float* trg_;    /* target coordinates */
#endif
  float* trgVal;  /* target potentials */
  int srcDim;	//152 for 6, 56 for 4, forget about 8

  int* trgBoxSize; /* number of points in target boxes */
  float* srcCtr;   /* center of the source box */
  float* srcRad;   /* radius of the source box */
//  FltNumMat samPos; /* sample position */
  float* srcDen;

  float *samPosF;	//pointer to sampos array

} dnComp_t;

#endif

/* eof */
