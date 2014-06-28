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
 * \file src/point3d.h
 * \brief Implements a structure for u-list computation
 */

#if !defined (INC_POINT3D_H)
#define INC_POINT3D_H /*!< point3d.h included */

#if defined (__cplusplus)
extern "C" {
#endif

/* ============================================================================
 */
//#define DS_ORG
/* \brief point3d structure */
typedef struct point3d {

/*	P->src_ has the form x1 y1 z1 d1 x2 y2 z2 d2 ......
	P->trg has the form x1 y1 z1 x2 y2 z2
	P->trgVal has the form t1 t2 t3.....*/

  int numSrc;     /* number of source points */
  int numTrg;     /* number of target points */
  int numTrgBox;  /* number of target boxes */
  int numSrcBox;  /* number of source boxes */
  int dim;	  /* dimension */
  int kernel_type;


#ifdef USE_DOUBLE
  double kernel_coef[2];   // kernel coefficients;

#ifdef DS_ORG
  double* src_;    /* source coordinates */
  double* trg_;    /* target coordinates */
#else
  double* sx_;    /* source x coordinates */
  double* sy_;    /* source y coordinates */
  double* sz_;    /* source z coordinates */

  double* tx_;    /* target x coordinates */
  double* ty_;    /* target y coordinates */
  double* tz_;    /* target z coordinates */
#endif
//   double* srcDen; /* source density values of size numSrc */
  double* trgVal; /* target values of size numTrg */
//   double *trgValC;	//TODO: remove

#else

  float kernel_coef[2];   // kernel coefficients;

#ifdef DS_ORG
  float* src_;    /* source coordinates */
  float* trg_;    /* target coordinates */
#else
  float* sx_;    /* source x coordinates */
  float* sy_;    /* source y coordinates */
  float* sz_;    /* source z coordinates */

  float* tx_;    /* target x coordinates */
  float* ty_;    /* target y coordinates */
  float* tz_;    /* target z coordinates */
#endif
//   float* srcDen; /* source density values of size numSrc */
  float* trgVal; /* target values of size numTrg */
//   float *trgValC;	//TODO: remove
#endif


  int** uList; /* u-list for each target box which is a leaf node */
  int* uListLen; /* u-list length for each target box */

  int* srcBoxSize; /* number of points in source boxes */
  int* trgBoxSize; /* number of points in target boxes */

} point3d_t;

#if defined (__cplusplus)
}
#endif
#endif

/* eof */
