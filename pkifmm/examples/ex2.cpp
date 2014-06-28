/* 
Parallel Kernel Independent Fast Multipole Method
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
02111-1307, USA.  


	 This example is similar to ex0.cpp but now the points are not in the unit cube. 
	 Appropriate scaling has to take place, both for the positions and the output potential. 
 */

#include <cassert>
#include <cstring>
#include "fmm3d_mpi.hpp"
#include "manage_petsc_events.hpp"
#include "sys/sys.h"
#include "parUtils.h"
#include <cstring>

using namespace std;

PetscInt  procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/3  /*dim*/; }
PetscInt  procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/3  /*dim*/; }

/* ************************************************** */
// this routine is used when the points are not contained in the unit cube
// the scale factor should be used to correct the force calculation.
int scalePointsToUnitCube( Vec points, double* out_scale, double* out_shift){
  PetscFunctionBegin;
  double min_coord,max_coord;
  PetscInt tmp;
  VecMax(points, &tmp, &max_coord);
  VecMin(points, &tmp, &min_coord);
  *out_scale = 1/(max_coord - min_coord + 0.02);
  *out_shift = -min_coord + 0.01;
  pC( VecShift(points, *out_shift));
  pC( VecScale(points, *out_scale ));
  PetscFunctionReturn(0);
}



/* ************************************************** */
int main(int argc, char** argv)
{
  PetscInitialize(&argc,&argv,"options",NULL); 
	MPI_Comm comm; comm = PETSC_COMM_WORLD;
  int mpirank;  MPI_Comm_rank(comm, &mpirank);
	int mpisize;  MPI_Comm_size(comm, &mpisize);
	int dim = 3;	
	srand48( mpirank );


	// set the number of sources and points per 
	PetscTruth flg;
	PetscInt lclnumsrc;  
	pC( PetscOptionsGetInt(0, "-numsrc", &lclnumsrc, &flg) );
	if(flg!=true)lclnumsrc=1000;	

	// setup kernel 
	PetscInt kt=111; //LAPLACE KERNEL that returns potential and its gradient.
	vector<double> tmp(2); 	tmp[0] = 1;	tmp[1] = 0.25; 
	Kernel3d_MPI knl(kt, tmp);

	// generate source positions, here we generate points 
	vector<double> srcPosarr(lclnumsrc*dim);
	for(PetscInt k=0; k<lclnumsrc; k++)
		{
			const double r=0.49;
			const double center [3] = { 1.5, 1.5, 1.5};    // notice that the points are not contained in the unit cube.
			double phi=2*M_PI*drand48();
			double theta=M_PI*drand48();
			srcPosarr[0+3*k]=center[0]+0.25*r*sin(theta)*cos(phi);
			srcPosarr[1+3*k]=center[1]+0.25*r*sin(theta)*sin(phi);
			srcPosarr[2+3*k]=center[2]+r*cos(theta);
		}

	Vec srcPosOriginal;
	VecCreateMPIWithArray(comm, lclnumsrc*dim, PETSC_DETERMINE, srcPosarr.size()? &srcPosarr[0]:0, &srcPosOriginal);

	// EX2:  scale FMM points to the unit box
	Vec srcPos;  double scale, shift;
	VecDuplicate( srcPosOriginal, &srcPos);
	VecCopy( srcPosOriginal, srcPos);
	scalePointsToUnitCube( srcPos, &scale, &shift);
	// FMM also requires a vector field called "srcNor()", which is not used 
  // in the single-layer Laplace kernel. 
	Vec srcNor;	VecDuplicate(srcPos,&srcNor); VecCopy(srcPos,srcNor); 


	// For simplicity, we will assume that the target and source points coincide.
	// Note howerver, the FMM allows different pointsets for targets and sources.
	Vec trgPos; 
	trgPos=srcPos; 

	
	//2. setup FMM
	FMM3d_MPI* fmm = new FMM3d_MPI("fmm3d_");
	fmm->srcPos()=srcPos;
	fmm->srcNor()=srcNor;
	fmm->trgPos()=trgPos;
	fmm->ctr() = Point3(0.5,0.5,0.5); // center for top level box
	fmm->rootLevel() = 1;             // must be always one. 
	fmm->knl() = knl;
 	MPI_Barrier(comm);
	/* ************************************************** */
	PetscPrintf(comm, "\n\n\t FMM SETUP BEGINS\n");
	pC( fmm->setup() );                                           // SETUP
	PetscPrintf(comm, "\t FMM SETUP ENDS\n");
	/* ************************************************** */

	// setup destroys srcPos, srcNor, trgPos (after creating new, redistributed ones) 
	srcPos=fmm->srcPos();
	trgPos=fmm->trgPos();
	srcNor=fmm->srcNor();

	// create source density and target potential vectors
	int srcDOF = knl.srcDOF();
	int trgDOF = knl.trgDOF();  // for 

	lclnumsrc = procLclNum(srcPos);
	PetscInt lclnumtrg = lclnumsrc;
	Vec srcDen;  pC( VecCreateMPI(comm, lclnumsrc*srcDOF, PETSC_DETERMINE, &srcDen) );
	double* srcDenarr; pC( VecGetArray(srcDen, &srcDenarr) );
	for(PetscInt k=0; k<lclnumsrc*srcDOF; k++)	srcDenarr[k] = drand48();
	pC( VecRestoreArray(srcDen, &srcDenarr) );


	Vec trgPot;  pC( VecCreateMPI(comm, lclnumtrg*trgDOF, PETSC_DETERMINE, &trgPot) );

	/* ************************************************** */
	pC( PetscPrintf(comm, "\n\n\t FMM EVALUATE BEGINS\n"));
	pC( fmm->evaluate(srcDen, trgPot) );                         // EVALUATE
	pC( PetscPrintf(comm, "\t FMM EVALUATE ENDS\n"));
	/* ************************************************** */

	
  // Here scale potential (since we scaled the points to [0,1]^3)                                         
  VecScale(trgPot,scale);

	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );

  PetscFinalize();
	return 0;
}
																																							 
