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


	 In this example, we create points on the surface of an ellipsoid
	 and we compute the Harmonic potential (Laplace kernel).

	 There are five  main steps: 
	 1. Generate the source points and densities.
	 2. Setup the kernel (Laplace, Stokes etc)
	 3. Setup the FMM data structures	
	 4. Evaluate at the target locations
	 5. Clean up
    
	 In this example, the target points are the same as the source
	 points coicide and we use the Laplace kernel.
	    See ../fmm3d_mpi/kernel3d_mpi.hpp for supported kernels
*/

#include <cassert>
#include <cstring>
#include "fmm3d_mpi.hpp"
#include "manage_petsc_events.hpp"
#include "sys/sys.h"
#include "parUtils.h"
#include <cstring>

using namespace std;

inline PetscInt  
 procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/3  /*dim*/; }
inline PetscInt  
 procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/3  /*dim*/; }


int main(int argc, char** argv)
{
	// Fist, ininalize PETSC
  PetscInitialize(&argc,&argv,"options",NULL); 
	/*
		By default, PETSC_COMM_WORLD is set to MPI_COMM_WORLD
    If you want a different communicator for FMM, you should create a new MPI comm, say MY_COMM
		set PETSC_COMM_WORLD = MY_COMM and then invoke PetscInitialize().
		See PETSc manual for more info.
	*/
	MPI_Comm comm; comm = PETSC_COMM_WORLD;
  int mpirank;  MPI_Comm_rank(comm, &mpirank);
	int mpisize;  MPI_Comm_size(comm, &mpisize);
	int dim = 3;	

	
	//--------------------------
	// STEP 1.  Generate geometry (points and densities)
	// set the number of sources and points per MPI process
	PetscTruth flg;
	PetscInt lclnumsrc;  
	pC( PetscOptionsGetInt(0, "-numsrc", &lclnumsrc, &flg) );
	if(flg!=true)lclnumsrc=1000;	



	// generate source positions randomply on the surface of an ellipse
	vector<double> srcPosarr(lclnumsrc*dim);
	srand48( mpirank );
	for(PetscInt k=0; k<lclnumsrc; k++)
		{
			const double r=0.49;
			const double center [3] = { 0.5, 0.5, 0.5};
			double phi=2*M_PI*drand48();
			double theta=M_PI*drand48();
			srcPosarr[0+3*k]=center[0]+0.25*r*sin(theta)*cos(phi);
			srcPosarr[1+3*k]=center[1]+0.25*r*sin(theta)*sin(phi);
			srcPosarr[2+3*k]=center[2]+r*cos(theta);
		}

	Vec srcPos;
	// when this vector is destroyed, srcPosarr won't be freed;
	// srcPosarr will be freed by it's destructor in the end of its
	// scope
	VecCreateMPIWithArray(comm, 
												lclnumsrc*dim, PETSC_DETERMINE, 
												srcPosarr.size()? &srcPosarr[0]:0, &srcPos);
	Vec srcNor;
	VecDuplicate(srcPos,&srcNor); 
	VecCopy(srcPos,srcNor); 
	Vec trgPos;  
  trgPos=srcPos; // we do assume petsc type "Vec" is actually a pointer 

	//--------------------------
	//STEP 2. 
	// First setup FMM kernel, 
	PetscInt kt=111;  // 111 defines the Laplace kernel, 
	vector<double> tmp(2); 	tmp[0] = 1;	tmp[1] = 0.25;  // kernel-related constants
	Kernel3d_MPI knl(kt, tmp); //


	//--------------------------
	//STEP 3: setup FMM data structures
	FMM3d_MPI* fmm = new FMM3d_MPI("fmm3d_");
	fmm->srcPos()=srcPos;
	fmm->srcNor()=srcNor;
	fmm->trgPos()=trgPos;
	fmm->ctr() = Point3(0.5,0.5,0.5); // center for top level box
	fmm->rootLevel() = 1;             // must be always one. 
	fmm->knl() = knl;
 	MPI_Barrier(comm);
	/* ************************************************** */
	// pC() is a macro for error checking 
	pC( PetscPrintf(comm, "\n\n\t FMM SETUP BEGINS\n"));
	pC( fmm->setup() );                                           // SETUP
	pC( PetscPrintf(comm, "\t FMM SETUP ENDS\n"));
	/* ************************************************** */

	// setup destroys srcPos, srcNor, trgPos (after creating new, redistributed ones) 
	srcPos=fmm->srcPos();
	trgPos=fmm->trgPos();
	srcNor=fmm->srcNor();


	// create source density and target potential vectors
	lclnumsrc = procLclNum(srcPos);
	PetscInt lclnumtrg = lclnumsrc;
	Vec srcDen;  pC( VecCreateMPI(comm, lclnumsrc, PETSC_DETERMINE, &srcDen) );
	double* srcDenarr; pC( VecGetArray(srcDen, &srcDenarr) );
	for(PetscInt k=0; k<lclnumsrc; k++)	srcDenarr[k] = drand48();
	pC( VecRestoreArray(srcDen, &srcDenarr) );
	Vec trgPot;  pC( VecCreateMPI(comm, lclnumtrg, PETSC_DETERMINE, &trgPot) );


	//--------------------------
	// STEP 4. Evaluate sum
	/* ************************************************** */
	pC( PetscPrintf(comm, "\n\n\t FMM EVALUATE BEGINS\n"));
	pC( fmm->evaluate(srcDen, trgPot) );                         // EVALUATE
	pC( PetscPrintf(comm, "\t FMM EVALUATE ENDS\n"));
	/* ************************************************** */

	// check error on 10 points (at every MPI rank)
  double rerr;
	pC( fmm->check(srcDen, trgPot, 10, rerr) );
	pC( PetscPrintf(comm, "\n\n \t FMM relative error is: %e\n\n", rerr) );


	//--------------------------
	// STEP 5:
	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );

  PetscFinalize();
	return 0;
}
																																							 
