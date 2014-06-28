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
	 and we compute the Harmonic potential (Laplace kernel) AND the
	 gradient of the potential.  Other than that everything else is the
	 same as ex0.cpp
	 
	 To compute the gradients, we select a different kernel and the 
	 output is longer. The potentials (u) are ouput blockwise per point:
	 u_pnt0, ux_pnt0, uy_pnt0, uz_pnt0, 	 u_pnt1, ux_pnt1, uy_pnt1, uz_pnt1, ...

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


int main(int argc, char** argv)
{
  PetscInitialize(&argc,&argv,"options",NULL); 
	MPI_Comm comm; comm = PETSC_COMM_WORLD;
  int mpirank;  MPI_Comm_rank(comm, &mpirank);
	int mpisize;  MPI_Comm_size(comm, &mpisize);
	int dim = 3;	

	// set the number of sources and points per mpi rank
	PetscTruth flg;
	PetscInt lclnumsrc;  
	pC( PetscOptionsGetInt(0, "-numsrc", &lclnumsrc, &flg) );
	if(flg!=true)lclnumsrc=1000;	

	// generate source positions, here we generate points 
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
	// when this vector is destroyed, srcPosarr won't be freed; srcPosarr will be freed by it's destructor in the end of its scope
	VecCreateMPIWithArray(comm, lclnumsrc*dim, 
												PETSC_DETERMINE, srcPosarr.size()? &srcPosarr[0]:0, &srcPos);
	Vec srcNor;
	VecDuplicate(srcPos,&srcNor); VecCopy(srcPos,srcNor); 
	Vec trgPos;  
  trgPos=srcPos; // we do assume petsc type "Vec" is actually a pointer 

	// setup fast multipole kernel 
	PetscInt kt=141; //LAPLACE KERNEL that returns potential and its gradient.
	vector<double> tmp(2); 	tmp[0] = 1;	tmp[1] = 0.25; 
	Kernel3d_MPI knl(kt, tmp);


	// setup FMM
	FMM3d_MPI* fmm = new FMM3d_MPI("fmm3d_");
	fmm->srcPos()=srcPos;
	fmm->srcNor()=srcNor;
	fmm->trgPos()=trgPos;
	fmm->ctr() = Point3(0.5,0.5,0.5); // center for top level box
	fmm->rootLevel() = 1;             // must be always one. 
	fmm->knl() = knl;
 	MPI_Barrier(comm);
	/* ************************************************** */
	pC( PetscPrintf(comm, "\n\n\t FMM SETUP BEGINS\n"));
	pC( fmm->setup() );                                           // SETUP
	pC( PetscPrintf(comm, "\t FMM SETUP ENDS\n"));
	/* ************************************************** */

	// setup destroys srcPos, srcNor, trgPos (after creating new, redistributed ones) 
	srcPos=fmm->srcPos();
	trgPos=fmm->trgPos();
	srcNor=fmm->srcNor();

	int srcDOF = knl.srcDOF();
	int trgDOF = knl.trgDOF();  

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


	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );

  PetscFinalize();
	return 0;
}
																																							 
