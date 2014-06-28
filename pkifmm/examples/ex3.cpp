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


	FMM repartitions the input points. This example demonstrates how the original
	ordering can be preserved.
	
	The fmm class maintains the original partitioning and we will use
  this information to map between the fmm partitioning and the original partitioning.
	
	The key routine is scatterCreatePnts2FMM()
	Currently, it is not possible to turn the repartitioning off. 
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


/*
 creates a PETSC VecScatter object that can be used to map between the
 user-defined partitioning of a PETSc Vec object and the FMM-defined
 partitioning of a PETSc Vec object.  "size_per_scatter_element" is
 used to allow for multi-component scatterings (for example, points
 have 3 components per PETSC Vec entry).
*/
#undef __FUNCT__
#define __FUNCT__ "scatterCreatePnts2FMM"
int scatterCreatePnts2FMM( MPI_Comm & comm,   // mpi communicator
													 unsigned int size_per_scatter_element, //block size per  
													 FMM3d_MPI &fmm, 
													 Vec pntsLayout, 
													 Vec fmmLayout,
													 VecScatter *scatter_fromInputOrdering_toFmmOrdering
													 ){
  PetscFunctionBegin;
	assert(size_per_scatter_element>0);

  IS is;
  vector<PetscInt> & pidx = fmm.let()->newSrcGlobalIndices;  

  for ( size_t i=0; i<pidx.size(); i++) pidx[i] *= size_per_scatter_element;
  ISCreateBlock(comm, size_per_scatter_element, pidx.size(),pidx.size()?&pidx[0]:0,&is);
  for (size_t i=0; i<pidx.size(); i++) pidx[i] /= size_per_scatter_element;
 
  VecScatterCreate(pntsLayout, PETSC_NULL, fmmLayout, is, scatter_fromInputOrdering_toFmmOrdering);
  ISDestroy(is);
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


	// set the number of sources and points per processor
	PetscTruth flg;
	PetscInt lclnumsrc;  
	pC( PetscOptionsGetInt(0, "-numsrc", &lclnumsrc, &flg) );
	if(flg!=true) lclnumsrc=1000;	

	// setup kernel for the fast multipole
	PetscInt kt=111; // LAPLACE KERNEL that returns the potential and its gradient at the target points.
	vector<double> tmp(2); 	tmp[0] = 1;	tmp[1] = 0.25; 
	Kernel3d_MPI knl(kt, tmp);
	int srcDOF = knl.srcDOF();
	int trgDOF = knl.trgDOF();   

	// generate source point  positions and their densities
	vector<double> srcPosarr(lclnumsrc*dim);
	vector<double> srcDenarr(lclnumsrc);
	for(PetscInt k=0; k<lclnumsrc; k++)
		{
			const double r=0.49;
			// notice that the points are not contained in the unit cube.
			const double center [3] = { 1.5, 1.5, 1.5};    
			double phi=2*M_PI*drand48();
			double theta=M_PI*drand48();
			srcPosarr[0+3*k]=center[0]+0.25*r*sin(theta)*cos(phi);
			srcPosarr[1+3*k]=center[1]+0.25*r*sin(theta)*sin(phi);
			srcPosarr[2+3*k]=center[2]+r*cos(theta);

			srcDenarr[k] = drand48();  // assign some random values for densities.
		}
	
	// create PETSc Vec objects to hold source positions and source densities.
	Vec srcPosOriginal;
	VecCreateMPIWithArray(comm, lclnumsrc*dim, PETSC_DETERMINE,  &srcPosarr[0], &srcPosOriginal);
	Vec srcDenOriginal;
	VecCreateMPIWithArray(comm, lclnumsrc, PETSC_DETERMINE, &srcDenarr[0], &srcDenOriginal);

	// scale source points to the unit box for the FMM 
  // this is related to a requirement in the tree construction algorithm
	Vec srcPos;  double scale, shift;
	VecDuplicate( srcPosOriginal, &srcPos);
	VecCopy( srcPosOriginal, srcPos);
	scalePointsToUnitCube( srcPos, &scale, &shift);
	Vec srcNor;	VecDuplicate(srcPos,&srcNor); 	VecCopy(srcPos,srcNor); 


	// For simplicity, we will assume that the target and source points coincide.
	Vec trgPos;    trgPos=srcPos; 
	// Create vector for output potential in original user-defined parallel data layout.
	Vec trgPotOriginal; 
	VecCreateMPI(comm, lclnumsrc*trgDOF, PETSC_DETERMINE, &trgPotOriginal);

	//2. setup FMM 
	// this will change the parallel paritioning of points
	FMM3d_MPI* fmm = new FMM3d_MPI("fmm3d_");
	fmm->srcPos()=srcPos;  
	fmm->trgPos()=trgPos;
	fmm->srcNor()=srcNor;
	fmm->ctr() = Point3(0.5,0.5,0.5); 
	fmm->rootLevel() = 1;             
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
	lclnumsrc = procLclNum(srcPos);

	// input density needs to be mapped to the FMM parallel partitioning layout
	Vec srcDen;
	VecCreateMPI(comm, lclnumsrc, PETSC_DETERMINE, &srcDen);
	VecScatter density_scatter;
	scatterCreatePnts2FMM(comm, srcDOF, *fmm, srcDenOriginal, srcDen, &density_scatter);
	VecScatterBegin(density_scatter, srcDenOriginal, srcDen, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(density_scatter, srcDenOriginal, srcDen, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterDestroy(density_scatter);

	// create vector to store the result of the calculation
	PetscInt lclnumtrg = lclnumsrc;
	Vec trgPot;  pC( VecCreateMPI(comm, lclnumtrg*trgDOF, PETSC_DETERMINE, &trgPot) );

	/* ************************************************** */
	pC( PetscPrintf(comm, "\n\n\t FMM EVALUATE BEGINS\n"));
	pC( fmm->evaluate(srcDen, trgPot) );                         // EVALUATE
	pC( PetscPrintf(comm, "\t FMM EVALUATE ENDS\n"));
	/* ************************************************** */
  // scale potential (since we scaled the points to [0,1]^3)
	VecScale(trgPot,scale);

	//Map potential back to the original layout.
	
	VecScatter potential_scatter;
	scatterCreatePnts2FMM(comm, trgDOF, *fmm, trgPotOriginal, trgPot, &potential_scatter);
	VecScatterBegin(potential_scatter, trgPot, trgPotOriginal, INSERT_VALUES, SCATTER_REVERSE);
	VecScatterEnd(potential_scatter, trgPot, trgPotOriginal, INSERT_VALUES, SCATTER_REVERSE);
	VecScatterDestroy(potential_scatter);


	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );
	pC( VecDestroy(trgPotOriginal));
	pC( VecDestroy(srcDenOriginal));
		 

  PetscFinalize();
	return 0;
}
																																							 
