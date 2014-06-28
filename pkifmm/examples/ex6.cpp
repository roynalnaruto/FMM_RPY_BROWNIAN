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
#include <iostream>
#include <fstream>
//#include <stdlib.h>



using namespace std;



#define constantC1 0.75
#define constantC2 0.50


PetscInt  procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/3  /*dim*/; }
PetscInt  procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/3  /*dim*/; }




void postCorrectionAll(int npos, double *srcpos, double *radii, int numpairs_p, int *pairs, 
						double *forceVec, double *output);

void postCorrection(int index1, int index2, double *srcPosition, double *forceVector, double *radii, double *output);


int part1rpy(int numsrc, double *srcposition, double *forceVec, double *radii, double* &output);
int part2rpy(int numsrc, double *srcposition, double *forceVec, double* &output);





int part1rpy(int numsrc, double *srcposition, double *forceVec, double *radii, double* &output){
	
	MPI_Comm comm; comm = PETSC_COMM_WORLD;
	int mpirank;  MPI_Comm_rank(comm, &mpirank);
	int mpisize;  MPI_Comm_size(comm, &mpisize);
	int dim = 3;	

	// set the number of sources and points per mpi rank
	PetscInt lclnumsrc;  
	lclnumsrc=numsrc;	

	srand48(mpirank);

	Vec srcPos;
	// when this vector is destroyed, srcposition won't be freed; 
	VecCreateMPIWithArray(comm, lclnumsrc*dim, 
												PETSC_DETERMINE, srcposition, &srcPos);
	
	Vec srcNor;
	VecDuplicate(srcPos,&srcNor); VecCopy(srcPos,srcNor); 
	Vec trgPos;  
	trgPos=srcPos; // we do assume petsc type "Vec" is actually a pointer 

	// setup fast multipole kernel 
	PetscInt kt=431; //RPY tensor part1
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
	Vec srcDen;
	pC( VecCreateMPI(comm, lclnumsrc*srcDOF, PETSC_DETERMINE, &srcDen) );
	
	double* srcDenarr;
	pC( VecGetArray(srcDen, &srcDenarr) );
	
	
	for(PetscInt k=0; k<lclnumsrc; k++){
		srcDenarr[k*srcDOF + 0] = radii[k];
		srcDenarr[k*srcDOF + 1] = forceVec[3*k + 0];
		srcDenarr[k*srcDOF + 2] = forceVec[3*k + 1];
		srcDenarr[k*srcDOF + 3] = forceVec[3*k + 2];
	}
	
	pC( VecRestoreArray(srcDen, &srcDenarr) );
	Vec trgPot;  pC( VecCreateMPI(comm, lclnumtrg*trgDOF, PETSC_DETERMINE, &trgPot) );

	/* ************************************************** */
	pC( PetscPrintf(comm, "\n\n\t FMM EVALUATE BEGINS\n"));
	pC( fmm->evaluate(srcDen, trgPot) );                         // EVALUATE
	pC( PetscPrintf(comm, "\t FMM EVALUATE ENDS\n"));
	/* ************************************************** */
	
	double *tempOutput;
	pC( VecGetArray(trgPot, &tempOutput) );
	
	 VecView(trgPot,PETSC_VIEWER_STDOUT_WORLD);
	 //output = tempOutput;


	for(PetscInt k=0; k<lclnumsrc*trgDOF; k++){
		output[k] = tempOutput[k];
	}

	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );
	
	return 0;	
} 





int part2rpy(int numsrc, double *srcposition, double *forceVec, double* &output){
	
	
	MPI_Comm comm; comm = PETSC_COMM_WORLD;
	int mpirank;  MPI_Comm_rank(comm, &mpirank);
	int mpisize;  MPI_Comm_size(comm, &mpisize);
	int dim = 3;	

	// set the number of sources and points per mpi rank
	PetscInt lclnumsrc;  
	lclnumsrc=numsrc;	

	srand48(mpirank);

	Vec srcPos;
	// when this vector is destroyed, srcposition won't be freed; 
	VecCreateMPIWithArray(comm, lclnumsrc*dim, 
												PETSC_DETERMINE, srcposition, &srcPos);
	
	Vec srcNor;
	VecDuplicate(srcPos,&srcNor); VecCopy(srcPos,srcNor); 
	Vec trgPos;  
	trgPos=srcPos; // we do assume petsc type "Vec" is actually a pointer 

	// setup fast multipole kernel 
	PetscInt kt=432; //RPY tensor part2
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
	Vec srcDen;
	pC( VecCreateMPI(comm, lclnumsrc*srcDOF, PETSC_DETERMINE, &srcDen) );
	
	double* srcDenarr;
	pC( VecGetArray(srcDen, &srcDenarr) );
	
	
	for(PetscInt k=0; k<lclnumsrc; k++){
		srcDenarr[k*srcDOF + 0] = forceVec[3*k + 0];
		srcDenarr[k*srcDOF + 1] = forceVec[3*k + 1];
		srcDenarr[k*srcDOF + 2] = forceVec[3*k + 2];
	}
	
	pC( VecRestoreArray(srcDen, &srcDenarr) );
	Vec trgPot;  pC( VecCreateMPI(comm, lclnumtrg*trgDOF, PETSC_DETERMINE, &trgPot) );

	/* ************************************************** */
	pC( PetscPrintf(comm, "\n\n\t FMM EVALUATE BEGINS\n"));
	pC( fmm->evaluate(srcDen, trgPot) );                         // EVALUATE
	pC( PetscPrintf(comm, "\t FMM EVALUATE ENDS\n"));
	/* ************************************************** */
	
	
	double *tempOutput;
	pC( VecGetArray(trgPot, &tempOutput) );
	//output = tempOutput;


	for(PetscInt k=0; k<lclnumsrc*trgDOF; k++){
		output[k] = tempOutput[k];
	}

	 VecView(trgPot,PETSC_VIEWER_STDOUT_WORLD);

	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );
	return 0;	
} 





void postCorrectionAll(int npos, double *srcpos, double *radii, int numpairs_p, int *pairs, 
						double *forceVec, double *output){
	
	double C1 = constantC1;
	for(int i=0; i<2*(numpairs_p); i+=2){
		if((pairs[i] <= npos) && (pairs[i+1] <= npos)){
			postCorrection(pairs[i], pairs[i+1], srcpos, forceVec, radii, output);
			postCorrection(pairs[i+1], pairs[i], srcpos, forceVec, radii, output);		
		}
	}
}




//Corrects stuff for index1 only!
void postCorrection(int index1, int index2, double *srcPosition, double *forceVector, double *radii, double *output){
	
	double effectiveRadius = (radii[index1] * radii[index1]) + (radii[index2] * radii[index2]);
	effectiveRadius = sqrt(effectiveRadius/2.0);
	
	double C1 = constantC1;
	double C2 = constantC2 * effectiveRadius * effectiveRadius;
	double C3 = (4 * constantC1)/(3 * effectiveRadius);
	double C4 = C3 * (3/(32 * effectiveRadius));
	
		
	double distanceVec[3];
	double pc[3][3];  //PostCorrection 3*3 matrix
	double modDistance=0.0;
	double identity=0.0;
	double rCrossR=0.0;
	
	for(int i=0;i<3;i++){
		distanceVec[i] = srcPosition[3*index1 + i] - srcPosition[3*index2 + i];
		modDistance += distanceVec[i] * distanceVec[i];
	}	
	
	modDistance = sqrt(modDistance);
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			identity = (i==j)? 1.0 : 0;
			rCrossR = distanceVec[i] * distanceVec[j];
			pc[i][j] = ((1 - ((9 * modDistance)/(32 * effectiveRadius))) * identity) * C3;
			pc[i][j] += (C4 * rCrossR)/modDistance;
			
			pc[i][j] -= (C1/modDistance) * (identity  + (rCrossR/(modDistance * modDistance))) ;
			pc[i][j] -= (C2/(modDistance * modDistance * modDistance))
							* (identity - ((3 * rCrossR)/(modDistance * modDistance)));
		}
	}
	
	for(int i=0;i<3;i++){
		output[3*index1 + i] += (pc[i][0] * forceVector[3*index2+0]) + 
								(pc[i][1] * forceVector[3*index2+1]) + 
								(pc[i][2] * forceVector[3*index2+2]);
	}
}




















/*
 * 
 * name: main
 * @param
 * @return
 * 
 */
int main(int argc, char** argv)
{
   PetscInitialize(&argc,&argv,"options",NULL); 
   
	// set the number of sources and points per mpi rank
	PetscTruth flg;
	PetscInt npos;  
	pC( PetscOptionsGetInt(0, "-numsrc", &npos, &flg) );
	if(flg!=true)npos=1000;	
	
	int dim = 3;
	int lclnumsrc = npos;
	
	double *srcPosarr = new double[lclnumsrc*dim];
	double *radii = new double[lclnumsrc];
	double *force = new double[lclnumsrc*dim];
	
	ifstream file1, file2, file3;
	file1.open ("pos.txt");
	file2.open("force.txt");
	file3.open("radii.txt");
	
	for(PetscInt k=0; k<lclnumsrc; k++){
		file1>>srcPosarr[3*k+0]>>srcPosarr[3*k+1]>>srcPosarr[3*k+2];
		file2>>force[3*k+0]>>force[3*k+1]>>force[3*k+2];
		file3>>radii[k];
	}	
	file1.close();	
	file2.close();
	file3.close();
	
	
	double *part1output = new double[lclnumsrc*dim];	
	double *part2output = new double[lclnumsrc*dim];
	
	
	part1rpy(lclnumsrc, srcPosarr, force, radii, part1output);
	//What's going on here?
	//PetscFinalize();
	//PetscInitialize(&argc,&argv,"options",NULL); 
	part2rpy(lclnumsrc, srcPosarr, force, part2output);
	
	for(int i=0; i<lclnumsrc; i++){
		for(int j=0; j<3; j++){
			part1output[3*i + j] += radii[i] * radii[i] * part2output[3*i + j];
			part1output[3*i + j] += force[3*i + j]/radii[i];  //Adding diagonal
		}
		cout<<part1output[3*i + 0]<<" "<<part1output[3*i + 1]<<" "<<part1output[3*i + 2]<<endl;	
	}
	

	PetscFinalize();
	return 0;
}
																																							 
