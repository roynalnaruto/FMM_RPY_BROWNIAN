#include "headers.h"


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
	
	cout<<"SRC DOF "<<srcDOF<<endl;
	cout<<"TRG DOF "<<trgDOF<<endl;
	

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
	
//	 VecView(trgPot,PETSC_VIEWER_STDOUT_WORLD);
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

//	 VecView(trgPot,PETSC_VIEWER_STDOUT_WORLD);

	// clean up
	delete fmm;
	pC( VecDestroy(srcPos) );
	pC( VecDestroy(srcNor) );
	pC( VecDestroy(srcDen) );
	pC( VecDestroy(trgPot) );
	return 0;	
} 





void computeRpy(int nsrc, double *srcPos, double *forceVec, double *radii, double *output, double *temp_output){

	///TODO! : Don't Allocate this array again and again. : DONE. use temp_output
	
	
	part1rpy(nsrc, srcPos, forceVec, radii, output);
	part2rpy(nsrc, srcPos, forceVec, temp_output);
	
	//cout<<"*****************************PART 1*****************************"<<endl;
	//printVectors(output, nsrc, 3, cout);
	//cout<<"******************************PART 2****************************"<<endl;
	//printVectors(temp_output, nsrc, 3, cout);
	//cout<<"**********************************************************"<<endl;
	
	
	for(int i=0; i<nsrc; i++){
		for(int j=0; j<3; j++){
			output[3*i + j] += radii[i] * radii[i] * temp_output[3*i + j];
			output[3*i + j] += forceVec[3*i + j]/radii[i];  //Adding diagonal
		}
	}
	 	
	///TODO! : PostCorrection
}



void postCorrectionAll(int npos, double *srcpos, double *radii, int numpairs_p, int *pairs, 
						double *forceVec, double *output){
	
	double C1 = constantC1;
	int count = 0;
	for(int i=0; i<2*(numpairs_p); i+=2){
		if((pairs[i] <= npos) && (pairs[i+1] <= npos)){
			count++;
			postCorrection(pairs[i]-1, pairs[i+1]-1, srcpos, forceVec, radii, output);
			postCorrection(pairs[i+1]-1, pairs[i]-1, srcpos, forceVec, radii, output);		
		}
	}	
	cout<<"NON_SHELL OVERLAPPING PARTICLES COUNT FROM FMM: "<<count<<endl;	
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











