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
02111-1307, USA.  */

#include <limits>
#include <cstring>
#include <fstream>
#include "fmm3d_mpi.hpp"
#include "manage_petsc_events.hpp"
#include "sys/sys.h"
#include "parUtils.h"
#ifdef HAVE_TAU
#include <Profile/Profiler.h>
#endif

#include "sstream"

using namespace std;

/*! For a certain processor, return local number of positions from Vec pos 
 * See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecGetLocalSize.html for more info */
PetscInt  procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/3  /*dim*/; }
/*! For a certain processor, return global number of positions from Vec pos */
PetscInt  procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/3  /*dim*/; }

void DumpCenterAndLeaks(Vec x, ofstream & CL_file)
{
  MPI_Comm comm=PETSC_COMM_WORLD;
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  // find minimal z
  double* x_arr;
  VecGetArray(x, &x_arr) ;
  double min_z = std::numeric_limits<double>::infinity();
  double global_min_z;
  int lclnumsrc = procLclNum(x);
  for(int i=0; i<lclnumsrc; i++)
    min_z = min(min_z,x_arr[3*i+2]);
  VecRestoreArray(x, &x_arr);

  MPI_Allreduce(&min_z,&global_min_z,1,MPI_DOUBLE,MPI_MIN,comm);

  vector<double> local_cm(3,0);
  vector<double> global_cm(3);

  PetscInt local_low=0, global_low;
  VecGetArray(x, &x_arr) ;
  for(int i=0; i<lclnumsrc; i++)
    if (x_arr[3*i+2]<global_min_z+3)
    {
      local_cm[0] += x_arr[3*i];
      local_cm[1] += x_arr[3*i+1];
      local_cm[2] += x_arr[3*i+2];
      local_low++;
    }
  VecRestoreArray(x, &x_arr);

  MPI_Allreduce(&local_cm[0],&global_cm[0],3,MPI_DOUBLE,MPI_SUM,comm);
  MPI_Allreduce(&local_low,&global_low,1,MPIU_INT,MPI_SUM,comm);

  for (int i=0; i<3; i++)
    global_cm[i] /= global_low;

  // now calculate the number of ``leaked'' particles
  PetscInt global_leaked, local_leaked=0;
  VecGetArray(x, &x_arr) ;
  for(int i=0; i<lclnumsrc; i++)
  {
    // calculate square of distance
    double distSq = 0;
    for (int j=0; j<3; j++)
      distSq += (x_arr[3*i+j]-global_cm[j])*(x_arr[3*i+j]-global_cm[j]);

    if (distSq > 2*2)
      local_leaked++;
  }
  VecRestoreArray(x, &x_arr);

  MPI_Allreduce(&local_leaked,&global_leaked,1,MPIU_INT,MPI_SUM,comm);
  if (!mpirank)
  {
    for (int i=0; i<3; i++)
      CL_file<<global_cm[i]<<" ";
    CL_file<<global_leaked<<endl;
  }
}


int eval_rhs(Vec & x, Vec & ids, Vec & k1, Kernel3d_MPI & knl, double eps, bool redistr )
  // k1 is created inside, so should  be destroyed outside
  // if redistr is true, "x" is redistributed; otherwise not;  in any case layout of k1 will be compatible with that of "x"
{
  PetscLogEventBegin(droplets_rhs,0,0,0,0);
  Vec l;
  Vec xx;
  Vec srcNor;

  MPI_Comm comm; comm = PETSC_COMM_WORLD;
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  if (redistr)
    xx=x;   // we do assume "Vec" is a pointer
  else
  {
    VecDuplicate(x,&xx);
    VecCopy(x, xx);
  }

  // allocate fmm 
  FMM3d_MPI* fmm = new FMM3d_MPI("fmm3d_");
  fmm->srcPos()=xx;

  VecDuplicate(xx,&srcNor);
  VecSet(srcNor,0);
  fmm->srcNor()=srcNor;

  fmm->trgPos()=xx;

  // at the moment, we shall use the box [0,1]^3 and shift&scale things so they fit there
  double min_coord, max_coord;
  PetscInt dummy;
  VecMax(xx,&dummy, &max_coord);
  VecMin(xx,&dummy, &min_coord);

  double scale_factor = 1/(max_coord-min_coord+0.02);
  double shift = -min_coord + 0.01;
  VecShift(xx, shift );
  VecScale(xx, scale_factor);

  fmm->ctr() = Point3(0.5,0.5,0.5); // CENTER OF THE TOPLEVEL BOX
  fmm->rootLevel() = 1;         // 2^(-rootlvl) is the RADIUS OF THE TOPLEVEL BOX
  
  fmm->knl() = knl;

  PetscLogStagePush(stages[1]);

  // setup destroys srcPos, srcNor, trgPos (after creating new, redistributed ones) 
  MPI_Barrier(comm);
  PetscLogEventBegin(fmm_setup_event,0,0,0,0);
  pC(fmm->setup()) ;
  PetscLogEventEnd(fmm_setup_event,0,0,0,0);

  PetscLogStagePop();
  PetscLogStagePush(stages[2]);

  // now sources and targets are re-distributed, pointers srcPos, trgPos, srcNor are invalid
  if (redistr)
    x=fmm->srcPos();
  xx = fmm->srcPos();
  srcNor=fmm->srcNor();

  int lclnumsrc = procLclNum(xx);

  int srcDOF = knl.srcDOF();
  int trgDOF = knl.trgDOF();
  if (!mpirank)
    cout<<"srcDOF="<<srcDOF<<" trgDOF="<<trgDOF<<endl;

  Vec srcDen;  pC( VecCreateMPI(comm, lclnumsrc*srcDOF, PETSC_DETERMINE, &srcDen) );
  double* srcDenarr; pC( VecGetArray(srcDen, &srcDenarr) );
  for(PetscInt k=0; k<lclnumsrc; k++)
  {
    srcDenarr[3*k] = 0;
    srcDenarr[3*k+1] = 0;
    srcDenarr[3*k+2] = -1; // -eps*eps;
  }
  pC( VecRestoreArray(srcDen, &srcDenarr) );

  VecDuplicate(xx,&l);

  //3. run fmm 
  PetscLogStagePop();
  PetscLogStagePush(stages[3]);

  MPI_Barrier(comm);
  PetscLogEventBegin(fmm_eval_event,0,0,0,0);
  pC( fmm->evaluate(srcDen, l) );
  PetscLogEventEnd(fmm_eval_event,0,0,0,0);
  PetscLogStagePop();

  // re-scale the result (since we scaled the points)
  VecScale(l,scale_factor);
  // finalize computation of RHS:
  VecAYPX(l,6*M_PI*eps,srcDen);

  if (!redistr)   // if we need to scatter l back to layout of x
  {
    vector<PetscInt>  & newSrcGlobalIndices = fmm->let()->newSrcGlobalIndices;

    // construct PETSc index set to redistribute source coordinates vector via VecScatter
    // basically, IS is a list of new global indices for local entries of _srcPos
    IS potIS;
    for (size_t i=0; i<newSrcGlobalIndices.size(); i++)
      newSrcGlobalIndices[i] *= 3;

    ISCreateBlock(comm,3, newSrcGlobalIndices.size(),newSrcGlobalIndices.size()?&newSrcGlobalIndices[0]:0,&potIS);

    // "newSrcGlobalIndices" might be used later with possibly different block size, so divide it back
    for (size_t i=0; i<newSrcGlobalIndices.size(); i++)
      newSrcGlobalIndices[i] /= 3;

    // construct new vector and scatter context to redistribute RHS vector via VecScatter back to the layout of "x"
    VecDuplicate(x,&k1);

    VecScatter ctx;
    VecScatterCreate(k1,PETSC_NULL,l,potIS, &ctx);
    ISDestroy(potIS);

    // do the actual communication 
    VecScatterBegin(ctx,l,k1,INSERT_VALUES,SCATTER_REVERSE);
    VecScatterEnd  (ctx,l,k1,INSERT_VALUES,SCATTER_REVERSE);
    VecScatterDestroy(ctx);

    VecDestroy(l);
  }
  else 
  {
    k1=l;
    // undo scaling and shifting of x (xx and x point to same structure)
    VecScale(xx, 1/scale_factor);
    VecShift(xx, -shift );
  }

  // redistribute IDs
  if(redistr && ids)
  {
    Vec new_ids;
    vector<PetscInt>  & newSrcGlobalIndices = fmm->let()->newSrcGlobalIndices;

    // construct PETSc index set to redistribute ids vector via VecScatter
    // basically, IS is a list of new global indices for local entries of _srcPos
    IS potIS;

    ISCreateBlock(comm,1, newSrcGlobalIndices.size(),newSrcGlobalIndices.size()?&newSrcGlobalIndices[0]:0,&potIS);

    // construct new vector and scatter context to redistribute ids vector via VecScatter according to new layout of "x"
    PetscInt lclnumsrc;
    VecGetLocalSize(x,&lclnumsrc);
    assert(lclnumsrc%3==0);
    lclnumsrc /= 3;
    VecCreateMPI(comm, lclnumsrc, PETSC_DETERMINE, &new_ids);

    VecScatter ctx;
    VecScatterCreate(ids,PETSC_NULL,new_ids,potIS, &ctx);
    ISDestroy(potIS);

    // do the actual communication 
    VecScatterBegin(ctx,ids,new_ids,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd  (ctx,ids,new_ids,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterDestroy(ctx);

    VecDestroy(ids);
    ids = new_ids;
  }

  delete fmm;


  if (!redistr)
    VecDestroy(xx);
  VecDestroy(srcDen);
  pC( VecDestroy(srcNor) );

  PetscLogEventEnd(droplets_rhs,0,0,0,0);
  return 0;
}


int main(int argc, char** argv)
{
#ifdef HAVE_TAU
  TAU_PROFILE_TIMER(tau_eval_timer,"fmm eval", "void (void)", TAU_USER);
  // TAU_PROFILE("int main(int, char **)", " ", TAU_DEFAULT);
  TAU_INIT(&argc, &argv); 
#ifndef TAU_MPI
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */
#endif
  PetscInitialize(&argc,&argv,NULL,NULL); 
  PetscPopSignalHandler();
  pC(PetscMemorySetGetMaximumUsage());
  ot::RegisterEvents();
  registerPetscEvents();

  // make "Main Stage" invisible
//   StageLog CurrentStageLog;
//   PetscLogGetStageLog(&CurrentStageLog);
//   int CurrentStage;
//   StageLogGetCurrent(CurrentStageLog, &CurrentStage);
//   pA(CurrentStage!=-1);
//   StageLogSetVisible(CurrentStageLog, CurrentStage, PETSC_FALSE);

  MPI_Comm comm; comm = PETSC_COMM_WORLD;
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  int dim = 3;
  srand48( mpirank + 257 );

  PetscTruth flg = PETSC_FALSE;

  //1. allocate random data
  PetscInt numsrc;  
  PetscOptionsGetInt(0, "-numsrc", &numsrc, &flg) ;
  pA(flg==PETSC_TRUE);

  if (!mpirank)
    cout<<"Using numsrc=" << numsrc << endl;

  PetscInt kt= 311; // Stokes velocity kernel
  vector<double> tmp(2); 
  tmp[0] = 1;
  tmp[1] = 0.25; // ignored for Stokes velocity kernel
  Kernel3d_MPI knl(kt, tmp);

  PetscTruth doing_isogranular;
  PetscOptionsHasName(0,"-isogranular",&doing_isogranular);

  const int distLen=50;
  char distribution[distLen];
  {
    PetscTruth distribution_flg;
    PetscOptionsGetString(0,"-distribution",distribution,distLen,&distribution_flg);
    assert(distribution_flg);
  }

  PetscInt lclnumsrc;

  if (doing_isogranular)
    // interpret "-numsrc" as per-processor number of sources
  {
    if (!mpirank)
      cout<<"Interpreting -numsrc as per-processor"<<endl;
    lclnumsrc = numsrc;
  }
  else
    // otherwise interpret "-numsrc" as global number of sources
  {
    if (!mpirank)
      cout<<"Interpreting -numsrc as global (NOT per-processor)"<<endl;
    lclnumsrc = (numsrc+mpirank)/mpisize;
  }

  // generate source positions
  vector<double> srcPosarr(lclnumsrc*dim);
  if (0==strcmp(distribution,"ball"))
  {
    if (!mpirank)
      cout<<"Creating initial ball"<<endl;

    for(PetscInt k=0; k<lclnumsrc; k++)
    {
      double r2=0;
      for(int d=0; d<dim; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = tmp;
      }
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }
  }
  else if(0==strcmp(distribution,"two_balls_vert"))
  {
    if (!mpirank)
      cout<<"Creating 2 balls, one above other"<<endl;

    for(PetscInt k=0; k<lclnumsrc/2; k++)
    {
      double r2=0;
      for(int d=0; d<dim; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = tmp;
      }
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }

    for(PetscInt k=lclnumsrc/2; k<lclnumsrc; k++)
    {
      double r2=0;
      for(int d=0; d<dim; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = (d==2? tmp+4:tmp) ;
      }
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }
  }
  else if(0==strcmp(distribution,"3_balls_horiz"))
  {
    if (!mpirank)
      cout<<"Creating 3 balls, on a horizontal line"<<endl;

    for(PetscInt k=0; k<lclnumsrc/3; k++)
    {
      double r2=0;
      for(int d=0; d<dim; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = tmp;
      }
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }

    for(PetscInt k=lclnumsrc/3; k<2*lclnumsrc/3; k++)
    {
      double r2=0;
      for(int d=0; d<dim; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = (d==0? tmp+4:tmp) ;
      }
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }

    for(PetscInt k=2*lclnumsrc/3; k<lclnumsrc; k++)
    {
      double r2=0;
      for(int d=0; d<dim; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = (d==0? tmp+8:tmp) ;
      }
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }
  }
  else if(0==strcmp(distribution,"mesh_of_balls"))
  {
    // for now, we'll do 10x10 horizontal mesh; ball radiuses will be 1, ball centers will have coordinates of the form
    // (2k, 2l, 0)    k,l,m >= 0

    if (!mpirank)
      cout<<"Creating 10x10 2D mesh of balls, on a horizontal line"<<endl;

    for(PetscInt k=0; k<lclnumsrc; k++)
    {
      double r2=0;
      int modules[]={1,10};
      for(int d=0; d<2; d++)
      {
	double tmp = 1-2*drand48();
	r2 += tmp*tmp;
	srcPosarr[d+dim*k] = tmp + 4*((k / modules[d]) % 10);
      }
      double tmp = 1-2*drand48();
      r2 += tmp*tmp;
      srcPosarr[2+dim*k] = tmp;
      if (r2 >= 1) // skip point outside ball
      {
	k--;
	continue;
      }
    }
  }
  else
    SETERRQ(1,"Invalid distribution of sources selected");

  Vec x;   // vector of particle coordinates
  // when this vector is destroyed, srcPosarr won't be freed; srcPosarr will be freed by it's destructor in the end of its scope
  VecCreateMPIWithArray(comm, lclnumsrc*dim, PETSC_DETERMINE, srcPosarr.size()? &srcPosarr[0]:0, &x);

  // now sort the sources and remove duplicates unless instructed not to
  PetscTruth skip_initial_sort;
  PetscOptionsHasName(0,"-skip_initial_sort",&skip_initial_sort);
  if (!skip_initial_sort)
  {
    if(!mpirank)
      cout<<"Sorting the sources in morton order and removing duplicates"<<endl;

    // at first, shift&scale the particles into [0,1] 
    double min_coord, max_coord;
    PetscInt dummy;
    VecMax(x, &dummy, &max_coord);
    VecMin(x, &dummy, &min_coord);

    double scale_factor = 1/(max_coord-min_coord+0.02);
    double shift = -min_coord + 0.01;
    VecShift(x, shift );
    VecScale(x, scale_factor);

    VecDestroy(x); // the array srcPosarr stays and is hopefully unchanged by VecDestroy

    const unsigned maxDepth = 30;
    std::vector<ot::TreeNode> tmpNodes(lclnumsrc);
    for (int i=0; i<lclnumsrc; i++)
    {
      unsigned X = unsigned( ldexp(srcPosarr[3*i],maxDepth) );
      unsigned Y = unsigned( ldexp(srcPosarr[3*i+1],maxDepth) );
      unsigned Z = unsigned( ldexp(srcPosarr[3*i+2],maxDepth) );
      tmpNodes[i]=ot::TreeNode(X,Y,Z,maxDepth,3,maxDepth);
    }
    par::removeDuplicates<ot::TreeNode>(tmpNodes,false,comm);	
    lclnumsrc = tmpNodes.size();
    srcPosarr.resize(lclnumsrc*dim);
    for (int i=0; i<lclnumsrc; i++)
    {
      srcPosarr[0+3*i]=ldexp(tmpNodes[i].getX()+0.5,-maxDepth);
      srcPosarr[1+3*i]=ldexp(tmpNodes[i].getY()+0.5,-maxDepth);
      srcPosarr[2+3*i]=ldexp(tmpNodes[i].getZ()+0.5,-maxDepth);
    }
    VecCreateMPIWithArray(comm, lclnumsrc*dim, PETSC_DETERMINE, srcPosarr.size()? &srcPosarr[0]:0, &x);
    // now shift and scale "x" back
    VecScale(x, 1/scale_factor);
    VecShift(x, -shift );
  }


  if(!mpirank)
    cout<<"Total number of sources on all processors: "<<procGlbNum(x) << endl;

  // vector of particles IDs;
  Vec ids;
  VecCreateMPI(comm,lclnumsrc,PETSC_DETERMINE,&ids);
  PetscInt first_idx, after_last_idx;
  VecGetOwnershipRange(ids,&first_idx,&after_last_idx);
  double* arr;
  VecGetArray(ids, &arr) ;
  for(PetscInt i=first_idx; i<after_last_idx; i++)
    arr[i-first_idx]=double(i);
  VecRestoreArray(ids, &arr);

  // vectors that we need for RK45  (they are created and destroyed in the course of computations)
  // layout is only changed during computation of k1
  // Vec xx;  // temporary vector of particle coordinates, will store things like x+0.5*h*k1
  Vec xx, k1, k2, k3, k4;            // layout is fixed during one iteration


  PetscInt num_iter;
  double h;
  double eps;
  double vol_ratio;
  PetscInt write_fraction;
  PetscTruth preserve_order;

  PetscOptionsGetReal(0,"-time_step",&h,&flg);
  pA(flg);
  PetscOptionsGetReal(0,"-particle_size",&eps,&flg);

  if (flg)
  {
    PetscOptionsHasName(0,"-volume_ratio", &flg);
    if (flg)
      SETERRQ(1,"-volume_ratio and -particle_size cannot be used at the same time");
  }
  else
  {
    PetscOptionsGetReal(0,"-volume_ratio",&vol_ratio,&flg);
    if(!flg)
      SETERRQ(1,"Either -volume_ratio or -particle_size must be provided");
    eps = exp( log(vol_ratio/procGlbNum(x))/3 );
  }

  PetscOptionsGetInt(0, "-num_iter", &num_iter, &flg) ;
  pA(flg==PETSC_TRUE);

  PetscOptionsGetInt(0, "-write_fraction", &write_fraction, &flg) ;
  pA(flg==PETSC_TRUE);

  PetscTruth dump_ids;
  PetscOptionsHasName(0,"-dump_ids", &dump_ids);

  PetscOptionsHasName(0,"-preserve_order", &preserve_order);

  if (!mpirank)
    cout<<"Using particle size: "<<eps<<" , time step: "<<h<<" , number of iterations: "<<num_iter<<" , write fraction: "<<write_fraction<<endl;


  ofstream CL_file; // in this file we'll save center of blob ana number of particles leaked at each time step

  // dump vector of initial coordinates to a file
//   {
//     ostringstream oss;
//     oss<<"coord0.txt";
//     PetscViewer viewer;
//     PetscViewerASCIIOpen (comm,oss.str().c_str(),&viewer);
//     PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_SYMMODU);
//     VecView(x,viewer);
//     PetscViewerDestroy(viewer);
// 
//     // dump center of mass and number of leaked particles (for single ball distribution only)
//     if (0==strcmp(distribution,"ball"))
//     {
//       // first open file for centers and leaks
//       if (!mpirank)
// 	CL_file.open("center_and_leaks.txt",ios::trunc);
// 
//       DumpCenterAndLeaks(x,CL_file);
//     }
//   }

  Vec dummy_vec=0;
  // main loop
  for (int i=1; i<=num_iter; i++)
  {
    //  ===== k1=f(x); change the layout of x ====
    eval_rhs(x, ids, k1, knl, eps, !preserve_order);   // VecCreate(k1) is called inside, so destroy k1 outside


    VecDuplicate(x,&xx);

    // xx = x+ 0.5*h*k1
    // l = f(xx)
    // k2 = scatter (l);
    VecCopy(x, xx);
    VecAXPY(xx, h/2,k1);
    eval_rhs(xx,dummy_vec, k2, knl, eps, false);   // k2 will have the layout of x

    // xx = x+ 0.5*h*k2
    // l = f(xx)
    // k3 = scatter (l);
    VecCopy(x, xx);
    VecAXPY(xx,h/2,k2);
    eval_rhs(xx,dummy_vec, k3, knl, eps, false);   // k3 will have the layout of x


    // xx = x+ h*k3
    // l = f(xx)
    // k4 = scatter (l);
    VecCopy(x, xx);
    VecAXPY(xx,h,k3);
    eval_rhs(xx,dummy_vec, k4, knl, eps, false);   // k4 will have the layout of x

    // x = x + h/6*k1;
    // x = x + h/3*k2;
    // x = x + h/3*k3;
    // x = x + h/6*k4;
    VecAXPY(x,h/6,k1);
    VecAXPY(x,h/3,k2);
    VecAXPY(x,h/3,k3);
    VecAXPY(x,h/6,k4);

    VecDestroy(k1);
    VecDestroy(k2);
    VecDestroy(k3);
    VecDestroy(k4);
    VecDestroy(xx);

    if(!(i%write_fraction))
    {
      ostringstream oss;
      oss<<"coord"<<i<<".txt";
      PetscViewer viewer;
      PetscViewerASCIIOpen (comm,oss.str().c_str(),&viewer);
      PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_SYMMODU);
      VecView(x,viewer);
      PetscViewerDestroy(viewer);

      // dump center of mass and number of leaked particles (for single ball distribution only)
      if (0==strcmp(distribution,"ball"))
	DumpCenterAndLeaks(x,CL_file);

      if(dump_ids)
      {
	ostringstream oss;
	oss<<"ids"<<i<<".txt";
	PetscViewer viewer;
	PetscViewerASCIIOpen (comm,oss.str().c_str(),&viewer);
	PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_SYMMODU);
	VecView(ids,viewer);
	PetscViewerDestroy(viewer);
      }
    }
  }

//   if (0==strcmp(distribution,"ball") && !mpirank)
//     CL_file.close();

  pC( VecDestroy(x) );
  pC( VecDestroy(ids) );

  PetscFinalize();
  return 0;
}
