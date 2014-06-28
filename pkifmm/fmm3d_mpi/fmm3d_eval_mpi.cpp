/* Parallel Kernel Independent Fast Multipole Method
   Copyright (C) 2004 Lexing Ying,  New York University
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
#include "fmm3d_mpi.hpp"
#include "common/vecmatop.hpp"
#include "manage_petsc_events.hpp"
#include "p3d/point3d.h"
#include "p3d/upComp.h"
#include "p3d/dnComp.h"
#include "gpu_setup.h"
#include "gpu_vlist.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

using std::cerr;
using std::cout;
using std::endl;

// ----------------------------------------------------------------------

int FMM3d_MPI::evaluate_downward()
{
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();

  vector<int> ordVec;
  pC( _let->dwnOrderCollect(ordVec) ); //BOTTOM UP collection of nodes
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);

#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif

  PetscTruth gpu_l2t;
  PetscOptionsHasName(0,"-gpu_l2t",&gpu_l2t);

  // non-leaf
  for(size_t i=0; i<ordVec.size(); i++)                                                                 
  { 
    int gNodeIdx = ordVec[i];
    if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {                                                    
      if(_let->depth(gNodeIdx)>=3)  {
	int pargNodeIdx = _let->parent(gNodeIdx);
	Index3 chdidx( _let->path2Node(gNodeIdx)-2 * _let->path2Node(pargNodeIdx) );                
	//L2L
	DblNumVec evaTrgDwnChkVal_gNodeIdx(evaTrgDwnChkVal(gNodeIdx));
	pC( _matmgnt->DwnEqu2DwnChk_dgemv(_let->depth(pargNodeIdx)+_rootLevel, chdidx, evaTrgDwnEquDen(pargNodeIdx), evaTrgDwnChkVal_gNodeIdx) );
      }                                                                                                 
      if(_let->depth(gNodeIdx)>=2) {                                                                    
	//L2L                                                                                       
	DblNumVec evaTrgDwnEquDen_gNodeIdx(evaTrgDwnEquDen(gNodeIdx));                              
	pC( _matmgnt->DwnChk2DwnEqu_dgemv(_let->depth(gNodeIdx)+_rootLevel, evaTrgDwnChkVal(gNodeIdx), evaTrgDwnEquDen_gNodeIdx) );
      }
    } 
  } 

  // leaf                                                                                                 
#pragma omp parallel for schedule(guided)                                                               
  for(int i=0; i<ordVec.size(); i++) {                                                                  
    int gNodeIdx = ordVec[i];
    if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) {                                                   
      if(_let->terminal(gNodeIdx)) {
	if(!gpu_l2t) // if node is a leaf and we are not going to use GPU for l2t translations            
	{
	  DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));                                        
	  iC( DwnEqu2TrgChk_dgemv(_let->center(gNodeIdx), _let->radius(gNodeIdx), evaTrgExaPos(gNodeIdx), evaTrgDwnEquDen(gNodeIdx), evaTrgExaVal_gNodeIdx, fuse_dense) );
	}     
      }
    } 
  }  

  if (gpu_l2t) {
#ifdef COMPILE_GPU
    const DblNumMat & sample_pos = _matmgnt->samPos(DE);
    vector<float> sample_pos_float(sample_pos.n()*sample_pos.m());

    int NumTrgBoxes=0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_EVTRNODE)
	NumTrgBoxes++;

    dnComp_t *DnC;
    if ( (DnC = (dnComp_t*) calloc (1, sizeof (dnComp_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for downward computation structure\n");
      return 1;
    }
    /* Copy data into the downward computation structure defined by 'DnC' */
    DnC->tag = DE;
    DnC->numTrg = procLclNum(_evaTrgExaPos);
    DnC->dim = 3;
    DnC->numTrgBox = NumTrgBoxes;

    DnC->trg_ = (float *) malloc(sizeof(float) * DnC->numTrg * DnC->dim);
    DnC->trgVal = (float*) malloc (sizeof(float*) * DnC->numTrg *trgDOF );
    DnC->trgBoxSize = (int *) calloc (DnC->numTrgBox, sizeof(int));
    DnC->srcCtr = (float *) calloc (DnC->numTrgBox * DnC->dim, sizeof(float));
    DnC->srcRad = (float *) calloc (DnC->numTrgBox, sizeof(float));

    for (size_t i=0; i<sample_pos_float.size(); i++)
      sample_pos_float[i]=*(sample_pos._data+i);

    DnC->srcDim=sample_pos.n();
    DnC->samPosF=&sample_pos_float[0];
    int fmm_srcDOF = _knl_mm.srcDOF();
    DnC->srcDen=(float*)calloc(DnC->numTrgBox*DnC->srcDim*fmm_srcDOF,sizeof(float));

    DnC->kernel_type = _knl_mm.kernelType();
    DnC->kernel_coef[0]=_knl_mm.coefs()[0];


    int trgIndex = 0;
    int trgBoxIdx = 0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_EVTRNODE)
      {
	/* Center of the box */
	for (int j = 0; j < DnC->dim; j++)
	  DnC->srcCtr[j+trgBoxIdx*DnC->dim] = _let->center(gNodeIdx)(j);
	/* Radius of the box */
	DnC->srcRad[trgBoxIdx] = _let->radius(gNodeIdx);

	DblNumMat targets (evaTrgExaPos(gNodeIdx));
	DnC->trgBoxSize[trgBoxIdx] = targets.n();
	for(int s = 0; s < DnC->trgBoxSize[trgBoxIdx]; s++) {
	  for(int d = 0; d < DnC->dim; d++) {
	    DnC->trg_[(s*(DnC->dim))+d+trgIndex] = targets(d,s);
	  }
	}

	for(int t = 0; t < _matmgnt->plnDatSze(DE); t++) {
	  DnC->srcDen[trgBoxIdx*_matmgnt->plnDatSze(DE)+t]=evaTrgDwnEquDen(gNodeIdx)(t);
	}

	trgIndex += (DnC->trgBoxSize[trgBoxIdx] * DnC->dim);
	trgBoxIdx++;
      }

    gpu_down(DnC);

    int trgValIdx=0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_EVTRNODE)
      {
	DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
	for (int j = 0; j < evaTrgExaVal_gNodeIdx.m(); j++)
	  evaTrgExaVal_gNodeIdx(j) += DnC->trgVal[trgValIdx+j];
	trgValIdx += evaTrgExaVal_gNodeIdx.m();
      }

    free (DnC->trg_);
    free (DnC->trgBoxSize);
    free (DnC->srcCtr);
    free (DnC->srcRad);
    //	  for (int i = 0; i < ordVec.size(); i++)
    //		free (DnC->trgVal[ordVec[i]]);
    free (DnC->trgVal);
    free(DnC->srcDen);
    free (DnC);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }

#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  return 0;
}

int FMM3d_MPI::evaluate_xlist()
{
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();

  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);

#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  PetscTruth gpu_xlist;
  PetscOptionsHasName(0,"-gpu_xlist",&gpu_xlist);
  if (gpu_xlist)
  {
#ifdef COMPILE_GPU
    // Interface U-list contribution calculation for GPU
    point3d_t *P;
    if ( (P = (point3d_t*) malloc (sizeof (point3d_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for u-list structure\n");
      return 1;
    }
    // Copy data into the u-list structure defined by 'P'

    // loop through the tree, make necessary mappings
    vector<int> XsrcBoxMap(_let->nodeVec().size(),-1);
    vector<int> XtrgBoxMap(_let->nodeVec().size(),-1);
    int numXtrgBoxes=0;
    int numXsources=0;
    int numXsrcBoxes=0;

    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE &&  _let->node(gNodeIdx).Xnodes().size()>0) 
      { // this box contains targets and has some boxes on it's X-list
	XtrgBoxMap[gNodeIdx]=numXtrgBoxes++;
	for (size_t i=0; i< _let->node(gNodeIdx).Xnodes().size(); i++)
	{
	  int x_node_idx = _let->node(gNodeIdx).Xnodes()[i];
	  if (XsrcBoxMap[x_node_idx]==-1)
	  {
	    XsrcBoxMap[x_node_idx]=numXsrcBoxes++;
	    numXsources += _let->node(x_node_idx).usrSrcExaNum();
	  }
	}
      }
    }

    P->numSrc =numXsources;
    P->numTrg = _matmgnt->samPos(DC).n()*numXtrgBoxes;
    P->dim = 3;
    P->kernel_type = _knl.kernelType();
    P->kernel_coef[0]=_knl.coefs()[0];

#ifdef USE_DOUBLE   
    P->src_ = (double *) malloc(sizeof(double) * P->numSrc * (P->dim+srcDOF));
    P->trg_ = (double *) malloc(sizeof(double) * P->numTrg * P->dim);
    P->trgVal = (double *) calloc(P->numTrg*trgDOF, sizeof(double));
#else
    P->src_ = (float *) malloc(sizeof(float) * P->numSrc * (P->dim+srcDOF));
    P->trg_ = (float *) malloc(sizeof(float) * P->numTrg * P->dim);
    P->trgVal = (float *) calloc(P->numTrg*trgDOF, sizeof(float));
#endif

    P->uList = (int **) malloc (sizeof(int*) * numXtrgBoxes);
    P->uListLen = (int *) calloc (numXtrgBoxes, sizeof(int));

    P->srcBoxSize = (int *) calloc (numXsrcBoxes, sizeof(int));
    // we'll also need an array of displacements for source densities and coordinates
    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int XsrcNodeIdx = XsrcBoxMap[gNodeIdx]; 
      if (XsrcNodeIdx!=-1)
	P->srcBoxSize[XsrcNodeIdx] = _let->node(gNodeIdx).usrSrcExaNum();
    }
    vector<int> XsrcDispls(numXsrcBoxes);
    if (!XsrcDispls.empty())
      XsrcDispls[0]=0;
    for(int i=1; i<numXsrcBoxes; i++)
      XsrcDispls[i]=XsrcDispls[i-1]+P->srcBoxSize[i-1];

    P->trgBoxSize = (int *) calloc (numXtrgBoxes, sizeof(int));
    P->numTrgBox = numXtrgBoxes;
    P->numSrcBox = numXsrcBoxes;

    int j;
    int trgIndex = 0;
    int srcIndex = 0;
    int d = 0;

    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int XtrgNodeIdx = XtrgBoxMap[gNodeIdx]; 
      if (XtrgNodeIdx!=-1)
      {
	Let3d_MPI::Node& curNode = _let->node(gNodeIdx);

	P->uList[XtrgNodeIdx] = (int*) malloc (sizeof(int) * curNode.Xnodes().size());
	P->uListLen[XtrgNodeIdx] = curNode.Xnodes().size();
	j = 0;
	for(vector<int>::iterator vi=curNode.Xnodes().begin(); vi!=curNode.Xnodes().end(); vi++) {
	  P->uList[XtrgNodeIdx][j] = XsrcBoxMap[*vi];
	  assert(XsrcBoxMap[*vi]>=0);
	  j++;
	}
	// for X-list targets are on downward check surface
	DblNumMat srcPos; 
	_matmgnt->locPos(DC, _let->center(gNodeIdx), _let->radius(gNodeIdx), srcPos);
	P->trgBoxSize[XtrgNodeIdx] =  srcPos.n();
	for(int t = 0; t < P->trgBoxSize[XtrgNodeIdx]; t++) {
	  for(d = 0; d < P->dim; d++)
	  {
	    P->trg_[(t*P->dim)+d+trgIndex] =  srcPos(d,t);
	  }
	}
	trgIndex += (P->trgBoxSize[XtrgNodeIdx] * P->dim);
      }

      int XsrcNodeIdx = XsrcBoxMap[gNodeIdx]; 
      if (XsrcNodeIdx!=-1)
      {
	srcIndex = XsrcDispls[XsrcNodeIdx]*(3+srcDOF);   // 3 is three coordinates
	for(int s = 0; s < P->srcBoxSize[XsrcNodeIdx]; s++) {
	  for(d = 0; d < P->dim; d++)
	    P->src_[(s*(P->dim+srcDOF))+d+srcIndex] = usrSrcExaPos(gNodeIdx)(d,s);
	  for(d=0; d<srcDOF; d++)
	    P->src_[(s*(P->dim+srcDOF))+3+d+srcIndex] = usrSrcExaDen(gNodeIdx)(s*srcDOF+d); // "3" is 3 coordinates
	}
      }
    }

    //  Calculate dense interations
    dense_inter_gpu(P);

    trgIndex = 0;
    // Copy target potentials back into the original structure
    // * for use by rest of the algorithm
    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int XtrgNodeIdx = XtrgBoxMap[gNodeIdx]; 
      if (XtrgNodeIdx!=-1)
      {
	for(int t = 0; t < P->trgBoxSize[XtrgNodeIdx]; t++)
	  for(d=0; d<trgDOF; d++)
	    evaTrgDwnChkVal(gNodeIdx)(t*trgDOF+d) += P->trgVal[t*trgDOF+d+trgIndex];
	trgIndex += P->trgBoxSize[XtrgNodeIdx]*trgDOF;
      }
    }

    // Free memory allocated for the interface
    free (P->src_);
    free (P->trg_);
    free (P->trgVal);
    free (P->uListLen);
    free (P->srcBoxSize);
    free (P->trgBoxSize);
    for(int i=0; i<P->numTrgBox; i++)
      free (P->uList[i]);
    free (P->uList);
    free (P);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }
  else
#pragma omp parallel for schedule(guided)
    for(size_t i=0; i<ordVec.size(); i++) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
	DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
	DblNumVec evaTrgDwnChkVal_gNodeIdx(evaTrgDwnChkVal(gNodeIdx));
	for(vector<int>::iterator vi=_let->node(gNodeIdx).Xnodes().begin(); vi!=_let->node(gNodeIdx).Xnodes().end(); vi++) {
	  if(_let->terminal(gNodeIdx) && _let->node(gNodeIdx).evaTrgExaNum()*trgDOF<_matmgnt->plnDatSze(DC)) { //use Exa instead
	    iC( SrcEqu2TrgChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), evaTrgExaPos(gNodeIdx), usrSrcExaDen(*vi), evaTrgExaVal_gNodeIdx, fuse_dense) );
	  } else {
	    //S2L
	    iC( SrcEqu2DwnChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), _let->center(gNodeIdx), _let->radius(gNodeIdx), usrSrcExaDen(*vi), evaTrgDwnChkVal_gNodeIdx, fuse_dense) );
	  }
	}
      }
    }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  return 0;
}

int FMM3d_MPI::evaluate_wlist()
{
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();

  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);

#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  PetscTruth gpu_wlist;
  PetscOptionsHasName(0,"-gpu_wlist",&gpu_wlist);
  if (gpu_wlist)
  {
#ifdef COMPILE_GPU
    // Interface U-list contribution calculation for GPU
    point3d_t *P;
    if ( (P = (point3d_t*) malloc (sizeof (point3d_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for u-list structure\n");
      return 1;
    }
    // Copy data into the u-list structure defined by 'P'

    // loop through the tree, make necessary mappings
    vector<int> WsrcBoxMap(_let->nodeVec().size(),-1);
    vector<int> WtrgBoxMap(_let->nodeVec().size(),-1);
    int numWtrgBoxes=0;
    int numWtargets=0;
    int numWsrcBoxes=0;


    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE &&  _let->node(gNodeIdx).Wnodes().size()>0) 
      { // this box contains targets and has some boxes on it's W-list
	WtrgBoxMap[gNodeIdx]=numWtrgBoxes++;
	numWtargets += _let->node(gNodeIdx).evaTrgExaNum();

	for (size_t i=0; i< _let->node(gNodeIdx).Wnodes().size(); i++)
	{
	  int w_node_idx = _let->node(gNodeIdx).Wnodes()[i];
	  if (WsrcBoxMap[w_node_idx]==-1)
	    WsrcBoxMap[w_node_idx]=numWsrcBoxes++;
	}
      }
    }

    P->numSrc =_matmgnt->samPos(UE).n()*numWsrcBoxes;
    P->numTrg = numWtargets;
    P->dim = 3;
    P->kernel_type = _knl_mm.kernelType();
    P->kernel_coef[0]=_knl_mm.coefs()[0];

    int fmm_srcDOF = _knl_mm.srcDOF();
#ifdef USE_DOUBLE
    P->src_ = (double *) malloc(sizeof(double) * P->numSrc * (P->dim+fmm_srcDOF));
    P->trg_ = (double *) malloc(sizeof(double) * P->numTrg * P->dim);
    P->trgVal = (double *) calloc(P->numTrg*trgDOF, sizeof(double));
#else
    P->src_ = (float *) malloc(sizeof(float) * P->numSrc * (P->dim+fmm_srcDOF));
    P->trg_ = (float *) malloc(sizeof(float) * P->numTrg * P->dim);
    P->trgVal = (float *) calloc(P->numTrg*trgDOF, sizeof(float));
#endif

    P->uList = (int **) malloc (sizeof(int*) * numWtrgBoxes);
    P->uListLen = (int *) calloc (numWtrgBoxes, sizeof(int));
    P->srcBoxSize = (int *) calloc (numWsrcBoxes, sizeof(int));
    P->trgBoxSize = (int *) calloc (numWtrgBoxes, sizeof(int));

    P->numTrgBox = numWtrgBoxes;
    P->numSrcBox = numWsrcBoxes;
    int j;
    int trgIndex = 0;
    int srcIndex = 0;
    int d = 0;

    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int WtrgNodeIdx = WtrgBoxMap[gNodeIdx]; 
      if (WtrgNodeIdx!=-1)
      {
	Let3d_MPI::Node& curNode = _let->node(gNodeIdx);

	P->uList[WtrgNodeIdx] = (int*) malloc (sizeof(int) * curNode.Wnodes().size());
	P->uListLen[WtrgNodeIdx] = curNode.Wnodes().size();
	j = 0;
	for(vector<int>::iterator vi=curNode.Wnodes().begin(); vi!=curNode.Wnodes().end(); vi++) {
	  P->uList[WtrgNodeIdx][j] = WsrcBoxMap[*vi];
	  // cout<<WsrcBoxMap[*vi]<<endl;
	  assert(WsrcBoxMap[*vi]>=0);
	  j++;
	}
	DblNumMat evaTrgExaPosgNodeIdx(evaTrgExaPos(gNodeIdx));
	P->trgBoxSize[WtrgNodeIdx] =  evaTrgExaPosgNodeIdx.n();  // curNode.evaTrgExaNum();
	assert (evaTrgExaPosgNodeIdx.n() == curNode.evaTrgExaNum());
	for(int t = 0; t < P->trgBoxSize[WtrgNodeIdx]; t++) {
	  for(d = 0; d < P->dim; d++)
	  {
	    P->trg_[(t*P->dim)+d+trgIndex] =  evaTrgExaPosgNodeIdx(d,t);
	  }
	}
	trgIndex += (P->trgBoxSize[WtrgNodeIdx] * P->dim);
      }

      int WsrcNodeIdx = WsrcBoxMap[gNodeIdx]; 
      if (WsrcNodeIdx!=-1)
      {
	// for W-list all source boxes have the same number of points
	srcIndex = (3+fmm_srcDOF)*_matmgnt->samPos(UE).n()*WsrcNodeIdx; // "3" is three coordinates
	P->srcBoxSize[WsrcNodeIdx] = _matmgnt->samPos(UE).n();
	DblNumMat srcPos; 
	_matmgnt->locPos(UE, _let->center(gNodeIdx), _let->radius(gNodeIdx), srcPos);

	for(int s = 0; s < P->srcBoxSize[WsrcNodeIdx]; s++) {
	  for(d = 0; d < P->dim; d++)
	    P->src_[(s*(P->dim+fmm_srcDOF))+d+srcIndex] =   srcPos(d,s);
	  for(d = 0; d < fmm_srcDOF; d++)
	    P->src_[(s*(P->dim+fmm_srcDOF))+P->dim+d+srcIndex] = usrSrcUpwEquDen(gNodeIdx)(s*fmm_srcDOF+d);
	}
      }
    }

    //  Calculate dense interations
    dense_inter_gpu(P);

    trgIndex = 0;
    // Copy target potentials back into the original structure
    // * for use by rest of the algorithm
    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int WtrgNodeIdx = WtrgBoxMap[gNodeIdx]; 
      if (WtrgNodeIdx!=-1)
      {
	for(int t = 0; t < P->trgBoxSize[WtrgNodeIdx]; t++)
	  for(d = 0; d < trgDOF; d++)
	    evaTrgExaVal(gNodeIdx)(t*trgDOF+d) += P->trgVal[t*trgDOF+d+trgIndex];
	trgIndex += P->trgBoxSize[WtrgNodeIdx]*trgDOF;
      }
    }

    // Free memory allocated for the interface
    free (P->src_);
    free (P->trg_);
    free (P->trgVal);
    free (P->uListLen);
    free (P->srcBoxSize);
    free (P->trgBoxSize);
    for(int i=0; i<P->numTrgBox; i++)
      free (P->uList[i]);
    free (P->uList);
    free (P);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }
  else
#pragma omp parallel for schedule(guided)
    for(size_t i=0; i<ordVec.size(); i++) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
	if( _let->terminal(gNodeIdx)==true ) {
	  DblNumVec evaTrgExaVal_gNodeIdx(this->evaTrgExaVal(gNodeIdx));
	  for(vector<int>::iterator vi=_let->node(gNodeIdx).Wnodes().begin(); vi!=_let->node(gNodeIdx).Wnodes().end(); vi++) {

	    // terminal nodes in LET might be parent nodes in global tree;
	    // thus, in some cases we instead need to check glbSrcExaNum or glbSrcExaBeg;
	    // both are guaranteed to be -1 for parent nodes in global tree
	    // and both guaranteed to be >=0 for leaves in global tree
	    if(_let->node(*vi).glbSrcExaBeg()>=0 && _let->node(*vi).usrSrcExaNum()*srcDOF<_matmgnt->plnDatSze(UE)) { //use Exa instead
	      //S2T
	      iC( SrcEqu2TrgChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), evaTrgExaPos(gNodeIdx), usrSrcExaDen(*vi), evaTrgExaVal_gNodeIdx, fuse_dense) );
	    } else {
	      //M2T
	      int vni = *vi;
	      iC( UpwEqu2TrgChk_dgemv(_let->center(vni), _let->radius(vni), evaTrgExaPos(gNodeIdx), usrSrcUpwEquDen(*vi), evaTrgExaVal_gNodeIdx, fuse_dense) );
	    }
	  }
	}
      }
    }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
}


int FMM3d_MPI::evaluate_vlist()
{
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();

  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);
  PetscTruth gpu_vlist;
  PetscOptionsHasName(0,"-gpu_vlist",&gpu_vlist);
  PetscTruth gpu_vlist_alternative;
  PetscOptionsHasName(0,"-gpu_vlist_alternative",&gpu_vlist_alternative);
  if (gpu_vlist)
  {
    //#ifdef COMPILE_GPU
    int effDatSze = _matmgnt->effDatSze(UE);

    // loop through the tree, make necessary mappings
    vector<int> VsrcBoxMap(_let->nodeVec().size(),-1);
    vector<int> VtrgBoxMap(_let->nodeVec().size(),-1);
    int numVtrgBoxes=0;
    int numVsrcBoxes=0;
    int totalVlistSize=0;


    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE &&  _let->node(gNodeIdx).Vnodes().size()>0) 
      { // this box contains targets and has some boxes on it's V-list
	VtrgBoxMap[gNodeIdx]=numVtrgBoxes++;
	totalVlistSize += _let->node(gNodeIdx).Vnodes().size();
      }

      if( _let->node(gNodeIdx).tag() & LET_USERNODE &&  node(gNodeIdx).vLstOthNum()>0) 
      { // this box contains sources and is on V-list of some other box
	VsrcBoxMap[gNodeIdx]=numVsrcBoxes++;
      }

    }

    // alloc.  fft-s of densities
    vector<float> fDen (effDatSze*numVsrcBoxes);
    // alloc.  fft-s of potentials:
    vector<float> fPot (effDatSze*numVtrgBoxes);

    // calculate number of translations and establish mapping for them
    int transl_map[7][7][7];
    int transl_counter=0;
    for (int i1=-3; i1<=3; i1++)
      for (int i2=-3; i2<=3; i2++)
	for (int i3=-3; i3<=3; i3++)
	  if (abs(i1)>1 || abs(i2)>1 || abs(i3)>1 )
	    transl_map[i1+3][i2+3][i3+3]=transl_counter++;
	  else
	    transl_map[i1+3][i2+3][i3+3]=-1; 

    // alloc fft-s of translations:
    vector<float> fTr (effDatSze*transl_counter);
    int numtransl = transl_counter;

    transl_counter=0;
    for (int i1=-3; i1<=3; i1++)
      for (int i2=-3; i2<=3; i2++)
	for (int i3=-3; i3<=3; i3++)
	  if (abs(i1)>1 || abs(i2)>1 || abs(i3)>1 )
	  {
	    // compute and copy translation operator
	    DblNumMat _UpwEqu2DwnChkii;
	    int effnum = effDatSze;
	    Index3 idx;
	    idx(0)=i1;
	    idx(1)=i2;
	    idx(2)=i3;
	    pA( idx.linfty()>1 );
	    double R = 1;
	    srcDOF=1;
	    trgDOF=1;
	    DblNumMat denPos(dim(),1);	 for(int i=0; i<dim(); i++)		denPos(i,0) = double(idx(i))*2.0*R; //shift
	    DblNumMat chkPos(dim(),_matmgnt->regPos().n());	 clear(chkPos);	 pC( daxpy(R, _matmgnt->regPos(), chkPos) );
	    DblNumMat tt(_matmgnt->regPos().n()*trgDOF, srcDOF);
	    pC( _knl.buildKnlIntCtx(denPos, denPos, chkPos, tt) );
	    // move data to tmp
	    DblNumMat tmp(trgDOF,_matmgnt->regPos().n()*srcDOF);
	    for(int k=0; k<_matmgnt->regPos().n();k++) {
	      for(int i=0; i<trgDOF; i++)
		for(int j=0; j<srcDOF; j++) {
		  tmp(i,j+k*srcDOF) = tt(i+k*trgDOF,j);
		}
	    }
	    _UpwEqu2DwnChkii.resize(trgDOF*srcDOF, effnum); 
	    //forward FFT from tmp to _UpwEqu2DwnChkii;

	    int nnn[3];
	    nnn[0] = 2*_np;
	    nnn[1] = 2*_np;
	    nnn[2] = 2*_np;

	    fftw_plan forplan = fftw_plan_many_dft_r2c(3,nnn,srcDOF*trgDOF, tmp.data(),NULL, srcDOF*trgDOF,1, (fftw_complex*)(_UpwEqu2DwnChkii.data()),NULL, srcDOF*trgDOF,1, FFTW_ESTIMATE);
	    fftw_execute(forplan);
	    fftw_destroy_plan(forplan);	 

	    for (int i=0; i<effDatSze; i++)
	      fTr[transl_counter*effDatSze + i]=_UpwEqu2DwnChkii._data[i];

	    transl_counter++;
	  }

    // extra integer in the end is necessary to calc. the length of last V-list 
    vector<int> vlist_starts(numVtrgBoxes+1,-1);
    vector<int> vlists(totalVlistSize,-1);
    // spv stores translation number for each box on each v-list
    vector<int> spv(totalVlistSize,-1);

    int vlist_ptr=0;
    int trgbox=0;
    int srcbox=0;
    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE &&  _let->node(gNodeIdx).Vnodes().size()>0) 
      { // this box contains targets and has some boxes on it's V-list
	Point3 gNodeIdxctr(_let->center(gNodeIdx));
	double D = 2.0 * _let->radius(gNodeIdx);
	vlist_starts[trgbox++]=vlist_ptr;
	for(size_t i=0; i<_let->node(gNodeIdx).Vnodes().size(); i++)
	{
	  Point3 victr(  _let->center( _let->node(gNodeIdx).Vnodes()[i] )  );
	  Index3 idx;
	  for(int d=0; d<dim(); d++)
	    idx(d) = int(floor( (victr[d]-gNodeIdxctr[d])/D+0.5));

	  spv[vlist_ptr] = transl_map[idx(0)+3][idx(1)+3][idx(2)+3];
	  vlists[vlist_ptr++]=VsrcBoxMap[ (_let->node(gNodeIdx).Vnodes())[i] ];
	}
      }

      // load densities
      if( _let->node(gNodeIdx).tag() & LET_USERNODE &&  node(gNodeIdx).vLstOthNum()>0) 
      { // this box contains sources and is on V-list of some other box
	// do fft
	Node& srcnode = node(gNodeIdx);
	srcnode.effDen().resize( _matmgnt->effDatSze(UE) );
	setvalue(srcnode.effDen(), 0.0);
	pC( _matmgnt->plnDen2EffDen(_let->depth(gNodeIdx)+_rootLevel, usrSrcUpwEquDen(gNodeIdx),  srcnode.effDen()) );
	double * data = srcnode.effDen()._data;
	for (int i=0; i<effDatSze; i++)
	  fDen[srcbox*effDatSze+i]=data[i];

	srcnode.effDen().resize(0);
	srcbox++;
      }
    }
    vlist_starts.back()=vlist_ptr;

    // now do something on gpu
#ifdef EMULATE_GPU_VLIST
    // (emulate)
    for (int i=0; i<numVtrgBoxes; i++)
    {
      for (int j=0; j<effDatSze; j++)
	fPot[i*effDatSze+j]=0;
      int v_size = vlist_starts[i+1]-vlist_starts[i];
      int * cur_vlist = &vlists[vlist_starts[i]];
      int * cur_trans = &spv[vlist_starts[i]];
      float * dst = &fPot[effDatSze*i];

      for (int Vbox=0; Vbox<v_size; Vbox++)
      {
	int VboxGlobNum = cur_vlist[Vbox];
	float * src = &fDen[effDatSze*VboxGlobNum]; 
	float * trn = &fTr[effDatSze*cur_trans[Vbox]];

	assert(effDatSze%2==0);
	int loop_len=effDatSze/2;
	for (int k=0; k<loop_len; k++)
	{
	  float a=src[2*k];
	  float b=src[2*k+1];
	  float c=trn[2*k];
	  float d=trn[2*k+1];

	  dst[2*k] += (a*c-b*d);
	  dst[2*k+1] +=(a*d+b*c); 
	}
      }
    }
#else
    //  if (!mpiRank())
    //    std::cout<<"Using GPU for V-list computations\n";
    //    if (numVsrcBoxes && numVtrgBoxes)
    //     cudavlistfunc(effDatSze/2,numtransl,numVtrgBoxes,numVsrcBoxes,&vlist_starts[0],&vlists[0],&spv[0],&fDen[0],&fTr[0],&fPot[0]);
#endif

    // write the results back:
    trgbox=0;
    float nrmfc = 1.0/_matmgnt->regPos().n();
    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE &&  _let->node(gNodeIdx).Vnodes().size()>0) 
      { // this box contains targets and has some boxes on it's V-list
	DblNumVec evaTrgDwnChkVal(this->evaTrgDwnChkVal(gNodeIdx));
	Node& trgnode = node(gNodeIdx);
	trgnode.effVal().resize( _matmgnt->effDatSze(DC) );

	double * data = trgnode.effVal()._data;
	for (int i=0; i<effDatSze; i++)
	  data[i] = fPot[effDatSze*trgbox + i]*nrmfc;
	pC( _matmgnt->effVal2PlnVal(_let->depth(gNodeIdx)+_rootLevel, trgnode.effVal(), evaTrgDwnChkVal) ); //1. transform from effval to DwnChkVal
	trgnode.effVal().resize(0); //2. resize effVal to 0
	trgbox++;
      }
    }
    //#else
    //    SETERRQ(1,"GPU code not compiled");
    //#endif
  }
  else if (gpu_vlist_alternative)
  {
#ifdef COMPILE_GPU
    // Interface U-list contribution calculation for GPU
    point3d_t *P;
    if ( (P = (point3d_t*) malloc (sizeof (point3d_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for u-list structure\n");
      return 1;
    }
    // Copy data into the u-list structure defined by 'P'

    // loop through the tree, make necessary mappings
    vector<int> VsrcBoxMap(_let->nodeVec().size(),-1);
    vector<int> VtrgBoxMap(_let->nodeVec().size(),-1);
    int numVtrgBoxes=0;
    int numVsrcBoxes=0;


    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE &&  _let->node(gNodeIdx).Vnodes().size()>0) 
      { // this box contains targets and has some boxes on it's V-list
	VtrgBoxMap[gNodeIdx]=numVtrgBoxes++;

	for (size_t i=0; i< _let->node(gNodeIdx).Vnodes().size(); i++)
	{
	  int v_node_idx = _let->node(gNodeIdx).Vnodes()[i];
	  if (VsrcBoxMap[v_node_idx]==-1)
	    VsrcBoxMap[v_node_idx]=numVsrcBoxes++;
	}
      }
    }

    P->numSrc =_matmgnt->samPos(UE).n()*numVsrcBoxes;
    P->numTrg = _matmgnt->samPos(DC).n()*numVtrgBoxes;
    P->dim = 3;
    P->kernel_type = _knl_mm.kernelType();
    P->kernel_coef[0]=_knl_mm.coefs()[0];

    int fmm_srcDOF = _knl_mm.srcDOF();
#ifdef USE_DOUBLE    
    P->src_ = (double *) malloc(sizeof(double) * P->numSrc * (P->dim+fmm_srcDOF));
    P->trg_ = (double *) malloc(sizeof(double) * P->numTrg * P->dim);
    P->trgVal = (double *) calloc(P->numTrg*trgDOF, sizeof(double));
#else
    P->src_ = (float *) malloc(sizeof(float) * P->numSrc * (P->dim+fmm_srcDOF));
    P->trg_ = (float *) malloc(sizeof(float) * P->numTrg * P->dim);
    P->trgVal = (float *) calloc(P->numTrg*trgDOF, sizeof(float));
#endif


    P->uList = (int **) malloc (sizeof(int*) * numVtrgBoxes);
    P->uListLen = (int *) calloc (numVtrgBoxes, sizeof(int));
    P->srcBoxSize = (int *) calloc (numVsrcBoxes, sizeof(int));
    P->trgBoxSize = (int *) calloc (numVtrgBoxes, sizeof(int));

    P->numTrgBox = numVtrgBoxes;
    P->numSrcBox = numVsrcBoxes;
    int j;
    int trgIndex = 0;
    int srcIndex = 0;
    int d = 0;

    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int VtrgNodeIdx = VtrgBoxMap[gNodeIdx]; 
      if (VtrgNodeIdx!=-1)
      {
	Let3d_MPI::Node& curNode = _let->node(gNodeIdx);

	P->uList[VtrgNodeIdx] = (int*) malloc (sizeof(int) * curNode.Vnodes().size());
	P->uListLen[VtrgNodeIdx] = curNode.Vnodes().size();
	j = 0;
	for(vector<int>::iterator vi=curNode.Vnodes().begin(); vi!=curNode.Vnodes().end(); vi++) {
	  P->uList[VtrgNodeIdx][j] = VsrcBoxMap[*vi];
	  // cout<<WsrcBoxMap[*vi]<<endl;
	  assert(VsrcBoxMap[*vi]>=0);
	  j++;
	}
	// for V-list targets are on downward check surface
	DblNumMat srcPos; 
	_matmgnt->locPos(DC, _let->center(gNodeIdx), _let->radius(gNodeIdx), srcPos);
	P->trgBoxSize[VtrgNodeIdx] =  srcPos.n();
	for(int t = 0; t < P->trgBoxSize[VtrgNodeIdx]; t++) {
	  for(d = 0; d < P->dim; d++)
	  {
	    P->trg_[(t*P->dim)+d+trgIndex] =  srcPos(d,t);
	  }
	}
	trgIndex += (P->trgBoxSize[VtrgNodeIdx] * P->dim);
      }

      int VsrcNodeIdx = VsrcBoxMap[gNodeIdx]; 
      if (VsrcNodeIdx!=-1)
      {
	// for V-list all source boxes have the same number of points
	srcIndex = (3+fmm_srcDOF)*_matmgnt->samPos(UE).n()*VsrcNodeIdx; // "3" is three coordinates
	P->srcBoxSize[VsrcNodeIdx] = _matmgnt->samPos(UE).n();
	DblNumMat srcPos; 
	_matmgnt->locPos(UE, _let->center(gNodeIdx), _let->radius(gNodeIdx), srcPos);

	for(int s = 0; s < P->srcBoxSize[VsrcNodeIdx]; s++) {
	  for(d = 0; d < P->dim; d++)
	    P->src_[(s*(P->dim+fmm_srcDOF))+d+srcIndex] =   srcPos(d,s);
	  for(d = 0; d < fmm_srcDOF; d++)
	    P->src_[(s*(P->dim+fmm_srcDOF))+P->dim+d+srcIndex] = usrSrcUpwEquDen(gNodeIdx)(s*fmm_srcDOF+d);
	}
      }
    }

    //  Calculate dense interations
    dense_inter_gpu(P);

    trgIndex = 0;
    // Copy target potentials back into the original structure
    // * for use by rest of the algorithm
    for(size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      int VtrgNodeIdx = VtrgBoxMap[gNodeIdx]; 
      if (VtrgNodeIdx!=-1)
      {
	for(int t = 0; t < P->trgBoxSize[VtrgNodeIdx]; t++)
	  for(d=0; d<trgDOF; d++)
	    evaTrgDwnChkVal(gNodeIdx)(t*trgDOF+d) += P->trgVal[t*trgDOF+d+trgIndex];
	trgIndex += P->trgBoxSize[VtrgNodeIdx]*trgDOF;
      }
    }

    // Free memory allocated for the interface
    free (P->src_);
    free (P->trg_);
    free (P->trgVal);
    free (P->uListLen);
    free (P->srcBoxSize);
    free (P->trgBoxSize);
    for(int i=0; i<P->numTrgBox; i++)
      free (P->uList[i]);
    free (P->uList);
    free (P);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }
  else
  {
#ifdef HAVE_PAPI
    // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
    // papi_real_time, papi_proc_time, papi_mflops are just discarded
    if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
      SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif

    //V - list contribution calculation
    // Compute FFT's for the source nodes
    int fftset = 0;
    int i = 0;
    while (fftset == 0 && i < ordVec.size()) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_USERNODE ) {
	Node& srcnode = node(gNodeIdx);
	srcnode.effDen().resize( _matmgnt->effDatSze(UE) );
	setvalue(srcnode.effDen(), 0.0);//1. resize effDen             
	iC( _matmgnt->plnDen2EffDen(_let->depth(gNodeIdx)+_rootLevel,
	      usrSrcUpwEquDen(gNodeIdx),  srcnode.effDen()) );                   //2. transform from upeDen to effDen
	fftset = 1; 
      }
      i++;
    }

#pragma omp parallel for schedule(static)
    for(int j = i; j<ordVec.size(); j++) {
      int gNodeIdx = ordVec[j];
      if( _let->node(gNodeIdx).tag() & LET_USERNODE ) {
	Node& srcnode = node(gNodeIdx);
	srcnode.effDen().resize( _matmgnt->effDatSze(UE) );
	setvalue(srcnode.effDen(), 0.0);
	iC( _matmgnt->plnDen2EffDen(_let->depth(gNodeIdx)+_rootLevel,
	      usrSrcUpwEquDen(gNodeIdx),  srcnode.effDen()) );                   //2. transform from upeDen to effDen
      }
    }


    // Compute FFT of the translation
    for (int l = 1; l < _let->maxLevel(); l++) {  //TODO: Fix max level
      _matmgnt->UpwEqu2DwnChk_mat(l);
    }

    i = 0;
    int ifftset = 0;
    while(ifftset == 0 && i < ordVec.size()) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //evaluator
	Point3 gNodeIdxctr(_let->center(gNodeIdx));
	double D = 2.0 * _let->radius(gNodeIdx);
	DblNumVec evaTrgDwnChkVal(this->evaTrgDwnChkVal(gNodeIdx));
	for(vector<int>::iterator vi=_let->node(gNodeIdx).Vnodes().begin(); vi!=_let->node(gNodeIdx).Vnodes().end(); vi++) {
	  Point3 victr(_let->center(*vi));
	  Index3 idx;		  for(int d=0; d<dim(); d++)			 idx(d) = int(floor( (victr[d]-gNodeIdxctr[d])/D+0.5));
	  Node& srcnode = node(*vi);
	  Node& trgnode = node(gNodeIdx);
	  if(trgnode.vLstInCnt()==0) {
	    trgnode.effVal().resize( _matmgnt->effDatSze(DC) );			 setvalue(trgnode.effVal(), 0.0); //1. resize effVal
	  }
	  //M2L
	  iC( _matmgnt->UpwEqu2DwnChk_dgemv(_let->depth(gNodeIdx)+_rootLevel, idx, srcnode.effDen(), trgnode.effVal()) );

	  srcnode.vLstOthCnt()++;
	  trgnode.vLstInCnt()++;
	  if(srcnode.vLstOthCnt()==srcnode.vLstOthNum()) {
	    srcnode.effDen().resize(0); //1. resize effDen to 0
	    srcnode.vLstOthCnt()=0;
	  }
	  if(trgnode.vLstInCnt()==trgnode.vLstInNum()) {
	    iC( _matmgnt->effVal2PlnVal(_let->depth(gNodeIdx)+_rootLevel, trgnode.effVal(), evaTrgDwnChkVal) ); //1. transform from effval to DwnChkVal
	    trgnode.effVal().resize(0); //2. resize effVal to 0
	    trgnode.vLstInCnt()=0;
	  }
	  ifftset = 1;
	}
      }
      i++;
    }

#pragma omp parallel for schedule(guided)
    for(size_t j=i; j<ordVec.size(); j++) {
      int gNodeIdx = ordVec[j];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //evaluator
	Point3 gNodeIdxctr(_let->center(gNodeIdx));
	double D = 2.0 * _let->radius(gNodeIdx);
	DblNumVec evaTrgDwnChkVal(this->evaTrgDwnChkVal(gNodeIdx));
	for(vector<int>::iterator vi=_let->node(gNodeIdx).Vnodes().begin(); vi!=_let->node(gNodeIdx).Vnodes().end(); vi++) {
	  Point3 victr(_let->center(*vi));
	  Index3 idx;		  for(int d=0; d<dim(); d++)			 idx(d) = int(floor( (victr[d]-gNodeIdxctr[d])/D+0.5));
	  Node& srcnode = node(*vi);
	  Node& trgnode = node(gNodeIdx);
	  if(trgnode.vLstInCnt()==0) {
	    trgnode.effVal().resize( _matmgnt->effDatSze(DC) );			 setvalue(trgnode.effVal(), 0.0); //1. resize effVal
	  }
	  //M2L
	  iC( _matmgnt->UpwEqu2DwnChk_dgemv(_let->depth(gNodeIdx)+_rootLevel, idx, srcnode.effDen(), trgnode.effVal()) );

	  trgnode.vLstInCnt()++;
	  if(trgnode.vLstInCnt()==trgnode.vLstInNum()) {
	    iC( _matmgnt->effVal2PlnVal(_let->depth(gNodeIdx)+_rootLevel, trgnode.effVal(), evaTrgDwnChkVal) ); //1. transform from effval to DwnChkVal
	    trgnode.effVal().resize(0); //2. resize effVal to 0
	    trgnode.vLstInCnt()=0;
	  }

	}
      }
    }

    for(size_t j=0; j<_let->nodeVec().size(); j++)
      if( _let->node(j).tag() & LET_USERNODE )
	node(j).effDen().resize(0);

#ifdef HAVE_PAPI
    // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
    // papi_real_time, papi_proc_time, papi_mflops are just discarded
    if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
      SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
    PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  }
  return 0;
}


int FMM3d_MPI::evaluate_ulist()
{
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();
  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);

  PetscTruth gpu_ulist;
  PetscOptionsHasName(0,"-gpu_ulist",&gpu_ulist);
  if (gpu_ulist)
  {
#ifdef COMPILE_GPU
    // Interface U-list contribution calculation for GPU
    point3d_t *P;
    if ( (P = (point3d_t*) malloc (sizeof (point3d_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for u-list structure\n");
      return 1;
    }
    // Copy data into the u-list structure defined by 'P'

    // P->numSrc = (*_srcPos).n();
    P->numSrc =procLclNum(_usrSrcExaPos) ;
    P->numTrg = procLclNum(_evaTrgExaPos);
    P->dim = 3;
    P->kernel_type = _knl.kernelType();
    P->kernel_coef[0]=_knl.coefs()[0];

#ifdef USE_DOUBLE
    P->src_ = (double *) malloc(sizeof(double) * P->numSrc * (P->dim + srcDOF));
    P->trg_ = (double *) malloc(sizeof(double) * P->numTrg * P->dim);
    P->trgVal = (double *) calloc(P->numTrg*trgDOF, sizeof(double));
#else
    P->src_ = (float *) malloc(sizeof(float) * P->numSrc * (P->dim + srcDOF));
    P->trg_ = (float *) malloc(sizeof(float) * P->numTrg * P->dim);
    P->trgVal = (float *) calloc(P->numTrg*trgDOF, sizeof(float));
#endif

    P->uList = (int **) malloc (sizeof(int*) * ordVec.size());
    P->uListLen = (int *) calloc (ordVec.size(), sizeof(int));
    P->srcBoxSize = (int *) calloc (ordVec.size(), sizeof(int));
    P->trgBoxSize = (int *) calloc (ordVec.size(), sizeof(int));

    P->numTrgBox = ordVec.size();
    P->numSrcBox = ordVec.size();		// TODO: Are the total number of source and target boxes always the same?
    int j;
    int trgIndex = 0;
    int srcIndex = 0;
    int tv = 0;
    int d = 0;
    for(int i=ordVec.size()-1; i >= 0; i--) {
      int gNodeIdx = ordVec[i];
      P->uList[gNodeIdx] = NULL;
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  Let3d_MPI::Node& curNode = _let->node(gNodeIdx);
	  P->uList[gNodeIdx] = (int*) malloc (sizeof(int) * curNode.Unodes().size());
	  P->uListLen[gNodeIdx] = curNode.Unodes().size();
	  j = 0;
	  for(vector<int>::iterator vi=curNode.Unodes().begin(); vi!=curNode.Unodes().end(); vi++) {
	    P->uList[gNodeIdx][j] = *vi;
	    j++;
	  }
	  // P->trgBoxSize[gNodeIdx] = curNode.evaTrgExaNum();
	  DblNumMat evaTrgExaPosgNodeIdx(evaTrgExaPos(gNodeIdx));
	  P->trgBoxSize[gNodeIdx] =  evaTrgExaPosgNodeIdx.n();  // curNode.evaTrgExaNum();
	  assert (evaTrgExaPosgNodeIdx.n() == curNode.evaTrgExaNum());
	  for(int t = 0; t < P->trgBoxSize[gNodeIdx]; t++) {
	    for(d = 0; d < P->dim; d++)
	    {
	      // std::cout<<evaTrgExaPosgNodeIdx(d,t)<<" ";
	      P->trg_[(t*P->dim)+d+trgIndex] =  evaTrgExaPosgNodeIdx(d,t);
	    }
	    // std::cout<<endl;
	  }
	}
	trgIndex += (P->trgBoxSize[gNodeIdx] * P->dim);
	tv += P->trgBoxSize[gNodeIdx];
      }

      if( _let->node(gNodeIdx).tag() & LET_USERNODE) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  P->srcBoxSize[gNodeIdx] = _let->node(gNodeIdx).usrSrcExaNum();
	  for(int s = 0; s < P->srcBoxSize[gNodeIdx]; s++) {
	    for(d = 0; d < P->dim; d++)
	      P->src_[(s*(P->dim+srcDOF))+d+srcIndex] = (usrSrcExaPos(gNodeIdx)(d,s));
	    for(d = 0; d < srcDOF; d++)
	      P->src_[(s*(P->dim+srcDOF))+P->dim+d+srcIndex] = usrSrcExaDen(gNodeIdx)(srcDOF*s+d);
	  }
	}
	srcIndex += (P->srcBoxSize[gNodeIdx] * (P->dim+srcDOF));
      }
    }

    //  Calculate dense interations
    dense_inter_gpu(P);

    trgIndex = 0;
    // Copy target potentials back into the original structure
    // * for use by rest of the algorithm
    for(int i=ordVec.size()-1; i >= 0; i--) // actually any order is fine
    {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  for(int t = 0; t < P->trgBoxSize[gNodeIdx]; t++) {
	    for(d = 0; d < trgDOF; d++)
	      evaTrgExaVal(gNodeIdx)(trgDOF*t+d)= P->trgVal[trgDOF*t+d+trgIndex];
	  }
	}
      }
      trgIndex += P->trgBoxSize[gNodeIdx]*trgDOF;
    }

    // Free memory allocated for the interface
    free (P->src_);
    free (P->trg_);
    free (P->trgVal);
    free (P->uListLen);
    free (P->srcBoxSize);
    free (P->trgBoxSize);
    for(int i=ordVec.size()-1; i >= 0; i--)
      free (P->uList[ordVec[i]]);
    free (P->uList);
    free (P);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }
  else
  {
#ifdef HAVE_PAPI
    // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
    // papi_real_time, papi_proc_time, papi_mflops are just discarded
    if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
      SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
#pragma omp parallel for schedule(guided)
    for(size_t i=0; i<ordVec.size(); i++) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  DblNumVec evaTrgExaValgNodeIdx(evaTrgExaVal(gNodeIdx));
	  DblNumMat evaTrgExaPosgNodeIdx(evaTrgExaPos(gNodeIdx));
	  for(vector<int>::iterator vi=_let->node(gNodeIdx).Unodes().begin(); vi!=_let->node(gNodeIdx).Unodes().end(); vi++) {
	    //S2T
	    iC( SrcEqu2TrgChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), evaTrgExaPosgNodeIdx, usrSrcExaDen(*vi), evaTrgExaValgNodeIdx, fuse_dense) );
	  }
	}
      }
    }
#ifdef HAVE_PAPI
    // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
    // papi_real_time, papi_proc_time, papi_mflops are just discarded
    if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
      SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
    PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  }
  return 0;
}

int FMM3d_MPI::evaluate_upward()
{
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();
  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);

  PetscScalar zero=0.0;
  int ctbSrcNodeCnt = _let->ctbSrcNodeCnt();

  pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcNodeCnt*datSze(UE), &_ctbSrcUpwEquDen) );
  pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcNodeCnt*datSze(UC), &_ctbSrcUpwChkVal) );

  //  Somewhere near this point strange crash on kraken happens (with particular optimization options)
  //  MPI_Barrier(mpiComm());
  //  if (!mpiRank())
  //    cout<<"All processes created _ctbSrcUpwEquDen and _ctbSrcUpwChkVal"<<endl;

  pC( VecSet(_ctbSrcUpwEquDen, zero) );
  pC( VecSet(_ctbSrcUpwChkVal, zero) );

#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif

#ifdef COMPILE_GPU
  upComp_t *UpC;
#endif

  PetscTruth gpu_s2m;
  PetscOptionsHasName(0,"-gpu_s2m",&gpu_s2m);
  if (gpu_s2m)
    // compute s2m for all leaves at once
  {
#ifdef COMPILE_GPU
    const DblNumMat & sample_pos = _matmgnt->samPos(UC);
    vector<float> sample_pos_float(sample_pos.n()*sample_pos.m());
    for (size_t i=0; i<sample_pos_float.size(); i++)
      sample_pos_float[i]=*(sample_pos._data+i);

    int NumSrcBoxes=0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_CBTRNODE)
	NumSrcBoxes++;

    /* Allocate memory for the upward computation structure for GPU */
    if ( (UpC = (upComp_t*) calloc (1, sizeof (upComp_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for upward computation structure\n");
      return 1;
    }	
    /* Copy data into the upward computation structure defined by 'UpC' */
    UpC->tag = UC;
    UpC->numSrc = procLclNum(_usrSrcExaPos);
    UpC->dim = 3;
    UpC->numSrcBox = NumSrcBoxes;
    UpC->src_ = (float *) malloc(sizeof(float) * UpC->numSrc * (UpC->dim+srcDOF));
    UpC->trgVal = (float*) malloc (sizeof(float*) * NumSrcBoxes*sample_pos.n()*trgDOF);
    UpC->srcBoxSize = (int *) calloc (NumSrcBoxes, sizeof(int));
    UpC->trgCtr = (float *) calloc (UpC->numSrcBox * UpC->dim, sizeof(float));
    UpC->trgRad = (float *) calloc (UpC->numSrcBox, sizeof(float));
    UpC->trgDim=sample_pos.n();
    UpC->samPosF=&sample_pos_float[0];
    UpC->kernel_type = _knl.kernelType();
    UpC->kernel_coef[0]=_knl.coefs()[0];

    int srcIndex = 0;
    int srcBoxIndex=0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_CBTRNODE)
      {
	for (int j = 0; j < UpC->dim; j++)
	  UpC->trgCtr[j+srcBoxIndex*UpC->dim] = _let->center(gNodeIdx)(j);

	/* Radius of the box */
	UpC->trgRad[srcBoxIndex] = _let->radius(gNodeIdx);

	/* Source points and density stored as x1 y1 z1 d1 x2 y2 z2 d2 ..... */
	DblNumMat sources = ctbSrcExaPos(gNodeIdx);
	DblNumVec densities = ctbSrcExaDen(gNodeIdx);
	UpC->srcBoxSize[srcBoxIndex] = sources.n();
	for(int s = 0; s < UpC->srcBoxSize[srcBoxIndex]; s++) {
	  for(int d = 0; d < UpC->dim; d++)
	    UpC->src_[(s*(UpC->dim+srcDOF))+d+srcIndex] = sources(d,s);
	  for(int d = 0; d < srcDOF; d++)
	    UpC->src_[(s*(UpC->dim+srcDOF))+3+srcIndex+d] = densities(s*srcDOF+d);
	}
	srcIndex += (UpC->srcBoxSize[srcBoxIndex] * (UpC->dim+srcDOF));
	srcBoxIndex++;
      }

    gpu_up(UpC);

    // copy results back
    int trgValIndex=0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_CBTRNODE)
      {
	DblNumVec ctbSrcUpwChkValgNodeIdx(ctbSrcUpwChkVal(gNodeIdx));
	for (int j = 0; j < ctbSrcUpwChkValgNodeIdx.m(); j++) 
	  ctbSrcUpwChkValgNodeIdx(j) = UpC->trgVal[trgValIndex+j];
	// assert(UpC->trgVal[trgValIndex+j] == (j%3));
	trgValIndex += ctbSrcUpwChkValgNodeIdx.m();
      }

    free (UpC->src_);
    free (UpC->srcBoxSize);
    free (UpC->trgCtr);
    free (UpC->trgRad);
    free (UpC->trgVal);
    free (UpC);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }

    int level = 0;
    _matmgnt->UpwChk2UpwEqu_mat(level); //TODO: fix for non-homogenous kernels

#pragma omp parallel for schedule(guided) 
    for(int i=0; i<ordVec.size(); i++) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_CBTRNODE) {
	if(_let->depth(gNodeIdx)>=0) {
	  if(_let->terminal(gNodeIdx)==true) {
	    DblNumVec ctbSrcUpwChkValgNodeIdx(ctbSrcUpwChkVal(gNodeIdx));
	    DblNumVec ctbSrcUpwEquDengNodeIdx(ctbSrcUpwEquDen(gNodeIdx));
	    //S2M
	    if (!gpu_s2m)
	      SrcEqu2UpwChk_dgemv(ctbSrcExaPos(gNodeIdx), ctbSrcExaNor(gNodeIdx), _let->center(gNodeIdx), _let->radius(gNodeIdx), ctbSrcExaDen(gNodeIdx), ctbSrcUpwChkValgNodeIdx, fuse_dense);
	    //M2M
	    iC( _matmgnt->UpwChk2UpwEqu_dgemv(_let->depth(gNodeIdx)+_rootLevel, ctbSrcUpwChkValgNodeIdx, ctbSrcUpwEquDengNodeIdx) );
	  }
	}
      }
    }

    // non-leaf
    for(size_t i=0; i<ordVec.size(); i++) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_CBTRNODE) {
	if(_let->depth(gNodeIdx)>=0) {
	  if(_let->terminal(gNodeIdx)==false) {
	    DblNumVec ctbSrcUpwChkValgNodeIdx(ctbSrcUpwChkVal(gNodeIdx));
	    DblNumVec ctbSrcUpwEquDengNodeIdx(ctbSrcUpwEquDen(gNodeIdx));
	    //M2M
	    for(int a=0; a<2; a++) 
	      for(int b=0; b<2; b++) 
		for(int c=0; c<2; c++) {
		  Index3 idx(a,b,c);
		  int chi = _let->child(gNodeIdx, idx);
		  if(_let->node(chi).tag() & LET_CBTRNODE) {
		    pC( _matmgnt->UpwEqu2UpwChk_dgemv(_let->depth(chi)+_rootLevel, idx, ctbSrcUpwEquDen(chi), ctbSrcUpwChkValgNodeIdx) );
		  }
		}
	    //M2M
	    pC( _matmgnt->UpwChk2UpwEqu_dgemv(_let->depth(gNodeIdx)+_rootLevel, ctbSrcUpwChkValgNodeIdx, ctbSrcUpwEquDengNodeIdx) );
	  }
	}
      }
    }

#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  pC( VecDestroy(_ctbSrcUpwChkVal));
  _ctbSrcUpwChkVal=0;

   return 0;
}

int FMM3d_MPI::EquDenReduceBcast()
{
  PetscLogEventBegin(reduce_broadcast_bookkeeping,0,0,0,0);
  using namespace std;
  Let3d_MPI::reduce_bcast_data & comm_pattern = _let->comm_pattern;

  // estimate length of queue, recvd, to_send and extra
  int num_iter = comm_pattern.to_send.size();
  assert(size_t(num_iter)==comm_pattern.recv_sizes.size());
  assert(size_t(num_iter)==comm_pattern.extra_sizes.size());
  assert(size_t(num_iter)==comm_pattern.merging_srcs.size());
  assert(size_t(num_iter)==comm_pattern.merging_types.size());


  // load partial densities, do a plain copy for interior octants
  int density_size = _matmgnt->plnDatSze(UE);
  const int MAX_QUEUE_SIZE = comm_pattern.max_queue_size;
  vector<double> dens_1, dens_2;
  dens_1.resize(MAX_QUEUE_SIZE*density_size);
  dens_2.resize(MAX_QUEUE_SIZE*density_size);

  vector<double> * densities = & dens_1; //  new vector<double>(density_size*l->size());
  vector<bool> is_shared(_let->nodeVec().size(),false);
  {
    int idx=0;
    for (size_t i=0; i<comm_pattern.shared_octant.size(); i++)
    {
      int q = comm_pattern.shared_octant[i];
      is_shared[q]=true;

      double * data = ctbSrcUpwEquDen(q)._data;
      for (int j=0; j<density_size; j++)
	(*densities)[idx*density_size+j] = data[j];
      idx++;
    }
  }

  for (size_t i=0; i<_let->nodeVec().size(); i++)
    if ( !is_shared[i] && _let->node(i).tag() & LET_USERNODE && _let->node(i).tag() & LET_CBTRNODE)
    {
      double * data = ctbSrcUpwEquDen(i)._data;
      double * dest_data = usrSrcUpwEquDen(i)._data;
      for (int j=0; j<density_size; j++)
	dest_data[j] = data[j];
    }

  pC( VecDestroy (_ctbSrcUpwEquDen));
  _ctbSrcUpwEquDen=0;

  // need to find out actual max size during some sort of setup for reduce-broadcast
  const int MAX_SEND_SIZE = comm_pattern.max_to_send_size;
  const int MAX_RECVD_SIZE = comm_pattern.max_recvd_size;
  const int MAX_EXTRA_SIZE = comm_pattern.max_extra_size;
  vector<double> densities_to_send;
  densities_to_send.resize(MAX_SEND_SIZE*density_size);
  vector<double> recvd_densities;
  recvd_densities.resize(MAX_RECVD_SIZE*density_size);
  vector<double> extra_densities;
  extra_densities.resize(MAX_EXTRA_SIZE*density_size);

  // dump some statistics for debugging
  //gb PetscLogDouble mem1;
  //gb PetscMemoryGetCurrentUsage(&mem1);
  //gb PetscLogDouble max_mem1;
  //gb MPI_Reduce ( &mem1, &max_mem1, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, 0, mpiComm() );
  //gb if (!mpiRank())    cout<<" PetscMemoryGetCurrentUsage (max) right before hypercube loop: "<<max_mem1<<endl;

  // communication loop:
  // We start with a "partition" consisting of all ranks; Each partition is split into two almost equal halves, halves exhange data and then each half becomes independent "partition", is split into new halves and so forth
  int first = 0;
  int last = mpiSize()-1;
  int iteration = 0;

  while (true)
  {
    assert(mpiRank()>=first && mpiRank()<=last);
    if (last==first)
      break; // this rank has finished reduce-broadcast

    int split = (first+last)/2;  // integer division!
    // first half of the partition is  first=<rank<=split
    // second half is remainder

    int half_size = (last-first)/2 + 1;  // e.g, for first=0, last=2 half_size will be 2
    int partner = mpiRank()<=split? mpiRank()+half_size:mpiRank()-half_size;
    assert(partner>=first);
    assert(partner<=last || mpiRank()==split && partner==last+1);

    // group densities-to-send together
    int send_size = comm_pattern.to_send[iteration].size();
    int recv_size = comm_pattern.recv_sizes[iteration];
    for (int i=0; i<send_size; i++)
    {
      double * data = &(*densities)[0] + density_size*comm_pattern.to_send[iteration][i];
      for (int j=0; j<density_size; j++)
	densities_to_send[i*density_size+j] = data[j];
    }

    PetscLogEventEnd(reduce_broadcast_bookkeeping,0,0,0,0);
    // exchange densities
    MPI_Status status;
    if (partner<=last) // this rank has a pair
      MPI_Sendrecv(send_size? &densities_to_send[0]:0, send_size*density_size, MPI_DOUBLE, partner, 0, recv_size? &recvd_densities[0]:0, recv_size*density_size, MPI_DOUBLE, partner, 0, mpiComm(), &status);
    else
      MPI_Send(send_size? &densities_to_send[0]:0, send_size*density_size, MPI_DOUBLE, last, 0, mpiComm());

    // for odd-sized partitions, last rank in partition will receive extra octants from rank "split"
    int extra_size=comm_pattern.extra_sizes[iteration];
    if ( (last-first+1)%2 &&  mpiRank()==last) // we are to receive "extra" octants from rank "split"
      MPI_Recv(extra_size? &extra_densities[0]:0, extra_size*density_size, MPI_DOUBLE, split, 0, mpiComm(), &status);

    PetscLogEventBegin(reduce_broadcast_bookkeeping,0,0,0,0);

    // merge l and received octants; take only those octants from l, which are adressed to this process or still have to be sent somewhere; we assume both l and recvd  are Morton sorted (in ascending order)
    assert(densities == &dens_1 || densities == &dens_2);
    vector<double> * new_densities = densities == &dens_1 ? &dens_2 : &dens_1 ;
    
    // merging loop, we assume that "l", "recvd" and "extra"  are Morton-sorted 
    size_t ll_ptr=0;
    size_t recv_ptr=0;
    size_t extra_ptr=0;
    size_t new_l_ptr=0;
    for(size_t i=0; i<comm_pattern.merging_srcs[iteration].size(); i++)
    {
      size_t *pp ;  // pointer to one of the indices ll_ptr, recv_ptr or extr_ptr
      double *pd ;  // pointer to corresponding location in one  of the density arrays: densities, recvd_densities or extra_densities

      switch (comm_pattern.merging_srcs[iteration][i])
      {
	case Let3d_MPI::QUEUE:
	  pp = &ll_ptr;
	  pd =  &(*densities)[ll_ptr*density_size];
	  break;
	case Let3d_MPI::RECVD:
	  pp = &recv_ptr;
	  pd = &recvd_densities[recv_ptr*density_size];
	  break;
	case Let3d_MPI::EXTRA:
	  pp = &extra_ptr;
	  pd = &extra_densities[extra_ptr*density_size];
	  break;
      }

      switch(comm_pattern.merging_types[iteration][i])
      {
	case Let3d_MPI::ADD: // same octant as previously pushed to new_l; we need to sum up densities
	  {
	    assert(new_l_ptr>0);
	    size_t new_density_idx = (new_l_ptr-1)*density_size; 
	    for (int j=0; j<density_size; j++)
	      (*new_densities)[new_density_idx+j] += pd[j];
	  }
	  break;
	case Let3d_MPI::INSERT:
	  {
	    size_t new_density_idx = new_l_ptr*density_size; 
	    for (int j=0; j<density_size; j++)
	      (*new_densities)[new_density_idx+j] = pd[j];
	    new_l_ptr++;
	  }
	  break;
	case Let3d_MPI::DROP:
	  break;
      }
      (*pp)++;    // increase the index in the array we picked an octant from (ll, recvd, or extra)
    }

    densities = new_densities;

    // preparing for next iteration:  each half of each partition becomes a partition of its own
    first = mpiRank()<=split ? first : split+1;
    last = mpiRank()<=split ? split : last;
    iteration++;

  } // end of the communication loop

  // now insert received octants in local tree
  // some received octants may already be present in the tree, then just set the global indices
  // for now, we'll do simplistic implementation:  for each  received octant we start from root and go down the tree to find an appropriate place
  for (size_t i=0; i<comm_pattern.usr_octants.size(); i++)
  {
    int q = comm_pattern.usr_octants[i];
    
    if (_let->node(q).tag() & LET_USERNODE ) //  if this octant is really used on this process (is on some list of some "local" octant)
    {
      double * data = usrSrcUpwEquDen(q)._data;
      for (int j=0; j<density_size; j++)
	data[j]= (*densities)[i*density_size+j];
    }
  }
  PetscLogEventEnd(reduce_broadcast_bookkeeping,0,0,0,0);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::evaluate"
int FMM3d_MPI::evaluate(Vec srcDen, Vec trgVal)
{
#ifdef HAVE_PAPI
  // these variables are for use with PAPI
  float papi_real_time, papi_proc_time, papi_mflops;
  long_long papi_flpops=0, papi_flpops2;
  int papi_retval;
#endif

  PetscLogEventBegin(EvalIni_event,0,0,0,0);
  //begin  //ebiLogInfo( "multiply.............");
  //-----------------------------------
  //cerr<<"fmm src and trg numbers "<<pglbnum(_srcPos)<<" "<<pglbnum(_trgPos)<<endl;
  PetscInt tmp;
  pC( VecGetSize(srcDen,&tmp) );  pA(tmp==srcDOF()*procGlbNum(_srcPos));
  pC( VecGetSize(trgVal,&tmp) );  pA(tmp==trgDOF()*procGlbNum(_trgPos));

  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();

  // shall we skip all communication? (results will be incorrect, of course)
  PetscTruth skip_communication;
  PetscOptionsHasName(0,"-eval_skip_communication",&skip_communication);
  if (skip_communication && !mpiRank())
    std::cout<<"!!!!! All communications during interaction evaluation are skipped. Results are incorrect !!!!"<<endl; 
      
  PetscTruth fuse_dense;
  PetscOptionsHasName(0,"-fuse_dense",&fuse_dense);

  //1. zero out vecs.  This includes all global, contributor, user, evaluator vectors.
  PetscScalar zero=0.0;
  pC( VecSet(trgVal, zero) );
  pC( VecSet(_glbSrcExaDen, zero) );
  pC( VecSet(_ctbSrcExaDen, zero) );
  pC( VecSet(_usrSrcExaDen, zero) );
  pC( VecSet(_usrSrcUpwEquDen, zero) );
  pC( VecSet(_evaTrgExaVal, zero) );
  pC( VecSet(_evaTrgDwnEquDen, zero) );
  pC( VecSet(_evaTrgDwnChkVal, zero) );

  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes

  //2. for contributors, load exact densities
  PetscInt procLclStart, procLclEnd; _let->procLclRan(_srcPos, procLclStart, procLclEnd);
  double* darr; pC( VecGetArray(srcDen, &darr) );
  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if(_let->node(gNodeIdx).tag() & LET_CBTRNODE) {
		if(_let->terminal(gNodeIdx)==true) {
		  DblNumVec ctbSrcExaDen(this->ctbSrcExaDen(gNodeIdx));
		  vector<PetscInt>& curVecIdxs = _let->node(gNodeIdx).ctbSrcOwnVecIdxs();
		  for(size_t k=0; k<curVecIdxs.size(); k++) {
			 PetscInt poff = curVecIdxs[k] - procLclStart;
			 for(int d=0; d<srcDOF; d++) {
				ctbSrcExaDen(k*srcDOF+d) = darr[poff*srcDOF+d];
			 }
		  }
		}
	 }
  }
  pC( VecRestoreArray(srcDen, &darr) );
  PetscLogEventEnd(EvalIni_event,0,0,0,0);

  if (!skip_communication)
  {
    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin
    PetscLogEventBegin(EvalCtb2GlbExa_event,0,0,0,0);
    // send source densities from contributors to owners; this now should not involve any MPI communication, since for all leaf nodes in global tree owners are the only contributors; maybe eventually I'll remove this scatter at all
    pC( VecScatterBegin(_ctb2GlbSrcExaDen, _ctbSrcExaDen, _glbSrcExaDen,    ADD_VALUES, SCATTER_FORWARD) );
    pC( VecScatterEnd(_ctb2GlbSrcExaDen,  _ctbSrcExaDen, _glbSrcExaDen,    ADD_VALUES, SCATTER_FORWARD) );
    PetscLogEventEnd(EvalCtb2GlbExa_event,0,0,0,0);

    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin
    PetscLogEventBegin(EvalGlb2UsrExaBeg_event,0,0,0,0);
    // we overlap sending of charge densities from owners to users with upward computation
    pC( VecScatterBegin(_usr2GlbSrcExaDen, _glbSrcExaDen, _usrSrcExaDen, INSERT_VALUES, SCATTER_REVERSE) );
    PetscLogEventEnd(EvalGlb2UsrExaBeg_event,0,0,0,0);
  }

  //3. up computation
  PetscLogEventBegin(EvalUpwComp_event,0,0,0,0);
  evaluate_upward();
  PetscLogEventEnd(EvalUpwComp_event,0,0,0,0);

  //4. vectbscatters
  if (!skip_communication)
  {
    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin/End
    PetscLogEventBegin(EvalGlb2UsrEquBeg_event,0,0,0,0);
    // dump current and maximum memory usage
    PetscLogDouble mem1, max_mem1;

    //gb PetscMemoryGetCurrentUsage(&mem1);
    //gbMPI_Reduce ( &mem1, &max_mem1, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    //gbif (!mpiRank())    cout<<"PetscMemoryGetCurrentUsage (max over processes) BEFORE reduce-broadcast: "<<max_mem1<<endl;

		//gb  PetscMemoryGetMaximumUsage(&mem1);
		//gb MPI_Reduce ( &mem1, &max_mem1, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
		//gb    if (!mpiRank())   cout<<"PetscMemoryGetMaximumUsage (max over processes) BEFORE reduce-broadcast: "<<max_mem1<<endl;

      EquDenReduceBcast();
      MPI_Barrier(mpiComm());

      if (!mpiRank())
	cout<<"All processes returned from reduce-broadcast"<<endl;
    PetscLogEventEnd(EvalGlb2UsrEquBeg_event,0,0,0,0);

    //gbPetscMemoryGetCurrentUsage(&mem1);
    //gbMPI_Reduce ( &mem1, &max_mem1, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    //gbif (!mpiRank())      cout<<"PetscMemoryGetCurrentUsage (max over processes) AFTER reduce-broadcast: "<<max_mem1<<endl;

    //gbPetscMemoryGetMaximumUsage(&mem1);
    //gbMPI_Reduce ( &mem1, &max_mem1, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    //gbif (!mpiRank())      cout<<"PetscMemoryGetMaximumUsage (max over processes) AFTER reduce-broadcast: "<<max_mem1<<endl;

    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin/End
    PetscLogEventBegin(EvalGlb2UsrExaEnd_event,0,0,0,0);
    // we overlap sending of charge densities from owners to users with upward computation (scatterBegin is several lines above)
    pC( VecScatterEnd(_usr2GlbSrcExaDen, _glbSrcExaDen, _usrSrcExaDen, INSERT_VALUES, SCATTER_REVERSE) );
    PetscLogEventEnd(EvalGlb2UsrExaEnd_event,0,0,0,0);
  }

  // U-list computation
  PetscLogEventBegin(EvalUList_event,0,0,0,0);
  evaluate_ulist();
  PetscLogEventEnd(EvalUList_event,0,0,0,0);

  //V
  PetscLogEventBegin(EvalVList_event,0,0,0,0);
  evaluate_vlist();
  PetscLogEventEnd(EvalVList_event,0,0,0,0);

  //W
  PetscLogEventBegin(EvalWList_event,0,0,0,0);
  evaluate_wlist();
  PetscLogEventEnd(EvalWList_event,0,0,0,0);

  //X
  PetscLogEventBegin(EvalXList_event,0,0,0,0);
  evaluate_xlist();
  PetscLogEventEnd(EvalXList_event,0,0,0,0);

  //7. combine
  PetscLogEventBegin(EvalCombine_event,0,0,0,0);
  evaluate_downward();
  PetscLogEventEnd(EvalCombine_event,0,0,0,0);

  PetscLogEventBegin(EvalFinalize_event,0,0,0,0);
  //8. save tdtExaVal
  _let->procLclRan(_trgPos, procLclStart, procLclEnd);
  double* varr; pC( VecGetArray(trgVal, &varr) );
  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) {
		if( _let->terminal(gNodeIdx)==true ) {
		  DblNumVec evaTrgExaVal(this->evaTrgExaVal(gNodeIdx));
		  vector<PetscInt>& curVecIdxs = _let->node(gNodeIdx).evaTrgOwnVecIdxs();
		  for(size_t k=0; k<curVecIdxs.size(); k++) {
			 PetscInt poff = curVecIdxs[k] - procLclStart;
			 for(int d=0; d<trgDOF; d++) {
				varr[poff*trgDOF+d] = evaTrgExaVal(k*trgDOF+d);
			 }
		  }
		}
	 }
  }
  pC( VecRestoreArray(trgVal, &varr) );
  PetscLogEventEnd(EvalFinalize_event,0,0,0,0);

  // I don't understand the role of barrier below. Let's remove it and see if things break.
  // pC( MPI_Barrier(mpiComm()) );  //check vLstInCnt, vLstOthCnt
  return(0);
}

