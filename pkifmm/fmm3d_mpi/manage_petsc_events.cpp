#include <iomanip>
#include <vector>
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "manage_petsc_events.hpp"
#include "petsc.h"
#include "petsctime.h"
#include <stdarg.h>
#include <sys/types.h>
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#include "petscfix.h"
// #include "plog.h"

PetscLogEvent  fmm_eval_init;
PetscLogEvent  let_setup_event;
PetscLogEvent  fmm_setup_event;
PetscLogEvent  fmm_eval_event;
PetscLogEvent  fmm_check_event;
PetscLogEvent  a2a_numOct_event;
PetscLogEvent  a2aV_octData_event;
PetscLogEvent  EvalIni_event;
PetscLogEvent  EvalCtb2GlbExa_event;
PetscLogEvent  EvalGlb2UsrExaBeg_event;
PetscLogEvent  EvalUpwComp_event;
PetscLogEvent  EvalCtb2GlbEqu_event;
PetscLogEvent  EvalGlb2UsrEquBeg_event;
PetscLogEvent  EvalGlb2UsrExaEnd_event;
PetscLogEvent  EvalUList_event;
PetscLogEvent  EvalGlb2UsrEquEnd_event;
PetscLogEvent  EvalVList_event;
PetscLogEvent  EvalWList_event;
PetscLogEvent  EvalXList_event;
PetscLogEvent  EvalCombine_event;
PetscLogEvent  EvalFinalize_event;
PetscLogEvent  Ctb2GlbSctCreate_event;
PetscLogEvent  Usr2GlbSctCreate_event;
PetscLogEvent  reduce_broadcast_bookkeeping;
PetscLogEvent  droplets_rhs;
PetscLogEvent  droplets_id_redistr;
PetscLogStage  stages[6];

PetscCookie kifmm3d_logClass;

void registerPetscEvents()
{
  PetscCookieRegister("KIFMM3D",&kifmm3d_logClass);

  PetscLogEventRegister("let setup",kifmm3d_logClass,&let_setup_event);
  PetscLogEventRegister("fmm setup",kifmm3d_logClass,&fmm_setup_event);
  PetscLogEventRegister("fmm eval",kifmm3d_logClass,&fmm_eval_event);
  PetscLogEventRegister("fmm check",kifmm3d_logClass,&fmm_check_event);
  PetscLogEventRegister("a2a num. oct",kifmm3d_logClass,&a2a_numOct_event);
  PetscLogEventRegister("a2aV octData",kifmm3d_logClass,&a2aV_octData_event);
  PetscLogEventRegister("EvlIni",kifmm3d_logClass,&EvalIni_event);
  PetscLogEventRegister("EvlCtb2GlbExa",kifmm3d_logClass,&EvalCtb2GlbExa_event);
  PetscLogEventRegister("EvlGlb2UsrExaBeg",kifmm3d_logClass,&EvalGlb2UsrExaBeg_event);
  PetscLogEventRegister("EvlUpwComp",kifmm3d_logClass,&EvalUpwComp_event);
  PetscLogEventRegister("EvlCtb2GlbEqu",kifmm3d_logClass,&EvalCtb2GlbEqu_event);
  PetscLogEventRegister("EvlGlb2UsrEquBeg",kifmm3d_logClass,&EvalGlb2UsrEquBeg_event);
  PetscLogEventRegister("EvlGlb2UsrExaEnd",kifmm3d_logClass,&EvalGlb2UsrExaEnd_event);
  PetscLogEventRegister("EvlUList",kifmm3d_logClass,&EvalUList_event);
  PetscLogEventRegister("EvlGlb2UsrEquEnd",kifmm3d_logClass,&EvalGlb2UsrEquEnd_event);
  PetscLogEventRegister("EvlVList",kifmm3d_logClass,&EvalVList_event);
  PetscLogEventRegister("EvlWList",kifmm3d_logClass,&EvalWList_event);
  PetscLogEventRegister("EvlXList",kifmm3d_logClass,&EvalXList_event);
  PetscLogEventRegister("EvlCombine",kifmm3d_logClass,&EvalCombine_event);
  PetscLogEventRegister("EvlFinalize",kifmm3d_logClass,&EvalFinalize_event);
  PetscLogEventRegister("Ctb2GlbSctCreate",kifmm3d_logClass,&Ctb2GlbSctCreate_event);
  PetscLogEventRegister("Usr2GlbSctCreate",kifmm3d_logClass,&Usr2GlbSctCreate_event);
  PetscLogEventRegister("RedBcastBookk",kifmm3d_logClass,&reduce_broadcast_bookkeeping);
  PetscLogEventRegister("droplets_rhs",kifmm3d_logClass,&droplets_rhs);
  PetscLogEventRegister("dropIDredist",kifmm3d_logClass,&droplets_id_redistr);
  PetscLogStageRegister("GenSrcTrgPosNor", stages+0);
  PetscLogStageRegister("FMM_Setup", stages+1);
  PetscLogStageRegister("GenSrcDen", stages+2);
  PetscLogStageRegister("FMM_evaluate1", stages+3);
  PetscLogStageRegister("FMM_evaluate2", stages+4);
  PetscLogStageRegister("FMM_check", stages+5);
}

PetscErrorCode DumpPerCoreSummary(MPI_Comm comm, const char filename[]) 
{
  FILE          *fd   = stdout;
  StageLog       stageLog;
  StageInfo     *stageInfo = PETSC_NULL;
  EventPerfInfo *eventInfo = PETSC_NULL;
  PetscMPIInt    size, rank;
  PetscTruth     *stageUsed;
  PetscTruth     *stageVisible;
  int            numStages, numEvents;
  int            stage;
  int            event;
  PetscErrorCode ierr;
  int            numActualEvents;
  int            * maxCtArray;

  PetscFunctionBegin;
  using namespace std;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Pop off any stages the user forgot to remove */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    ierr = StageLogPop(stageLog);CHKERRQ(ierr);
    ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }

  numStages = stageLog->numStages;
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageVisible);CHKERRQ(ierr);
  if (numStages > 0) {
    stageInfo = stageLog->stageInfo;
    for(stage = 0; stage < numStages; stage++) {
      if (stage < stageLog->numStages) {
        stageUsed[stage]    = stageInfo[stage].used;
        stageVisible[stage] = stageInfo[stage].perfInfo.visible;
      } else {
        stageUsed[stage]    = PETSC_FALSE;
        stageVisible[stage] = PETSC_TRUE;
      }
    }
  }

  /* Report events */
  /* Problem: The stage name will not show up unless the stage executed on proc 1 */
  for(stage = 0; stage < numStages; stage++) {
    if (!stageVisible[stage]  || !stageUsed[stage]) continue;
    /* Open the summary file */
    if (filename && !rank){
      ostringstream oss;
      oss<<stageInfo[stage].name<<"_"<<filename;
      // cout<<oss.str();
      fd = fopen(oss.str().c_str(), "w"); CHKERRQ(fd==0);
    }

    /* Get total number of events in this stage --
    */
    eventInfo      = stageLog->stageInfo[stage].eventLog->eventInfo;
    numEvents = stageLog->stageInfo[stage].eventLog->numEvents;

    ierr = PetscMalloc(numEvents * sizeof(int), &maxCtArray);CHKERRQ(ierr);
    numActualEvents=0;
    for(event = 0; event < numEvents; event++)
    {
      ierr = MPI_Allreduce(&eventInfo[event].count, maxCtArray+event, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
      if (maxCtArray[event]) numActualEvents++;
    }

    // process 0 writes event names
    if (!rank)
    {
      fprintf(fd,"%d processors\n%d events (names follow)\n",size,numActualEvents);
      for(event = 0; event < numEvents; event++)
	if (maxCtArray[event])
	  fprintf(fd,"%s\n",stageLog->eventLog->eventInfo[event].name);
      fprintf(fd,"6 Fields (names follow)\nCount\nTime-sec\nFlops\nMess\nTotal-len\nReduct\n");
    }

    for(event = 0; event < numEvents; event++)
      if (maxCtArray[event] != 0) {
	ostringstream oss;
	oss<<setprecision(3)<<eventInfo[event].count<<" "<<eventInfo[event].time<<" "<<eventInfo[event].flops<<" "<<eventInfo[event].numMessages<<" "<<eventInfo[event].messageLength<<" "<<eventInfo[event].numReductions<<endl;
	int locSize=oss.str().length();
	if (0==rank)
	{
	  vector<int> locSizes(size);
	  // cout<<"About to perform Gathers"<<endl;
	  MPI_Barrier(comm);
	  MPI_Gather ( &locSize, 1, MPI_INT, &locSizes[0], 1, MPI_INT, 0, comm );
	  vector<int> displs(size);
	  displs[0]=0;
	  for (int i=1; i<size; i++)
	    displs[i]=displs[i-1]+locSizes[i-1];
	  vector<char> globalStr(displs.back()+locSizes.back()+1);
	  globalStr.back()=0;
	  MPI_Barrier(comm);
	  MPI_Gatherv (const_cast<char *>(oss.str().data()), oss.str().length(), MPI_CHAR, &globalStr[0], &locSizes[0], &displs[0], MPI_CHAR, 0, comm);
	  // cout<<"Both Gathers worked"<<endl;
	  fprintf(fd,"%s", &globalStr[0]);
	}
	else
	{
	  MPI_Barrier(comm);
	  MPI_Gather ( &locSize, 1, MPI_INT, 0, 1, MPI_INT, 0, comm );
	  MPI_Barrier(comm);
	  MPI_Gatherv (const_cast<char *>(oss.str().data()), oss.str().length(), MPI_CHAR, 0, 0, 0, MPI_CHAR, 0, comm);
	}
      }
    if (!rank && filename)
      fclose(fd);
    ierr = PetscFree(maxCtArray);CHKERRQ(ierr);
  }

  ierr = PetscFree(stageUsed);CHKERRQ(ierr);
  ierr = PetscFree(stageVisible);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
