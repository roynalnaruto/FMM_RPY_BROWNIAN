#ifndef MANAGE_PETSC_EVENTS_HEADER
#define MANAGE_PETSC_EVENTS_HEADER

#include "petsc.h"

extern PetscLogEvent  let_setup_event;
extern PetscLogEvent  fmm_setup_event;
extern PetscLogEvent  fmm_eval_event;
extern PetscLogEvent  fmm_check_event;
extern PetscLogEvent  a2a_numOct_event;
extern PetscLogEvent  a2aV_octData_event;
extern PetscLogEvent  EvalIni_event;
extern PetscLogEvent  EvalCtb2GlbExa_event;
extern PetscLogEvent  EvalGlb2UsrExaBeg_event;
extern PetscLogEvent  EvalUpwComp_event;
extern PetscLogEvent  EvalCtb2GlbEqu_event;
extern PetscLogEvent  EvalGlb2UsrEquBeg_event;
extern PetscLogEvent  EvalGlb2UsrExaEnd_event;
extern PetscLogEvent  EvalUList_event;
extern PetscLogEvent  EvalGlb2UsrEquEnd_event;
extern PetscLogEvent  EvalVList_event;
extern PetscLogEvent  EvalWList_event;
extern PetscLogEvent  EvalXList_event;
extern PetscLogEvent  EvalCombine_event;
extern PetscLogEvent  EvalFinalize_event;
extern PetscLogEvent  Ctb2GlbSctCreate_event; 
extern PetscLogEvent  Usr2GlbSctCreate_event; 
extern PetscLogEvent  reduce_broadcast_bookkeeping;
extern PetscLogEvent  droplets_rhs;
extern PetscLogEvent  droplets_id_redistr;
extern PetscLogStage  stages[6];

void registerPetscEvents();
PetscErrorCode DumpPerCoreSummary(MPI_Comm comm, const char * fname);

#endif
