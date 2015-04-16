//This is the header file for the study of irregular applications by Qi Zheng and Cao Gao at University of Michigan, Ann Arbor.
//Important parameters: 
//    INSTCOMMIT_MAX_WARP -- this value must be consistent with gpgpusim.config file.
//    INSTCOMMIT_SM -- this value must be consistent with gpgpusim.config.file.

//#define IRREGULAR_MOT
//#define DEBUG_ORACLE_INSTCOMMIT

#include <map>
#include <vector>

#ifndef INST_COMMIT_H
#define INST_COMMIT_H

#define INSTCOMMIT_MAX_WARP 48 
#define INSTCOMMIT_SM 4
extern unsigned instcommit_dyn_warp_id[INSTCOMMIT_SM][INSTCOMMIT_MAX_WARP]; 
extern unsigned long long warp_committed_inst[INSTCOMMIT_SM][INSTCOMMIT_MAX_WARP];
//#endif
#endif /*INST_COMMIT_H*/
