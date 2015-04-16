//Qi Zheng -- VCA study

#include <map>
#include <vector>

//#define RF_STUDY
//#define RF_TRACE

#ifndef RF_STUDY_H
#define RF_STUDY_H
//#ifdef RF_STUDY
extern std::map<unsigned, unsigned  >* warp_idx; 
extern std::vector< std::map<unsigned, unsigned long long> >* reg_tot_dist; 
extern std::vector< std::map<unsigned, unsigned long long> >* reg_acc_count;
extern std::vector< std::map<unsigned, unsigned long long> >* reg_last_acc;
extern unsigned long long* vector_size;
//#endif
#endif /*RF_STUDY_H*/
