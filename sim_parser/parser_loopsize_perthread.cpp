//Unpublished Copyright (c) 2015 Qi Zheng, All Rights Reserved.
//This is a parser program that reads the output of a cuda program, and records the loop size of every thread. The input of the parser is a csv file that contains the thread id and corresponding loop size. The output of the parser is an estimation of the performance improvement with dynamic thread merging and stealing.
//The input format is:
//  <thread id> <loop size>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#define WARPSIZE 32

using namespace std;

char *benchmark;
int SMnum;
int64_t alllines;

//for oracle formatting
vector<int>threadloopsize;
int maxtid;

//for reality
int curwid;
int curmaxloopsize;
vector<int>warploopsize;

bool isfirst;

int kernelcount;

void init(){
  alllines = 0;
  kernelcount = 0;
  maxtid = -1;
  threadloopsize.clear();
  curwid = -1;
  warploopsize.clear();
  curmaxloopsize = -1;
  isfirst = true;
}

void reinit(){
  maxtid = -1;
  threadloopsize.clear();
  curwid = -1;
  warploopsize.clear();
  curmaxloopsize = -1;
}

void print_result(){
  if(isfirst){
    isfirst = false;
    return;
  }

  int tot_warp_num = (maxtid/WARPSIZE) + 1;
  if(tot_warp_num != warploopsize.size()){
    fprintf(stderr,"At lines %ld -- Inconsistent warp size: %d vs. %lu\n",alllines, tot_warp_num, warploopsize.size());
    exit(-1);
  }
  if(maxtid != threadloopsize.size() - 1){
    fprintf(stderr,"At lines %ld -- Inconsistent thread size: %d vs. %lu\n",alllines, maxtid + 1, threadloopsize.size());
    exit(-1);
  }
  int warp_per_core = tot_warp_num/SMnum;
  int extra = tot_warp_num - (warp_per_core * SMnum);
  int itr = 0;
  printf("Kernel: %d\n",kernelcount);
  for(int i=0;i<SMnum;i++){
    int mywarp_per_core = warp_per_core;
    if(extra > 0)
      mywarp_per_core += 1;
    int thread_range = mywarp_per_core * WARPSIZE;
    sort(threadloopsize.begin()+itr, threadloopsize.begin()+itr+thread_range);//from little to large
    int oracleloopsize = 0;
    int realloopsize = 0;
    for(int j=0;j<mywarp_per_core;j++){
      oracleloopsize += threadloopsize.at((j+1)*WARPSIZE-1);
      realloopsize += warploopsize.at(j);
    }
    printf("  Core %d improves %.2f%\n",i,((float)realloopsize/(float)oracleloopsize-1.0) * 100.0);

    extra--;
    itr += thread_range;
  }
}

int parse_line(char* input){
  int tid, curloopsize;
  sscanf(input, "%d %d",&tid,&curloopsize);

  threadloopsize.push_back(curloopsize);
  maxtid = tid;

  int tempwid = tid/WARPSIZE;
  if((tid+1)%WARPSIZE == 0){//last thread of a warp
    if(curmaxloopsize < curloopsize) 
      curmaxloopsize = curloopsize;
    warploopsize.push_back(curmaxloopsize);
    curwid = -1;
    curmaxloopsize = -1;
  }else{
    if(curmaxloopsize < curloopsize) 
      curmaxloopsize = curloopsize;
  }

  return 1;
}

int detect_new(char* input){
  const char* ref = "Kernel:";
  char curhead[7];
  memcpy(curhead, input, 7);
  if(memcmp(ref, curhead,7)){
    return 0;
  }

  print_result();
  reinit();
  kernelcount++;
  return 1;
}

void process_trace(){
        int i;
        char* line = NULL;
        size_t len = 0;
        ssize_t readlen;

        FILE* fp = fopen(benchmark,"r");
        if(fp == NULL){
            fprintf(stderr,"Cannot open %s\n", benchmark);
        }
       	while((readlen = getline(&line, &len, fp)) != -1){
		alllines++;
                if(detect_new(line)){
                  continue;
                }
                parse_line(line);
        }	

        if(line) free(line);
        print_result();
}

int main(int argc, char **argv){
        if(argc != 3){
                fprintf(stderr, "Usage: executable <csv file> <#SM>\n");
                exit(-1);
        }
	benchmark = argv[1];
        SMnum = atoi(argv[2]);
        if(SMnum <= 0){
          fprintf(stderr,"Wrong #SM!\n");
          assert(0);
        }

	init();	
	process_trace();
	return 0;
}

