#!/usr/bin/python
## GPGPU irregular application project: parsing script

import os, sys, re

try:
  filename = sys.argv[1]
  if filename[-4:] == '.out':
    filename = filename[:-4]
except:
  print 'Usage: trace.py filename'
  sys.exit()

warps = []
tracefile = open(filename + '.out', 'r')
line = tracefile.readline()
while line:
  if line[0:4] == 'TEST':  # profile outputs
    line = line.split('\t')
    if line[2] == 'start\n':
      line = tracefile.readline().split('\t')
    elif line[2] == 'exit\n':
      line = tracefile.readline().replace('\n','')
      line = line.split('\t')
#      core_id = line[2]
#      warps_id = line[4]
#      dynamic_warps_id = line[6]
      start_cycle = int(line[8])
#      finish_cycle = line[10]
      run_time = int(line[12])
      inst_committed = int(line[14])
      warps.append([start_cycle, run_time, inst_committed])
#      run_time_list.append(run_time)
#      inst_committed_list.append(inst_committed)
  line = tracefile.readline()

warps = sorted(warps, key=lambda warps: (warps[0], warps[1]))    # sort based on start cycle first, then run time
csvfile = open(filename + '.csv', 'w+')
csvfile.write('start_cycle' + '\t' + 'run_time' + '\t' + 'inst_committed' + '\n')
for warp in warps:
  for item in warp: 
    csvfile.write(str(item))
    csvfile.write('\t')
  csvfile.write('\n')

tracefile.close()
csvfile.close()
