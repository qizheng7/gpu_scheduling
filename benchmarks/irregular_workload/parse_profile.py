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

kernel = -1
warps = [[]]
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
      warps[kernel].append([start_cycle, run_time, inst_committed])
#      run_time_list.append(run_time)
#      inst_committed_list.append(inst_committed)
  if line[0:17] == 'New kernel starts':  # profile outputs
    kernel = kernel + 1
    warps.append([])
  line = tracefile.readline()

for kernel_n in xrange(kernel+1):
  csvfile = open(filename + '_' + str(kernel_n) + '.csv', 'w+')
  csvfile.write('new kernel' + str(kernel_n) + '\n')
  warp = warps[kernel_n]
  warp = sorted(warp, key=lambda warp: (warp[0], warp[1]))    # sort based on start cycle first, then run time
  csvfile.write('start_cycle' + '\t' + 'run_time' + '\t' + 'inst_committed' + '\n')
  for single_warp in warp:
    for item in single_warp: 
      csvfile.write(str(item))
      csvfile.write('\t')
    csvfile.write('\n')
  csvfile.close()

tracefile.close()
