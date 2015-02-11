#!/usr/bin/python
## GPGPU irregular application project: parsing script

import os, sys, re

try:
  filename = sys.argv[1]
  if filename[-6:] == '.trace':
    filename = filename[:-6]
except:
  print 'Usage: trace.py filename'
  sys.exit()

tracefile = open(filename, 'r')
csvfile = open(filename + '.csv', 'w+')
csvfile.write('run_time' + '\t' + 'inst_committed' + '\n')
#run_time_list = []
#inst_committed_list = []

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
#      warp_id = line[4]
#      dynamic_warp_id = line[6]
#      start_cycle = line[8]
#      finish_cycle = line[10]
      run_time = line[12]
      csvfile.write(run_time + '\t')
      inst_committed = line[14]
      csvfile.write(inst_committed + '\t')
      csvfile.write('\n')
#      run_time_list.append(run_time)
#      inst_committed_list.append(inst_committed)
  line = tracefile.readline()

tracefile.close()
csvfile.close()
