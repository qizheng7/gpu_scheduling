#!/usr/bin/python
### submit all jobs and the parse the results (all in local)
#################################
## ALL JOBS TO SUBMIT
#jobs_list = ["GFC", "NQU", "TSP", "bfs", "cfd", "bh", "dmr"]
jobs_list = ["dmr"]
gpgpu_path = "/home/caogao/gpu_scheduling/v3-cuda4.2/"
#################################

#######################
import os, sys, time
from subprocess import check_output, call, Popen
def ck_out(command):
  return check_output(command, executable='/bin/bash', shell=True)
def cmd(command):
  return call(command, executable='/bin/bash', shell=True)
def popen(command):
  return Popen(command, executable='/bin/bash', shell=True)
###############
try:
  output_postfix = '_' + sys.argv[1]
except:
  output_postfix = "" 
###############
rod = "rodinia_2.4/"
lone = "lonestargpu/"
jobs = {}
#### {job_name : [binary, parameters, outfile, process]}
jobs['GFC'] = ["GFC/GFC11", ""]
jobs['NQU'] = ["NQU/bin/release/NQU", ""] 
jobs['TSP'] = ["TSP/TSPGPU11", "TSP/Input/att48.tsp 50"]
jobs['bfs'] = [rod + "cuda/bfs/bfs", rod + "data/bfs/graph4096.txt"] 
jobs['cfd'] = [rod + "cuda/cfd/euler3d", rod + "data/bfs/cfd/fvcorr.domn.097K"]
jobs['bh'] = [lone + "v4.2/apps/bh/bh", "30000 1 0"]
jobs['dmr'] = [lone + "v4.2/apps/dmr/run", lone + "250k.2 20"]
###############
# submit all jobs
#os.system("source " + gpgpu_path + "setup_environment")
date = time.strftime("%m_%d_%H")  
cmd("source " + gpgpu_path + "setup_environment")
print "submit all jobs.."
for name, para in jobs.iteritems():
  if (name in jobs_list): 
    jobs[name].append(name + '_' + date + output_postfix + '.out')
    outfile = open(jobs[name][-1], 'w+')
    jobs[name].append(Popen('./' + para[0] + ' ' + para[1], stdout=outfile, stderr=outfile, shell=True)) 
    outfile.close()
print "wait for all jobs.."
for name, para in jobs.iteritems():
  if (name in jobs_list):
    para[3].wait()
    print name + " finished, parsing"
    Popen('./parse_profile.py ' + para[2], shell=True)
###############
try:
  cmd("rm _*")
except:
  pass
try:
  cmd("rm gpgpusim_power_report__*")
except:
  pass
