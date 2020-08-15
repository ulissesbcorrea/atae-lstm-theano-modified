
# -*- coding: UTF-8 -*-
import os
import time
start_time = time.time()
SEEDS = [94196879,76466153,8269641,96259793,59870115,19470948,51319742,38792324,23867758,91518904]
for seed in SEEDS:
    os.makedirs(os.path.join('results',str(seed)),exist_ok = True)
for seed in SEEDS:
    print 'Trainning ATAE with seed ' + str(seed)
    start_it = time.time()
    print 'python2  main.py --seed '+ str(seed)
    os.system('python2  main.py --seed '+ str(seed))
    # os.system('killall python & killall python2 & killall python3')
    print 'Excuted with ' + str(seed) + ' in ' + str(time.time() - start_it) +  ' s'

print 'executed all seeds in ' + str(time.time() - start_time) + ' s' 
