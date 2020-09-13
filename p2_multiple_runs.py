# -*- coding: UTF-8 -*-
import os
import time
start_time = time.time()
# SEEDS = [94196879,76466153,8269641,96259793,59870115,19470948,51319742,38792324,23867758,91518904]
epochs = 50

# for seed in SEEDS:
#     p = os.path.join('results',str(seed))

#     if not os.path.exists(p):
#         os.makedirs(p)

# for seed in SEEDS:
#     print 'Trainning ATAE with seed ' + str(seed)
#     start_it = time.time()
#     print 'python2  main.py --seed '+ str(seed)
#     os.system('python2  main.py --seed '+ str(seed) + ' --epoch ' + str(epochs))
#     print 'Excuted with ' + str(seed) + ' in ' + str(time.time() - start_it) +  ' s'

# print 'executed all seeds in ' + str(time.time() - start_time) + ' s' 

n_folds = 10
for fold in range(n_folds):
    p = os.path.join('results','fold_'+str(fold))

    if not os.path.exists(p):
        os.makedirs(p)

for fold in range(n_folds):
    print 'Trainning ATAE with fold ' + str(fold)
    start_it = time.time()
    print 'python2  main.py --fold '+ str(fold) + ' --epoch ' + str(epochs) + ' --dataset ' + 'fold_'+str(fold)
    os.system('python2  main.py --fold '+ str(fold) + ' --epoch ' + str(epochs) + ' --dataset ' + 'data/fold_'+str(fold))
    print 'Excuted with fold_' + str(fold) + ' in ' + str(time.time() - start_it) +  ' s'

print 'executed all seeds in ' + str(time.time() - start_time) + ' s' 
