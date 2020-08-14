# -*- coding: UTF-8 -*-
import os
SEEDS = [94196879,76466153,8269641,96259793,59870115,19470948,51319742,38792324,23867758,91518904]
for seed in SEEDS:
    print(f'Trainning ATAE with seed {seed}')
    os.system('time python  main.py --seed '+ str(seed))
    # os.system('killall python & killall python2 & killall python3')
