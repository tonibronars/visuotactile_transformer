import glob
import numpy as np
import os 

aa = glob.glob('local_shape_1_*.png')
bb = glob.glob('local_shape_35_*.png')

remove_aa = aa[160:]
remove_bb = bb[160:]

print('removing 1')
for s in remove_aa:
    os.rename(s, 'remove/' + s)
    os.rename(s.replace('local_shape', 'depth').replace('png','npy'), 'remove/' + s.replace('local_shape', 'depth').replace('png','npy'))
    os.rename(s.replace('local_shape', 'transformation').replace('png','npy'), 'remove/' + s.replace('local_shape', 'transformation').replace('png','npy'))

print('removing 35')
for s in remove_bb:
    os.rename(s, 'remove/' + s)
    os.rename(s.replace('local_shape', 'depth').replace('png','npy'), 'remove/' + s.replace('local_shape', 'depth').replace('png','npy'))
    os.rename(s.replace('local_shape', 'transformation').replace('png','npy'), 'remove/' + s.replace('local_shape', 'transformation').replace('png','npy'))

