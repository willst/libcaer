from __future__ import print_function

import sys
if sys.version_info >= (3, 0):
    sys.stdout.write("pybind11 compiled to only work with Python2.x currently.\n")
    sys.exit(1)

import optic_flow
import numpy as np
import time

if __name__ == "__main__":

    ltime = time.time()
    of = optic_flow.DVSOpticFlow()

    while of.processFrame():
        c = of.getCount()
        x = of.getVx()
        y = of.getVy()

        cnt = int(np.sum(c))

        print("Events on frame is %i" % cnt)

        avg_x = float(np.sum(x) / cnt)
        avg_y = float(np.sum(y) / cnt)

        print("Avg velocity %f, %f" % (avg_x, avg_y))
        ntime = time.time()
        print('Process rate is %f' % (1.0 / (ntime - ltime)))
        ltime = ntime

