import numpy as np
import sys
import os

IS_TEST = (os.getenv("IS_TEST") == "True") if ("IS_TEST" in os.environ) else False
if IS_TEST:
    sys.path.insert(0, "/deployed/radiation/python")
else:
    sys.path.insert(0, "/work/radiation/python")
from radlw.radlw_main_gt4py import RadLWClass

me = 0
iovrlw = 1
isubclw = 2

for rank in range(6):
    rlw = RadLWClass(rank, iovrlw, isubclw)
    rlw.create_input_data(rank)
    rlw.lwrad(rank, do_subtest=True)
