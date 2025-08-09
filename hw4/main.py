import os
os.environ["PYTHONPATH"] = "/data/coding/dlsys_hw/hw4/python"
os.environ["NEEDLE_BACKEND"] = "nd"
import sys
sys.path.append('/data/coding/dlsys_hw/hw4/python')


# import tests.hw4.test_nd_backend as test_nd_backend
# import needle as ndl

# test_nd_backend.test_logsumexp((5, 3), 0, ndl.cpu())


import tests.hw4.test_cifar_ptb_data as test_cifar_ptb_data

test_cifar_ptb_data.test_cifar10_dataset(True)