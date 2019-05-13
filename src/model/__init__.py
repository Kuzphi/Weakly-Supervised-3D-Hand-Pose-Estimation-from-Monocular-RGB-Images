# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
for name in os.listdir(os.path.dirname(__file__)):
    if name[0] == '.' or name == '__init__.py' or name[-3:] != '.py':
        continue
    module = __import__(name[:-3], locals(), globals(), ['*'])
    if not hasattr(module, '__all__'):
    	print (module)
    	continue
    for key in module.__all__:
    	locals()[key] = getattr(module, key)
