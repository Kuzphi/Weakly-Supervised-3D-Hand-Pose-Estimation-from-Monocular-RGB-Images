# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_path, '3rdparty'))

from progress.bar import Bar as Bar