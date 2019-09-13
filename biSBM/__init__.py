#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# det_k_bisbm -- a python module for partitioning bipartite networks using the bipartite stochastic block model
#
# Copyright (C) 2016-2019 Tzu-Chi Yen <tzuchi@netscied.tw>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""``det_k_bisbm`` - Statistical inference of the bipartite stochastic block model
----------------------------------------------------------------------------------

This module contains algorithms for the identification of large-scale network
structure via the statistical inference of the bipartite stochastic block model.

.. note::

   TODO.

"""
from biSBM.optimalks import *
from biSBM.ioutils import *
import engines

__package__ = 'bisbm'
__title__ = 'biSBM: a python package for partitioning bipartite networks using the bipartite stochastic block model'
__description__ = ''
__copyright__ = 'Copyright 2016-2019 Tzu-Chi Yen'
__author__ = """\n""".join([
    'Tzu-Chi Yen <tzuchi.yen@colorado.edu>',
])
__URL__ = "https://docs.netscied.tw/bipartiteSBM/index.html"
__version__ = '0.90.0'
__release__ = '0.90.0'

__all__ = ["OptimalKs", "engines", "__author__", "__URL__", "__version__", "__copyright__"]
