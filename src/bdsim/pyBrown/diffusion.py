# pyBrown is a Brownian and Stokesian dynamics simulation tool
# Copyright (C) 2021  Tomasz Skora (tskora@ichf.edu.pl)
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
# along with this program.  If not, see https://www.gnu.org/licenses.

import ctypes
import numpy as np
import os.path	

from array import array
from ctypes.util import find_library

lib_name = "pyBrown"
lib_path = find_library(lib_name)
lib = ctypes.cdll.LoadLibrary( lib_path )

#-------------------------------------------------------------------------------

def RPY_M_matrix(beads, pointers):

	c_double = ctypes.c_double

	N = len(beads);

	a_list = [b.a for b in beads]
	v0 = array('d', a_list)
	a = (c_double * len(v0)).from_buffer(v0)

	p_list = np.ravel( -pointers[np.triu_indices(n=N, k=1)] ).tolist()
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	N_c = ctypes.c_int(N)

	len_my_list = 9*N*N

	my_list = [0.0]*len_my_list
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	lib.RPY_M_matrix(a, p, N_c, my_arr)

	M = np.reshape(my_arr, (3*N, 3*N))

	return M

#-------------------------------------------------------------------------------

def RPY_Smith_M_matrix(beads, pointers, box_length, alpha, m, n):

	c_double = ctypes.c_double

	N = len(beads);

	a_list = [b.a for b in beads]
	v0 = array('d', a_list)
	a = (c_double * len(v0)).from_buffer(v0)

	p_list = np.ravel( -pointers[np.triu_indices(n=N, k=1)] ).tolist()
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	box_length_c = ctypes.c_double(box_length)

	alpha_c = ctypes.c_double(alpha)

	m_c = ctypes.c_int(m)

	n_c = ctypes.c_int(n)

	N_c = ctypes.c_int(N)

	len_my_list = 9*N*N

	my_list = [0.0]*len_my_list
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	lib.RPY_Smith_M_matrix(a, p, box_length_c, alpha_c, m_c, n_c, N_c, my_arr)

	M = np.reshape(my_arr, (3*N, 3*N))

	return M

#-------------------------------------------------------------------------------

def JO_2B_R_matrix(beadi, beadj):

	c_double = ctypes.c_double

	ai = ctypes.c_double(beadi.a)

	aj = ctypes.c_double(beadj.a)

	rx = ctypes.c_double(beadj.r[0] - beadi.r[0])

	ry = ctypes.c_double(beadj.r[1] - beadi.r[1])

	rz = ctypes.c_double(beadj.r[2] - beadi.r[2])

	my_list = [0.0]*36
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	lib.JO_2B_R_matrix(ai, aj, rx, ry, rz, my_arr)

	my_arr = list(my_arr)

	R = np.zeros((6,6))

	R[0][0] = my_arr[0]
	R[1][1] = my_arr[1]
	R[2][2] = my_arr[2]
	R[0][1] = R[1][0] = my_arr[3]
	R[0][2] = R[2][0] = my_arr[4]
	R[1][2] = R[2][1] = my_arr[5]
	R[3][3] = my_arr[6]
	R[4][4] = my_arr[7]
	R[5][5] = my_arr[8]
	R[3][4] = R[4][3] = my_arr[9]
	R[3][5] = R[5][3] = my_arr[10]
	R[4][5] = R[5][4] = my_arr[11]
	R[3][0] = R[0][3] = my_arr[12]
	R[4][1] = R[1][4] = my_arr[13]
	R[5][2] = R[2][5] = my_arr[14]
	R[4][0] = R[0][4] = R[3][1] = R[1][3] = my_arr[15]
	R[5][0] = R[0][5] = R[3][2] = R[2][3] = my_arr[16]
	R[5][1] = R[1][5] = R[4][2] = R[2][4] = my_arr[17]

	return R

#-------------------------------------------------------------------------------

def JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff, cichocki_correction):

	c_double = ctypes.c_double

	N = len(beads)

	a_list = [b.a for b in beads]
	v0 = array('d', a_list)
	a = (c_double * len(v0)).from_buffer(v0)

	p_list = np.ravel( -pointers[np.triu_indices(n=N, k=1)] ).tolist()
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	N_c = ctypes.c_int(N)
	lubrication_cutoff_c = ctypes.c_double(lubrication_cutoff)

	len_my_list = 9*N*N

	my_list = [0.0]*len_my_list
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	if cichocki_correction: cichocki = ctypes.c_int(1)
	else: cichocki = ctypes.c_int(0)

	lib.JO_R_lubrication_correction_matrix(a, p, N_c, lubrication_cutoff_c, cichocki, my_arr)

	M = np.reshape(my_arr, (3*N, 3*N))

	return M

#-------------------------------------------------------------------------------