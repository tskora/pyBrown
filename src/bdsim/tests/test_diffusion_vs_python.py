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

import math
import numpy as np
import os.path
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from scipy.special import erfc

from pyBrown.bead import Bead, compute_pointer_pbc_matrix
from pyBrown.diffusion import RPY_M_matrix, RPY_Smith_M_matrix, JO_2B_R_matrix, JO_R_lubrication_correction_matrix

#-------------------------------------------------------------------------------

def M_rpy_python(beads, pointers):

	M = [ [ None for j in range( len(beads) ) ] for i in range( len(beads) ) ]

	for i, bi in enumerate(beads):

		M[i][i] = Mii_rpy_python(bi.a)

		for j in range(i):

			bj = beads[j]

			M[i][j] = Mij_rpy_python(bi.a, bj.a, pointers[i][j])
			M[j][i] = np.transpose( M[i][j] )

	return np.block(M)

#-------------------------------------------------------------------------------

def M_rpy_smith_python(beads, pointers, box_length, alpha, m, n):

	M = [ [ None for j in range( len(beads) ) ] for i in range( len(beads) ) ]

	for i, bi in enumerate(beads):

		M[i][i] = Mii_rpy_smith_python(bi.a, box_length, alpha, m, n)

		for j in range(i):

			bj = beads[j]

			M[i][j] = Mij_rpy_smith_python(bi.a, bj.a, pointers[i][j], box_length, alpha, m, n)
			M[j][i] = np.transpose( M[i][j] )

	return np.block(M)

#-------------------------------------------------------------------------------

def R_lub_corr_python(beads, pointers, lubrication_cutoff, cichocki_correction):

	corr = [ [ np.zeros((3,3)) for j in range( len(beads) ) ] for i in range( len(beads) ) ]

	if cichocki_correction: q = 0.5 * np.block( [ [np.identity(3), -np.identity(3)], [-np.identity(3), np.identity(3)] ] )

	for i, bi in enumerate(beads):

		for j in range(i):

			bj = beads[j]

			dist = np.sqrt( np.sum( pointers[i][j]**2 ) )

			if ( dist - bi.a - bj.a ) / ( bi.a + bj.a ) > lubrication_cutoff:
				continue 

			R = R_jeffrey_python( bi.a, bj.a, pointers[i][j] )

			if cichocki_correction:

				nf2b = np.transpose(q) @ R @ q

				ff2b = np.transpose(q) @ np.linalg.inv( M_rpy_python( [bi, bj], np.array([[pointers[i][i], pointers[i][j]], [pointers[j][i], pointers[j][j]]]) ) ) @ q

			else:

				nf2b = R

				ff2b = np.linalg.inv( M_rpy_python( [bi, bj], np.array([[pointers[i][i], pointers[i][j]], [pointers[j][i], pointers[j][j]]]) ) )

			lub_corr = nf2b - ff2b

			corrii = lub_corr[0:3,0:3]
			corrjj = lub_corr[3:6,3:6]
			corrij = lub_corr[3:6,0:3]

			corr[i][i] += corrii
			corr[j][j] += corrjj
			corr[i][j] += corrij
			corr[j][i] += np.transpose( corrij )

	return np.block(corr)

#-------------------------------------------------------------------------------

def Mii_rpy_python(a):

	return np.identity(3) / ( 6 * np.pi * a )

#-------------------------------------------------------------------------------

def Mij_rpy_python(ai, aj, pointer):

	Rh_larger = max( ai, aj )
	Rh_smaller = min( ai, aj )

	dist2 = pointer[0]**2 + pointer[1]**2 + pointer[2]**2
	dist = math.sqrt( dist2 )
	outer = np.outer(pointer, pointer)/dist2

	aij2 = ai**2 + aj**2

	if dist > ( ai + aj ):
	
		coef_1 = 1.0 / ( 8 * np.pi * dist )
		coef_2 = 1.0 + aij2 / ( 3 * dist2 )
		coef_3 = 1.0 - aij2 / dist2
	
		answer = coef_2 * np.identity(3)
		answer += coef_3 * outer
		answer *= coef_1
	
		return answer
        
	elif dist <= ( Rh_larger - Rh_smaller ):

		return np.identity(3) / ( 6 * np.pi * Rh_larger )

	else:

		dist3 = dist * dist2

		coef_1 = 1.0 / ( 6 * np.pi * ai * aj )
		coef_2 = 16 * dist3 * ( ai + aj )
		coef_3 = (ai - aj)**2 + 3 * dist2
		coef_3 *= coef_3
		coef_4 = ( coef_2 - coef_3 ) / ( 32 * dist3 )
		coef_5 = 3 * ( (ai - aj)**2 - dist2 )**2
		coef_6 = coef_5 / ( 32 * dist3 )

		answer = coef_4 * np.identity(3)
		answer += coef_6 * outer
		answer *= coef_1
        
		return answer

#-------------------------------------------------------------------------------

def O_python(r):

	dist = math.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )

	return ( np.identity(3) + np.outer( r / dist, r / dist ) ) / dist

#-------------------------------------------------------------------------------

def Q_python(r):

	dist = math.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )

	return ( np.identity(3) - 3.0 * np.outer( r / dist, r / dist ) ) / dist**3

#-------------------------------------------------------------------------------

def Oii_pbc_smith_python(a, box_length, alpha, m, n ):

	ms = [ np.array([ mi,mj,mk ], float) for mi in range(-m, m+1) for mj in range(-m, m+1) for mk in range(-m, m+1) if (np.abs(mi)+np.abs(mj)+np.abs(mk)<=m) if not (mi == 0 and mj == 0 and mk == 0) ]

	ns = [ np.array([ ni,nj,nk ], float) for ni in range(-n, n+1) for nj in range(-n, n+1) for nk in range(-n, n+1) if (np.abs(ni)+np.abs(nj)+np.abs(nk)<=n) if not (ni == 0 and nj == 0 and nk == 0) ]

	answer = np.zeros((3, 3))

	for mvec in ms:

		mlength = math.sqrt( mvec[0]**2 + mvec[1]**2 + mvec[2]**2 )

		answer += erfc( alpha * mlength ) * O_python(mvec)

		answer += 2.0 * alpha / math.sqrt(np.pi) * math.exp( - alpha**2 * mlength**2 ) * np.outer(mvec/mlength, mvec/mlength)

	for nvec in ns:

		nlength = math.sqrt( nvec[0]**2 + nvec[1]**2 + nvec[2]**2 )

		mult = np.identity(3) - ( 1 + np.pi**2 * nlength**2 / alpha**2 ) * np.outer(nvec/nlength, nvec/nlength)

		answer += 2.0 / ( np.pi * nlength**2 ) * math.exp( - np.pi**2 * nlength**2 / alpha**2 ) * mult

	return answer - 3.0 * alpha * a / ( 2.0 * math.sqrt(np.pi) * box_length ) * np.identity(3)

#-------------------------------------------------------------------------------

def Qii_pbc_smith_python(a, box_length, alpha, m, n):

	ms = [ np.array([ mi,mj,mk ], float) for mi in range(-m, m+1) for mj in range(-m, m+1) for mk in range(-m, m+1) if (np.abs(mi)+np.abs(mj)+np.abs(mk)<=m) if not (mi == 0 and mj == 0 and mk == 0) ]

	ns = [ np.array([ ni,nj,nk ], float) for ni in range(-n, n+1) for nj in range(-n, n+1) for nk in range(-n, n+1) if (np.abs(ni)+np.abs(nj)+np.abs(nk)<=n) if not (ni == 0 and nj == 0 and nk == 0) ]

	answer = np.zeros((3, 3))

	for mvec in ms:

		mlength = math.sqrt( mvec[0]**2 + mvec[1]**2 + mvec[2]**2 )

		mult = erfc( alpha * mlength ) + 2.0 * alpha / math.sqrt(np.pi) * mlength * math.exp( -alpha**2 * mlength**2 )

		answer += mult * Q_python(mvec)

		answer -= 4.0 * alpha**3 / math.sqrt(np.pi) * math.exp(-alpha**2 * mlength**2) * np.outer(mvec/mlength, mvec/mlength)

	for nvec in ns:

		nlength = math.sqrt( nvec[0]**2 + nvec[1]**2 + nvec[2]**2 )

		answer += 4.0 * np.pi * math.exp( -np.pi**2 * nlength**2 / alpha**2) * np.outer(nvec/nlength, nvec/nlength)

	return answer - 1.0 / ( 3.0 * math.sqrt(np.pi) ) * (alpha * a / box_length)**3 * np.identity(3)

#-------------------------------------------------------------------------------

def Mii_rpy_smith_python(a, box_length, alpha, m, n):

	coef1 = 1.0 / ( 6 * np.pi * a )

	coef2 = 3.0 * a / ( 4.0 * box_length )

	coef3 = ( a / box_length )**3 / 2.0

	comp1 = np.identity(3)

	comp2 = Oii_pbc_smith_python( a, box_length, alpha, m, n )

	comp3 = Qii_pbc_smith_python( a, box_length, alpha, m, n )

	return coef1 * ( comp1 + coef2 * comp2 + coef3 * comp3 )

#-------------------------------------------------------------------------------

def Oij_pbc_smith_python(sigma, alpha, m, n):

	ms = [ np.array([ mi,mj,mk ], float) for mi in range(-m, m+1) for mj in range(-m, m+1) for mk in range(-m, m+1) if (np.abs(mi)+np.abs(mj)+np.abs(mk)<=m) ]

	ns = [ np.array([ ni,nj,nk ], float) for ni in range(-n, n+1) for nj in range(-n, n+1) for nk in range(-n, n+1) if (np.abs(ni)+np.abs(nj)+np.abs(nk)<=n) if not (ni == 0 and nj == 0 and nk == 0) ]

	answer = np.zeros((3, 3))

	for mvec in ms:

		msvec = mvec + sigma

		mslength = math.sqrt( msvec[0]**2 + msvec[1]**2 + msvec[2]**2 )

		answer += erfc( alpha * mslength ) * O_python( mvec + sigma )

		answer += 2.0 * alpha / math.sqrt(np.pi) * math.exp( - alpha**2 * mslength**2 ) * np.outer(mvec+sigma, mvec+sigma)/mslength**2

	for nvec in ns:

		nlength = math.sqrt( nvec[0]**2 + nvec[1]**2 + nvec[2]**2 )

		mult = 2.0 / ( np.pi * nlength**2 ) * math.exp( - np.pi**2 * nlength**2 / alpha**2 ) * np.exp( 2 * np.pi * 1j * np.dot(nvec, sigma)  )

		mult_real = mult.real

		answer += mult_real * ( np.identity(3) - (1 + np.pi**2 * nlength**2 / alpha**2 ) * np.outer(nvec / nlength, nvec / nlength) )

	return answer

#-------------------------------------------------------------------------------

def Qij_pbc_smith_python( sigma, alpha, m, n ):

	ms = [ np.array([ mi,mj,mk ], float) for mi in range(-m, m+1) for mj in range(-m, m+1) for mk in range(-m, m+1) if (np.abs(mi)+np.abs(mj)+np.abs(mk)<=m) ]

	ns = [ np.array([ ni,nj,nk ], float) for ni in range(-n, n+1) for nj in range(-n, n+1) for nk in range(-n, n+1) if (np.abs(ni)+np.abs(nj)+np.abs(nk)<=n) if not (ni == 0 and nj == 0 and nk == 0) ]

	answer = np.zeros((3, 3))

	for mvec in ms:

		msvec = mvec + sigma

		mslength = math.sqrt( msvec[0]**2 + msvec[1]**2 + msvec[2]**2 )

		mult = erfc( alpha * mslength ) + 2.0 * alpha / np.sqrt( np.pi ) * mslength * math.exp(- alpha**2 * mslength**2)

		answer += mult * Q_python(mvec + sigma)

		answer -= 4.0 * alpha**3 / np.sqrt(np.pi) * math.exp( -alpha**2 * mslength**2 ) * np.outer(mvec+sigma,mvec+sigma)/mslength**2

	for nvec in ns:

		nlength = math.sqrt( nvec[0]**2 + nvec[1]**2 + nvec[2]**2 )

		addi = 4.0 * np.pi * math.exp( -np.pi**2 * nlength**2 / alpha**2 ) * np.exp(2*np.pi*1j*np.dot(nvec,sigma)) * np.outer(nvec/nlength, nvec/nlength)

		addi_real = addi.real

		answer += addi_real

	return answer

#-------------------------------------------------------------------------------

def Mij_rpy_smith_python(ai, aj, pointer, box_length, alpha, m, n):

	sigma = np.array( pointer / box_length, float )

	coef1 = 1.0 / ( 6 * np.pi * ai )

	coef2 = 3.0 * ai / ( 4.0 * box_length )

	if ai == aj: coef3 = ( ai / box_length )**3 / 2.0

	else: coef3 = ai * ( ai**2 + aj**2 ) / box_length**3 / 4

	comp1 = Oij_pbc_smith_python( sigma, alpha, m, n )

	comp2 = Qij_pbc_smith_python( sigma, alpha, m, n )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	result = coef1 * ( coef2 * comp1 + coef3 * comp2 )

	if dist2 < ( ai + aj ) * ( ai + aj ):

		dist = math.sqrt(dist2)

		outer = np.outer(pointer, pointer)/dist2

		aij2 = ai**2 + aj**2

		coef_1 = 1.0 / ( 8 * np.pi * dist )
		coef_2 = 1.0 + aij2 / ( 3 * dist2 )
		coef_3 = 1.0 - aij2 / dist2

		result += Mij_rpy_python(ai, aj, pointer) - coef_1 * (coef_2 * np.identity(3) + coef_3 * outer)

	return result

#-------------------------------------------------------------------------------

def results_position(i, j, N):

	return i + j*N - j*(j+1)//2

#-------------------------------------------------------------------------------

def X_f_poly_python(l, rank):

	if rank == 0: return 1
	if rank == 1: return 3 * l
	if rank == 2: return 9 * l
	if rank == 3: return -4 * l + 27 * l**2 - 4 * l**3
	if rank == 4: return -24 * l + 81 * l**2 + 36 * l**3
	if rank == 5: return 72 * l**2 + 243 * l**3 + 72 * l**4
	if rank == 6: return 16 * l + 108 * l**2 + 281 * l**3 + 648 * l**4 + 144 * l**5
	if rank == 7: return 288 * l**2 + 1620 * l**3 + 1515 * l**4 + 1620 * l**5 + 288 * l**6
	if rank == 8: return 576 * l**2 + 4848 * l**3 + 5409 * l**4 + 4524 * l**5 + 3888 * l**6 + 576 * l**7
	if rank == 9: return 1152 * l**2 + 9072 * l**3 + 14752 * l**4 + 26163 * l**5 + 14752 * l**6 + 9072 * l**7 + 1152 * l**8
	if rank == 10: return 2304 * l**2 + 20736 * l**3 + 42804 * l**4 + 115849 * l**5 + 76176 * l**6 + 39264 * l**7 + 20736 * l**8 + 2304 * l**9
	if rank == 11: return 4608 * l**2 + 46656 * l**3 + 108912 * l**4 + 269100 * l**5 + 319899 * l**6 + 269100 * l**7 + 108912 * l**8 + 46656 * l**9 + 4608 * l**10
	else: return None

#-------------------------------------------------------------------------------

def Y_f_poly_python(l, rank):

	if rank == 0: return 1
	if rank == 1: return 3 / 2 * l
	if rank == 2: return 9 / 4 * l
	if rank == 3: return 2 * l + 27 / 8 * l**2 + 2 * l**3
	if rank == 4: return 6 * l + 81 / 16 * l**2 + 18 * l**3
	if rank == 5: return 63 / 2 * l**2 + 243 / 32 * l**3 + 63 / 2 * l**4
	if rank == 6: return 4 * l + 54 * l**2 + 1241 / 64 * l**3 + 81 * l**4 + 72 * l**5
	if rank == 7: return 144 * l**2 + 1053 / 8 * l**3 + 19083 / 128 * l**4 + 1053 / 8 * l**5 + 144 * l**6
	if rank == 8: return 279 * l**2 + 4261 / 8 * l**3 + 126369 / 256 * l**4 - 117 / 8 * l**5 + 648 * l**6 + 288 * l**7
	if rank == 9: return 576 * l**2 + 1134 * l**3 + 60443 / 32 * l**4 + 766179 / 512 * l**5 + 60443 / 32 * l**6 + 1134 * l**7 + 576 * l**8
	if rank == 10: return 1152 * l**2 + 7857 / 4 * l**3 + 98487 / 16 * l**4 + 10548393 / 1024 * l**5 + 67617 / 8 * l**6 - 351 / 2 * l**7 + 3888 * l**8 + 1152 * l**9
	if rank == 11: return 2304 * l**2 + 7128 * l**3 + 22071 / 2 * l**4 + 2744505 / 128 * l**5 + 95203835 / 2048 * l**6 + 2744505 / 128 * l**7 + 22071 / 2 * l**8 + 7128 * l**9 + 2304 * l**10
	else: return None

#-------------------------------------------------------------------------------

def X_g_poly_python(l, rank):

	if rank == 1: return 2 * l**2 * ( 1 + l )**(-3)
	if rank == 2: return 1 / 5 * l * ( 1 + 7 * l + l**2 ) * ( 1 + l )**(-3)
	if rank == 3: return 1 / 42 * ( 1 + 18 * l - 29 * l**2 + 18 * l**3 + l**4 ) * ( 1 + l )**(-3)
	else: return None

#-------------------------------------------------------------------------------

def Y_g_poly_python(l, rank):

	if rank == 2: return 4 / 15 * l * ( 2 + l + 2 * l**2 ) * ( 1 + l )**(-3)
	if rank == 3: return 2 / 375 * ( 16 - 45 * l + 58 * l**2 - 45 * l**3 + 16 * l**4 ) * ( 1 + l )**(-3)
	else: return None

#-------------------------------------------------------------------------------

def XA11_python(s, l):

	answer = 0.0

	answer += X_g_poly_python(l, 1) * ( 1 - 4 * s**(-2) )**(-1)

	answer -= X_g_poly_python(l, 2) * np.log( 1 - 4 * s**(-2) )

	answer -= X_g_poly_python(l, 3) * ( 1 - 4 * s**(-2) ) * np.log( 1 - 4 * s**(-2) )

	answer += X_f_poly_python(l, 0) - X_g_poly_python(l, 1)

	for m in [ mi for mi in range(1, 12) if ( mi%2 == 0 ) ]:

		if m == 2: m1 = -2
		else: m1 = m - 2

		mult = ( 2 / s )**m

		answer += mult * ( 2**(-m) * ( 1 + l )**(-m) * X_f_poly_python(l, m) - X_g_poly_python(l, 1) )

		answer += mult * ( 4 * m**(-1) * m1**(-1) * X_g_poly_python(l, 3) - 2 * m**(-1) * X_g_poly_python(l, 2) )

	return answer

#-------------------------------------------------------------------------------

def YA11_python(s, l):

	answer = 0.0

	answer -= Y_g_poly_python(l, 2) * np.log( 1 - 4 * s**(-2) )

	answer -= Y_g_poly_python(l, 3) * ( 1 - 4 * s**(-2) ) * np.log( 1 - 4 * s**(-2) )

	answer += Y_f_poly_python(l, 0)

	for m in [ mi for mi in range(1, 12) if ( mi%2 == 0 ) ]:

		if m == 2: m1 = -2
		else: m1 = m - 2

		mult = ( 2 / s )**m

		answer += mult * ( 2**(-m) * ( 1 + l )**(-m) * Y_f_poly_python(l, m) - 2 * m**(-1) * Y_g_poly_python(l, 2) )

		answer += mult * 4 * m**(-1) * m1**(-1) * Y_g_poly_python(l, 3)

	return answer

#-------------------------------------------------------------------------------

def XA12_python(s, l):

	answer = 0.0

	answer += 2 * s**(-1) * X_g_poly_python(l, 1) * ( 1 - 4 * s**(-2) )**(-1)

	answer += X_g_poly_python(l, 2) * np.log( ( s + 2 ) / ( s - 2 ) )

	answer += X_g_poly_python(l, 3) * ( 1 - 4 * s**(-2) ) * np.log( ( s + 2 ) / ( s - 2 ) ) + 4 * X_g_poly_python(l, 3) * s**(-1)

	for m in [ mi for mi in range(1, 12) if ( mi%2 == 1 ) ]:

		if m == 2: m1 = -2
		else: m1 = m - 2

		mult = ( 2 / s )**m

		answer += mult * ( 2**(-m) * ( 1 + l )**(-m) * X_f_poly_python(l, m) - X_g_poly_python(l, 1) )

		answer += mult * ( 4 * m**(-1) * m1**(-1) * X_g_poly_python(l, 3) - 2 * m**(-1) * X_g_poly_python(l, 2) )

	divisor = -1 / 2 * ( 1 + l )

	return answer / divisor

#-------------------------------------------------------------------------------

def YA12_python(s, l):

	answer = 0.0

	answer += Y_g_poly_python(l, 2) * np.log( ( s + 2 ) / ( s - 2 ) )

	answer += Y_g_poly_python(l, 3) * ( 1 - 4 * s**(-2) ) * np.log( ( s + 2 ) / ( s - 2 ) )

	answer += 4 * Y_g_poly_python(l, 3) * s**(-1)

	for m in [ mi for mi in range(1, 12) if ( mi%2 == 1 ) ]:

		if m == 2: m1 = -2
		else: m1 = m - 2

		mult = ( 2 / s )**m

		answer += mult * ( 2**(-m) * ( 1 + l )**(-m) * Y_f_poly_python(l, m) - 2 * m**(-1) * Y_g_poly_python(l, 2) )

		answer += mult * ( 4 * m**(-1) * m1**(-1) * Y_g_poly_python(l, 3) )

	divisor = -1 / 2 * ( 1 + l )

	return answer / divisor

#-------------------------------------------------------------------------------

def R_jeffrey_python(ai, aj, pointer):

	dist = math.sqrt( pointer[0]**2 + pointer[1]**2 + pointer[2]**2 )

	s = 2 * dist / ( ai + aj )

	l = aj / ai

	R = [ [ None , None ], [ None, None ] ]

	const = 6 * np.pi * ai

	R[0][0] = XA11_python(s, l) * np.outer(pointer/dist, pointer/dist) + YA11_python(s, l) * ( np.identity(3) - np.outer(pointer/dist, pointer/dist) )

	R[1][1] = l * ( XA11_python(s, 1/l) * np.outer(pointer/dist, pointer/dist) + YA11_python(s, 1/l) * ( np.identity(3) - np.outer(pointer/dist, pointer/dist) ) )

	R[0][1] = 0.5 * ( l + 1.0 ) * ( XA12_python(s, l) * np.outer(pointer/dist, pointer/dist) + YA12_python(s, l) * ( np.identity(3) - np.outer(pointer/dist, pointer/dist) ) )

	R[1][0] = 0.5 * ( l + 1.0 ) * ( XA12_python(s, 1/l) * np.outer(pointer/dist, pointer/dist) + YA12_python(s, 1/l) * ( np.identity(3) - np.outer(pointer/dist, pointer/dist) ) )

	return const * np.block(R)

#-------------------------------------------------------------------------------

class TestDiffusionVsPython(unittest.TestCase):

	def test_M_rpy_smith(self):

		N_beads = 100

		box_length = 20.0

		alpha = np.sqrt(np.pi)

		m = 3

		n = 3

		beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(N_beads) ]

		pointers = compute_pointer_pbc_matrix(beads, box_length)

		c_ish = RPY_Smith_M_matrix(beads, pointers, box_length, alpha, m, n)

		python_ish = M_rpy_smith_python(beads, pointers, box_length, alpha, m, n)

		for i in range(3*N_beads):
			for j in range(3*N_beads):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

	#---------------------------------------------------------------------------

	def test_M_rpy(self):

		N_beads = 100

		box_length = 20.0

		beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(N_beads) ]

		pointers = compute_pointer_pbc_matrix(beads, box_length)

		c_ish = RPY_M_matrix(beads, pointers)

		python_ish = M_rpy_python(beads, pointers)

		for i in range(3*N_beads):
			for j in range(3*N_beads):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

	#---------------------------------------------------------------------------

	def test_R_2B(self):

		np.random.seed(0)

		beads = [ Bead([x, y, z], np.random.choice([1.0, 0.5])) for x in [0,3] for y in [0,3] for z in [0,3] ]

		for i in range( len(beads)-1 ):

			for j in range( i+1, len(beads) ):

				c_ish = JO_2B_R_matrix(beads[i], beads[j])

				python_ish = R_jeffrey_python(beads[i].a, beads[j].a, beads[j].r - beads[i].r)

				for ii in range(6):
					for jj in range(6):
						self.assertAlmostEqual(c_ish[ii][jj], python_ish[ii][jj], places = 7)

	#---------------------------------------------------------------------------

	def test_R_lub_corr_large_cutoff_cichocki(self):

		box_length = 20.0

		lubrication_cutoff = 10.0

		beads = [ Bead([x, y, z], 1.0) for x in [0,3,6] for y in [0,3,6] for z in [0,3,6] ]

		pointers = compute_pointer_pbc_matrix(beads, box_length)

		c_ish = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff, cichocki_correction = True)

		python_ish = R_lub_corr_python(beads, pointers, lubrication_cutoff, cichocki_correction = True)

		for i in range(3*len(beads)):
			for j in range(3*len(beads)):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

	#---------------------------------------------------------------------------

	def test_R_lub_corr_large_cutoff_no_cichocki(self):

		box_length = 20.0

		lubrication_cutoff = 10.0

		beads = [ Bead([x, y, z], 1.0) for x in [0,3,6] for y in [0,3,6] for z in [0,3,6] ]

		pointers = compute_pointer_pbc_matrix(beads, box_length)

		c_ish = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff, cichocki_correction = False)

		python_ish = R_lub_corr_python(beads, pointers, lubrication_cutoff, cichocki_correction = False)

		for i in range(3*len(beads)):
			for j in range(3*len(beads)):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

	#---------------------------------------------------------------------------

	def test_R_lub_corr_small_cutoff_cichocki(self):

		box_length = 20.0

		lubrication_cutoff = 1.0

		beads = [ Bead([x, y, z], 1.0) for x in [0,3,6] for y in [0,3,6] for z in [0,3,6] ]

		pointers = compute_pointer_pbc_matrix(beads, box_length)

		c_ish = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff, cichocki_correction = True)

		python_ish = R_lub_corr_python(beads, pointers, lubrication_cutoff, cichocki_correction = True)

		for i in range(3*len(beads)):
			for j in range(3*len(beads)):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

	#---------------------------------------------------------------------------

	def test_R_lub_corr_small_cutoff_no_cichocki(self):

		box_length = 20.0

		lubrication_cutoff = 1.0

		beads = [ Bead([x, y, z], 1.0) for x in [0,3,6] for y in [0,3,6] for z in [0,3,6] ]

		pointers = compute_pointer_pbc_matrix(beads, box_length)

		c_ish = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff, cichocki_correction = False)

		python_ish = R_lub_corr_python(beads, pointers, lubrication_cutoff, cichocki_correction = False)

		for i in range(3*len(beads)):
			for j in range(3*len(beads)):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------