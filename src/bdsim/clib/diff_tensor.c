// pyBrown is a Brownian and Stokesian dynamics simulation tool
// Copyright (C) 2021  Tomasz Skora (tskora@ichf.edu.pl)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see https://www.gnu.org/licenses.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// LU decomoposition of a general matrix
void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

static inline double RPY_Mii_tt_block(double a);

static inline double RPY_Mii_rr_block(double a);

static void RPY_Mij_tt_block(double ai, double aj, double rx, double ry, double rz, double* matrix);

static void RPY_Mij_rr_block(double ai, double aj, double rx, double ry, double rz, double* matrix);

static void RPY_Mij_rt_block(double ai, double aj, double rx, double ry, double rz, double* matrix);

static void RPY_Smith_Mii_block(double a, double box_length, double alpha, int m_max, int n_max, double* matrix);

static void RPY_Smith_Mij_block(double ai, double aj, double rx, double ry, double rz, double box_length, double alpha, int m_max, int n_max, double* matrix);

static void RPY_Smith_Oii_block(double a, double box_length, double alpha, int m_max, int n_max, double* matrix);

static void RPY_Smith_Qii_block(double a, double box_length, double alpha, int m_max, int n_max, double* matrix);

static void RPY_Smith_Oij_block(double sigmax, double sigmay, double sigmaz, double alpha, int m_max, int n_max, double* matrix);

static void RPY_Smith_Qij_block(double sigmax, double sigmay, double sigmaz, double alpha, int m_max, int n_max, double* matrix);

static void RPY_O_function(double rx, double ry, double rz, double* matrix);

static void RPY_Q_function(double rx, double ry, double rz, double* matrix);

static void RPY_2B_RA_matrix(double ai, double aj, double rx, double ry, double rz, double* R_matrix);

static void Cichocki_2B_R_correction(double* temp_2b, double* result_2b);

static double JO_XA11_term(double s, double l);

static double JO_YA11_term(double s, double l);

static double JO_XA12_term(double s, double l);

static double JO_YA12_term(double s, double l);

static double JO_YB11_term(double s, double l);

static double JO_YB12_term(double s, double l);

static double JO_XC11_term(double s, double l);

static double JO_YC11_term(double s, double l);

static double JO_XC12_term(double s, double l);

static double JO_YC12_term(double s, double l);

static double JO_XAf_polynomial(double l, int degree);

static double JO_YAf_polynomial(double l, int degree);

static double JO_XAg_polynomial(double l, int degree);

static double JO_YAg_polynomial(double l, int degree);

static double JO_YBf_polynomial(double l, int degree);

static double JO_YBg_polynomial(double l, int degree);

static double JO_XCf_polynomial(double l, int degree);

static double JO_YCf_polynomial(double l, int degree);

static double JO_YCg_polynomial(double l, int degree);

static void inverse(double* matrix, int matrix_dimension);

static inline int results_position(int i, int j, int N);

// -------------------------------------------------------------------------------

void RPY_M_matrix(double* as, double* pointers, int number_of_beads, double* M_matrix)
{
	register int i = 0;
	register int j = 0;

	double* Mij_tt_values;
	double* Mij_rr_values;
	double* Mij_rt_values;

	double* results = calloc(9*(number_of_beads*number_of_beads+number_of_beads), sizeof(double));

	int shift_results;

	int shift_pointers;

	double rx, ry, rz, diag_tt, diag_rr;

	int r, I, I1, I2, I3, I4, I5, J, J1, J2, J3, J4, J5;

	int N = 6*number_of_beads;

	int tr_shift = 3*(number_of_beads*number_of_beads+number_of_beads);

	for (j = 0; j < number_of_beads; j++)
	{
		diag_tt = RPY_Mii_tt_block(*(as+j));
		diag_rr = RPY_Mii_rr_block(*(as+j));

		shift_results = 6*results_position(j,j,number_of_beads);

		*(results + shift_results) = diag_tt;
		*(results + shift_results + 1) = diag_tt;
		*(results + shift_results + 2) = diag_tt;

		*(results + shift_results + tr_shift) = diag_rr;
		*(results + shift_results + tr_shift + 1) = diag_rr;
		*(results + shift_results + tr_shift + 2) = diag_rr;

		for (i = j + 1; i < number_of_beads; i++)
		{
			Mij_tt_values = calloc(6, sizeof(double));
			Mij_rr_values = calloc(6, sizeof(double));
			Mij_rt_values = calloc(6, sizeof(double));

			shift_results = 6*results_position(i,j,number_of_beads);
			shift_pointers = 3*results_position(i-1,j,number_of_beads-1);

			rx = *(pointers + shift_pointers);
			ry = *(pointers + shift_pointers + 1);
			rz = *(pointers + shift_pointers + 2);

			RPY_Mij_tt_block(*(as+i), *(as+j), rx, ry, rz, Mij_tt_values);

			*(results + shift_results) = *Mij_tt_values;
			*(results + shift_results + 1) = *(Mij_tt_values + 1);
			*(results + shift_results + 2) = *(Mij_tt_values + 2);
			*(results + shift_results + 3) = *(Mij_tt_values + 3);
			*(results + shift_results + 4) = *(Mij_tt_values + 4);
			*(results + shift_results + 5) = *(Mij_tt_values + 5);

			RPY_Mij_rr_block(*(as+i), *(as+j), rx, ry, rz, Mij_rr_values);

			*(results + shift_results + tr_shift) = *Mij_rr_values;
			*(results + shift_results + tr_shift + 1) = *(Mij_rr_values + 1);
			*(results + shift_results + tr_shift + 2) = *(Mij_rr_values + 2);
			*(results + shift_results + tr_shift + 3) = *(Mij_rr_values + 3);
			*(results + shift_results + tr_shift + 4) = *(Mij_rr_values + 4);
			*(results + shift_results + tr_shift + 5) = *(Mij_rr_values + 5);

			RPY_Mij_rt_block(*(as+i), *(as+j), rx, ry, rz, Mij_rt_values);

			*(results + shift_results + 2*tr_shift) = *Mij_rt_values;
			*(results + shift_results + 2*tr_shift + 1) = *(Mij_rt_values + 1);
			*(results + shift_results + 2*tr_shift + 2) = *(Mij_rt_values + 2);
			*(results + shift_results + 2*tr_shift + 3) = *(Mij_rt_values + 3);
			*(results + shift_results + 2*tr_shift + 4) = *(Mij_rt_values + 4);
			*(results + shift_results + 2*tr_shift + 5) = *(Mij_rt_values + 5);

			free(Mij_tt_values);
			free(Mij_rr_values);
			free(Mij_rt_values);
		}
	}

	for (j = 0; j < number_of_beads; j++)
	{
		r = 6*results_position(j, j, number_of_beads);
		J = 3*j;
		J1 = J + 1;
		J2 = J + 2;
		J3 = J + 3*number_of_beads;
		J4 = J1 + 3*number_of_beads;
		J5 = J2 + 3*number_of_beads;

		M_matrix[J + J*N] = results[r];
		M_matrix[J1 + J1*N] = results[r + 1];
		M_matrix[J2 + J2*N] = results[r + 2];
		M_matrix[J3 + J3*N] = results[r + tr_shift];
		M_matrix[J4 + J4*N] = results[r + tr_shift + 1];
		M_matrix[J5 + J5*N] = results[r + tr_shift + 2];

		for (i = j+1; i < number_of_beads; i++)
		{
			r = 6*results_position(i, j, number_of_beads);
			I = 3*i;
			I1 = I + 1;
			I2 = I + 2;
			I3 = I + 3*number_of_beads;
			I4 = I1 + 3*number_of_beads;
			I5 = I2 + 3*number_of_beads;

			M_matrix[I + J*N] = results[r];
			M_matrix[J + I*N] = results[r];
			M_matrix[I1 + J1*N] = results[r + 1];
			M_matrix[J1 + I1*N] = results[r + 1];
			M_matrix[I2 + J2*N] = results[r + 2];
			M_matrix[J2 + I2*N] = results[r + 2];
			M_matrix[I1 + J*N] = results[r + 3];
			M_matrix[I + J1*N] = results[r + 3];
			M_matrix[J + I1*N] = results[r + 3];
			M_matrix[J1 + I*N] = results[r + 3];
			M_matrix[I2 + J*N] = results[r + 4];
			M_matrix[I + J2*N] = results[r + 4];
			M_matrix[J + I2*N] = results[r + 4];
			M_matrix[J2 + I*N] = results[r + 4];
			M_matrix[I2 + J1*N] = results[r + 5];
			M_matrix[I1 + J2*N] = results[r + 5];
			M_matrix[J1 + I2*N] = results[r + 5];
			M_matrix[J2 + I1*N] = results[r + 5];

			M_matrix[I3 + J3*N] = results[r + tr_shift];
			M_matrix[J3 + I3*N] = results[r + tr_shift];
			M_matrix[I4 + J4*N] = results[r + tr_shift + 1];
			M_matrix[J4 + I4*N] = results[r + tr_shift + 1];
			M_matrix[I5 + J5*N] = results[r + tr_shift + 2];
			M_matrix[J5 + I5*N] = results[r + tr_shift + 2];
			M_matrix[I4 + J3*N] = results[r + tr_shift + 3];
			M_matrix[I3 + J4*N] = results[r + tr_shift + 3];
			M_matrix[J3 + I4*N] = results[r + tr_shift + 3];
			M_matrix[J4 + I3*N] = results[r + tr_shift + 3];
			M_matrix[I5 + J3*N] = results[r + tr_shift + 4];
			M_matrix[I3 + J5*N] = results[r + tr_shift + 4];
			M_matrix[J3 + I5*N] = results[r + tr_shift + 4];
			M_matrix[J5 + I3*N] = results[r + tr_shift + 4];
			M_matrix[I5 + J4*N] = results[r + tr_shift + 5];
			M_matrix[I4 + J5*N] = results[r + tr_shift + 5];
			M_matrix[J4 + I5*N] = results[r + tr_shift + 5];
			M_matrix[J5 + I4*N] = results[r + tr_shift + 5];

			// M_matrix[I3 + J*N] = results[r + 2*tr_shift];
			// M_matrix[J + I3*N] = results[r + 2*tr_shift];
			// M_matrix[I4 + J1*N] = results[r + 2*tr_shift + 1];
			// M_matrix[J1 + I4*N] = results[r + 2*tr_shift + 1];
			// M_matrix[I5 + J2*N] = results[r + 2*tr_shift + 2];
			// M_matrix[J2 + I5*N] = results[r + 2*tr_shift + 2];
			M_matrix[I4 + J*N] = -results[r + 2*tr_shift + 3];
			M_matrix[J1 + I3*N] = -results[r + 2*tr_shift + 3];
			M_matrix[I3 + J1*N] = results[r + 2*tr_shift + 3];
			M_matrix[J + I4*N] = results[r + 2*tr_shift + 3];
			M_matrix[I5 + J*N] = -results[r + 2*tr_shift + 4];
			M_matrix[J2 + I3*N] = -results[r + 2*tr_shift + 4];
			M_matrix[I3 + J2*N] = results[r + 2*tr_shift + 4];
			M_matrix[J + I5*N] = results[r + 2*tr_shift + 4];
			M_matrix[I5 + J1*N] = -results[r + 2*tr_shift + 5];
			M_matrix[J2 + I4*N] = -results[r + 2*tr_shift + 5];
			M_matrix[I4 + J2*N] = results[r + 2*tr_shift + 5];
			M_matrix[J1 + I5*N] = results[r + 2*tr_shift + 5];

		}
	}

	free(results);
}

// -------------------------------------------------------------------------------

void RPY_M_tt_matrix(double* as, double* pointers, int number_of_beads, double* M_matrix)
{
	register int i = 0;
	register int j = 0;

	double* Mij_values;

	double* results = calloc(3*(number_of_beads*number_of_beads+number_of_beads), sizeof(double));

	int shift_results;

	int shift_pointers;

	double rx, ry, rz, diag;

	int r, I, I1, I2, J, J1, J2;

	int N = 3*number_of_beads;

	for (j = 0; j < number_of_beads; j++)
	{
		diag = RPY_Mii_tt_block(*(as+j));

		shift_results = 6*results_position(j,j,number_of_beads);

		*(results + shift_results) = diag;
		*(results + shift_results + 1) = diag;
		*(results + shift_results + 2) = diag;

		for (i = j + 1; i < number_of_beads; i++)
		{
			Mij_values = calloc(6, sizeof(double));

			shift_results = 6*results_position(i,j,number_of_beads);
			shift_pointers = 3*results_position(i-1,j,number_of_beads-1);

			rx = *(pointers + shift_pointers);
			ry = *(pointers + shift_pointers + 1);
			rz = *(pointers + shift_pointers + 2);

			RPY_Mij_tt_block(*(as+i), *(as+j), rx, ry, rz, Mij_values);

			*(results + shift_results) = *Mij_values;
			*(results + shift_results + 1) = *(Mij_values + 1);
			*(results + shift_results + 2) = *(Mij_values + 2);
			*(results + shift_results + 3) = *(Mij_values + 3);
			*(results + shift_results + 4) = *(Mij_values + 4);
			*(results + shift_results + 5) = *(Mij_values + 5);

			free(Mij_values);
		}
	}

	for (j = 0; j < number_of_beads; j++)
	{
		r = 6*results_position(j, j, number_of_beads);
		J = 3*j;
		J1 = J + 1;
		J2 = J + 2;

		M_matrix[J + J*N] = results[r];
		M_matrix[J1 + J1*N] = results[r + 1];
		M_matrix[J2 + J2*N] = results[r + 2];

		for (i = j+1; i < number_of_beads; i++)
		{
			r = 6*results_position(i, j, number_of_beads);
			I = 3*i;
			I1 = I + 1;
			I2 = I + 2;

			M_matrix[I + J*N] = results[r];
			M_matrix[J + I*N] = results[r];
			M_matrix[I1 + J1*N] = results[r + 1];
			M_matrix[J1 + I1*N] = results[r + 1];
			M_matrix[I2 + J2*N] = results[r + 2];
			M_matrix[J2 + I2*N] = results[r + 2];
			M_matrix[I1 + J*N] = results[r + 3];
			M_matrix[I + J1*N] = results[r + 3];
			M_matrix[J + I1*N] = results[r + 3];
			M_matrix[J1 + I*N] = results[r + 3];
			M_matrix[I2 + J*N] = results[r + 4];
			M_matrix[I + J2*N] = results[r + 4];
			M_matrix[J + I2*N] = results[r + 4];
			M_matrix[J2 + I*N] = results[r + 4];
			M_matrix[I2 + J1*N] = results[r + 5];
			M_matrix[I1 + J2*N] = results[r + 5];
			M_matrix[J1 + I2*N] = results[r + 5];
			M_matrix[J2 + I1*N] = results[r + 5];
		}
	}

	free(results);
}

// -------------------------------------------------------------------------------

void RPY_Smith_M_tt_matrix(double* as, double* pointers, double box_length, double alpha, int m_max, int n_max, int number_of_beads, double* M_matrix)
{
	register int i = 0;
	register int j = 0;

	int I, J, J1, J2, I1, I2, r, shift_results, shift_pointers;

	int N = 3*number_of_beads;

	double rx, ry, rz;

	double* vector;

	double* results = calloc(3*(number_of_beads*number_of_beads+number_of_beads), sizeof(double));

	#pragma omp parallel for private(i, vector, shift_results, shift_pointers, rx, ry, rz)
	for (j = 0; j < number_of_beads; j++)
	{
		vector = calloc(6, sizeof(double));

		RPY_Smith_Mii_block(*(as+j), box_length, alpha, m_max, n_max, vector);

		shift_results = 6*results_position(j,j,number_of_beads);

		*(results + shift_results) = *vector;
		*(results + shift_results + 1) = *(vector + 1);
		*(results + shift_results + 2) = *(vector + 2);

		for (i = j + 1; i < number_of_beads; i++)
		{
			shift_results = 6*results_position(i,j,number_of_beads);
			shift_pointers = 3*results_position(i-1,j,number_of_beads-1);

			rx = *(pointers + shift_pointers);
			ry = *(pointers + shift_pointers + 1);
			rz = *(pointers + shift_pointers + 2);

			RPY_Smith_Mij_block(*(as+i), *(as+j), rx, ry, rz, box_length, alpha, m_max, n_max, vector);

			*(results + shift_results) = *vector;
			*(results + shift_results + 1) = *(vector + 1);
			*(results + shift_results + 2) = *(vector + 2);
			*(results + shift_results + 3) = *(vector + 3);
			*(results + shift_results + 4) = *(vector + 4);
			*(results + shift_results + 5) = *(vector + 5);
		}

		free(vector);
	}

	for (j = 0; j < number_of_beads; j++)
	{
		r = 6*results_position(j, j, number_of_beads);
		J = 3*j;
		J1 = J + 1;
		J2 = J + 2;

		M_matrix[J + J*N] = results[r];
		M_matrix[J1 + J1*N] = results[r + 1];
		M_matrix[J2 + J2*N] = results[r + 2];
		M_matrix[J1 + J*N] = results[r + 3];
		M_matrix[J + J1*N] = results[r + 3];
		M_matrix[J2 + J*N] = results[r + 4];
		M_matrix[J + J2*N] = results[r + 4];
		M_matrix[J2 + J1*N] = results[r + 5];
		M_matrix[J1 + J2*N] = results[r + 5];

		for (i = j+1; i < number_of_beads; i++)
		{
			r = 6*results_position(i, j, number_of_beads);
			I = 3*i;
			I1 = I + 1;
			I2 = I + 2;

			M_matrix[I + J*N] = results[r];
			M_matrix[J + I*N] = results[r];
			M_matrix[I1 + J1*N] = results[r + 1];
			M_matrix[J1 + I1*N] = results[r + 1];
			M_matrix[I2 + J2*N] = results[r + 2];
			M_matrix[J2 + I2*N] = results[r + 2];
			M_matrix[I1 + J*N] = results[r + 3];
			M_matrix[I + J1*N] = results[r + 3];
			M_matrix[J + I1*N] = results[r + 3];
			M_matrix[J1 + I*N] = results[r + 3];
			M_matrix[I2 + J*N] = results[r + 4];
			M_matrix[I + J2*N] = results[r + 4];
			M_matrix[J + I2*N] = results[r + 4];
			M_matrix[J2 + I*N] = results[r + 4];
			M_matrix[I2 + J1*N] = results[r + 5];
			M_matrix[I1 + J2*N] = results[r + 5];
			M_matrix[J1 + I2*N] = results[r + 5];
			M_matrix[J2 + I1*N] = results[r + 5];
		}
	}

	free(results);
}

// -------------------------------------------------------------------------------

void JO_2B_RA_matrix(double ai, double aj, double rx, double ry, double rz, double* R_matrix)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double s = 2*dist/(ai + aj);

	double l = aj/ai;

	double xa11l = JO_XA11_term(s, l);

	double ya11l = JO_YA11_term(s, l);

	double xa11linv = l * JO_XA11_term(s, 1/l);

	double ya11linv = l * JO_YA11_term(s, 1/l);

	double xa12l = JO_XA12_term(s, l);

	double ya12l = JO_YA12_term(s, l);

	double mult = 6 * M_PI * ai;

	int i;

	// block 00

	*R_matrix = xa11l*rx*rx/dist2 + ya11l*(1 - rx*rx/dist2); // 00

	*(R_matrix+1) = xa11l*ry*ry/dist2 + ya11l*(1 - ry*ry/dist2); // 11

	*(R_matrix+2) = xa11l*rz*rz/dist2 + ya11l*(1 - rz*rz/dist2); // 22

	*(R_matrix+3) = (xa11l - ya11l)*rx*ry/dist2; // 10

	*(R_matrix+4) = (xa11l - ya11l)*rx*rz/dist2; // 20

	*(R_matrix+5) = (xa11l - ya11l)*ry*rz/dist2; // 21

	// block 11

	*(R_matrix+6) = xa11linv*rx*rx/dist2 + ya11linv*(1 - rx*rx/dist2); // 33

	*(R_matrix+7) = xa11linv*ry*ry/dist2 + ya11linv*(1 - ry*ry/dist2); // 44

	*(R_matrix+8) = xa11linv*rz*rz/dist2 + ya11linv*(1 - rz*rz/dist2); // 55

	*(R_matrix+9) = (xa11linv - ya11linv)*rx*ry/dist2; // 43

	*(R_matrix+10) = (xa11linv - ya11linv)*rx*rz/dist2; // 53

	*(R_matrix+11) = (xa11linv - ya11linv)*ry*rz/dist2; // 54

	// block 10

	*(R_matrix+12) = xa12l*rx*rx/dist2 + ya12l*(1 - rx*rx/dist2); // 30

	*(R_matrix+13) = xa12l*ry*ry/dist2 + ya12l*(1 - ry*ry/dist2); // 41

	*(R_matrix+14) = xa12l*rz*rz/dist2 + ya12l*(1 - rz*rz/dist2); // 52

	*(R_matrix+15) = (xa12l - ya12l)*rx*ry/dist2; // 40

	*(R_matrix+16) = (xa12l - ya12l)*rx*rz/dist2; // 50

	*(R_matrix+17) = (xa12l - ya12l)*ry*rz/dist2; // 51

	for (i = 0; i < 18; i++)
	{
		*(R_matrix+i) *= mult;
	}
}

// -------------------------------------------------------------------------------

void JO_2B_RB_matrix(double ai, double aj, double rx, double ry, double rz, double* R_matrix)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double s = 2*dist/(ai + aj);

	double l = aj/ai;

	double yb11l = JO_YB11_term(s, l);

	double yb11linv = l * l * JO_YB11_term(s, 1/l);

	double yb12l = JO_YB12_term(s, l);

	double mult = 4 * M_PI * ai * ai;

	int i;

	// block 00

	*(R_matrix) = yb11l*rz/dist; // 01

	*(R_matrix+1) = -yb11l*ry/dist; // 02

	*(R_matrix+2) = yb11l*rx/dist; // 12

	// block 11

	*(R_matrix+3) = yb11linv*rz/dist; // 34

	*(R_matrix+4) = -yb11linv*ry/dist; // 35

	*(R_matrix+5) = yb11linv*rx/dist; // 45

	// block 10

	*(R_matrix+6) = yb12l*rz/dist; // 04

	*(R_matrix+7) = -yb12l*ry/dist; // 05

	*(R_matrix+8) = yb12l*rx/dist; // 15

	for (i = 0; i < 9; i++)
	{
		*(R_matrix+i) *= mult;
	}
}

// -------------------------------------------------------------------------------

void JO_2B_RC_matrix(double ai, double aj, double rx, double ry, double rz, double* R_matrix)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double s = 2*dist/(ai + aj);

	double l = aj/ai;

	double xc11l = JO_XC11_term(s, l);

	double yc11l = JO_YC11_term(s, l);

	double xc11linv = l * l * l * JO_XC11_term(s, 1/l);

	double yc11linv = l * l * l * JO_YC11_term(s, 1/l);

	double xc12l = JO_XC12_term(s, l);

	double yc12l = JO_YC12_term(s, l);

	double mult = 8 * M_PI * ai * ai * ai;

	int i;

	// block 00

	*R_matrix = xc11l*rx*rx/dist2 + yc11l*(1 - rx*rx/dist2); // 00

	*(R_matrix+1) = xc11l*ry*ry/dist2 + yc11l*(1 - ry*ry/dist2); // 11

	*(R_matrix+2) = xc11l*rz*rz/dist2 + yc11l*(1 - rz*rz/dist2); // 22

	*(R_matrix+3) = (xc11l - yc11l)*rx*ry/dist2; // 10

	*(R_matrix+4) = (xc11l - yc11l)*rx*rz/dist2; // 20

	*(R_matrix+5) = (xc11l - yc11l)*ry*rz/dist2; // 21

	// block 11

	*(R_matrix+6) = xc11linv*rx*rx/dist2 + yc11linv*(1 - rx*rx/dist2); // 33

	*(R_matrix+7) = xc11linv*ry*ry/dist2 + yc11linv*(1 - ry*ry/dist2); // 44

	*(R_matrix+8) = xc11linv*rz*rz/dist2 + yc11linv*(1 - rz*rz/dist2); // 55

	*(R_matrix+9) = (xc11linv - yc11linv)*rx*ry/dist2; // 43

	*(R_matrix+10) = (xc11linv - yc11linv)*rx*rz/dist2; // 53

	*(R_matrix+11) = (xc11linv - yc11linv)*ry*rz/dist2; // 54

	// block 10

	*(R_matrix+12) = xc12l*rx*rx/dist2 + yc12l*(1 - rx*rx/dist2); // 30

	*(R_matrix+13) = xc12l*ry*ry/dist2 + yc12l*(1 - ry*ry/dist2); // 41

	*(R_matrix+14) = xc12l*rz*rz/dist2 + yc12l*(1 - rz*rz/dist2); // 52

	*(R_matrix+15) = (xc12l - yc12l)*rx*ry/dist2; // 40

	*(R_matrix+16) = (xc12l - yc12l)*rx*rz/dist2; // 50

	*(R_matrix+17) = (xc12l - yc12l)*ry*rz/dist2; // 51

	for (i = 0; i < 18; i++)
	{
		*(R_matrix+i) *= mult;
	}
}

// -------------------------------------------------------------------------------

void JO_R_lubrication_correction_F_matrix(double* as, double* pointers, int number_of_beads, double cutoff_distance, int cichocki_correction, double* correction_matrix)
{
	register int i = 0;

	register int j = 0;

	register int k = 0;

	double dist2, dist;

	double* nf2b;

	double* ff2b;

	double* temp_nf2b;

	double* temp_ff2b;

	double rx, ry, rz;

	int I, I1, I2, J, J1, J2, r;

	int N = 3*number_of_beads;

	int shift_pointers;

	double* results;

	for (j = 0; j < number_of_beads; j++)
	{
		for (i = j + 1; i < number_of_beads; i++)
		{

			shift_pointers = 3*results_position(i-1,j,number_of_beads-1);

			rx = *(pointers + shift_pointers);
			ry = *(pointers + shift_pointers + 1);
			rz = *(pointers + shift_pointers + 2);

			dist2 = rx*rx + ry*ry + rz*rz;
			dist = sqrt(dist2);

			if ( (dist - *(as+j) - *(as+i)) / ( *(as+j) + *(as+i) ) <= cutoff_distance)
			{
				nf2b = calloc(18, sizeof(double));

				ff2b = calloc(18, sizeof(double));

				if (cichocki_correction)
				{
					temp_nf2b = calloc(18, sizeof(double));
					temp_ff2b = calloc(18, sizeof(double));

					JO_2B_RA_matrix(*(as+j), *(as+i), rx, ry, rz, temp_nf2b);
					Cichocki_2B_R_correction(temp_nf2b, nf2b);
					RPY_2B_RA_matrix(*(as+j), *(as+i), rx, ry, rz, temp_ff2b);
					Cichocki_2B_R_correction(temp_ff2b, ff2b);

					free(temp_nf2b);
					free(temp_ff2b);
				}
				else
				{
					JO_2B_RA_matrix(*(as+j), *(as+i), rx, ry, rz, nf2b);
					RPY_2B_RA_matrix(*(as+j), *(as+i), rx, ry, rz, ff2b);
				}

				results = calloc(18, sizeof(double));

				for (k = 0; k < 18; k++)
				{
					*(results+k) = *(nf2b+k) - *(ff2b+k);
				}

				J = 3*j;
				J1 = J + 1;
				J2 = J + 2;

				correction_matrix[J + J*N] += results[0];
				correction_matrix[J1 + J1*N] += results[1];
				correction_matrix[J2 + J2*N] += results[2];
				correction_matrix[J1 + J*N] += results[3];
				correction_matrix[J + J1*N] += results[3];
				correction_matrix[J2 + J*N] += results[4];
				correction_matrix[J + J2*N] += results[4];
				correction_matrix[J2 + J1*N] += results[5];
				correction_matrix[J1 + J2*N] += results[5];

				r = 6;
				I = 3*i;
				I1 = I + 1;
				I2 = I + 2;

				correction_matrix[I + I*N] += results[r];
				correction_matrix[I1 + I1*N] += results[r + 1];
				correction_matrix[I2 + I2*N] += results[r + 2];
				correction_matrix[I1 + I*N] += results[r + 3];
				correction_matrix[I + I1*N] += results[r + 3];
				correction_matrix[I2 + I*N] += results[r + 4];
				correction_matrix[I + I2*N] += results[r + 4];
				correction_matrix[I2 + I1*N] += results[r + 5];
				correction_matrix[I1 + I2*N] += results[r + 5];

				r = 12;

				correction_matrix[I + J*N] += results[r];
				correction_matrix[J + I*N] += results[r];
				correction_matrix[I1 + J1*N] += results[r + 1];
				correction_matrix[J1 + I1*N] += results[r + 1];
				correction_matrix[I2 + J2*N] += results[r + 2];
				correction_matrix[J2 + I2*N] += results[r + 2];
				correction_matrix[I1 + J*N] += results[r + 3];
				correction_matrix[I + J1*N] += results[r + 3];
				correction_matrix[J + I1*N] += results[r + 3];
				correction_matrix[J1 + I*N] += results[r + 3];
				correction_matrix[I2 + J*N] += results[r + 4];
				correction_matrix[I + J2*N] += results[r + 4];
				correction_matrix[J + I2*N] += results[r + 4];
				correction_matrix[J2 + I*N] += results[r + 4];
				correction_matrix[I2 + J1*N] += results[r + 5];
				correction_matrix[I1 + J2*N] += results[r + 5];
				correction_matrix[J1 + I2*N] += results[r + 5];
				correction_matrix[J2 + I1*N] += results[r + 5];


				free(nf2b);

				free(ff2b);

				free(results);
			}
		}
	}
}

// -------------------------------------------------------------------------------

void JO_R_lubrication_correction_FT_matrix(double* as, double* pointers, int number_of_beads, double cutoff_distance, int cichocki_correction, double* correction_matrix)
{
	register int i = 0;

	register int j = 0;

	register int k = 0;

	double dist2, dist;

	double* nf2b_tt;

	double* nf2b_rr;

	double* nf2b_rt;

	double* ff2b;

	double* temp_nf2b_tt;

	double* temp_nf2b_rr;

	double* temp_nf2b_rt;

	double* temp_ff2b;

	double rx, ry, rz;

	int I, I1, I2, J, J1, J2, r;

	int N = 6*number_of_beads;

	// int tr_shift = 3*(number_of_beads*number_of_beads+number_of_beads);

	int shift_pointers;

	double* results;

	for (j = 0; j < number_of_beads; j++)
	{
		for (i = j + 1; i < number_of_beads; i++)
		{

			shift_pointers = 3*results_position(i-1,j,number_of_beads-1);

			rx = *(pointers + shift_pointers);
			ry = *(pointers + shift_pointers + 1);
			rz = *(pointers + shift_pointers + 2);

			dist2 = rx*rx + ry*ry + rz*rz;
			dist = sqrt(dist2);

			if ( (dist - *(as+j) - *(as+i)) / ( *(as+j) + *(as+i) ) <= cutoff_distance)
			{
				nf2b_tt = calloc(18, sizeof(double));

				nf2b_rr = calloc(18, sizeof(double));

				nf2b_rt = calloc(9, sizeof(double));

				ff2b = calloc(45, sizeof(double));

				if (cichocki_correction)
				{
					temp_nf2b_tt = calloc(18, sizeof(double));

					temp_nf2b_rr = calloc(18, sizeof(double));

					temp_nf2b_rt = calloc(9, sizeof(double));

					temp_ff2b = calloc(45, sizeof(double));

					JO_2B_RA_matrix(*(as+j), *(as+i), rx, ry, rz, temp_nf2b_tt);
					JO_2B_RC_matrix(*(as+j), *(as+i), rx, ry, rz, temp_nf2b_rr);
					JO_2B_RB_matrix(*(as+j), *(as+i), rx, ry, rz, temp_nf2b_rt);
					Cichocki_2B_R_correction(temp_nf2b_tt, nf2b_tt);
					Cichocki_2B_R_correction(temp_nf2b_rr, nf2b_rr);
					Cichocki_2B_R_correction(temp_nf2b_rt, nf2b_rt);
					RPY_2B_R_matrix(*(as+j), *(as+i), rx, ry, rz, temp_ff2b);
					Cichocki_2B_R_correction(temp_ff2b, ff2b);

					free(temp_nf2b_tt);
					free(temp_nf2b_rr);
					free(temp_nf2b_rt);
					free(temp_ff2b);
				}
				else
				{
					JO_2B_RA_matrix(*(as+j), *(as+i), rx, ry, rz, nf2b_tt);
					JO_2B_RC_matrix(*(as+j), *(as+i), rx, ry, rz, nf2b_rr);
					JO_2B_RB_matrix(*(as+j), *(as+i), rx, ry, rz, nf2b_rt);
					RPY_2B_R_matrix(*(as+j), *(as+i), rx, ry, rz, ff2b);
				}

				results = calloc(45, sizeof(double));

				for (k = 0; k < 18; k++)
				{
					*(results+k) = *(nf2b_tt+k) - *(ff2b+k);
					*(results+k+18) = *(nf2b_rr+k) - *(ff2b+k+18);
				}

				for (k = 0; k > 9; k++)
					*(results+k+36) = *(nf2b_rt+k) - *(ff2b+k+36);
				}

				J = 3*j;
				J1 = J + 1;
				J2 = J + 2;
				J3 = J + 3*number_of_beads;
				J4 = J1 + 3*number_of_beads;
				J5 = J2 + 3*number_of_beads;

				I = 3*i;
				I1 = I + 1;
				I2 = I + 2;
				I3 = I + 3*number_of_beads;
				I1 = I1 + 3*number_of_beads;
				I2 = I2 + 3*number_of_beads;

				correction_matrix[J + J*N] += results[0];
				correction_matrix[J1 + J1*N] += results[1];
				correction_matrix[J2 + J2*N] += results[2];
				correction_matrix[J1 + J*N] += results[3];
				correction_matrix[J + J1*N] += results[3];
				correction_matrix[J2 + J*N] += results[4];
				correction_matrix[J + J2*N] += results[4];
				correction_matrix[J2 + J1*N] += results[5];
				correction_matrix[J1 + J2*N] += results[5];

				correction_matrix[I + I*N] += results[6];
				correction_matrix[I1 + I1*N] += results[7];
				correction_matrix[I2 + I2*N] += results[8];
				correction_matrix[I1 + I*N] += results[9];
				correction_matrix[I + I1*N] += results[9];
				correction_matrix[I2 + I*N] += results[10];
				correction_matrix[I + I2*N] += results[10];
				correction_matrix[I2 + I1*N] += results[11];
				correction_matrix[I1 + I2*N] += results[11];

				correction_matrix[I + J*N] += results[12];
				correction_matrix[J + I*N] += results[12];
				correction_matrix[I1 + J1*N] += results[13];
				correction_matrix[J1 + I1*N] += results[13];
				correction_matrix[I2 + J2*N] += results[14];
				correction_matrix[J2 + I2*N] += results[14];
				correction_matrix[I1 + J*N] += results[15];
				correction_matrix[I + J1*N] += results[15];
				correction_matrix[J1 + I*N] += results[15];
				correction_matrix[J + I1*N] += results[15];
				correction_matrix[I2 + J*N] += results[16];
				correction_matrix[I + J2*N] += results[16];
				correction_matrix[J2 + I*N] += results[16];
				correction_matrix[J + I2*N] += results[16];
				correction_matrix[I2 + J1*N] += results[17];
				correction_matrix[I1 + J2*N] += results[17];
				correction_matrix[J2 + I1*N] += results[17];
				correction_matrix[J1 + I2*N] += results[17];

				correction_matrix[J3 + J3*N] += results[18];
				correction_matrix[J4 + J4*N] += results[19];
				correction_matrix[J5 + J5*N] += results[20];
				correction_matrix[J4 + J3*N] += results[21];
				correction_matrix[J3 + J4*N] += results[21];
				correction_matrix[J5 + J3*N] += results[22];
				correction_matrix[J3 + J5*N] += results[22];
				correction_matrix[J5 + J4*N] += results[23];
				correction_matrix[J4 + J5*N] += results[23];

				correction_matrix[I3 + I3*N] += results[24];
				correction_matrix[I4 + I4*N] += results[25];
				correction_matrix[I5 + I5*N] += results[26];
				correction_matrix[I4 + I3*N] += results[27];
				correction_matrix[I3 + I4*N] += results[27];
				correction_matrix[I5 + I3*N] += results[28];
				correction_matrix[I3 + I5*N] += results[28];
				correction_matrix[I5 + I4*N] += results[29];
				correction_matrix[I4 + I5*N] += results[29];

				correction_matrix[I3 + J3*N] += results[30];
				correction_matrix[J3 + I3*N] += results[30];
				correction_matrix[I4 + J4*N] += results[31];
				correction_matrix[J4 + I4*N] += results[31];
				correction_matrix[I5 + J5*N] += results[32];
				correction_matrix[J5 + I5*N] += results[32];
				correction_matrix[I4 + J3*N] += results[33];
				correction_matrix[I3 + J4*N] += results[33];
				correction_matrix[J4 + I3*N] += results[33];
				correction_matrix[J3 + I4*N] += results[33];
				correction_matrix[I5 + J3*N] += results[34];
				correction_matrix[I3 + J5*N] += results[34];
				correction_matrix[J5 + I3*N] += results[34];
				correction_matrix[J3 + I5*N] += results[34];
				correction_matrix[I5 + J4*N] += results[35];
				correction_matrix[I4 + J5*N] += results[35];
				correction_matrix[J5 + I4*N] += results[35];
				correction_matrix[J4 + I5*N] += results[35];

				correction_matrix[I4 + I*N] += results[36];
				correction_matrix[I3 + I1*N] -= results[36];
				correction_matrix[I1 + I3*N] += results[36];
				correction_matrix[I + I4*N] -= results[36];
				correction_matrix[I5 + I*N] += results[37];
				correction_matrix[I3 + I2*N] -= results[37];
				correction_matrix[I2 + I3*N] += results[37];
				correction_matrix[I + I5*N] -= results[37];
				correction_matrix[I5 + I*N] += results[38];
				correction_matrix[I3 + I2*N] -= results[38];
				correction_matrix[I2 + I3*N] += results[38];
				correction_matrix[I + I5*N] -= results[38];
				correction_matrix[J4 + J*N] += results[39];
				correction_matrix[J3 + J1*N] -= results[39];
				correction_matrix[J1 + J3*N] += results[39];
				correction_matrix[J + J4*N] -= results[39];
				correction_matrix[J5 + J*N] += results[40];
				correction_matrix[J3 + J2*N] -= results[40];
				correction_matrix[J2 + J3*N] += results[40];
				correction_matrix[J + J5*N] -= results[40];
				correction_matrix[J5 + J*N] += results[41];
				correction_matrix[J3 + J2*N] -= results[41];
				correction_matrix[J2 + J3*N] += results[41];
				correction_matrix[J + J5*N] -= results[41];
				correction_matrix[J4 + I*N] += results[42];
				correction_matrix[J3 + I1*N] -= results[42];
				correction_matrix[I4 + J*N] -= results[42];
				correction_matrix[I3 + J1*N] += results[42];
				correction_matrix[J1 + I3*N] += results[42];
				correction_matrix[J + I4*N] -= results[42];
				correction_matrix[I1 + J3*N] -= results[42];
				correction_matrix[I + J4*N] += results[42];
				correction_matrix[J5 + I*N] += results[43];
				correction_matrix[J3 + I2*N] -= results[43];
				correction_matrix[I5 + J*N] -= results[43];
				correction_matrix[I3 + J2*N] += results[43];
				correction_matrix[J2 + I3*N] += results[43];
				correction_matrix[J + I5*N] -= results[43];
				correction_matrix[I2 + J3*N] -= results[43];
				correction_matrix[I + J5*N] += results[43];
				correction_matrix[J5 + I1*N] += results[44];
				correction_matrix[J4 + I2*N] -= results[44];
				correction_matrix[I5 + J1*N] -= results[44];
				correction_matrix[I4 + J2*N] += results[44];
				correction_matrix[J2 + I4*N] += results[44];
				correction_matrix[J1 + I5*N] -= results[44];
				correction_matrix[I2 + J4*N] -= results[44];
				correction_matrix[I1 + J5*N] += results[44];

				free(nf2b_tt);

				free(nf2b_rr);

				free(nf2b_rt);

				free(ff2b);

				free(results);
			}
		}
	}
}

// -------------------------------------------------------------------------------

static inline double RPY_Mii_tt_block(double a)
{
	return 1.0 / ( 6 * M_PI * a );
}

// -------------------------------------------------------------------------------

static inline double RPY_Mii_rr_block(double a)
{
	return 1.0 / ( 8 * M_PI * a * a * a );
}

// -------------------------------------------------------------------------------

static void RPY_Mij_tt_block(double ai, double aj, double rx, double ry, double rz, double* matrix)
{
	double al, as;

	if (ai > aj)
	{
		al = ai;
		as = aj;
	}
	else
	{
		al = aj;
		as = ai;
	}
	double dist2 = rx*rx + ry*ry + rz*rz;
	double dist = sqrt(dist2);
	double aij2 = ai*ai + aj*aj;

	if (dist > (ai + aj))
	{
		double coef1 = 1.0 / ( 8 * M_PI * dist );
		double coef2 = 1.0 + aij2 / ( 3 * dist2 );
		double coef3 = ( 1.0 - aij2 / dist2 ) / dist2;

		*matrix = coef1 * ( coef2 + coef3 * rx * rx );
		*(matrix+1) = coef1 * ( coef2 + coef3 * ry * ry );
		*(matrix+2) = coef1 * ( coef2 + coef3 * rz * rz );
		*(matrix+3) = coef1 * coef3 * rx * ry;
		*(matrix+4) = coef1 * coef3 * rx * rz;
		*(matrix+5) = coef1 * coef3 * ry * rz;
	}
	else if (dist <= (al - as))
	{
		double temp = 1.0 / ( 6 * M_PI * al );

		*matrix = temp;
		*(matrix+1) = temp;
		*(matrix+2) = temp;
	}
	else
	{
		double dist3 = dist2 * dist;

		double coef1 = 1.0 / ( 6 * M_PI * ai * aj );
		double coef2 = 16 * dist3 * ( ai + aj );
		double coef3 = (ai - aj) * (ai - aj) + 3*dist2;
		coef3 *= coef3;
		double coef4 = (coef2 - coef3) / 32 / dist3;
		double coef5 = ( (ai - aj)*(ai - aj) - dist2 );
		coef5 *= 3 * coef5;
		double coef6 = coef5 / ( 32 * dist3 ) / dist2;

		*matrix = coef1 * ( coef4 + coef6 * rx * rx );
		*(matrix+1) = coef1 * ( coef4 + coef6 * ry * ry );
		*(matrix+2) = coef1 * ( coef4 + coef6 * rz * rz );
		*(matrix+3) = coef1 * coef6 * rx * ry;
		*(matrix+4) = coef1 * coef6 * rx * rz;
		*(matrix+5) = coef1 * coef6 * ry * rz;
	}
}

// -------------------------------------------------------------------------------

static void RPY_Mij_rr_block(double ai, double aj, double rx, double ry, double rz, double* matrix)
{
	double al, as;

	if (ai > aj)
	{
		al = ai;
		as = aj;
	}
	else
	{
		al = aj;
		as = ai;
	}
	double dist2 = rx*rx + ry*ry + rz*rz;
	double dist = sqrt(dist2);
	double dist3 = dist2 * dist;
	double aij2 = ai*ai + aj*aj;

	if (dist > (ai + aj))
	{
		double coef1 = 1.0 / ( 16 * M_PI * dist3 );
		double coef2 = 3 * coef1 / dist2;

		*matrix = coef2 * rx * rx - coef1;
		*(matrix+1) = coef2 * ry * ry - coef1;
		*(matrix+2) = coef2 * rz * rz - coef1;
		*(matrix+3) = coef2 * rx * ry;
		*(matrix+4) = coef2 * rx * rz;
		*(matrix+5) = coef2 * ry * rz;
	}
	else if (dist <= (al - as))
	{
		double temp = 1.0 / ( 8 * M_PI * ai * ai * ai );

		*matrix = temp;
		*(matrix+1) = temp;
		*(matrix+2) = temp;
	}
	else
	{
		double coef1 = 1.0 / ( 8 * M_PI * ai * ai * ai * aj * aj * aj );
		double coef2 = 5 * dist3 * dist3 - 27 * dist2 * dist2 * aij2;
		double coef3 = 32 * dist3 * (ai*ai*ai + aj*aj*aj);
		double coef4 = -9 * dist2 * pow(ai*ai - aj*aj, 2) - pow(ai-aj, 4)*(aij2 + 4*ai*aj);
		double coef5 = 64.0 * dist3;
		double coef6 = coef1 * (coef2 + coef3 + coef4) / coef5;

		double coef7 = 3 * pow((ai - aj)*(ai - aj) - dist2, 2);
		double coef8 = aij2 + 4*ai*aj - dist2;
		double coef9 = coef1 * coef7 * coef8 / coef5 / dist2;

		*matrix = coef6 + coef9 * rx * rx;
		*(matrix+1) = coef6 + coef9 * ry * ry;
		*(matrix+2) = coef6 + coef9 * rz * rz;
		*(matrix+3) = coef9 * rx * ry;
		*(matrix+4) = coef9 * rx * rz;
		*(matrix+5) = coef9 * ry * rz;
	}
}

// -------------------------------------------------------------------------------

static void RPY_Mij_rt_block(double ai, double aj, double rx, double ry, double rz, double* matrix)
{
	double al, as;

	if (ai > aj)
	{
		al = ai;
		as = aj;
	}
	else
	{
		al = aj;
		as = ai;
	}
	double dist2 = rx*rx + ry*ry + rz*rz;
	double dist = sqrt(dist2);
	// double aij2 = ai*ai + aj*aj;

	if (dist > (ai + aj))
	{
		double coef1 = 1.0 / ( 8 * M_PI * dist2 );

		*(matrix+3) = -coef1 * rz / dist;
		*(matrix+4) = coef1 * ry / dist;
		*(matrix+5) = -coef1 * rx / dist;
	}
	else if (dist <= (al - as))
	{
		if (ai > aj)
		{
			double dist3 = dist2 * dist;
			double coef1 = dist / (8 * M_PI * dist3);

			*(matrix+3) = -coef1 * rz / dist;
			*(matrix+4) = coef1 * ry / dist;
			*(matrix+5) = -coef1 * rx / dist;
		}
	}
	else
	{
		double coef1 = 1.0 / ( 18 * M_PI * ai * ai * ai * aj );
		double coef2 = pow(ai - aj + dist, 2);
		double coef3 = aj*aj + 2*aj*(ai+dist) - 3*pow(ai-dist, 2);
		double coef4 = 8*dist2;
		double coef5 = coef1 * coef2 * coef3 / coef4;

		*(matrix+3) = -coef5 * rz / dist;
		*(matrix+4) = coef5 * ry / dist;
		*(matrix+5) = -coef5 * rx / dist;
	}
}

// -------------------------------------------------------------------------------

static void RPY_Smith_Mii_block(double a, double box_length, double alpha, int m_max, int n_max, double* matrix)
{
	double coef1 = 1.0 / ( 6 * M_PI * a );
	double coef2 = 3 * a / ( 4 * box_length );
	double coef3 = a * a * a / ( 2 * box_length * box_length * box_length );

	double* comp1 = calloc(6, sizeof(double));
	double* comp2 = calloc(6, sizeof(double));

	RPY_Smith_Oii_block(a, box_length, alpha, m_max, n_max, comp1);
	RPY_Smith_Qii_block(a, box_length, alpha, m_max, n_max, comp2);

	*matrix = coef1 * ( 1.0 + coef2 * *comp1 + coef3 * *comp2 );
	*(matrix+1) = coef1 * ( 1.0 + coef2 * *(comp1+1) + coef3 * *(comp2+1) );
	*(matrix+2) = coef1 * ( 1.0 + coef2 * *(comp1+2) + coef3 * *(comp2+2) );

	*(matrix+3) = coef1 * ( coef2 * *(comp1+3) + coef3 * *(comp2+3) );
	*(matrix+4) = coef1 * ( coef2 * *(comp1+4) + coef3 * *(comp2+4) );
	*(matrix+5) = coef1 * ( coef2 * *(comp1+5) + coef3 * *(comp2+5) );

	free(comp1);
	free(comp2);
}

// -------------------------------------------------------------------------------

static void RPY_Smith_Mij_block(double ai, double aj, double rx, double ry, double rz, double box_length, double alpha, int m_max, int n_max, double* matrix)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double sigmax = rx / box_length;
	double sigmay = ry / box_length;
	double sigmaz = rz / box_length;

	double coef1 = 1.0 / ( 6 * M_PI * ai );
	double coef2 = 3 * ai / ( 4 * box_length );
	double coef3;

	if (ai == aj)
	{
		coef3 = ai * ai * ai / ( 2 * box_length * box_length * box_length );
	}
	else
	{
		coef3 = ai * ( ai * ai + aj * aj ) / ( 4 * box_length * box_length * box_length );
	}

	double* comp1 = calloc(6, sizeof(double));
	double* comp2 = calloc(6, sizeof(double));

	RPY_Smith_Oij_block(sigmax, sigmay, sigmaz, alpha, m_max, n_max, comp1);
	RPY_Smith_Qij_block(sigmax, sigmay, sigmaz, alpha, m_max, n_max, comp2);

	*matrix = coef1 * ( coef2 * *comp1 + coef3 * *comp2 );
	*(matrix+1) = coef1 * ( coef2 * *(comp1+1) + coef3 * *(comp2+1) );
	*(matrix+2) = coef1 * ( coef2 * *(comp1+2) + coef3 * *(comp2+2) );
	*(matrix+3) = coef1 * ( coef2 * *(comp1+3) + coef3 * *(comp2+3) );
	*(matrix+4) = coef1 * ( coef2 * *(comp1+4) + coef3 * *(comp2+4) );
	*(matrix+5) = coef1 * ( coef2 * *(comp1+5) + coef3 * *(comp2+5) );


	if (dist2 < (ai + aj)*(ai + aj))
	{
		double dist = sqrt(dist2);

		double aij2 = ai*ai + aj*aj;

		double* Aij = calloc(6, sizeof(double));

		RPY_Mij_tt_block(ai, aj, rx, ry, rz, Aij);

		*matrix += *(Aij);
		*(matrix+1) += *(Aij+1);
		*(matrix+2) += *(Aij+2);
		*(matrix+3) += *(Aij+3);
		*(matrix+4) += *(Aij+4);
		*(matrix+5) += *(Aij+5);

		coef1 = 1.0 / ( 8 * M_PI * dist );
		coef2 = 1.0 + aij2 / ( 3 * dist2 );
		coef3 = ( 1.0 - aij2 / dist2 ) / dist2;

		*matrix -= coef1 * ( coef2 + coef3 * rx * rx);
		*(matrix+1) -= coef1 * ( coef2 + coef3 * ry * ry );
		*(matrix+2) -= coef1 * ( coef2 + coef3 * rz * rz );
		*(matrix+3) -= coef1 * coef3 * rx * ry;
		*(matrix+4) -= coef1 * coef3 * rx * rz;
		*(matrix+5) -= coef1 * coef3 * ry * rz;

		free(Aij);
	}

	free(comp1);

	free(comp2);
}

// -------------------------------------------------------------------------------

static void RPY_Smith_Oii_block(double a, double box_length, double alpha, int m_max, int n_max, double* matrix)
{
	register int mx;
	register int my;
	register int mz;
	register int nx;
	register int ny;
	register int nz;

	double mlen, mlen2, nlen2, mult, temp;

	int mbis, mtris, nbis, ntris;

	double* O_values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	double mult0 = 2 * alpha / sqrt_pi;

	double exp_const = -M_PI*M_PI/alpha2;

	for (mx = -m_max; mx <= m_max; mx++)
	{
		mbis = m_max - abs(mx);
		for (my = -mbis; my <= mbis; my++)
		{
			mtris = mbis - abs(my);
			for (mz = -mtris; mz <= mtris; mz++)
			{
				if (!(mx==0 && my==0 && mz==0))
				{
					mlen2 = mx*mx + my*my + mz*mz;
					mlen = sqrt(mlen2);
					RPY_O_function(mx, my, mz, O_values);
					mult = erfc( alpha*mlen );

					*matrix += mult * *O_values;
					*(matrix+1) += mult * *(O_values+1);
					*(matrix+2) += mult * *(O_values+2);
					*(matrix+3) += mult * *(O_values+3);
					*(matrix+4) += mult * *(O_values+4);
					*(matrix+5) += mult * *(O_values+5);

					mult = mult0 * exp( - alpha2 * mlen2 ) / mlen2;

					*matrix += mult * mx * mx;
					*(matrix+1) += mult * my * my;
					*(matrix+2) += mult * mz * mz;
					*(matrix+3) += mult * mx * my;
					*(matrix+4) += mult * mx * mz;
					*(matrix+5) += mult * my * mz;
				}
			}
		}
	}

	mult0 = 2.0 / M_PI;

	for (nx = -n_max; nx <= n_max; nx++)
	{
		nbis = n_max - abs(nx);
		for (ny = -nbis; ny <= nbis; ny++)
		{
			ntris = nbis - abs(ny);
			for (nz = -ntris; nz <= ntris; nz++)
			{
				if (!(nx==0 && ny==0 && nz==0))
				{
					nlen2 = nx*nx + ny*ny + nz*nz;
					mult = mult0 / nlen2 * exp(exp_const * nlen2);
					temp = 1.0 / nlen2 - exp_const;

					*matrix += mult * ( 1.0 - temp * nx * nx );
					*(matrix+1) += mult * ( 1.0 - temp * ny * ny );
					*(matrix+2) += mult * ( 1.0 - temp * nz * nz );
					*(matrix+3) += mult * ( - temp * nx * ny );
					*(matrix+4) += mult * ( - temp * nx * nz );
					*(matrix+5) += mult * ( - temp * ny * nz );
				}
			}
		}
	}

	temp = 1.5 * alpha * a / ( sqrt_pi * box_length );

	*matrix -= temp;
	*(matrix+1) -= temp;
	*(matrix+2) -= temp;

	free(O_values);
}

// -------------------------------------------------------------------------------

static void RPY_Smith_Qii_block(double a, double box_length, double alpha, int m_max, int n_max, double* matrix)
{
	register int mx;
	register int my;
	register int mz;
	register int nx;
	register int ny;
	register int nz;

	double amlen, amlen2, expamlen2, mlen2, nlen2, mult;

	int mbis, mtris, nbis, ntris;

	double* Q_values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	double alpha3 = alpha * alpha2;

	double a3 = a * a * a;

	double box_length3 = box_length * box_length * box_length;

	double temp1 = 2 / sqrt_pi;

	double temp2 = 2 * alpha3;

	for (mx = -m_max; mx <= m_max; mx++)
	{
		mbis = m_max - abs(mx);
		for (my = -mbis; my <= mbis; my++)
		{
			mtris = mbis - abs(my);
			for (mz = -mtris; mz <= mtris; mz++)
			{
				if (!(mx==0 && my==0 && mz==0))
				{
					mlen2 = mx*mx + my*my + mz*mz;
					amlen = alpha * sqrt(mlen2);
					amlen2 = amlen * amlen;
					expamlen2 = exp( -amlen2 );
					RPY_Q_function(mx, my, mz, Q_values);
					mult = erfc( amlen ) + temp1 * amlen * expamlen2;

					*matrix += mult * *Q_values;
					*(matrix+1) += mult * *(Q_values+1);
					*(matrix+2) += mult * *(Q_values+2);
					*(matrix+3) += mult * *(Q_values+3);
					*(matrix+4) += mult * *(Q_values+4);
					*(matrix+5) += mult * *(Q_values+5);

					mult = temp1 * temp2 * expamlen2 / mlen2;

					*matrix -= mult * mx * mx;
					*(matrix+1) -= mult * my * my;
					*(matrix+2) -= mult * mz * mz;
					*(matrix+3) -= mult * mx * my;
					*(matrix+4) -= mult * mx * mz;
					*(matrix+5) -= mult * my * mz;
				}
			}
		}
	}

	temp1 = -M_PI * M_PI / alpha2;

	temp2 = 4 * M_PI;

	for (nx = -n_max; nx <= n_max; nx++)
	{
		nbis = n_max - abs(nx);
		for (ny = -nbis; ny <= nbis; ny++)
		{
			ntris = nbis - abs(ny);
			for (nz = -ntris; nz <= ntris; nz++)
			{
				if (!(nx==0 && ny==0 && nz==0))
				{
					nlen2 = nx*nx + ny*ny + nz*nz;
					mult = temp2 * exp( temp1 * nlen2 ) / nlen2;

					*matrix += mult * nx * nx;
					*(matrix+1) += mult * ny * ny;
					*(matrix+2) += mult * nz * nz;
					*(matrix+3) += mult * nx * ny;
					*(matrix+4) += mult * nx * nz;
					*(matrix+5) += mult * ny * nz;
				}
			}
		}
	}

	temp1 = alpha3 * a3 / ( box_length3 * 3.0 * sqrt_pi );

	*matrix -= temp1;
	*(matrix+1) -= temp1;
	*(matrix+2) -= temp1;

	free(Q_values);
}

// -------------------------------------------------------------------------------

static void RPY_Smith_Oij_block(double sigmax, double sigmay, double sigmaz, double alpha, int m_max, int n_max, double* matrix)
{
	register int mx;
	register int my;
	register int mz;
	register int nx;
	register int ny;
	register int nz;

	double mslen, mslen2, nlen2, msx, msy, msz, nsdot, mult, mult2, exp_const;

	int mbis, mtris, nbis, ntris;

	double* O_values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double sqrt_pi = sqrt(M_PI);

	double temp = 2 * alpha / sqrt_pi;

	for (mx = -m_max; mx <= m_max; mx++)
	{
		mbis = m_max - abs(mx);
		for (my = -mbis; my <= mbis; my++)
		{
			mtris = mbis - abs(my);
			for (mz = -mtris; mz <= mtris; mz++)
			{
				msx = mx + sigmax;
				msy = my + sigmay;
				msz = mz + sigmaz;

				mslen2 = msx*msx + msy*msy + msz*msz;
				mslen = sqrt(mslen2);
				mult = erfc( alpha * mslen );
				RPY_O_function(msx, msy, msz, O_values);

				*matrix += mult * *O_values;
				*(matrix+1) += mult * *(O_values+1);
				*(matrix+2) += mult * *(O_values+2);
				*(matrix+3) += mult * *(O_values+3);
				*(matrix+4) += mult * *(O_values+4);
				*(matrix+5) += mult * *(O_values+5);

				mult = temp * exp( -alpha2 * mslen2 ) / mslen2;

				*matrix += mult * msx * msx;
				*(matrix+1) += mult * msy * msy;
				*(matrix+2) += mult * msz * msz;
				*(matrix+3) += mult * msx * msy;
				*(matrix+4) += mult * msx * msz;
				*(matrix+5) += mult * msy * msz;
			}
		}
	}

	temp = M_PI * M_PI / alpha2;

	double temp2 = 2.0 / M_PI;

	for (nx = -n_max; nx <= n_max; nx++)
	{
		nbis = n_max - abs(nx);
		for (ny = -nbis; ny <= nbis; ny++)
		{
			ntris = nbis - abs(ny);
			for (nz = -ntris; nz <= ntris; nz++)
			{
				if (!(nx==0 && ny==0 && nz==0))
				{	
					nlen2 = nx*nx + ny*ny + nz*nz;
					nsdot = nx*sigmax + ny*sigmay + nz*sigmaz;
					exp_const = temp * nlen2;
					mult = temp2 / nlen2 * exp( -exp_const ) * cos(2 * M_PI * nsdot );
					mult2 = (1.0 + exp_const) / nlen2;

					*matrix += mult * ( 1.0 - mult2 * nx * nx );
					*(matrix+1) += mult * ( 1.0 - mult2 * ny * ny );
					*(matrix+2) += mult * ( 1.0 - mult2 * nz * nz );
					*(matrix+3) += mult * ( - mult2 * nx * ny );
					*(matrix+4) += mult * ( - mult2 * nx * nz );
					*(matrix+5) += mult * ( - mult2 * ny * nz );
				}
			}
		}
	}

	free(O_values);
}

// -------------------------------------------------------------------------------

static void RPY_Smith_Qij_block(double sigmax, double sigmay, double sigmaz, double alpha, int m_max, int n_max, double* matrix)
{
	register int mx;
	register int my;
	register int mz;
	register int nx;
	register int ny;
	register int nz;

	double amslen, amslen2, mslen2, expamslen2, nlen2, msx, msy, msz, nsdot, mult;

	int mbis, mtris, nbis, ntris;

	double* Q_values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double alpha3 = alpha2 * alpha;

	double sqrt_pi = sqrt(M_PI);

	double temp1 = 2.0 / sqrt_pi;

	double temp2 = 4 * alpha3 / sqrt_pi;

	for (mx = -m_max; mx <= m_max; mx++)
	{
		mbis = m_max - abs(mx);
		for (my = -mbis; my <= mbis; my++)
		{
			mtris = mbis - abs(my);
			for (mz = -mtris; mz <= mtris; mz++)
			{
				msx = mx + sigmax;
				msy = my + sigmay;
				msz = mz + sigmaz;

				mslen2 = msx*msx + msy*msy + msz*msz;
				amslen = alpha * sqrt(mslen2);
				amslen2 = amslen * amslen;
				expamslen2 = exp( -amslen2 );
				mult = erfc( amslen ) + temp1 * amslen * expamslen2;
				RPY_Q_function(msx, msy, msz, Q_values);

				*matrix += mult * *Q_values;
				*(matrix+1) += mult * *(Q_values+1);
				*(matrix+2) += mult * *(Q_values+2);
				*(matrix+3) += mult * *(Q_values+3);
				*(matrix+4) += mult * *(Q_values+4);
				*(matrix+5) += mult * *(Q_values+5);

				mult = temp2 * expamslen2 / mslen2;

				*matrix -= mult * msx * msx;
				*(matrix+1) -= mult * msy * msy;
				*(matrix+2) -= mult * msz * msz;
				*(matrix+3) -= mult * msx * msy;
				*(matrix+4) -= mult * msx * msz;
				*(matrix+5) -= mult * msy * msz;
			}
		}
	}

	temp1 = -M_PI*M_PI/alpha2;

	temp2 = 4 * M_PI;

	for (nx = -n_max; nx <= n_max; nx++)
	{
		nbis = n_max - abs(nx);
		for (ny = -nbis; ny <= nbis; ny++)
		{
			ntris = nbis - abs(ny);
			for (nz = -ntris; nz <= ntris; nz++)
			{
				if (!(nx==0 && ny==0 && nz==0))
				{
					nlen2 = nx*nx + ny*ny + nz*nz;
					nsdot = nx*sigmax + ny*sigmay + nz*sigmaz;
					mult = temp2 * exp(temp1 * nlen2) * cos(2*M_PI*nsdot) / nlen2;

					*matrix += mult * nx * nx;
					*(matrix+1) += mult * ny * ny;
					*(matrix+2) += mult * nz * nz;
					*(matrix+3) += mult * nx * ny;
					*(matrix+4) += mult * nx * nz;
					*(matrix+5) += mult * ny * nz;
				}
			}
		}
	}

	free(Q_values);
}

// -------------------------------------------------------------------------------

static void RPY_O_function(double rx, double ry, double rz, double* matrix)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double dist3 = dist2 * dist;

	*matrix = ( 1 + rx*rx / dist2 ) / dist;

	*(matrix+1) = ( 1 + ry*ry / dist2 ) / dist;

	*(matrix+2) = ( 1 + rz*rz / dist2 ) / dist;

	*(matrix+3) = rx*ry / dist3;

	*(matrix+4) = rx*rz / dist3;

	*(matrix+5) = ry*rz / dist3;
}

// -------------------------------------------------------------------------------

static void RPY_Q_function(double rx, double ry, double rz, double* matrix)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double dist3 = dist2 * dist;

	double dist5 = dist3 * dist2;

	*matrix = ( 1 - 3 * rx*rx / dist2 ) / dist3;

	*(matrix+1) = ( 1 - 3 * ry*ry / dist2 ) / dist3;

	*(matrix+2) = ( 1 - 3 * rz*rz / dist2 ) / dist3;

	*(matrix+3) = -3 * rx*ry / dist5;

	*(matrix+4) = -3 * rx*rz / dist5;

	*(matrix+5) = -3 * ry*rz / dist5;
}

// -------------------------------------------------------------------------------

static void RPY_2B_R_matrix(double ai, double aj, double rx, double ry, double rz, double* R_matrix)
{
	double Mti = RPY_Mii_tt_block(ai);

	double Mtj = RPY_Mii_tt_block(aj);

	double Mri = RPY_Mii_rr_block(ai);

	double Mrj = RPY_Mii_rr_block(aj);

	double* Mttij = calloc(6, sizeof(double));

	double* Mrrij = calloc(6, sizeof(double));

	double* Mrtij = calloc(6, sizeof(double));

	RPY_Mij_tt_block(ai, aj, rx, ry, rz, Mttij);

	RPY_Mij_rr_block(ai, aj, rx, ry, rz, Mrrij);

	RPY_Mij_rt_block(ai, aj, rx, ry, rz, Mrtij);

	double* matrix = calloc(12*12, sizeof(double));

	// *matrix = Mi;
	// *(matrix+7) = Mi;
	// *(matrix+14) = Mi;
	// *(matrix+21) = Mj;
	// *(matrix+28) = Mj;
	// *(matrix+35) = Mj;

	// *(matrix+3) = *(Mij);
	// *(matrix+4) = *(Mij+3);
	// *(matrix+5) = *(Mij+4);
	// *(matrix+9) = *(Mij+3);
	// *(matrix+10) = *(Mij+1);
	// *(matrix+11) = *(Mij+5);
	// *(matrix+15) = *(Mij+4);
	// *(matrix+16) = *(Mij+5);
	// *(matrix+17) = *(Mij+2);

	// *(matrix+18) = *(Mij);
	// *(matrix+19) = *(Mij+3);
	// *(matrix+20) = *(Mij+4);
	// *(matrix+24) = *(Mij+3);
	// *(matrix+25) = *(Mij+1);
	// *(matrix+26) = *(Mij+5);
	// *(matrix+30) = *(Mij+4);
	// *(matrix+31) = *(Mij+5);
	// *(matrix+32) = *(Mij+2);

	// inverse(matrix,6);

	// // block 00

	// *R_matrix = *matrix;
	// *(R_matrix+1) = *(matrix+7);
	// *(R_matrix+2) = *(matrix+14);
	// *(R_matrix+3) = *(matrix+6);
	// *(R_matrix+4) = *(matrix+12);
	// *(R_matrix+5) = *(matrix+13);

	// // block 11

	// *(R_matrix+6) = *(matrix+21);
	// *(R_matrix+7) = *(matrix+28);
	// *(R_matrix+8) = *(matrix+35);
	// *(R_matrix+9) = *(matrix+27);
	// *(R_matrix+10) = *(matrix+33);
	// *(R_matrix+11) = *(matrix+34);

	// // block 10

	// *(R_matrix+12) = *(matrix+3);
	// *(R_matrix+13) = *(matrix+10);
	// *(R_matrix+14) = *(matrix+17);
	// *(R_matrix+15) = *(matrix+9);
	// *(R_matrix+16) = *(matrix+15);
	// *(R_matrix+17) = *(matrix+16);

	free(Mttij);

	free(Mrrij);
	
	free(Mrtij);

	free(matrix);
}


static void RPY_2B_RA_matrix(double ai, double aj, double rx, double ry, double rz, double* R_matrix)
{
	double Mi = RPY_Mii_tt_block(ai);

	double Mj = RPY_Mii_tt_block(aj);

	double* Mij = calloc(6, sizeof(double));

	RPY_Mij_tt_block(ai, aj, rx, ry, rz, Mij);

	double* matrix = calloc(6*6, sizeof(double));

	*matrix = Mi;
	*(matrix+7) = Mi;
	*(matrix+14) = Mi;
	*(matrix+21) = Mj;
	*(matrix+28) = Mj;
	*(matrix+35) = Mj;

	*(matrix+3) = *(Mij);
	*(matrix+4) = *(Mij+3);
	*(matrix+5) = *(Mij+4);
	*(matrix+9) = *(Mij+3);
	*(matrix+10) = *(Mij+1);
	*(matrix+11) = *(Mij+5);
	*(matrix+15) = *(Mij+4);
	*(matrix+16) = *(Mij+5);
	*(matrix+17) = *(Mij+2);

	*(matrix+18) = *(Mij);
	*(matrix+19) = *(Mij+3);
	*(matrix+20) = *(Mij+4);
	*(matrix+24) = *(Mij+3);
	*(matrix+25) = *(Mij+1);
	*(matrix+26) = *(Mij+5);
	*(matrix+30) = *(Mij+4);
	*(matrix+31) = *(Mij+5);
	*(matrix+32) = *(Mij+2);

	inverse(matrix,6);

	// block 00

	*R_matrix = *matrix;
	*(R_matrix+1) = *(matrix+7);
	*(R_matrix+2) = *(matrix+14);
	*(R_matrix+3) = *(matrix+6);
	*(R_matrix+4) = *(matrix+12);
	*(R_matrix+5) = *(matrix+13);

	// block 11

	*(R_matrix+6) = *(matrix+21);
	*(R_matrix+7) = *(matrix+28);
	*(R_matrix+8) = *(matrix+35);
	*(R_matrix+9) = *(matrix+27);
	*(R_matrix+10) = *(matrix+33);
	*(R_matrix+11) = *(matrix+34);

	// block 10

	*(R_matrix+12) = *(matrix+3);
	*(R_matrix+13) = *(matrix+10);
	*(R_matrix+14) = *(matrix+17);
	*(R_matrix+15) = *(matrix+9);
	*(R_matrix+16) = *(matrix+15);
	*(R_matrix+17) = *(matrix+16);

	free(Mij);

	free(matrix);
}

// -------------------------------------------------------------------------------

static void Cichocki_2B_R_correction(double* temp_2b, double* result_2b)
{
	*result_2b = ( *temp_2b + *(temp_2b + 6) - 2 * *(temp_2b + 12) ) / 4;
	*(result_2b + 1) = ( *(temp_2b + 1) + *(temp_2b + 7) - 2 * *(temp_2b + 13) ) / 4;
	*(result_2b + 2) = ( *(temp_2b + 2) + *(temp_2b + 8) - 2 * *(temp_2b + 14) ) / 4;
	*(result_2b + 3) = ( *(temp_2b + 3) + *(temp_2b + 9) - 2 * *(temp_2b + 15) ) / 4;
	*(result_2b + 4) = ( *(temp_2b + 4) + *(temp_2b + 10) - 2 * *(temp_2b + 16) ) / 4;
	*(result_2b + 5) = ( *(temp_2b + 5) + *(temp_2b + 11) - 2 * *(temp_2b + 17) ) / 4;
	*(result_2b + 6) = ( *temp_2b + *(temp_2b + 6) - 2 * *(temp_2b + 12) ) / 4;
	*(result_2b + 7) = ( *(temp_2b + 1) + *(temp_2b + 7) - 2 * *(temp_2b + 13) ) / 4;
	*(result_2b + 8) = ( *(temp_2b + 2) + *(temp_2b + 8) - 2 * *(temp_2b + 14) ) / 4;
	*(result_2b + 9) = ( *(temp_2b + 3) + *(temp_2b + 9) - 2 * *(temp_2b + 15) ) / 4;
	*(result_2b + 10) = ( *(temp_2b + 4) + *(temp_2b + 10) - 2 * *(temp_2b + 16) ) / 4;
	*(result_2b + 11) = ( *(temp_2b + 5) + *(temp_2b + 11) - 2 * *(temp_2b + 17) ) / 4;
	*(result_2b + 12) = ( 2 * *(temp_2b + 12) - *temp_2b - *(temp_2b + 6) ) / 4;
	*(result_2b + 13) = ( 2 * *(temp_2b + 13) - *(temp_2b + 1) - *(temp_2b + 7) ) / 4;
	*(result_2b + 14) = ( 2 * *(temp_2b + 14) - *(temp_2b + 2) - *(temp_2b + 8) ) / 4;
	*(result_2b + 15) = ( 2 * *(temp_2b + 15) - *(temp_2b + 3) - *(temp_2b + 9) ) / 4;
	*(result_2b + 16) = ( 2 * *(temp_2b + 16) - *(temp_2b + 4) - *(temp_2b + 10) ) / 4;
	*(result_2b + 17) = ( 2 * *(temp_2b + 17) - *(temp_2b + 5) - *(temp_2b + 11) ) / 4;
}

// -------------------------------------------------------------------------------

static double JO_XA11_term(double s, double l)
{
	double XA11_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	int m, m1;

	double mult;

	XA11_value += JO_XAg_polynomial(l, 1)/s2;

	XA11_value -= JO_XAg_polynomial(l, 2)*logs2;

	XA11_value -= JO_XAg_polynomial(l, 3)*s2*logs2;

	XA11_value += JO_XAf_polynomial(l, 0) - JO_XAg_polynomial(l, 1);

	for (m = 1; m < 12; m++)
	{
		if (m % 2 == 1)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -2;
			}
			else
			{
				m1 = m - 2;
			}
			
			mult = pow(2/s, m);

			XA11_value += mult * ( pow(2, -m) * pow(1+l, -m) * JO_XAf_polynomial(l, m) - JO_XAg_polynomial(l, 1) );

			XA11_value += mult * ( 4.0 / ( m * m1 ) * JO_XAg_polynomial(l, 3) - 2.0 / m * JO_XAg_polynomial(l, 2) );
		}
	}

	return XA11_value;
}

// -------------------------------------------------------------------------------

static double JO_YA11_term(double s, double l)
{
	double YA11_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	int m, m1;

	double mult;

	YA11_value -= JO_YAg_polynomial(l, 2)*logs2;

	YA11_value -= JO_YAg_polynomial(l, 3)*s2*logs2;

	YA11_value += JO_YAf_polynomial(l, 0);

	for (m = 1; m < 12; m++)
	{
		if (m % 2 == 1)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -2;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2/s, m);

			YA11_value += mult * ( pow(2, -m)*pow(1+l, -m)*JO_YAf_polynomial(l, m) - 2.0/m*JO_YAg_polynomial(l, 2) );

			YA11_value += mult*4.0/(m*m1)*JO_YAg_polynomial(l, 3);
		}
	}

	return YA11_value;
}

// -------------------------------------------------------------------------------

static double JO_XA12_term(double s, double l)
{
	double XA12_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs = log((s + 2)/(s - 2));

	int m, m1;

	double mult;

	XA12_value += 2.0/s*JO_XAg_polynomial(l, 1)/s2;

	XA12_value += JO_XAg_polynomial(l, 2)*logs;

	XA12_value += JO_XAg_polynomial(l, 3)*s2*logs + 4*JO_XAg_polynomial(l, 3)/s;

	for (m = 1; m < 12; m++)
	{
		if (m % 2 == 0)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -2;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2/s, m);

			XA12_value += mult * ( pow(2, -m)*pow(1 + l, -m)*JO_XAf_polynomial(l, m) - JO_XAg_polynomial(l, 1) );

			XA12_value += mult * ( 4.0/(m*m1) * JO_XAg_polynomial(l, 3) - 2.0 / m * JO_XAg_polynomial(l, 2) );
		}

	}

	return -XA12_value;
}

// -------------------------------------------------------------------------------

static double JO_YA12_term(double s, double l)
{
	double YA12_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs = log((s + 2)/(s - 2));

	int m, m1;

	double mult;

	YA12_value += JO_YAg_polynomial(l, 2)*logs;

	YA12_value += JO_YAg_polynomial(l, 3)*s2*logs;

	YA12_value += 4*JO_YAg_polynomial(l, 3)/s;

	for (m = 1; m < 12; m++)
	{
		if (m % 2 == 0)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -1;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2/s, m);

			YA12_value += mult*(pow(2, -m)*pow(1+l, -m)*JO_YAf_polynomial(l, m) - 2.0/m*JO_YAg_polynomial(l, 2));

			YA12_value += mult*(4.0/(m*m1)*JO_YAg_polynomial(l, 3));
		}
	}

	return -YA12_value;
}

// -------------------------------------------------------------------------------

static double JO_YB11_term(double s, double l)
{
	double YB11_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs = log((s + 2)/(s - 2));

	// double logs2 = log(s2);

	int m, m1;

	double mult;

	YB11_value += JO_YBg_polynomial(l, 2)*logs;

	YB11_value += JO_YBg_polynomial(l, 3)*s2*logs;

	YB11_value += 4.0 * JO_YBg_polynomial(l, 3) / s;

	for (m = 1; m < 12; m++)
	{
		if (m % 2 == 0)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -2;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2/s, m);

			YB11_value += mult * ( pow(2, -m)*pow(1+l, -m)*JO_YBf_polynomial(l, m) - 2.0/m*JO_YBg_polynomial(l, 2) );

			YB11_value += mult*4.0/(m*m1)*JO_YBg_polynomial(l, 3);
		}
	}

	return YB11_value;
}

// -------------------------------------------------------------------------------

static double JO_YB12_term(double s, double l)
{
	double YB12_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	int m, m1;

	double mult;

	YB12_value -= JO_YBg_polynomial(l, 2)*logs2;

	YB12_value -= JO_YBg_polynomial(l, 3)*s2*logs2;

	// printf("partial efore summation C: %lf", YB12_value);

	for (m = 2; m < 12; m++)
	{
		if (m % 2 == 1)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -2;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2.0/s, m);

			YB12_value += mult*(pow(2, -m)*pow(1+l, -m)*JO_YBf_polynomial(l, m) - 2.0/m*JO_YBg_polynomial(l, 2));

			YB12_value += mult*(4.0/(m*m1)*JO_YBg_polynomial(l, 3));
		}
	}

	// printf("partial after summation C: %lf", YB12_value);

	return -YB12_value;
}

// -------------------------------------------------------------------------------

static double JO_XC11_term(double s, double l)
{
	double XC11_value = 1.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	double l2 = l * l / (1.0 + l);

	int k;

	double mult;

	XC11_value += l2 / 2 * logs2;

	// printf("C: %lf'n", XC11_value);

	XC11_value += l2 / s * log((s+2.0)/(s-2.0));

	// printf("C: %lf'n", XC11_value);

	for (k = 1; k < 6; k++)
	{
		mult = pow(s, -2*k);

		XC11_value += mult * ( pow(1+l, -2*k) * JO_XCf_polynomial(l, 2*k) );

		XC11_value -= mult * ( pow(2, 2*k+1) / k / (2 * k - 1) * l2 / 4 );
	}

	// printf("C: %lf'n", XC11_value);

	return XC11_value;
}

// -------------------------------------------------------------------------------

static double JO_YC11_term(double s, double l)
{
	double YC11_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	int m, m1;

	double mult;

	YC11_value -= JO_YCg_polynomial(l, 2) * log(s2);

	YC11_value -= JO_YCg_polynomial(l, 3) * s2 * log(s2);

	YC11_value += JO_YCf_polynomial(l, 0);

	for (m = 2; m < 12; m++)
	{
		if (m % 2 == 1)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -2;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2.0/s, m);

			// printf("fm C: %lf\ng2 C: %lf\ng3 C: %lf\n", JO_YCf_polynomial(l, m), JO_YCg_polynomial(l, 2), JO_YCg_polynomial(l, 3));

			YC11_value += mult*(pow(2.0, -m)*pow(1+l, -m)*JO_YCf_polynomial(l, m) - 2.0/m*JO_YCg_polynomial(l, 2));

			YC11_value += mult*(4.0/(m*m1)*JO_YCg_polynomial(l, 3));

			// printf("partial YC11 C: %lf\n", YC11_value);
		}
	}

	return YC11_value;
}

// -------------------------------------------------------------------------------

static double JO_XC12_term(double s, double l)
{
	double XC12_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	double l2 = l * l / (1.0 + l);

	double l3 = 1.0 / (1.0 + l) / (1.0 + l) / (1.0 + l);

	int k;

	double mult;

	XC12_value += 4.0 * l2 * l3 *log((s+2.0)/(s-2.0));

	XC12_value += 8.0 * l2 * l3 / s * logs2;

	// errata
	XC12_value -= 16.0 * l2 * l3 / s;

	for (k = 1; k < 6; k++)
	{
		mult = pow(s, -2*k - 1) * 8.0 * l3;

		XC12_value -= mult * ( pow(1+l, -2*k - 1) * JO_XCf_polynomial(l, 2*k + 1) );

		// errata
		XC12_value += mult * ( pow(2, 2*k+2) / k / (2 * k + 1) * l2 / 4 );
	}

	return 1.0 / 8.0 * pow(1+l, 3) * XC12_value;
}

// -------------------------------------------------------------------------------

static double JO_YC12_term(double s, double l)
{
	double YC12_value = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	int m, m1;

	double mult;

	YC12_value += JO_YCg_polynomial(l, 4) * log((s+2.0)/(s-2.0));

	YC12_value += JO_YCg_polynomial(l, 5) * s2 * log((s+2.0)/(s-2.0));

	YC12_value += 4.0 * JO_YCg_polynomial(l, 5) / s;

	for (m = 1; m < 12; m++)
	{
		if (m % 2 == 0)
		{
			continue;
		}
		else
		{
			if (m == 2)
			{
				m1 = -1;
			}
			else
			{
				m1 = m - 2;
			}

			mult = pow(2/s, m);

			YC12_value += mult*(pow(2, -m)*pow(1+l, -m)*JO_YCf_polynomial(l, m) - 2.0/m*JO_YCg_polynomial(l, 4));

			YC12_value += mult*(4.0/(m*m1)*JO_YCg_polynomial(l, 5));
		}
	}

	return YC12_value;
}

// -------------------------------------------------------------------------------

static double JO_XAf_polynomial(double l, int degree)
{
	double Xf_value;
	double l2, l3, l4, l5, l6, l7, l8, l9, l10;

	switch(degree)
	{
		case 0:
			Xf_value = 1;
			break;
		case 1:
			Xf_value = 3*l;
			break;
		case 2:
			Xf_value = 9*l;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l; 
			Xf_value = -4*l + 27*l2 - 4*l3;
			break;
		case 4:
			l2 = l*l;
			l3 = l2*l;
			Xf_value = -24*l + 81*l2 + 36*l3;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			Xf_value = 72*l2 + 243*l3 + 72*l4;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			Xf_value = 16*l + 108*l2 + 281*l3 + 648*l4 + 144*l5;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			Xf_value = 288*l2 + 1620*l3 + 1515*l4 + 1620*l5 + 288*l6;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			Xf_value = 576*l2 + 4848*l3 + 5409*l4 + 4524*l5 + 3888*l6 + 576*l7;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			Xf_value = 1152*l2 + 9072*l3 + 14752*l4 + 26163*l5 + 14752*l6 + 9072*l7 + 1152*l8;
			break;
		case 10:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			l9 = l5*l4;
			Xf_value = 2304*l2 + 20736*l3 + 42804*l4 + 115849*l5 + 76176*l6 + 39264*l7 + 20736*l8 + 2304*l9;
			break;
		case 11:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			l9 = l5*l4;
			l10 = l5*l5;
			Xf_value = 4608*l2 + 46656*l3 + 108912*l4 + 269100*l5 + 319899*l6 + 269100*l7 + 108912*l8 + 46656*l9 + 4608*l10;
			break;
	}

	return Xf_value;
}

// -------------------------------------------------------------------------------

static double JO_YAf_polynomial(double l, int degree)
{
	double Yf_value;
	double l2, l3, l4, l5, l6, l7, l8, l9, l10;

	switch(degree)
	{
		case 0:
			Yf_value = 1;
			break;
		case 1:
			Yf_value = 3.0/2*l;
			break;
		case 2:
			Yf_value = 9.0/4*l;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l;
			Yf_value = 2*l + 3.375*l2 + 2*l3;
			break;
		case 4:
			l2 = l*l;
			l3 = l2*l;
			Yf_value = 6*l + 5.0625*l2 + 18*l3;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			Yf_value = 31.5*l2 + 7.59375*l3 + 31.5*l4;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			Yf_value = 4*l + 54*l2 + 19.390625*l3 + 81*l4 + 72*l5;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			Yf_value = 144*l2 + 131.625*l3 + 149.0859375*l4 + 131.625*l5 + 144*l6;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			Yf_value = 279*l2 + 532.625*l3 + 493.62890625*l4 - 14.625*l5 + 648*l6 + 288*l7;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			Yf_value = 576*l2 + 1134*l3 + 1888.84375*l4 + 1496.443359375*l5 + 1888.84375*l6 + 1134*l7 + 576*l8;
			break;
		case 10:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			l9 = l5*l4;
			Yf_value = 1152*l2 + 1964.25*l3 + 6155.4375*l4 + 10301.1650390625*l5 + 8452.125*l6 - 175.5*l7 + 3888*l8 + 1152*l9;
			break;
		case 11:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			l9 = l5*l4;
			l10 = l5*l5;
			Yf_value = 2304*l2 + 7128*l3 + 11035.5*l4 + 21441.4453125*l5 + 46486.24755859375*l6 + 21441.4453125*l7 + 11035.5*l8 + 7128*l9 + 2304*l10;
			break;
	}

	return Yf_value;
}

// -------------------------------------------------------------------------------

static double JO_XAg_polynomial(double l, int degree)
{
	double Xg_value;
	double l2 = l*l;
	double plus = 1+l;
	double plus3 = plus * plus * plus;
	double l3, l4;

	switch(degree)
	{
		case 1:
			Xg_value = 2*l2/plus3;
			break;
		case 2:
			Xg_value = 0.2*l*(1 + 7*l + l2)/plus3;
			break;
		case 3:
			l3 = l2*l;
			l4 = l2*l2;
			Xg_value = 1.0/42*(1 + 18*l - 29*l2 + 18*l3 + l4)/plus3;
			break;
	}

	return Xg_value;
}

// -------------------------------------------------------------------------------

static double JO_YAg_polynomial(double l, int degree)
{
	double Yg_value;
	double l2 = l*l;
	double plus = 1+l;
	double plus3 = plus * plus * plus;
	double l3, l4;

	switch(degree)
	{
		case 2:
			Yg_value = 4.0/15*l*(2 + l + 2*l2)/plus3;
			break;
		case 3:
			l3 = l2*l;
			l4 = l2*l2;
			Yg_value = 2.0/375*(16 - 45*l + 58*l2 - 45*l3 + 16*l4)/plus3;
	}

	return Yg_value;
}

// -------------------------------------------------------------------------------

static double JO_YBf_polynomial(double l, int degree)
{
	double Yf_value;
	double l2, l3, l4, l5, l6, l7, l8, l9;

	switch(degree)
	{
		case 0:
			Yf_value = 0;
			break;
		case 1:
			Yf_value = 0;
			break;
		case 2:
			Yf_value = -6.0*l;
			break;
		case 3:
			Yf_value = -9.0*l;
			break;
		case 4:
			l2 = l*l;
			Yf_value = -13.5*l2;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			Yf_value = -12.0*l - 20.25*l2 - 36.0*l3;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			Yf_value = -108.0*l2 - 30.375*l3 - 72.0*l4;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			Yf_value = -189.0*l2 - 525.5625*l3 - 243.0*l4 - 144.0*l5;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			Yf_value = -432.0*l2 - 486.0*l3 - 2420.34375*l4 - 405.0*l5 - 288.0*l6;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			Yf_value = -864.0*l2 - 789.75*l3 - 4422.515625*l4 - 7631.25*l5 - 1620.0*l6 - 576.0*l7;
			break;
		case 10:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			Yf_value = -1728.0*l2 - 3888.0*l3 - 14888.25*l4 - 8793.7734375*l5 - 22002.0*l6 - 2916.0*l7 - 1152.0*l8;
			break;
		case 11:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			l9 = l5*l4;
			Yf_value = -3456.0*l2 - 6804.0*l3 - 38405.0625*l4 - 17888.6601563*l5 - 33542.4375*l6 - 73989.0*l7 - 9072.0*l8 - 2304.0*l9;
			break;
	}

	return Yf_value;
}

// -------------------------------------------------------------------------------

static double JO_YBg_polynomial(double l, int degree)
{
	double Yg_value;
	double l2, l3;
	double plus = 1+l;
	double plus2 = plus * plus;

	switch(degree)
	{
		case 2:
			Yg_value = -1.0/5.0*l*(4 + l)/plus2;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l;
			Yg_value = -1.0/250.0*(32.0 - 33.0*l + 83.0*l2 + 43.0*l3)/plus2;
	}

	return Yg_value;
}

// -------------------------------------------------------------------------------

static double JO_XCf_polynomial(double l, int degree)
{
	double Xf_value;
	double l2, l3, l4, l5, l6, l7, l8;

	switch(degree)
	{
		case 0:
			Xf_value = 1.0;
			break;
		case 1:
			Xf_value = 0;
			break;
		case 2:
			Xf_value = 0;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l;
			Xf_value = 8.0*l3;
			break;
		case 4:
			Xf_value = 0;
			break;
		case 5:
			Xf_value = 0;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			Xf_value = 64.0*l3;
			break;
		case 7:
			Xf_value = 0;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			Xf_value = 768.0*l5;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			Xf_value = 512.0*l6;
			break;
		case 10:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			Xf_value = 6144.0*l7;
			break;
		case 11:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			Xf_value = 6144.0*(l6 + l8);
			break;
	}

	return Xf_value;
}

// -------------------------------------------------------------------------------

static double JO_YCf_polynomial(double l, int degree)
{
	double Yf_value;
	double l2, l3, l4, l5, l6, l7, l8, l9, l10;

	switch(degree)
	{
		case 0:
			Yf_value = 1.0;
			break;
		case 1:
			Yf_value = 0;
			break;
		case 2:
			Yf_value = 0;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l;
			Yf_value = 4.0*l3;
			break;
		case 4:
			Yf_value = 12.0*l;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			Yf_value = 18.0*l4;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			Yf_value = 27.0*l2 + 256.0*l3;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			Yf_value = 72.0*l4 + 40.5*l5 + 72.0*l6;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			Yf_value = 216.0*l2 + 60.75*l3 + 216.0*l4 + 2496.0*l5;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			Yf_value = 288.0*l4 + 486.0*l5 - 804.875*l6 + 486.0*l7 + 288.0*l8;
			break;
		case 10:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			Yf_value = 864.0*l2 + 972.0*l3 + 9448.6875*l4 + 972.0*l5 + 1296.0*l6 + 18432.0*l7;
			break;
		case 11:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			l9 = l5*l4;
			l10 = l5*l5;
			Yf_value = 1152.0*l4 + 3240.0*l5 - 5473.5*l6 + 16189.03125*l7 - 5473.5*l8 + 3240.0*l9 + 1152.0*l10;
			break;
	}

	return Yf_value;
}

// -------------------------------------------------------------------------------

static double JO_YCg_polynomial(double l, int degree)
{
	double Yg_value;
	double l2;
	double plus = 1+l;
	double plus2 = plus * plus;

	switch(degree)
	{
		case 2:
			Yg_value = 2.0 / 5.0 * l / plus;
			break;
		case 3:
			l2 = l*l;
			Yg_value = 1.0/125.0*(8.0 + 6.0*l + 33.0*l2)/plus;
			break;
		case 4:
			l2 = l*l;
			Yg_value = 4.0/5.0*l2/plus2/plus2;
			break;
		case 5:
			// errata
			l2 = l*l;
			Yg_value = 1.0/125.0*l*( 43.0 - 24.0*l + 43.0*l2 )/(plus2*plus2);
			break;
	}

	return Yg_value;
}

// -------------------------------------------------------------------------------

static void inverse(double* matrix, int matrix_dimension)
{
    int *IPIV = calloc(matrix_dimension, sizeof(int));
    int LWORK = matrix_dimension * matrix_dimension;
    double *WORK = calloc(LWORK, sizeof(double));
    int INFO;

    dgetrf_(&matrix_dimension, &matrix_dimension, matrix, &matrix_dimension, IPIV, &INFO);
    dgetri_(&matrix_dimension, matrix, &matrix_dimension, IPIV, WORK, &LWORK, &INFO);

    free(IPIV);
    free(WORK);
}

// -------------------------------------------------------------------------------

static inline int results_position(int i, int j, int N)
{
	int position = i + j*N;

	return position - j*(j+1)/2;
}

// -------------------------------------------------------------------------------