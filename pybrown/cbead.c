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

static int overlap_pbc(double ai, double aj, double rx, double ry, double rz, double box_length, double overlap_treshold);

static double distance_pbc(double rx, double ry, double rz, double box_length);

// -------------------------------------------------------------------------------

void pointer_pbc_matrix(double* positions, int number_of_beads, double box_length, double* pointers, int dims)
{
	register int i = 0;
	register int j = 0;

	double* shifted_pointers;

	double* temp;

	for (j = 0; j < number_of_beads; j++)
	{
		for (i = j + 1; i < number_of_beads; i++)
		{
			temp = calloc(dims, sizeof(double));

			*temp = *(positions+dims*i) - *(positions+dims*j);

			while (*temp >= box_length/2.0)
			{
				*temp -= box_length;
			}
			while (*temp < -box_length/2.0)
			{
				*temp += box_length;
			}

			*(temp+1) = *(positions+dims*i+1) - *(positions+dims*j+1);

			while (*(temp+1) >= box_length/2.0)
			{
				*(temp+1) -= box_length;
			}
			while (*(temp+1) < -box_length/2.0)
			{
				*(temp+1) += box_length;
			}

			if (dims == 3)
			{
				*(temp+2) = *(positions+dims*i+2) - *(positions+dims*j+2);

				while (*(temp+2) >= box_length/2.0)
				{
					*(temp+2) -= box_length;
				}
				while (*(temp+2) < -box_length/2.0)
				{
					*(temp+2) += box_length;
				}
			}

			shifted_pointers = pointers + dims*i + dims*j*number_of_beads;

			*shifted_pointers = *temp;

			*(shifted_pointers+1) = *(temp+1);

			if (dims == 3)
			{
				*(shifted_pointers+2) = *(temp+2);
			}

			shifted_pointers = pointers + dims*j + dims*i*number_of_beads;

			*shifted_pointers = -*temp;

			*(shifted_pointers+1) = -*(temp+1);

			if (dims == 3)
			{
				*(shifted_pointers+2) = -*(temp+2);
			}

			free(temp);

		}
	}
}

// -------------------------------------------------------------------------------

void pointer_immobile_pbc_matrix(double* positions_mobile, double* positions_immobile, int number_of_mobile, int number_of_immobile, double box_length, double* pointers, int dims)
{
	register int i = 0;
	register int j = 0;

	double* shifted_pointers;

	double* temp;

	for (j = 0; j < number_of_immobile; j++)
	{
		for (i = 0; i < number_of_mobile; i++)
		{
			temp = calloc(dims, sizeof(double));

			*temp = *(positions_immobile+dims*j) - *(positions_mobile+dims*i);

			while (*temp >= box_length/2.0)
			{
				*temp -= box_length;
			}
			while (*temp < -box_length/2.0)
			{
				*temp += box_length;
			}

			*(temp+1) = *(positions_immobile+dims*j+1) - *(positions_mobile+dims*i+1);

			while (*(temp+1) >= box_length/2.0)
			{
				*(temp+1) -= box_length;
			}
			while (*(temp+1) < -box_length/2.0)
			{
				*(temp+1) += box_length;
			}

			if (dims == 3)
			{
				*(temp+2) = *(positions_immobile+dims*j+2) - *(positions_mobile+dims*i+2);

				while (*(temp+2) >= box_length/2.0)
				{
					*(temp+2) -= box_length;
				}
				while (*(temp+2) <= -box_length/2.0)
				{
					*(temp+2) += box_length;
				}
			}

			shifted_pointers = pointers + dims*j + dims*i*number_of_immobile;

			*shifted_pointers = *temp;

			*(shifted_pointers+1) = *(temp+1);

			if (dims == 3)
			{
				*(shifted_pointers+2) = *(temp+2);
			}

			free(temp);
		}
	}
}

// -------------------------------------------------------------------------------

static int is_there_bond_between(int* connection_matrix, int number_of_beads, int i, int j)
{
	return *(connection_matrix + i + number_of_beads*j);
}

// -------------------------------------------------------------------------------

int check_overlaps(double* positions, double *as, int number_of_beads, double box_length, double overlap_treshold, int* connection_matrix, int dims)
{
	register int i = 0;
	register int j = 0;

	double rx, ry, rz, radii_sum, radii_sum_pbc;

	for (j = 0; j < number_of_beads; j++)
	{
		for (i = j + 1; i < number_of_beads; i++)
		{
			rx = *(positions + dims*i) - *(positions + dims*j);

			ry = *(positions + dims*i + 1) - *(positions + dims*j + 1);

			if (dims == 3)
			{
				rz = *(positions + dims*i + 2) - *(positions + dims*j + 2);
			}
			else
			{
				rz = 0.0;
			}

			radii_sum = *(as+i) + *(as+j);

			radii_sum_pbc = box_length - radii_sum;

			if ( is_there_bond_between(connection_matrix, number_of_beads, j, i) )
			{
				continue;
			}

			if ( ( (rx > radii_sum) && (rx < radii_sum_pbc) ) || ( (rx < -radii_sum) && (rx > -radii_sum_pbc) ) )
			{
				continue;
			}
			else if ( ( (ry > radii_sum) && (ry < radii_sum_pbc) ) || ( (ry < -radii_sum) && (ry > -radii_sum_pbc) ) )
			{
				continue;
			}
			else if ( ( (rz > radii_sum) && (rz < radii_sum_pbc) ) || ( (rz < -radii_sum) && (rz > -radii_sum_pbc) ) )
			{
				continue;
			}
			else
			{
				if (overlap_pbc(*(as+i), *(as+j), rx, ry, rz, box_length, overlap_treshold))
				{
					return 1;
				}
			}
		}
	}

	return 0;
}

// -------------------------------------------------------------------------------

static int overlap_pbc(double ai, double aj, double rx, double ry, double rz, double box_length, double overlap_treshold)
{
	double dist = distance_pbc(rx, ry, rz, box_length);

	return ( dist <= ai + aj + overlap_treshold );
}

// -------------------------------------------------------------------------------

static double distance_pbc(double rx, double ry, double rz, double box_length)
{
	while (rx >= box_length/2.0)
	{
		rx -= box_length;
	}
	while (rx < -box_length/2.0)
	{
		rx += box_length;
	}
	while (ry >= box_length/2.0)
	{
		ry -= box_length;
	}
	while (ry < -box_length/2.0)
	{
		ry += box_length;
	}
	while (rz >= box_length/2.0)
	{
		rz -= box_length;
	}
	while (rz < -box_length/2.0)
	{
		rz += box_length;
	}


	return sqrt( rx*rx + ry*ry + rz*rz );
}

// -------------------------------------------------------------------------------