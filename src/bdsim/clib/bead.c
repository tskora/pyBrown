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

void pointer_pbc_matrix(double* positions, int number_of_beads, double box_length, double* pointers)
{
	register int i = 0;
	register int j = 0;

	double* shifted_pointers;

	double* temp;

	for (j = 0; j < number_of_beads; j++)
	{
		for (i = j + 1; i < number_of_beads; i++)
		{
			temp = calloc(3, sizeof(double));

			*temp = *(positions+3*i) - *(positions+3*j);

			while (*temp >= box_length/2.0)
			{
				*temp -= box_length;
			}
			while (*temp <= -box_length/2.0)
			{
				*temp += box_length;
			}

			*(temp+1) = *(positions+3*i+1) - *(positions+3*j+1);

			while (*(temp+1) >= box_length/2.0)
			{
				*(temp+1) -= box_length;
			}
			while (*(temp+1) <= -box_length/2.0)
			{
				*(temp+1) += box_length;
			}

			*(temp+2) = *(positions+3*i+2) - *(positions+3*j+2);

			while (*(temp+2) >= box_length/2.0)
			{
				*(temp+2) -= box_length;
			}
			while (*(temp+2) <= -box_length/2.0)
			{
				*(temp+2) += box_length;
			}

			shifted_pointers = pointers + 3*i + 3*j*number_of_beads;

			*shifted_pointers = *temp;

			*(shifted_pointers+1) = *(temp+1);

			*(shifted_pointers+2) = *(temp+2);

			shifted_pointers = pointers + 3*j + 3*i*number_of_beads;

			*shifted_pointers = -*temp;

			*(shifted_pointers+1) = -*(temp+1);

			*(shifted_pointers+2) = -*(temp+2);

			free(temp);

		}
	}
}

// -------------------------------------------------------------------------------

int check_overlaps(double* positions, double *as, int number_of_beads, double box_length, double overlap_treshold)
{
	register int i = 0;
	register int j = 0;

	double rx, ry, rz, radii_sum, radii_sum_pbc;

	for (j = 0; j < number_of_beads; j++)
	{
		for (i = j + 1; i < number_of_beads; i++)
		{
			rx = *(positions + 3*i) - *(positions + 3*j);

			ry = *(positions + 3*i + 1) - *(positions + 3*j + 1);

			rz = *(positions + 3*i + 2) - *(positions + 3*j + 2);

			radii_sum = *(as+i) + *(as+j);

			radii_sum_pbc = box_length - radii_sum;

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
	while (rx <= -box_length/2.0)
	{
		rx += box_length;
	}
	while (ry >= box_length/2.0)
	{
		ry -= box_length;
	}
	while (ry <= -box_length/2.0)
	{
		ry += box_length;
	}
	while (rz >= box_length/2.0)
	{
		rz -= box_length;
	}
	while (rz <= -box_length/2.0)
	{
		rz += box_length;
	}


	return sqrt( rx*rx + ry*ry + rz*rz );
}

// -------------------------------------------------------------------------------