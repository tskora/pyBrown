// pyBD is a Brownian and Stokesian dynamics simulation tool
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// -------------------------------------------------------------------------------

void O(double rx, double ry, double rz, double* answer)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	*answer = ( 1 + rx*rx / dist2 ) / dist;

	*(answer+1) = ( 1 + ry*ry / dist2 ) / dist;

	*(answer+2) = ( 1 + rz*rz / dist2 ) / dist;

	*(answer+3) = rx*ry / dist2 / dist;

	*(answer+4) = rx*rz / dist2 / dist;

	*(answer+5) = ry*rz / dist2 / dist;
}

// -------------------------------------------------------------------------------

void Q(double rx, double ry, double rz, double* answer)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double dist3 = dist * dist2;

	*answer = ( 1 - 3 * rx*rx / dist2 ) / dist3;

	*(answer+1) = ( 1 - 3 * ry*ry / dist2 ) / dist3;

	*(answer+2) = ( 1 - 3 * rz*rz / dist2 ) / dist3;

	*(answer+3) = -3 * rx*ry / dist2 / dist3;

	*(answer+4) = -3 * rx*rz / dist2 / dist3;

	*(answer+5) = -3 * ry*rz / dist2 / dist3;
}

// -------------------------------------------------------------------------------

void Oii(double a, double box_length, double alpha, int m, int n, double* answer)
{
	double mlen, mlen2, nlen, nlen2, mult;

	int mx, my, mz, nx, ny, nz, i;

	double* values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	for (mx = -m; mx <= m; mx++)
	{
		for (my = -m; my <= m; my++)
		{
			for (mz = -m; mz <= m; mz++)
			{
				if (abs(mx)+abs(my)+abs(mz)<=m)
				{
					if (!(mx==0 && my==0 && mz==0))
					{
						mlen2 = mx*mx + my*my + mz*mz;

						mlen = sqrt(mlen2);

						O(mx, my, mz, values);

						mult = erfc( alpha*mlen );

						for (i = 0; i < 6; i++)
						{
							*(answer+i) += mult * *(values+i);
						}

						mult = 2.0 * alpha / sqrt_pi * exp( - alpha2 * mlen2 );

						*(answer) += mult * mx * mx / mlen2;
						*(answer+1) += mult * my * my / mlen2;
						*(answer+2) += mult * mz * mz / mlen2;
						*(answer+3) += mult * mx * my / mlen2;
						*(answer+4) += mult * mx * mz / mlen2;
						*(answer+5) += mult * my * mz / mlen2;
					}
				}
			}
		}
	}

	for (nx = -n; nx <= n; nx++)
	{
		for (ny = -n; ny <= n; ny++)
		{
			for (nz = -n; nz <= n; nz++)
			{
				if (abs(nx)+abs(ny)+abs(nz)<=n)
				{
					if (!(nx==0 && ny==0 && nz==0))
					{
						nlen2 = nx*nx + ny*ny + nz*nz;

						nlen = sqrt(nlen2);

						mult = 2.0 / M_PI / nlen2 * exp( -M_PI * M_PI * nlen2 / alpha / alpha);

						*(answer) += mult * ( 1.0 - ( 1.0 + M_PI * M_PI * nlen2 / alpha2 ) * nx * nx / nlen2 );					
						*(answer+1) += mult * ( 1.0 - ( 1.0 + M_PI * M_PI * nlen2 / alpha2 ) * ny * ny / nlen2 );					
						*(answer+2) += mult * ( 1.0 - ( 1.0 + M_PI * M_PI * nlen2 / alpha2 ) * nz * nz / nlen2 );					
						*(answer+3) += mult * ( - ( 1.0 + M_PI * M_PI * nlen2 / alpha2 ) * nx * ny / nlen2 );				
						*(answer+4) += mult * ( - ( 1.0 + M_PI * M_PI * nlen2 / alpha2 ) * nx * nz / nlen2 );				
						*(answer+5) += mult * ( - ( 1.0 + M_PI * M_PI * nlen2 / alpha2 ) * ny * nz / nlen2 );				
					}
				}
			}
		}
	}

	*(answer) -= 3.0 * alpha * a / 2.0 / sqrt_pi / box_length;
	*(answer+1) -= 3.0 * alpha * a / 2.0 / sqrt_pi / box_length;
	*(answer+2) -= 3.0 * alpha * a / 2.0 / sqrt_pi / box_length;
}

// -------------------------------------------------------------------------------

void Qii(double a, double box_length, double alpha, int m, int n, double* answer)
{
	double mlen, mlen2, nlen, nlen2, mult;

	int mx, my, mz, nx, ny, nz, i;

	double* values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	double alpha3 = alpha * alpha2;

	double a3 = a * a * a;

	double box_length3 = box_length * box_length * box_length;

	for (mx = -m; mx <= m; mx++)
	{
		for (my = -m; my <= m; my++)
		{
			for (mz = -m; mz <= m; mz++)
			{
				if (abs(mx)+abs(my)+abs(mz)<=m)
				{
					if (!(mx==0 && my==0 && mz==0))
					{
						mlen2 = mx*mx + my*my + mz*mz;

						mlen = sqrt(mlen2);

						Q(mx, my, mz, values);

						mult = erfc( alpha * mlen ) + 2 * alpha / sqrt_pi * mlen * exp( -alpha2 * mlen2 );

						for (i = 0; i < 6; i++)
						{
							*(answer+i) += mult * *(values+i);
						}

						mult = 4 * alpha3 / sqrt_pi * exp(-alpha2*mlen2);

						*(answer) -= mult * mx * mx / mlen2;
						*(answer+1) -= mult * my * my / mlen2;
						*(answer+2) -= mult * mz * mz / mlen2;
						*(answer+3) -= mult * mx * my / mlen2;
						*(answer+4) -= mult * mx * mz / mlen2;
						*(answer+5) -= mult * my * mz / mlen2;
					}
				}
			}
		}
	}

	for (nx = -n; nx <= n; nx++)
	{
		for (ny = -n; ny <= n; ny++)
		{
			for (nz = -n; nz <= n; nz++)
			{
				if (abs(nx)+abs(ny)+abs(nz)<=n)
				{
					if (!(nx==0 && ny==0 && nz==0))
					{
						nlen2 = nx*nx + ny*ny + nz*nz;

						nlen = sqrt(nlen2);

						mult = 4 * M_PI * exp( -M_PI * M_PI * nlen2 / alpha2 );

						*(answer) += mult * nx * nx / nlen2;
						*(answer+1) += mult * ny * ny / nlen2;
						*(answer+2) += mult * nz * nz / nlen2;
						*(answer+3) += mult * nx * ny / nlen2;
						*(answer+4) += mult * nx * nz / nlen2;
						*(answer+5) += mult * ny * nz / nlen2;
					}
				}
			}
		}
	}

	*(answer) -= 1.0 / 3.0 / sqrt_pi * alpha3 * a3 / box_length3;
	*(answer+1) -= 1.0 / 3.0 / sqrt_pi * alpha3 * a3 / box_length3;
	*(answer+2) -= 1.0 / 3.0 / sqrt_pi * alpha3 * a3 / box_length3;	
}

// -------------------------------------------------------------------------------

void Oij(double sigmax, double sigmay, double sigmaz, double alpha, int m, int n, double* answer)
{
	double mslen, mslen2, nlen, nlen2, msx, msy, msz, nsdot, mult;

	int mx, my, mz, nx, ny, nz, i;

	double* values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double sqrt_pi = sqrt(M_PI);

	for (mx = -m; mx <= m; mx++)
	{
		for (my = -m; my <= m; my++)
		{
			for (mz = -m; mz <= m; mz++)
			{
				if (abs(mx)+abs(my)+abs(mz)<=m)
				{
					msx = mx + sigmax;
					msy = my + sigmay;
					msz = mz + sigmaz;

					mslen2 = msx*msx + msy*msy + msz*msz;

					mslen = sqrt(mslen2);

					mult = erfc( alpha * mslen );

					O(msx, msy, msz, values);

					for (i = 0; i < 6; i++)
					{
						*(answer+i) += mult * *(values+i);
					}

					mult = 2 * alpha / sqrt_pi * exp( -alpha2 * mslen2 );

					*(answer) += mult * msx * msx / mslen2;
					*(answer+1) += mult * msy * msy / mslen2;
					*(answer+2) += mult * msz * msz / mslen2;
					*(answer+3) += mult * msx * msy / mslen2;
					*(answer+4) += mult * msx * msz / mslen2;
					*(answer+5) += mult * msy * msz / mslen2;
				}
			}
		}
	}

	for (nx = -n; nx <= n; nx++)
	{
		for (ny = -n; ny <= n; ny++)
		{
			for (nz = -n; nz <= n; nz++)
			{
				if (abs(nx)+abs(ny)+abs(nz)<=n)
				{
					if (!(nx==0 && ny==0 && nz==0))
					{	
						nlen2 = nx*nx + ny*ny + nz*nz;
						nlen = sqrt(nlen2);

						nsdot = nx*sigmax + ny*sigmay + nz*sigmaz;

						mult = 2.0 / M_PI / nlen2 * exp( -M_PI * M_PI * nlen2 / alpha2 ) * cos(2 * M_PI * nsdot );

						*(answer) += mult * ( 1.0 - (1.0 + M_PI * M_PI * nlen2 / alpha2) * nx * nx / nlen2 );
						*(answer+1) += mult * ( 1.0 - (1.0 + M_PI * M_PI * nlen2 / alpha2) * ny * ny / nlen2 );
						*(answer+2) += mult * ( 1.0 - (1.0 + M_PI * M_PI * nlen2 / alpha2) * nz * nz / nlen2 );
						*(answer+3) += mult * ( - (1.0 + M_PI * M_PI * nlen2 / alpha2) * nx * ny / nlen2 );
						*(answer+4) += mult * ( - (1.0 + M_PI * M_PI * nlen2 / alpha2) * nx * nz / nlen2 );
						*(answer+5) += mult * ( - (1.0 + M_PI * M_PI * nlen2 / alpha2) * ny * nz / nlen2 );
					}
				}
			}
		}
	}
}

// -------------------------------------------------------------------------------

void Qij(double sigmax, double sigmay, double sigmaz, double alpha, int m, int n, double* answer)
{
	double mslen, mslen2, nlen, nlen2, msx, msy, msz, nsdot, mult;

	int mx, my, mz, nx, ny, nz, i;

	double* values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double alpha3 = alpha2 * alpha;

	double sqrt_pi = sqrt(M_PI);

	for (mx = -m; mx <= m; mx++)
	{
		for (my = -m; my <= m; my++)
		{
			for (mz = -m; mz <= m; mz++)
			{
				if (abs(mx)+abs(my)+abs(mz)<=m)
				{
					msx = mx + sigmax;
					msy = my + sigmay;
					msz = mz + sigmaz;

					mslen2 = msx*msx + msy*msy + msz*msz;

					mslen = sqrt(mslen2);

					mult = erfc( alpha * mslen ) + 2.0 * alpha / sqrt_pi * mslen * exp( -alpha2 * mslen2 );

					Q(msx, msy, msz, values);

					for (i = 0; i < 6; i++)
					{
						*(answer+i) += mult * *(values+i);
					}

					mult = 4 * alpha3 / sqrt_pi * exp( -alpha2 * mslen2 );

					*(answer) -= mult * msx * msx / mslen2;
					*(answer+1) -= mult * msy * msy / mslen2;
					*(answer+2) -= mult * msz * msz / mslen2;
					*(answer+3) -= mult * msx * msy / mslen2;
					*(answer+4) -= mult * msx * msz / mslen2;
					*(answer+5) -= mult * msy * msz / mslen2;
				}
			}
		}
	}

	for (nx = -n; nx <= n; nx++)
	{
		for (ny = -n; ny <= n; ny++)
		{
			for (nz = -n; nz <= n; nz++)
			{
				if (abs(nx)+abs(ny)+abs(nz)<=n)
				{
					if (!(nx==0 && ny==0 && nz==0))
					{
						nlen2 = nx*nx + ny*ny + nz*nz;

						nlen = sqrt(nlen2);

						nsdot = nx*sigmax + ny*sigmay + nz*sigmaz;

						mult = 4 * M_PI * exp( -M_PI * M_PI * nlen2 / alpha2 ) * cos(2*M_PI*nsdot);

						*(answer) += mult * nx * nx / nlen2;
						*(answer+1) += mult * ny * ny / nlen2;
						*(answer+2) += mult * nz * nz / nlen2;
						*(answer+3) += mult * nx * ny / nlen2;
						*(answer+4) += mult * nx * nz / nlen2;
						*(answer+5) += mult * ny * nz / nlen2;
					}
				}
			}
		}
	}
}

// -------------------------------------------------------------------------------

void Mij_rpy(double ai, double aj, double rx, double ry, double rz, double* answer)
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
		double coef1 = 1.0 / 8 / M_PI / dist;
		double coef2 = 1.0 + aij2 / 3 / dist2;
		double coef3 = 1.0 - aij2 / dist2;

		*(answer) = coef1 * ( coef2 + coef3 * rx * rx / dist2 );
		*(answer+1) = coef1 * ( coef2 + coef3 * ry * ry / dist2 );
		*(answer+2) = coef1 * ( coef2 + coef3 * rz * rz / dist2 );
		*(answer+3) = coef1 * coef3 * rx * ry / dist2;
		*(answer+4) = coef1 * coef3 * rx * rz / dist2;
		*(answer+5) = coef1 * coef3 * ry * rz / dist2;
	}
	else if (dist <= (al - as))
	{
		*(answer) = 1.0 / 6 / M_PI / al;
		*(answer+1) = 1.0 / 6 / M_PI / al;
		*(answer+2) = 1.0 / 6 / M_PI / al;
	}
	else
	{
		double dist3 = dist2 * dist;

		double coef1 = 1.0 / 6 / M_PI / ai / aj;
		double coef2 = 16 * dist3 * ( ai + aj );
		double coef3 = (ai - aj) * (ai - aj) + 3*dist2;
		coef3 *= coef3;
		double coef4 = (coef2 - coef3) / 32 / dist3;
		double coef5 = ( (ai - aj)*(ai - aj) - dist2 );
		coef5 *= 3 * coef5;
		double coef6 = coef5 / 32 / dist3;

		*(answer) = coef1 * ( coef4 + coef6 * rx * rx / dist2 );
		*(answer+1) = coef1 * ( coef4 + coef6 * ry * ry / dist2 );
		*(answer+2) = coef1 * ( coef4 + coef6 * rz * rz / dist2 );
		*(answer+3) = coef1 * coef6 * rx * ry / dist2;
		*(answer+4) = coef1 * coef6 * rx * rz / dist2;
		*(answer+5) = coef1 * coef6 * ry * rz / dist2;
	}
}

// -------------------------------------------------------------------------------

void Mii_rpy_smith(double a, double box_length, double alpha, int m, int n, double* answer)
{
	int i;

	double coef1 = 1.0 / 6 / M_PI / a;
	double coef2 = 3 * a / 4 / box_length;
	double coef3 = a * a * a / box_length / box_length / box_length / 2;

	double* comp1 = calloc(6, sizeof(double));
	double* comp2 = calloc(6, sizeof(double));

	Oii(a, box_length, alpha, m, n, comp1);
	Qii(a, box_length, alpha, m, n, comp2);

	for (i = 0; i < 3; i++)
	{
		*(answer+i) = coef1 * ( 1.0 + coef2 * *(comp1+i) + coef3 * *(comp2+i) );
	}
	for (i = 3; i < 6; i++)
	{
		*(answer+i) = coef1 * ( coef2 * *(comp1+i) + coef3 * *(comp2+i) );
	}
}

// -------------------------------------------------------------------------------

void Mij_rpy_smith(double ai, double aj, double rx, double ry, double rz, double box_length, double alpha, int m, int n, double* answer)
{
	int i;
	double dist2 = rx*rx + ry*ry + rz*rz;

	double sigmax = rx / box_length;
	double sigmay = ry / box_length;
	double sigmaz = rz / box_length;

	double coef1 = 1.0 / 6 / M_PI / ai;
	double coef2 = 3 * ai / 4 / box_length;
	double coef3;
	if (ai == aj)
	{
		coef3 = ai * ai * ai / box_length / box_length / box_length / 2;
	}
	else
	{
		coef3 = ai * ( ai * ai + aj * aj ) / box_length / box_length / box_length / 4;
	}

	double* comp1 = calloc(6, sizeof(double));
	double* comp2 = calloc(6, sizeof(double));

	Oij(sigmax, sigmay, sigmaz, alpha, m, n, comp1);
	Qij(sigmax, sigmay, sigmaz, alpha, m, n, comp2);

	for (i = 0; i < 6; i++)
	{
		*(answer+i) = coef1 * ( coef2 * *(comp1+i) + coef3 * *(comp2+i) );
	}

	if (dist2 < (ai + aj)*(ai + aj)) //new added
	{
		double dist = sqrt(dist2);

		double aij2 = ai*ai + aj*aj;

		double* Aij = calloc(6, sizeof(double));

		Mij_rpy(ai, aj, rx, ry, rz, Aij);

		for (i = 0; i < 6; i++)
		{
			*(answer+i) += *(Aij+i);
		}

		coef1 = 1.0 / 8 / M_PI / dist;
		coef2 = 1.0 + aij2 / 3 / dist2;
		coef3 = 1.0 - aij2 / dist2;

		*(answer) -= coef1 * ( coef2 + coef3 * rx * rx / dist2 );
		*(answer+1) -= coef1 * ( coef2 + coef3 * ry * ry / dist2 );
		*(answer+2) -= coef1 * ( coef2 + coef3 * rz * rz / dist2 );
		*(answer+3) -= coef1 * coef3 * rx * ry / dist2;
		*(answer+4) -= coef1 * coef3 * rx * rz / dist2;
		*(answer+5) -= coef1 * coef3 * ry * rz / dist2;
	}
}

// -------------------------------------------------------------------------------

int results_position(int i, int j, int N)
{
	int k;

	int position = i + j*N;

	// for (k = 1; k <= j; k++)
	// {
	// 	position -= k;
	// }

	return position - (j+1)/2*j;
}

// -------------------------------------------------------------------------------

void M_rpy_smith(double* as, double* pointers, double box_length, double alpha, int m, int n, int N, double* results)
{
	int i, j, k;

	double* vector;

	double* shifted_results;

	double* shifted_pointers;

	double rx, ry, rz;

	for (j = 0; j < N; j++)
	{
		vector = calloc(6, sizeof(double));

		Mii_rpy_smith(*(as+j), box_length, alpha, m, n, vector);

		shifted_results = results + 6*results_position(j,j,N);

		for (k = 0; k < 3; k++)
		{
			*(shifted_results + k) = *(vector + k);
		}

		for (i = j + 1; i < N; i++)
		{
			vector = calloc(6, sizeof(double));

			shifted_results = results + 6*results_position(i,j,N);
			shifted_pointers = pointers + 3*results_position(i-1,j,N-1);

			rx = *(shifted_pointers);
			ry = *(shifted_pointers + 1);
			rz = *(shifted_pointers + 2);

			Mij_rpy_smith(*(as+i), *(as+j), rx, ry, rz, box_length, alpha, m, n, vector);

			for (k = 0; k < 6; k++)
			{
				*(shifted_results + k) = *(vector + k);
			}
		}
	}
}

// -------------------------------------------------------------------------------

double X_f_poly(double l, int rank)
{
	double answer;
	double l2, l3, l4, l5, l6, l7, l8, l9, l10;

	switch(rank)
	{
		case 0:
			answer = 1;
			break;
		case 1:
			answer = 3*l;
			break;
		case 2:
			answer = 9*l;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l; 
			answer = -4*l + 27*l2 - 4*l3;
			break;
		case 4:
			l2 = l*l;
			l3 = l2*l;
			answer = -24*l + 81*l2 + 36*l3;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			answer = 72*l2 + 243*l3 + 72*l4;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			answer = 16*l + 108*l2 + 281*l3 + 648*l4 + 144*l5;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			answer = 288*l2 + 1620*l3 + 1515*l4 + 1620*l5 + 288*l6;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			answer = 576*l2 + 4848*l3 + 5409*l4 + 4524*l5 + 3888*l6 + 576*l7;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			answer = 1152*l2 + 9072*l3 + 14752*l4 + 26163*l5 + 14752*l6 + 9072*l7 + 1152*l8;
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
			answer = 2304*l2 + 20736*l3 + 42804*l4 + 115849*l5 + 76176*l6 + 39264*l7 + 20736*l8 + 2304*l9;
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
			answer = 4608*l2 + 46656*l3 + 108912*l4 + 269100*l5 + 319899*l6 + 269100*l7 + 108912*l8 + 46656*l9 + 4608*l10;
			break;
	}

	return answer;
}

// -------------------------------------------------------------------------------

// def Y_f_poly_python(l, rank):

// 	if rank == 0: return 1
// 	if rank == 1: return 3 / 2 * l
// 	if rank == 2: return 9 / 4 * l
// 	if rank == 3: return 2 * l + 27 / 8 * l**2 + 2 * l**3
// 	if rank == 4: return 6 * l + 81 / 16 * l**2 + 18 * l**3
// 	if rank == 5: return 63 / 2 * l**2 + 243 / 32 * l**3 + 63 / 2 * l**4
// 	if rank == 6: return 4 * l + 54 * l**2 + 1241 / 64 * l**3 + 81 * l**4 + 72 * l**5
// 	if rank == 7: return 144 * l**2 + 1053 / 8 * l**3 + 19083 / 128 * l**4 + 1053 / 8 * l**5 + 144 * l**6
// 	if rank == 8: return 279 * l**2 + 4261 / 8 * l**3 + 126369 / 256 * l**4 - 117 / 8 * l**5 + 648 * l**6 + 288 * l**7
// 	if rank == 9: return 576 * l**2 + 1134 * l**3 + 60443 / 32 * l**4 + 766179 / 512 * l**5 + 60443 / 32 * l**6 + 1134 * l**7 + 576 * l**8
// 	if rank == 10: return 1152 * l**2 + 7857 / 4 * l**3 + 98487 / 16 * l**4 + 10548393 / 1024 * l**5 + 67617 / 8 * l**6 - 351 / 2 * l**7 + 3888 * l**8 + 1152 * l**9
// 	if rank == 11: return 2304 * l**2 + 7128 * l**3 + 22071 / 2 * l**4 + 2744505 / 128 * l**5 + 95203835 / 2048 * l**6 + 2744505 / 128 * l**7 + 22071 / 2 * l**8 + 7128 * l**9 + 2304 * l**10
// 	else: return None

double Y_f_poly(double l, int rank)
{
	double answer;
	double l2, l3, l4, l5, l6, l7, l8, l9, l10;

	switch(rank)
	{
		case 0:
			answer = 1;
			break;
		case 1:
			answer = 3.0/2*l;
			break;
		case 2:
			answer = 9.0/4*l;
			break;
		case 3:
			l2 = l*l;
			l3 = l2*l;
			answer = 2*l + 27.0/8*l2 + 2*l3;
			break;
		case 4:
			l2 = l*l;
			l3 = l2*l;
			answer = 6*l + 81.0/16*l2 + 18*l3;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			answer = 63.0/2*l2 + 243.0/32*l3 + 63.0/2*l4;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			answer = 4*l + 54*l2 + 1241.0/64*l3 + 81*l4 + 72*l5;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			answer = 144*l2 + 1053.0/8*l3 + 19083.0/128*l4 + 1053.0/8*l5 + 144*l6;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			answer = 279*l2 + 4261.0/8*l3 + 126369.0/256*l4 - 117.0/8*l5 + 648*l6 + 288*l7;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			answer = 576*l2 + 1134*l3 + 60443.0/32*l4 + 766179.0/512*l5 + 60443.0/32*l6 + 1134*l7 + 576*l8;
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
			answer = 1152*l2 + 7857.0/4*l3 + 98487.0/16*l4 + 10548393.0/1024*l5 + 67617.0/8*l6 - 351.0/2*l7 + 3888*l8 + 1152*l9;
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
			answer = 2304*l2 + 7128*l3 + 22071.0/2*l4 + 2744505.0/128*l5 + 95203835.0/2048*l6 + 2744505.0/128*l7 + 22071.0/2*l8 + 7128*l9 + 2304*l10;
			break;
	}

	return answer;
}

// -------------------------------------------------------------------------------

double X_g_poly(double l, int rank)
{
	double answer;
	double l2 = l*l;
	double plus = 1+l;
	double plus3 = plus * plus * plus;
	double l3, l4;

	switch(rank)
	{
		case 1:
			answer = 2*l2/plus3;
			break;
		case 2:
			answer = 1.0/5*l*(1 + 7*l + l2)/plus3;
			break;
		case 3:
			l3 = l2*l;
			l4 = l2*l2;
			answer = 1.0/42*(1 + 18*l - 29*l2 + 18*l3 + l4)/plus3;
			break;
	}

	return answer;
}

// -------------------------------------------------------------------------------

double Y_g_poly(double l, int rank)
{
	double answer;
	double l2 = l*l;
	double plus = 1+l;
	double plus3 = plus * plus * plus;
	double l3, l4;

	switch(rank)
	{
		case 2:
			answer = 4.0/15*l*(2 + l + 2*l2)/plus3;
			break;
		case 3:
			l3 = l2*l;
			l4 = l2*l2;
			answer = 2.0/375*(16 - 45*l + 58*l2 - 45*l3 + 16*l4)/plus3;
	}

	return answer;
}

// -------------------------------------------------------------------------------

double XA11(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 / s / s;

	int m, m1;

	double mult;

	answer += X_g_poly(l, 1)/(1.0 - 4.0*s2);

	answer -= X_g_poly(l, 2)*log(1.0 - 4.0*s2);

	answer -= X_g_poly(l, 3)*(1.0 - 4.0*s2)*log(1.0 - 4.0*s2);

	answer += X_f_poly(l, 0) - X_g_poly(l, 1);

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

			answer += mult * ( pow(2, -m) * pow(1+l, -m) * X_f_poly(l, m) - X_g_poly(l, 1) );

			answer += mult * ( 4.0 / m / m1 * X_g_poly(l, 3) - 2.0 / m * X_g_poly(l, 2) );
		}
	}

	return answer;
}

// -------------------------------------------------------------------------------

double YA11(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 / s / s;

	int m, m1;

	double mult;

	answer -= Y_g_poly(l, 2)*log(1.0 - 4.0*s2 );

	answer -= Y_g_poly(l, 3)*(1.0 - 4.0*s2)*log(1.0 - 4.0*s2 );

	answer += Y_f_poly(l, 0);

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

			answer += mult * ( pow(2, -m)*pow(1+l, -m)*Y_f_poly(l, m) - 2.0/m*Y_g_poly(l, 2) );

			answer += mult*4.0/m/m1*Y_g_poly(l, 3);
		}
	}

	return answer;
}

// -------------------------------------------------------------------------------

double XA12(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 / s / s;

	int m, m1;

	double mult, divisor;

	answer += 2.0/s*X_g_poly(l, 1)/(1.0 - 4.0*s2);

	answer += X_g_poly(l, 2)*log((s + 2)/(s - 2));

	answer += X_g_poly(l, 3)*(1.0 - 4.0*s2)*log((s + 2)/(s - 2)) + 4*X_g_poly(l, 3)/s;

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

			answer += mult * ( pow(2, -m)*pow(1 + l, -m)*X_f_poly(l, m) - X_g_poly(l, 1) );

			answer += mult * ( 4.0/m/m1 * X_g_poly(l, 3) - 2.0 / m * X_g_poly(l, 2) );
		}

	}

	divisor = -(1.0 + l)/2;

	return answer / divisor;
}

// -------------------------------------------------------------------------------

double YA12(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 / s / s;

	int m, m1;

	double mult, divisor;

	answer += Y_g_poly(l, 2)*log((s + 2)/(s - 2));

	answer += Y_g_poly(l, 3)*(1.0 - 4.0*s2)*log((s + 2)/(s - 2));

	answer += 4*Y_g_poly(l, 3)/s;

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

			answer += mult*(pow(2, -m)*pow(1+l, -m)*Y_f_poly(l, m) - 2.0/m*Y_g_poly(l, 2));

			answer += mult*(4.0/m/m1*Y_g_poly(l, 3));
		}
	}

	divisor = -(1.0 + l)/2;

	return answer / divisor;
}

// -------------------------------------------------------------------------------

void R_jeffrey(double ai, double aj, double rx, double ry, double rz, double* answer)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double s = 2*dist/(ai + aj);

	double l = aj/ai;

	int i;

	// block 00

	*answer = XA11(s, l)*rx*rx/dist2 + YA11(s, l)*(1 - rx*rx/dist2); // 00

	*(answer+1) = XA11(s, l)*ry*ry/dist2 + YA11(s, l)*(1 - ry*ry/dist2); // 11

	*(answer+2) = XA11(s, l)*rz*rz/dist2 + YA11(s, l)*(1 - rz*rz/dist2); // 22

	*(answer+3) = (XA11(s, l) - YA11(s, l))*rx*ry/dist2; // 10

	*(answer+4) = (XA11(s, l) - YA11(s, l))*rx*rz/dist2; // 20

	*(answer+5) = (XA11(s, l) - YA11(s, l))*ry*rz/dist2; // 21

	// block 11

	*(answer+6) = XA11(s, 1/l)*rx*rx/dist2 + YA11(s, 1/l)*(1 - rx*rx/dist2); // 33

	*(answer+7) = XA11(s, 1/l)*ry*ry/dist2 + YA11(s, 1/l)*(1 - ry*ry/dist2); // 44

	*(answer+8) = XA11(s, 1/l)*rz*rz/dist2 + YA11(s, 1/l)*(1 - rz*rz/dist2); // 55

	*(answer+9) = (XA11(s, 1/l) - YA11(s, 1/l))*rx*ry/dist2; // 43

	*(answer+10) = (XA11(s, 1/l) - YA11(s, 1/l))*rx*rz/dist2; // 53

	*(answer+11) = (XA11(s, 1/l) - YA11(s, 1/l))*ry*rz/dist2; // 54

	// block 10

	*(answer+12) = XA12(s, 1/l)*rx*rx/dist2 + YA12(s, 1/l)*(1 - rx*rx/dist2); // 30

	*(answer+13) = XA12(s, 1/l)*ry*ry/dist2 + YA12(s, 1/l)*(1 - ry*ry/dist2); // 41

	*(answer+14) = XA12(s, 1/l)*rz*rz/dist2 + YA12(s, 1/l)*(1 - rz*rz/dist2); // 52

	*(answer+15) = (XA12(s, 1/l) - YA12(s, 1/l))*rx*ry/dist2; // 40

	*(answer+16) = (XA12(s, 1/l) - YA12(s, 1/l))*rx*rz/dist2; // 50

	*(answer+17) = (XA12(s, 1/l) - YA12(s, 1/l))*ry*rz/dist2; // 51

	for (i = 0; i < 18; i++)
	{
		*(answer+i) *= 3 * M_PI * ( ai + aj );
	}
}

// -------------------------------------------------------------------------------