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

	double dist3 = dist2 * dist;

	*answer = ( 1 + rx*rx / dist2 ) / dist;

	*(answer+1) = ( 1 + ry*ry / dist2 ) / dist;

	*(answer+2) = ( 1 + rz*rz / dist2 ) / dist;

	*(answer+3) = rx*ry / dist3;

	*(answer+4) = rx*rz / dist3;

	*(answer+5) = ry*rz / dist3;
}

// -------------------------------------------------------------------------------

void Q(double rx, double ry, double rz, double* answer)
{
	double dist2 = rx*rx + ry*ry + rz*rz;

	double dist = sqrt(dist2);

	double dist3 = dist2 * dist;

	double dist5 = dist3 * dist2;

	*answer = ( 1 - 3 * rx*rx / dist2 ) / dist3;

	*(answer+1) = ( 1 - 3 * ry*ry / dist2 ) / dist3;

	*(answer+2) = ( 1 - 3 * rz*rz / dist2 ) / dist3;

	*(answer+3) = -3 * rx*ry / dist5;

	*(answer+4) = -3 * rx*rz / dist5;

	*(answer+5) = -3 * ry*rz / dist5;
}

// -------------------------------------------------------------------------------

void Oii(double a, double box_length, double alpha, int m, int n, double* answer)
{
	double mlen, mlen2, nlen, nlen2, mult, temp;

	int mbis, mtris, nbis, ntris;

	register int mx;

	register int my;

	register int mz;
	
	register int nx;
	
	register int ny;
	
	register int nz;

	double* values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	double mult0 = 2 * alpha / sqrt_pi;

	double exp_const = -M_PI*M_PI/alpha2;

	for (mx = -m; mx <= m; mx++)
	{
		mbis = m - abs(mx);
		for (my = -mbis; my <= mbis; my++)
		{
			mtris = mbis - abs(my);
			for (mz = -mtris; mz <= mtris; mz++)
			{
				if (!(mx==0 && my==0 && mz==0))
				{
					mlen2 = mx*mx + my*my + mz*mz;
					mlen = sqrt(mlen2);
					O(mx, my, mz, values);
					mult = erfc( alpha*mlen );

					*(answer) += mult * *(values);
					*(answer+1) += mult * *(values+1);
					*(answer+2) += mult * *(values+2);
					*(answer+3) += mult * *(values+3);
					*(answer+4) += mult * *(values+4);
					*(answer+5) += mult * *(values+5);

					mult = mult0 * exp( - alpha2 * mlen2 ) / mlen2;

					*(answer) += mult * mx * mx;
					*(answer+1) += mult * my * my;
					*(answer+2) += mult * mz * mz;
					*(answer+3) += mult * mx * my;
					*(answer+4) += mult * mx * mz;
					*(answer+5) += mult * my * mz;
				}
			}
		}
	}

	mult0 = 2.0 / M_PI;

	for (nx = -n; nx <= n; nx++)
	{
		nbis = n - abs(nx);
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

					*(answer) += mult * ( 1.0 - temp * nx * nx );					
					*(answer+1) += mult * ( 1.0 - temp * ny * ny );					
					*(answer+2) += mult * ( 1.0 - temp * nz * nz );					
					*(answer+3) += mult * ( - temp * nx * ny );				
					*(answer+4) += mult * ( - temp * nx * nz );				
					*(answer+5) += mult * ( - temp * ny * nz );				
				}
			}
		}
	}

	temp = 1.5 * alpha * a / ( sqrt_pi * box_length );

	*(answer) -= temp;
	*(answer+1) -= temp;
	*(answer+2) -= temp;
}

// -------------------------------------------------------------------------------

void Qii(double a, double box_length, double alpha, int m, int n, double* answer)
{
	double amlen, amlen2, expamlen2, mlen2, nlen, nlen2, mult;

	int mbis, mtris, nbis, ntris;

	register int mx;

	register int my;

	register int mz;
	
	register int nx;
	
	register int ny;
	
	register int nz;

	double* values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	double alpha3 = alpha * alpha2;

	double a3 = a * a * a;

	double box_length3 = box_length * box_length * box_length;

	double temp1 = 2 / sqrt_pi;

	double temp2 = 2 * alpha3;

	for (mx = -m; mx <= m; mx++)
	{
		mbis = m -abs(mx);
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
					Q(mx, my, mz, values);
					mult = erfc( amlen ) + temp1 * amlen * expamlen2;

					*(answer) += mult * *(values);
					*(answer+1) += mult * *(values+1);
					*(answer+2) += mult * *(values+2);
					*(answer+3) += mult * *(values+3);
					*(answer+4) += mult * *(values+4);
					*(answer+5) += mult * *(values+5);

					mult = temp1 * temp2 * expamlen2 / mlen2;

					*(answer) -= mult * mx * mx;
					*(answer+1) -= mult * my * my;
					*(answer+2) -= mult * mz * mz;
					*(answer+3) -= mult * mx * my;
					*(answer+4) -= mult * mx * mz;
					*(answer+5) -= mult * my * mz;
				}
			}
		}
	}

	temp1 = -M_PI * M_PI / alpha2;

	temp2 = 4 * M_PI;

	for (nx = -n; nx <= n; nx++)
	{
		nbis = n - abs(nx);
		for (ny = -nbis; ny <= nbis; ny++)
		{
			ntris = nbis - abs(ny);
			for (nz = -ntris; nz <= ntris; nz++)
			{
				if (!(nx==0 && ny==0 && nz==0))
				{
					nlen2 = nx*nx + ny*ny + nz*nz;
					mult = temp2 * exp( temp1 * nlen2 ) / nlen2;

					*(answer) += mult * nx * nx;
					*(answer+1) += mult * ny * ny;
					*(answer+2) += mult * nz * nz;
					*(answer+3) += mult * nx * ny;
					*(answer+4) += mult * nx * nz;
					*(answer+5) += mult * ny * nz;
				}
			}
		}
	}

	temp1 = alpha3 * a3 / ( box_length3 * 3.0 * sqrt_pi );

	*(answer) -= temp1;
	*(answer+1) -= temp1;
	*(answer+2) -= temp1;	
}

// -------------------------------------------------------------------------------

void Oij(double sigmax, double sigmay, double sigmaz, double alpha, int m, int n, double* answer)
{
	double mslen, mslen2, nlen, nlen2, msx, msy, msz, nsdot, mult, mult2, exp_const;

	int mbis, mtris, nbis, ntris;

	register int mx;

	register int my;

	register int mz;
	
	register int nx;
	
	register int ny;
	
	register int nz;

	double* values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double sqrt_pi = sqrt(M_PI);

	double temp = 2 * alpha / sqrt_pi;

	for (mx = -m; mx <= m; mx++)
	{
		mbis = m - abs(mx);
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
				O(msx, msy, msz, values);

				*(answer) += mult * *(values);
				*(answer+1) += mult * *(values+1);
				*(answer+2) += mult * *(values+2);
				*(answer+3) += mult * *(values+3);
				*(answer+4) += mult * *(values+4);
				*(answer+5) += mult * *(values+5);

				mult = temp * exp( -alpha2 * mslen2 ) / mslen2;

				*(answer) += mult * msx * msx;
				*(answer+1) += mult * msy * msy;
				*(answer+2) += mult * msz * msz;
				*(answer+3) += mult * msx * msy;
				*(answer+4) += mult * msx * msz;
				*(answer+5) += mult * msy * msz;
			}
		}
	}

	temp = M_PI * M_PI / alpha2;

	double temp2 = 2.0 / M_PI;

	for (nx = -n; nx <= n; nx++)
	{
		nbis = n - abs(nx);
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

					*(answer) += mult * ( 1.0 - mult2 * nx * nx );
					*(answer+1) += mult * ( 1.0 - mult2 * ny * ny );
					*(answer+2) += mult * ( 1.0 - mult2 * nz * nz );
					*(answer+3) += mult * ( - mult2 * nx * ny );
					*(answer+4) += mult * ( - mult2 * nx * nz );
					*(answer+5) += mult * ( - mult2 * ny * nz );
				}
			}
		}
	}
}

// -------------------------------------------------------------------------------

void Qij(double sigmax, double sigmay, double sigmaz, double alpha, int m, int n, double* answer)
{
	double amslen, amslen2, mslen2, expamslen2, nlen, nlen2, msx, msy, msz, nsdot, mult;

	int mbis, mtris, nbis, ntris;

	register int mx;

	register int my;

	register int mz;
	
	register int nx;
	
	register int ny;
	
	register int nz;

	double* values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double alpha3 = alpha2 * alpha;

	double sqrt_pi = sqrt(M_PI);

	double temp1 = 2.0 / sqrt_pi;

	double temp2 = 4 * alpha3 / sqrt_pi;

	for (mx = -m; mx <= m; mx++)
	{
		mbis = m - abs(mx);
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
				Q(msx, msy, msz, values);

				*(answer) += mult * *(values);
				*(answer+1) += mult * *(values+1);
				*(answer+2) += mult * *(values+2);
				*(answer+3) += mult * *(values+3);
				*(answer+4) += mult * *(values+4);
				*(answer+5) += mult * *(values+5);

				mult = temp2 * expamslen2 / mslen2;

				*(answer) -= mult * msx * msx;
				*(answer+1) -= mult * msy * msy;
				*(answer+2) -= mult * msz * msz;
				*(answer+3) -= mult * msx * msy;
				*(answer+4) -= mult * msx * msz;
				*(answer+5) -= mult * msy * msz;
			}
		}
	}

	temp1 = -M_PI*M_PI/alpha2;

	temp2 = 4 * M_PI;

	for (nx = -n; nx <= n; nx++)
	{
		nbis = n - abs(nx);
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

					*(answer) += mult * nx * nx;
					*(answer+1) += mult * ny * ny;
					*(answer+2) += mult * nz * nz;
					*(answer+3) += mult * nx * ny;
					*(answer+4) += mult * nx * nz;
					*(answer+5) += mult * ny * nz;
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
		double coef1 = 1.0 / ( 8 * M_PI * dist );
		double coef2 = 1.0 + aij2 / ( 3 * dist2 );
		double coef3 = ( 1.0 - aij2 / dist2 ) / dist2;

		*(answer) = coef1 * ( coef2 + coef3 * rx * rx );
		*(answer+1) = coef1 * ( coef2 + coef3 * ry * ry );
		*(answer+2) = coef1 * ( coef2 + coef3 * rz * rz );
		*(answer+3) = coef1 * coef3 * rx * ry;
		*(answer+4) = coef1 * coef3 * rx * rz;
		*(answer+5) = coef1 * coef3 * ry * rz;
	}
	else if (dist <= (al - as))
	{
		double temp = 1.0 / ( 6 * M_PI * al );

		*(answer) = temp;
		*(answer+1) = temp;
		*(answer+2) = temp;
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

		*(answer) = coef1 * ( coef4 + coef6 * rx * rx );
		*(answer+1) = coef1 * ( coef4 + coef6 * ry * ry );
		*(answer+2) = coef1 * ( coef4 + coef6 * rz * rz );
		*(answer+3) = coef1 * coef6 * rx * ry;
		*(answer+4) = coef1 * coef6 * rx * rz;
		*(answer+5) = coef1 * coef6 * ry * rz;
	}
}

// -------------------------------------------------------------------------------

void Mii_rpy_smith(double a, double box_length, double alpha, int m, int n, double* answer)
{
	int i;

	double coef1 = 1.0 / ( 6 * M_PI * a );
	double coef2 = 3 * a / ( 4 * box_length );
	double coef3 = a * a * a / ( 2 * box_length * box_length * box_length );

	double* comp1 = calloc(6, sizeof(double));
	double* comp2 = calloc(6, sizeof(double));

	Oii(a, box_length, alpha, m, n, comp1);
	Qii(a, box_length, alpha, m, n, comp2);

	*(answer) = coef1 * ( 1.0 + coef2 * *(comp1) + coef3 * *(comp2) );
	*(answer+1) = coef1 * ( 1.0 + coef2 * *(comp1+1) + coef3 * *(comp2+1) );
	*(answer+2) = coef1 * ( 1.0 + coef2 * *(comp1+2) + coef3 * *(comp2+2) );

	*(answer+3) = coef1 * ( coef2 * *(comp1+3) + coef3 * *(comp2+3) );
	*(answer+4) = coef1 * ( coef2 * *(comp1+4) + coef3 * *(comp2+4) );
	*(answer+5) = coef1 * ( coef2 * *(comp1+5) + coef3 * *(comp2+5) );
}

// -------------------------------------------------------------------------------

void Mij_rpy_smith(double ai, double aj, double rx, double ry, double rz, double box_length, double alpha, int m, int n, double* answer)
{
	int i;
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

	Oij(sigmax, sigmay, sigmaz, alpha, m, n, comp1);
	Qij(sigmax, sigmay, sigmaz, alpha, m, n, comp2);

	*(answer) = coef1 * ( coef2 * *(comp1) + coef3 * *(comp2) );
	*(answer+1) = coef1 * ( coef2 * *(comp1+1) + coef3 * *(comp2+1) );
	*(answer+2) = coef1 * ( coef2 * *(comp1+2) + coef3 * *(comp2+2) );
	*(answer+3) = coef1 * ( coef2 * *(comp1+3) + coef3 * *(comp2+3) );
	*(answer+4) = coef1 * ( coef2 * *(comp1+4) + coef3 * *(comp2+4) );
	*(answer+5) = coef1 * ( coef2 * *(comp1+5) + coef3 * *(comp2+5) );


	if (dist2 < (ai + aj)*(ai + aj))
	{
		double dist = sqrt(dist2);

		double aij2 = ai*ai + aj*aj;

		double* Aij = calloc(6, sizeof(double));

		Mij_rpy(ai, aj, rx, ry, rz, Aij);

		*(answer) += *(Aij);
		*(answer+1) += *(Aij+1);
		*(answer+2) += *(Aij+2);
		*(answer+3) += *(Aij+3);
		*(answer+4) += *(Aij+4);
		*(answer+5) += *(Aij+5);

		coef1 = 1.0 / ( 8 * M_PI * dist );
		coef2 = 1.0 + aij2 / ( 3 * dist2 );
		coef3 = ( 1.0 - aij2 / dist2 ) / dist2;

		*(answer) -= coef1 * ( coef2 + coef3 * rx * rx);
		*(answer+1) -= coef1 * ( coef2 + coef3 * ry * ry );
		*(answer+2) -= coef1 * ( coef2 + coef3 * rz * rz );
		*(answer+3) -= coef1 * coef3 * rx * ry;
		*(answer+4) -= coef1 * coef3 * rx * rz;
		*(answer+5) -= coef1 * coef3 * ry * rz;
	}
}

// -------------------------------------------------------------------------------

int results_position(int i, int j, int N)
{
	int k;

	int position = i + j*N;

	return position - j*(j+1)/2;
}

// -------------------------------------------------------------------------------

void M_rpy_smith(double* as, double* pointers, double box_length, double alpha, int m, int n, int N, double* results2)
{
	register int i = 0;

	register int j = 0;

	int I, J, J1, J2, I1, I2;

	int r;

	double* vector;

	double* shifted_results;

	double* shifted_pointers;

	double rx, ry, rz;

	double* results = calloc(3*(N*N+N), sizeof(double));

	for (j = 0; j < N; j++)
	{
		vector = calloc(6, sizeof(double));

		Mii_rpy_smith(*(as+j), box_length, alpha, m, n, vector);

		shifted_results = results + 6*results_position(j,j,N);

		*(shifted_results) = *(vector);
		*(shifted_results + 1) = *(vector + 1);
		*(shifted_results + 2) = *(vector + 2);

		for (i = j + 1; i < N; i++)
		{
			vector = calloc(6, sizeof(double));

			shifted_results = results + 6*results_position(i,j,N);
			shifted_pointers = pointers + 3*results_position(i-1,j,N-1);

			rx = *(shifted_pointers);
			ry = *(shifted_pointers + 1);
			rz = *(shifted_pointers + 2);

			Mij_rpy_smith(*(as+i), *(as+j), rx, ry, rz, box_length, alpha, m, n, vector);

			*(shifted_results) = *(vector);
			*(shifted_results + 1) = *(vector + 1);
			*(shifted_results + 2) = *(vector + 2);
			*(shifted_results + 3) = *(vector + 3);
			*(shifted_results + 4) = *(vector + 4);
			*(shifted_results + 5) = *(vector + 5);

		}
	}

	int N3 = 3*N;

	for (j = 0; j < N; j++)
	{
		r = 6*results_position(j, j, N);
		J = 3*j;
		J1 = J + 1;
		J2 = J + 2;

		results2[J + J*N3] = results[r];
		results2[J1 + J1*N3] = results[r + 1];
		results2[J2 + J2*N3] = results[r + 2];
		results2[J1 + J*N3] = results[r + 3];
		results2[J + J1*N3] = results[r + 3];
		results2[J2 + J*N3] = results[r + 4];
		results2[J + J2*N3] = results[r + 4];
		results2[J2 + J1*N3] = results[r + 5];
		results2[J1 + J2*N3] = results[r + 5];

		for (i = j+1; i < N; i++)
		{
			r = 6*results_position(i, j, N);
			I = 3*i;
			I1 = I + 1;
			I2 = I + 2;

			results2[I + J*N3] = results[r];
			results2[J + I*N3] = results[r];
			results2[I1 + J1*N3] = results[r + 1];
			results2[J1 + I1*N3] = results[r + 1];
			results2[I2 + J2*N3] = results[r + 2];
			results2[J2 + I2*N3] = results[r + 2];
			results2[I1 + J*N3] = results[r + 3];
			results2[I + J1*N3] = results[r + 3];
			results2[J + I1*N3] = results[r + 3];
			results2[J1 + I*N3] = results[r + 3];
			results2[I2 + J*N3] = results[r + 4];
			results2[I + J2*N3] = results[r + 4];
			results2[J + I2*N3] = results[r + 4];
			results2[J2 + I*N3] = results[r + 4];
			results2[I2 + J1*N3] = results[r + 5];
			results2[I1 + J2*N3] = results[r + 5];
			results2[J1 + I2*N3] = results[r + 5];
			results2[J2 + I1*N3] = results[r + 5];
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
			answer = 2*l + 3.375*l2 + 2*l3;
			break;
		case 4:
			l2 = l*l;
			l3 = l2*l;
			answer = 6*l + 5.0625*l2 + 18*l3;
			break;
		case 5:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			answer = 31.5*l2 + 7.59375*l3 + 31.5*l4;
			break;
		case 6:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			answer = 4*l + 54*l2 + 19.390625*l3 + 81*l4 + 72*l5;
			break;
		case 7:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			answer = 144*l2 + 131.625*l3 + 149.0859375*l4 + 131.625*l5 + 144*l6;
			break;
		case 8:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			answer = 279*l2 + 532.625*l3 + 493.62890625*l4 - 14.625*l5 + 648*l6 + 288*l7;
			break;
		case 9:
			l2 = l*l;
			l3 = l2*l;
			l4 = l2*l2;
			l5 = l3*l2;
			l6 = l3*l3;
			l7 = l4*l3;
			l8 = l4*l4;
			answer = 576*l2 + 1134*l3 + 1888.84375*l4 + 1496.443359375*l5 + 1888.84375*l6 + 1134*l7 + 576*l8;
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
			answer = 1152*l2 + 1964.25*l3 + 6155.4375*l4 + 10301.1650390625*l5 + 8452.125*l6 - 175.5*l7 + 3888*l8 + 1152*l9;
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
			answer = 2304*l2 + 7128*l3 + 11035.5*l4 + 21441.4453125*l5 + 46486.24755859375*l6 + 21441.4453125*l7 + 11035.5*l8 + 7128*l9 + 2304*l10;
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
			answer = 0.2*l*(1 + 7*l + l2)/plus3;
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

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	int m, m1;

	double mult;

	answer += X_g_poly(l, 1)/s2;

	answer -= X_g_poly(l, 2)*logs2;

	answer -= X_g_poly(l, 3)*s2*logs2;

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

			answer += mult * ( 4.0 / ( m * m1 ) * X_g_poly(l, 3) - 2.0 / m * X_g_poly(l, 2) );
		}
	}

	return answer;
}

// -------------------------------------------------------------------------------

double YA11(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs2 = log(s2);

	int m, m1;

	double mult;

	answer -= Y_g_poly(l, 2)*logs2;

	answer -= Y_g_poly(l, 3)*s2*logs2;

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

			answer += mult*4.0/(m*m1)*Y_g_poly(l, 3);
		}
	}

	return answer;
}

// -------------------------------------------------------------------------------

double XA12(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs = log((s + 2)/(s - 2));

	int m, m1;

	double mult, divisor;

	answer += 2.0/s*X_g_poly(l, 1)/s2;

	answer += X_g_poly(l, 2)*logs;

	answer += X_g_poly(l, 3)*s2*logs + 4*X_g_poly(l, 3)/s;

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

			answer += mult * ( 4.0/(m*m1) * X_g_poly(l, 3) - 2.0 / m * X_g_poly(l, 2) );
		}

	}

	divisor = -(1.0 + l)/2;

	return answer / divisor;
}

// -------------------------------------------------------------------------------

double YA12(double s, double l)
{
	double answer = 0.0;

	double s2 = 1.0 - 4.0 / ( s * s );

	double logs = log((s + 2)/(s - 2));

	int m, m1;

	double mult, divisor;

	answer += Y_g_poly(l, 2)*logs;

	answer += Y_g_poly(l, 3)*s2*logs;

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

			answer += mult*(4.0/(m*m1)*Y_g_poly(l, 3));
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

	double xa11l = XA11(s, l);

	double ya11l = YA11(s, l);

	double xa11linv = XA11(s, 1/l);

	double ya11linv = YA11(s, 1/l);

	double xa12linv = XA12(s, 1/l);

	double ya12linv = YA12(s, 1/l);

	double mult = 3 * M_PI * ( ai + aj );

	int i;

	// block 00

	*answer = xa11l*rx*rx/dist2 + ya11l*(1 - rx*rx/dist2); // 00

	*(answer+1) = xa11l*ry*ry/dist2 + ya11l*(1 - ry*ry/dist2); // 11

	*(answer+2) = xa11l*rz*rz/dist2 + ya11l*(1 - rz*rz/dist2); // 22

	*(answer+3) = (xa11l - ya11l)*rx*ry/dist2; // 10

	*(answer+4) = (xa11l - ya11l)*rx*rz/dist2; // 20

	*(answer+5) = (xa11l - ya11l)*ry*rz/dist2; // 21

	// block 11

	*(answer+6) = xa11linv*rx*rx/dist2 + ya11linv*(1 - rx*rx/dist2); // 33

	*(answer+7) = xa11linv*ry*ry/dist2 + ya11linv*(1 - ry*ry/dist2); // 44

	*(answer+8) = xa11linv*rz*rz/dist2 + ya11linv*(1 - rz*rz/dist2); // 55

	*(answer+9) = (xa11linv - ya11linv)*rx*ry/dist2; // 43

	*(answer+10) = (xa11linv - ya11linv)*rx*rz/dist2; // 53

	*(answer+11) = (xa11linv - ya11linv)*ry*rz/dist2; // 54

	// block 10

	*(answer+12) = xa12linv*rx*rx/dist2 + ya12linv*(1 - rx*rx/dist2); // 30

	*(answer+13) = xa12linv*ry*ry/dist2 + ya12linv*(1 - ry*ry/dist2); // 41

	*(answer+14) = xa12linv*rz*rz/dist2 + ya12linv*(1 - rz*rz/dist2); // 52

	*(answer+15) = (xa12linv - ya12linv)*rx*ry/dist2; // 40

	*(answer+16) = (xa12linv - ya12linv)*rx*rz/dist2; // 50

	*(answer+17) = (xa12linv - ya12linv)*ry*rz/dist2; // 51

	for (i = 0; i < 18; i++)
	{
		*(answer+i) *= mult;
	}
}

// -------------------------------------------------------------------------------