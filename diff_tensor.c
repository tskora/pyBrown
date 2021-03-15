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

void Oii(double a, double box_length, double alpha, int m, int n, double* answer)
{
	double mlen, mlen2, nlen, nlen2, mult;

	double* values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	for (int mx = -m; mx <= m; mx++)
	{
		for (int my = -m; my <= m; my++)
		{
			for (int mz = -m; mz <= m; mz++)
			{
				if (abs(mx)+abs(my)+abs(mz)<=m)
				{
					if (!(mx==0 && my==0 && mz==0))
					{
						mlen2 = mx*mx + my*my + mz*mz;

						mlen = sqrt(mlen2);

						O(mx, my, mz, values);

						mult = erfc( alpha*mlen );

						for (int i = 0; i < 6; i++)
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

	for (int nx = -n; nx <= n; nx++)
	{
		for (int ny = -n; ny <= n; ny++)
		{
			for (int nz = -n; nz <= n; nz++)
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

void Qii(double a, double box_length, double alpha, int m, int n, double* answer)
{
	double mlen, mlen2, nlen, nlen2, mult;

	double* values = calloc(6, sizeof(double));

	double sqrt_pi = sqrt(M_PI);

	double alpha2 = alpha * alpha;

	double alpha3 = alpha * alpha2;

	double a3 = a * a * a;

	double box_length3 = box_length * box_length * box_length;

	for (int mx = -m; mx <= m; mx++)
	{
		for (int my = -m; my <= m; my++)
		{
			for (int mz = -m; mz <= m; mz++)
			{
				if (abs(mx)+abs(my)+abs(mz)<=m)
				{
					if (!(mx==0 && my==0 && mz==0))
					{
						mlen2 = mx*mx + my*my + mz*mz;

						mlen = sqrt(mlen2);

						Q(mx, my, mz, values);

						mult = erfc( alpha * mlen ) + 2 * alpha / sqrt_pi * mlen * exp( -alpha2 * mlen2 );

						for (int i = 0; i < 6; i++)
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

	for (int nx = -n; nx <= n; nx++)
	{
		for (int ny = -n; ny <= n; ny++)
		{
			for (int nz = -n; nz <= n; nz++)
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

void Oij(double sigmax, double sigmay, double sigmaz, double alpha, int m, int n, double* answer)
{
	double mslen, mslen2, nlen, nlen2, msx, msy, msz, nsdot, mult;

	double* values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double sqrt_pi = sqrt(M_PI);

	for (int mx = -m; mx <= m; mx++)
	{
		for (int my = -m; my <= m; my++)
		{
			for (int mz = -m; mz <= m; mz++)
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

					for (int i = 0; i < 6; i++)
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

	for (int nx = -n; nx <= n; nx++)
	{
		for (int ny = -n; ny <= n; ny++)
		{
			for (int nz = -n; nz <= n; nz++)
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

void Qij(double sigmax, double sigmay, double sigmaz, double alpha, int m, int n, double* answer)
{
	double mslen, mslen2, nlen, nlen2, msx, msy, msz, nsdot, mult;

	double* values = calloc(6, sizeof(double));

	double alpha2 = alpha * alpha;

	double alpha3 = alpha2 * alpha;

	double sqrt_pi = sqrt(M_PI);

	for (int mx = -m; mx <= m; mx++)
	{
		for (int my = -m; my <= m; my++)
		{
			for (int mz = -m; mz <= m; mz++)
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

					for (int i = 0; i < 6; i++)
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

	for (int nx = -n; nx <= n; nx++)
	{
		for (int ny = -n; ny <= n; ny++)
		{
			for (int nz = -n; nz <= n; nz++)
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