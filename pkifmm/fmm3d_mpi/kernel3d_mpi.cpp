/* Parallel Kernel Independent Fast Multipole Method
   Copyright (C) 2004 Lexing Ying,  New York University
   Copyright (C) 2010 Denis Zorin, New York University
   Copyright (C) 2010 George Biros, Harper Langston, Ilya Lashuk
   Copyright (C) 2010, Aparna Chandramowlishwaran, Aashay Shingrapure, Rich Vuduc

	 This program is free software; you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation; either version 2, or (at your option)
	 any later version.

	 This program is distributed in the hope that it will be useful, but WITHOUT
	 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
	 FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
	 for more details.

	 You should have received a copy of the GNU General Public License
	 along with this program; see the file COPYING.  If not, write to the Free
	 Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
	 02111-1307, USA.  */

#ifdef USE_SSE
#include <emmintrin.h>
#endif

#include <iostream>
using namespace std;
#include "common/vecmatop.hpp"
#include "kernel3d_mpi.hpp"
#include "comobject_mpi.hpp"

/** \file
 * This code implements the few Kernel3d_MPI functions.  For example,
 * based on kernelType, return source and target degrees of freedom.
 * Most importantly, based on the type of kernel being used
 * to do the evaluations, evaluate a multiplied context and the degrees of
 * freedom necessary in the buildKnlIntCtx function */


/* Most of this code is self-explanatory.  Based on the type of kernel being used
 * to do the evaluations, evaluate a multiplied context and the degrees of
 * freedom necessary */

/* Minimum difference */
double Kernel3d_MPI::_mindif = 1e-17;


// ----------------------------------------------------------------------
/* Depending on kernel type, set the degree of freedom for the sources
 * See kernel3d_mpi.hpp for kernelType descriptions */
int Kernel3d_MPI::srcDOF() const
{
  int dof = 0;
  switch(_kernelType) {
    //laplace kernels
	case KNL_LAP_S_U:  dof = 1; break;
	case KNL_LAP_D_U:  dof = 1; break;
	case KNL_LAP_I  :  dof = 1; break;
	case KNL_LAP_S_UG: dof = 1; break;

		//stokes kernels
	case KNL_STK_F_U: dof = 4; break;
	case KNL_STK_F_UG: dof = 4; break;
	case KNL_STK_S_U: dof = 3; break;
	case KNL_STK_S_UG: dof = 3; break;
	case KNL_STK_S_P: dof = 3; break;
	case KNL_STK_D_U: dof = 3; break;
	case KNL_STK_D_P: dof = 3; break;
	case KNL_STK_R_U: dof = 3; break;
	case KNL_STK_R_P: dof = 3; break;
	case KNL_STK_I  : dof = 3; break;
	case KNL_STK_E  : dof = 3; break;

		//navier kernels:	 
    // case KNL_NAV_F_U: dof = 3; break; //used for fmm
	case KNL_NAV_S_U: dof = 3; break;
	case KNL_NAV_D_U: dof = 3; break;
	case KNL_NAV_R_U: dof = 3; break;
	case KNL_NAV_I  : dof = 3; break;
	case KNL_NAV_E  : dof = 3; break;
		//others
	case KNL_LAP_S_UG_WD : dof = 5; break;
	case KNL_RPY : dof = 4; break;	
	case KNL_RPY2 : dof = 3; break;	
	
		//error
	case KNL_ERR:     dof = 0; break;
	default:
		abort();
  }
  return dof;
}

// ----------------------------------------------------------------------
/* Depending on the kernel type, set the degree of freedom for the target values
 * See kernel3d_mpi.hpp for kernelType descriptions */
int Kernel3d_MPI::trgDOF() const
{
  int dof = 0;
  switch(_kernelType) {
		//laplace kernels
  case KNL_LAP_S_U: dof = 1; break;
  case KNL_LAP_D_U: dof = 1; break;
  case KNL_LAP_I  : dof = 1; break;
  case KNL_LAP_S_UG: dof = 4; break;

		//stokes kernels
  case KNL_STK_F_U: dof = 3; break;
  case KNL_STK_F_UG: dof = 12; break;
  case KNL_STK_S_U: dof = 3; break;
  case KNL_STK_S_UG: dof = 12; break;
  case KNL_STK_S_P: dof = 1; break;
  case KNL_STK_D_U: dof = 3; break;
  case KNL_STK_D_P: dof = 1; break;
  case KNL_STK_R_U: dof = 3; break;
  case KNL_STK_R_P: dof = 1; break;
  case KNL_STK_I  : dof = 3; break;
  case KNL_STK_E  : dof = 3; break;
		//navier kernels:	 

		// case KNL_NAV_F_U: dof = 3; break; //used for fmm
  case KNL_NAV_S_U: dof = 3; break;
  case KNL_NAV_D_U: dof = 3; break;
  case KNL_NAV_R_U: dof = 3; break;
  case KNL_NAV_I  : dof = 3; break;
  case KNL_NAV_E  : dof = 3; break;
		//others
  case KNL_LAP_S_UG_WD : dof = 4; break;
  case KNL_RPY : dof = 3; break;	
  case KNL_RPY2 : dof = 3; break;
		
		//error
  case KNL_ERR:     dof = 0; break;
  }
  return dof;
}

// ---------------------------------------------------------------------- 
bool Kernel3d_MPI::homogeneous() const
{
  bool ret = false;
  switch(_kernelType) {
		//laplace kernels
  case KNL_LAP_S_U: ret = true; break;
		//stokes kernels
  case KNL_STK_F_U: ret = true; break;
		//stokes kernels
  case KNL_NAV_S_U: ret = true; break;
  default: assert(0);
  }
  return ret;
}

// ---------------------------------------------------------------------- 
void Kernel3d_MPI::homogeneousDeg(vector<double>& degVec) const
{
  switch(_kernelType) {
  case KNL_LAP_S_U: degVec.resize(1); degVec[0]=1; break;
  case KNL_STK_F_U: degVec.resize(4); degVec[0]=1; degVec[1]=1; degVec[2]=1; degVec[3]=2; break;
  case KNL_NAV_S_U: degVec.resize(3); degVec[0]=1; degVec[1]=1; degVec[2]=1; break;
  default: assert(0);
  }
  return;
}

#ifdef USE_SSE
namespace
{
#define IDEAL_ALIGNMENT 16
#define SIMD_LEN (IDEAL_ALIGNMENT / sizeof(double))
#define REG_BLOCK_SIZE 1
#define DECL_SIMD_ALIGNED  __declspec(align(IDEAL_ALIGNMENT))
#define OOEP_R  1.0/(8.0 * M_PI)

  void stokesDirectVecSSE(
													const int ns,
													const int nt,
													const double *sx,
													const double *sy,
													const double *sz,
													const double *tx,
													const double *ty,
													const double *tz,
													const double *srcDen,
													double *trgVal,
													const double cof )
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();
    double mu = cof;

    double OOEP = 1.0/(8.0*M_PI);
    __m128d tempx;
    __m128d tempy;
    __m128d tempz;
    double oomeu = 1/mu;
    
    double aux_arr[3*SIMD_LEN+1]; 
    double *tempvalx; 
    double *tempvaly; 
    double *tempvalz; 
    if (size_t(aux_arr)%IDEAL_ALIGNMENT)  // if aux_arr is misaligned
			{
				tempvalx = aux_arr + 1;
				if (size_t(tempvalx)%IDEAL_ALIGNMENT)
					abort();
			}
    else
      tempvalx = aux_arr;
    tempvaly=tempvalx+SIMD_LEN;
    tempvalz=tempvaly+SIMD_LEN;
    
    
    /*! One over eight pi */
    __m128d ooep = _mm_set1_pd (OOEP_R);
    __m128d half = _mm_set1_pd (0.5);
    __m128d one = _mm_set1_pd (1.0);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();
    __m128d oomu = _mm_set1_pd (1/mu);

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      tempx = _mm_setzero_pd();
      tempy = _mm_setzero_pd();
      tempz = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
				__m128d sxj = _mm_load_pd (&sx[j]);
				__m128d syj = _mm_load_pd (&sy[j]);
				__m128d szj = _mm_load_pd (&sz[j]); 
				__m128d sdenx = _mm_set_pd (srcDen[(j+1)*3],   srcDen[j*3]);
				__m128d sdeny = _mm_set_pd (srcDen[(j+1)*3+1], srcDen[j*3+1]);
				__m128d sdenz = _mm_set_pd (srcDen[(j+1)*3+2], srcDen[j*3+2]);

				__m128d dX, dY, dZ;
				__m128d dR2;
				__m128d S;

				dX = _mm_sub_pd(txi , sxj);
				dY = _mm_sub_pd(tyi , syj);
				dZ = _mm_sub_pd(tzi , szj);

				sxj = _mm_mul_pd(dX, dX); 
				syj = _mm_mul_pd(dY, dY);
				szj = _mm_mul_pd(dZ, dZ);

				dR2 = _mm_add_pd(sxj, syj);
				dR2 = _mm_add_pd(szj, dR2);
        __m128d temp = _mm_cmpeq_pd (dR2, zero);
        
				__m128d xhalf = _mm_mul_pd (half, dR2);
				__m128 dR2_s  =  _mm_cvtpd_ps(dR2);
				__m128 S_s    = _mm_rsqrt_ps(dR2_s);
				__m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (temp, S_d);

				S = _mm_mul_pd (S_d, S_d);
				S = _mm_mul_pd (S, xhalf);
				S = _mm_sub_pd (opf, S);
				S = _mm_mul_pd (S, S_d);

				__m128d dotx = _mm_mul_pd (dX, sdenx);
				__m128d doty = _mm_mul_pd (dY, sdeny);
				__m128d dotz = _mm_mul_pd (dZ, sdenz);

				__m128d dot_sum = _mm_add_pd (dotx, doty);
				dot_sum = _mm_add_pd (dot_sum, dotz);

				dot_sum = _mm_mul_pd (dot_sum, S);
				dot_sum = _mm_mul_pd (dot_sum, S);
				dotx = _mm_mul_pd (dot_sum, dX);
				doty = _mm_mul_pd (dot_sum, dY);
				dotz = _mm_mul_pd (dot_sum, dZ);

				sdenx = _mm_add_pd (sdenx, dotx);
				sdeny = _mm_add_pd (sdeny, doty);
				sdenz = _mm_add_pd (sdenz, dotz);

				sdenx = _mm_mul_pd (sdenx, S);
				sdeny = _mm_mul_pd (sdeny, S);
				sdenz = _mm_mul_pd (sdenz, S);

				tempx = _mm_add_pd (sdenx, tempx);
				tempy = _mm_add_pd (sdeny, tempy);
				tempz = _mm_add_pd (sdenz, tempz);

      }
      tempx = _mm_mul_pd (tempx, ooep);
      tempy = _mm_mul_pd (tempy, ooep);
      tempz = _mm_mul_pd (tempz, ooep);

      tempx = _mm_mul_pd (tempx, oomu);
      tempy = _mm_mul_pd (tempy, oomu);
      tempz = _mm_mul_pd (tempz, oomu);

      _mm_store_pd(tempvalx, tempx); 
      _mm_store_pd(tempvaly, tempy); 
      _mm_store_pd(tempvalz, tempz); 
      for (int k = 0; k < SIMD_LEN; k++) {
				trgVal[i*3]   += tempvalx[k];
				trgVal[i*3+1] += tempvaly[k];
				trgVal[i*3+2] += tempvalz[k];
      }

      for (; j < ns; j++) {
				double x = tx[i] - sx[j];
				double y = ty[i] - sy[j];
				double z = tz[i] - sz[j];
				double r2 = x*x + y*y + z*z;
				double r = sqrt(r2);
				double invdr;
				if (r == 0)
					invdr = 0;
				else 
					invdr = 1/r;
				double dot = (x*srcDen[j*3] + y*srcDen[j*3+1] + z*srcDen[j*3+2]) * invdr * invdr;
				double denx = srcDen[j*3] + dot*x;
				double deny = srcDen[j*3+1] + dot*y;
				double denz = srcDen[j*3+2] + dot*z;

				trgVal[i*3] += denx*invdr*OOEP*oomeu;
				trgVal[i*3+1] += deny*invdr*OOEP*oomeu;
				trgVal[i*3+2] += denz*invdr*OOEP*oomeu;
      }
    }

    return;
  }

  void stokesDirectSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], const double kernel_coef)
  {
#define X(s,k) (s)[(k)*DIM]
#define Y(s,k) (s)[(k)*DIM+1]
#define Z(s,k) (s)[(k)*DIM+2]
#define DIM 3

    vector<double> xs(ns+1);   vector<double> xt(nt);
    vector<double> ys(ns+1);   vector<double> yt(nt);
    vector<double> zs(ns+1);   vector<double> zt(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation 
    stokesDirectVecSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot,kernel_coef);
    return;
  }
}
#undef X
#undef Y
#undef Z
#undef LEN
#endif

#undef __FUNCT__
#define __FUNCT__ "Kernel3d_MPI::density2potential"
int Kernel3d_MPI::density2potential(const DblNumMat& srcPos, const DblNumVec& srcDen, const DblNumMat& srcNor, const DblNumMat& trgPos, DblNumVec& trgVal)
{
#define PI_8I 0.039788735772974
  int ns=srcPos.n();
  int nt=trgPos.n();
  double * src=srcPos.data();
  double * trg=trgPos.data();
  double * den=srcDen.data();
  double * pot=trgVal.data();
  double kernel_coef=_coefs[0];

  switch(_kernelType)
		{
    case KNL_STK_S_U:
#ifdef USE_SSE
      stokesDirectSSEShuffle(ns, nt, src, trg, den, pot, kernel_coef);
#else
      for (int t=0; t<nt; t++)
				{
					double p[3]={0,0,0};
					double tx=trg[3*t];
					double ty=trg[3*t+1];
					double tz=trg[3*t+2];

					for (int s=0; s<ns; s++)
						{
							double dX_reg=src[3*s]-tx;
							double dY_reg=src[3*s+1]-ty;
							double dZ_reg=src[3*s+2]-tz;

							double invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
							if (invR!=0)
								invR = 1.0/sqrt(invR);

							double cur_pot_x = den[3*s];
							double cur_pot_y = den[3*s+1];
							double cur_pot_z = den[3*s+2];

							double tmp_scalar = (dX_reg*cur_pot_x + dY_reg*cur_pot_y + dZ_reg*cur_pot_z)*invR*invR;
							cur_pot_x += tmp_scalar*dX_reg;
							cur_pot_y += tmp_scalar*dY_reg;
							cur_pot_z += tmp_scalar*dZ_reg;

							p[0] += cur_pot_x*invR;
							p[1] += cur_pot_y*invR;
							p[2] += cur_pot_z*invR;
						}
					pot[3*t] += p[0]*PI_8I*kernel_coef;
					pot[3*t+1] += p[1]*PI_8I*kernel_coef;
					pot[3*t+2] += p[2]*PI_8I*kernel_coef;
				}
#endif
      for (int i=0; i<nt*3; i++)
				if (pot[i]!=pot[i])
					abort();
      break;
    case KNL_STK_S_UG:
      // The output of this kernel is 12 numbers per point, stored
      // COLUMNWISE as 3x4 matrix; first column is the velocity;
      // second column is x-derivative of velocity, then column for
      // y-derivative and column for z-derivative

      for (int t=0; t<nt; t++)
				{
					double p[3][4] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };
					double tx=trg[3*t];
					double ty=trg[3*t+1];
					double tz=trg[3*t+2];

					for (int s=0; s<ns; s++)
						{
							double r[3] = { -src[3*s]+tx, -src[3*s+1]+ty, -src[3*s+2]+tz }; 

							double invR = 0;
							for(int i=0; i<3; i++)
								invR += r[i]*r[i];
							if (invR!=0)
								invR = 1.0/sqrt(invR);

							double mu[3] = { den[3*s], den[3*s+1], den[3*s+2] };
							double invR2 = invR*invR;
							double prod = 0;
							for(int i=0; i<3; i++)
								prod += r[i]*mu[i];
							double tmp_scalar = prod*invR2;

							for(int i=0; i<3; i++)
								p[i][0] += (mu[i] + tmp_scalar*r[i]) * invR;

							double invR3 = invR2*invR;
							for (int i=0; i<3; i++)
								for (int j=0; j<3; j++)
									p[i][j+1] += (r[i]*mu[j] - mu[i]*r[j] -  3*r[i]*r[j]*tmp_scalar)*invR3;

							for (int i=0; i<3; i++)
								p[i][i+1] += prod*invR3;
						}

					for (int i=0; i<3; i++)
						for (int j=0; j<4; j++)
							pot[12*t+3*j+i] += p[i][j]*PI_8I*kernel_coef;
				}
      break;
    case KNL_STK_F_UG:
      // the output of this kernel is 12 numbers per point, stored
      // COLUMNWISE as 3x4 matrix; first column is the velocity;
      // second column is x-derivative of velocity, then column for
      // y-derivative and column for z-derivative
      for (int t=0; t<nt; t++)
				{
					double p[3][4] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };
					double tx=trg[3*t];
					double ty=trg[3*t+1];
					double tz=trg[3*t+2];

					for (int s=0; s<ns; s++)
						{
							double r[3] = { -src[3*s]+tx, -src[3*s+1]+ty, -src[3*s+2]+tz }; 

							double invR = 0;
							for(int i=0; i<3; i++)
								invR += r[i]*r[i];
							if (invR!=0)
								invR = 1.0/sqrt(invR);

							double mu[4] = { den[4*s], den[4*s+1], den[4*s+2], den[4*s+3] };
							double invR2 = invR*invR;
							double prod = 0;
							for(int i=0; i<3; i++)
								prod += r[i]*mu[i];
							prod += mu[3]*2/kernel_coef;
							double tmp_scalar = prod*invR2;

							for(int i=0; i<3; i++)
								p[i][0] += (mu[i] + tmp_scalar*r[i]) * invR;

							double invR3 = invR2*invR;
							for (int i=0; i<3; i++)
								for (int j=0; j<3; j++)
									p[i][j+1] += (r[i]*mu[j] - mu[i]*r[j] -  3*r[i]*r[j]*tmp_scalar)*invR3;

							for (int i=0; i<3; i++)
								p[i][i+1] += prod*invR3;
						}

					for (int i=0; i<3; i++)
						for (int j=0; j<4; j++)
							pot[12*t+3*j+i] += p[i][j]*PI_8I*kernel_coef;
				}
      break;
    case KNL_STK_F_U:
      for (int t=0; t<nt; t++)
				{
					double p[3]={0,0,0};
					double tx=trg[3*t];
					double ty=trg[3*t+1];
					double tz=trg[3*t+2];

					for (int s=0; s<ns; s++)
						{
							double dX_reg=src[3*s]-tx;
							double dY_reg=src[3*s+1]-ty;
							double dZ_reg=src[3*s+2]-tz;

							double invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
							if (invR!=0)
								invR = 1.0/sqrt(invR);

							double cur_pot_x = den[4*s];
							double cur_pot_y = den[4*s+1];
							double cur_pot_z = den[4*s+2];
							double cur_pot_w = den[4*s+3];

							double tmp_scalar = (dX_reg*cur_pot_x + dY_reg*cur_pot_y + dZ_reg*cur_pot_z - 2/kernel_coef*cur_pot_w)*invR*invR;
							cur_pot_x += tmp_scalar*dX_reg;
							cur_pot_y += tmp_scalar*dY_reg;
							cur_pot_z += tmp_scalar*dZ_reg;

							p[0] += cur_pot_x*invR;
							p[1] += cur_pot_y*invR;
							p[2] += cur_pot_z*invR;
						}
					pot[3*t] += p[0]*PI_8I*kernel_coef;
					pot[3*t+1] += p[1]*PI_8I*kernel_coef;
					pot[3*t+2] += p[2]*PI_8I*kernel_coef;
				}
      break;
  case KNL_RPY: {
		//const double OOFP = 1.0/(4.0*M_PI);
		const double OOFP = 1.0;
				for (int t=0; t<nt; t++)
					{
						double p[3]={0,0,0};
						double tx=trg[3*t];
						double ty=trg[3*t+1];
						double tz=trg[3*t+2];

						for (int s=0; s<ns; s++)
							{
								
								double distance[3];
								//this or negative? : Doesn't matter!
								distance[0]=src[3*s]-tx;
								distance[1]=src[3*s+1]-ty;
								distance[2]=src[3*s+2]-tz;
								
								double invR = (distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2]);
								if (invR!=0)
									invR = 1.0/sqrt(invR);
								
								double invR2 = invR * invR;
								double invR3 = invR2 * invR;	
								
								double C1 = 0.75;
								double C2 = 0.25;
								double identity = 0.0;
								
								double effectiveRadius1 = den[4*s + 0] * den[4*s + 0];

								for(int ii=0;ii<3;ii++){
									for(int jj=0;jj<3;jj++){
										identity = (ii==jj)? 1.0 : 0.0;
										p[ii] += C1 * invR * den[4*s+1+jj] *
											(identity + ((distance[ii] * distance[jj]) * invR2));
										p[ii] += C2 * invR3 * effectiveRadius1 * den[4*s+1+jj] * 
														(identity - (3 * (distance[ii] * distance[jj]) * invR2));
									}
								}
								//cout<<"1) --------- "<<den[4*s + 1]<<" "<<den[4*s + 2]<<" "<<den[4*s + 3]<<endl;;					
							}
						pot[3*t]   += p[0]*OOFP*kernel_coef;
						pot[3*t+1] += p[1]*OOFP*kernel_coef;
						pot[3*t+2] += p[2]*OOFP*kernel_coef;
					}
	}
	break;
	
	
	case KNL_RPY2: {
		//const double OOFP = 1.0/(4.0*M_PI);
		const double OOFP = 1.0;
				for (int t=0; t<nt; t++)
					{
						double p[3]={0.0,0.0,0.0};
						double tx=trg[3*t];
						double ty=trg[3*t+1];
						double tz=trg[3*t+2];

						for (int s=0; s<ns; s++)
							{
								
								double distance[3];
								//this or negative? : Doesn't matter!
								distance[0]=src[3*s]-tx;
								distance[1]=src[3*s+1]-ty;
								distance[2]=src[3*s+2]-tz;
								
								double invR = (distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2]);
								if (invR!=0)
									invR = 1.0/sqrt(invR);
								
								double invR2 = invR * invR;
								double invR3 = invR2 * invR;	
								
								double C2 = 0.25;
								double identity = 0.0;
								

								for(int ii=0;ii<3;ii++){
									for(int jj=0;jj<3;jj++){
										identity = (ii==jj)? 1.0 : 0.0;
										p[ii] += C2 * invR3 * den[3*s+jj] * 
														(identity - (3 * (distance[ii] * distance[jj]) * invR2));
									}
								}
								//cout<<"2) --------- "<<den[3*s + 0]<<" "<<den[3*s + 1]<<" "<<den[3*s + 2]<<endl;;					
						}
						pot[3*t]   += p[0]*OOFP*kernel_coef;
						pot[3*t+1] += p[1]*OOFP*kernel_coef;
						pot[3*t+2] += p[2]*OOFP*kernel_coef;
					}
	}
	break;


    case KNL_LAP_S_UG_WD: {
		//const double OOFP = 1.0/(4.0*M_PI);
		const double OOFP = 1.0;
				for (int t=0; t<nt; t++)
					{
						double p[4]={0,0,0,0};
						double tx=trg[3*t];
						double ty=trg[3*t+1];
						double tz=trg[3*t+2];

						for (int s=0; s<ns; s++)
							{
								double dX_reg=src[3*s]-tx;
								double dY_reg=src[3*s+1]-ty;
								double dZ_reg=src[3*s+2]-tz;

								double invR = (dX_reg*dX_reg + dY_reg*dY_reg + dZ_reg*dZ_reg);
								if (invR!=0)
									invR = 1.0/sqrt(invR);

								double phi = den[5*s + 0]*invR;
								
								double phiDip = den[5*s+2]*dX_reg + den[5*s+3]*dY_reg + den[5*s+4]*dZ_reg;
								phiDip *= den[5*s+1];
								phiDip *= invR * invR * invR;
								
								double phiDipTemp =  den[5*s+1]*invR*invR*invR;
								 
								p[0] += phi + phiDip;
								p[1] += (phi*dX_reg*invR*invR + 3*phiDip*dX_reg*invR*invR) - (phiDipTemp*den[5*s+2]);
								p[2] += (phi*dY_reg*invR*invR + 3*phiDip*dY_reg*invR*invR) - (phiDipTemp*den[5*s+3]);
								p[3] += (phi*dZ_reg*invR*invR + 3*phiDip*dZ_reg*invR*invR) - (phiDipTemp*den[5*s+4]);
							}
						pot[4*t]   += p[0]*OOFP*kernel_coef;
						pot[4*t+1] += p[1]*OOFP*kernel_coef;
						pot[4*t+2] += p[2]*OOFP*kernel_coef;
						pot[4*t+3] += p[3]*OOFP*kernel_coef;
					}
	}
	break;
    case KNL_LAP_S_UG:
      {
				const double OOFP = 1.0/(4.0*M_PI);
				for (int t=0; t<nt; t++)
					{
						double p[4]={0,0,0,0};
						double tx=trg[3*t];
						double ty=trg[3*t+1];
						double tz=trg[3*t+2];

						for (int s=0; s<ns; s++)
							{
								double dX_reg=src[3*s]-tx;
								double dY_reg=src[3*s+1]-ty;
								double dZ_reg=src[3*s+2]-tz;

								double invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
								if (invR!=0)
									invR = 1.0/sqrt(invR);

								double phi = den[s]*invR;
								p[0] += phi;
								p[1] += phi*dX_reg*invR*invR;
								p[2] += phi*dY_reg*invR*invR;
								p[3] += phi*dZ_reg*invR*invR;
							}
						pot[4*t]   += p[0]*OOFP*kernel_coef;
						pot[4*t+1] += p[1]*OOFP*kernel_coef;
						pot[4*t+2] += p[2]*OOFP*kernel_coef;
						pot[4*t+3] += p[3]*OOFP*kernel_coef;
					}
      }
      break;
    case KNL_LAP_S_U:
      {
				const double OOFP = 1.0/(4.0*M_PI);
				for (int t=0; t<nt; t++)
					{
						double p=0;
						double tx=trg[3*t];
						double ty=trg[3*t+1];
						double tz=trg[3*t+2];

						for (int s=0; s<ns; s++)
							{
								double dX_reg=src[3*s]-tx;
								double dY_reg=src[3*s+1]-ty;
								double dZ_reg=src[3*s+2]-tz;

								double invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
								if (invR!=0)
									invR = 1.0/sqrt(invR);

								p += den[s]*invR;
							}
						pot[t] += p*OOFP*kernel_coef;
					}
      }
      break;
    default:
      abort();
		}
  return 0;
}

// ----------------------------------------------------------------------
/* This is the main function for Kernel3d_MPI.
 * Given source positions, normals and target positions, build a matrix inter
 * which will be used as a multiplier during the multiplication phase.
 * This functino is just a set of cases, based on the kernelType variable.
 * Mathematical details on these kernels are available in the papers
 */
#undef __FUNCT__
#define __FUNCT__ "Kernel3d_MPI::kernel"
int Kernel3d_MPI::buildKnlIntCtx(const DblNumMat& srcPos, const DblNumMat& srcNor, const DblNumMat& trgPos, DblNumMat& inter)
{
  //begin
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();
  pA(srcPos.m()==dim() && srcNor.m()==dim() && trgPos.m()==dim());
  pA(srcPos.n()==srcNor.n());
  pA(srcPos.n()*srcDOF == inter.n());
  pA(trgPos.n()*trgDOF == inter.m());
  /* Single-Layer Laplacian Position/Velocity */
  if(       _kernelType==KNL_LAP_S_U) {
		//----------------------------------------------------------------------------------
		//---------------------------------
		double OOFP = 1.0/(4.0*M_PI);
		for(int i=0; i<trgPos.n(); i++) {
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  				  double r = sqrt(r2);
				if(r<_mindif) {
					inter(i,j) = 0;
				} else {
					inter(i,j) = OOFP / r;
				}
			}
		}
  }
  /* Double-Layer Laplacian Position/Velocity */
  else if(_kernelType==KNL_LAP_D_U) {
		//---------------------------------
		double OOFP = 1.0/(4.0*M_PI);
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
					double rn = x*nx + y*ny + z*nz;
					double r3 = r2*r;
					inter(i,j) = - OOFP / r3 * rn;
				}
			}
  }
  /* Laplacian Identity Kernel */
  else if(_kernelType==KNL_LAP_I) {
		//---------------------------------
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++)
				inter(i,j) = 1;
  }
  /* Stokes - Fmm3d velocity kernel */
  else if(_kernelType==KNL_STK_F_U) {
		//----------------------------------------------------------------------------------
		//---------------------------------
		pA(_coefs.size()>=1);
		double mu = _coefs[0];
		double OOFP = 1.0/(4.0*M_PI);
		double OOEP = 1.0/(8.0*M_PI);
		double oomu = 1.0/mu;
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double r3 = r2*r;
					double G = oomu * OOEP / r;
					double H = oomu * OOEP / r3;
					double A = OOFP / r3;
					int is = i*3;			 int js = j*4;
					inter(is,   js)   = G + H*x*x; inter(is,   js+1) =     H*x*y; inter(is,   js+2) =     H*x*z; inter(is,   js+3) = A*x;
					inter(is+1, js)   =     H*y*x; inter(is+1, js+1) = G + H*y*y; inter(is+1, js+2) =     H*y*z; inter(is+1, js+3) = A*y;
					inter(is+2, js)   =     H*z*x; inter(is+2, js+1) =     H*z*y; inter(is+2, js+2) = G + H*z*z; inter(is+2, js+3) = A*z;
				}
			}
  }
  /* Stokes Single-Layer Position/Velocity kernel */
  else if(_kernelType==KNL_STK_S_U) {
		//---------------------------------
		pA(_coefs.size()>=1);
		double mu = _coefs[0];
		double OOEP = 1.0/(8.0*M_PI);
		double oomu = 1.0/mu;
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double r3 = r2*r;
					double G = OOEP / r;
					double H = OOEP / r3;
					inter(i*3,   j*3)   = oomu*(G + H*x*x); inter(i*3,   j*3+1) = oomu*(    H*x*y); inter(i*3,   j*3+2) = oomu*(    H*x*z);
					inter(i*3+1, j*3)   = oomu*(    H*y*x); inter(i*3+1, j*3+1) = oomu*(G + H*y*y); inter(i*3+1, j*3+2) = oomu*(    H*y*z);
					inter(i*3+2, j*3)   = oomu*(    H*z*x); inter(i*3+2, j*3+1) = oomu*(    H*z*y); inter(i*3+2, j*3+2) = oomu*(G + H*z*z);
				}
			}
  }
  /* Stokes Single-Layer Pressure kernel */
  else if(_kernelType==KNL_STK_S_P) {
		//---------------------------------
		pA(_coefs.size()>=1);
		// double mu = _coefs[0];
		double OOFP = 1.0/(4.0*M_PI);
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double r3 = r2*r;
					inter(i  ,j*3  ) = OOFP*x/r3;			 inter(i  ,j*3+1) = OOFP*y/r3;			 inter(i  ,j*3+2) = OOFP*z/r3;
				}
			}
  }
  /* Stokes Double-Layer Position/Velocity kernel */
  else if(_kernelType==KNL_STK_D_U) {
		//---------------------------------
		pA(_coefs.size()>=1);
		// double mu = _coefs[0];
		double SOEP = 6.0/(8.0*M_PI);
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
					double rn = x*nx + y*ny + z*nz;
					double r5 = r2*r2*r;
					double C = - SOEP / r5;
					inter(i*3,   j*3)   = C*rn*x*x;				  inter(i*3,   j*3+1) = C*rn*x*y;				  inter(i*3,   j*3+2) = C*rn*x*z;
					inter(i*3+1, j*3)   = C*rn*y*x;				  inter(i*3+1, j*3+1) = C*rn*y*y;				  inter(i*3+1, j*3+2) = C*rn*y*z;
					inter(i*3+2, j*3)   = C*rn*z*x;				  inter(i*3+2, j*3+1) = C*rn*z*y;				  inter(i*3+2, j*3+2) = C*rn*z*z;
				}
			}
  }
  /* Stokes Double-Layer Pressure kernel */
  else if(_kernelType==KNL_STK_D_P) {
		//---------------------------------
		pA(_coefs.size()>=1);
		double mu = _coefs[0];
		double OOTP = 1.0/(2.0*M_PI);
		double coef = mu*OOTP;
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
					double rn = x*nx + y*ny + z*nz;
					double r3 = r2*r;
					double r5 = r3*r2;
					int is = i;			 int js = j*3;
					inter(is  ,js  ) = coef*(nx/r3 - 3*rn*x/r5);
					inter(is  ,js+1) = coef*(ny/r3 - 3*rn*y/r5);
					inter(is  ,js+2) = coef*(nz/r3 - 3*rn*z/r5);
				}
			}
  }
  else if(_kernelType==KNL_STK_R_U) {
		//---------------------------------
		pA(_coefs.size()>=1);
		double mu = _coefs[0];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double r3 = r2*r;
					double coef = 1.0/(8.0*M_PI*mu)/r3;
					inter(i*3,   j*3)   = coef*0;			 inter(i*3,   j*3+1) = coef*z;			 inter(i*3,   j*3+2) = coef*(-y);
					inter(i*3+1, j*3)   = coef*(-z);		 inter(i*3+1, j*3+1) = coef*0;			 inter(i*3+1, j*3+2) = coef*x;
					inter(i*3+2, j*3)   = coef*y;			 inter(i*3+2, j*3+1) = coef*(-x);		 inter(i*3+2, j*3+2) = coef*0;
				}
			}
  } else if(_kernelType==KNL_STK_R_P) {
		//---------------------------------
		pA(_coefs.size()>=1);
		// double mu = _coefs[0];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				inter(i,j*3  ) = 0;
				inter(i,j*3+1) = 0;
				inter(i,j*3+2) = 0;
			}
  } else if(_kernelType==KNL_STK_I) {
		//---------------------------------
		pA(_coefs.size()>=1);
		// double mu = _coefs[0];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				inter(i*3,   j*3  ) = 1;		 inter(i*3,   j*3+1) = 0;		 inter(i*3,   j*3+2) = 0;
				inter(i*3+1, j*3  ) = 0;		 inter(i*3+1, j*3+1) = 1;		 inter(i*3+1, j*3+2) = 0;
				inter(i*3+2, j*3  ) = 0;		 inter(i*3+2, j*3+1) = 0;		 inter(i*3+2, j*3+2) = 1;
			}
  }
  /* levi-civita tensor */
  else if(_kernelType==KNL_STK_E) {
		//---------------------------------
		pA(_coefs.size()>=1);
		// double mu = _coefs[0];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				inter(i*3,   j*3  ) = 0;		 inter(i*3,   j*3+1) = z;		 inter(i*3,   j*3+2) = -y;
				inter(i*3+1, j*3  ) = -z;  	 inter(i*3+1, j*3+1) = 0;		 inter(i*3+1, j*3+2) = x;
				inter(i*3+2, j*3  ) = y;		 inter(i*3+2, j*3+1) = -x;		 inter(i*3+2, j*3+2) = 0;
			}
  }
  /* Navier-Stokes Single-Layer Position/Velocity Kernel */
  else if(_kernelType==KNL_NAV_S_U) {
		//----------------------------------------------------------------------------------
		//---------------------------------
		pA(_coefs.size()>=2);
		double mu = _coefs[0];	 double ve = _coefs[1];
		double sc1 = (3.0-4.0*ve)/(16.0*M_PI*(1.0-ve));
		double sc2 = 1.0/(16.0*M_PI*(1.0-ve));
		double oomu = 1.0/mu;
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double r3 = r2*r;
					double G = sc1 / r;
					double H = sc2 / r3;
					inter(i*3,   j*3)   = oomu*(G + H*x*x);  inter(i*3,   j*3+1) = oomu*(    H*x*y);  inter(i*3,   j*3+2) = oomu*(    H*x*z);
					inter(i*3+1, j*3)   = oomu*(    H*y*x);  inter(i*3+1, j*3+1) = oomu*(G + H*y*y);  inter(i*3+1, j*3+2) = oomu*(    H*y*z);
					inter(i*3+2, j*3)   = oomu*(    H*z*x);  inter(i*3+2, j*3+1) = oomu*(    H*z*y);  inter(i*3+2, j*3+2) = oomu*(G + H*z*z);
				}
			}
  }
  /* Navier-Stokes Double-Layer Position/Velocity Kernel */
  else if(_kernelType==KNL_NAV_D_U) {
		//---------------------------------
		pA(_coefs.size()>=2);
		// double mu = _coefs[0];	 
		double ve = _coefs[1];
		double dc1 = -(1-2.0*ve)/(8.0*M_PI*(1.0-ve));
		double dc2 =  (1-2.0*ve)/(8.0*M_PI*(1.0-ve));
		double dc3 = -3.0/(8.0*M_PI*(1.0-ve));
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
					double rn = x*nx + y*ny + z*nz;
					double r3 = r2*r;
					double r5 = r3*r2;
					double A = dc1 / r3;			 double B = dc2 / r3;			 double C = dc3 / r5;
					double rx = x;			 double ry = y;			 double rz = z;
					//&&&&&&&
					inter(i*3,   j*3)   = A*(rn+nx*rx) + B*(rx*nx) + C*rn*rx*rx;
					inter(i*3,   j*3+1) = A*(   nx*ry) + B*(rx*ny) + C*rn*rx*ry;
					inter(i*3,   j*3+2) = A*(   nx*rz) + B*(rx*nz) + C*rn*rx*rz;
					inter(i*3+1, j*3)   = A*(   ny*rx) + B*(ry*nx) + C*rn*ry*rx;
					inter(i*3+1, j*3+1) = A*(rn+ny*ry) + B*(ry*ny) + C*rn*ry*ry;
					inter(i*3+1, j*3+2) = A*(   ny*rz) + B*(ry*nz) + C*rn*ry*rz;
					inter(i*3+2, j*3)   = A*(   nz*rx) + B*(rz*nx) + C*rn*rz*rx;
					inter(i*3+2, j*3+1) = A*(   nz*ry) + B*(rz*ny) + C*rn*rz*ry;
					inter(i*3+2, j*3+2) = A*(rn+nz*rz) + B*(rz*nz) + C*rn*rz*rz;
				}
			}
  } else if(_kernelType==KNL_NAV_R_U) {
		//---------------------------------
		pA(_coefs.size()>=2);
		double mu = _coefs[0];	 // double ve = _coefs[1];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
				if(r<_mindif) {
					for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
				} else {
					double r3 = r2*r;
					double coef = 1.0/(8.0*M_PI*mu)/r3;
					inter(i*3,   j*3)   = coef*0;			 inter(i*3,   j*3+1) = coef*z;			 inter(i*3,   j*3+2) = coef*(-y);
					inter(i*3+1, j*3)   = coef*(-z);		 inter(i*3+1, j*3+1) = coef*0;			 inter(i*3+1, j*3+2) = coef*x;
					inter(i*3+2, j*3)   = coef*y;			 inter(i*3+2, j*3+1) = coef*(-x);		 inter(i*3+2, j*3+2) = coef*0;
				}
			}
  } else if(_kernelType==KNL_NAV_I) {
		//---------------------------------
		pA(_coefs.size()>=2);
		// double mu = _coefs[0];	 
		// double ve = _coefs[1];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				inter(i*3,   j*3  ) = 1;		 inter(i*3,   j*3+1) = 0;		 inter(i*3,   j*3+2) = 0;
				inter(i*3+1, j*3  ) = 0;		 inter(i*3+1, j*3+1) = 1;		 inter(i*3+1, j*3+2) = 0;
				inter(i*3+2, j*3  ) = 0;		 inter(i*3+2, j*3+1) = 0;		 inter(i*3+2, j*3+2) = 1;
			}
  } else if(_kernelType==KNL_NAV_E) {
		//---------------------------------
		pA(_coefs.size()>=2);
		// double mu = _coefs[0];	 
		// double ve = _coefs[1];
		for(int i=0; i<trgPos.n(); i++)
			for(int j=0; j<srcPos.n(); j++) {
				double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
				inter(i*3,   j*3  ) = 0;		 inter(i*3,   j*3+1) = z;		 inter(i*3,   j*3+2) = -y;
				inter(i*3+1, j*3  ) = -z;  	 inter(i*3+1, j*3+1) = 0;		 inter(i*3+1, j*3+2) = x;
				inter(i*3+2, j*3  ) = y;		 inter(i*3+2, j*3+1) = -x;		 inter(i*3+2, j*3+2) = 0;
			}
  } else if(_kernelType==KNL_ERR) {
		//---------------------------------
		pA(0);
  }
  return(0);
}




