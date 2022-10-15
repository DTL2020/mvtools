// Functions that computes VIF DWT metrics between blocks

// See legal notice in Copying.txt for more information

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .


#ifndef __VIF_FUNC__
#define __VIF_FUNC__

#include "def.h"
#include "types.h"
#include <stdint.h>
#include "emmintrin.h"

struct DWT_DECOMP_INT {
  int a[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  int v[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  int h[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  int d[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
};

struct DWT_DECOMP_FLOAT {
  float a[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float v[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float h[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float d[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
};


VIFFunction* get_vif_function_a(int BlockX, int BlockY, int bits_per_pixel, arch_t arch);

VIFFunction* get_vif_function_e(int BlockX, int BlockY, int bits_per_pixel, arch_t arch);

VIFFunction* get_vif_function_full(int BlockX, int BlockY, int bits_per_pixel, arch_t arch);

/*
float mvt_ssim_full_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
float mvt_ssim_full_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
float mvt_ssim_full_16x16_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);

float mvt_ssim_l_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
float mvt_ssim_l_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
float mvt_ssim_l_16x16_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);

float mvt_ssim_cs_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
float mvt_ssim_cs_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
float mvt_ssim_cs_16x16_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
*/

#endif
