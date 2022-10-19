// Functions that computes SAD and Covar metrics between blocks

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


#ifndef __SADCOVAR_FUNC__
#define __SADCOVAR_FUNC__

#include "def.h"
#include "types.h"
#include <stdint.h>
#include "emmintrin.h"


SADCOVARFunction* get_sadcovar_function(int BlockX, int BlockY, int bits_per_pixel, arch_t arch);


float mvt_sadcovar_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch, float* pfCov);
float mvt_sadcovar_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch, float* pfCov);
float mvt_sadcovar_16x16_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch, float* pfCov);


#endif
