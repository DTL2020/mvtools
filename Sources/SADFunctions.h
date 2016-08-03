// Functions that computes distances between blocks

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

/*! \file SADFunctions.h
 *  \brief Calculate the sum of absolute differences between two blocks.
 *
 *	The sum of absolute differences between two blocks represents the distance
 *	between these two blocks. The lower this value, the closer the blocks will look like
 *  There are two versions, one in plain C, one in iSSE assembler.
 */

#ifndef __SAD_FUNC__
#define __SAD_FUNC__

#include "types.h"
#include <stdint.h>

typedef unsigned int (SADFunction)(const uint8_t *pSrc, int nSrcPitch,
								    const uint8_t *pRef, int nRefPitch);
SADFunction* get_sad_function(int BlockX, int BlockY, int pixelsize, arch_t arch);

inline unsigned int SADABS(int x) {	return ( x < 0 ) ? -x : x; }
//inline unsigned int SADABS(int x) {	return ( x < -16 ) ? 16 : ( x < 0 ) ? -x : ( x > 16) ? 16 : x; }

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
unsigned int Sad_C(const uint8_t *pSrc, int nSrcPitch,const uint8_t *pRef,
					     int nRefPitch)
{
	unsigned int sum = 0; // int is probably enough for 32x32
	for ( int y = 0; y < nBlkHeight; y++ )
	{
		for ( int x = 0; x < nBlkWidth; x++ )
			sum += SADABS(reinterpret_cast<const pixel_t *>(pSrc)[x] - reinterpret_cast<const pixel_t *>(pRef)[x]);
      pSrc += nSrcPitch;
      pRef += nRefPitch;
	}
	return sum;
}
template<int nBlkSize, typename pixel_t>
unsigned int Sad_C(const uint8_t *pSrc, int nSrcPitch,const uint8_t *pRef,
                    int nRefPitch)
{
   return Sad_C<nBlkSize, nBlkSize, pixel_t>(pSrc, pRef, nSrcPitch, nRefPitch);
}
// a litle more fast unrolled (Fizick)
/* //seems to be dead code (TSchniede)
inline unsigned int Sad2_C(const uint8_t *pSrc, const uint8_t *pRef,int nSrcPitch,
					     int nRefPitch)
{
	unsigned int sum = 0;
		sum += SADABS(pSrc[0] - pRef[0]);
		sum += SADABS(pSrc[1] - pRef[1]);
      pSrc += nSrcPitch;
      pRef += nRefPitch;
		sum += SADABS(pSrc[0] - pRef[0]);
		sum += SADABS(pSrc[1] - pRef[1]);
	return sum;
}

inline unsigned int Sad2x4_C(const uint8_t *pSrc, const uint8_t *pRef,int nSrcPitch,
					     int nRefPitch)
{
	unsigned int sum = 0;
		sum += SADABS(pSrc[0] - pRef[0]);
		sum += SADABS(pSrc[1] - pRef[1]);
      pSrc += nSrcPitch;
      pRef += nRefPitch;
		sum += SADABS(pSrc[0] - pRef[0]);
		sum += SADABS(pSrc[1] - pRef[1]);
      pSrc += nSrcPitch;
      pRef += nRefPitch;
		sum += SADABS(pSrc[0] - pRef[0]);
		sum += SADABS(pSrc[1] - pRef[1]);
      pSrc += nSrcPitch;
      pRef += nRefPitch;
		sum += SADABS(pSrc[0] - pRef[0]);
		sum += SADABS(pSrc[1] - pRef[1]);
	return sum;
}
*/
#define MK_CFUNC(functionname) extern "C" unsigned int __cdecl functionname (const uint8_t *pSrc, int nSrcPitch, const uint8_t *pRef, int nRefPitch)

#define SAD_ISSE(blsizex, blsizey) extern "C" unsigned int __cdecl Sad##blsizex##x##blsizey##_iSSE(const uint8_t *pSrc, int nSrcPitch, const uint8_t *pRef, int nRefPitch)
//Sad16x16_iSSE( x,y can be: 32 16 8 4 2
SAD_ISSE(32,32);
SAD_ISSE(16,32);
SAD_ISSE(32,16);
SAD_ISSE(32,8);
SAD_ISSE(16,16);
SAD_ISSE(16,8);
SAD_ISSE(16,4);
SAD_ISSE(16,2);
SAD_ISSE(16,1);
SAD_ISSE(8,16);
SAD_ISSE(8,8);
SAD_ISSE(8,4);
SAD_ISSE(8,2);
SAD_ISSE(8,1);
SAD_ISSE(4,8);
SAD_ISSE(4,4);
SAD_ISSE(4,2);
SAD_ISSE(2,4);
SAD_ISSE(2,2);

MK_CFUNC(Sad2x2_iSSE_T); //only work with packed, aligned source block copy
MK_CFUNC(Sad2x4_iSSE_T);
//MK_CFUNC(Sad2x2_iSSE_O);


#undef SAD_ISSE
/* //allows simple wrapping SAD functions, implementation in SADFunctions.h
#define MK_CPPWRAP(blkx, blky) unsigned int  Sad##blkx##x##blky##_wrap(const uint8_t *pSrc, int nSrcPitch, const uint8_t *pRef, int nRefPitch)
MK_CPPWRAP(16,16);
MK_CPPWRAP(16,8);
MK_CPPWRAP(16,2);
MK_CPPWRAP(8,16);
MK_CPPWRAP(8,8);
MK_CPPWRAP(8,4);
MK_CPPWRAP(8,2);
MK_CPPWRAP(8,1);
MK_CPPWRAP(4,8);
MK_CPPWRAP(4,4);
MK_CPPWRAP(4,2);
MK_CPPWRAP(2,4);
MK_CPPWRAP(2,2);
#undef MK_CPPWRAP
*/
/* included from x264 */
#define SAD_x264(blsizex, blsizey) extern "C" unsigned int __cdecl x264_pixel_sad_##blsizex##x##blsizey##_mmx2(const uint8_t *pSrc, int nSrcPitch, const uint8_t *pRef, int nRefPitch)
//x264_pixel_sad_16x16_mmx2(   x,y can be: 16 8 4
SAD_x264(16,16);
SAD_x264(16,8);
SAD_x264(8,16);
SAD_x264(8,8);
SAD_x264(8,4);
SAD_x264(4,8);
SAD_x264(4,4);
#undef SAD_x264
//parameter is function name
MK_CFUNC(x264_pixel_sad_16x16_sse2); //non optimized cache access, for AMD?
MK_CFUNC(x264_pixel_sad_16x8_sse2);	 //non optimized cache access, for AMD?
MK_CFUNC(x264_pixel_sad_16x16_sse3); //LDDQU Pentium4E (Core1?), not for Core2!
MK_CFUNC(x264_pixel_sad_16x8_sse3);  //LDDQU Pentium4E (Core1?), not for Core2!
MK_CFUNC(x264_pixel_sad_16x16_cache64_sse2);//core2 optimized
MK_CFUNC(x264_pixel_sad_16x8_cache64_sse2);//core2 optimized
MK_CFUNC(x264_pixel_sad_16x16_cache64_ssse3);//core2 optimized
MK_CFUNC(x264_pixel_sad_16x8_cache64_ssse3); //core2 optimized

//MK_CFUNC(x264_pixel_sad_16x16_cache32_mmx2);
//MK_CFUNC(x264_pixel_sad_16x8_cache32_mmx2);
MK_CFUNC(x264_pixel_sad_16x16_cache64_mmx2);
MK_CFUNC(x264_pixel_sad_16x8_cache64_mmx2);
MK_CFUNC(x264_pixel_sad_8x16_cache32_mmx2);
MK_CFUNC(x264_pixel_sad_8x8_cache32_mmx2);
MK_CFUNC(x264_pixel_sad_8x4_cache32_mmx2);
MK_CFUNC(x264_pixel_sad_8x16_cache64_mmx2);
MK_CFUNC(x264_pixel_sad_8x8_cache64_mmx2);
MK_CFUNC(x264_pixel_sad_8x4_cache64_mmx2);

//1.9.5.3: added ssd & SATD (TSchniede)
/* alternative to SAD - SSD: squared sum of differences, VERY sensitive to noise */
MK_CFUNC(x264_pixel_ssd_16x16_mmx);
MK_CFUNC(x264_pixel_ssd_16x8_mmx);
MK_CFUNC(x264_pixel_ssd_8x16_mmx);
MK_CFUNC(x264_pixel_ssd_8x8_mmx);
MK_CFUNC(x264_pixel_ssd_8x4_mmx);
MK_CFUNC(x264_pixel_ssd_4x8_mmx);
MK_CFUNC(x264_pixel_ssd_4x4_mmx);

/* SATD: Sum of Absolute Transformed Differences, more sensitive to noise, frequency domain based - replacement to dct/SAD */
MK_CFUNC(x264_pixel_satd_16x16_mmx2);
MK_CFUNC(x264_pixel_satd_16x8_mmx2);
MK_CFUNC(x264_pixel_satd_8x16_mmx2);
MK_CFUNC(x264_pixel_satd_8x8_mmx2);
MK_CFUNC(x264_pixel_satd_8x4_mmx2);
MK_CFUNC(x264_pixel_satd_4x8_mmx2);
MK_CFUNC(x264_pixel_satd_4x4_mmx2);

MK_CFUNC(x264_pixel_satd_32x32_mmx2);
MK_CFUNC(x264_pixel_satd_32x16_mmx2);

#define SATD_SSE(blsizex, blsizey, type) extern "C" unsigned int __cdecl x264_pixel_satd_##blsizex##x##blsizey##_##type##(const uint8_t *pSrc, int nSrcPitch, const uint8_t *pRef, int nRefPitch)

//x264_pixel_satd_16x16_%1
SATD_SSE(16, 16, sse2);
SATD_SSE(16,  8, sse2);
SATD_SSE( 8, 16, sse2);
SATD_SSE( 8,  8, sse2);
SATD_SSE( 8,  4, sse2);
SATD_SSE(16, 16, ssse3);
SATD_SSE(16,  8, ssse3);
SATD_SSE( 8, 16, ssse3);
SATD_SSE( 8,  8, ssse3);
SATD_SSE( 8,  4, ssse3);
#if 0
SATD_SSE(16, 16, ssse3_phadd); //identical to ssse3, for Penryn useful only?
SATD_SSE(16,  8, ssse3_phadd); //identical to ssse3
SATD_SSE( 8, 16, ssse3_phadd);
SATD_SSE( 8,  8, ssse3_phadd);
SATD_SSE( 8,  4, ssse3_phadd);
#endif

SATD_SSE(32, 32, sse2);
SATD_SSE(32, 16, sse2);
SATD_SSE(32, 32, ssse3);
SATD_SSE(32, 16, ssse3);
SATD_SSE(32, 32, ssse3_phadd);
SATD_SSE(32, 16, ssse3_phadd);
#undef SATD_SSE

//dummy for testing and deactivate SAD
MK_CFUNC(SadDummy);
#undef MK_CFUNC


#endif
