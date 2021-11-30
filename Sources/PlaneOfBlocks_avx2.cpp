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

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined(__INTEL_LLVM_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined(__INTEL_LLVM_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined (__FMA__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER))  && ! defined (__INTEL_COMPILER) 
#include <fmaintrin.h>
#endif // __FMA__ 

#ifndef _mm256_loadu2_m128i
#define _mm256_loadu2_m128i(/* __m128i const* */ hiaddr, \
                            /* __m128i const* */ loaddr) \
    _mm256_set_m128i(_mm_loadu_si128(hiaddr), _mm_loadu_si128(loaddr))
#endif

#ifndef _mm_storeu_si16
#define _mm_storeu_si16(p, a) (void)(*(short*)(p) = (short)_mm_cvtsi128_si32((a)))
#endif

#define _mm256_mpsadbw_8_2(Ref_2, Src_2) _mm256_adds_epu16(_mm256_mpsadbw_epu8(Ref_2, Src_2, 0), _mm256_mpsadbw_epu8(Ref_2, Src_2, 45))

#define Sads_block_8x8 \
	ymm_block_ress = _mm256_mpsadbw_8_2(ymm0_Ref_01, ymm4_Src_01); \
	ymm_block_ress = _mm256_adds_epu16(ymm_block_ress, _mm256_mpsadbw_8_2(ymm1_Ref_23, ymm5_Src_23)); \
	ymm_block_ress = _mm256_adds_epu16(ymm_block_ress, _mm256_mpsadbw_8_2(ymm2_Ref_45, ymm6_Src_45)); \
	ymm_block_ress = _mm256_adds_epu16(ymm_block_ress, _mm256_mpsadbw_8_2(ymm3_Ref_67, ymm7_Src_67)); \
	ymm_block_ress = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm_block_ress, 1)), ymm_block_ress);

#define Push_Ref_8x8_row(push_row) \
	ymm0_Ref_01 = _mm256_permute2x128_si256(ymm0_Ref_01, ymm1_Ref_23, 33); \
	ymm1_Ref_23 = _mm256_permute2x128_si256(ymm1_Ref_23, ymm2_Ref_45, 33); \
	ymm2_Ref_45 = _mm256_permute2x128_si256(ymm2_Ref_45, ymm3_Ref_67, 33); \
	ymm3_Ref_67 = _mm256_permute2x128_si256(ymm3_Ref_67, ymm3_Ref_67, 17); \
	ymm3_Ref_67 = _mm256_inserti128_si256(ymm3_Ref_67, _mm_loadu_si128((__m128i*)(pucRef + nRefPitch[0] * push_row)), 1);

#define Half_16x16_sads \
	ymm8_Src0 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (0 + iSrcShift)))), 80); \
	ymm9_Src1 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (1 + iSrcShift)))), 80); \
	ymm10_Src2 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (2 + iSrcShift)))), 80); \
	ymm11_Src3 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (3 + iSrcShift)))), 80); \
	ymm12_Src4 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (4 + iSrcShift)))), 80); \
	ymm13_Src5 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (5 + iSrcShift)))), 80); \
	ymm14_Src6 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (6 + iSrcShift)))), 80); \
	ymm15_Src7 = _mm256_permute4x64_epi64(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(pucCurr + nSrcPitch[0] * (7 + iSrcShift)))), 80); \
 \
	ymm0_Ref0 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (0 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (0 + iRefShift))); \
	ymm1_Ref1 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (1 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (1 + iRefShift))); \
	ymm2_Ref2 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (2 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (2 + iRefShift))); \
	ymm3_Ref3 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (3 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (3 + iRefShift))); \
	ymm4_Ref4 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (4 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (4 + iRefShift))); \
	ymm5_Ref5 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (5 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (5 + iRefShift))); \
	ymm6_Ref6 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (6 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (6 + iRefShift))); \
	ymm7_Ref7 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (7 + iRefShift) + 8), (__m128i*)(pucRef + nRefPitch[0] * (7 + iRefShift))); \
\
	ymm0_Ref0 = _mm256_mpsadbw_8_2(ymm0_Ref0, ymm8_Src0); \
	ymm1_Ref1 = _mm256_mpsadbw_8_2(ymm1_Ref1, ymm9_Src1); \
	ymm2_Ref2 = _mm256_mpsadbw_8_2(ymm2_Ref2, ymm10_Src2); \
	ymm3_Ref3 = _mm256_mpsadbw_8_2(ymm3_Ref3, ymm11_Src3); \
	ymm4_Ref4 = _mm256_mpsadbw_8_2(ymm4_Ref4, ymm12_Src4); \
	ymm5_Ref5 = _mm256_mpsadbw_8_2(ymm5_Ref5, ymm13_Src5); \
	ymm6_Ref6 = _mm256_mpsadbw_8_2(ymm6_Ref6, ymm14_Src6); \
	ymm7_Ref7 = _mm256_mpsadbw_8_2(ymm7_Ref7, ymm15_Src7); \
 \
	ymm0_Ref0 = _mm256_adds_epu16(ymm0_Ref0, ymm1_Ref1); \
	ymm2_Ref2 = _mm256_adds_epu16(ymm2_Ref2, ymm3_Ref3); \
	ymm4_Ref4 = _mm256_adds_epu16(ymm4_Ref4, ymm5_Ref5); \
	ymm6_Ref6 = _mm256_adds_epu16(ymm6_Ref6, ymm7_Ref7); \
 \
	ymm0_Ref0 = _mm256_adds_epu16(ymm0_Ref0, ymm2_Ref2); \
	ymm4_Ref4 = _mm256_adds_epu16(ymm4_Ref4, ymm6_Ref6); \
\
 	ymm0_Ref0 = _mm256_adds_epu16(ymm0_Ref0, ymm4_Ref4); \
	ymm_half16x16_sads = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm0_Ref0, 1)), ymm0_Ref0);


#include "PlaneOfBlocks.h"
#include <map>
#include <tuple>

#include <stdint.h>
#include "def.h"

// prefetches still experimental and need tests if helps and for best locality hints (NTA hint may be not best for large frames ?)
// source block prefetch looks like not needed if ALIGNSOURCEBLOCK > 1

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp4_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !
    // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 4, mvy - 4))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 3, mvy + 4))
  {
    return;
  }
  /*
  if (!workarea.IsVectorOK(mvx - 4, mvy + 3))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 3, mvy - 4))
  {
    return;
  }
  */
  // still 8 H search positions, so -4..+3.
//  unsigned short ArrSADs[8][9];
  alignas(16) __m128i Arr128iSADs[9];
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 4, mvy - 4); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];


  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // 2x12bytes store, require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_tmp, ymm5_tmp;

  __m256i ymm10_Src_2031, ymm11_Src_6475;
  __m128i xmm10_Src_20, xmm11_Src_31, xmm12_Src_64, xmm13_Src_75;

  __m256i ymm6_part_sads = _mm256_setzero_si256();

  __m256i ymm7_minsad8; // vectors of minsads for SSE4.1 _mm_minpos_epu16() minsad and pos search

#ifdef _DEBUG
  ymm7_minsad8 = _mm256_setzero_si256(); // to prevent debug break on non-init reg, it fully filled later.
#endif

  xmm10_Src_20 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 2)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  xmm11_Src_31 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 1)));
  ymm10_Src_2031 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm10_Src_20), _mm256_castsi128_si256(xmm11_Src_31), 32);

  xmm12_Src_64 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 6)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  xmm13_Src_75 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 5)));
  ymm11_Src_6475 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm12_Src_64), _mm256_castsi128_si256(xmm13_Src_75), 32);
  // current block for search loaded into ymm10 and ymm11

  for (int i = 0; i < 9; i++)
  {
    ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
    ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
    ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
    ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
    // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

    // process sad[-4,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -4,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 1 of 8 in hi and low 128bits

    // shift is possibly faster at IceLake and newer
    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[-3,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -3,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 2 of 8 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[-2,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -2,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 3 of 8 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[-1,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -1,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 4 of 8 in hi and low 128bits

    // process 4 partial sads in 2 partial sads
    ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
    ymm7_minsad8 = _mm256_blend_epi16(ymm7_minsad8, ymm6_part_sads, 15);

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[0,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 0,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 5 of 8 in hi and low 128bits

    // shift is possibly faster at IceLake and newer
    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[1,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 1,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 6 of 8 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[2,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 2,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 7 of 8 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[3,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 3,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 8 of 8 in hi and low 128bits

    ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
    ymm7_minsad8 = _mm256_slli_si256(ymm7_minsad8, 8);
    ymm7_minsad8 = _mm256_blend_epi16(ymm7_minsad8, ymm6_part_sads, 15);

    ymm7_minsad8 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm7_minsad8, 1)), ymm7_minsad8);   // minsad8 ready

    // sad 3,i-4 ready in low of mm4
    _mm_store_si128(&Arr128iSADs[i], _mm256_castsi256_si128(ymm7_minsad8));
  }

  unsigned short minsad = 65535;
  int idx_min_sad = 0;
  for (int y = 0; y < 9; y++)
  {
    unsigned int uiSADRes = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm_load_si128(&Arr128iSADs[y])));
    if ((unsigned short)uiSADRes < minsad)
    {
      minsad = (unsigned short)uiSADRes;
      idx_min_sad = 7 - (uiSADRes >> 16) + y * 8;
    }
  }

  //  x_minsad = (idx_min_sad % 8) - 4; - just comment where from x,y minsad come from
  //  y_minsad = (idx_min_sad / 8) - 4;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + (idx_min_sad % 8) - 4; // 8 and 9 need check
  workarea.bestMV.y = mvy + (idx_min_sad / 8) - 4;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();
}

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp3_avx2(WorkingArea& workarea, int mvx, int mvy) // debugged 28.10.21 (?)
{
  // debug check !
    // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 3, mvy - 3))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 3, mvy + 3))
  {
    return;
  }
  /*
  if (!workarea.IsVectorOK(mvx - 4, mvy + 3))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 3, mvy - 4))
  {
    return;
  }
  */
  alignas(16) __m128i Arr128iSADs[7];
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 3, mvy - 3); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // 2x12bytes store, require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_tmp, ymm5_tmp;

  __m256i ymm10_Src_2031, ymm11_Src_6475;
  __m128i xmm10_Src_20, xmm11_Src_31, xmm12_Src_64, xmm13_Src_75;

  __m256i ymm6_part_sads = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()); // set to all 65535;

  __m256i ymm7_minsad7; // vectors of minsads for SSE4.1 _mm_minpos_epu16() minsad and pos search, need FF-set 

  ymm7_minsad7 = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()); // set to all 65535

  xmm10_Src_20 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 2)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  xmm11_Src_31 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 1)));
  ymm10_Src_2031 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm10_Src_20), _mm256_castsi128_si256(xmm11_Src_31), 32);

  xmm12_Src_64 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 6)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  xmm13_Src_75 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 5)));
  ymm11_Src_6475 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm12_Src_64), _mm256_castsi128_si256(xmm13_Src_75), 32);
  // current block for search loaded into ymm10 and ymm11

  for (int i = 0; i < 7; i++)
  {
    ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
    ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
    ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
    ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
    // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

    // process sad[-3,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -3,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 1 of 7 in hi and low 128bits

    // shift is possibly faster at IceLake and newer
    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[-2,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -2,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 2 of 7 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[-1,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad -1,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 3 of 7 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[0,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 0,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 4 of 7 in hi and low 128bits

    // process 4 partial sads in 2 partial sads
    ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
    ymm7_minsad7 = _mm256_blend_epi16(ymm7_minsad7, ymm6_part_sads, 15);

    ymm6_part_sads = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()); // set max

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[1,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 1,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 5 of 7 in hi and low 128bits

    // shift is possibly faster at IceLake and newer
    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[2,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 2,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 6 of 7 in hi and low 128bits

    ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
    ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
    ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
    ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

    // process sad[3,i-4]
    ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
    ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

    ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
    // sad 3,i-4 4 parts ready in low of mm4
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 7 of 8 in hi and low 128bits

    ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);

    __m256i ymm8_tmpFFFF = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()); // set to all 65535;
    ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
    ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm8_tmpFFFF, 17);

    ymm7_minsad7 = _mm256_slli_si256(ymm7_minsad7, 8);
    ymm7_minsad7 = _mm256_blend_epi16(ymm7_minsad7, ymm6_part_sads, 15);

    ymm7_minsad7 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm7_minsad7, 1)), ymm7_minsad7);   // minsad7 ready

    // sad 3,i-4 ready in low of mm4
    _mm_store_si128(&Arr128iSADs[i], _mm256_castsi256_si128(ymm7_minsad7));

    ymm7_minsad7 = _mm256_cmpeq_epi64(ymm7_minsad7, ymm7_minsad7); // set to all 65535

  }

  unsigned short minsad = 65535;
  int idx_min_sad = 0;
  for (int y = 0; y < 7; y++)
  {
    unsigned int uiSADRes = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm_load_si128(&Arr128iSADs[y])));
    if ((unsigned short)uiSADRes < minsad)
    {
      minsad = (unsigned short)uiSADRes;
      idx_min_sad = 7 - (uiSADRes >> 16) + y * 7;
    }
  }

  //  x_minsad = (idx_min_sad % 8) - 4; - just comment where from x,y minsad come from
  //  y_minsad = (idx_min_sad / 9) - 4;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + (idx_min_sad % 7) - 3;
  workarea.bestMV.y = mvy + (idx_min_sad / 7) - 3;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();
}


void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp2_mpsadbw_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 2, mvy - 2))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 2, mvy + 2))
  {
    return;
  }

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_Src_01, ymm5_Src_23, ymm6_Src_45, ymm7_Src_67; // require buf padding to allow 16bytes reads to xmm

  __m256i ymm10_sads_r0, ymm11_sads_r1, ymm12_sads_r2, ymm13_sads_r3, ymm14_sads_r4;

  __m256i ymm13_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  __m256i ymm_block_ress;

  // load src as low 8bytes to each 128bit lane of 256
  ymm4_Src_01 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 1)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  ymm5_Src_23 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 2)));
  ymm6_Src_45 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 5)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  ymm7_Src_67 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 6)));

  // 1st row
  // gathering should be faster on Skylake and newer ?
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 1), (__m128i*)(pucRef));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 3), (__m128i*)(pucRef + nRefPitch[0] * 2));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 5), (__m128i*)(pucRef + nRefPitch[0] * 4));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 7), (__m128i*)(pucRef + nRefPitch[0] * 6));

  Sads_block_8x8
    ymm10_sads_r0 = ymm_block_ress;

  // 2nd row
  Push_Ref_8x8_row(8)

  Sads_block_8x8
    ymm11_sads_r1 = ymm_block_ress;

  // 3rd row
  Push_Ref_8x8_row(9)

  Sads_block_8x8
    ymm12_sads_r2 = ymm_block_ress;

  // 4th row
  Push_Ref_8x8_row(10)

  Sads_block_8x8
    ymm13_sads_r3 = ymm_block_ress;

  // 5th row
  Push_Ref_8x8_row(11)

  Sads_block_8x8
    ymm14_sads_r4 = ymm_block_ress;

  // set high sads, leave only 4,3,2,1,0
  ymm10_sads_r0 = _mm256_blend_epi16(ymm10_sads_r0, ymm13_all_ones, 224);
  ymm11_sads_r1 = _mm256_blend_epi16(ymm11_sads_r1, ymm13_all_ones, 224);
  ymm12_sads_r2 = _mm256_blend_epi16(ymm12_sads_r2, ymm13_all_ones, 224);
  ymm13_sads_r3 = _mm256_blend_epi16(ymm13_sads_r3, ymm13_all_ones, 224);
  ymm14_sads_r4 = _mm256_blend_epi16(ymm14_sads_r4, ymm13_all_ones, 224);

  __m128i xmm_res_R0 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm10_sads_r0));
  __m128i xmm_res_R1 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm11_sads_r1));
  __m128i xmm_res_R2 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm12_sads_r2));
  __m128i xmm_res_R3 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm13_sads_r3));
  __m128i xmm_res_R4 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm14_sads_r4));

  __m128i xmm_res_R0_R4 = _mm256_castsi256_si128(ymm13_all_ones);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, xmm_res_R0, 1);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R1, 2), 2);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R2, 4), 4);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R3, 6), 8);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R4, 8), 16);

  unsigned int uiRes_R0_R4 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm_res_R0_R4));

  int dx_minsad, dy_minsad, minsad;

  minsad = (unsigned short)uiRes_R0_R4;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  int iRow_minsad = (uiRes_R0_R4 >> 16);

  switch (iRow_minsad)
  {
    case 0:
      dy_minsad = -2;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R0) >> 16) - 2;
      break;

    case 1:
      dy_minsad = -1;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R1) >> 16) - 2;
      break;

    case 2:
      dy_minsad = 0;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R2) >> 16) - 2;
      break;

    case 3:
      dy_minsad = 1;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R3) >> 16) - 2;
      break;

    case 4:
      dy_minsad = 2;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R4) >> 16) - 2;
      break;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}

void PlaneOfBlocks::ExhaustiveSearch16x16_uint8_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 2, mvy - 2))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 2, mvy + 2))
  {
    return;
  }

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref0, ymm1_Ref1, ymm2_Ref2, ymm3_Ref3, ymm4_Ref4, ymm5_Ref5, ymm6_Ref6, ymm7_Ref7;
  __m256i ymm8_Src0, ymm9_Src1, ymm10_Src2, ymm11_Src3, ymm12_Src4, ymm13_Src5, ymm14_Src6, ymm15_Src7;

  __m256i ymm_half16x16_sads;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2, ymm_sads_R3, ymm_sads_R4;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  int iSrcShift = 0;
  int iRefShift = 0;

  // 1st row
  Half_16x16_sads

  ymm_sads_R0 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 8;

  Half_16x16_sads

  ymm_sads_R0 = _mm256_adds_epu16(ymm_sads_R0, ymm_half16x16_sads);

  // 2nd row
  iSrcShift = 0;
  iRefShift = 1;

  Half_16x16_sads

  ymm_sads_R1 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 9;

  Half_16x16_sads

  ymm_sads_R1 = _mm256_adds_epu16(ymm_sads_R1, ymm_half16x16_sads);

  // 3rd row
  iSrcShift = 0;
  iRefShift = 2;

  Half_16x16_sads

  ymm_sads_R2 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 10;

  Half_16x16_sads

  ymm_sads_R2 = _mm256_adds_epu16(ymm_sads_R2, ymm_half16x16_sads);

  // 4th row
  iSrcShift = 0;
  iRefShift = 3;

  Half_16x16_sads

  ymm_sads_R3 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 11;

  Half_16x16_sads

  ymm_sads_R3 = _mm256_adds_epu16(ymm_sads_R3, ymm_half16x16_sads);

  // 5th row
  iSrcShift = 0;
  iRefShift = 4;

  Half_16x16_sads

  ymm_sads_R4 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 12;

  Half_16x16_sads

  ymm_sads_R4 = _mm256_adds_epu16(ymm_sads_R4, ymm_half16x16_sads);

  // set high sads, leave only 4,3,2,1,0
  ymm_sads_R0 = _mm256_blend_epi16(ymm_sads_R0, ymm_all_ones, 224);
  ymm_sads_R1 = _mm256_blend_epi16(ymm_sads_R1, ymm_all_ones, 224);
  ymm_sads_R2 = _mm256_blend_epi16(ymm_sads_R2, ymm_all_ones, 224);
  ymm_sads_R3 = _mm256_blend_epi16(ymm_sads_R3, ymm_all_ones, 224);
  ymm_sads_R4 = _mm256_blend_epi16(ymm_sads_R4, ymm_all_ones, 224);

  __m128i xmm_res_R0 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R0));
  __m128i xmm_res_R1 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R1));
  __m128i xmm_res_R2 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R2));
  __m128i xmm_res_R3 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R3));
  __m128i xmm_res_R4 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R4));

  __m128i xmm_res_R0_R4 = _mm256_castsi256_si128(ymm_all_ones);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, xmm_res_R0, 1);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R1, 2), 2);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R2, 4), 4);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R3, 6), 8);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R4, 8), 16);

  unsigned int uiRes_R0_R4 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm_res_R0_R4));

  int dx_minsad, dy_minsad, minsad;

  minsad = (unsigned short)uiRes_R0_R4;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  int iRow_minsad = (uiRes_R0_R4 >> 16);

  switch (iRow_minsad)
  {
    case 0:
      dy_minsad = -2;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R0) >> 16) - 2;
      break;

    case 1:
      dy_minsad = -1;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R1) >> 16) - 2;
      break;

    case 2:
      dy_minsad = 0;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R2) >> 16) - 2;
      break;

    case 3:
      dy_minsad = 1;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R3) >> 16) - 2;
      break;

    case 4:
      dy_minsad = 2;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R4) >> 16) - 2;
      break;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();
}


void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 2, mvy - 2))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 2, mvy + 2))
  {
    return;
  }

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // 2x12bytes store, require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_tmp, ymm5_tmp;

  __m256i ymm10_Src_2031, ymm11_Src_6475;
  __m128i xmm10_Src_20, xmm11_Src_31, xmm12_Src_64, xmm13_Src_75;

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm7_minsad8_1, ymm8_minsad8_2, ymm9_minsad8_3; // vectors of minsads for SSE4.1 _mm_minpos_epu16() minsad and pos search

#ifdef _DEBUG
  ymm7_minsad8_1 = _mm256_setzero_si256();
  ymm8_minsad8_2 = _mm256_setzero_si256();
  ymm9_minsad8_3 = _mm256_setzero_si256();
#endif

  __m256i ymm6_part_sads = _mm256_setzero_si256();

  xmm10_Src_20 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 2)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  xmm11_Src_31 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 1)));
  ymm10_Src_2031 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm10_Src_20), _mm256_castsi128_si256(xmm11_Src_31), 32);

  xmm12_Src_64 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 6)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  xmm13_Src_75 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 5)));
  ymm11_Src_6475 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm12_Src_64), _mm256_castsi128_si256(xmm13_Src_75), 32);
  // current block for search loaded into ymm10 and ymm11

  // 1st row 
  int i = 0;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 1 of 8 in hi and low 128bits

  // shift is possibly faster at IceLake and newer
  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 2 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  //
  //2 row 
  //
  i = 1;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 2 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 2 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 2 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_slli_si256(ymm7_minsad8_1, 8);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm7_minsad8_1 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm7_minsad8_1, 1)), ymm7_minsad8_1);   // minsad8_1 ready

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 1 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 2 of 8 in hi and low 128bits


  //
  //3 row 
  //
  i = 2;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm8_minsad8_2 = _mm256_blend_epi16(ymm8_minsad8_2, ymm6_part_sads, 15);

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  //
  //4 row 
  //
  i = 3;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm8_minsad8_2 = _mm256_slli_si256(ymm8_minsad8_2, 8);
  ymm8_minsad8_2 = _mm256_blend_epi16(ymm8_minsad8_2, ymm6_part_sads, 15);

  ymm8_minsad8_2 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm8_minsad8_2, 1)), ymm8_minsad8_2);   // minsad8_2 ready

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 1 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 2 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm9_minsad8_3 = _mm256_blend_epi16(ymm9_minsad8_3, ymm6_part_sads, 15);
  //
  //5 row 
  //
  i = 4;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);


  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);


  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm9_minsad8_3 = _mm256_slli_si256(ymm9_minsad8_3, 8);
  ymm9_minsad8_3 = _mm256_blend_epi16(ymm9_minsad8_3, ymm6_part_sads, 15);

  ymm9_minsad8_3 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm9_minsad8_3, 1)), ymm9_minsad8_3);   // minsad8_3 ready

  // last column in 5 row of scan is not processed because of 3x8=24 only values for minsad SIMD, may be proc and stored separately for compare
  // if needed, but r2 search with 'round' do not require corners in fast processing. here only 1 of 4 corners is not checked.

  unsigned short minsad = 65535;
  int idx_min_sad = 0;

  unsigned int uiSADRes1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm7_minsad8_1)));
  unsigned int uiSADRes2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm8_minsad8_2)));
  unsigned int uiSADRes3 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm9_minsad8_3)));

  if ((unsigned short)uiSADRes1 < minsad)
  {
    minsad = (unsigned short)uiSADRes1;
    idx_min_sad = 7 - (uiSADRes1 >> 16);
  }

  if ((unsigned short)uiSADRes2 < minsad)
  {
    minsad = (unsigned short)uiSADRes2;
    idx_min_sad = 7 - (uiSADRes2 >> 16) + 8;
  }

  if ((unsigned short)uiSADRes3 < minsad)
  {
    minsad = (unsigned short)uiSADRes3;
    idx_min_sad = 7 - (uiSADRes3 >> 16) + 16;
  }

  //  x_minsad = (idx_min_sad % 5) - 2; - just comment where from x,y minsad come from
  //  y_minsad = (idx_min_sad / 5) - 2;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + (idx_min_sad % 5) - 2;
  workarea.bestMV.y = mvy + (idx_min_sad / 5) - 2;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 1, mvy - 1))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 1, mvy + 1))
  {
    return;
  }
  /*
  if (!workarea.IsVectorOK(mvx - 1, mvy + 1))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 1, mvy - 1))
  {
    return;
  }
  */

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // 2x12bytes store, require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_tmp, ymm5_tmp;

  __m256i ymm10_Src_2031, ymm11_Src_6475;
  __m128i xmm10_Src_20, xmm11_Src_31, xmm12_Src_64, xmm13_Src_75;

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm7_minsad8_1; // vectors of minsads for SSE4.1 _mm_minpos_epu16() minsad and pos search

#ifdef _DEBUG
  ymm7_minsad8_1 = _mm256_setzero_si256(); // to prevent debug break on access non-init value, it will be replaced total at using
#endif

  __m256i ymm6_part_sads = _mm256_setzero_si256(); // also last +1,+1 sad

  xmm10_Src_20 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 2)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  xmm11_Src_31 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 1)));
  ymm10_Src_2031 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm10_Src_20), _mm256_castsi128_si256(xmm11_Src_31), 32);

  xmm12_Src_64 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 6)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  xmm13_Src_75 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 5)));
  ymm11_Src_6475 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm12_Src_64), _mm256_castsi128_si256(xmm13_Src_75), 32);
  // current block for search loaded into ymm10 and ymm11

  // 1st row 
  int i = 0;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-1,-1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 1 of 8 in hi and low 128bits

  // shift is possibly faster at IceLake and newer
  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,-1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 2 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,-1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  //
  //2 row 
  //
  i = 1;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-1,0]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,0 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[0,0]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,0 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,0]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,0 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  //
  //3 row 
  //
  i = 2;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-1,1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,1 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[0,1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,1 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_slli_si256(ymm7_minsad8_1, 8);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm7_minsad8_1 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm7_minsad8_1, 1)), ymm7_minsad8_1);   // minsad8_1 ready

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm6_part_sads = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm6_part_sads, 1)), ymm6_part_sads);


  unsigned short minsad = 65535;
  int idx_min_sad = 0;

  unsigned int uiSADRes1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm7_minsad8_1)));

  if ((unsigned short)uiSADRes1 < minsad)
  {
    minsad = (unsigned short)uiSADRes1;
    idx_min_sad = 7 - (uiSADRes1 >> 16);
  }

  if ((unsigned short)_mm_cvtsi128_si32(_mm256_castsi256_si128(ymm6_part_sads)) < minsad)
  {
    minsad = (unsigned short)_mm_cvtsi128_si32(_mm256_castsi256_si128(ymm6_part_sads));
    idx_min_sad = 8;
  }

  //  x_minsad = (idx_min_sad % 3) - 1; - just comment where from x,y minsad come from
  //  y_minsad = (idx_min_sad / 3) - 1;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + (idx_min_sad % 3) - 1;
  workarea.bestMV.y = mvy + (idx_min_sad / 3) - 1;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}

void PlaneOfBlocks::ExhaustiveSearch16x16_uint8_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 1, mvy - 1))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 1, mvy + 1))
  {
    return;
  }

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref0, ymm1_Ref1, ymm2_Ref2, ymm3_Ref3, ymm4_Ref4, ymm5_Ref5, ymm6_Ref6, ymm7_Ref7;
  __m256i ymm8_Src0, ymm9_Src1, ymm10_Src2, ymm11_Src3, ymm12_Src4, ymm13_Src5, ymm14_Src6, ymm15_Src7;

  __m256i ymm_half16x16_sads;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  int iSrcShift = 0;
  int iRefShift = 0;

  // 1st row
  Half_16x16_sads

  ymm_sads_R0 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 8;

  Half_16x16_sads

  ymm_sads_R0 = _mm256_adds_epu16(ymm_sads_R0, ymm_half16x16_sads);

  // 2nd row
  iSrcShift = 0;
  iRefShift = 1;

  Half_16x16_sads

  ymm_sads_R1 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 9;

  Half_16x16_sads

  ymm_sads_R1 = _mm256_adds_epu16(ymm_sads_R1, ymm_half16x16_sads);

  // 3rd row
  iSrcShift = 0;
  iRefShift = 2;

  Half_16x16_sads

  ymm_sads_R2 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 10;

  Half_16x16_sads

  ymm_sads_R2 = _mm256_adds_epu16(ymm_sads_R2, ymm_half16x16_sads);

  // set high sads, leave only 2,1,0
  ymm_sads_R0 = _mm256_blend_epi16(ymm_sads_R0, ymm_all_ones, 248);
  ymm_sads_R1 = _mm256_blend_epi16(ymm_sads_R1, ymm_all_ones, 248);
  ymm_sads_R2 = _mm256_blend_epi16(ymm_sads_R2, ymm_all_ones, 248);

  unsigned int uiRes_R0 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R0)));
  unsigned int uiRes_R1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R1)));
  unsigned int uiRes_R2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R2)));

  int dx_minsad, dy_minsad, minsad;

  if ((unsigned short)uiRes_R0 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R0;
    dy_minsad = -1;
    dx_minsad = (uiRes_R0 >> 16) - 1;
  }
  else // minsad r1 >= minsad r0
  {
    minsad = (unsigned short)uiRes_R1;
    dy_minsad = 0;
    dx_minsad = (uiRes_R1 >> 16) - 1;
  }

  if ((unsigned short)uiRes_R2 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R2;
    dy_minsad = 1;
    dx_minsad = (uiRes_R2 >> 16) - 1;
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}



void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp1_mpsadbw_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 1, mvy - 1))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 1, mvy + 1))
  {
    return;
  }

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_Src_01, ymm5_Src_23, ymm6_Src_45, ymm7_Src_67; // require buf padding to allow 16bytes reads to xmm

  __m256i ymm10_sads_r0, ymm11_sads_r1, ymm12_sads_r2;

  __m256i ymm13_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  __m256i ymm_block_ress;

  // load src as low 8bytes to each 128bit lane of 256
  ymm4_Src_01 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 1)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  ymm5_Src_23 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 2)));
  ymm6_Src_45 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 5)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  ymm7_Src_67 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 6)));

  // 1st row
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 1), (__m128i*)(pucRef));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 3), (__m128i*)(pucRef + nRefPitch[0] * 2));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 5), (__m128i*)(pucRef + nRefPitch[0] * 4));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 7), (__m128i*)(pucRef + nRefPitch[0] * 6));

  Sads_block_8x8
  ymm10_sads_r0 = ymm_block_ress;

  // 2nd row
  Push_Ref_8x8_row(8)

  Sads_block_8x8
  ymm11_sads_r1 = ymm_block_ress;

  // 3rd row
  Push_Ref_8x8_row(9)

  Sads_block_8x8
  ymm12_sads_r2 = ymm_block_ress;

  // set high sads, leave only 2,1,0
  ymm10_sads_r0 = _mm256_blend_epi16(ymm10_sads_r0, ymm13_all_ones, 248);
  ymm11_sads_r1 = _mm256_blend_epi16(ymm11_sads_r1, ymm13_all_ones, 248);
  ymm12_sads_r2 = _mm256_blend_epi16(ymm12_sads_r2, ymm13_all_ones, 248);

  unsigned int uiRes_R0 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm10_sads_r0)));
  unsigned int uiRes_R1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm11_sads_r1)));
  unsigned int uiRes_R2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm12_sads_r2)));

  int dx_minsad, dy_minsad, minsad;

  if ((unsigned short)uiRes_R0 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R0;
    dy_minsad = -1;
    dx_minsad = (uiRes_R0 >> 16) - 1;
  }
  else // minsad r1 >= minsad r0
  {
    minsad = (unsigned short)uiRes_R1;
    dy_minsad = 0;
    dx_minsad = (uiRes_R1 >> 16) - 1;
  }

  if ((unsigned short)uiRes_R2 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R2;
    dy_minsad = 1;
    dx_minsad = (uiRes_R2 >> 16) - 1;
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}


// SO2 versions without IsVectorOK check
void PlaneOfBlocks::ExhaustiveSearch16x16_uint8_SO2_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref0, ymm1_Ref1, ymm2_Ref2, ymm3_Ref3, ymm4_Ref4, ymm5_Ref5, ymm6_Ref6, ymm7_Ref7;
  __m256i ymm8_Src0, ymm9_Src1, ymm10_Src2, ymm11_Src3, ymm12_Src4, ymm13_Src5, ymm14_Src6, ymm15_Src7;

  __m256i ymm_half16x16_sads;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2, ymm_sads_R3, ymm_sads_R4;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  int iSrcShift = 0;
  int iRefShift = 0;

  // 1st row
  Half_16x16_sads

  ymm_sads_R0 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 8;

  Half_16x16_sads

  ymm_sads_R0 = _mm256_adds_epu16(ymm_sads_R0, ymm_half16x16_sads);

  // 2nd row
  iSrcShift = 0;
  iRefShift = 1;

  Half_16x16_sads

  ymm_sads_R1 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 9;

  Half_16x16_sads

  ymm_sads_R1 = _mm256_adds_epu16(ymm_sads_R1, ymm_half16x16_sads);

  // 3rd row
  iSrcShift = 0;
  iRefShift = 2;

  Half_16x16_sads

  ymm_sads_R2 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 10;

  Half_16x16_sads

  ymm_sads_R2 = _mm256_adds_epu16(ymm_sads_R2, ymm_half16x16_sads);

  // 4th row
  iSrcShift = 0;
  iRefShift = 3;

  Half_16x16_sads

  ymm_sads_R3 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 11;

  Half_16x16_sads

  ymm_sads_R3 = _mm256_adds_epu16(ymm_sads_R3, ymm_half16x16_sads);

  // 5th row
  iSrcShift = 0;
  iRefShift = 4;

  Half_16x16_sads

  ymm_sads_R4 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 12;

  Half_16x16_sads

  ymm_sads_R4 = _mm256_adds_epu16(ymm_sads_R4, ymm_half16x16_sads);

  // set high sads, leave only 4,3,2,1,0
  ymm_sads_R0 = _mm256_blend_epi16(ymm_sads_R0, ymm_all_ones, 224);
  ymm_sads_R1 = _mm256_blend_epi16(ymm_sads_R1, ymm_all_ones, 224);
  ymm_sads_R2 = _mm256_blend_epi16(ymm_sads_R2, ymm_all_ones, 224);
  ymm_sads_R3 = _mm256_blend_epi16(ymm_sads_R3, ymm_all_ones, 224);
  ymm_sads_R4 = _mm256_blend_epi16(ymm_sads_R4, ymm_all_ones, 224);

  __m128i xmm_res_R0 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R0));
  __m128i xmm_res_R1 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R1));
  __m128i xmm_res_R2 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R2));
  __m128i xmm_res_R3 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R3));
  __m128i xmm_res_R4 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R4));

  __m128i xmm_res_R0_R4 = _mm256_castsi256_si128(ymm_all_ones);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, xmm_res_R0, 1);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R1, 2), 2);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R2, 4), 4);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R3, 6), 8);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R4, 8), 16);

  unsigned int uiRes_R0_R4 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm_res_R0_R4));

  int dx_minsad, dy_minsad, minsad;

  minsad = (unsigned short)uiRes_R0_R4;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  int iRow_minsad = (uiRes_R0_R4 >> 16);

  switch (iRow_minsad)
  {
  case 0:
    dy_minsad = -2;
    dx_minsad = (_mm_cvtsi128_si32(xmm_res_R0) >> 16) - 2;
    break;

  case 1:
    dy_minsad = -1;
    dx_minsad = (_mm_cvtsi128_si32(xmm_res_R1) >> 16) - 2;
    break;

  case 2:
    dy_minsad = 0;
    dx_minsad = (_mm_cvtsi128_si32(xmm_res_R2) >> 16) - 2;
    break;

  case 3:
    dy_minsad = 1;
    dx_minsad = (_mm_cvtsi128_si32(xmm_res_R3) >> 16) - 2;
    break;

  case 4:
    dy_minsad = 2;
    dx_minsad = (_mm_cvtsi128_si32(xmm_res_R4) >> 16) - 2;
    break;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();
}



void PlaneOfBlocks::ExhaustiveSearch16x16_uint8_SO2_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm0_Ref0, ymm1_Ref1, ymm2_Ref2, ymm3_Ref3, ymm4_Ref4, ymm5_Ref5, ymm6_Ref6, ymm7_Ref7;
  __m256i ymm8_Src0, ymm9_Src1, ymm10_Src2, ymm11_Src3, ymm12_Src4, ymm13_Src5, ymm14_Src6, ymm15_Src7;

  __m256i ymm_half16x16_sads;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  int iSrcShift = 0;
  int iRefShift = 0;

  // 1st row
  Half_16x16_sads

  ymm_sads_R0 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 8;

  Half_16x16_sads

  ymm_sads_R0 = _mm256_adds_epu16(ymm_sads_R0, ymm_half16x16_sads);

  // 2nd row
  iSrcShift = 0;
  iRefShift = 1;

  Half_16x16_sads

  ymm_sads_R1 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 9;

  Half_16x16_sads

  ymm_sads_R1 = _mm256_adds_epu16(ymm_sads_R1, ymm_half16x16_sads);

  // 3rd row
  iSrcShift = 0;
  iRefShift = 2;

  Half_16x16_sads

  ymm_sads_R2 = ymm_half16x16_sads;

  iSrcShift = 8;
  iRefShift = 10;

  Half_16x16_sads

  ymm_sads_R2 = _mm256_adds_epu16(ymm_sads_R2, ymm_half16x16_sads);

  // set high sads, leave only 2,1,0
  ymm_sads_R0 = _mm256_blend_epi16(ymm_sads_R0, ymm_all_ones, 248);
  ymm_sads_R1 = _mm256_blend_epi16(ymm_sads_R1, ymm_all_ones, 248);
  ymm_sads_R2 = _mm256_blend_epi16(ymm_sads_R2, ymm_all_ones, 248);

  unsigned int uiRes_R0 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R0)));
  unsigned int uiRes_R1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R1)));
  unsigned int uiRes_R2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R2)));

  int dx_minsad, dy_minsad, minsad;

  if ((unsigned short)uiRes_R0 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R0;
    dy_minsad = -1;
    dx_minsad = (uiRes_R0 >> 16) - 1;
  }
  else // minsad r1 >= minsad r0
  {
    minsad = (unsigned short)uiRes_R1;
    dy_minsad = 0;
    dx_minsad = (uiRes_R1 >> 16) - 1;
  }

  if ((unsigned short)uiRes_R2 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R2;
    dy_minsad = 1;
    dx_minsad = (uiRes_R2 >> 16) - 1;
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}



void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp1_mpsadbw_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  const long long* pucRef = (long long*)GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const long long* pucCurr = (long long*)workarea.pSrc[0];

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_Src_01, ymm5_Src_23, ymm6_Src_45, ymm7_Src_67; // require buf padding to allow 16bytes reads to xmm

  __m256i ymm10_sads_r0, ymm11_sads_r1, ymm12_sads_r2;

  __m256i ymm13_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  __m256i ymm_block_ress;

  // load src as low 8bytes to each 128bit lane of 256
/*  ymm4_Src_01 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 1)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  ymm5_Src_23 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 2)));
  ymm6_Src_45 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 5)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  ymm7_Src_67 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 6)));*/
  __m256i ymm_Src0123 = _mm256_i64gather_epi64(pucCurr, _mm256_set_epi64x(nSrcPitch[0] * 3, nSrcPitch[0] * 2, nSrcPitch[0] * 1, nSrcPitch[0] * 0), 1);
  __m256i ymm_Src4567 = _mm256_i64gather_epi64(pucCurr, _mm256_set_epi64x(nSrcPitch[0] * 7, nSrcPitch[0] * 6, nSrcPitch[0] * 5, nSrcPitch[0] * 4), 1);

  ymm4_Src_01 = _mm256_permute4x64_epi64(ymm_Src0123, 80);
  ymm5_Src_23 = _mm256_permute4x64_epi64(ymm_Src0123, 250);
  ymm6_Src_45 = _mm256_permute4x64_epi64(ymm_Src4567, 80);
  ymm7_Src_67 = _mm256_permute4x64_epi64(ymm_Src4567, 250);

  // 1st row
/*  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 1), (__m128i*)(pucRef));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 3), (__m128i*)(pucRef + nRefPitch[0] * 2));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 5), (__m128i*)(pucRef + nRefPitch[0] * 4));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 7), (__m128i*)(pucRef + nRefPitch[0] * 6));*/
  ymm0_Ref_01 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 1) + 8, (nRefPitch[0] * 1) + 0, 8, 0), 1);
  ymm1_Ref_23 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 3) + 8, (nRefPitch[0] * 3) + 0, (nRefPitch[0] * 2) + 8, (nRefPitch[0] * 2) + 0), 1);
  ymm2_Ref_45 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 5) + 8, (nRefPitch[0] * 5) + 0, (nRefPitch[0] * 4) + 8, (nRefPitch[0] * 4) + 0), 1);
  ymm3_Ref_67 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 7) + 8, (nRefPitch[0] * 7) + 0, (nRefPitch[0] * 6) + 8, (nRefPitch[0] * 6) + 0), 1);

  Sads_block_8x8
  ymm10_sads_r0 = ymm_block_ress;

  // 2nd row
  Push_Ref_8x8_row(8)
  Sads_block_8x8
  ymm11_sads_r1 = ymm_block_ress;

  // 3rd row
  Push_Ref_8x8_row(9)
  Sads_block_8x8
  ymm12_sads_r2 = ymm_block_ress;

  // set high sads, leave only 2,1,0
  ymm10_sads_r0 = _mm256_blend_epi16(ymm10_sads_r0, ymm13_all_ones, 248);
  ymm11_sads_r1 = _mm256_blend_epi16(ymm11_sads_r1, ymm13_all_ones, 248);
  ymm12_sads_r2 = _mm256_blend_epi16(ymm12_sads_r2, ymm13_all_ones, 248);

  unsigned int uiRes_R0 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm10_sads_r0)));
  unsigned int uiRes_R1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm11_sads_r1)));
  unsigned int uiRes_R2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm12_sads_r2)));

  int dx_minsad, dy_minsad, minsad;

  if ((unsigned short)uiRes_R0 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R0;
    dy_minsad = -1;
    dx_minsad = (uiRes_R0 >> 16) - 1;
  }
  else // minsad r1 >= minsad r0
  {
    minsad = (unsigned short)uiRes_R1;
    dy_minsad = 0;
    dx_minsad = (uiRes_R1 >> 16) - 1;
  }

  if ((unsigned short)uiRes_R2 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R2;
    dy_minsad = 1;
    dx_minsad = (uiRes_R2 >> 16) - 1;
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}


void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp2_mpsadbw_avx2(WorkingArea& workarea, int mvx, int mvy)
{

  const long long* pucRef = (long long*)GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const long long* pucCurr = (long long*)workarea.pSrc[0];

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_Src_01, ymm5_Src_23, ymm6_Src_45, ymm7_Src_67; // require buf padding to allow 16bytes reads to xmm

  __m256i ymm10_sads_r0, ymm11_sads_r1, ymm12_sads_r2, ymm13_sads_r3, ymm14_sads_r4;

  __m256i ymm13_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  __m256i ymm_block_ress;

  // load src as low 8bytes to each 128bit lane of 256
/*  ymm4_Src_01 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 1)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  ymm5_Src_23 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 2)));
  ymm6_Src_45 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 5)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  ymm7_Src_67 = _mm256_set_m128i(_mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 6)));*/
  __m256i ymm_Src0123 = _mm256_i64gather_epi64(pucCurr, _mm256_set_epi64x(nSrcPitch[0] * 3, nSrcPitch[0] * 2, nSrcPitch[0] * 1, nSrcPitch[0] * 0), 1);
  __m256i ymm_Src4567 = _mm256_i64gather_epi64(pucCurr, _mm256_set_epi64x(nSrcPitch[0] * 7, nSrcPitch[0] * 6, nSrcPitch[0] * 5, nSrcPitch[0] * 4), 1);

  ymm4_Src_01 = _mm256_permute4x64_epi64(ymm_Src0123, 80);
  ymm5_Src_23 = _mm256_permute4x64_epi64(ymm_Src0123, 250);
  ymm6_Src_45 = _mm256_permute4x64_epi64(ymm_Src4567, 80);
  ymm7_Src_67 = _mm256_permute4x64_epi64(ymm_Src4567, 250);

  // 1st row
  // gathering should be faster on Skylake and newer ?
/*  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 1), (__m128i*)(pucRef));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 3), (__m128i*)(pucRef + nRefPitch[0] * 2));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 5), (__m128i*)(pucRef + nRefPitch[0] * 4));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * 7), (__m128i*)(pucRef + nRefPitch[0] * 6));*/
  ymm0_Ref_01 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 1) + 8, (nRefPitch[0] * 1) + 0, 8, 0), 1);
  ymm1_Ref_23 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 3) + 8, (nRefPitch[0] * 3) + 0, (nRefPitch[0] * 2) + 8, (nRefPitch[0] * 2) + 0), 1);
  ymm2_Ref_45 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 5) + 8, (nRefPitch[0] * 5) + 0, (nRefPitch[0] * 4) + 8, (nRefPitch[0] * 4) + 0), 1);
  ymm3_Ref_67 = _mm256_i64gather_epi64(pucRef, _mm256_set_epi64x((nRefPitch[0] * 7) + 8, (nRefPitch[0] * 7) + 0, (nRefPitch[0] * 6) + 8, (nRefPitch[0] * 6) + 0), 1);

  Sads_block_8x8
  ymm10_sads_r0 = ymm_block_ress;

  // 2nd row
  Push_Ref_8x8_row(8)
  Sads_block_8x8
  ymm11_sads_r1 = ymm_block_ress;

  // 3rd row
  Push_Ref_8x8_row(9)
  Sads_block_8x8
  ymm12_sads_r2 = ymm_block_ress;

  // 4th row
  Push_Ref_8x8_row(10)
  Sads_block_8x8
  ymm13_sads_r3 = ymm_block_ress;

  // 5th row
  Push_Ref_8x8_row(11)
  Sads_block_8x8
  ymm14_sads_r4 = ymm_block_ress;

  // set high sads, leave only 4,3,2,1,0
  ymm10_sads_r0 = _mm256_blend_epi16(ymm10_sads_r0, ymm13_all_ones, 224);
  ymm11_sads_r1 = _mm256_blend_epi16(ymm11_sads_r1, ymm13_all_ones, 224);
  ymm12_sads_r2 = _mm256_blend_epi16(ymm12_sads_r2, ymm13_all_ones, 224);
  ymm13_sads_r3 = _mm256_blend_epi16(ymm13_sads_r3, ymm13_all_ones, 224);
  ymm14_sads_r4 = _mm256_blend_epi16(ymm14_sads_r4, ymm13_all_ones, 224);
  
  __m128i xmm_res_R0 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm10_sads_r0));
  __m128i xmm_res_R1 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm11_sads_r1));
  __m128i xmm_res_R2 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm12_sads_r2));
  __m128i xmm_res_R3 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm13_sads_r3));
  __m128i xmm_res_R4 = _mm_minpos_epu16(_mm256_castsi256_si128(ymm14_sads_r4));
  
  __m128i xmm_res_R0_R4 = _mm256_castsi256_si128(ymm13_all_ones);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, xmm_res_R0, 1);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R1, 2), 2);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R2, 4), 4);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R3, 6), 8);
  xmm_res_R0_R4 = _mm_blend_epi16(xmm_res_R0_R4, _mm_slli_si128(xmm_res_R4, 8), 16);

  unsigned int uiRes_R0_R4 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm_res_R0_R4));

  int dx_minsad, dy_minsad, minsad;

  minsad = (unsigned short)uiRes_R0_R4;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  int iRow_minsad = (uiRes_R0_R4 >> 16);

  switch (iRow_minsad)
  {
    case 0:
      dy_minsad = -2;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R0) >> 16) - 2;
      break;

    case 1:
      dy_minsad = -1;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R1) >> 16) - 2;
      break;

    case 2:
      dy_minsad = 0;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R2) >> 16) - 2;
      break;

    case 3:
      dy_minsad = 1;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R3) >> 16) - 2;
      break;

    case 4:
      dy_minsad = 2;
      dx_minsad = (_mm_cvtsi128_si32(xmm_res_R4) >> 16) - 2;
      break;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}


void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy)
{

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // 2x12bytes store, require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_tmp, ymm5_tmp;

  __m256i ymm10_Src_2031, ymm11_Src_6475;
  __m128i xmm10_Src_20, xmm11_Src_31, xmm12_Src_64, xmm13_Src_75;

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm7_minsad8_1, ymm8_minsad8_2, ymm9_minsad8_3; // vectors of minsads for SSE4.1 _mm_minpos_epu16() minsad and pos search

#ifdef _DEBUG
  ymm7_minsad8_1 = _mm256_setzero_si256();
  ymm8_minsad8_2 = _mm256_setzero_si256();
  ymm9_minsad8_3 = _mm256_setzero_si256();
#endif

  __m256i ymm6_part_sads = _mm256_setzero_si256();

  xmm10_Src_20 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 2)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  xmm11_Src_31 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 1)));
  ymm10_Src_2031 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm10_Src_20), _mm256_castsi128_si256(xmm11_Src_31), 32);

  xmm12_Src_64 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 6)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  xmm13_Src_75 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 5)));
  ymm11_Src_6475 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm12_Src_64), _mm256_castsi128_si256(xmm13_Src_75), 32);
  // current block for search loaded into ymm10 and ymm11

  // 1st row 
  int i = 0;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 1 of 8 in hi and low 128bits

  // shift is possibly faster at IceLake and newer
  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 2 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  //
  //2 row 
  //
  i = 1;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 2 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 2 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 2 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_slli_si256(ymm7_minsad8_1, 8);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm7_minsad8_1 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm7_minsad8_1, 1)), ymm7_minsad8_1);   // minsad8_1 ready

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 1 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 2 of 8 in hi and low 128bits


  //
  //3 row 
  //
  i = 2;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm8_minsad8_2 = _mm256_blend_epi16(ymm8_minsad8_2, ymm6_part_sads, 15);

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  //
  //4 row 
  //
  i = 3;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm8_minsad8_2 = _mm256_slli_si256(ymm8_minsad8_2, 8);
  ymm8_minsad8_2 = _mm256_blend_epi16(ymm8_minsad8_2, ymm6_part_sads, 15);

  ymm8_minsad8_2 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm8_minsad8_2, 1)), ymm8_minsad8_2);   // minsad8_2 ready

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 1 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 2 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm9_minsad8_3 = _mm256_blend_epi16(ymm9_minsad8_3, ymm6_part_sads, 15);
  //
  //5 row 
  //
  i = 4;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-2,i-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -2,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);


  // process sad[-0,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);


  // process sad[1,-2]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,i-2 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm9_minsad8_3 = _mm256_slli_si256(ymm9_minsad8_3, 8);
  ymm9_minsad8_3 = _mm256_blend_epi16(ymm9_minsad8_3, ymm6_part_sads, 15);

  ymm9_minsad8_3 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm9_minsad8_3, 1)), ymm9_minsad8_3);   // minsad8_3 ready

  // last column in 5 row of scan is not processed because of 3x8=24 only values for minsad SIMD, may be proc and stored separately for compare
  // if needed, but r2 search with 'round' do not require corners in fast processing. here only 1 of 4 corners is not checked.

  unsigned short minsad = 65535;
  int idx_min_sad = 0;

  unsigned int uiSADRes1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm7_minsad8_1)));
  unsigned int uiSADRes2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm8_minsad8_2)));
  unsigned int uiSADRes3 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm9_minsad8_3)));

  if ((unsigned short)uiSADRes1 < minsad)
  {
    minsad = (unsigned short)uiSADRes1;
    idx_min_sad = 7 - (uiSADRes1 >> 16);
  }

  if ((unsigned short)uiSADRes2 < minsad)
  {
    minsad = (unsigned short)uiSADRes2;
    idx_min_sad = 7 - (uiSADRes2 >> 16) + 8;
  }

  if ((unsigned short)uiSADRes3 < minsad)
  {
    minsad = (unsigned short)uiSADRes3;
    idx_min_sad = 7 - (uiSADRes3 >> 16) + 16;
  }

  //  x_minsad = (idx_min_sad % 5) - 2; - just comment where from x,y minsad come from
  //  y_minsad = (idx_min_sad / 5) - 2;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + (idx_min_sad % 5) - 2;
  workarea.bestMV.y = mvy + (idx_min_sad / 5) - 2;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy)
{

  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // 2x12bytes store, require buf padding to allow 16bytes reads to xmm
  __m256i ymm4_tmp, ymm5_tmp;

  __m256i ymm10_Src_2031, ymm11_Src_6475;
  __m128i xmm10_Src_20, xmm11_Src_31, xmm12_Src_64, xmm13_Src_75;

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m256i ymm7_minsad8_1; // vectors of minsads for SSE4.1 _mm_minpos_epu16() minsad and pos search

#ifdef _DEBUG
  ymm7_minsad8_1 = _mm256_setzero_si256(); // to prevent debug break on access non-init value, it will be replaced total at using
#endif

  __m256i ymm6_part_sads = _mm256_setzero_si256(); // also last +1,+1 sad

  xmm10_Src_20 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 2)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 0)));
  xmm11_Src_31 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 3)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 1)));
  ymm10_Src_2031 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm10_Src_20), _mm256_castsi128_si256(xmm11_Src_31), 32);

  xmm12_Src_64 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 6)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 4)));
  xmm13_Src_75 = _mm_unpacklo_epi64(_mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 7)), _mm_loadu_si64((__m128i*)(pucCurr + nSrcPitch[0] * 5)));
  ymm11_Src_6475 = _mm256_permute2x128_si256(_mm256_castsi128_si256(xmm12_Src_64), _mm256_castsi128_si256(xmm13_Src_75), 32);
  // current block for search loaded into ymm10 and ymm11

  // 1st row 
  int i = 0;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-1,-1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,i-1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 1 of 8 in hi and low 128bits

  // shift is possibly faster at IceLake and newer
  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[-0,-1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,-1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); //  partial sums 2 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,-1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,-1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 3 of 8 in hi and low 128bits

  //
  //2 row 
  //
  i = 1;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-1,0]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,0 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 4 of 8 in hi and low 128bits

  // process 4 partial sads in 2 partial sads
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[0,0]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,0 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 5 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,0]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,0 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 6 of 8 in hi and low 128bits

  //
  //3 row 
  //
  i = 2;
  ymm0_Ref_01 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 1)), (__m128i*)(pucRef + nRefPitch[0] * (i + 0)));
  ymm1_Ref_23 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 3)), (__m128i*)(pucRef + nRefPitch[0] * (i + 2)));
  ymm2_Ref_45 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 5)), (__m128i*)(pucRef + nRefPitch[0] * (i + 4)));
  ymm3_Ref_67 = _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * (i + 7)), (__m128i*)(pucRef + nRefPitch[0] * (i + 6)));
  // loaded 8 rows of Ref plane 16samples wide into ymm0..ymm3

  // process sad[-1,1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad -1,1 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 7 of 8 in hi and low 128bits

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[0,1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 0,1 ready in low of mm4
  ymm6_part_sads = _mm256_slli_si256(ymm6_part_sads, 2);
  ymm6_part_sads = _mm256_blend_epi16(ymm6_part_sads, ymm4_tmp, 17); // partial sums 8 of 8 in hi and low 128bits

  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm7_minsad8_1 = _mm256_slli_si256(ymm7_minsad8_1, 8);
  ymm7_minsad8_1 = _mm256_blend_epi16(ymm7_minsad8_1, ymm6_part_sads, 15);

  ymm7_minsad8_1 = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm7_minsad8_1, 1)), ymm7_minsad8_1);   // minsad8_1 ready

  ymm0_Ref_01 = _mm256_srli_si256(ymm0_Ref_01, 1);
  ymm1_Ref_23 = _mm256_srli_si256(ymm1_Ref_23, 1);
  ymm2_Ref_45 = _mm256_srli_si256(ymm2_Ref_45, 1);
  ymm3_Ref_67 = _mm256_srli_si256(ymm3_Ref_67, 1);

  // process sad[1,1]
  ymm4_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm0_Ref_01, 8), ymm1_Ref_23, 51), ymm10_Src_2031);
  ymm5_tmp = _mm256_sad_epu8(_mm256_blend_epi32(_mm256_slli_si256(ymm2_Ref_45, 8), ymm3_Ref_67, 51), ymm11_Src_6475);

  ymm4_tmp = _mm256_adds_epu16(ymm4_tmp, ymm5_tmp);
  // sad 1,1 4 parts ready in low of mm4
  ymm6_part_sads = _mm256_adds_epu16(_mm256_srli_si256(ymm6_part_sads, 8), ymm6_part_sads);
  ymm6_part_sads = _mm256_adds_epu16(_mm256_castsi128_si256(_mm256_extracti128_si256(ymm6_part_sads, 1)), ymm6_part_sads);


  unsigned short minsad = 65535;
  int idx_min_sad = 0;

  unsigned int uiSADRes1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm7_minsad8_1)));

  if ((unsigned short)uiSADRes1 < minsad)
  {
    minsad = (unsigned short)uiSADRes1;
    idx_min_sad = 7 - (uiSADRes1 >> 16);
  }

  if ((unsigned short)_mm_cvtsi128_si32(_mm256_castsi256_si128(ymm6_part_sads)) < minsad)
  {
    minsad = (unsigned short)_mm_cvtsi128_si32(_mm256_castsi256_si128(ymm6_part_sads));
    idx_min_sad = 8;
  }

  //  x_minsad = (idx_min_sad % 3) - 1; - just comment where from x,y minsad come from
  //  y_minsad = (idx_min_sad / 3) - 1;

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + (idx_min_sad % 3) - 1;
  workarea.bestMV.y = mvy + (idx_min_sad / 3) - 1;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}



void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy, int* pBlkData)
{

#define SAD_4blocks \
  /*load ref*/ \
  ymm8_r1 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 0) + x)); \
  ymm9_r2 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 1) + x)); \
  ymm10_r3 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 2) + x)); \
  ymm11_r4 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 3) + x)); \
  ymm12_r5 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 4) + x)); \
  ymm13_r6 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 5) + x)); \
  ymm14_r7 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 6) + x)); \
  ymm15_r8 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 7) + x)); \
  /* calc sads */ \
  ymm8_r1 = _mm256_sad_epu8(ymm8_r1, ymm0_src_r1); \
  ymm9_r2 = _mm256_sad_epu8(ymm9_r2, ymm1_src_r2); \
  ymm10_r3 = _mm256_sad_epu8(ymm10_r3, ymm2_src_r3); \
  ymm11_r4 = _mm256_sad_epu8(ymm11_r4, ymm3_src_r4); \
  ymm12_r5 = _mm256_sad_epu8(ymm12_r5, ymm4_src_r5); \
  ymm13_r6 = _mm256_sad_epu8(ymm13_r6, ymm5_src_r6); \
  ymm14_r7 = _mm256_sad_epu8(ymm14_r7, ymm6_src_r7); \
  ymm15_r8 = _mm256_sad_epu8(ymm15_r8, ymm7_src_r8); \
  \
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm9_r2); \
  ymm10_r3 = _mm256_adds_epu16(ymm10_r3, ymm11_r4); \
  ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm13_r6); \
  ymm14_r7 = _mm256_adds_epu16(ymm14_r7, ymm15_r8); \
  \
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm10_r3); \
  ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm14_r7); \
  \
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm12_r5);

#define _mm_cmpge_epu16(a, b) \
        _mm_cmpeq_epi16(_mm_max_epu16(a, b), a)

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  // 4 blocks at once proc, pel=1
  __m256i ymm0_src_r1 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 0));
  __m256i	ymm1_src_r2 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 1));
  __m256i	ymm2_src_r3 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 2));
  __m256i	ymm3_src_r4 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 3));
  __m256i	ymm4_src_r5 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 4));
  __m256i	ymm5_src_r6 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 5));
  __m256i	ymm6_src_r7 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 6));
  __m256i	ymm7_src_r8 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 7));

  __m256i ymm8_r1, ymm9_r2, ymm10_r3, ymm11_r4, ymm12_r5, ymm13_r6, ymm14_r7, ymm15_r8;

  int x, y;
  __m256i part_sads1, part_sads2;
#ifdef _DEBUG
  part_sads1 = _mm256_setzero_si256();
  part_sads2 = _mm256_setzero_si256();
#endif

  // 1st 4sads
  y = 0; x = 0;
  SAD_4blocks
    /*			// load ref
      ymm8_r1 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 0) + x));
      ymm9_r2 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 1) + x));
      ymm10_r3 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 2) + x));
      ymm11_r4 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 3) + x));
      ymm12_r5 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 4) + x));
      ymm13_r6 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 5) + x));
      ymm14_r7 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 6) + x));
      ymm15_r8 = _mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * (y + 7) + x));
      // calc sads
      ymm8_r1 = _mm256_sad_epu8(ymm8_r1, ymm0_src_r1);
      ymm9_r2 = _mm256_sad_epu8(ymm9_r2, ymm1_src_r2);
      ymm10_r3 = _mm256_sad_epu8(ymm10_r3, ymm2_src_r3);
      ymm11_r4 = _mm256_sad_epu8(ymm11_r4, ymm3_src_r4);
      ymm12_r5 = _mm256_sad_epu8(ymm12_r5, ymm4_src_r5);
      ymm13_r6 = _mm256_sad_epu8(ymm13_r6, ymm5_src_r6);
      ymm14_r7 = _mm256_sad_epu8(ymm14_r7, ymm6_src_r7);
      ymm15_r8 = _mm256_sad_epu8(ymm15_r8, ymm7_src_r8);
      ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm9_r2);
      ymm10_r3 = _mm256_adds_epu16(ymm10_r3, ymm11_r4);
      ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm13_r6);
      ymm14_r7 = _mm256_adds_epu16(ymm14_r7, ymm15_r8);
      ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm10_r3);
      ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm14_r7);
      ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm12_r5);
      */
  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 17);
  part_sads1 = _mm256_slli_si256(part_sads1, 2);

  // 2nd 4sads
  x = 1; y = 0;
  SAD_4blocks

  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 17);
  part_sads1 = _mm256_slli_si256(part_sads1, 2);

  // 3rd 4sads
  x = 2; y = 0;
  SAD_4blocks

  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 17);
  part_sads1 = _mm256_slli_si256(part_sads1, 2);

  // 4th 4sads
  x = 0; y = 1;
  SAD_4blocks

  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 17); // part_sads1 ready 4x4

  // 5th 4sads
  x = 1; y = 1;
  SAD_4blocks

  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 17);
  part_sads2 = _mm256_slli_si256(part_sads2, 2);

  // 6th 4sads
  x = 2; y = 1;
  SAD_4blocks

  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 17);
  part_sads2 = _mm256_slli_si256(part_sads2, 2);

  // 7th 4sads
  x = 0; y = 2;
  SAD_4blocks

  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 17);
  part_sads2 = _mm256_slli_si256(part_sads2, 2);

  // 8th 4sads
  x = 1; y = 2;
  SAD_4blocks

  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 17); // part_sads2 ready 4x4

  // 9th 4sads
  x = 2; y = 2;
  SAD_4blocks

    // 9th 4 sads in ymm8_r1
  /*	unsigned int ui9SAD1 = _mm256_extract_epi16(ymm8_r1, 0);
    unsigned int ui9SAD2 = _mm256_extract_epi16(ymm8_r1, 3);
    unsigned int ui9SAD3 = _mm256_extract_epi16(ymm8_r1, 7);
    unsigned int ui9SAD4 = _mm256_extract_epi16(ymm8_r1, 15);
    */
    __m256i ymm_tmp1, ymm_tmp2;
  // 8 SADs of 1 block
  ymm_tmp1 = _mm256_slli_si256(part_sads1, 8);
  ymm_tmp2 = _mm256_blend_epi16(part_sads2, ymm_tmp1, 240);

  __m128i xmm0_sad1 = _mm256_castsi256_si128(ymm_tmp2);

  // 8 SADS of 2 block
  ymm_tmp1 = _mm256_srli_si256(part_sads2, 8);
  ymm_tmp2 = _mm256_blend_epi16(part_sads1, ymm_tmp1, 15);

  __m128i xmm1_sad2 = _mm256_castsi256_si128(ymm_tmp2);

  part_sads1 = _mm256_permute4x64_epi64(part_sads1, 14); // move high 128bits to low 128 bits
  part_sads2 = _mm256_permute4x64_epi64(part_sads2, 14); // move high 128bits to low 128 bits

  // 8 SADs of 3 block
  ymm_tmp1 = _mm256_slli_si256(part_sads1, 8);
  ymm_tmp2 = _mm256_blend_epi16(part_sads2, ymm_tmp1, 240);

  __m128i xmm2_sad3 = _mm256_castsi256_si128(ymm_tmp2);

  // 8 SADs of 4 block
  ymm_tmp1 = _mm256_srli_si256(part_sads2, 8);
  ymm_tmp2 = _mm256_blend_epi16(part_sads1, ymm_tmp1, 15);

  __m128i xmm3_sad4 = _mm256_castsi256_si128(ymm_tmp2);

  unsigned int uiSADRes1 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm0_sad1));
  unsigned int uiSADRes2 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm1_sad2));
  unsigned int uiSADRes3 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm2_sad3));
  unsigned int uiSADRes4 = _mm_cvtsi128_si32(_mm_minpos_epu16(xmm3_sad4));


  unsigned short minsad = 65535;
  int idx_min_sad = 0;
  /* SIMD it too
    if ((unsigned short)uiSADRes1 < minsad)
    {
      minsad = (unsigned short)uiSADRes1;
      idx_min_sad = 7 - (uiSADRes1 >> 16);
    }
    */
  __m128i xmm_sad_res1234 = _mm_cvtsi32_si128(uiSADRes1); // replace with shift + blend
  xmm_sad_res1234 = _mm_insert_epi32(xmm_sad_res1234, uiSADRes2, 1);
  xmm_sad_res1234 = _mm_insert_epi32(xmm_sad_res1234, uiSADRes3, 2);
  xmm_sad_res1234 = _mm_insert_epi32(xmm_sad_res1234, uiSADRes4, 3);

  __m128i xmm_minsad1234 = xmm_sad_res1234;//_mm_set1_epi16(-1);
  __m128i idx_minsad1234 = _mm_set1_epi16(7);

  //	xmm_minsad = _mm_min_epu16(xmm_minsad, xmm_sad_res1234);
  idx_minsad1234 = _mm_subs_epu16(idx_minsad1234, xmm_sad_res1234);
  /*
    if ((unsigned short)ui9SAD1 < minsad)
    {
      minsad = (unsigned short)ui9SAD1 < minsad;
      idx_min_sad = 8;
    }
    */
    /*	__m128i xmm_sad_res1234_9 = _mm_cvtsi32_si128(ui9SAD1); // repack ymm8_r1 !!!
      xmm_sad_res1234_9 = _mm_insert_epi32(xmm_sad_res1234, ui9SAD2, 1);
      xmm_sad_res1234_9 = _mm_insert_epi32(xmm_sad_res1234, ui9SAD3, 2);
      xmm_sad_res1234_9 = _mm_insert_epi32(xmm_sad_res1234, ui9SAD4, 3);
      */
  __m128i xmm_sad_res1234_9 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(ymm8_r1, _mm256_setr_epi32(7, 7, 7, 7, 6, 4, 2, 0))); // check

  __m128i idx_minsad8 = _mm_set1_epi16(8);
  __m128i mask_idx8 = _mm_cmpge_epu16(xmm_sad_res1234, xmm_sad_res1234_9);

  idx_minsad1234 = _mm_blendv_epi8(idx_minsad1234, idx_minsad8, mask_idx8);
  xmm_minsad1234 = _mm_min_epu16(xmm_minsad1234, xmm_sad_res1234_9);

  int idx_min_sad1 = _mm_extract_epi16(idx_minsad1234, 0);
  int idx_min_sad2 = _mm_extract_epi16(idx_minsad1234, 2);
  int idx_min_sad3 = _mm_extract_epi16(idx_minsad1234, 4);
  int idx_min_sad4 = _mm_extract_epi16(idx_minsad1234, 6);

  int x_minsad1 = (idx_min_sad1 % 3) - 1; //- just comment where from x,y minsad come from
  int y_minsad1 = (idx_min_sad1 / 3) - 1;

  int x_minsad2 = (idx_min_sad2 % 3) - 1; //- just comment where from x,y minsad come from
  int y_minsad2 = (idx_min_sad2 / 3) - 1;

  int x_minsad3 = (idx_min_sad3 % 3) - 1; //- just comment where from x,y minsad come from
  int y_minsad3 = (idx_min_sad3 / 3) - 1;

  int x_minsad4 = (idx_min_sad4 % 3) - 1; //- just comment where from x,y minsad come from
  int y_minsad4 = (idx_min_sad4 / 3) - 1;

  int minsad1 = _mm_extract_epi16(idx_minsad1234, 1);
  int minsad2 = _mm_extract_epi16(idx_minsad1234, 3);
  int minsad3 = _mm_extract_epi16(idx_minsad1234, 5);
  int minsad4 = _mm_extract_epi16(idx_minsad1234, 7);

 //  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);

  __m128i xmm_penaltyNew = _mm_set1_epi32(penaltyNew);
  __m128i xmm_minsad32 = _mm_srli_epi32(xmm_minsad1234, 16);
  xmm_minsad32 = _mm_mullo_epi32(xmm_minsad32, xmm_penaltyNew);
  xmm_minsad32 = _mm_srli_epi32(xmm_minsad32, 8);
  int cost1 = _mm_extract_epi16(idx_minsad1234, 1);
  int cost2 = _mm_extract_epi16(idx_minsad1234, 3);
  int cost3 = _mm_extract_epi16(idx_minsad1234, 5);
  int cost4 = _mm_extract_epi16(idx_minsad1234, 7);

  //  if (cost >= workarea.nMinCost)

  if (cost1 < workarea.nMinCost)
  {
    pBlkData[workarea.blkx * N_PER_BLOCK + 0] = pBlkData[workarea.blkx * N_PER_BLOCK + 0] + x_minsad1;
    pBlkData[workarea.blkx * N_PER_BLOCK + 1] = pBlkData[workarea.blkx * N_PER_BLOCK + 1] + y_minsad1;
    pBlkData[workarea.blkx * N_PER_BLOCK + 2] = (uint32_t)(minsad1);

    workarea.bestMV.x = mvx + x_minsad1;
    workarea.bestMV.y = mvy + y_minsad1;
    workarea.nMinCost = cost1;
    workarea.bestMV.sad = minsad1;
  }

  if (cost2 < workarea.nMinCost)
  {
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 0] = pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 0] + x_minsad2;
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 1] = pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 1] + y_minsad2;
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = (uint32_t)(minsad2);
  }

  if (cost3 < workarea.nMinCost)
  {
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 0] = pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 0] + x_minsad2;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 1] = pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 1] + y_minsad2;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = (uint32_t)(minsad3);
  }

  if (cost4 < workarea.nMinCost)
  {
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 0] = pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 0] + x_minsad2;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 1] = pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 1] + y_minsad2;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = (uint32_t)(minsad4);
  }
  
  _mm256_zeroupper();

}

////////////// To file PlaneofBlocks_avx512.cpp
#define _mm512_gather16x16_src(r2, r1) \
			_mm512_permutexvar_epi32(idx_set_src, _mm512_i64gather_epi64(_mm512_set_epi64( \
				nSrcPitch[0] * r2 + 8, nSrcPitch[0] * r2 + 8, nSrcPitch[0] * r2 + 0, nSrcPitch[0] * r2 + 0, \
				nSrcPitch[0] * r1 + 8, nSrcPitch[0] * r1 + 8, nSrcPitch[0] * r1 + 0, nSrcPitch[0] * r1 + 0), pucCurr, 1))

#define Sad_16x16_avx512 \
	zmm16_sad01 = _mm512_dbsad_epu8(zmm8_Src01, zmm0_Ref01, 148); \
	zmm17_sad23 = _mm512_dbsad_epu8(zmm9_Src23, zmm1_Ref23, 148); \
	zmm18_sad45 = _mm512_dbsad_epu8(zmm10_Src45, zmm2_Ref45, 148); \
	zmm19_sad67 = _mm512_dbsad_epu8(zmm11_Src67, zmm3_Ref67, 148); \
	zmm20_sad89 = _mm512_dbsad_epu8(zmm12_Src89, zmm4_Ref89, 148); \
	zmm21_sad1011 = _mm512_dbsad_epu8(zmm13_Src1011, zmm5_Ref1011, 148); \
	zmm22_sad1213 = _mm512_dbsad_epu8(zmm14_Src1213, zmm6_Ref1213, 148); \
	zmm23_sad1415 = _mm512_dbsad_epu8(zmm15_Src1415, zmm7_Ref1415, 148); \
	zmm16_sad01 = _mm512_adds_epu16(zmm16_sad01, zmm17_sad23); \
	zmm18_sad45 = _mm512_adds_epu16(zmm18_sad45, zmm19_sad67); \
	zmm20_sad89 = _mm512_adds_epu16(zmm20_sad89, zmm21_sad1011); \
	zmm22_sad1213 = _mm512_adds_epu16(zmm22_sad1213, zmm23_sad1415); \
	zmm16_sad01 = _mm512_adds_epu16(zmm16_sad01, zmm18_sad45); \
	zmm20_sad89 = _mm512_adds_epu16(zmm20_sad89, zmm22_sad1213); \
	zmm16_sad01 = _mm512_adds_epu16(zmm16_sad01, zmm20_sad89); \
	zmm16_sad01 = _mm512_adds_epu16(zmm16_sad01, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm16_sad01)); \
	zmm16_sad01 = _mm512_adds_epu16(zmm16_sad01, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm16_sad01)); \
	xmm_sad_ress = _mm512_castsi512_si128(zmm16_sad01); \
	xmm_sad_ress = _mm_adds_epi16(xmm_sad_ress, _mm_srli_si128(xmm_sad_ress, 8));



void PlaneOfBlocks::ExhaustiveSearch16x16_uint8_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy)
{
  // debug check !! need to fix caller to now allow illegal vectors 
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 1, mvy - 1))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 1, mvy + 1))
  {
    return;
  }

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m512i zmm0_Ref01, zmm1_Ref23, zmm2_Ref45, zmm3_Ref67, zmm4_Ref89, zmm5_Ref1011, zmm6_Ref1213, zmm7_Ref1415;
  __m512i zmm8_Src01, zmm9_Src23, zmm10_Src45, zmm11_Src67, zmm12_Src89, zmm13_Src1011, zmm14_Src1213, zmm15_Src1415;

  __m512i zmm16_sad01, zmm17_sad23, zmm18_sad45, zmm19_sad67, zmm20_sad89, zmm21_sad1011, zmm22_sad1213, zmm23_sad1415;

  __m128i xmm_sad_ress;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  const __m512i idx_set_src = _mm512_set_epi32(13, 13, 12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 1, 1, 0, 0);
  const __m512i imm_shift_insert_rows = _mm512_set_epi32(21, 20, 19, 18, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8);
  const __m512i imm_vshift_rows = _mm512_set_epi64(11, 10, 9, 8, 7, 6, 5, 4);

  // src
  zmm8_Src01 = _mm512_gather16x16_src(1, 0);
  zmm9_Src23 = _mm512_gather16x16_src(3, 2);
  zmm10_Src45 = _mm512_gather16x16_src(5, 4);
  zmm11_Src67 = _mm512_gather16x16_src(7, 6);
  zmm12_Src89 = _mm512_gather16x16_src(9, 8);
  zmm13_Src1011 = _mm512_gather16x16_src(11, 10);
  zmm14_Src1213 = _mm512_gather16x16_src(13, 12);
  zmm15_Src1415 = _mm512_gather16x16_src(15, 14);

  // ref 
  zmm0_Ref01 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 1 + 16, nRefPitch[0] * 1 + 8, nRefPitch[0] * 1 + 8, nRefPitch[0] * 1 + 0, nRefPitch[0] * 0 + 16, nRefPitch[0] * 0 + 8, nRefPitch[0] * 0 + 8, nRefPitch[0] * 0 + 0), pucRef, 1);
  zmm1_Ref23 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 3 + 16, nRefPitch[0] * 3 + 8, nRefPitch[0] * 3 + 8, nRefPitch[0] * 3 + 0, nRefPitch[0] * 2 + 16, nRefPitch[0] * 2 + 8, nRefPitch[0] * 2 + 8, nRefPitch[0] * 2 + 0), pucRef, 1);
  zmm2_Ref45 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 5 + 16, nRefPitch[0] * 5 + 8, nRefPitch[0] * 5 + 8, nRefPitch[0] * 5 + 0, nRefPitch[0] * 4 + 16, nRefPitch[0] * 4 + 8, nRefPitch[0] * 4 + 8, nRefPitch[0] * 4 + 0), pucRef, 1);
  zmm3_Ref67 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 7 + 16, nRefPitch[0] * 7 + 8, nRefPitch[0] * 7 + 8, nRefPitch[0] * 7 + 0, nRefPitch[0] * 6 + 16, nRefPitch[0] * 6 + 8, nRefPitch[0] * 6 + 8, nRefPitch[0] * 6 + 0), pucRef, 1);
  zmm4_Ref89 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 9 + 16, nRefPitch[0] * 9 + 8, nRefPitch[0] * 9 + 8, nRefPitch[0] * 9 + 0, nRefPitch[0] * 8 + 16, nRefPitch[0] * 8 + 8, nRefPitch[0] * 8 + 8, nRefPitch[0] * 8 + 0), pucRef, 1);
  zmm5_Ref1011 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 11 + 16, nRefPitch[0] * 11 + 8, nRefPitch[0] * 11 + 8, nRefPitch[0] * 11 + 0, nRefPitch[0] * 10 + 16, nRefPitch[0] * 10 + 8, nRefPitch[0] * 10 + 8, nRefPitch[0] * 10 + 0), pucRef, 1);
  zmm6_Ref1213 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 13 + 16, nRefPitch[0] * 13 + 8, nRefPitch[0] * 13 + 8, nRefPitch[0] * 13 + 0, nRefPitch[0] * 12 + 16, nRefPitch[0] * 12 + 8, nRefPitch[0] * 12 + 8, nRefPitch[0] * 12 + 0), pucRef, 1);
  zmm7_Ref1415 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 15 + 16, nRefPitch[0] * 15 + 8, nRefPitch[0] * 15 + 8, nRefPitch[0] * 15 + 0, nRefPitch[0] * 14 + 16, nRefPitch[0] * 14 + 8, nRefPitch[0] * 14 + 8, nRefPitch[0] * 14 + 0), pucRef, 1);

  Sad_16x16_avx512
  ymm_sads_R0 = _mm256_castsi128_si256(xmm_sad_ress);

  // shift 1 row
  zmm0_Ref01 = _mm512_permutex2var_epi64(zmm0_Ref01, imm_vshift_rows, zmm1_Ref23);
  zmm1_Ref23 = _mm512_permutex2var_epi64(zmm1_Ref23, imm_vshift_rows, zmm2_Ref45);
  zmm2_Ref45 = _mm512_permutex2var_epi64(zmm2_Ref45, imm_vshift_rows, zmm3_Ref67);
  zmm3_Ref67 = _mm512_permutex2var_epi64(zmm3_Ref67, imm_vshift_rows, zmm4_Ref89);
  zmm4_Ref89 = _mm512_permutex2var_epi64(zmm4_Ref89, imm_vshift_rows, zmm5_Ref1011);
  zmm5_Ref1011 = _mm512_permutex2var_epi64(zmm5_Ref1011, imm_vshift_rows, zmm6_Ref1213);
  zmm6_Ref1213 = _mm512_permutex2var_epi64(zmm6_Ref1213, imm_vshift_rows, zmm7_Ref1415);
  // ref need to be padded to allow 32bytes loads ?
  zmm7_Ref1415 = _mm512_permutex2var_epi32(zmm7_Ref1415, imm_shift_insert_rows, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * 16))));

  Sad_16x16_avx512
  ymm_sads_R1 = _mm256_castsi128_si256(xmm_sad_ress);

  // shift 1 row
  zmm0_Ref01 = _mm512_permutex2var_epi64(zmm0_Ref01, imm_vshift_rows, zmm1_Ref23);
  zmm1_Ref23 = _mm512_permutex2var_epi64(zmm1_Ref23, imm_vshift_rows, zmm2_Ref45);
  zmm2_Ref45 = _mm512_permutex2var_epi64(zmm2_Ref45, imm_vshift_rows, zmm3_Ref67);
  zmm3_Ref67 = _mm512_permutex2var_epi64(zmm3_Ref67, imm_vshift_rows, zmm4_Ref89);
  zmm4_Ref89 = _mm512_permutex2var_epi64(zmm4_Ref89, imm_vshift_rows, zmm5_Ref1011);
  zmm5_Ref1011 = _mm512_permutex2var_epi64(zmm5_Ref1011, imm_vshift_rows, zmm6_Ref1213);
  zmm6_Ref1213 = _mm512_permutex2var_epi64(zmm6_Ref1213, imm_vshift_rows, zmm7_Ref1415);
  // ref need to be padded to allow 32bytes loads ?
  zmm7_Ref1415 = _mm512_permutex2var_epi32(zmm7_Ref1415, imm_shift_insert_rows, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * 17))));

  Sad_16x16_avx512
  ymm_sads_R2 = _mm256_castsi128_si256(xmm_sad_ress);

  // set high sads, leave only 2,1,0
  ymm_sads_R0 = _mm256_blend_epi16(ymm_sads_R0, ymm_all_ones, 248);
  ymm_sads_R1 = _mm256_blend_epi16(ymm_sads_R1, ymm_all_ones, 248);
  ymm_sads_R2 = _mm256_blend_epi16(ymm_sads_R2, ymm_all_ones, 248);

  unsigned int uiRes_R0 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R0)));
  unsigned int uiRes_R1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R1)));
  unsigned int uiRes_R2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R2)));

  int dx_minsad, dy_minsad, minsad;

  if ((unsigned short)uiRes_R0 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R0;
    dy_minsad = -1;
    dx_minsad = (uiRes_R0 >> 16) - 1;
  }
  else // minsad r1 >= minsad r0
  {
    minsad = (unsigned short)uiRes_R1;
    dy_minsad = 0;
    dx_minsad = (uiRes_R1 >> 16) - 1;
  }

  if ((unsigned short)uiRes_R2 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R2;
    dy_minsad = 1;
    dx_minsad = (uiRes_R2 >> 16) - 1;
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}

void PlaneOfBlocks::ExhaustiveSearch16x16_uint8_SO2_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy)
{
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m512i zmm0_Ref01, zmm1_Ref23, zmm2_Ref45, zmm3_Ref67, zmm4_Ref89, zmm5_Ref1011, zmm6_Ref1213, zmm7_Ref1415;
  __m512i zmm8_Src01, zmm9_Src23, zmm10_Src45, zmm11_Src67, zmm12_Src89, zmm13_Src1011, zmm14_Src1213, zmm15_Src1415;

  __m512i zmm16_sad01, zmm17_sad23, zmm18_sad45, zmm19_sad67, zmm20_sad89, zmm21_sad1011, zmm22_sad1213, zmm23_sad1415;

  __m128i xmm_sad_ress;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  const __m512i idx_set_src = _mm512_set_epi32(13, 13, 12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 1, 1, 0, 0);
  const __m512i imm_shift_insert_rows = _mm512_set_epi32(21, 20, 19, 18, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8);
  const __m512i imm_vshift_rows = _mm512_set_epi64(11, 10, 9, 8, 7, 6, 5, 4);

  // src
  zmm8_Src01 = _mm512_gather16x16_src(1, 0);
  zmm9_Src23 = _mm512_gather16x16_src(3, 2);
  zmm10_Src45 = _mm512_gather16x16_src(5, 4);
  zmm11_Src67 = _mm512_gather16x16_src(7, 6);
  zmm12_Src89 = _mm512_gather16x16_src(9, 8);
  zmm13_Src1011 = _mm512_gather16x16_src(11, 10);
  zmm14_Src1213 = _mm512_gather16x16_src(13, 12);
  zmm15_Src1415 = _mm512_gather16x16_src(15, 14);

  // ref 
  zmm0_Ref01 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 1 + 16, nRefPitch[0] * 1 + 8, nRefPitch[0] * 1 + 8, nRefPitch[0] * 1 + 0, nRefPitch[0] * 0 + 16, nRefPitch[0] * 0 + 8, nRefPitch[0] * 0 + 8, nRefPitch[0] * 0 + 0), pucRef, 1);
  zmm1_Ref23 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 3 + 16, nRefPitch[0] * 3 + 8, nRefPitch[0] * 3 + 8, nRefPitch[0] * 3 + 0, nRefPitch[0] * 2 + 16, nRefPitch[0] * 2 + 8, nRefPitch[0] * 2 + 8, nRefPitch[0] * 2 + 0), pucRef, 1);
  zmm2_Ref45 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 5 + 16, nRefPitch[0] * 5 + 8, nRefPitch[0] * 5 + 8, nRefPitch[0] * 5 + 0, nRefPitch[0] * 4 + 16, nRefPitch[0] * 4 + 8, nRefPitch[0] * 4 + 8, nRefPitch[0] * 4 + 0), pucRef, 1);
  zmm3_Ref67 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 7 + 16, nRefPitch[0] * 7 + 8, nRefPitch[0] * 7 + 8, nRefPitch[0] * 7 + 0, nRefPitch[0] * 6 + 16, nRefPitch[0] * 6 + 8, nRefPitch[0] * 6 + 8, nRefPitch[0] * 6 + 0), pucRef, 1);
  zmm4_Ref89 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 9 + 16, nRefPitch[0] * 9 + 8, nRefPitch[0] * 9 + 8, nRefPitch[0] * 9 + 0, nRefPitch[0] * 8 + 16, nRefPitch[0] * 8 + 8, nRefPitch[0] * 8 + 8, nRefPitch[0] * 8 + 0), pucRef, 1);
  zmm5_Ref1011 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 11 + 16, nRefPitch[0] * 11 + 8, nRefPitch[0] * 11 + 8, nRefPitch[0] * 11 + 0, nRefPitch[0] * 10 + 16, nRefPitch[0] * 10 + 8, nRefPitch[0] * 10 + 8, nRefPitch[0] * 10 + 0), pucRef, 1);
  zmm6_Ref1213 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 13 + 16, nRefPitch[0] * 13 + 8, nRefPitch[0] * 13 + 8, nRefPitch[0] * 13 + 0, nRefPitch[0] * 12 + 16, nRefPitch[0] * 12 + 8, nRefPitch[0] * 12 + 8, nRefPitch[0] * 12 + 0), pucRef, 1);
  zmm7_Ref1415 = _mm512_i64gather_epi64(_mm512_set_epi64(nRefPitch[0] * 15 + 16, nRefPitch[0] * 15 + 8, nRefPitch[0] * 15 + 8, nRefPitch[0] * 15 + 0, nRefPitch[0] * 14 + 16, nRefPitch[0] * 14 + 8, nRefPitch[0] * 14 + 8, nRefPitch[0] * 14 + 0), pucRef, 1);

  Sad_16x16_avx512
    ymm_sads_R0 = _mm256_castsi128_si256(xmm_sad_ress);

  // shift 1 row
  zmm0_Ref01 = _mm512_permutex2var_epi64(zmm0_Ref01, imm_vshift_rows, zmm1_Ref23);
  zmm1_Ref23 = _mm512_permutex2var_epi64(zmm1_Ref23, imm_vshift_rows, zmm2_Ref45);
  zmm2_Ref45 = _mm512_permutex2var_epi64(zmm2_Ref45, imm_vshift_rows, zmm3_Ref67);
  zmm3_Ref67 = _mm512_permutex2var_epi64(zmm3_Ref67, imm_vshift_rows, zmm4_Ref89);
  zmm4_Ref89 = _mm512_permutex2var_epi64(zmm4_Ref89, imm_vshift_rows, zmm5_Ref1011);
  zmm5_Ref1011 = _mm512_permutex2var_epi64(zmm5_Ref1011, imm_vshift_rows, zmm6_Ref1213);
  zmm6_Ref1213 = _mm512_permutex2var_epi64(zmm6_Ref1213, imm_vshift_rows, zmm7_Ref1415);
  // ref need to be padded to allow 32bytes loads ?
  zmm7_Ref1415 = _mm512_permutex2var_epi32(zmm7_Ref1415, imm_shift_insert_rows, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * 16))));

  Sad_16x16_avx512
    ymm_sads_R1 = _mm256_castsi128_si256(xmm_sad_ress);

  // shift 1 row
  zmm0_Ref01 = _mm512_permutex2var_epi64(zmm0_Ref01, imm_vshift_rows, zmm1_Ref23);
  zmm1_Ref23 = _mm512_permutex2var_epi64(zmm1_Ref23, imm_vshift_rows, zmm2_Ref45);
  zmm2_Ref45 = _mm512_permutex2var_epi64(zmm2_Ref45, imm_vshift_rows, zmm3_Ref67);
  zmm3_Ref67 = _mm512_permutex2var_epi64(zmm3_Ref67, imm_vshift_rows, zmm4_Ref89);
  zmm4_Ref89 = _mm512_permutex2var_epi64(zmm4_Ref89, imm_vshift_rows, zmm5_Ref1011);
  zmm5_Ref1011 = _mm512_permutex2var_epi64(zmm5_Ref1011, imm_vshift_rows, zmm6_Ref1213);
  zmm6_Ref1213 = _mm512_permutex2var_epi64(zmm6_Ref1213, imm_vshift_rows, zmm7_Ref1415);
  // ref need to be padded to allow 32bytes loads ?
  zmm7_Ref1415 = _mm512_permutex2var_epi32(zmm7_Ref1415, imm_shift_insert_rows, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(pucRef + nRefPitch[0] * 17))));

  Sad_16x16_avx512
    ymm_sads_R2 = _mm256_castsi128_si256(xmm_sad_ress);

  // set high sads, leave only 2,1,0
  ymm_sads_R0 = _mm256_blend_epi16(ymm_sads_R0, ymm_all_ones, 248);
  ymm_sads_R1 = _mm256_blend_epi16(ymm_sads_R1, ymm_all_ones, 248);
  ymm_sads_R2 = _mm256_blend_epi16(ymm_sads_R2, ymm_all_ones, 248);

  unsigned int uiRes_R0 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R0)));
  unsigned int uiRes_R1 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R1)));
  unsigned int uiRes_R2 = _mm_cvtsi128_si32(_mm_minpos_epu16(_mm256_castsi256_si128(ymm_sads_R2)));

  int dx_minsad, dy_minsad, minsad;

  if ((unsigned short)uiRes_R0 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R0;
    dy_minsad = -1;
    dx_minsad = (uiRes_R0 >> 16) - 1;
  }
  else // minsad r1 >= minsad r0
  {
    minsad = (unsigned short)uiRes_R1;
    dy_minsad = 0;
    dx_minsad = (uiRes_R1 >> 16) - 1;
  }

  if ((unsigned short)uiRes_R2 < (unsigned short)uiRes_R1)
  {
    minsad = (unsigned short)uiRes_R2;
    dy_minsad = 1;
    dx_minsad = (uiRes_R2 >> 16) - 1;
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost)
  {
    _mm256_zeroupper();
    return;
  }

  workarea.bestMV.x = mvx + dx_minsad;
  workarea.bestMV.y = mvy + dy_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

  _mm256_zeroupper();

}

