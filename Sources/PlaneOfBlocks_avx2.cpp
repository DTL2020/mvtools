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


#define sad_block_8x8 \
  xmm8_Ref0 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 0)); \
  xmm9_Ref1 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 1)); \
  xmm10_Ref2 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 2)); \
  xmm11_Ref3 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 3)); \
  xmm12_Ref4 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 4)); \
  xmm13_Ref5 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 5)); \
  xmm14_Ref6 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 6)); \
  xmm15_Ref7 = _mm_loadl_epi64((__m128i*)(pucRef + nRefPitch[0] * 7)); \
 \
  xmm8_Ref0 = _mm_sad_epu8(xmm8_Ref0, xmm0_Src0); \
  xmm9_Ref1 = _mm_sad_epu8(xmm9_Ref1, xmm1_Src1); \
  xmm10_Ref2 = _mm_sad_epu8(xmm10_Ref2, xmm2_Src2); \
  xmm11_Ref3 = _mm_sad_epu8(xmm11_Ref3, xmm3_Src3); \
  xmm12_Ref4 = _mm_sad_epu8(xmm12_Ref4, xmm4_Src4); \
  xmm13_Ref5 = _mm_sad_epu8(xmm13_Ref5, xmm5_Src5); \
  xmm14_Ref6 = _mm_sad_epu8(xmm14_Ref6, xmm6_Src6); \
  xmm15_Ref7 = _mm_sad_epu8(xmm15_Ref7, xmm7_Src7); \
 \
  xmm8_Ref0 = _mm_adds_epu16(xmm8_Ref0, xmm9_Ref1); \
  xmm10_Ref2 = _mm_adds_epu16(xmm10_Ref2, xmm11_Ref3); \
  xmm12_Ref4 = _mm_adds_epu16(xmm12_Ref4, xmm13_Ref5); \
  xmm14_Ref6 = _mm_adds_epu16(xmm14_Ref6, xmm15_Ref7); \
  xmm8_Ref0 = _mm_adds_epu16(xmm8_Ref0, xmm10_Ref2); \
  xmm12_Ref4 = _mm_adds_epu16(xmm12_Ref4, xmm14_Ref6); \
  xmm8_Ref0 = _mm_adds_epu16(xmm8_Ref0, xmm12_Ref4);

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



void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy)
{
  const uint8_t* pucRef = (uint8_t*)GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = (uint8_t*)workarea.pSrc[0];

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


void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy)
{

  const uint8_t* pucRef = (uint8_t*)GetRefBlock(workarea, mvx - 2, mvy - 2); // upper left corner
  const uint8_t* pucCurr = (uint8_t*)workarea.pSrc[0];

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

#define SAD_4blocks8x8 /*AVX2*/\
/* calc sads with ref */ \
ymm8_r1 = _mm256_sad_epu8(workarea.ymm0_src_r1, *(__m256i*)(pucRef + nRefPitch[0] * (y + 0) + x)); \
ymm9_r2 = _mm256_sad_epu8(workarea.ymm1_src_r2, *(__m256i*)(pucRef + nRefPitch[0] * (y + 1) + x)); \
ymm10_r3 = _mm256_sad_epu8(workarea.ymm2_src_r3, *(__m256i*)(pucRef + nRefPitch[0] * (y + 2) + x)); \
ymm11_r4 = _mm256_sad_epu8(workarea.ymm3_src_r4, *(__m256i*)(pucRef + nRefPitch[0] * (y + 3) + x)); \
ymm12_r5 = _mm256_sad_epu8(workarea.ymm4_src_r5, *(__m256i*)(pucRef + nRefPitch[0] * (y + 4) + x)); \
ymm13_r6 = _mm256_sad_epu8(workarea.ymm5_src_r6, *(__m256i*)(pucRef + nRefPitch[0] * (y + 5) + x)); \
ymm14_r7 = _mm256_sad_epu8(workarea.ymm6_src_r7, *(__m256i*)(pucRef + nRefPitch[0] * (y + 6) + x)); \
ymm15_r8 = _mm256_sad_epu8(workarea.ymm7_src_r8, *(__m256i*)(pucRef + nRefPitch[0] * (y + 7) + x)); \
\
ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm9_r2); \
ymm10_r3 = _mm256_adds_epu16(ymm10_r3, ymm11_r4); \
ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm13_r6); \
ymm14_r7 = _mm256_adds_epu16(ymm14_r7, ymm15_r8); \
\
ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm10_r3); \
ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm14_r7); \
\
ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm12_r5);\
\
ymm8_r1 = _mm256_slli_epi64(ymm8_r1, 48);

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy, int* pBlkData)
{

  /*
ymm8_r1 = _mm256_sad_epu8(ymm0_src_r1, *(__m256i*)(pucRef + nRefPitch[0] * (y + 0) + x)); \
ymm9_r2 = _mm256_sad_epu8(ymm1_src_r2, *(__m256i*)(pucRef + nRefPitch[0] * (y + 1) + x)); \
ymm10_r3 = _mm256_sad_epu8(ymm2_src_r3, *(__m256i*)(pucRef + nRefPitch[0] * (y + 2) + x)); \
ymm11_r4 = _mm256_sad_epu8(ymm3_src_r4, *(__m256i*)(pucRef + nRefPitch[0] * (y + 3) + x)); \
ymm12_r5 = _mm256_sad_epu8(ymm4_src_r5, *(__m256i*)(pucRef + nRefPitch[0] * (y + 4) + x)); \
ymm13_r6 = _mm256_sad_epu8(ymm5_src_r6, *(__m256i*)(pucRef + nRefPitch[0] * (y + 5) + x)); \
ymm14_r7 = _mm256_sad_epu8(ymm6_src_r7, *(__m256i*)(pucRef + nRefPitch[0] * (y + 6) + x)); \
ymm15_r8 = _mm256_sad_epu8(ymm7_src_r8, *(__m256i*)(pucRef + nRefPitch[0] * (y + 7) + x)); \

  */

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];
/*
  // 4 blocks at once proc, pel=1
  __m256i ymm0_src_r1 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 0));
  __m256i	ymm1_src_r2 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 1));
  __m256i	ymm2_src_r3 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 2));
  __m256i	ymm3_src_r4 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 3));
  __m256i	ymm4_src_r5 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 4));
  __m256i	ymm5_src_r6 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 5));
  __m256i	ymm6_src_r7 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 6));
  __m256i	ymm7_src_r8 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 7));
  */
  __m256i ymm8_r1, ymm9_r2, ymm10_r3, ymm11_r4, ymm12_r5, ymm13_r6, ymm14_r7, ymm15_r8;

  int x, y;
  __m256i part_sads1, part_sads2;
#ifdef _DEBUG
  part_sads1 = _mm256_setzero_si256();
  part_sads2 = _mm256_setzero_si256();
#endif

  // 1st 4sads
  y = 0; x = 0;
  SAD_4blocks8x8
  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136);
  part_sads1 = _mm256_srli_si256(part_sads1, 2);

  // 2nd 4sads
  x = 1; y = 0;
  SAD_4blocks8x8

  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136);
  part_sads1 = _mm256_srli_si256(part_sads1, 2);

  // 3rd 4sads
  x = 2; y = 0;
  SAD_4blocks8x8
  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136);
  part_sads1 = _mm256_srli_si256(part_sads1, 2);

  // 4th 4sads
  x = 0; y = 1;
  SAD_4blocks8x8
  part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136); // part_sads1 ready 4x

  /*	// 5th 4sads
    x = 1; y = 1; //
    SAD_4blocks // skip check zero position, add 1 to minpos if > 4 !
    */

  // 5th 4sads
  x = 2; y = 1;
  SAD_4blocks8x8
  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136);
  part_sads2 = _mm256_srli_si256(part_sads2, 2);

  // 6th 4sads
  x = 0; y = 2;
  SAD_4blocks8x8
  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136);
  part_sads2 = _mm256_srli_si256(part_sads2, 2);

  // 7th 4sads
  x = 1; y = 2;
  SAD_4blocks8x8
  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136);
  part_sads2 = _mm256_srli_si256(part_sads2, 2);

  // 8th 4sads
  x = 2; y = 2;
  SAD_4blocks8x8
  part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136); // part_sads2 ready 4x4

  // 8 SADs of 1 block
  __m128i xmm0_sad1 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads1, _mm256_slli_si256(part_sads2, 8), 240));

  // 8 SADS of 2 block
  __m128i xmm1_sad2 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads2, _mm256_srli_si256(part_sads1, 8), 15));

  part_sads1 = _mm256_permute4x64_epi64(part_sads1, 14); // move high 128bits to low 128 bits
  part_sads2 = _mm256_permute4x64_epi64(part_sads2, 14); // move high 128bits to low 128 bits

  // 8 SADs of 3 block
  __m128i xmm2_sad3 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads1, _mm256_slli_si256(part_sads2, 8), 240));

  // 8 SADs of 4 block
  __m128i xmm3_sad4 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads2, _mm256_srli_si256(part_sads1, 8), 15));

  xmm0_sad1 = _mm_minpos_epu16(xmm0_sad1);
  xmm1_sad2 = _mm_minpos_epu16(xmm1_sad2);
  xmm2_sad3 = _mm_minpos_epu16(xmm2_sad3);
  xmm3_sad4 = _mm_minpos_epu16(xmm3_sad4);

  // add +1 to minpos if > 3
  __m128i xmm_uint3 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 3, 0);
  __m128i xmm_uint1 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 1, 0);
 
  xmm0_sad1 = _mm_blendv_epi8(xmm0_sad1, _mm_add_epi16(xmm0_sad1, xmm_uint1), _mm_cmpgt_epi16(xmm0_sad1, xmm_uint3));
  xmm1_sad2 = _mm_blendv_epi8(xmm1_sad2, _mm_add_epi16(xmm1_sad2, xmm_uint1), _mm_cmpgt_epi16(xmm1_sad2, xmm_uint3));
  xmm2_sad3 = _mm_blendv_epi8(xmm2_sad3, _mm_add_epi16(xmm2_sad3, xmm_uint1), _mm_cmpgt_epi16(xmm2_sad3, xmm_uint3));
  xmm3_sad4 = _mm_blendv_epi8(xmm3_sad4, _mm_add_epi16(xmm3_sad4, xmm_uint1), _mm_cmpgt_epi16(xmm3_sad4, xmm_uint3));

  unsigned short minsad1 = (unsigned short)_mm_cvtsi128_si32(xmm0_sad1);
  unsigned short minsad2 = (unsigned short)_mm_cvtsi128_si32(xmm1_sad2);
  unsigned short minsad3 = (unsigned short)_mm_cvtsi128_si32(xmm2_sad3);
  unsigned short minsad4 = (unsigned short)_mm_cvtsi128_si32(xmm3_sad4);

  __m128i xmm_idx = xmm0_sad1;
  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_slli_si128(xmm1_sad2, 4), 8);
  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_slli_si128(xmm2_sad3, 8), 32);
  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_slli_si128(xmm3_sad4, 12), 128);

  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_srli_si128(xmm_idx, 2), 85);

  //	int dx_minsad = (idx_min_sad % 3) - 1; - just comment where from x,y minsad come from
  //	int dy_minsad = (idx_min_sad / 3) - 1;

  // dx = idx*11>>5 , dy = (((idx*11)&31)*3)>>5
  __m128i xmm_uint11 = _mm_set_epi16(11, 11, 11, 11, 11, 11, 11, 11);
  __m128i xmm_and = _mm_set_epi16(-1, 31, -1, 31, -1, 31, -1, 31);
  __m128i xmm_1_3 = _mm_set_epi16(1, 3, 1, 3, 1, 3, 1, 3);
  __m128i xmm_1 = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);

  xmm_idx = _mm_mullo_epi16(xmm_idx, xmm_uint11);
  xmm_idx = _mm_and_si128(xmm_idx, xmm_and);
  xmm_idx = _mm_mullo_epi16(xmm_idx, xmm_1_3);
  xmm_idx = _mm_srli_epi16(xmm_idx, 5);
  xmm_idx = _mm_sub_epi16(xmm_idx, xmm_1); // global -1
/*
  int dx_minsad1 = _mm_extract_epi16(xmm_idx, 0);
  int dy_minsad1 = _mm_extract_epi16(xmm_idx, 1);

  int dx_minsad2 = _mm_extract_epi16(xmm_idx, 2);
  int dy_minsad2 = _mm_extract_epi16(xmm_idx, 3);

  int dx_minsad3 = _mm_extract_epi16(xmm_idx, 4);
  int dy_minsad3 = _mm_extract_epi16(xmm_idx, 5);

  int dx_minsad4 = _mm_extract_epi16(xmm_idx, 6); 
  int dy_minsad4 = _mm_extract_epi16(xmm_idx, 7);*/

  //  if (cost >= workarea.nMinCost)

  if (minsad1 < workarea.bestMV.sad)
  {
    pBlkData[workarea.blkx * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 0); // dx_minsad1
    pBlkData[workarea.blkx * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 1); // dy_minsad1
    pBlkData[workarea.blkx * N_PER_BLOCK + 2] = (uint32_t)(minsad1);

    workarea.bestMV.x += _mm_extract_epi16(xmm_idx, 0);
    workarea.bestMV.y += _mm_extract_epi16(xmm_idx, 1);
    workarea.nMinCost = minsad1 + ((penaltyNew * minsad1) >> 8);
    workarea.bestMV.sad = minsad1;
  }

  if (minsad2 < pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 2); // dx_minsad2
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 3); // dy_minsad2;
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = (uint32_t)(minsad2);
  }

  if (minsad3 < pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 4); // dx_minsad3;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 5); // dy_minsad3;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = (uint32_t)(minsad3);
  }

  if (minsad4 < pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 6); // dx_minsad4;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 7); // dy_minsad4;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = (uint32_t)(minsad4);
  }
  
//  _mm256_zeroupper(); - compiler should add if required ?

}

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_4Blks_Z_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy, int* pBlkData)
{

//#define SAD_4blocks8x8_Z /*AVX2*/\
// calc sads with ref  \
ymm8_r1 = _mm256_sad_epu8(ymm0_src_r1, *(__m256i*)(pucRef + nRefPitch[0] * (y + 0) + x)); \
ymm9_r2 = _mm256_sad_epu8(ymm1_src_r2, *(__m256i*)(pucRef + nRefPitch[0] * (y + 1) + x)); \
ymm10_r3 = _mm256_sad_epu8(ymm2_src_r3, *(__m256i*)(pucRef + nRefPitch[0] * (y + 2) + x)); \
ymm11_r4 = _mm256_sad_epu8(ymm3_src_r4, *(__m256i*)(pucRef + nRefPitch[0] * (y + 3) + x)); \
ymm12_r5 = _mm256_sad_epu8(ymm4_src_r5, *(__m256i*)(pucRef + nRefPitch[0] * (y + 4) + x)); \
ymm13_r6 = _mm256_sad_epu8(ymm5_src_r6, *(__m256i*)(pucRef + nRefPitch[0] * (y + 5) + x)); \
ymm14_r7 = _mm256_sad_epu8(ymm6_src_r7, *(__m256i*)(pucRef + nRefPitch[0] * (y + 6) + x)); \
ymm15_r8 = _mm256_sad_epu8(ymm7_src_r8, *(__m256i*)(pucRef + nRefPitch[0] * (y + 7) + x)); \
\
ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm9_r2); \
ymm10_r3 = _mm256_adds_epu16(ymm10_r3, ymm11_r4); \
ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm13_r6); \
ymm14_r7 = _mm256_adds_epu16(ymm14_r7, ymm15_r8); \
\
ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm10_r3); \
ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm14_r7); \
\
ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm12_r5);\
\
ymm8_r1 = _mm256_slli_epi64(ymm8_r1, 48);

  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  // 4 blocks at once proc, pel=1
/*  __m256i ymm0_src_r1 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 0));
  __m256i	ymm1_src_r2 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 1));
  __m256i	ymm2_src_r3 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 2));
  __m256i	ymm3_src_r4 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 3));
  __m256i	ymm4_src_r5 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 4));
  __m256i	ymm5_src_r6 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 5));
  __m256i	ymm6_src_r7 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 6));
  __m256i	ymm7_src_r8 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 7));
*/
  workarea.ymm0_src_r1 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 0));
  workarea.ymm1_src_r2 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 1));
  workarea.ymm2_src_r3 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 2));
  workarea.ymm3_src_r4 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 3));
  workarea.ymm4_src_r5 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 4));
  workarea.ymm5_src_r6 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 5));
  workarea.ymm6_src_r7 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 6));
  workarea.ymm7_src_r8 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 7));

  __m256i ymm8_r1, ymm9_r2, ymm10_r3, ymm11_r4, ymm12_r5, ymm13_r6, ymm14_r7, ymm15_r8;

  int x, y;
  __m256i part_sads1, part_sads2;
#ifdef _DEBUG
  part_sads1 = _mm256_setzero_si256();
  part_sads2 = _mm256_setzero_si256();
#endif
  // check zero pos first
  x = 1; y = 1; //
  ymm8_r1 = _mm256_sad_epu8(workarea.ymm0_src_r1, *(__m256i*)(pucRef + nRefPitch[0] * (y + 0) + x)); 
  ymm9_r2 = _mm256_sad_epu8(workarea.ymm1_src_r2, *(__m256i*)(pucRef + nRefPitch[0] * (y + 1) + x));
  ymm10_r3 = _mm256_sad_epu8(workarea.ymm2_src_r3, *(__m256i*)(pucRef + nRefPitch[0] * (y + 2) + x));
  ymm11_r4 = _mm256_sad_epu8(workarea.ymm3_src_r4, *(__m256i*)(pucRef + nRefPitch[0] * (y + 3) + x));
  ymm12_r5 = _mm256_sad_epu8(workarea.ymm4_src_r5, *(__m256i*)(pucRef + nRefPitch[0] * (y + 4) + x));
  ymm13_r6 = _mm256_sad_epu8(workarea.ymm5_src_r6, *(__m256i*)(pucRef + nRefPitch[0] * (y + 5) + x));
  ymm14_r7 = _mm256_sad_epu8(workarea.ymm6_src_r7, *(__m256i*)(pucRef + nRefPitch[0] * (y + 6) + x));
  ymm15_r8 = _mm256_sad_epu8(workarea.ymm7_src_r8, *(__m256i*)(pucRef + nRefPitch[0] * (y + 7) + x));
  
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm9_r2); 
  ymm10_r3 = _mm256_adds_epu16(ymm10_r3, ymm11_r4); 
  ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm13_r6);
  ymm14_r7 = _mm256_adds_epu16(ymm14_r7, ymm15_r8); 
  
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm10_r3); 
  ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm14_r7); 

  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm12_r5); 

  workarea.bestMV.sad = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm8_r1));
  pBlkData[(workarea.blkx + 0) * N_PER_BLOCK + 2] = workarea.bestMV.sad;
  pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = _mm256_extract_epi32(ymm8_r1, 2);
  pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = _mm256_extract_epi32(ymm8_r1, 4);
  pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = _mm256_extract_epi32(ymm8_r1, 6);


  // 1st 4sads
  y = 0; x = 0;
  SAD_4blocks8x8
    part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136);
  part_sads1 = _mm256_srli_si256(part_sads1, 2);

  // 2nd 4sads
  x = 1; y = 0;
  SAD_4blocks8x8

    part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136);
  part_sads1 = _mm256_srli_si256(part_sads1, 2);

  // 3rd 4sads
  x = 2; y = 0;
  SAD_4blocks8x8
    part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136);
  part_sads1 = _mm256_srli_si256(part_sads1, 2);

  // 4th 4sads
  x = 0; y = 1;
  SAD_4blocks8x8
    part_sads1 = _mm256_blend_epi16(part_sads1, ymm8_r1, 136); // part_sads1 ready 4x

    /*	// 5th 4sads
      x = 1; y = 1; //
      SAD_4blocks // skip check zero position, add 1 to minpos if > 4 !
      */

      // 5th 4sads
  x = 2; y = 1;
  SAD_4blocks8x8
    part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136);
  part_sads2 = _mm256_srli_si256(part_sads2, 2);

  // 6th 4sads
  x = 0; y = 2;
  SAD_4blocks8x8
    part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136);
  part_sads2 = _mm256_srli_si256(part_sads2, 2);

  // 7th 4sads
  x = 1; y = 2;
  SAD_4blocks8x8
    part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136);
  part_sads2 = _mm256_srli_si256(part_sads2, 2);

  // 8th 4sads
  x = 2; y = 2;
  SAD_4blocks8x8
    part_sads2 = _mm256_blend_epi16(part_sads2, ymm8_r1, 136); // part_sads2 ready 4x4

    // 8 SADs of 1 block
  __m128i xmm0_sad1 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads1, _mm256_slli_si256(part_sads2, 8), 240));

  // 8 SADS of 2 block
  __m128i xmm1_sad2 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads2, _mm256_srli_si256(part_sads1, 8), 15));

  part_sads1 = _mm256_permute4x64_epi64(part_sads1, 14); // move high 128bits to low 128 bits
  part_sads2 = _mm256_permute4x64_epi64(part_sads2, 14); // move high 128bits to low 128 bits

  // 8 SADs of 3 block
  __m128i xmm2_sad3 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads1, _mm256_slli_si256(part_sads2, 8), 240));

  // 8 SADs of 4 block
  __m128i xmm3_sad4 = _mm256_castsi256_si128(_mm256_blend_epi16(part_sads2, _mm256_srli_si256(part_sads1, 8), 15));

  xmm0_sad1 = _mm_minpos_epu16(xmm0_sad1);
  xmm1_sad2 = _mm_minpos_epu16(xmm1_sad2);
  xmm2_sad3 = _mm_minpos_epu16(xmm2_sad3);
  xmm3_sad4 = _mm_minpos_epu16(xmm3_sad4);

  // add +1 to minpos if > 3
  __m128i xmm_uint3 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 3, 0);
  __m128i xmm_uint1 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 1, 0);

  xmm0_sad1 = _mm_blendv_epi8(xmm0_sad1, _mm_add_epi16(xmm0_sad1, xmm_uint1), _mm_cmpgt_epi16(xmm0_sad1, xmm_uint3));
  xmm1_sad2 = _mm_blendv_epi8(xmm1_sad2, _mm_add_epi16(xmm1_sad2, xmm_uint1), _mm_cmpgt_epi16(xmm1_sad2, xmm_uint3));
  xmm2_sad3 = _mm_blendv_epi8(xmm2_sad3, _mm_add_epi16(xmm2_sad3, xmm_uint1), _mm_cmpgt_epi16(xmm2_sad3, xmm_uint3));
  xmm3_sad4 = _mm_blendv_epi8(xmm3_sad4, _mm_add_epi16(xmm3_sad4, xmm_uint1), _mm_cmpgt_epi16(xmm3_sad4, xmm_uint3));

  unsigned short minsad1 = (unsigned short)_mm_cvtsi128_si32(xmm0_sad1);
  unsigned short minsad2 = (unsigned short)_mm_cvtsi128_si32(xmm1_sad2);
  unsigned short minsad3 = (unsigned short)_mm_cvtsi128_si32(xmm2_sad3);
  unsigned short minsad4 = (unsigned short)_mm_cvtsi128_si32(xmm3_sad4);

  __m128i xmm_idx = xmm0_sad1;
  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_slli_si128(xmm1_sad2, 4), 8);
  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_slli_si128(xmm2_sad3, 8), 32);
  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_slli_si128(xmm3_sad4, 12), 128);

  xmm_idx = _mm_blend_epi16(xmm_idx, _mm_srli_si128(xmm_idx, 2), 85);

  //	int dx_minsad = (idx_min_sad % 3) - 1; - just comment where from x,y minsad come from
  //	int dy_minsad = (idx_min_sad / 3) - 1;

  // dx = idx*11>>5 , dy = (((idx*11)&31)*3)>>5
  __m128i xmm_uint11 = _mm_set_epi16(11, 11, 11, 11, 11, 11, 11, 11);
  __m128i xmm_and = _mm_set_epi16(-1, 31, -1, 31, -1, 31, -1, 31);
  __m128i xmm_1_3 = _mm_set_epi16(1, 3, 1, 3, 1, 3, 1, 3);
  __m128i xmm_1 = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);

  xmm_idx = _mm_mullo_epi16(xmm_idx, xmm_uint11);
  xmm_idx = _mm_and_si128(xmm_idx, xmm_and);
  xmm_idx = _mm_mullo_epi16(xmm_idx, xmm_1_3);
  xmm_idx = _mm_srli_epi16(xmm_idx, 5);
  xmm_idx = _mm_sub_epi16(xmm_idx, xmm_1); // global -1
/*
  int dx_minsad1 = _mm_extract_epi16(xmm_idx, 0);
  int dy_minsad1 = _mm_extract_epi16(xmm_idx, 1);

  int dx_minsad2 = _mm_extract_epi16(xmm_idx, 2);
  int dy_minsad2 = _mm_extract_epi16(xmm_idx, 3);

  int dx_minsad3 = _mm_extract_epi16(xmm_idx, 4);
  int dy_minsad3 = _mm_extract_epi16(xmm_idx, 5);

  int dx_minsad4 = _mm_extract_epi16(xmm_idx, 6);
  int dy_minsad4 = _mm_extract_epi16(xmm_idx, 7);*/

  //  if (cost >= workarea.nMinCost)

  if (minsad1 < workarea.bestMV.sad)
  {
    pBlkData[workarea.blkx * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 0); // dx_minsad1
    pBlkData[workarea.blkx * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 1); // dy_minsad1
    pBlkData[workarea.blkx * N_PER_BLOCK + 2] = (uint32_t)(minsad1);

    workarea.bestMV.x += (_mm_extract_epi16(xmm_idx, 0) - 1);
    workarea.bestMV.y += (_mm_extract_epi16(xmm_idx, 1) - 1);
    workarea.nMinCost = minsad1 + ((penaltyNew * minsad1) >> 8);
    workarea.bestMV.sad = minsad1;
  }

  if (minsad2 < pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 2); // dx_minsad2
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 3); // dy_minsad2;
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = (uint32_t)(minsad2);
  }

  if (minsad3 < pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 4); // dx_minsad3;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 5); // dy_minsad3;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = (uint32_t)(minsad3);
  }

  if (minsad4 < pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx, 6); // dx_minsad4;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx, 7); // dy_minsad4;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = (uint32_t)(minsad4);
  }

  //  _mm256_zeroupper(); - compiler should add if required ?

}

