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

#include "PlaneOfBlocks.h"
#include <map>
#include <tuple>

#include <stdint.h>
#include "def.h"

#define Load_src16x16(r2,r1) \
  _mm512_permutexvar_epi32(idx_set_src2, _mm512_castsi256_si512(_mm256_loadu2_m128i((__m128i*)(pucCurr + nSrcPitch[0] * r2), (__m128i*)(pucCurr + nSrcPitch[0] * r1))))

#define Load_src8x8(r4,r3,r2,r1) \
  _mm512_permutexvar_epi32(idx_set_src2, _mm512_castsi256_si512(_mm256_set_epi64x(*(pucCurr + nSrcPitch[0] * r4), *(pucCurr + nSrcPitch[0] * r3), *(pucCurr + nSrcPitch[0] * r2),*(pucCurr + nSrcPitch[0] * r1))))


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

#define Load_ref16x16(r2,r1) \
    _mm512_set_epi64(*(pucRef + nRefPitch[0] * r2 + 16), *(pucRef + nRefPitch[0] * r2 + 8), *(pucRef + nRefPitch[0] * r2 + 8), *(pucRef + nRefPitch[0] * r2 + 0), \
    *(pucRef + nRefPitch[0] * r1 + 16), *(pucRef + nRefPitch[0] * r1 + 8), *(pucRef + nRefPitch[0] * r1 + 8), *(pucRef + nRefPitch[0] * r1 + 0))

#define Load_ref8x8(r4,r3,r2,r1) \
    _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * r2), (__m128i*)(pucRef + nRefPitch[0] * r1))), \
      _mm256_loadu2_m128i((__m128i*)(pucRef + nRefPitch[0] * r4), (__m128i*)(pucRef + nRefPitch[0] * r3)), 1)

#define SAD_16blocks8x8 /*AVX512*/\
/* calc sads src with ref */ \
zmm16_r1_b0007 = _mm512_sad_epu8(workarea.zmm0_Src_r1_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 0) + x + 0)); \
zmm17_r1_b0815 = _mm512_sad_epu8(workarea.zmm1_Src_r1_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 0) + x + 64)); \
zmm18_r2_b0007 = _mm512_sad_epu8(workarea.zmm2_Src_r2_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 1) + x + 0)); \
zmm19_r2_b0815 = _mm512_sad_epu8(workarea.zmm3_Src_r2_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 1) + x + 64)); \
zmm20_r3_b0007 = _mm512_sad_epu8(workarea.zmm4_Src_r3_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 2) + x + 0)); \
zmm21_r3_b0815 = _mm512_sad_epu8(workarea.zmm5_Src_r3_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 2) + x + 64)); \
zmm22_r4_b0007 = _mm512_sad_epu8(workarea.zmm6_Src_r4_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 3) + x + 0)); \
zmm23_r4_b0815 = _mm512_sad_epu8(workarea.zmm7_Src_r4_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 3) + x + 64)); \
zmm24_r5_b0007 = _mm512_sad_epu8(workarea.zmm8_Src_r5_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 4) + x + 0)); \
zmm25_r5_b0815 = _mm512_sad_epu8(workarea.zmm9_Src_r5_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 4) + x + 64)); \
zmm26_r6_b0007 = _mm512_sad_epu8(workarea.zmm10_Src_r6_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 5) + x + 0)); \
zmm27_r6_b0815 = _mm512_sad_epu8(workarea.zmm11_Src_r6_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 5) + x + 64)); \
zmm28_r7_b0007 = _mm512_sad_epu8(workarea.zmm12_Src_r7_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 6) + x + 0)); \
zmm29_r7_b0815 = _mm512_sad_epu8(workarea.zmm13_Src_r7_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 6) + x + 64)); \
zmm30_r8_b0007 = _mm512_sad_epu8(workarea.zmm14_Src_r8_b0007, *(__m512i*)(pucRef + nRefPitch[0] * (y + 7) + x + 0)); \
zmm31_r8_b0815 = _mm512_sad_epu8(workarea.zmm15_Src_r8_b0815, *(__m512i*)(pucRef + nRefPitch[0] * (y + 7) + x + 64)); \
\
zmm16_r1_b0007 = _mm512_adds_epi16(zmm16_r1_b0007, zmm18_r2_b0007); \
zmm20_r3_b0007 = _mm512_adds_epi16(zmm20_r3_b0007, zmm22_r4_b0007); \
zmm24_r5_b0007 = _mm512_adds_epi16(zmm24_r5_b0007, zmm26_r6_b0007); \
zmm28_r7_b0007 = _mm512_adds_epi16(zmm28_r7_b0007, zmm30_r8_b0007); \
 \
zmm17_r1_b0815 = _mm512_adds_epi16(zmm17_r1_b0815, zmm19_r2_b0815); \
zmm21_r3_b0815 = _mm512_adds_epi16(zmm21_r3_b0815, zmm23_r4_b0815); \
zmm25_r5_b0815 = _mm512_adds_epi16(zmm25_r5_b0815, zmm27_r6_b0815); \
zmm29_r7_b0815 = _mm512_adds_epi16(zmm29_r7_b0815, zmm31_r8_b0815); \
\
zmm16_r1_b0007 = _mm512_adds_epi16(zmm16_r1_b0007, zmm20_r3_b0007); \
zmm24_r5_b0007 = _mm512_adds_epi16(zmm24_r5_b0007, zmm28_r7_b0007); \
\
zmm17_r1_b0815 = _mm512_adds_epi16(zmm17_r1_b0815, zmm21_r3_b0815); \
zmm25_r5_b0815 = _mm512_adds_epi16(zmm25_r5_b0815, zmm29_r7_b0815); \
\
zmm16_r1_b0007 = _mm512_adds_epi16(zmm16_r1_b0007, zmm24_r5_b0007); \
\
zmm17_r1_b0815 = _mm512_adds_epi16(zmm17_r1_b0815, zmm25_r5_b0815);

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

  //  const __m512i idx_set_src = _mm512_set_epi32(13, 13, 12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 1, 1, 0, 0);
  const __m512i idx_set_src2 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i imm_shift_insert_rows = _mm512_set_epi32(21, 20, 19, 18, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8);
  const __m512i imm_vshift_rows = _mm512_set_epi64(11, 10, 9, 8, 7, 6, 5, 4);

  // src
  zmm8_Src01 = Load_src16x16(1, 0);
  zmm9_Src23 = Load_src16x16(3, 2);
  zmm10_Src45 = Load_src16x16(5, 4);
  zmm11_Src67 = Load_src16x16(7, 6);
  zmm12_Src89 = Load_src16x16(9, 8);
  zmm13_Src1011 = Load_src16x16(11, 10);
  zmm14_Src1213 = Load_src16x16(13, 12);
  zmm15_Src1415 = Load_src16x16(15, 14);

  // ref 
  zmm0_Ref01 = Load_ref16x16(1, 0);
  zmm1_Ref23 = Load_ref16x16(3, 2);
  zmm2_Ref45 = Load_ref16x16(5, 4);
  zmm3_Ref67 = Load_ref16x16(7, 6);
  zmm4_Ref89 = Load_ref16x16(9, 8);
  zmm5_Ref1011 = Load_ref16x16(11, 10);
  zmm6_Ref1213 = Load_ref16x16(13, 12);
  zmm7_Ref1415 = Load_ref16x16(15, 14);


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

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy)
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

  __m512i zmm0_Ref0123, zmm1_Ref4567;
  __m512i zmm2_Src0123, zmm3_Src4567;

  __m512i zmm4_sad_8;

  __m256i ymm_sad_ress;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  const __m512i idx_set_src2 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i imm_shift_insert_rows8 = _mm512_set_epi32(19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
  const __m512i imm_vshift_rows8 = _mm512_set_epi64(9, 8, 7, 6, 5, 4, 3, 2);

  zmm2_Src0123 = Load_src8x8(3, 2, 1, 0);
  zmm3_Src4567 = Load_src8x8(7, 6, 5, 4);

  zmm0_Ref0123 = Load_ref8x8(3, 2, 1, 0);
  zmm1_Ref4567 = Load_ref8x8(7, 6, 5, 4);

  zmm4_sad_8 = _mm512_adds_epu16(_mm512_dbsad_epu8(zmm2_Src0123, zmm0_Ref0123, 148), _mm512_dbsad_epu8(zmm3_Src4567, zmm1_Ref4567, 148));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm4_sad_8));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm4_sad_8));
  ymm_sad_ress = _mm512_castsi512_si256(zmm4_sad_8);
  ymm_sads_R0 = _mm256_adds_epi16(ymm_sad_ress, _mm256_srli_si256(ymm_sad_ress, 8));

  // shift-push 1 row
  zmm0_Ref0123 = _mm512_permutex2var_epi64(zmm0_Ref0123, imm_vshift_rows8, zmm1_Ref4567);
  // ref need to be padded to allow 16bytes loads ?
  zmm1_Ref4567 = _mm512_permutex2var_epi32(zmm1_Ref4567, imm_shift_insert_rows8, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(pucRef + nRefPitch[0] * 8))));

  zmm4_sad_8 = _mm512_adds_epu16(_mm512_dbsad_epu8(zmm2_Src0123, zmm0_Ref0123, 148), _mm512_dbsad_epu8(zmm3_Src4567, zmm1_Ref4567, 148));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm4_sad_8));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm4_sad_8));
  ymm_sad_ress = _mm512_castsi512_si256(zmm4_sad_8);
  ymm_sads_R1 = _mm256_adds_epi16(ymm_sad_ress, _mm256_srli_si256(ymm_sad_ress, 8));

  // shift-push 1 row
  zmm0_Ref0123 = _mm512_permutex2var_epi64(zmm0_Ref0123, imm_vshift_rows8, zmm1_Ref4567);
  // ref need to be padded to allow 16bytes loads ?
  zmm1_Ref4567 = _mm512_permutex2var_epi32(zmm1_Ref4567, imm_shift_insert_rows8, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(pucRef + nRefPitch[0] * 9))));
  zmm4_sad_8 = _mm512_adds_epu16(_mm512_dbsad_epu8(zmm2_Src0123, zmm0_Ref0123, 148), _mm512_dbsad_epu8(zmm3_Src4567, zmm1_Ref4567, 148));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm4_sad_8));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm4_sad_8));
  ymm_sad_ress = _mm512_castsi512_si256(zmm4_sad_8);
  ymm_sads_R2 = _mm256_adds_epi16(ymm_sad_ress, _mm256_srli_si256(ymm_sad_ress, 8));

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

  const __m512i idx_set_src2 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i imm_shift_insert_rows = _mm512_set_epi32(21, 20, 19, 18, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8);
  const __m512i imm_vshift_rows = _mm512_set_epi64(11, 10, 9, 8, 7, 6, 5, 4);

  // src
  zmm8_Src01 = Load_src16x16(1, 0);
  zmm9_Src23 = Load_src16x16(3, 2);
  zmm10_Src45 = Load_src16x16(5, 4);
  zmm11_Src67 = Load_src16x16(7, 6);
  zmm12_Src89 = Load_src16x16(9, 8);
  zmm13_Src1011 = Load_src16x16(11, 10);
  zmm14_Src1213 = Load_src16x16(13, 12);
  zmm15_Src1415 = Load_src16x16(15, 14);

  // ref 
  zmm0_Ref01 = Load_ref16x16(1, 0);
  zmm1_Ref23 = Load_ref16x16(3, 2);
  zmm2_Ref45 = Load_ref16x16(5, 4);
  zmm3_Ref67 = Load_ref16x16(7, 6);
  zmm4_Ref89 = Load_ref16x16(9, 8);
  zmm5_Ref1011 = Load_ref16x16(11, 10);
  zmm6_Ref1213 = Load_ref16x16(13, 12);
  zmm7_Ref1415 = Load_ref16x16(15, 14);

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

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy)
{
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];

  __m512i zmm0_Ref0123, zmm1_Ref4567;
  __m512i zmm2_Src0123, zmm3_Src4567;

  __m512i zmm4_sad_8;

  __m256i ymm_sad_ress;
  __m256i ymm_sads_R0, ymm_sads_R1, ymm_sads_R2;

  __m256i ymm_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  const __m512i idx_set_src2 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i imm_shift_insert_rows8 = _mm512_set_epi32(19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
  const __m512i imm_vshift_rows8 = _mm512_set_epi64(9, 8, 7, 6, 5, 4, 3, 2);

  zmm2_Src0123 = Load_src8x8(3, 2, 1, 0);
  zmm3_Src4567 = Load_src8x8(7, 6, 5, 4);

  zmm0_Ref0123 = Load_ref8x8(3, 2, 1, 0);
  zmm1_Ref4567 = Load_ref8x8(7, 6, 5, 4);

  zmm4_sad_8 = _mm512_adds_epu16(_mm512_dbsad_epu8(zmm2_Src0123, zmm0_Ref0123, 148), _mm512_dbsad_epu8(zmm3_Src4567, zmm1_Ref4567, 148));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm4_sad_8));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm4_sad_8));
  ymm_sad_ress = _mm512_castsi512_si256(zmm4_sad_8);
  ymm_sads_R0 = _mm256_adds_epi16(ymm_sad_ress, _mm256_srli_si256(ymm_sad_ress, 8));

  // shift-push 1 row
  zmm0_Ref0123 = _mm512_permutex2var_epi64(zmm0_Ref0123, imm_vshift_rows8, zmm1_Ref4567);
  // ref need to be padded to allow 16bytes loads ?
  zmm1_Ref4567 = _mm512_permutex2var_epi32(zmm1_Ref4567, imm_shift_insert_rows8, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(pucRef + nRefPitch[0] * 8))));

  zmm4_sad_8 = _mm512_adds_epu16(_mm512_dbsad_epu8(zmm2_Src0123, zmm0_Ref0123, 148), _mm512_dbsad_epu8(zmm3_Src4567, zmm1_Ref4567, 148));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm4_sad_8));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm4_sad_8));
  ymm_sad_ress = _mm512_castsi512_si256(zmm4_sad_8);
  ymm_sads_R1 = _mm256_adds_epi16(ymm_sad_ress, _mm256_srli_si256(ymm_sad_ress, 8));

  // shift-push 1 row
  zmm0_Ref0123 = _mm512_permutex2var_epi64(zmm0_Ref0123, imm_vshift_rows8, zmm1_Ref4567);
  // ref need to be padded to allow 16bytes loads ?
  zmm1_Ref4567 = _mm512_permutex2var_epi32(zmm1_Ref4567, imm_shift_insert_rows8, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(pucRef + nRefPitch[0] * 9))));
  zmm4_sad_8 = _mm512_adds_epu16(_mm512_dbsad_epu8(zmm2_Src0123, zmm0_Ref0123, 148), _mm512_dbsad_epu8(zmm3_Src4567, zmm1_Ref4567, 148));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), zmm4_sad_8));
  zmm4_sad_8 = _mm512_adds_epu16(zmm4_sad_8, _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 7, 7, 7, 7, 3, 2), zmm4_sad_8));
  ymm_sad_ress = _mm512_castsi512_si256(zmm4_sad_8);
  ymm_sads_R2 = _mm256_adds_epi16(ymm_sad_ress, _mm256_srli_si256(ymm_sad_ress, 8));

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

void PlaneOfBlocks::ExhaustiveSearch8x8_uint8_16Blks_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy, int* pBlkData)
{
  const uint8_t* pucRef = GetRefBlock(workarea, mvx - 1, mvy - 1); // upper left corner
  const uint8_t* pucCurr = workarea.pSrc[0];
  // 16 blocks AVX512

  __m512i zmm16_r1_b0007, zmm18_r2_b0007, zmm20_r3_b0007, zmm22_r4_b0007, zmm24_r5_b0007, zmm26_r6_b0007, zmm28_r7_b0007, zmm30_r8_b0007;
  __m512i	zmm17_r1_b0815, zmm19_r2_b0815, zmm21_r3_b0815, zmm23_r4_b0815, zmm25_r5_b0815, zmm27_r6_b0815, zmm29_r7_b0815, zmm31_r8_b0815;

/*  zmm0_Src_r1_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 0));
  zmm1_Src_r1_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 0 + 64));
  zmm2_Src_r2_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 1));
  zmm3_Src_r2_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 1 + 64));
  zmm4_Src_r3_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 2));
  zmm5_Src_r3_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 2 + 64));
  zmm6_Src_r4_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 3));
  zmm7_Src_r4_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 3 + 64));
  zmm8_Src_r5_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 4));
  zmm9_Src_r5_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 4 + 64));
  zmm10_Src_r6_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 5));
  zmm11_Src_r6_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 5 + 64));
  zmm12_Src_r7_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 6));
  zmm13_Src_r7_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 6 + 64));
  zmm14_Src_r8_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 7));
  zmm15_Src_r8_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 7 + 64));*/

  workarea.zmm0_Src_r1_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 0));
  workarea.zmm1_Src_r1_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 0 + 64));
  workarea.zmm2_Src_r2_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 1));
  workarea.zmm3_Src_r2_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 1 + 64));
  workarea.zmm4_Src_r3_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 2));
  workarea.zmm5_Src_r3_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 2 + 64));
  workarea.zmm6_Src_r4_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 3));
  workarea.zmm7_Src_r4_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 3 + 64));
  workarea.zmm8_Src_r5_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 4));
  workarea.zmm9_Src_r5_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 4 + 64));
  workarea.zmm10_Src_r6_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 5));
  workarea.zmm11_Src_r6_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 5 + 64));
  workarea.zmm12_Src_r7_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 6));
  workarea.zmm13_Src_r7_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 6 + 64));
  workarea.zmm14_Src_r8_b0007 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 7));
  workarea.zmm15_Src_r8_b0815 = _mm512_loadu_si512((__m512i*)(pucCurr + nSrcPitch[0] * 7 + 64));

  int x, y;
  __m512i part_sads1_0007, part_sads2_0007, part_sads1_0815, part_sads2_0815;
#ifdef _DEBUG
  part_sads1_0007 = _mm512_setzero_si512();
  part_sads2_0007 = _mm512_setzero_si512();
  part_sads1_0815 = _mm512_setzero_si512();
  part_sads2_0815 = _mm512_setzero_si512();
#endif

  // 1st 16sads
  y = 0; x = 0;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads1_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0007, zmm16_r1_b0007);
  part_sads1_0007 = _mm512_srli_epi64(part_sads1_0007, 16);

  part_sads1_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0815, zmm17_r1_b0815);
  part_sads1_0815 = _mm512_srli_epi64(part_sads1_0815, 16);

  // 2nd 16sads
  x = 1; y = 0;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads1_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0007, zmm16_r1_b0007);
  part_sads1_0007 = _mm512_srli_epi64(part_sads1_0007, 16);

  part_sads1_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0815, zmm17_r1_b0815);
  part_sads1_0815 = _mm512_srli_epi64(part_sads1_0815, 16);

  // 3rd 16sads
  x = 2; y = 0;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads1_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0007, zmm16_r1_b0007);
  part_sads1_0007 = _mm512_srli_epi64(part_sads1_0007, 16);

  part_sads1_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0815, zmm17_r1_b0815);
  part_sads1_0815 = _mm512_srli_epi64(part_sads1_0815, 16);

  // 4th 16sads
  x = 0; y = 1;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads1_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0007, zmm16_r1_b0007);
  part_sads1_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads1_0815, zmm17_r1_b0815); // 4 x 16 sads ready

// skip check zero position, add 1 to minpos if > 4 !

  // 5th 16sads
  x = 2; y = 1;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads2_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0007, zmm16_r1_b0007);
  part_sads2_0007 = _mm512_srli_epi64(part_sads1_0007, 16);

  part_sads2_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0815, zmm17_r1_b0815);
  part_sads2_0815 = _mm512_srli_epi64(part_sads2_0815, 16);


  // 6th 4sads
  x = 0; y = 2;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads2_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0007, zmm16_r1_b0007);
  part_sads2_0007 = _mm512_srli_epi64(part_sads1_0007, 16);

  part_sads2_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0815, zmm17_r1_b0815);
  part_sads2_0815 = _mm512_srli_epi64(part_sads2_0815, 16);


  // 7th 4sads
  x = 1; y = 2;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads2_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0007, zmm16_r1_b0007);
  part_sads2_0007 = _mm512_srli_epi64(part_sads1_0007, 16);

  part_sads2_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0815, zmm17_r1_b0815);
  part_sads2_0815 = _mm512_srli_epi64(part_sads2_0815, 16);


  // 8th 4sads
  x = 2; y = 2;
  SAD_16blocks8x8

    zmm16_r1_b0007 = _mm512_slli_epi64(zmm16_r1_b0007, 48);
  zmm17_r1_b0815 = _mm512_slli_epi64(zmm17_r1_b0815, 48);

  part_sads2_0007 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0007, zmm16_r1_b0007);
  part_sads2_0815 = _mm512_mask_blend_epi16(0x88888888, part_sads2_0815, zmm17_r1_b0815);


  // 8 SADs of 1 block
  __m128i xmm0_sad1 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 3, 32 + 2, 32 + 1, 32 + 0, 3, 2, 1, 0), part_sads2_0007));

  // 8 SADs of 2 block
  __m128i xmm1_sad2 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 7, 32 + 6, 32 + 5, 32 + 4, 7, 6, 5, 4), part_sads2_0007));

  // 8 SADs of 3 block
  __m128i xmm2_sad3 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 11, 32 + 10, 32 + 9, 32 + 8, 11, 10, 9, 8), part_sads2_0007));

  // 8 SADs of 4 block
  __m128i xmm3_sad4 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 15, 32 + 14, 32 + 13, 32 + 12, 15, 14, 13, 12), part_sads2_0007));

  // 8 SADs of 5 block
  __m128i xmm4_sad5 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 19, 32 + 18, 32 + 17, 32 + 16, 19, 18, 17, 16), part_sads2_0007));

  // 8 SADs of 6 block
  __m128i xmm5_sad6 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 23, 32 + 22, 32 + 21, 32 + 20, 23, 22, 21, 20), part_sads2_0007));

  // 8 SADs of 7 block
  __m128i xmm6_sad7 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 27, 32 + 26, 32 + 25, 32 + 24, 27, 26, 25, 24), part_sads2_0007));

  // 8 SADs of 8 block
  __m128i xmm7_sad8 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0007, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 31, 32 + 30, 32 + 29, 32 + 28, 31, 30, 29, 28), part_sads2_0007));

  // 8 SADs of 9 block
  __m128i xmm8_sad9 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 3, 32 + 2, 32 + 1, 32 + 0, 3, 2, 1, 0), part_sads2_0815));

  // 8 SADs of 10 block
  __m128i xmm9_sad10 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 7, 32 + 6, 32 + 5, 32 + 4, 7, 6, 5, 4), part_sads2_0815));

  // 8 SADs of 11 block
  __m128i xmm10_sad11 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 11, 32 + 10, 32 + 9, 32 + 8, 11, 10, 9, 8), part_sads2_0815));

  // 8 SADs of 12 block
  __m128i xmm11_sad12 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 15, 32 + 14, 32 + 13, 32 + 12, 15, 14, 13, 12), part_sads2_0815));

  // 8 SADs of 13 block
  __m128i xmm12_sad13 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 19, 32 + 18, 32 + 17, 32 + 16, 19, 18, 17, 16), part_sads2_0815));

  // 8 SADs of 14 block
  __m128i xmm13_sad14 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 23, 32 + 22, 32 + 21, 32 + 20, 23, 22, 21, 20), part_sads2_0815));

  // 8 SADs of 15 block
  __m128i xmm14_sad15 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 27, 32 + 26, 32 + 25, 32 + 24, 27, 26, 25, 24), part_sads2_0815));

  // 8 SADs of 16 block
  __m128i xmm15_sad16 = _mm512_castsi512_si128(_mm512_permutex2var_epi16(part_sads1_0815, _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    32 + 31, 32 + 30, 32 + 29, 32 + 28, 31, 30, 29, 28), part_sads2_0815));

  xmm0_sad1 = _mm_minpos_epu16(xmm0_sad1);
  xmm1_sad2 = _mm_minpos_epu16(xmm1_sad2);
  xmm2_sad3 = _mm_minpos_epu16(xmm2_sad3);
  xmm3_sad4 = _mm_minpos_epu16(xmm3_sad4);
  xmm4_sad5 = _mm_minpos_epu16(xmm4_sad5);
  xmm5_sad6 = _mm_minpos_epu16(xmm5_sad6);
  xmm6_sad7 = _mm_minpos_epu16(xmm6_sad7);
  xmm7_sad8 = _mm_minpos_epu16(xmm7_sad8);
  xmm8_sad9 = _mm_minpos_epu16(xmm8_sad9);
  xmm9_sad10 = _mm_minpos_epu16(xmm9_sad10);
  xmm10_sad11 = _mm_minpos_epu16(xmm10_sad11);
  xmm11_sad12 = _mm_minpos_epu16(xmm11_sad12);
  xmm12_sad13 = _mm_minpos_epu16(xmm12_sad13);
  xmm13_sad14 = _mm_minpos_epu16(xmm13_sad14);
  xmm14_sad15 = _mm_minpos_epu16(xmm14_sad15);
  xmm15_sad16 = _mm_minpos_epu16(xmm15_sad16);

  // add +1 to minpos if > 3
  __m128i xmm_uint3 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 3, 0);
  __m128i xmm_uint1 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 1, 0);

  xmm0_sad1 = _mm_blendv_epi8(xmm0_sad1, _mm_add_epi16(xmm0_sad1, xmm_uint1), _mm_cmpgt_epi16(xmm0_sad1, xmm_uint3));
  xmm1_sad2 = _mm_blendv_epi8(xmm1_sad2, _mm_add_epi16(xmm1_sad2, xmm_uint1), _mm_cmpgt_epi16(xmm1_sad2, xmm_uint3));
  xmm2_sad3 = _mm_blendv_epi8(xmm2_sad3, _mm_add_epi16(xmm2_sad3, xmm_uint1), _mm_cmpgt_epi16(xmm2_sad3, xmm_uint3));
  xmm3_sad4 = _mm_blendv_epi8(xmm3_sad4, _mm_add_epi16(xmm3_sad4, xmm_uint1), _mm_cmpgt_epi16(xmm3_sad4, xmm_uint3));
  xmm4_sad5 = _mm_blendv_epi8(xmm4_sad5, _mm_add_epi16(xmm4_sad5, xmm_uint1), _mm_cmpgt_epi16(xmm4_sad5, xmm_uint3));
  xmm5_sad6 = _mm_blendv_epi8(xmm5_sad6, _mm_add_epi16(xmm5_sad6, xmm_uint1), _mm_cmpgt_epi16(xmm5_sad6, xmm_uint3));
  xmm6_sad7 = _mm_blendv_epi8(xmm6_sad7, _mm_add_epi16(xmm6_sad7, xmm_uint1), _mm_cmpgt_epi16(xmm6_sad7, xmm_uint3));
  xmm7_sad8 = _mm_blendv_epi8(xmm7_sad8, _mm_add_epi16(xmm7_sad8, xmm_uint1), _mm_cmpgt_epi16(xmm7_sad8, xmm_uint3));
  xmm8_sad9 = _mm_blendv_epi8(xmm8_sad9, _mm_add_epi16(xmm8_sad9, xmm_uint1), _mm_cmpgt_epi16(xmm8_sad9, xmm_uint3));
  xmm9_sad10 = _mm_blendv_epi8(xmm9_sad10, _mm_add_epi16(xmm9_sad10, xmm_uint1), _mm_cmpgt_epi16(xmm9_sad10, xmm_uint3));
  xmm10_sad11 = _mm_blendv_epi8(xmm10_sad11, _mm_add_epi16(xmm10_sad11, xmm_uint1), _mm_cmpgt_epi16(xmm10_sad11, xmm_uint3));
  xmm11_sad12 = _mm_blendv_epi8(xmm11_sad12, _mm_add_epi16(xmm11_sad12, xmm_uint1), _mm_cmpgt_epi16(xmm11_sad12, xmm_uint3));
  xmm12_sad13 = _mm_blendv_epi8(xmm12_sad13, _mm_add_epi16(xmm12_sad13, xmm_uint1), _mm_cmpgt_epi16(xmm12_sad13, xmm_uint3));
  xmm13_sad14 = _mm_blendv_epi8(xmm13_sad14, _mm_add_epi16(xmm13_sad14, xmm_uint1), _mm_cmpgt_epi16(xmm13_sad14, xmm_uint3));
  xmm14_sad15 = _mm_blendv_epi8(xmm14_sad15, _mm_add_epi16(xmm14_sad15, xmm_uint1), _mm_cmpgt_epi16(xmm14_sad15, xmm_uint3));
  xmm15_sad16 = _mm_blendv_epi8(xmm15_sad16, _mm_add_epi16(xmm15_sad16, xmm_uint1), _mm_cmpgt_epi16(xmm15_sad16, xmm_uint3));

  unsigned short minsad1 = (unsigned short)_mm_cvtsi128_si32(xmm0_sad1);
  unsigned short minsad2 = (unsigned short)_mm_cvtsi128_si32(xmm1_sad2);
  unsigned short minsad3 = (unsigned short)_mm_cvtsi128_si32(xmm2_sad3);
  unsigned short minsad4 = (unsigned short)_mm_cvtsi128_si32(xmm3_sad4);
  unsigned short minsad5 = (unsigned short)_mm_cvtsi128_si32(xmm4_sad5);
  unsigned short minsad6 = (unsigned short)_mm_cvtsi128_si32(xmm5_sad6);
  unsigned short minsad7 = (unsigned short)_mm_cvtsi128_si32(xmm6_sad7);
  unsigned short minsad8 = (unsigned short)_mm_cvtsi128_si32(xmm7_sad8);
  unsigned short minsad9 = (unsigned short)_mm_cvtsi128_si32(xmm8_sad9);
  unsigned short minsad10 = (unsigned short)_mm_cvtsi128_si32(xmm9_sad10);
  unsigned short minsad11 = (unsigned short)_mm_cvtsi128_si32(xmm10_sad11);
  unsigned short minsad12 = (unsigned short)_mm_cvtsi128_si32(xmm11_sad12);
  unsigned short minsad13 = (unsigned short)_mm_cvtsi128_si32(xmm12_sad13);
  unsigned short minsad14 = (unsigned short)_mm_cvtsi128_si32(xmm13_sad14);
  unsigned short minsad15 = (unsigned short)_mm_cvtsi128_si32(xmm14_sad15);
  unsigned short minsad16 = (unsigned short)_mm_cvtsi128_si32(xmm15_sad16);

  __m128i xmm_idx0003 = xmm0_sad1;
  xmm_idx0003 = _mm_blend_epi16(xmm_idx0003, _mm_slli_si128(xmm1_sad2, 4), 8);
  xmm_idx0003 = _mm_blend_epi16(xmm_idx0003, _mm_slli_si128(xmm2_sad3, 8), 32);
  xmm_idx0003 = _mm_blend_epi16(xmm_idx0003, _mm_slli_si128(xmm3_sad4, 12), 128);

  xmm_idx0003 = _mm_blend_epi16(xmm_idx0003, _mm_srli_si128(xmm_idx0003, 2), 85);

  __m128i xmm_idx0407 = xmm4_sad5;
  xmm_idx0407 = _mm_blend_epi16(xmm_idx0407, _mm_slli_si128(xmm5_sad6, 4), 8);
  xmm_idx0407 = _mm_blend_epi16(xmm_idx0407, _mm_slli_si128(xmm6_sad7, 8), 32);
  xmm_idx0407 = _mm_blend_epi16(xmm_idx0407, _mm_slli_si128(xmm7_sad8, 12), 128);

  xmm_idx0407 = _mm_blend_epi16(xmm_idx0407, _mm_srli_si128(xmm_idx0407, 2), 85);

  __m128i xmm_idx0811 = xmm8_sad9;
  xmm_idx0811 = _mm_blend_epi16(xmm_idx0811, _mm_slli_si128(xmm9_sad10, 4), 8);
  xmm_idx0811 = _mm_blend_epi16(xmm_idx0811, _mm_slli_si128(xmm10_sad11, 8), 32);
  xmm_idx0811 = _mm_blend_epi16(xmm_idx0811, _mm_slli_si128(xmm11_sad12, 12), 128);

  xmm_idx0811 = _mm_blend_epi16(xmm_idx0811, _mm_srli_si128(xmm_idx0811, 2), 85);

  __m128i xmm_idx1215 = xmm12_sad13;
  xmm_idx1215 = _mm_blend_epi16(xmm_idx1215, _mm_slli_si128(xmm13_sad14, 4), 8);
  xmm_idx1215 = _mm_blend_epi16(xmm_idx1215, _mm_slli_si128(xmm14_sad15, 8), 32);
  xmm_idx1215 = _mm_blend_epi16(xmm_idx1215, _mm_slli_si128(xmm15_sad16, 12), 128);

  xmm_idx1215 = _mm_blend_epi16(xmm_idx0811, _mm_srli_si128(xmm_idx0811, 2), 85);

  //	int dx_minsad = (idx_min_sad % 3) - 1; - just comment where from x,y minsad come from
  //	int dy_minsad = (idx_min_sad / 3) - 1;

  // dx = idx*11>>5 , dy = (((idx*11)&31)*3)>>5
  __m128i xmm_uint11 = _mm_set_epi16(11, 11, 11, 11, 11, 11, 11, 11);
  __m128i xmm_and = _mm_set_epi16(-1, 31, -1, 31, -1, 31, -1, 31);
  __m128i xmm_1_3 = _mm_set_epi16(1, 3, 1, 3, 1, 3, 1, 3);
  __m128i xmm_1 = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);

  xmm_idx0003 = _mm_mullo_epi16(xmm_idx0003, xmm_uint11);
  xmm_idx0003 = _mm_and_si128(xmm_idx0003, xmm_and);
  xmm_idx0003 = _mm_mullo_epi16(xmm_idx0003, xmm_1_3);
  xmm_idx0003 = _mm_srli_epi16(xmm_idx0003, 5);
  xmm_idx0003 = _mm_sub_epi16(xmm_idx0003, xmm_1);

  xmm_idx0407 = _mm_mullo_epi16(xmm_idx0407, xmm_uint11);
  xmm_idx0407 = _mm_and_si128(xmm_idx0407, xmm_and);
  xmm_idx0407 = _mm_mullo_epi16(xmm_idx0407, xmm_1_3);
  xmm_idx0407 = _mm_srli_epi16(xmm_idx0407, 5);
  xmm_idx0407 = _mm_sub_epi16(xmm_idx0407, xmm_1);

  xmm_idx0811 = _mm_mullo_epi16(xmm_idx0811, xmm_uint11);
  xmm_idx0811 = _mm_and_si128(xmm_idx0811, xmm_and);
  xmm_idx0811 = _mm_mullo_epi16(xmm_idx0811, xmm_1_3);
  xmm_idx0811 = _mm_srli_epi16(xmm_idx0811, 5);
  xmm_idx0811 = _mm_sub_epi16(xmm_idx0811, xmm_1);

  xmm_idx1215 = _mm_mullo_epi16(xmm_idx1215, xmm_uint11);
  xmm_idx1215 = _mm_and_si128(xmm_idx1215, xmm_and);
  xmm_idx1215 = _mm_mullo_epi16(xmm_idx1215, xmm_1_3);
  xmm_idx1215 = _mm_srli_epi16(xmm_idx1215, 5);
  xmm_idx1215 = _mm_sub_epi16(xmm_idx1215, xmm_1);
  /*
  int dx_minsad1 = _mm_extract_epi16(xmm_idx0003, 0);
  int dy_minsad1 = _mm_extract_epi16(xmm_idx0003, 1);

  int dx_minsad2 = _mm_extract_epi16(xmm_idx0003, 2);
  int dy_minsad2 = _mm_extract_epi16(xmm_idx0003, 3);

  int dx_minsad3 = _mm_extract_epi16(xmm_idx0003, 4);
  int dy_minsad3 = _mm_extract_epi16(xmm_idx0003, 5);

  int dx_minsad4 = _mm_extract_epi16(xmm_idx0003, 6);
  int dy_minsad4 = _mm_extract_epi16(xmm_idx0003, 7);

  int dx_minsad5 = _mm_extract_epi16(xmm_idx0407, 0);
  int dy_minsad5 = _mm_extract_epi16(xmm_idx0407, 1);

  int dx_minsad6 = _mm_extract_epi16(xmm_idx0407, 2);
  int dy_minsad6 = _mm_extract_epi16(xmm_idx0407, 3);

  int dx_minsad7 = _mm_extract_epi16(xmm_idx0407, 4);
  int dy_minsad7 = _mm_extract_epi16(xmm_idx0407, 5);

  int dx_minsad8 = _mm_extract_epi16(xmm_idx0407, 6);
  int dy_minsad8 = _mm_extract_epi16(xmm_idx0407, 7);

  int dx_minsad9 = _mm_extract_epi16(xmm_idx0811, 0);
  int dy_minsad9 = _mm_extract_epi16(xmm_idx0811, 1);

  int dx_minsad10 = _mm_extract_epi16(xmm_idx0811, 2);
  int dy_minsad10 = _mm_extract_epi16(xmm_idx0811, 3);

  int dx_minsad11 = _mm_extract_epi16(xmm_idx0811, 4);
  int dy_minsad11 = _mm_extract_epi16(xmm_idx0811, 5);

  int dx_minsad12 = _mm_extract_epi16(xmm_idx0811, 6);
  int dy_minsad12 = _mm_extract_epi16(xmm_idx0811, 7);

  int dx_minsad13 = _mm_extract_epi16(xmm_idx1215, 0);
  int dy_minsad13 = _mm_extract_epi16(xmm_idx1215, 1);

  int dx_minsad14 = _mm_extract_epi16(xmm_idx1215, 2);
  int dy_minsad14 = _mm_extract_epi16(xmm_idx1215, 3);

  int dx_minsad15 = _mm_extract_epi16(xmm_idx1215, 4);
  int dy_minsad15 = _mm_extract_epi16(xmm_idx1215, 5);

  int dx_minsad16 = _mm_extract_epi16(xmm_idx1215, 6);
  int dy_minsad16 = _mm_extract_epi16(xmm_idx1215, 7);*/

  if (minsad1 < workarea.bestMV.sad)
  {
    pBlkData[workarea.blkx * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0003, 0); // dx_minsad1
    pBlkData[workarea.blkx * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0003, 1); // dy_minsad1
    pBlkData[workarea.blkx * N_PER_BLOCK + 2] = (uint32_t)(minsad1);

    workarea.bestMV.x += _mm_extract_epi16(xmm_idx0003, 0); // is it need at all ??
    workarea.bestMV.y += _mm_extract_epi16(xmm_idx0003, 1);
    workarea.nMinCost = minsad1 + ((penaltyNew * minsad1) >> 8);
    workarea.bestMV.sad = minsad1;
  }

  if (minsad2 < pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0003, 2); // dx_minsad2
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0003, 3); // dy_minsad2;
    pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = (uint32_t)(minsad2);
  }

  if (minsad3 < pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0003, 4); // dx_minsad3;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0003, 5); // dy_minsad3;
    pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = (uint32_t)(minsad3);
  }

  if (minsad4 < pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0003, 6); // dx_minsad4;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0003, 7); // dy_minsad4;
    pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = (uint32_t)(minsad4);
  }

  if (minsad5 < pBlkData[(workarea.blkx + 4) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 4) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0407, 0); // dx_minsad5;
    pBlkData[(workarea.blkx + 4) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0407, 1); // dy_minsad5;
    pBlkData[(workarea.blkx + 4) * N_PER_BLOCK + 2] = (uint32_t)(minsad5);
  }

  if (minsad6 < pBlkData[(workarea.blkx + 5) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 5) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0407, 2); // dx_minsad6;
    pBlkData[(workarea.blkx + 5) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0407, 3); // dy_minsad6;
    pBlkData[(workarea.blkx + 5) * N_PER_BLOCK + 2] = (uint32_t)(minsad6);
  }

  if (minsad7 < pBlkData[(workarea.blkx + 6) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 6) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0407, 4); // dx_minsad7;
    pBlkData[(workarea.blkx + 6) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0407, 5); // dy_minsad7;
    pBlkData[(workarea.blkx + 6) * N_PER_BLOCK + 2] = (uint32_t)(minsad7);
  }

  if (minsad8 < pBlkData[(workarea.blkx + 7) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 7) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0407, 6); // dx_minsad8;
    pBlkData[(workarea.blkx + 7) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0407, 7); // dy_minsad8;
    pBlkData[(workarea.blkx + 7) * N_PER_BLOCK + 2] = (uint32_t)(minsad8);
  }

  if (minsad9 < pBlkData[(workarea.blkx + 8) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 8) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0811, 0); // dx_minsad9;
    pBlkData[(workarea.blkx + 8) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0811, 1); // dy_minsad9;
    pBlkData[(workarea.blkx + 8) * N_PER_BLOCK + 2] = (uint32_t)(minsad9);
  }

  if (minsad10 < pBlkData[(workarea.blkx + 9) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 9) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0811, 2); // dx_minsad10;
    pBlkData[(workarea.blkx + 9) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0811, 3); // dy_minsad10;
    pBlkData[(workarea.blkx + 9) * N_PER_BLOCK + 2] = (uint32_t)(minsad10);
  }

  if (minsad11 < pBlkData[(workarea.blkx + 10) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 10) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0811, 4); // dx_minsad11;
    pBlkData[(workarea.blkx + 10) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0811, 5); // dy_minsad11;
    pBlkData[(workarea.blkx + 10) * N_PER_BLOCK + 2] = (uint32_t)(minsad11);
  }

  if (minsad12 < pBlkData[(workarea.blkx + 11) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 11) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx0811, 6); // dx_minsad12;
    pBlkData[(workarea.blkx + 11) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx0811, 7); // dy_minsad12;
    pBlkData[(workarea.blkx + 11) * N_PER_BLOCK + 2] = (uint32_t)(minsad11);
  }

  if (minsad13 < pBlkData[(workarea.blkx + 12) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 12) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx1215, 0); // dx_minsad13;
    pBlkData[(workarea.blkx + 12) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx1215, 1); // dy_minsad13;
    pBlkData[(workarea.blkx + 12) * N_PER_BLOCK + 2] = (uint32_t)(minsad11);
  }

  if (minsad14 < pBlkData[(workarea.blkx + 13) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 13) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx1215, 2); // dx_minsad14;
    pBlkData[(workarea.blkx + 13) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx1215, 3); // dy_minsad14;
    pBlkData[(workarea.blkx + 13) * N_PER_BLOCK + 2] = (uint32_t)(minsad11);
  }

  if (minsad15 < pBlkData[(workarea.blkx + 14) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 14) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx1215, 4); // dx_minsad15;
    pBlkData[(workarea.blkx + 14) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx1215, 5); // dy_minsad15;
    pBlkData[(workarea.blkx + 14) * N_PER_BLOCK + 2] = (uint32_t)(minsad11);
  }

  if (minsad16 < pBlkData[(workarea.blkx + 15) * N_PER_BLOCK + 2])
  {
    pBlkData[(workarea.blkx + 15) * N_PER_BLOCK + 0] += _mm_extract_epi16(xmm_idx1215, 6); // dx_minsad16;
    pBlkData[(workarea.blkx + 15) * N_PER_BLOCK + 1] += _mm_extract_epi16(xmm_idx1215, 7); // dy_minsad16;
    pBlkData[(workarea.blkx + 15) * N_PER_BLOCK + 2] = (uint32_t)(minsad11);
  }

}
