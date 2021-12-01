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
    _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu2_m128i((__m128i*)(pucCurr + nSrcPitch[0] * r2), (__m128i*)(pucCurr + nSrcPitch[0] * r1))), \
      _mm256_loadu2_m128i((__m128i*)(pucCurr + nSrcPitch[0] * r4), (__m128i*)(pucCurr + nSrcPitch[0] * r3)), 1)

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
