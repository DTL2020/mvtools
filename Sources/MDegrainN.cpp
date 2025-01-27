

#include  "ClipFnc.h"
#include  "CopyCode.h"
#include	"def.h"
#include	"MDegrainN.h"
#include  "MVDegrain3.h"
#include  "MVFrame.h"
#include  "MVPlane.h"
#include  "MVFilter.h"
#include  "profile.h"
#include  "SuperParams64Bits.h"
#include  "SADFunctions.h"

#include	<emmintrin.h>
#include	<mmintrin.h>

#include	<cassert>
#include	<cmath>
#include  <map>
#include  <tuple>
#include  <stdint.h>
#include  "commonfunctions.h"

/*
MV_FORCEINLINE float DiamondAngle(int y, int x)
{
  if ((x + y) == 0 || (y - x) == 0 || (-y - x) == 0 || (x - y) == 0)
    return 0;

  float fy = (float)y;
  float fx = (float)x;

  if (y >= 0)
    return (x >= 0 ? fy / (fx + fy) : 1 - fx / (-fx + fy));
  else
    return (x < 0 ? 2 - fy / (-fx - fy) : 3 + fx / (fx - fy));

}
*/

MV_FORCEINLINE float fDiffAngleVect(int x1, int y1, int x2, int y2)
{
  float fResult = 0.0f;
  // check if any of 2 input vectors is zero vector - return 0
  if ((x1 == 0) && (y1 == 0) || (x2 == 0) && (y2 == 0))
    return 0.0f;

  int iUpper = x1 * x2 + y1 * y2;

  if (iUpper > 0)
  {
    fResult = 1.0f - ((float)(iUpper * iUpper) / (float((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2))));
  }
  else
  {
    fResult = 1.0f + ((float)(iUpper * iUpper) / (float((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2))));
  }

  return fResult;

}

// out16_type: 
//   0: native 8 or 16
//   1: 8bit in, lsb
//   2: 8bit in, native16 out
template <typename pixel_t, int blockWidth, int blockHeight, int out16_type>
void DegrainN_C(
  BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  const BYTE* pRef[], int Pitch[],
  int Wall[], int trad
)
{
  // for less template, see solution in Degrain1to6_C

  constexpr bool lsb_flag = (out16_type == 1);
  constexpr bool out16 = (out16_type == 2);

  if constexpr (lsb_flag || out16)
  {
    // 8 bit base only
    for (int h = 0; h < blockHeight; ++h)
    {
      for (int x = 0; x < blockWidth; ++x)
      {
        int val = pSrc[x] * Wall[0];
        for (int k = 0; k < trad; ++k)
        {
          val += pRef[k * 2][x] * (short)Wall[k * 2 + 1]
            + pRef[k * 2 + 1][x] * (short)Wall[k * 2 + 2]; // to be compatible with 2x16bit 32bit weight
        }
        if constexpr (lsb_flag) {
          pDst[x] = (uint8_t)(val >> 8);
          pDstLsb[x] = (uint8_t)(val & 255);
        }
        else { // out16
          reinterpret_cast<uint16_t*>(pDst)[x] = (uint16_t)val;
        }
      }

      pDst += nDstPitch;
      if constexpr (lsb_flag)
        pDstLsb += nDstPitch;
      pSrc += nSrcPitch;
      for (int k = 0; k < trad; ++k)
      {
        pRef[k * 2] += Pitch[k * 2];
        pRef[k * 2 + 1] += Pitch[k * 2 + 1];
      }
    }
  }

  else
  {
    typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float>::type target_t;
    constexpr target_t rounder = (sizeof(pixel_t) <= 2) ? 128 : 0;
    constexpr float scaleback = 1.0f / (1 << DEGRAIN_WEIGHT_BITS);

    // Wall: 8 bit. rounding: 128
    for (int h = 0; h < blockHeight; ++h)
    {
      for (int x = 0; x < blockWidth; ++x)
      {
        target_t val = reinterpret_cast<const pixel_t*>(pSrc)[x] * (target_t)Wall[0] + rounder;
        for (int k = 0; k < trad; ++k)
        {
          val += reinterpret_cast<const pixel_t*>(pRef[k * 2])[x] * (target_t)Wall[k * 2 + 1]
            + reinterpret_cast<const pixel_t*>(pRef[k * 2 + 1])[x] * (target_t)Wall[k * 2 + 2]; // do it compatible with 2x16bit weight ?
        }
        if constexpr (sizeof(pixel_t) <= 2)
          reinterpret_cast<pixel_t*>(pDst)[x] = (pixel_t)(val >> 8); // 8-16bit
        else
          reinterpret_cast<pixel_t*>(pDst)[x] = val * scaleback; // 32bit float
      }

      pDst += nDstPitch;
      pSrc += nSrcPitch;
      for (int k = 0; k < trad; ++k)
      {
        pRef[k * 2] += Pitch[k * 2];
        pRef[k * 2 + 1] += Pitch[k * 2 + 1];
      }
    }
  }
}

#if 0
template <typename pixel_t, int blockWidth, int blockHeight>
void SubtractBlockN_C(
  BYTE* pDst, /*BYTE* pDstLsb,*/ int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  const BYTE* pRef[], int Pitch[],
  int Wall[], int trad, int iN
)
{
    typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float>::type target_t;
    constexpr target_t rounder = (sizeof(pixel_t) <= 2) ? 128 : 0;
    constexpr float scaleback = 1.0f / (1 << DEGRAIN_WEIGHT_BITS);

    // subtract block N = (Src - BlockN*wN) * (1/(1-wN))
    target_t wN = Wall[iN]; // weight of block N
    target_t mul = (target_t)(1.0f / ((1 << DEGRAIN_WEIGHT_BITS) - (target_t)wN));

  // Wall: 8 bit. rounding: 128
  for (int h = 0; h < blockHeight; ++h)
  {
    for (int x = 0; x < blockWidth; ++x)
    {
//      target_t val = reinterpret_cast<const pixel_t*>(pSrc)[x] * (target_t)Wall[0] + rounder;
      target_t val = reinterpret_cast<const pixel_t*>(pSrc)[x] + rounder;
/*      for (int k = 0; k < trad; ++k)
      {
        val += reinterpret_cast<const pixel_t*>(pRef[k * 2])[x] * (target_t)Wall[k * 2 + 1]
          + reinterpret_cast<const pixel_t*>(pRef[k * 2 + 1])[x] * (target_t)Wall[k * 2 + 2]; // do it compatible with 2x16bit weight ?
      }*/
      if constexpr (sizeof(pixel_t) <= 2)
      {
        val = ((val << 8) - (reinterpret_cast<const pixel_t*>(pRef[iN])[x] * (target_t)Wall[iN + 1]));
        val = val >> 8;
        val *= mul;
      }
      else
      {
        val = (val - (reinterpret_cast<const pixel_t*>(pRef[iN])[x] * (target_t)Wall[iN + 1]));
        val = val * scaleback;
        val *= mul;
      }

      if constexpr (sizeof(pixel_t) <= 2)
        reinterpret_cast<pixel_t*>(pDst)[x] = (pixel_t)(val >> 8); // 8-16bit
      else
        reinterpret_cast<pixel_t*>(pDst)[x] = val * scaleback; // 32bit float
    }

    pDst += nDstPitch;
    pSrc += nSrcPitch;
/*    for (int k = 0; k < trad; ++k)
    {
      pRef[k * 2] += Pitch[k * 2];
      pRef[k * 2 + 1] += Pitch[k * 2 + 1];
    }*/
    pRef[iN] += Pitch[iN];
  }

}
#endif

void SubtractBlock_C_uint8(
  BYTE* pDst, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  uint8_t* pRefBlock, const int PitchRef,
  int wN, int blockWidth, int blockHeight
)
{
  /*  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float>::type target_t;
    constexpr target_t rounder = (sizeof(pixel_t) <= 2) ? 128 : 0;*/
  constexpr float scaleback = 1.0f / (1 << DEGRAIN_WEIGHT_BITS);

  int one = (1 << DEGRAIN_WEIGHT_BITS);

  // subtract block N = (Src - BlockN*wN) * (1/(1-wN))
  int mul = (int)(((float)one * ((float)(one) / (float)(one - wN))) + 0.5f);

  float  fmul = 256.0f / (float)(256 - wN);
  int rounder = 128;

  // Wall: 8 bit. 
  for (int h = 0; h < blockHeight; ++h)
  {
    for (int x = 0; x < blockWidth; ++x)
    {
      int val = reinterpret_cast<const uint8_t*>(pSrc)[x];

      val = ((val << 8) - (reinterpret_cast<const uint8_t*>(pRefBlock)[x] * (uint8_t)wN));
      val += rounder;
      val = val >> 8;
      val *= mul;
      val += rounder;

      reinterpret_cast<uint8_t*>(pDst)[x] = (uint8_t)(val >> 8); // 8bit
#ifdef _DEBUG
      int src = reinterpret_cast<const uint8_t*>(pSrc)[x];
      src = src << 8; // *256
      src = (src - (reinterpret_cast<const uint8_t*>(pRefBlock)[x] * (uint8_t)wN));

      float fres = ((float)src * fmul);
      uint8_t res8 = ((fres / 256.0f) + 0.5f);
      reinterpret_cast<uint8_t*>(pDst)[x] = res8;

      if (abs(res8 - pDst[x]) > 0)
      {
        int idbr = 0;
      }
#endif
    }

    pDst += nDstPitch;
    pSrc += nSrcPitch;
    pRefBlock += PitchRef;
  }

}

void SubtractBlock_uint8_sse2(
  BYTE* pDst, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  uint8_t* pRefBlock, const int PitchRef,
  int wN, int blockWidth, int blockHeight
)
{
  assert(blockWidth % 4 == 0);
  // only mod4 supported

  /*  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float>::type target_t;
    constexpr target_t rounder = (sizeof(pixel_t) <= 2) ? 128 : 0;*/
  constexpr float scaleback = 1.0f / (1 << DEGRAIN_WEIGHT_BITS);

  int one = (1 << DEGRAIN_WEIGHT_BITS);

  // subtract block N = (Src - BlockN*wN) * (1/(1-wN))
  int mul = (int)(((float)one * ((float)(one) / (float)(one - wN))) + 0.5f);
  int rounder_i = 128;

  const __m128i rounder = _mm_set1_epi16(rounder_i);

  const __m128i z = _mm_setzero_si128();

  bool is_mod8 = blockWidth % 8 == 0; // constexpr later
  int pixels_at_a_time = is_mod8 ? 8 : 4; // 4 for 4 and 12; 8 for all others 8, 16, 24, 32...

  // Wall: 8 bit. 
  for (int h = 0; h < blockHeight; ++h)
  {
    for (int x = 0; x < blockWidth; x += pixels_at_a_time)
    {
      /*      int val = reinterpret_cast<const uint8_t*>(pSrc)[x];

            val = ((val << 8) - (reinterpret_cast<const uint8_t*>(pRefBlock)[x] * (uint8_t)wN));
            val += rounder;
            val = val >> 8;
            val *= mul;
            val += rounder;

            reinterpret_cast<uint8_t*>(pDst)[x] = (uint8_t)(val >> 8); // 8bit*/
      __m128i src;
      if (is_mod8) // load 8 pixels
        src = _mm_loadl_epi64((__m128i*) (pSrc + x));
      else // load 4 pixels
        src = _mm_cvtsi32_si128(*(uint32_t*)(pSrc + x));

      __m128i val = _mm_unpacklo_epi8(src, z);
      val = _mm_slli_epi16(val, 8); // val << 8

      __m128i ref;
      if (is_mod8) // load 8 pixels
      {
        ref = _mm_loadl_epi64((__m128i*) (pRefBlock + x));
      }
      else { // 4 pixels
        ref = _mm_cvtsi32_si128(*(uint32_t*)(pRefBlock + x));
      }

      __m128i mul1 = _mm_mullo_epi16(_mm_unpacklo_epi8(ref, z), _mm_set1_epi16(wN));

      val = _mm_sub_epi16(val, mul1);
      val = _mm_adds_epi16(val, rounder);
      val = _mm_srli_epi16(val, 8);
      val = _mm_mullo_epi16(val, _mm_set1_epi16(mul));
      val = _mm_adds_epi16(val, rounder);

      auto res = _mm_packus_epi16(_mm_srli_epi16(val, 8), z);

      if (is_mod8) {
        _mm_storel_epi64((__m128i*)(pDst + x), res);
      }
      else {
        *(uint32_t*)(pDst + x) = _mm_cvtsi128_si32(res);
      }

    }

    pDst += nDstPitch;
    pSrc += nSrcPitch;
    pRefBlock += PitchRef;
  }

}


// Debug note: DegrainN filter is calling Degrain1-6 instead if ThSAD(C) == ThSAD(C)2.
// To reach DegrainN_ functions, set the above parameters to different values

// out16_type: 
//   0: native 8 or 16
//   1: 8bit in, lsb
//   2: 8bit in, native16 out
template <int blockWidth, int blockHeight, int out16_type>
void DegrainN_sse2(
  BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  const BYTE* pRef[], int Pitch[],
  int Wall[], int trad
)
{
  assert(blockWidth % 4 == 0);
  // only mod4 supported

  constexpr bool lsb_flag = (out16_type == 1);
  constexpr bool out16 = (out16_type == 2);

  const __m128i z = _mm_setzero_si128();

  constexpr bool is_mod8 = blockWidth % 8 == 0;
  constexpr int pixels_at_a_time = is_mod8 ? 8 : 4; // 4 for 4 and 12; 8 for all others 8, 16, 24, 32...

  if constexpr (lsb_flag || out16)
  {
    // no rounding
    const __m128i m = _mm_set1_epi16(255);

    for (int h = 0; h < blockHeight; ++h)
    {
      for (int x = 0; x < blockWidth; x += pixels_at_a_time)
      {
        __m128i src;
        if (is_mod8) // load 8 pixels
          src = _mm_loadl_epi64((__m128i*) (pSrc + x));
        else // load 4 pixels
          src = _mm_cvtsi32_si128(*(uint32_t*)(pSrc + x));
        __m128i val = _mm_mullo_epi16(_mm_unpacklo_epi8(src, z), _mm_set1_epi16(Wall[0]));
        for (int k = 0; k < trad; ++k)
        {
          __m128i src1, src2;
          if constexpr (is_mod8) // load 8-8 pixels
          {
            src1 = _mm_loadl_epi64((__m128i*) (pRef[k * 2] + x));
            src2 = _mm_loadl_epi64((__m128i*) (pRef[k * 2 + 1] + x));
          }
          else { // 4-4 pixels
            src1 = _mm_cvtsi32_si128(*(uint32_t*)(pRef[k * 2] + x));
            src2 = _mm_cvtsi32_si128(*(uint32_t*)(pRef[k * 2 + 1] + x));
          }
          const __m128i	s1 = _mm_mullo_epi16(_mm_unpacklo_epi8(src1, z), _mm_set1_epi16(Wall[k * 2 + 1]));
          const __m128i	s2 = _mm_mullo_epi16(_mm_unpacklo_epi8(src2, z), _mm_set1_epi16(Wall[k * 2 + 2]));
          val = _mm_add_epi16(val, s1);
          val = _mm_add_epi16(val, s2);
        }
        if constexpr (is_mod8) {
          if constexpr (lsb_flag) {
            _mm_storel_epi64((__m128i*)(pDst + x), _mm_packus_epi16(_mm_srli_epi16(val, 8), z));
            _mm_storel_epi64((__m128i*)(pDstLsb + x), _mm_packus_epi16(_mm_and_si128(val, m), z));
          }
          else {
            _mm_storeu_si128((__m128i*)(pDst + x * 2), val);
          }
          }
        else {
          if constexpr (lsb_flag) {
            *(uint32_t*)(pDst + x) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_srli_epi16(val, 8), z));
            *(uint32_t*)(pDstLsb + x) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_and_si128(val, m), z));
          }
          else {
            _mm_storel_epi64((__m128i*)(pDst + x * 2), val);
          }
        }
        }
      pDst += nDstPitch;
      if constexpr (lsb_flag)
        pDstLsb += nDstPitch;
      pSrc += nSrcPitch;
      for (int k = 0; k < trad; ++k)
      {
        pRef[k * 2] += Pitch[k * 2];
        pRef[k * 2 + 1] += Pitch[k * 2 + 1];
      }
      }
    }

  else
  {
    // base 8 bit -> 8 bit
    const __m128i o = _mm_set1_epi16(128); // rounding

    for (int h = 0; h < blockHeight; ++h)
    {
      for (int x = 0; x < blockWidth; x += pixels_at_a_time)
      {
        __m128i src;
        if constexpr (is_mod8) // load 8 pixels
          src = _mm_loadl_epi64((__m128i*) (pSrc + x));
        else // load 4 pixels
          src = _mm_cvtsi32_si128(*(uint32_t*)(pSrc + x));

        __m128i val = _mm_add_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8(src, z), _mm_set1_epi16(Wall[0])), o);
        for (int k = 0; k < trad; ++k)
        {
          __m128i src1, src2;
          if constexpr (is_mod8) // load 8-8 pixels
          {
            src1 = _mm_loadl_epi64((__m128i*) (pRef[k * 2] + x));
            src2 = _mm_loadl_epi64((__m128i*) (pRef[k * 2 + 1] + x));
          }
          else { // 4-4 pixels
            src1 = _mm_cvtsi32_si128(*(uint32_t*)(pRef[k * 2] + x));
            src2 = _mm_cvtsi32_si128(*(uint32_t*)(pRef[k * 2 + 1] + x));
          }
          const __m128i s1 = _mm_mullo_epi16(_mm_unpacklo_epi8(src1, z), _mm_set1_epi16(Wall[k * 2 + 1]));
          const __m128i s2 = _mm_mullo_epi16(_mm_unpacklo_epi8(src2, z), _mm_set1_epi16(Wall[k * 2 + 2]));
          val = _mm_add_epi16(val, s1);
          val = _mm_add_epi16(val, s2);
        }
        auto res = _mm_packus_epi16(_mm_srli_epi16(val, 8), z);
        if constexpr (is_mod8) {
          _mm_storel_epi64((__m128i*)(pDst + x), res);
        }
        else {
          *(uint32_t*)(pDst + x) = _mm_cvtsi128_si32(res);
        }
      }

      pDst += nDstPitch;
      pSrc += nSrcPitch;
      for (int k = 0; k < trad; ++k)
      {
        pRef[k * 2] += Pitch[k * 2];
        pRef[k * 2 + 1] += Pitch[k * 2 + 1];
      }
    }
  }
}

template<int blockWidth, int blockHeight, bool lessThan16bits>
void DegrainN_16_sse41(
  BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  const BYTE* pRef[], int Pitch[],
  int Wall[], int trad
)
{
  assert(blockWidth % 4 == 0);
  // only mod4 supported

  // able to do madd for real 16 bit uint16_t data
  const auto signed16_shifter = _mm_set1_epi16(-32768);
  const auto signed16_shifter_si32 = _mm_set1_epi32(32768 << DEGRAIN_WEIGHT_BITS);

  const __m128i z = _mm_setzero_si128();
  constexpr int SHIFTBACK = DEGRAIN_WEIGHT_BITS;
  constexpr int rounder_i = (1 << SHIFTBACK) / 2;
  // note: DEGRAIN_WEIGHT_BITS is fixed 8 bits, so no rounding occurs on 8 bit in 16 bit out

  __m128i rounder = _mm_set1_epi32(rounder_i); // rounding: 128 (mul by 8 bit wref scale back)

  for (int h = 0; h < blockHeight; ++h)
  {
    for (int x = 0; x < blockWidth; x += 8 / sizeof(uint16_t)) // up to 4 pixels per cycle
    {
      // load 4 pixels
      auto src = _mm_loadl_epi64((__m128i*)(pSrc + x * sizeof(uint16_t)));

      // weights array structure: center, forward1, backward1, forward2, backward2, etc
      //                          Wall[0] Wall[1]   Wall[2]    Wall[3]   Wall[4] ...
      // inputs structure:        pSrc    pRef[0]   pRef[1]    pRef[2]   pRef[3] ...

      __m128i res;
      // make signed when unsigned 16 bit mode
      if constexpr (!lessThan16bits)
        src = _mm_add_epi16(src, signed16_shifter);

      // Interleave Src 0 Src 0 ...
      src = _mm_cvtepu16_epi32(src); // sse4 unpacklo_epi16 w/ zero

      // interleave 0 and center weight
      auto ws = _mm_set1_epi32((0 << 16) + Wall[0]);
      // pSrc[x] * WSrc + 0 * 0
      res = _mm_madd_epi16(src, ws);

      // pRefF[n][x] * WRefF[n] + pRefB[n][x] * WRefB[n]
      for (int k = 0; k < trad; ++k)
      {
        // Interleave SrcF SrcB
        src = _mm_unpacklo_epi16(
          _mm_loadl_epi64((__m128i*)(pRef[k * 2] + x * sizeof(uint16_t))), // from forward
          _mm_loadl_epi64((__m128i*)(pRef[k * 2 + 1] + x * sizeof(uint16_t)))); // from backward
        if constexpr (!lessThan16bits)
          src = _mm_add_epi16(src, signed16_shifter);

        // Interleave Forward and Backward 16 bit weights for madd
        // backward << 16 | forward in a 32 bit
        auto weightBF = _mm_set1_epi32((Wall[k * 2 + 2] << 16) + Wall[k * 2 + 1]);
        res = _mm_add_epi32(res, _mm_madd_epi16(src, weightBF));
      }

      res = _mm_add_epi32(res, rounder); // round

      res = _mm_packs_epi32(_mm_srai_epi32(res, SHIFTBACK), z);
      // make unsigned when unsigned 16 bit mode
      if constexpr (!lessThan16bits)
        res = _mm_add_epi16(res, signed16_shifter);

      // we are supporting only mod4
      // 4, 8, 12, ...
      _mm_storel_epi64((__m128i*)(pDst + x * sizeof(uint16_t)), res);

#if 0
      // sample from MDegrainX, not only mod4
      if constexpr (blockWidth == 6) {
        // special, 4+2
        if (x == 0)
          _mm_storel_epi64((__m128i*)(pDst + x * sizeof(uint16_t)), res);
        else
          *(uint32_t*)(pDst + x * sizeof(uint16_t)) = _mm_cvtsi128_si32(res);
      }
      else if constexpr (blockWidth >= 8 / sizeof(uint16_t)) { // block 4 is already 8 bytes
        // 4, 8, 12, ...
        _mm_storel_epi64((__m128i*)(pDst + x * sizeof(uint16_t)), res);
      }
      else if constexpr (blockWidth == 3) { // blockwidth 3 is 6 bytes
        // x == 0 always
        *(uint32_t*)(pDst) = _mm_cvtsi128_si32(res); // 1-4 bytes
        uint32_t res32 = _mm_cvtsi128_si32(_mm_srli_si128(res, 4)); // 5-8 byte
        *(uint16_t*)(pDst + sizeof(uint32_t)) = (uint16_t)res32; // 2 bytes needed
      }
      else { // blockwidth 2 is 4 bytes
        *(uint32_t*)(pDst + x * sizeof(uint16_t)) = _mm_cvtsi128_si32(res);
      }
#endif

    }

    pDst += nDstPitch;
    pSrc += nSrcPitch;
    for (int k = 0; k < trad; ++k)
    {
      pRef[k * 2] += Pitch[k * 2];
      pRef[k * 2 + 1] += Pitch[k * 2 + 1];
    }
  }

}

MDegrainN::DenoiseNFunction* MDegrainN::get_denoiseN_function(int BlockX, int BlockY, int _bits_per_pixel, bool _lsb_flag, bool _out16_flag, arch_t arch)
{
  //---------- DENOISE/DEGRAIN
  const int DEGRAIN_TYPE_8BIT = 1;
  const int DEGRAIN_TYPE_8BIT_STACKED = 2;
  const int DEGRAIN_TYPE_8BIT_OUT16 = 4;
  const int DEGRAIN_TYPE_10to14BIT = 8;
  const int DEGRAIN_TYPE_16BIT = 16;
  const int DEGRAIN_TYPE_32BIT = 32;
  // BlkSizeX, BlkSizeY, degrain_type, arch_t
  std::map<std::tuple<int, int, int, arch_t>, DenoiseNFunction*> func_degrain;
  using std::make_tuple;

  int type_to_search;
  if (_bits_per_pixel == 8) {
    if (_out16_flag)
      type_to_search = DEGRAIN_TYPE_8BIT_OUT16;
    else if (_lsb_flag)
      type_to_search = DEGRAIN_TYPE_8BIT_STACKED;
    else
      type_to_search = DEGRAIN_TYPE_8BIT;
  }
  else if (_bits_per_pixel <= 14)
    type_to_search = DEGRAIN_TYPE_10to14BIT;
  else if (_bits_per_pixel == 16)
    type_to_search = DEGRAIN_TYPE_16BIT;
  else if (_bits_per_pixel == 32)
    type_to_search = DEGRAIN_TYPE_32BIT;
  else
    return nullptr;


  // 8bit C, 8bit lsb C, 8bit out16 C, 10-16 bit C, float C (same for all, no blocksize templates)
#define MAKE_FN(x, y) \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_8BIT, NO_SIMD)] = DegrainN_C<uint8_t, x, y, 0>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_8BIT_STACKED, NO_SIMD)] = DegrainN_C<uint8_t, x, y, 1>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_8BIT_OUT16, NO_SIMD)] = DegrainN_C<uint8_t, x, y, 2>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_10to14BIT, NO_SIMD)] = DegrainN_C<uint16_t, x, y, 0>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_16BIT, NO_SIMD)] = DegrainN_C<uint16_t, x, y, 0>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_32BIT, NO_SIMD)] = DegrainN_C<float, x, y, 0>;
    MAKE_FN(64, 64)
    MAKE_FN(64, 48)
    MAKE_FN(64, 32)
    MAKE_FN(64, 16)
    MAKE_FN(48, 64)
    MAKE_FN(48, 48)
    MAKE_FN(48, 24)
    MAKE_FN(48, 12)
    MAKE_FN(32, 64)
    MAKE_FN(32, 32)
    MAKE_FN(32, 24)
    MAKE_FN(32, 16)
    MAKE_FN(32, 8)
    MAKE_FN(24, 48)
    MAKE_FN(24, 32)
    MAKE_FN(24, 24)
    MAKE_FN(24, 12)
    MAKE_FN(24, 6)
    MAKE_FN(16, 64)
    MAKE_FN(16, 32)
    MAKE_FN(16, 16)
    MAKE_FN(16, 12)
    MAKE_FN(16, 8)
    MAKE_FN(16, 4)
    MAKE_FN(16, 2)
    MAKE_FN(16, 1)
    MAKE_FN(12, 48)
    MAKE_FN(12, 24)
    MAKE_FN(12, 16)
    MAKE_FN(12, 12)
    MAKE_FN(12, 6)
    MAKE_FN(12, 3)
    MAKE_FN(8, 32)
    MAKE_FN(8, 16)
    MAKE_FN(8, 8)
    MAKE_FN(8, 4)
    MAKE_FN(8, 2)
    MAKE_FN(8, 1)
    MAKE_FN(6, 24)
    MAKE_FN(6, 12)
    MAKE_FN(6, 6)
    MAKE_FN(6, 3)
    MAKE_FN(4, 8)
    MAKE_FN(4, 4)
    MAKE_FN(4, 2)
    MAKE_FN(4, 1)
    MAKE_FN(3, 6)
    MAKE_FN(3, 3)
    MAKE_FN(2, 4)
    MAKE_FN(2, 2)
    MAKE_FN(2, 1)
#undef MAKE_FN
#undef MAKE_FN_LEVEL

      // and the SSE2 versions for 8 bit
#define MAKE_FN(x, y) \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_8BIT, USE_SSE2)] = DegrainN_sse2<x, y, 0>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_8BIT_STACKED, USE_SSE2)] = DegrainN_sse2<x, y, 1>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_8BIT_OUT16, USE_SSE2)] = DegrainN_sse2<x, y, 2>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_10to14BIT, USE_SSE41)] = DegrainN_16_sse41<x, y, true>; \
func_degrain[make_tuple(x, y, DEGRAIN_TYPE_16BIT, USE_SSE41)] = DegrainN_16_sse41<x, y, false>;

  MAKE_FN(64, 64)
    MAKE_FN(64, 48)
    MAKE_FN(64, 32)
    MAKE_FN(64, 16)
    MAKE_FN(48, 64)
    MAKE_FN(48, 48)
    MAKE_FN(48, 24)
    MAKE_FN(48, 12)
    MAKE_FN(32, 64)
    MAKE_FN(32, 32)
    MAKE_FN(32, 24)
    MAKE_FN(32, 16)
    MAKE_FN(32, 8)
    MAKE_FN(24, 48)
    MAKE_FN(24, 32)
    MAKE_FN(24, 24)
    MAKE_FN(24, 12)
    MAKE_FN(24, 6)
    MAKE_FN(16, 64)
    MAKE_FN(16, 32)
    MAKE_FN(16, 16)
    MAKE_FN(16, 12)
    MAKE_FN(16, 8)
    MAKE_FN(16, 4)
    MAKE_FN(16, 2)
    MAKE_FN(16, 1)
    MAKE_FN(12, 48)
    MAKE_FN(12, 24)
    MAKE_FN(12, 16)
    MAKE_FN(12, 12)
    MAKE_FN(12, 6)
    MAKE_FN(12, 3) 
    MAKE_FN(8, 32)
    MAKE_FN(8, 16)
    MAKE_FN(8, 8)
    MAKE_FN(8, 4)
    MAKE_FN(8, 2)
    MAKE_FN(8, 1)
    //MAKE_FN(6, 24) // w is mod4 only supported
    //MAKE_FN(6, 12)
    //MAKE_FN(6, 6)
    //MAKE_FN(6, 3)
    MAKE_FN(4, 8)
    MAKE_FN(4, 4)
    MAKE_FN(4, 2)
    MAKE_FN(4, 1)
    //MAKE_FN(3, 6) // w is mod4 only supported
    //MAKE_FN(3, 3)
    //MAKE_FN(2, 4) // no 2 byte width, only C
    //MAKE_FN(2, 2) // no 2 byte width, only C
    //MAKE_FN(2, 1) // no 2 byte width, only C
#undef MAKE_FN
#undef MAKE_FN_LEVEL

  DenoiseNFunction* result = nullptr;
  arch_t archlist[] = { USE_AVX2, USE_AVX, USE_SSE41, USE_SSE2, NO_SIMD };
  int index = 0;
  while (result == nullptr) {
    arch_t current_arch_try = archlist[index++];
    if (current_arch_try > arch) continue;
    result = func_degrain[make_tuple(BlockX, BlockY, type_to_search, current_arch_try)];
    if (result == nullptr && current_arch_try == NO_SIMD)
      break;
  }
  return result;
}



MDegrainN::MDegrainN(
  PClip child, PClip super, PClip mvmulti, int trad,
  sad_t thsad, sad_t thsadc, int yuvplanes, float nlimit, float nlimitc,
  sad_t nscd1, int nscd2, bool isse_flag, bool planar_flag, bool lsb_flag,
  sad_t thsad2, sad_t thsadc2, bool mt_flag, bool out16_flag, int wpow, float adjSADzeromv, float adjSADcohmv, int thCohMV,
  float MVLPFCutoff, float MVLPFSlope, float MVLPFGauss, int thMVLPFCorr, float adjSADLPFedmv,
  int UseSubShift, int InterpolateOverlap, PClip _mvmultirs, int _thFWBWmvpos,
  int _MPBthSub, int _MPBthAdd, int _MPBNumIt, float _MPB_SPC_sub, float _MPB_SPC_add, bool _MPB_PartBlend,
  int _MPBthIVS, bool _showIVSmask, ::PClip _mvmultivs, int _MPB_DMFlags, int _MPBchroma, int _MPBtgtTR, int _MPB_MVlth,
  int _pmode, int _TTH_DMFlags, int _TTH_thUPD, int _TTH_BAS, bool _TTH_chroma, PClip _dnmask,
  float _thSADA_a, float _thSADA_b, int _MVMedF, int _MVMedF_em, int _MVMedF_cm, int _MVF_fm,
  int _MGR, int _MGR_sr, int _MGR_st, int _MGR_pm,
  int _LtComp, int _NEW_DMFlags,
  IScriptEnvironment* env_ptr
)
  : GenericVideoFilter(child)
  , MVFilter(mvmulti, "MDegrainN", env_ptr, 1, 0)
  , _mv_clip_arr()
  , _trad(trad)
  , _yuvplanes(yuvplanes)
  , _nlimit(nlimit)
  , _nlimitc(nlimitc)
  , _super(super)
  , _planar_flag(planar_flag)
  , _lsb_flag(lsb_flag)
  , _mt_flag(mt_flag)
  , _out16_flag(out16_flag)
  , _height_lsb_or_out16_mul((lsb_flag || out16_flag) ? 2 : 1)
  , _nsupermodeyuv(-1)
  , _dst_planes(nullptr)
  , _src_planes(nullptr)
  , _overwins()
  , _overwins_uv()
  , _oversluma_ptr(0)
  , _overschroma_ptr(0)
  , _oversluma16_ptr(0)
  , _overschroma16_ptr(0)
  , _oversluma32_ptr(0)
  , _overschroma32_ptr(0)
  , _oversluma_lsb_ptr(0)
  , _overschroma_lsb_ptr(0)
  , _degrainluma_ptr(0)
  , _degrainchroma_ptr(0)
  , _dst_short()
  , _dst_short_pitch()
  , _dst_int()
  , _dst_int_pitch()
  , _dst_shortUV1()
  , _dst_intUV1()
  , _dst_shortUV2()
  , _dst_intUV2()
  //,	_usable_flag_arr ()
  //,	_planes_ptr ()
  //,	_dst_ptr_arr ()
  //,	_src_ptr_arr ()
  //,	_dst_pitch_arr ()
  //,	_src_pitch_arr ()
  //,	_lsb_offset_arr ()
  , _covered_width(0)
  , _covered_height(0)
  , _boundary_cnt_arr()
  , fadjSADzeromv(adjSADzeromv)
  , fadjSADcohmv(adjSADcohmv)
  , fadjSADLPFedmv(adjSADLPFedmv)
  , ithCohMV(thCohMV) // need to scale to pel value ?
  , fMVLPFCutoff(MVLPFCutoff)
  , fMVLPFSlope(MVLPFSlope)
  , fMVLPFGauss(MVLPFGauss)
  , ithMVLPFCorr(thMVLPFCorr)
  , nUseSubShift(UseSubShift)
  , iInterpolateOverlap(InterpolateOverlap)
  , mvmultirs(_mvmultirs)
  , thFWBWmvpos(_thFWBWmvpos)
  , MPBthSub(_MPBthSub)
  , MPBthAdd(_MPBthAdd)
  , MPBNumIt(_MPBNumIt)
  , MPB_SPC_sub(_MPB_SPC_sub)
  , MPB_SPC_add(_MPB_SPC_add)
  , MPB_PartBlend(_MPB_PartBlend)
  , MPB_thIVS(_MPBthIVS)
  , showIVSmask(_showIVSmask)
  , mvmultivs(_mvmultivs)
  , MPBchroma(_MPBchroma)
  , MPBtgtTR(_MPBtgtTR)
  , MPB_DMFlags(_MPB_DMFlags)
  , MPB_MVlth(_MPB_MVlth)
  , pmode((PMode)_pmode)
  , TTH_DMFlags(_TTH_DMFlags)
  , TTH_thUPD(_TTH_thUPD)
  , TTH_BAS(_TTH_BAS)
  , TTH_chroma(_TTH_chroma)
  , dnmask(_dnmask)
  , thSADA_a(_thSADA_a)
  , thSADA_b(_thSADA_b)
  , iMVMedF(_MVMedF)
  , iMVMedF_em(_MVMedF_em)
  , iMVMedF_cm(_MVMedF_cm)
  , iMVF_fm(_MVF_fm)
  , iMGR(_MGR)
  , iMGR_sr(_MGR_sr)
  , iMGR_st(_MGR_st)
  , iMGR_pm(_MGR_pm)
  , iLtComp(_LtComp)
  , iNEW_DMFlags(_NEW_DMFlags)
  , veryBigSAD(3 * nBlkSizeX * nBlkSizeY * (pixelsize == 4 ? 1 : (1 << bits_per_pixel))) // * 256, pixelsize==2 -> 65536. Float:1
{
  has_at_least_v8 = true;
  try { env_ptr->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  if (trad > MAX_TEMP_RAD)
  {
    env_ptr->ThrowError(
      "MDegrainN: temporal radius too large (max %d)",
      MAX_TEMP_RAD
    );
  }
  else if (trad < 1)
  {
    env_ptr->ThrowError("MDegrainN: temporal radius must be at least 1.");
  }

  // check if padding is enough - must be at least blksize/2
  if (nHPadding < nBlkSizeX / 2) env_ptr->ThrowError("MDegrainN: hpad in MSuper must be >= blksize (%d).", nBlkSizeX);
  if (nVPadding < nBlkSizeY / 2) env_ptr->ThrowError("MDegrainN: vpad in MSuper must be >= blksize (%d).", nBlkSizeY);

  if (wpow < 1 || wpow > 7)
  {
    env_ptr->ThrowError("MDegrainN: wpow must be from 1 to 7. 7 = equal weights.");
  }

  if (iInterpolateOverlap > 0 && (nOverlapX > 0 || nOverlapY > 0))
  {
    env_ptr->ThrowError("MDegrainN: IntOvlp > 0 but input MVs clip already have overlap.");
  }

  if ((iInterpolateOverlap > 4) || (iInterpolateOverlap < 0))
  {
    env_ptr->ThrowError("MDegrainN: IntOvlp can be only from 0 to 4.");
  }

  // adjust main params of current MVFilter to overlapped, remember source params
      // save original input MVFilter params
  nInputBlkX = nBlkX;
  nInputBlkY = nBlkY;
  nInputBlkCount = nBlkCount;

  if ((iInterpolateOverlap == 1) || (iInterpolateOverlap == 2)) // full 4x blocknum H and V overlap
  {
    // save original input MVFilter params
    nInputBlkX = nBlkX;
    nInputBlkY = nBlkY;
    nInputBlkCount = nBlkCount;

    //assume interpolated overlap is always BlkSize/2
    nOverlapX = nBlkSizeX / 2;
    nOverlapY = nBlkSizeY / 2;

    nBlkX = (nWidth - nOverlapX)
      / (nBlkSizeX - nOverlapX);
    nBlkY = (nHeight - nOverlapY)
      / (nBlkSizeY - nOverlapY);

    nBlkCount = nBlkX * nBlkY;

  }

  if ((iInterpolateOverlap == 3) || (iInterpolateOverlap == 4)) // diagonal 2x blocknum V +0.5H overlap
  {
    // save original input MVFilter params
    nInputBlkX = nBlkX;
    nInputBlkY = nBlkY;
    nInputBlkCount = nBlkCount;

    //assume interpolated overlap is always BlkSize/2
    nOverlapX = 0;
    nOverlapY = nBlkSizeY / 2;

//    nBlkX = (nWidth) / (nBlkSizeX); - not changed
    nBlkY = (nHeight - nOverlapY)
      / (nBlkSizeY - nOverlapY);

    nBlkCount = nBlkX * nBlkY;
    bDiagOvlp = true;

  }
  else
  {
    bDiagOvlp = false;
  }

  _wpow = wpow;
  // scale to nPel^2
  MPB_thIVS *= (nPel * nPel);

  _mv_clip_arr.resize(_trad * 2);
  for (int k = 0; k < _trad * 2; ++k)
  {
    _mv_clip_arr[k]._clip_sptr = SharedPtr <MVClip>(
      new MVClip(mvmulti, nscd1, nscd2, env_ptr, _trad * 2, k, true) // use MVsArray only, not blocks[]
      );

    static const char *name_0[2] = { "mvbw", "mvfw" };
    char txt_0[127 + 1];
    sprintf(txt_0, "%s%d", name_0[k & 1], 1 + k / 2);
//    CheckSimilarity(*(_mv_clip_arr[k]._clip_sptr), txt_0, env_ptr);
    if (iInterpolateOverlap == 0)
      CheckSimilarity(*(_mv_clip_arr[k]._clip_sptr), txt_0, env_ptr);
    else
      CheckSimilarityEO(*(_mv_clip_arr[k]._clip_sptr), txt_0, env_ptr);
  }

  if (mvmultirs != 0) // separate reverse search MVclip provided
  {
    for (int k = 0; k < _trad * 2; ++k)
    {
      _mv_clip_arr[k]._cliprs_sptr = SharedPtr <MVClip>(
        new MVClip(mvmultirs, nscd1, nscd2, env_ptr, _trad * 2, k, true) // use MVsArray only, not blocks[]
        );

      static const char* name_0[2] = { "mvbw", "mvfw" };
      char txt_0[127 + 1];
      sprintf(txt_0, "%s%d", name_0[k & 1], 1 + k / 2);
      //    CheckSimilarity(*(_mv_clip_arr[k]._clip_sptr), txt_0, env_ptr);
      if (iInterpolateOverlap == 0)
        CheckSimilarity(*(_mv_clip_arr[k]._cliprs_sptr), txt_0, env_ptr);
      else
        CheckSimilarityEO(*(_mv_clip_arr[k]._cliprs_sptr), txt_0, env_ptr);
    }
  }

  if (mvmultivs != 0) // separate MVclip provided for IVS check/mask
  {
    for (int k = 0; k < _trad * 2; ++k)
    {
      _mv_clip_arr[k]._clipvs_sptr = SharedPtr <MVClip>(
        new MVClip(mvmultivs, nscd1, nscd2, env_ptr, _trad * 2, k, true) // use MVsArray only, not blocks[]
        );

      static const char* name_0[2] = { "mvbw", "mvfw" };
      char txt_0[127 + 1];
      sprintf(txt_0, "%s%d", name_0[k & 1], 1 + k / 2);
      //    CheckSimilarity(*(_mv_clip_arr[k]._clip_sptr), txt_0, env_ptr);
      if (iInterpolateOverlap == 0)
        CheckSimilarity(*(_mv_clip_arr[k]._clipvs_sptr), txt_0, env_ptr);
      else
        CheckSimilarityEO(*(_mv_clip_arr[k]._clipvs_sptr), txt_0, env_ptr);
    }
  }

  const sad_t mv_thscd1 = _mv_clip_arr[0]._clip_sptr->GetThSCD1();
  thsad = (uint64_t)thsad   * mv_thscd1 / nscd1;	// normalize to block SAD
  thsadc = (uint64_t)thsadc  * mv_thscd1 / nscd1;	// chroma
  thsad2 = (uint64_t)thsad2  * mv_thscd1 / nscd1;
  thsadc2 = (uint64_t)thsadc2 * mv_thscd1 / nscd1;

  thSAD_param_norm = thsad;
  thSAD2_param_norm = thsad2;
  thSADC_param_norm = thsadc;
  thSADC2_param_norm = thsadc2;
  fthSAD12_ratio = (float)thSAD2_param_norm / (float)thSAD_param_norm;
  fthSADC12_ratio = (float)thSADC2_param_norm / (float)thSADC_param_norm;
  fthSAD_LC_ratio = (float)thSADC_param_norm / (float)thSAD_param_norm;
  thSCD1 = nscd1;

  if ((thsad != thsadc) || (thsad2 != thsadc2))
    bthLC_diff = true; // thSAD luma and chroma different - use separate DegrainWeight and normweights proc in YUV single pass processing
  else
    bthLC_diff = false;

  const ::VideoInfo &vi_super = _super->GetVideoInfo();

  if (!vi.IsSameColorspace(_super->GetVideoInfo()))
    env_ptr->ThrowError("MDegrainN: source and super clip video format is different!");

  // v2.7.39- make subsampling independent from motion vector's origin:
  // because xRatioUV and yRatioUV: in MVFilter, property of motion vectors
  xRatioUV_super = 1;
  yRatioUV_super = 1;
  if (!vi.IsY() && !vi.IsRGB()) {
    xRatioUV_super = vi.IsYUY2() ? 2 : (1 << vi.GetPlaneWidthSubsampling(PLANAR_U));
    yRatioUV_super = vi.IsYUY2() ? 1 : (1 << vi.GetPlaneHeightSubsampling(PLANAR_U));
  }
  nLogxRatioUV_super = ilog2(xRatioUV_super);
  nLogyRatioUV_super = ilog2(yRatioUV_super);

  pixelsize_super = vi_super.ComponentSize(); // of MVFilter
  bits_per_pixel_super = vi_super.BitsPerComponent();

  _cpuFlags = isse_flag ? env_ptr->GetCPUFlags() : 0;

// get parameters of prepared super clip - v2.0
  SuperParams64Bits params;
  memcpy(&params, &vi_super.num_audio_samples, 8);
  const int nHeightS = params.nHeight;
  const int nSuperHPad = params.nHPad;
  const int nSuperVPad = params.nVPad;
  const int nSuperPel = params.nPel;
  const int nSuperLevels = params.nLevels;
  const int nSuperParam = params.param;
  _nsupermodeyuv = params.nModeYUV;

  const bool bPelRefine = (nSuperParam & 1); // LSB of free param member

  if (!bPelRefine && (UseSubShift == 0) && (nSuperPel > 1))
  {
    env_ptr->ThrowError("MDegrainN: super clip do not have refined planes for pel > 1 and no internal subshifting is used");
  }

  // no need for SAD scaling, it is coming from the mv clip analysis. nSCD1 is already scaled in MVClip constructor
  /* must be good from 2.7.13.22
  thsad = sad_t(thsad / 255.0 * ((1 << bits_per_pixel) - 1));
  thsadc = sad_t(thsadc / 255.0 * ((1 << bits_per_pixel) - 1));
  thsad2 = sad_t(thsad2 / 255.0 * ((1 << bits_per_pixel) - 1));
  thsadc2 = sad_t(thsadc2 / 255.0 * ((1 << bits_per_pixel) - 1));
  */

  for (int k = 0; k < _trad * 2; ++k)
  {
    MvClipInfo &c_info = _mv_clip_arr[k];

    c_info._gof_sptr = SharedPtr <MVGroupOfFrames>(new MVGroupOfFrames(
      nSuperLevels,
      nWidth,
      nHeight,
      nSuperPel,
      nSuperHPad,
      nSuperVPad,
      _nsupermodeyuv,
      _cpuFlags,
      xRatioUV_super,
      yRatioUV_super,
      pixelsize_super,
      bits_per_pixel_super,
      mt_flag
    ));

    // Computes the SAD thresholds for this source frame, a cosine-shaped
    // smooth transition between thsad(c) and thsad(c)2.
    const int		d = k / 2 + 1;
    c_info._thsad = ClipFnc::interpolate_thsad(thsad, thsad2, d, _trad);
    c_info._thsadc = ClipFnc::interpolate_thsad(thsadc, thsadc2, d, _trad);
    //    c_info._thsad_sq = double(c_info._thsad) * double(c_info._thsad); // 2.7.46
    //    c_info._thsadc_sq = double(c_info._thsadc) * double(c_info._thsadc);
    c_info._thsad_sq = double(c_info._thsad);
    c_info._thsadc_sq = double(c_info._thsadc);
    for (int i = 0; i < _wpow - 1; i++)
    {
      c_info._thsad_sq *= double(c_info._thsad);
      c_info._thsadc_sq *= double(c_info._thsadc);
    }
  }

  const int nSuperWidth = vi_super.width;
  pixelsize_super_shift = ilog2(pixelsize_super);

  if (nHeight != nHeightS
    || nHeight != vi.height
    || nWidth != nSuperWidth - nSuperHPad * 2
    || nWidth != vi.width
    || nPel != nSuperPel)
  {
    env_ptr->ThrowError("MDegrainN : wrong source or super frame size");
  }

  if(lsb_flag && (pixelsize != 1 || pixelsize_super != 1))
    env_ptr->ThrowError("MDegrainN : lsb_flag only for 8 bit sources");

  if (out16_flag) {
    if (pixelsize != 1 || pixelsize_super != 1)
      env_ptr->ThrowError("MDegrainN : out16 flag only for 8 bit sources");
    if (!vi.IsY8() && !vi.IsYV12() && !vi.IsYV16() && !vi.IsYV24())
      env_ptr->ThrowError("MDegrainN : only YV8, YV12, YV16 or YV24 allowed for out16");
  }

  if (lsb_flag && out16_flag)
    env_ptr->ThrowError("MDegrainN : cannot specify both lsb and out16 flag");

  // output can be different bit depth from input
  pixelsize_output = pixelsize_super;
  bits_per_pixel_output = bits_per_pixel_super;
  pixelsize_output_shift = pixelsize_super_shift;
  if (out16_flag) {
    pixelsize_output = sizeof(uint16_t);
    bits_per_pixel_output = 16;
    pixelsize_output_shift = ilog2(pixelsize_output);
  }

  if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !_planar_flag)
  {
    _dst_planes = std::unique_ptr <YUY2Planes>(
      new YUY2Planes(nWidth, nHeight * _height_lsb_or_out16_mul)
      );
    _src_planes = std::unique_ptr <YUY2Planes>(
      new YUY2Planes(nWidth, nHeight)
      );
  }
  _dst_short_pitch = ((nWidth + 15) / 16) * 16;
  _dst_int_pitch = _dst_short_pitch;

  if (nOverlapX > 0 || nOverlapY > 0)
  {
    if (!bDiagOvlp)
    {
      _overwins = std::unique_ptr <OverlapWindows>(
        new OverlapWindows(nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY)
        );
      _overwins_uv = std::unique_ptr <OverlapWindows>(new OverlapWindows(
        nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super,
        nOverlapX >> nLogxRatioUV_super, nOverlapY >> nLogyRatioUV_super
      ));
    }
    else // interpolated diagonal overlap
    {
      _overwins = std::unique_ptr <OverlapWindows>(
        new OverlapWindows(nBlkSizeX, nBlkSizeY, nBlkSizeX / 2, nOverlapY, true)
        );
      _overwins_uv = std::unique_ptr <OverlapWindows>(new OverlapWindows(
        nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super,
        (nBlkSizeX / 2) >> nLogxRatioUV_super, nOverlapY >> nLogyRatioUV_super, true
      ));
    }

    if (_lsb_flag || pixelsize_output > 1)
    {
      _dst_int.resize(_dst_int_pitch * nHeight);
      _dst_intUV1.resize(_dst_int_pitch * nHeight);
      _dst_intUV2.resize(_dst_int_pitch * nHeight);
    }
    else
    {
      _dst_short.resize(_dst_short_pitch * nHeight);
      _dst_shortUV1.resize(_dst_short_pitch * nHeight); // really may be down to 4 times less for 4:2:0 ?
      _dst_shortUV2.resize(_dst_short_pitch * nHeight);
    }
  }
  if (nOverlapY > 0)
  {
    _boundary_cnt_arr.resize(nBlkY);
  }

    // in overlaps.h
    // OverlapsLsbFunction
    // OverlapsFunction
    // in M(V)DegrainX: DenoiseXFunction
  arch_t arch;

  if ((nBlkSizeX == 8) && (nBlkSizeY == 8))
  {
    if ((_cpuFlags & CPUF_AVX2) != 0)
      arch = USE_AVX2;
    else if ((_cpuFlags & CPUF_AVX) != 0)
      arch = USE_AVX;
    else if ((_cpuFlags & CPUF_SSE4_1) != 0)
      arch = USE_SSE41;
    else if ((_cpuFlags & CPUF_SSE2) != 0)
      arch = USE_SSE2;
    else
      arch = NO_SIMD;
  }
  else // no SSSE3 x264 SAD - requires 16bytes ref block align
  {
    if ((_cpuFlags & CPUF_AVX2) != 0)
      arch = USE_AVX2;
    else if ((_cpuFlags & CPUF_AVX) != 0)
      arch = USE_AVX;
    else
      arch = NO_SIMD;
  }

  SAD = get_sad_function(nBlkSizeX, nBlkSizeY, bits_per_pixel, arch);
  SADCHROMA = get_sad_function(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, arch);

  COVAR = get_covar_function(nBlkSizeX, nBlkSizeY, bits_per_pixel, arch);
  COVARCHROMA = get_covar_function(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, arch);

  DM_Luma = new DisMetric(nBlkSizeX, nBlkSizeY, bits_per_pixel, pixelsize, arch, MPB_DMFlags);
  DM_Chroma = new DisMetric(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, pixelsize, arch, MPB_DMFlags);

  DM_TTH_Luma = new DisMetric(nBlkSizeX, nBlkSizeY, bits_per_pixel, pixelsize, arch, TTH_DMFlags);
  DM_TTH_Chroma = new DisMetric(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, pixelsize, arch, TTH_DMFlags);

  if (iNEW_DMFlags != 0)
  {
    DM_NEW_Luma = new DisMetric(nBlkSizeX, nBlkSizeY, bits_per_pixel, pixelsize, arch, iNEW_DMFlags);
    DM_NEW_Chroma = new DisMetric(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, pixelsize, arch, iNEW_DMFlags);
  }

// C only -> NO_SIMD
  _oversluma_lsb_ptr = get_overlaps_lsb_function(nBlkSizeX, nBlkSizeY, sizeof(uint8_t), NO_SIMD);
  _overschroma_lsb_ptr = get_overlaps_lsb_function(nBlkSizeX / xRatioUV_super, nBlkSizeY / yRatioUV_super, sizeof(uint8_t), NO_SIMD);

  _oversluma_ptr = get_overlaps_function(nBlkSizeX, nBlkSizeY, sizeof(uint8_t), false, arch);
  _overschroma_ptr = get_overlaps_function(nBlkSizeX / xRatioUV_super, nBlkSizeY / yRatioUV_super, sizeof(uint8_t), false, arch);

  _oversluma16_ptr = get_overlaps_function(nBlkSizeX, nBlkSizeY, sizeof(uint16_t), false, arch);
  _overschroma16_ptr = get_overlaps_function(nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super, sizeof(uint16_t), false, arch);

  _oversluma32_ptr = get_overlaps_function(nBlkSizeX, nBlkSizeY, sizeof(float), false, arch);
  _overschroma32_ptr = get_overlaps_function(nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super, sizeof(float), false, arch);

  _degrainluma_ptr = get_denoiseN_function(nBlkSizeX, nBlkSizeY, bits_per_pixel_super, lsb_flag, out16_flag, arch);
  _degrainchroma_ptr = get_denoiseN_function(nBlkSizeX / xRatioUV_super, nBlkSizeY / yRatioUV_super, bits_per_pixel_super, lsb_flag, out16_flag, arch);

  if (!_oversluma_lsb_ptr)
    env_ptr->ThrowError("MDegrainN : no valid _oversluma_lsb_ptr function for %dx%d, pixelsize=%d, lsb_flag=%d", nBlkSizeX, nBlkSizeY, pixelsize_super, (int)lsb_flag);
  if (!_overschroma_lsb_ptr)
    env_ptr->ThrowError("MDegrainN : no valid _overschroma_lsb_ptr function for %dx%d, pixelsize=%d, lsb_flag=%d", nBlkSizeX, nBlkSizeY, pixelsize_super, (int)lsb_flag);
  if (!_oversluma_ptr)
    env_ptr->ThrowError("MDegrainN : no valid _oversluma_ptr function for %dx%d, pixelsize=%d, lsb_flag=%d", nBlkSizeX, nBlkSizeY, pixelsize_super, (int)lsb_flag);
  if (!_overschroma_ptr)
    env_ptr->ThrowError("MDegrainN : no valid _overschroma_ptr function for %dx%d, pixelsize=%d, lsb_flag=%d", nBlkSizeX, nBlkSizeY, pixelsize_super, (int)lsb_flag);
  if (!_degrainluma_ptr)
    env_ptr->ThrowError("MDegrainN : no valid _degrainluma_ptr function for %dx%d, pixelsize=%d, lsb_flag=%d", nBlkSizeX, nBlkSizeY, pixelsize_super, (int)lsb_flag);
  if (!_degrainchroma_ptr)
    env_ptr->ThrowError("MDegrainN : no valid _degrainchroma_ptr function for %dx%d, pixelsize=%d, lsb_flag=%d", nBlkSizeX, nBlkSizeY, pixelsize_super, (int)lsb_flag);

  if ((_cpuFlags & CPUF_SSE2) != 0)
  {
    if(out16_flag)
      LimitFunction = LimitChanges_src8_target16_c; // todo SSE2
    else if (pixelsize_super == 1)
      LimitFunction = LimitChanges_sse2_new<uint8_t, 0>;
    else if (pixelsize_super == 2) { // pixelsize_super == 2
      if ((_cpuFlags & CPUF_SSE4_1) != 0)
        LimitFunction = LimitChanges_sse2_new<uint16_t, 1>;
      else
        LimitFunction = LimitChanges_sse2_new<uint16_t, 0>;
    }
    else {
      LimitFunction = LimitChanges_float_c; // no SSE2
    }
  }
  else
  {
    if (out16_flag)
      LimitFunction = LimitChanges_src8_target16_c; // todo SSE2
    else if (pixelsize_super == 1)
      LimitFunction = LimitChanges_c<uint8_t>;
    else if (pixelsize_super == 2)
      LimitFunction = LimitChanges_c<uint16_t>;
    else
      LimitFunction = LimitChanges_float_c;
  }

  //---------- end of functions

  // 16 bit output hack
  if (_lsb_flag)
  {
    vi.height <<= 1;
  }

  if (out16_flag) {
    if (vi.IsY8())
      vi.pixel_type = VideoInfo::CS_Y16;
    else if (vi.IsYV12())
      vi.pixel_type = VideoInfo::CS_YUV420P16;
    else if (vi.IsYV16())
      vi.pixel_type = VideoInfo::CS_YUV422P16;
    else if (vi.IsYV24())
      vi.pixel_type = VideoInfo::CS_YUV444P16;
  }

  if ((fadjSADzeromv != 1.0f) || (fadjSADcohmv != 1.0f))
  {
    use_block_y_func = &MDegrainN::use_block_y_thSADzeromv_thSADcohmv;
    use_block_uv_func = &MDegrainN::use_block_uv_thSADzeromv_thSADcohmv;
  }
  else // old funcs
  {
    use_block_y_func = &MDegrainN::use_block_y;
    use_block_uv_func = &MDegrainN::use_block_uv;
  }

  // calculate MV LPF filter kernel (from fMVLPFCutoff and (in future) fMVLPFSlope params)
  // for interlaced field-based content the +-0.5 V shift should be added after filtering ?
  if (fMVLPFCutoff < 1.0f || fMVLPFGauss > 0.0f || iMVMedF > 0)
    bMVsAddProc = true;
  else
    bMVsAddProc = false;

  float fPi = 3.14159265f;
  int iKS_d2 = MVLPFKERNELSIZE / 2;

  if (fMVLPFGauss == 0.0f)
  {
    for (int i = 0; i < MVLPFKERNELSIZE; i++)
    {
      float fArg = (float)(i - iKS_d2) * fPi * fMVLPFCutoff;
      fMVLPFKernel[i] = fSinc(fArg);

      // Lanczos weighting
      float fArgLz = (float)(i - iKS_d2) * fPi / (float)(iKS_d2);
      fMVLPFKernel[i] *= fSinc(fArgLz);
    }
  }
  else // gaussian impulse kernel
  {
    for (int i = 0; i < MVLPFKERNELSIZE; i++)
    {
      float fArg = (float)(i - iKS_d2) * fMVLPFGauss;
      fMVLPFKernel[i] = powf(2.0, -fArg * fArg);
    }
  }

  float fSum = 0.0f;
  for (int i = 0; i < MVLPFKERNELSIZE; i++)
  {
    fSum += fMVLPFKernel[i];
  }

  for (int i = 0; i < MVLPFKERNELSIZE; i++)
  {
    fMVLPFKernel[i] /= fSum;
  }

  // allocate filtered MVs arrays
  for (int k = 0; k < _trad * 2; ++k)
  {
    uint8_t* pTmp_a;
    VECTOR* pTmp;
#ifdef _WIN32
    // to prevent cache set overloading when accessing fpob MVs arrays - add linear L2L3_CACHE_LINE_SIZE-bytes sized offset to different allocations
    SIZE_T stSizeToAlloc = nBlkCount * sizeof(VECTOR) + _trad * 2 * L2L3_CACHE_LINE_SIZE;

    pTmp_a = (BYTE*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
    pFilteredMVsPlanesArrays_a[k] = pTmp_a;
    pTmp = (VECTOR*)(pTmp_a + k * L2L3_CACHE_LINE_SIZE);
#else
    pTmp = new VECTOR[nBlkCount]; // allocate in heap ?
    pFilteredMVsPlanesArrays[k] = pTmp;
#endif
    pFilteredMVsPlanesArrays[k] = pTmp;

  }

  // allocate interpolated overlap MVs arrays
  for (int k = 0; k < _trad * 2; ++k)
  {
    uint8_t* pTmp_a;
    VECTOR* pTmp;
#ifdef _WIN32
    // to prevent cache set overloading when accessing fpob MVs arrays - add linear L2L3_CACHE_LINE_SIZE-bytes sized offset to different allocations
    SIZE_T stSizeToAlloc = nBlkCount * sizeof(VECTOR) + _trad * 2 * L2L3_CACHE_LINE_SIZE;

    pTmp_a = (BYTE*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
    pMVsIntOvlpPlanesArrays_a[k] = pTmp_a;
    pTmp = (VECTOR*)(pTmp_a + k * L2L3_CACHE_LINE_SIZE);
#else
    pTmp = new VECTOR[nBlkCount]; // allocate in heap ?
    pMVsIntOvlpPlanesArrays_a[k] = pTmp;
#endif
    pMVsIntOvlpPlanesArrays[k] = pTmp;

    if (mvmultivs)
    {
#ifdef _WIN32
      pTmp_a = (BYTE*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
      pMVsIntOvlpPlanesArraysVS_a[k] = pTmp_a;
      pTmp = (VECTOR*)(pTmp_a + k * L2L3_CACHE_LINE_SIZE);
#else
      pTmp = new VECTOR[nBlkCount]; // allocate in heap ?
      pMVsIntOvlpPlanesArraysVS_a[k] = pTmp;
#endif
      pMVsIntOvlpPlanesArraysVS[k] = pTmp;

    }

  }

  // allocate temp single subtracted blocks memory area
  SIZE_T stSizeToAlloc = nBlkSizeX * nBlkSizeY * pixelsize + (_trad * 2 + 2); // to hold (trad*2 + 2) number of temp blocks, full blended + subtracted current + all refs
#ifdef _WIN32
  pMPBTempBlocks = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
  pMPBTempBlocksUV1 = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
  pMPBTempBlocksUV2 = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
#else
  pMPBTempBlocks = new uint8_t[stSizeToAlloc];
  pMPBTempBlocksUV1 = new uint8_t[stSizeToAlloc];
  pMPBTempBlocksUV2 = new uint8_t[stSizeToAlloc];
#endif

  // allocate MEL IIR filter memory storage
  if (TTH_thUPD > 0) // TTH in some mode enabled
  {
    SIZE_T stSizeToAlloc = nBlkSizeX * nBlkSizeY * pixelsize * nBlkCount;
    SIZE_T stSizeToAllocSum = nBlkCount * sizeof(int);

#ifdef _WIN32
    pMELmemY = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
    pMELmemUV1 = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
    pMELmemUV2 = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address

    pMELmemYSum = (int*)VirtualAlloc(0, stSizeToAllocSum, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
    pMELmemUV1Sum = (int*)VirtualAlloc(0, stSizeToAllocSum, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
    pMELmemUV2Sum = (int*)VirtualAlloc(0, stSizeToAllocSum, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address

#else
    pMELmemY = new uint8_t[stSizeToAlloc];
    pMELmemUV1 = new uint8_t[stSizeToAlloc];
    pMELmemUV2 = new uint8_t[stSizeToAlloc];

    pMELmemYSum = new int[stSizeToAllocSum];
    pMELmemUV1Sum = new int[stSizeToAllocSum];
    pMELmemUV2Sum = new int[stSizeToAllocSum];
#endif

    int iMaxSum = (trad * 2 + 1) * veryBigSAD; // do not overflow 32bit int ?
    for (int i = 0; i < nBlkCount; i++)
    {
      pMELmemYSum[i] = iMaxSum;
      pMELmemUV1Sum[i] = iMaxSum;
      pMELmemUV2Sum[i] = iMaxSum;
    }

    BA_Yarr = new BlockArea* [nBlkCount];
    BA_UV1arr = new BlockArea* [nBlkCount];
    BA_UV2arr = new BlockArea* [nBlkCount];

    DM_cache_arr = new DM_cache * [nBlkCount];

    for (int i = 0; i < nBlkCount; i++)
    {
      BA_Yarr[i] = new BlockArea(nBlkSizeX, nBlkSizeY, TTH_BAS, pixelsize, nPel, arch, TTH_DMFlags);
      BA_UV1arr[i] = new BlockArea(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, TTH_BAS, pixelsize, nPel, arch, TTH_DMFlags);
      BA_UV2arr[i] = new BlockArea(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, TTH_BAS, pixelsize, nPel, arch, TTH_DMFlags);

      DM_cache_arr[i] = new DM_cache(((_trad * 2 + 1) * (_trad * 2 + 1)) / 2);
    }

  }

  // calculate limits of blx/bly once in constructor
  if (nUseSubShift == 0)
  {
    iMinBlx = -nBlkSizeX * nPel;
    iMaxBlx = nWidth * nPel;
    iMinBly = -nBlkSizeY * nPel;
    iMaxBly = nHeight * nPel;
  }
  else
  {
    int iKS_sh_d2 = ((SHIFTKERNELSIZE / 2) + 2); // +2 is to prevent run out of buffer for UV planes
    iMinBlx = (-nBlkSizeX + iKS_sh_d2) * nPel;
    iMaxBlx = (nWidth - iKS_sh_d2) * nPel;
    iMinBly = (-nBlkSizeY + iKS_sh_d2) * nPel;
    iMaxBly = (nHeight - iKS_sh_d2) * nPel;
  }

  // Lighting compensation - memory allocation for temp buf for compensated ref blocks
  if (iLtComp > 0)
  {
    SIZE_T stSizeToAlloc = nBlkSizeX * nBlkSizeY * pixelsize * ( 2 * _trad); // all ref blocks, Y only blocks for now
    pCompRefsBlksY = new uint8_t[stSizeToAlloc];
  }

  // DN mask check and select
  if (dnmask != 0)
  {
    const ::VideoInfo& vi_dnmask = dnmask->GetVideoInfo();

    if (!vi_dnmask.IsY8())
      env_ptr->ThrowError("MDegrainN: dnmask clip must be Y8 format only!");

    /* TODO - make samples mode also
    if ((vi_dnmask.width == nWidth) && (vi_dnmask.height == nHeight))
      dn_mm = DN_MM_SAMPLES;
    else if ((vi_dnmask.width == nBlkX) && (vi_dnmask.height == nBlkY))
      dn_mm = DN_MM_BLOCKS;
    else
      env_ptr->ThrowError("MDegrainN: dnmask clip size must be either equal to input clip size for sample-based mode or block W and H count in block-based mode (full block count with overlap) !");
      */
    if ((vi_dnmask.width == nBlkX) && (vi_dnmask.height == nBlkY))
      dn_mm = DN_MM_BLOCKS;
    else
      env_ptr->ThrowError("MDegrainN: dnmask clip frame size must be of blocks horizontal and vertical count in block-based mode (full block count in current overlap mode used) ! \n Current blocks count H=%d V=%d", nBlkX, nBlkY);
  }
  else
    dn_mm = DN_MM_NONE;
}


MDegrainN::~MDegrainN()
{
  // Nothing
  for (int k = 0; k < _trad * 2; ++k)
  {
#ifdef _WIN32
    VirtualFree((LPVOID)pFilteredMVsPlanesArrays_a[k], 0, MEM_FREE);
    VirtualFree((LPVOID)pMVsIntOvlpPlanesArrays_a[k], 0, MEM_FREE);
    VirtualFree((LPVOID)pMVsIntOvlpPlanesArraysVS_a[k], 0, MEM_FREE);
    VirtualFree((LPVOID)pMPBTempBlocks, 0, MEM_FREE);
    VirtualFree((LPVOID)pMPBTempBlocksUV1, 0, MEM_FREE);
    VirtualFree((LPVOID)pMPBTempBlocksUV2, 0, MEM_FREE);
#else
    delete pFilteredMVsPlanesArrays[k];
    delete pMVsIntOvlpPlanesArrays[k];
    delete pMPBTempBlocks;
    delete pMPBTempBlocksUV1;
    delete pMPBTempBlocksUV2;
#endif
  }

  if (pmode == PM_MEL)
  {
#ifdef _WIN32
    VirtualFree((LPVOID)pMELmemY, 0, MEM_FREE);
    VirtualFree((LPVOID)pMELmemUV1, 0, MEM_FREE);
    VirtualFree((LPVOID)pMELmemUV2, 0, MEM_FREE);

    VirtualFree((LPVOID)pMELmemYSum, 0, MEM_FREE);
    VirtualFree((LPVOID)pMELmemUV1Sum, 0, MEM_FREE);
    VirtualFree((LPVOID)pMELmemUV2Sum, 0, MEM_FREE);

#else
    delete pMELmemY;
    delete pMELmemUV1;
    delete pMELmemUV2;

    delete pMELmemYSum;
    delete pMELmemUV1Sum;
    delete pMELmemUV2Sum;

#endif

    for (int i = 0; i < nBlkCount; ++i)
    {
      delete BA_Yarr[i];
      delete BA_UV1arr[i];
      delete BA_UV2arr[i];

      delete DM_cache_arr[i];
    }
  }

  if (iLtComp > 0)
  {
    delete pCompRefsBlksY;
  }

}

static void plane_copy_8_to_16_c(uint8_t *dstp, int dstpitch, const uint8_t *srcp, int srcpitch, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      reinterpret_cast<uint16_t *>(dstp)[x] = srcp[x] << 8;
    }
    dstp += dstpitch;
    srcp += srcpitch;
  }
}


::PVideoFrame __stdcall MDegrainN::GetFrame(int n, ::IScriptEnvironment* env_ptr)
{
  _covered_width = nBlkX * (nBlkSizeX - nOverlapX) + nOverlapX;
  _covered_height = nBlkY * (nBlkSizeY - nOverlapY) + nOverlapY;

  const BYTE * pRef[MAX_TEMP_RAD * 2][3];
  int nRefPitches[MAX_TEMP_RAD * 2][3];
  unsigned char *pDstYUY2;
  const unsigned char *pSrcYUY2;
  int nDstPitchYUY2;
  int nSrcPitchYUY2;

  iFrameNumRequested = n;// save to local var to use in DM cache

  for (int k2 = 0; k2 < _trad * 2; ++k2)
  {
    // reorder ror regular frames order in v2.0.9.2
    const int k = reorder_ref(k2);

    // v2.0.9.2 - it seems we do not need in vectors clip anymore when we
    // finished copying them to fakeblockdatas
    MVClip &mv_clip = *(_mv_clip_arr[k]._clip_sptr);
    ::PVideoFrame mv = mv_clip.GetFrame(n, env_ptr);
    mv_clip.Update(mv, env_ptr);
    _usable_flag_arr[k] = mv_clip.IsUsable();

    if (mv_clip.GetTrad() != _trad) env_ptr->ThrowError("MDegrainN : nTrad in mvmulti %d not equal to MDegrain(tr=%d), possibly wrong tr params in MAnalyse and MDegrain", mv_clip.GetTrad(), _trad);
    
    if (mvmultirs != 0) // get and update reverse search MVs
    {
      MVClip& mv_clip_rs = *(_mv_clip_arr[k]._cliprs_sptr);
      ::PVideoFrame mv_rs = mv_clip_rs.GetFrame(n, env_ptr);
      mv_clip_rs.Update(mv_rs, env_ptr);
    }

    if (mvmultivs != 0) // get and update IVS check MVs
    {
      MVClip& mv_clip_vs = *(_mv_clip_arr[k]._clipvs_sptr);
      ::PVideoFrame mv_vs = mv_clip_vs.GetFrame(n, env_ptr);
      mv_clip_vs.Update(mv_vs, env_ptr);
    }
  }

  if (dn_mm != DN_MM_NONE)
  {
    src_dnmask = dnmask->GetFrame(n, env_ptr);
    dnmask_pitch = YPITCH(src_dnmask);
    pDNMask = (BYTE*)YRPLAN(src_dnmask);
  }

  PVideoFrame src = child->GetFrame(n, env_ptr);
  PVideoFrame dst = has_at_least_v8 ? env_ptr->NewVideoFrameP(vi, &src) : env_ptr->NewVideoFrame(vi); // frame property support
   
  if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2)
  {
    if (!_planar_flag)
    {
      pDstYUY2 = dst->GetWritePtr();
      nDstPitchYUY2 = dst->GetPitch();
      _dst_ptr_arr[0] = _dst_planes->GetPtr();
      _dst_ptr_arr[1] = _dst_planes->GetPtrU();
      _dst_ptr_arr[2] = _dst_planes->GetPtrV();
      _dst_pitch_arr[0] = _dst_planes->GetPitch();
      _dst_pitch_arr[1] = _dst_planes->GetPitchUV();
      _dst_pitch_arr[2] = _dst_planes->GetPitchUV();

      pSrcYUY2 = src->GetReadPtr();
      nSrcPitchYUY2 = src->GetPitch();
      _src_ptr_arr[0] = _src_planes->GetPtr();
      _src_ptr_arr[1] = _src_planes->GetPtrU();
      _src_ptr_arr[2] = _src_planes->GetPtrV();
      _src_pitch_arr[0] = _src_planes->GetPitch();
      _src_pitch_arr[1] = _src_planes->GetPitchUV();
      _src_pitch_arr[2] = _src_planes->GetPitchUV();

      YUY2ToPlanes(
        pSrcYUY2, nSrcPitchYUY2, nWidth, nHeight,
        _src_ptr_arr[0], _src_pitch_arr[0],
        _src_ptr_arr[1], _src_ptr_arr[2], _src_pitch_arr[1],
        _cpuFlags
      );
    }
    else
    {
      _dst_ptr_arr[0] = dst->GetWritePtr();
      _dst_ptr_arr[1] = _dst_ptr_arr[0] + nWidth;
      _dst_ptr_arr[2] = _dst_ptr_arr[1] + nWidth / 2; //yuy2 xratio
      _dst_pitch_arr[0] = dst->GetPitch();
      _dst_pitch_arr[1] = _dst_pitch_arr[0];
      _dst_pitch_arr[2] = _dst_pitch_arr[0];
      _src_ptr_arr[0] = src->GetReadPtr();
      _src_ptr_arr[1] = _src_ptr_arr[0] + nWidth;
      _src_ptr_arr[2] = _src_ptr_arr[1] + nWidth / 2;
      _src_pitch_arr[0] = src->GetPitch();
      _src_pitch_arr[1] = _src_pitch_arr[0];
      _src_pitch_arr[2] = _src_pitch_arr[0];
    }
  }
  else
  {
    _dst_ptr_arr[0] = YWPLAN(dst);
    _dst_ptr_arr[1] = UWPLAN(dst);
    _dst_ptr_arr[2] = VWPLAN(dst);
    _dst_pitch_arr[0] = YPITCH(dst);
    _dst_pitch_arr[1] = UPITCH(dst);
    _dst_pitch_arr[2] = VPITCH(dst);
    _src_ptr_arr[0] = YRPLAN(src);
    _src_ptr_arr[1] = URPLAN(src);
    _src_ptr_arr[2] = VRPLAN(src);
    _src_pitch_arr[0] = YPITCH(src);
    _src_pitch_arr[1] = UPITCH(src);
    _src_pitch_arr[2] = VPITCH(src);
  }

//  DWORD dwOldProt;
//  BYTE* pbAVS = (BYTE*)_dst_ptr_arr[0];

  _lsb_offset_arr[0] = _dst_pitch_arr[0] * nHeight;
  _lsb_offset_arr[1] = _dst_pitch_arr[1] * (nHeight >> nLogyRatioUV_super);
  _lsb_offset_arr[2] = _dst_pitch_arr[2] * (nHeight >> nLogyRatioUV_super);

  if (_lsb_flag)
  {
    memset(_dst_ptr_arr[0] + _lsb_offset_arr[0], 0, _lsb_offset_arr[0]);
    if (!_planar_flag)
    {
      memset(_dst_ptr_arr[1] + _lsb_offset_arr[1], 0, _lsb_offset_arr[1]);
      memset(_dst_ptr_arr[2] + _lsb_offset_arr[2], 0, _lsb_offset_arr[2]);
    }
  }

  ::PVideoFrame ref[MAX_TEMP_RAD * 2];

  for (int k2 = 0; k2 < _trad * 2; ++k2)
  {
    // reorder ror regular frames order in v2.0.9.2
    const int k = reorder_ref(k2);
    MVClip &mv_clip = *(_mv_clip_arr[k]._clip_sptr);
    mv_clip.use_ref_frame(ref[k], _usable_flag_arr[k], _super, n, env_ptr);
  }

  if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2)
  {
    for (int k2 = 0; k2 < _trad * 2; ++k2)
    {
      const int k = reorder_ref(k2);
      if (_usable_flag_arr[k])
      {
        pRef[k][0] = ref[k]->GetReadPtr();
        pRef[k][1] = pRef[k][0] + ref[k]->GetRowSize() / 2;
        pRef[k][2] = pRef[k][1] + ref[k]->GetRowSize() / 4;
        nRefPitches[k][0] = ref[k]->GetPitch();
        nRefPitches[k][1] = nRefPitches[k][0];
        nRefPitches[k][2] = nRefPitches[k][0];
      }
    }
  }
  else
  {
    for (int k2 = 0; k2 < _trad * 2; ++k2)
    {
      const int k = reorder_ref(k2);
      if (_usable_flag_arr[k])
      {
        pRef[k][0] = YRPLAN(ref[k]);
        pRef[k][1] = URPLAN(ref[k]);
        pRef[k][2] = VRPLAN(ref[k]);
        nRefPitches[k][0] = YPITCH(ref[k]);
        nRefPitches[k][1] = UPITCH(ref[k]);
        nRefPitches[k][2] = VPITCH(ref[k]);
      }
    }
  }

  memset(_planes_ptr, 0, _trad * 2 * sizeof(_planes_ptr[0]));

  for (int k2 = 0; k2 < _trad * 2; ++k2)
  {
    const int k = reorder_ref(k2);
    MVGroupOfFrames &gof = *(_mv_clip_arr[k]._gof_sptr);
    gof.Update(
      _yuvplanes,
      const_cast <BYTE *> (pRef[k][0]), nRefPitches[k][0],
      const_cast <BYTE *> (pRef[k][1]), nRefPitches[k][1],
      const_cast <BYTE *> (pRef[k][2]), nRefPitches[k][2]
    );
    if (_yuvplanes & YPLANE)
    {
      _planes_ptr[k][0] = gof.GetFrame(0)->GetPlane(YPLANE);
      // set block size for MVplane
      _planes_ptr[k][0]->SetBlockSize(nBlkSizeX, nBlkSizeY); // hope it is never zero ptr ?
    }
    if (_yuvplanes & UPLANE)
    {
      _planes_ptr[k][1] = gof.GetFrame(0)->GetPlane(UPLANE);
      // set block size for MVplane
      if (_planes_ptr[k][1] != 0)
        _planes_ptr[k][1]->SetBlockSize(nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super); 
    }
    if (_yuvplanes & VPLANE)
    {
      _planes_ptr[k][2] = gof.GetFrame(0)->GetPlane(VPLANE);
      // set block size for MVplane
      if (_planes_ptr[k][2] != 0)
        _planes_ptr[k][2]->SetBlockSize(nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super); 
    }
  }

  // process reverse search MVs data to update SAD of std search to mark too bad MVs
  if (mvmultirs)
    ProcessRSMVdata();

  // load pMVsArray into temp buf once, 2.7.46
  if (iInterpolateOverlap > 0)
  {
    for (int k = 0; k < _trad * 2; ++k)
    {
      if ((iInterpolateOverlap == 1) || (iInterpolateOverlap == 2))
      {
        InterpolateOverlap_4x(pMVsIntOvlpPlanesArrays[k], _mv_clip_arr[k]._clip_sptr->GetpMVsArray(0), k);
        if (mvmultivs != 0)
          InterpolateOverlap_4x(pMVsIntOvlpPlanesArraysVS[k], _mv_clip_arr[k]._clipvs_sptr->GetpMVsArray(0), k);
      }
      else if ((iInterpolateOverlap == 3) || (iInterpolateOverlap == 4))
      {
        InterpolateOverlap_2x(pMVsIntOvlpPlanesArrays[k], _mv_clip_arr[k]._clip_sptr->GetpMVsArray(0), k);
        if (mvmultivs != 0)
          InterpolateOverlap_2x(pMVsIntOvlpPlanesArraysVS[k], _mv_clip_arr[k]._clipvs_sptr->GetpMVsArray(0), k);
      }
      pMVsWorkPlanesArrays[k] = pMVsIntOvlpPlanesArrays[k];
      if (mvmultivs != 0)
        pMVsPlanesArraysVS[k] = pMVsIntOvlpPlanesArraysVS[k];
    }
  }
  else
  {
    for (int k = 0; k < _trad * 2; ++k)
    {
//      pMVsPlanesArrays[k] = _mv_clip_arr[k]._clip_sptr->GetpMVsArray(0);
//      pMVsWorkPlanesArrays[k] = (VECTOR*)pMVsPlanesArrays[k];
      pMVsWorkPlanesArrays[k] = (VECTOR*)_mv_clip_arr[k]._clip_sptr->GetpMVsArray(0);

      if (mvmultivs != 0)
        pMVsPlanesArraysVS[k] = _mv_clip_arr[k]._clipvs_sptr->GetpMVsArray(0);
    }
  }

  //call Filter MVs here because it equal for luma and all chroma planes
//  const BYTE* pSrcCur = _src_ptr_arr[0] + td._y_beg * rowsize * _src_pitch_arr[0]; // P.F. why *rowsize? (*nBlkSizeY)

  bYUVProc = (_planes_ptr[0][1] != 0) && (_planes_ptr[0][2] != 0);// colour planes exist, use single pass YUV proc for colour formats with colour processing

   // check if auto-thSAD required
  if ((thSADA_a != 0) || (thSADA_b != 0))
    CalcAutothSADs();

    // it is currently faster to call once because of interconnectin of Y+UV via chroma blocks SADs,
  // will be faster with per-block processing may be only in the combined Y+UV colour data processing (possibly).
  if (bMVsAddProc && !bYUVProc) // if interpolate overlap - may be it is better (and definitely faster) to make with input non-overlapped MVs ?
  {
    FilterMVs();
  }
  // TEST with use_block_yuv


  PROFILE_START(MOTION_PROFILE_COMPENSATION);

  //-------------------------------------------------------------------------
  // LUMA plane Y

  if ((_yuvplanes & YPLANE) == 0)
  {
    if (_out16_flag) {
      // copy 8 bit source to 16bit target
      plane_copy_8_to_16_c(_dst_ptr_arr[0], _dst_pitch_arr[0],
        _src_ptr_arr[0], _src_pitch_arr[0],
        nWidth, nHeight);
    }
    else {
      BitBlt(
        _dst_ptr_arr[0], _dst_pitch_arr[0],
        _src_ptr_arr[0], _src_pitch_arr[0],
        nWidth << pixelsize_super_shift, nHeight
      );
    }
  }
  else
  {
    Slicer slicer(_mt_flag);

    if (nOverlapX == 0 && nOverlapY == 0)
    {
      {
        if (bYUVProc) // YUV planes all present, single pass all 3 planes process
        {
          slicer.start(
            nBlkY,
            *this,
            &MDegrainN::process_luma_and_chroma_normal_slice 
          );

          // finish and return !
          if (_nlimit < 255)
          {
            nlimit_luma();
          }

          if (_nlimitc < 255)
          {
            nlimit_chroma(1);
            nlimit_chroma(2);
          }

#ifndef _M_X64 
          _mm_empty(); // (we may use double-float somewhere) Fizick
#endif

          PROFILE_STOP(MOTION_PROFILE_COMPENSATION);

          if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !_planar_flag)
          {
            YUY2FromPlanes(
              pDstYUY2, nDstPitchYUY2, nWidth, nHeight * _height_lsb_or_out16_mul,
              _dst_ptr_arr[0], _dst_pitch_arr[0],
              _dst_ptr_arr[1], _dst_ptr_arr[2], _dst_pitch_arr[1], _cpuFlags);
          }

          return (dst); // here is end of YUV single pass proc and GetFrame additional return

        }
        else
        {
          slicer.start(
            nBlkY,
            *this,
            &MDegrainN::process_luma_normal_slice // Y plane only
          );
        }
      }
      slicer.wait();
    }

    // Overlap
    else
    {
      if (bYUVProc) // single pass YUV proc overlap
      {
        // luma
        uint16_t* pDstShort = (_dst_short.empty()) ? 0 : &_dst_short[0];
        int* pDstInt = (_dst_int.empty()) ? 0 : &_dst_int[0];
        MemZoneSetY(pDstShort, pDstInt);

        // chroma plane 1
        uint16_t* pDstShortUV = (_dst_shortUV1.empty()) ? 0 : &_dst_shortUV1[0];
        int* pDstIntUV = (_dst_intUV1.empty()) ? 0 : &_dst_intUV1[0];
        MemZoneSetUV(pDstShortUV, pDstIntUV);

        //chroma plane 2
        pDstShortUV = (_dst_shortUV2.empty()) ? 0 : &_dst_shortUV2[0];
        pDstIntUV = (_dst_intUV2.empty()) ? 0 : &_dst_intUV2[0];
        MemZoneSetUV(pDstShortUV, pDstIntUV);
        
        if (nOverlapY > 0)
        {
          memset(
            &_boundary_cnt_arr[0],
            0,
            _boundary_cnt_arr.size() * sizeof(_boundary_cnt_arr[0])
          );
        }

        slicer.start(
          nBlkY,
          *this,
          &MDegrainN::process_luma_and_chroma_overlap_slice,
          2
        );
        slicer.wait();

        // luma
        post_overlap_luma_plane();

        // chroma 1
        pDstShortUV = (_dst_shortUV1.empty()) ? 0 : &_dst_shortUV1[0];
        pDstIntUV = (_dst_intUV1.empty()) ? 0 : &_dst_intUV1[0];
        post_overlap_chroma_plane(1, pDstShortUV, pDstIntUV);

        // chroma 2
        pDstShortUV = (_dst_shortUV2.empty()) ? 0 : &_dst_shortUV2[0];
        pDstIntUV = (_dst_intUV2.empty()) ? 0 : &_dst_intUV2[0];
        post_overlap_chroma_plane(2, pDstShortUV, pDstIntUV);

        if (_nlimit < 255)
        {
          nlimit_luma();
        }

        if (_nlimitc < 255)
        {
          nlimit_chroma(1);
          nlimit_chroma(2);
        }

#ifndef _M_X64 
        _mm_empty(); // (we may use double-float somewhere) Fizick
#endif

        PROFILE_STOP(MOTION_PROFILE_COMPENSATION);

        if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !_planar_flag)
        {
          YUY2FromPlanes(
            pDstYUY2, nDstPitchYUY2, nWidth, nHeight * _height_lsb_or_out16_mul,
            _dst_ptr_arr[0], _dst_pitch_arr[0],
            _dst_ptr_arr[1], _dst_ptr_arr[2], _dst_pitch_arr[1], _cpuFlags);
        }

        return (dst); // here is end of YUV single pass proc and GetFrame additional return

      }
      else
      {
        uint16_t* pDstShort = (_dst_short.empty()) ? 0 : &_dst_short[0];
        int* pDstInt = (_dst_int.empty()) ? 0 : &_dst_int[0];
        MemZoneSetY(pDstShort, pDstInt);

        if (nOverlapY > 0)
        {
          memset(
            &_boundary_cnt_arr[0],
            0,
            _boundary_cnt_arr.size() * sizeof(_boundary_cnt_arr[0])
          );
        }

        slicer.start(
          nBlkY,
          *this,
          &MDegrainN::process_luma_overlap_slice,
          2
        );
        slicer.wait();

        post_overlap_luma_plane();

      } // !bYUVProc - old separated planes proc overlap
    }	// overlap - end

    if (_nlimit < 255)
    {
      nlimit_luma();
    }
  }

  //-------------------------------------------------------------------------
  // CHROMA planes

  process_chroma <1>(UPLANE & _nsupermodeyuv);
  process_chroma <2>(VPLANE & _nsupermodeyuv);

  //-------------------------------------------------------------------------

#ifndef _M_X64 
  _mm_empty(); // (we may use double-float somewhere) Fizick
#endif

  PROFILE_STOP(MOTION_PROFILE_COMPENSATION);

  if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !_planar_flag)
  {
    YUY2FromPlanes(
      pDstYUY2, nDstPitchYUY2, nWidth, nHeight * _height_lsb_or_out16_mul,
      _dst_ptr_arr[0], _dst_pitch_arr[0],
      _dst_ptr_arr[1], _dst_ptr_arr[2], _dst_pitch_arr[1], _cpuFlags);
  }

  return (dst);
}



// Fn...F1 B1...Bn
int MDegrainN::reorder_ref(int index) const
{
  assert(index >= 0);
  assert(index < _trad * 2);

  const int k = (index < _trad)
    ? (_trad - index) * 2 - 1
    : (index - _trad) * 2;

  return (k);
}



template <int P>
void	MDegrainN::process_chroma(int plane_mask)
{
  if ((_yuvplanes & plane_mask) == 0)
  {
    if (_out16_flag) {
      // copy 8 bit source to 16bit target
      plane_copy_8_to_16_c(_dst_ptr_arr[P], _dst_pitch_arr[P],
        _src_ptr_arr[P], _src_pitch_arr[P],
        nWidth >> nLogxRatioUV_super, nHeight >> nLogyRatioUV_super
      );
    }
    else {
      BitBlt(
        _dst_ptr_arr[P], _dst_pitch_arr[P],
        _src_ptr_arr[P], _src_pitch_arr[P],
        (nWidth >> nLogxRatioUV_super) << pixelsize_super_shift, nHeight >> nLogyRatioUV_super
      );
    }
  }

  else
  {
    Slicer slicer(_mt_flag);

    if (nOverlapX == 0 && nOverlapY == 0)
    {
      slicer.start(
        nBlkY,
        *this,
        &MDegrainN::process_chroma_normal_slice <P>
      );
      slicer.wait();
    }

    // Overlap
    else
    {
      uint16_t * pDstShort = (_dst_short.empty()) ? 0 : &_dst_short[0];
      int * pDstInt = (_dst_int.empty()) ? 0 : &_dst_int[0];
      MemZoneSetUV(pDstShort, pDstInt);

      if (nOverlapY > 0)
      {
        memset(
          &_boundary_cnt_arr[0],
          0,
          _boundary_cnt_arr.size() * sizeof(_boundary_cnt_arr[0])
        );
      }

      slicer.start(
        nBlkY,
        *this,
        &MDegrainN::process_chroma_overlap_slice <P>,
        2
      );
      slicer.wait();
      
      if (_lsb_flag)
      {
        Short2BytesLsb(
          _dst_ptr_arr[P],
          _dst_ptr_arr[P] + _lsb_offset_arr[P], // 8 bit only
          _dst_pitch_arr[P],
          &_dst_int[0], _dst_int_pitch,
          _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
        );
      }
      else if (_out16_flag)
      {
        Short2Bytes_Int32toWord16(
          (uint16_t *)_dst_ptr_arr[P], _dst_pitch_arr[P],
          &_dst_int[0], _dst_int_pitch,
          _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
          bits_per_pixel_output
        );
      }
      else if (pixelsize_super == 1)
      {
        Short2Bytes(
          _dst_ptr_arr[P], _dst_pitch_arr[P],
          &_dst_short[0], _dst_short_pitch,
          _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
        );
      }
      else if (pixelsize_super == 2)
      {
        Short2Bytes_Int32toWord16(
          (uint16_t *)_dst_ptr_arr[P], _dst_pitch_arr[P],
          &_dst_int[0], _dst_int_pitch,
          _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
          bits_per_pixel_super
        );
      }
      else if (pixelsize_super == 4)
      {
        Short2Bytes_FloatInInt32ArrayToFloat(
          (float *)_dst_ptr_arr[P], _dst_pitch_arr[P],
          (float *)&_dst_int[0], _dst_int_pitch,
          _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
        );
      }

      if (_covered_width < nWidth)
      {
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(_dst_ptr_arr[P] + ((_covered_width >> nLogxRatioUV_super) << pixelsize_output_shift), _dst_pitch_arr[P],
            _src_ptr_arr[P] + (_covered_width >> nLogxRatioUV_super), _src_pitch_arr[P],
            (nWidth - _covered_width) >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
          );
        }
        else {
          BitBlt(
            _dst_ptr_arr[P] + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _dst_pitch_arr[P],
            _src_ptr_arr[P] + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[P],
            ((nWidth - _covered_width) >> nLogxRatioUV_super) << pixelsize_super_shift, _covered_height >> nLogyRatioUV_super
          );
        }
      }
      if (_covered_height < nHeight) // bottom noncovered region
      {
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(_dst_ptr_arr[P] + ((_dst_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _dst_pitch_arr[P],
            _src_ptr_arr[P] + ((_src_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _src_pitch_arr[P],
            nWidth >> nLogxRatioUV_super, ((nHeight - _covered_height) >> nLogyRatioUV_super)
          );
        }
        else {
          BitBlt(
            _dst_ptr_arr[P] + ((_dst_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _dst_pitch_arr[P],
            _src_ptr_arr[P] + ((_src_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _src_pitch_arr[P],
            (nWidth >> nLogxRatioUV_super) << pixelsize_super_shift, ((nHeight - _covered_height) >> nLogyRatioUV_super)
          );
        }
      }
    } // overlap - end

    if (_nlimitc < 255)
    {
      // limit is 0-255 relative, for any bit depth
      float realLimit;
      if (pixelsize_output <= 2)
        realLimit = _nlimitc * (1 << (bits_per_pixel_output - 8));
      else
        realLimit = (float)_nlimitc / 255.0f;
      LimitFunction(_dst_ptr_arr[P], _dst_pitch_arr[P],
        _src_ptr_arr[P], _src_pitch_arr[P],
        nWidth >> nLogxRatioUV_super, nHeight >> nLogyRatioUV_super,
        realLimit
      );
    }
  }
}

void	MDegrainN::process_luma_normal_slice(Slicer::TaskData &td)
{
  assert(&td != 0);

  const int rowsize = nBlkSizeY;
  BYTE *pDstCur = _dst_ptr_arr[0] + td._y_beg * rowsize * _dst_pitch_arr[0]; // P.F. why *rowsize? (*nBlkSizeY)
  const BYTE *pSrcCur = _src_ptr_arr[0] + td._y_beg * rowsize * _src_pitch_arr[0]; // P.F. why *rowsize? (*nBlkSizeY)

  for (int by = td._y_beg; by < td._y_end; ++by)
  {
    int xx = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array

    // prefetch source full row in linear lines reading
    for (int iH = 0; iH < nBlkSizeY; ++iH)
    {
      HWprefetch_T1((char*)pSrcCur + _src_pitch_arr[0] * iH, nBlkX * nBlkSizeX);
    }

    for (int bx = 0; bx < nBlkX; ++bx)
    {

      int i = by * nBlkX + bx;
      const BYTE *ref_data_ptr_arr[MAX_TEMP_RAD * 2];
      int pitch_arr[MAX_TEMP_RAD * 2];
      int weight_arr[1 + MAX_TEMP_RAD * 2];

      PrefetchMVs(i);

      for (int k = 0; k < _trad * 2; ++k)
      {
        if (!bMVsAddProc)
        {
          (this->*use_block_y_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][0],
            pSrcCur,
            xx << pixelsize_super_shift,
            _src_pitch_arr[0],
            bx,
            by,
//            pMVsPlanesArrays[k]
              pMVsWorkPlanesArrays[k]
            );
        }
        else
        {
          (this->*use_block_y_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][0],
            pSrcCur,
            xx << pixelsize_super_shift,
            _src_pitch_arr[0],
            bx,
            by,
            (const VECTOR*)pFilteredMVsPlanesArrays[k]
            );
        }
      }

      if (dn_mm == DN_MM_BLOCKS)
      {
        int iDN_MM_Weight = 255 - pDNMask[by * dnmask_pitch + bx]; // invert mask - 255 is zero refs weight - no denoise 
        apply_dn_mask_weights(weight_arr, _trad, iDN_MM_Weight);
      }

      norm_weights(weight_arr, _trad);

      // luma
      if (MPBNumIt == 0 || !isMVsStable(pMVsWorkPlanesArrays, i, weight_arr))
      _degrainluma_ptr(
        pDstCur + (xx << pixelsize_output_shift), pDstCur + _lsb_offset_arr[0] + (xx << pixelsize_super_shift), _dst_pitch_arr[0],
        pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
        ref_data_ptr_arr, pitch_arr, weight_arr, _trad
      );
      else
      {
        MPB_SP(pDstCur + (xx << pixelsize_output_shift), pDstCur + _lsb_offset_arr[0] + (xx << pixelsize_super_shift), _dst_pitch_arr[0],
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
          ref_data_ptr_arr, pitch_arr, weight_arr, nBlkSizeX, nBlkSizeY, false, i);
/*        int iNumItCurr = MPBNumIt;
        do
        {
          // initial blend or each iteration blend
          _degrainluma_ptr(
            pMPBTempBlocks, 0, (nBlkSizeX * pixelsize),
            pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
            ref_data_ptr_arr, pitch_arr, weight_arr, _trad
          );
        
          int iNumAlignedBlocks = AlignBlockWeights(
            ref_data_ptr_arr, pitch_arr,
            pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
            weight_arr, nBlkSizeX, nBlkSizeY, false
          );

          if ((iNumAlignedBlocks == 0) || (iNumItCurr < 0))
          {
            // final output blend
            if (_lsb_flag || iNumAlignedBlocks != 0) // make full blend (with lsb) again
            {
              _degrainluma_ptr(
                pDstCur + (xx << pixelsize_output_shift), pDstCur + _lsb_offset_arr[0] + (xx << pixelsize_super_shift), _dst_pitch_arr[0],
                pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
                ref_data_ptr_arr, pitch_arr, weight_arr, _trad
              );
            }
            else // simply copy current blended block
            {
              CopyBlock(pDstCur + (xx << pixelsize_output_shift), _dst_pitch_arr[0], pMPBTempBlocks, nBlkSizeX, nBlkSizeY);
            }
            break;
          }

          iNumItCurr--;

        } while (1);
        */
      }
      
      xx += (nBlkSizeX); // xx: indexing offset

      if (bx == nBlkX - 1 && _covered_width < nWidth) // right non-covered region
      {
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(
            pDstCur + (_covered_width << pixelsize_output_shift), _dst_pitch_arr[0],
            pSrcCur + (_covered_width << pixelsize_super_shift), _src_pitch_arr[0],
            nWidth - _covered_width, nBlkSizeY
          );
        }
        else {
          // luma
          BitBlt(
            pDstCur + (_covered_width << pixelsize_super_shift), _dst_pitch_arr[0],
            pSrcCur + (_covered_width << pixelsize_super_shift), _src_pitch_arr[0],
            (nWidth - _covered_width) << pixelsize_super_shift, nBlkSizeY);
        }
      }
    }	// for bx

    pDstCur += rowsize * _dst_pitch_arr[0];
    pSrcCur += rowsize * _src_pitch_arr[0];

    if (by == nBlkY - 1 && _covered_height < nHeight) // bottom uncovered region
    {
      // luma
      if (_out16_flag) {
        // copy 8 bit source to 16bit target
        plane_copy_8_to_16_c(
          pDstCur, _dst_pitch_arr[0],
          pSrcCur, _src_pitch_arr[0],
          nWidth, nHeight - _covered_height
        );
      }
      else {
        BitBlt(
          pDstCur, _dst_pitch_arr[0],
          pSrcCur, _src_pitch_arr[0],
          nWidth << pixelsize_super_shift, nHeight - _covered_height
        );
      }
    }
  }	// for by

}

void	MDegrainN::process_luma_overlap_slice(Slicer::TaskData &td)
{
  assert(&td != 0);

  if (nOverlapY == 0
    || (td._y_beg == 0 && td._y_end == nBlkY))
  {
    process_luma_overlap_slice(td._y_beg, td._y_end);
  }

  else
  {
    assert(td._y_end - td._y_beg >= 2);

    process_luma_overlap_slice(td._y_beg, td._y_end - 1);

    const conc::AioAdd <int>	inc_ftor(+1);

    const int cnt_top = conc::AtomicIntOp::exec_new(
      _boundary_cnt_arr[td._y_beg],
      inc_ftor
    );
    if (td._y_beg > 0 && cnt_top == 2)
    {
      process_luma_overlap_slice(td._y_beg - 1, td._y_beg);
    }

    int cnt_bot = 2;
    if (td._y_end < nBlkY)
    {
      cnt_bot = conc::AtomicIntOp::exec_new(
        _boundary_cnt_arr[td._y_end],
        inc_ftor
      );
    }
    if (cnt_bot == 2)
    {
      process_luma_overlap_slice(td._y_end - 1, td._y_end);
    }
  }
}

void	MDegrainN::process_luma_and_chroma_overlap_slice(Slicer::TaskData& td)
{
  assert(&td != 0);

  if (nOverlapY == 0
    || (td._y_beg == 0 && td._y_end == nBlkY))
  {
    process_luma_and_chroma_overlap_slice(td._y_beg, td._y_end);
  }

  else
  {
    assert(td._y_end - td._y_beg >= 2);

    process_luma_and_chroma_overlap_slice(td._y_beg, td._y_end - 1);

    const conc::AioAdd <int>	inc_ftor(+1);

    const int cnt_top = conc::AtomicIntOp::exec_new(
      _boundary_cnt_arr[td._y_beg],
      inc_ftor
    );
    if (td._y_beg > 0 && cnt_top == 2)
    {
      process_luma_and_chroma_overlap_slice(td._y_beg - 1, td._y_beg);
    }

    int cnt_bot = 2;
    if (td._y_end < nBlkY)
    {
      cnt_bot = conc::AtomicIntOp::exec_new(
        _boundary_cnt_arr[td._y_end],
        inc_ftor
      );
    }
    if (cnt_bot == 2)
    {
      process_luma_and_chroma_overlap_slice(td._y_end - 1, td._y_end);
    }
  }
}



void	MDegrainN::process_luma_overlap_slice(int y_beg, int y_end)
{
  TmpBlock       tmp_block;

  const int      rowsize = nBlkSizeY - nOverlapY;
  const BYTE *   pSrcCur = _src_ptr_arr[0] + y_beg * rowsize * _src_pitch_arr[0];

  uint16_t * pDstShort = (_dst_short.empty()) ? 0 : &_dst_short[0] + y_beg * rowsize * _dst_short_pitch;
  int *pDstInt = (_dst_int.empty()) ? 0 : &_dst_int[0] + y_beg * rowsize * _dst_int_pitch;
  const int tmpPitch = nBlkSizeX;
  assert(tmpPitch <= TmpBlock::MAX_SIZE);

  for (int by = y_beg; by < y_end; ++by)
  {
    // indexing overlap windows weighting table: top=0 middle=3 bottom=6
    /*
    0 = Top Left    1 = Top Middle    2 = Top Right
    3 = Middle Left 4 = Middle Middle 5 = Middle Right
    6 = Bottom Left 7 = Bottom Middle 8 = Bottom Right
    */

    int wby = (by == 0) ? 0 * 3 : (by == nBlkY - 1) ? 2 * 3 : 1 * 3; // 0 for very first, 2*3 for very last, 1*3 for all others in the middle
    int xx = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array
    int ibxLast = nBlkX;
    if (bDiagOvlp)
    {
      if ((by % 2) != 0)
      {
        xx += nBlkSizeX / 2; // shift src start at odd lines
        ibxLast = nBlkX - 1; // not process last block/MV in odd rows
      }
    }
    
    // prefetch source full row in linear lines reading
    for (int iH = 0; iH < nBlkSizeY; ++iH)
    {
      HWprefetch_T1((char*)pSrcCur + _src_pitch_arr[0] * iH, nBlkX * nBlkSizeX);
    }

    for (int bx = 0; bx < ibxLast; ++bx)
    {
      // select window
      int wbx;
      // indexing overlap windows weighting table: left=+0 middle=+1 rightmost=+2
      if (bDiagOvlp && ((by % 2) != 0))
        wbx = 1; // all rows diagonally half blocksize shifted are internal
      else
        wbx = (bx == 0) ? 0 : (bx == nBlkX - 1) ? 2 : 1; // 0 for very first, 2 for very last, 1 for all others in the middle

      short *winOver = _overwins->GetWindow(wby + wbx);

      int i = by * nBlkX + bx;
      const BYTE *ref_data_ptr_arr[MAX_TEMP_RAD * 2];
      int pitch_arr[MAX_TEMP_RAD * 2];
      int weight_arr[1 + MAX_TEMP_RAD * 2];

      PrefetchMVs(i);

      for (int k = 0; k < _trad * 2; ++k)
      {
        if (!bMVsAddProc)
        {
          (this->*use_block_y_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][0],
            pSrcCur,
            xx << pixelsize_super_shift,
            _src_pitch_arr[0],
            bx,
            by,
//            pMVsPlanesArrays[k]
              pMVsWorkPlanesArrays[k]
            );
        }
        else
        {
          (this->*use_block_y_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][0],
            pSrcCur,
            xx << pixelsize_super_shift,
            _src_pitch_arr[0],
            bx,
            by,
            (const VECTOR*)pFilteredMVsPlanesArrays[k]
            );
        }
      }

      if (dn_mm == DN_MM_BLOCKS)
      {
        int iDN_MM_Weight = 255 - pDNMask[by * dnmask_pitch + bx]; // invert mask - 255 is zero refs weight - no denoise 
        apply_dn_mask_weights(weight_arr, _trad, iDN_MM_Weight);
      }

      norm_weights(weight_arr, _trad);

      // luma
/*      _degrainluma_ptr(
        &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
        pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
        ref_data_ptr_arr, pitch_arr, weight_arr, _trad
      );
      */
      if (MPBNumIt == 0 || !isMVsStable(pMVsWorkPlanesArrays, i, weight_arr))
      {
        _degrainluma_ptr(
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
          ref_data_ptr_arr, pitch_arr, weight_arr, _trad
        );
      }
      else
      {
        MPB_SP(&tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0],
          ref_data_ptr_arr, pitch_arr, weight_arr, nBlkSizeX, nBlkSizeY, false, i);
      }

      if (_lsb_flag)
      {
        _oversluma_lsb_ptr(
          pDstInt + xx, _dst_int_pitch,
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch,
          winOver, nBlkSizeX
        );
      }
      else if (_out16_flag)
      {
        // cast to match the prototype
        _oversluma16_ptr((uint16_t *)(pDstInt + xx), _dst_int_pitch, &tmp_block._d[0], tmpPitch << pixelsize_output_shift, winOver, nBlkSizeX);
      }
      else if (pixelsize_super == 1)
      {
        _oversluma_ptr(
          pDstShort + xx, _dst_short_pitch,
          &tmp_block._d[0], tmpPitch,
          winOver, nBlkSizeX
        );
      }
      else if (pixelsize_super == 2) {
        _oversluma16_ptr((uint16_t *)(pDstInt + xx), _dst_int_pitch, &tmp_block._d[0], tmpPitch << pixelsize_super_shift, winOver, nBlkSizeX);
      }
      else { // pixelsize_super == 4
        _oversluma32_ptr((uint16_t *)(pDstInt + xx), _dst_int_pitch, &tmp_block._d[0], tmpPitch << pixelsize_super_shift, winOver, nBlkSizeX);
      }

      xx += nBlkSizeX - nOverlapX;
    } // for bx

    pSrcCur += rowsize * _src_pitch_arr[0]; // byte pointer
    pDstShort += rowsize * _dst_short_pitch; // short pointer
    pDstInt += rowsize * _dst_int_pitch; // int pointer
  } // for by

}


//  To reuse subshifted blocks in MVPlane in both MVLPF and MDegrainN, also use single block weight calc 
void	MDegrainN::process_luma_and_chroma_normal_slice(Slicer::TaskData& td)
{
  assert(&td != 0);

  const int rowsize = nBlkSizeY;
  BYTE* pDstCur = _dst_ptr_arr[0] + td._y_beg * rowsize * _dst_pitch_arr[0]; // P.F. why *rowsize? (*nBlkSizeY)
  const BYTE* pSrcCur = _src_ptr_arr[0] + td._y_beg * rowsize * _src_pitch_arr[0]; // P.F. why *rowsize? (*nBlkSizeY)

  const int rowsizeUV = nBlkSizeY >> nLogyRatioUV_super; // bad name. it's height really
  BYTE* pDstCurUV1 = _dst_ptr_arr[1] + td._y_beg * rowsizeUV * _dst_pitch_arr[1];
  BYTE* pDstCurUV2 = _dst_ptr_arr[2] + td._y_beg * rowsizeUV * _dst_pitch_arr[2];
  const BYTE* pSrcCurUV1 = _src_ptr_arr[1] + td._y_beg * rowsizeUV * _src_pitch_arr[1];
  const BYTE* pSrcCurUV2 = _src_ptr_arr[2] + td._y_beg * rowsizeUV * _src_pitch_arr[2];

  int effective_nSrcPitchUV1 = (nBlkSizeY >> nLogyRatioUV_super)* _src_pitch_arr[1]; // pitch is byte granularity
  int effective_nDstPitchUV1 = (nBlkSizeY >> nLogyRatioUV_super)* _dst_pitch_arr[1]; // pitch is short granularity

  int effective_nSrcPitchUV2 = (nBlkSizeY >> nLogyRatioUV_super)* _src_pitch_arr[2]; // pitch is byte granularity
  int effective_nDstPitchUV2 = (nBlkSizeY >> nLogyRatioUV_super)* _dst_pitch_arr[2]; // pitch is short granularity

#ifdef _DEBUG
  if (pmode == PM_MEL)
  {
    iMEL_non_zero_blocks = 0;
    iMEL_mem_hits = 0;
    iMEL_mem_updates = 0;
  }
#endif

  for (int by = td._y_beg; by < td._y_end; ++by)
  {
    int xx = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array
    int xx_uv = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array

    // prefetch source full row in linear lines reading
    for (int iH = 0; iH < nBlkSizeY; ++iH)
    {
      HWprefetch_T1((char*)pSrcCur + _src_pitch_arr[0] * iH, nBlkX * nBlkSizeX);
    }

    for (int bx = 0; bx < nBlkX; ++bx)
    {

      int i = by * nBlkX + bx;

#ifdef _DEBUG
      if ((bx == 0) && (by == 0) && (pmode == PM_MEL))
      {
        int idbr = 0;
      }

      if (pmode == PM_MEL)
      {
        int idbr = 0;
      }
#endif
      PrefetchMVs(i);

      if (pmode == PM_BLEND)
      {
        DegrainBlendBlock_LC(
          pDstCur + (xx << pixelsize_output_shift), pDstCur +_lsb_offset_arr[0] + (xx << pixelsize_super_shift), _dst_pitch_arr[0],
          pSrcCur,
          pDstCurUV1 + (xx_uv << pixelsize_output_shift), pDstCurUV1 + (xx_uv << pixelsize_super_shift) + _lsb_offset_arr[1], _dst_pitch_arr[1],
          pSrcCurUV1,
          pDstCurUV2 + (xx_uv << pixelsize_output_shift), pDstCurUV2 + (xx_uv << pixelsize_super_shift) + _lsb_offset_arr[2], _dst_pitch_arr[2],
          pSrcCurUV2,
          i, bx, by, xx, xx_uv);

        if (iMGR > 0)
        {
          MGR_LC(
            pDstCur + (xx << pixelsize_output_shift), pDstCur + _lsb_offset_arr[0] + (xx << pixelsize_super_shift), _dst_pitch_arr[0],
            pSrcCur,
            pDstCurUV1 + (xx_uv << pixelsize_output_shift), pDstCurUV1 + (xx_uv << pixelsize_super_shift) + _lsb_offset_arr[1], _dst_pitch_arr[1],
            pSrcCurUV1,
            pDstCurUV2 + (xx_uv << pixelsize_output_shift), pDstCurUV2 + (xx_uv << pixelsize_super_shift) + _lsb_offset_arr[2], _dst_pitch_arr[2],
            pSrcCurUV2,
            i, bx, by, xx, xx_uv
          );
        }

      } // pmode BLEND end
      else
      {
        /*
            BYTE* pDstCur, int iDstPitch,
      const BYTE* pSrcCur,
      BYTE* pDstCurUV1, int iDstUV1Pitch,
      const BYTE* pSrcCurUV1,
      BYTE* pDstCurUV2, int iDstUV2Pitch,
      const BYTE* pSrcCurUV2,
      int xx, int xx_uv, int ibx, int iby, int iBlkNum
        */
        if (!_out16_flag)
        {
          MEL_LC(pDstCur + xx, _dst_pitch_arr[0],
            pSrcCur,
            pDstCurUV1 + xx_uv, _dst_pitch_arr[1],
            pSrcCurUV1,
            pDstCurUV2 + xx_uv, _dst_pitch_arr[2],
            pSrcCurUV2,
            xx, xx_uv, bx, by, i);
        }
        else
        {
          MEL_LC(pDstCur + (xx << pixelsize_output_shift), _dst_pitch_arr[0],
            pSrcCur,
            pDstCurUV1 + (xx_uv << pixelsize_output_shift), _dst_pitch_arr[1],
            pSrcCurUV1,
            pDstCurUV2 + (xx_uv << pixelsize_output_shift), _dst_pitch_arr[2],
            pSrcCurUV2,
            xx, xx_uv, bx, by, i);

        }
      } // pmode MEL select end

      xx += (nBlkSizeX); // xx: indexing offset
      xx_uv += (nBlkSizeX >> nLogxRatioUV_super); // xx: indexing offset

      if (bx == nBlkX - 1 && _covered_width < nWidth) // right non-covered region
      {
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(
            pDstCur + (_covered_width << pixelsize_output_shift), _dst_pitch_arr[0],
            pSrcCur + (_covered_width << pixelsize_super_shift), _src_pitch_arr[0],
            nWidth - _covered_width, nBlkSizeY
          );
        }
        else {
          // luma
          BitBlt(
            pDstCur + (_covered_width << pixelsize_super_shift), _dst_pitch_arr[0],
            pSrcCur + (_covered_width << pixelsize_super_shift), _src_pitch_arr[0],
            (nWidth - _covered_width) << pixelsize_super_shift, nBlkSizeY);
        }
      }

      // first chroma plane right non-covered
      if (bx == nBlkX - 1 && _covered_width < nWidth) // right non-covered region
      {
        // chroma
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(
            pDstCurUV1 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_output_shift), _dst_pitch_arr[1],
            pSrcCurUV1 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[1],
            (nWidth - _covered_width) >> nLogxRatioUV_super, rowsizeUV
          );
        }
        else {
          BitBlt(
            pDstCurUV1 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _dst_pitch_arr[1],
            pSrcCurUV1 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[1],
            ((nWidth - _covered_width) >> nLogxRatioUV_super) << pixelsize_super_shift, rowsizeUV
          );
        }
      }

      // second chroma plane right non-covered
      if (bx == nBlkX - 1 && _covered_width < nWidth) // right non-covered region
      {
        // chroma
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(
            pDstCurUV2 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_output_shift), _dst_pitch_arr[2],
            pSrcCurUV2 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[2],
            (nWidth - _covered_width) >> nLogxRatioUV_super, rowsizeUV
          );
        }
        else {
          BitBlt(
            pDstCurUV2 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _dst_pitch_arr[2],
            pSrcCurUV2 + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[2],
            ((nWidth - _covered_width) >> nLogxRatioUV_super) << pixelsize_super_shift, rowsizeUV
          );
        }
      }

    }	// for bx

    pDstCur += rowsize * _dst_pitch_arr[0];
    pSrcCur += rowsize * _src_pitch_arr[0];

    pDstCurUV1 += effective_nDstPitchUV1;
    pSrcCurUV1 += effective_nSrcPitchUV2;

    pDstCurUV2 += effective_nDstPitchUV2;
    pSrcCurUV2 += effective_nSrcPitchUV2;

    if (by == nBlkY - 1 && _covered_height < nHeight) // bottom uncovered region
    {
      // luma
      if (_out16_flag) {
        // copy 8 bit source to 16bit target
        plane_copy_8_to_16_c(
          pDstCur, _dst_pitch_arr[0],
          pSrcCur, _src_pitch_arr[0],
          nWidth, nHeight - _covered_height
        );
      }
      else {
        BitBlt(
          pDstCur, _dst_pitch_arr[0],
          pSrcCur, _src_pitch_arr[0],
          nWidth << pixelsize_super_shift, nHeight - _covered_height
        );
      }

      // chroma plane 1
      if (_out16_flag) {
        // copy 8 bit source to 16bit target
        plane_copy_8_to_16_c(
          pDstCurUV1, _dst_pitch_arr[1],
          pSrcCurUV1, _src_pitch_arr[1],
          nWidth >> nLogxRatioUV_super, (nHeight - _covered_height) >> nLogyRatioUV_super /* height */
        );
      }
      else {
        BitBlt(
          pDstCurUV1, _dst_pitch_arr[1],
          pSrcCurUV1, _src_pitch_arr[1],
          (nWidth >> nLogxRatioUV_super) << pixelsize_super_shift, (nHeight - _covered_height) >> nLogyRatioUV_super /* height */
        );
      }

      // chroma plane 2
      if (_out16_flag) {
        // copy 8 bit source to 16bit target
        plane_copy_8_to_16_c(
          pDstCurUV2, _dst_pitch_arr[2],
          pSrcCurUV2, _src_pitch_arr[2],
          nWidth >> nLogxRatioUV_super, (nHeight - _covered_height) >> nLogyRatioUV_super /* height */
        );
      }
      else {
        BitBlt(
          pDstCurUV2, _dst_pitch_arr[2],
          pSrcCurUV2, _src_pitch_arr[2],
          (nWidth >> nLogxRatioUV_super) << pixelsize_super_shift, (nHeight - _covered_height) >> nLogyRatioUV_super /* height */
        );
      }
    }
  }	// for by

#ifdef _DEBUG
  if (pmode == PM_MEL)
  {
    float fRatioMEL_nz_blocks = (float)iMEL_non_zero_blocks / (float)nBlkCount;
    float fRatioMEL_hits_blocks = (float)iMEL_mem_hits / (float)nBlkCount;
    float fRatioMEL_updates_blocks = (float)iMEL_mem_updates / (float)nBlkCount;
    int idbr = 0;
  }
#endif

}

void	MDegrainN::process_luma_and_chroma_overlap_slice(int y_beg, int y_end)
{
  //luma
  TmpBlock       tmp_block;

  const int      rowsize = nBlkSizeY - nOverlapY;
  const BYTE* pSrcCur = _src_ptr_arr[0] + y_beg * rowsize * _src_pitch_arr[0];

  uint16_t* pDstShort = (_dst_short.empty()) ? 0 : &_dst_short[0] + y_beg * rowsize * _dst_short_pitch;
  int* pDstInt = (_dst_int.empty()) ? 0 : &_dst_int[0] + y_beg * rowsize * _dst_int_pitch;
  const int tmpPitch = nBlkSizeX;
  assert(tmpPitch <= TmpBlock::MAX_SIZE);
  
  // chroma
 
  TmpBlock       tmp_blockUV1;
  TmpBlock       tmp_blockUV2;

  const int rowsizeUV = (nBlkSizeY - nOverlapY) >> nLogyRatioUV_super; // bad name. it's height really
  const BYTE* pSrcCurUV1 = _src_ptr_arr[1] + y_beg * rowsizeUV * _src_pitch_arr[1];
  const BYTE* pSrcCurUV2 = _src_ptr_arr[2] + y_beg * rowsizeUV * _src_pitch_arr[2];

  uint16_t* pDstShortUV1 = (_dst_shortUV1.empty()) ? 0 : &_dst_shortUV1[0] + y_beg * rowsizeUV * _dst_short_pitch;
  int* pDstIntUV1 = (_dst_intUV1.empty()) ? 0 : &_dst_intUV1[0] + y_beg * rowsizeUV * _dst_int_pitch;
  uint16_t* pDstShortUV2 = (_dst_shortUV2.empty()) ? 0 : &_dst_shortUV2[0] + y_beg * rowsizeUV * _dst_short_pitch;
  int* pDstIntUV2 = (_dst_intUV2.empty()) ? 0 : &_dst_intUV2[0] + y_beg * rowsizeUV * _dst_int_pitch;

  int effective_nSrcPitchUV1 = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _src_pitch_arr[1]; // pitch is byte granularity
  int effective_dstShortPitchUV1 = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _dst_short_pitch; // pitch is short granularity
  int effective_dstIntPitchUV1 = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _dst_int_pitch; // pitch is int granularity

  int effective_nSrcPitchUV2 = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _src_pitch_arr[2]; // pitch is byte granularity
  int effective_dstShortPitchUV2 = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _dst_short_pitch; // pitch is short granularity
  int effective_dstIntPitchUV2 = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _dst_int_pitch; // pitch is int granularity

  int iBlkScanDir;
  int iBlkxStart;

#ifdef _DEBUG
  iMEL_non_zero_blocks = 0;
  iMEL_mem_hits = 0;
  if (pmode == PM_MEL)
  {
    int idbr = 0;
  }
#endif

  for (int by = y_beg; by < y_end; ++by)
  {
    // indexing overlap windows weighting table: top=0 middle=3 bottom=6
    /*
    0 = Top Left    1 = Top Middle    2 = Top Right
    3 = Middle Left 4 = Middle Middle 5 = Middle Right
    6 = Bottom Left 7 = Bottom Middle 8 = Bottom Right
    */

    int wby = (by == 0) ? 0 * 3 : (by == nBlkY - 1) ? 2 * 3 : 1 * 3; // 0 for very first, 2*3 for very last, 1*3 for all others in the middle
    int xx = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array

    int xx_uv = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array

    int ibxLast = nBlkX;
    if (bDiagOvlp)
    {
      if ((by % 2) != 0)
      {
        xx += nBlkSizeX / 2; // shift src start at odd lines
        xx_uv += ((nBlkSizeX / 2) >> nLogxRatioUV_super);
        ibxLast = nBlkX - 1; // not process last block/MV in odd rows
      }
    }

    if (by % 2 == 0) // meander scan - left to right
    {
      iBlkScanDir = 1;
      iBlkxStart = 0;
    }
    else // right to left scan
    {
      iBlkScanDir = -1;
      iBlkxStart = ibxLast - 1;
      xx += (nBlkSizeX - nOverlapX) * iBlkxStart;
      xx_uv += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super) * iBlkxStart;
    }
    
    // prefetch source full row in linear lines reading
    for (int iH = 0; iH < nBlkSizeY; ++iH)
    {
      HWprefetch_T1((char*)pSrcCur + _src_pitch_arr[0] * iH, nBlkX * nBlkSizeX);
    }

    for (int iH = 0; iH < (nBlkSizeY >> nLogyRatioUV_super); ++iH)
    {
      HWprefetch_T1((char*)pSrcCurUV1 + _src_pitch_arr[1] * iH, nBlkX * (nBlkSizeX >> nLogxRatioUV_super));
      HWprefetch_T1((char*)pSrcCurUV2 + _src_pitch_arr[2] * iH, nBlkX * (nBlkSizeX >> nLogxRatioUV_super));
    }

//    for (int bx = 0; bx < ibxLast; ++bx)
    for (int ibx = 0; ibx < ibxLast; ++ibx)
    {
      int bx = iBlkxStart + ibx * iBlkScanDir;

      // select window
      int wbx;
      // indexing overlap windows weighting table: left=+0 middle=+1 rightmost=+2
      if (bDiagOvlp && ((by % 2) != 0))
        wbx = 1; // all rows diagonally half blocksize shifted are internal
      else
        wbx = (bx == 0) ? 0 : (bx == nBlkX - 1) ? 2 : 1; // 0 for very first, 2 for very last, 1 for all others in the middle

      short* winOver = _overwins->GetWindow(wby + wbx);
      short* winOverUV = _overwins_uv->GetWindow(wby + wbx);

      int i = by * nBlkX + bx;

      PrefetchMVs(i);

      if (pmode == PM_BLEND)
      {

        DegrainBlendBlock_LC(
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCur,
          &tmp_blockUV1._d[0], tmp_blockUV1._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCurUV1,
          &tmp_blockUV2._d[0], tmp_blockUV2._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCurUV2,
          i, bx, by, xx, xx_uv);

        if (iMGR > 0)
        {
          MGR_LC(
            &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
            pSrcCur,
            &tmp_blockUV1._d[0], tmp_blockUV1._lsb_ptr, tmpPitch << pixelsize_output_shift,
            pSrcCurUV1,
            &tmp_blockUV2._d[0], tmp_blockUV2._lsb_ptr, tmpPitch << pixelsize_output_shift,
            pSrcCurUV2,
            i, bx, by, xx, xx_uv);
        }

      } // if pmode == PM_BLEND end here
      else
      {
        MEL_LC(&tmp_block._d[0], tmpPitch << pixelsize_output_shift,
          pSrcCur,
          &tmp_blockUV1._d[0], tmpPitch << pixelsize_output_shift,
          pSrcCurUV1,
          &tmp_blockUV2._d[0], tmpPitch << pixelsize_output_shift,
          pSrcCurUV2,
          xx, xx_uv, bx, by, i);
      } // pmode MEL select end


      // chroma
      // currently use common preprocessed weight-arr or no MPB chroma
/*
        _degrainchroma_ptr(
          &tmp_blockUV1._d[0], tmp_blockUV1._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCurUV1 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[1],
          ref_data_ptr_arrUV1, pitch_arrUV1, weight_arrUV, _trad
        );

     // currently use common preprocessed weight_arr from Y or no MPB chroma
        _degrainchroma_ptr(
        &tmp_blockUV2._d[0], tmp_blockUV2._lsb_ptr, tmpPitch << pixelsize_output_shift,
        pSrcCurUV2 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[2],
        ref_data_ptr_arrUV2, pitch_arrUV2, weight_arrUV, _trad
      );
  */    

      // luma
      if (_lsb_flag)
      {
        _oversluma_lsb_ptr(
          pDstInt + xx, _dst_int_pitch,
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch,
          winOver, nBlkSizeX
        );
      }
      else if (_out16_flag)
      {
        // cast to match the prototype
        _oversluma16_ptr((uint16_t*)(pDstInt + xx), _dst_int_pitch, &tmp_block._d[0], tmpPitch << pixelsize_output_shift, winOver, nBlkSizeX);
      }
      else if (pixelsize_super == 1)
      {
        _oversluma_ptr(
          pDstShort + xx, _dst_short_pitch,
          &tmp_block._d[0], tmpPitch,
          winOver, nBlkSizeX
        );
      }
      else if (pixelsize_super == 2) {
        _oversluma16_ptr((uint16_t*)(pDstInt + xx), _dst_int_pitch, &tmp_block._d[0], tmpPitch << pixelsize_super_shift, winOver, nBlkSizeX);
      }
      else { // pixelsize_super == 4
        _oversluma32_ptr((uint16_t*)(pDstInt + xx), _dst_int_pitch, &tmp_block._d[0], tmpPitch << pixelsize_super_shift, winOver, nBlkSizeX);
      }

      // chroma 1
      if (_lsb_flag)
      {
        _overschroma_lsb_ptr(
          pDstIntUV1 + xx_uv, _dst_int_pitch,
          &tmp_blockUV1._d[0], tmp_blockUV1._lsb_ptr, tmpPitch,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super
        );
      }
      else if (_out16_flag)
      {
        // cast to match the prototype
        _overschroma16_ptr(
          (uint16_t*)(pDstIntUV1 + xx_uv), _dst_int_pitch,
          &tmp_blockUV1._d[0], tmpPitch << pixelsize_output_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else if (pixelsize_super == 1)
      {
        _overschroma_ptr(
          pDstShortUV1 + xx_uv, _dst_short_pitch,
          &tmp_blockUV1._d[0], tmpPitch,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else if (pixelsize_super == 2)
      {
        _overschroma16_ptr(
          (uint16_t*)(pDstIntUV1 + xx_uv), _dst_int_pitch,
          &tmp_blockUV1._d[0], tmpPitch << pixelsize_super_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else // if (pixelsize_super == 4)
      {
        _overschroma32_ptr(
          (uint16_t*)(pDstIntUV1 + xx_uv), _dst_int_pitch,
          &tmp_blockUV1._d[0], tmpPitch << pixelsize_super_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }

      // chroma 2
      if (_lsb_flag)
      {
        _overschroma_lsb_ptr(
          pDstIntUV2 + xx_uv, _dst_int_pitch,
          &tmp_blockUV2._d[0], tmp_blockUV2._lsb_ptr, tmpPitch,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super
        );
      }
      else if (_out16_flag)
      {
        // cast to match the prototype
        _overschroma16_ptr(
          (uint16_t*)(pDstIntUV2 + xx_uv), _dst_int_pitch,
          &tmp_blockUV2._d[0], tmpPitch << pixelsize_output_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else if (pixelsize_super == 1)
      {
        _overschroma_ptr(
          pDstShortUV2 + xx_uv, _dst_short_pitch,
          &tmp_blockUV2._d[0], tmpPitch,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else if (pixelsize_super == 2)
      {
        _overschroma16_ptr(
          (uint16_t*)(pDstIntUV2 + xx_uv), _dst_int_pitch,
          &tmp_blockUV2._d[0], tmpPitch << pixelsize_super_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else // if (pixelsize_super == 4)
      {
        _overschroma32_ptr(
          (uint16_t*)(pDstIntUV2 + xx_uv), _dst_int_pitch,
          &tmp_blockUV2._d[0], tmpPitch << pixelsize_super_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      
//      xx += nBlkSizeX - nOverlapX;
//      xx_uv += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super); // no pixelsize here
      xx += (nBlkSizeX - nOverlapX) * iBlkScanDir;
      xx_uv += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super) * iBlkScanDir; // no pixelsize here

    } // for bx

    pSrcCur += rowsize * _src_pitch_arr[0]; // byte pointer
    pDstShort += rowsize * _dst_short_pitch; // short pointer
    pDstInt += rowsize * _dst_int_pitch; // int pointer

    pSrcCurUV1 += effective_nSrcPitchUV1; // pitch is byte granularity
    pDstShortUV1 += effective_dstShortPitchUV1; // pitch is short granularity
    pDstIntUV1 += effective_dstIntPitchUV1; // pitch is int granularity

    pSrcCurUV2 += effective_nSrcPitchUV2; // pitch is byte granularity
    pDstShortUV2 += effective_dstShortPitchUV2; // pitch is short granularity
    pDstIntUV2 += effective_dstIntPitchUV2; // pitch is int granularity
  } // for by

#ifdef _DEBUG
  float fRatioMEL_nz_blocks = (float)iMEL_non_zero_blocks / (float)nBlkCount;
  float fRatioMEL_mem_blocks = (float)iMEL_mem_hits / (float)nBlkCount;
  int idbr = 0;
#endif

}



template <int P>
void	MDegrainN::process_chroma_normal_slice(Slicer::TaskData &td)
{
  assert(&td != 0);
  const int rowsize = nBlkSizeY >> nLogyRatioUV_super; // bad name. it's height really
  BYTE *pDstCur = _dst_ptr_arr[P] + td._y_beg * rowsize * _dst_pitch_arr[P];
  const BYTE *pSrcCur = _src_ptr_arr[P] + td._y_beg * rowsize * _src_pitch_arr[P];

  int effective_nSrcPitch = (nBlkSizeY >> nLogyRatioUV_super) * _src_pitch_arr[P]; // pitch is byte granularity
  int effective_nDstPitch = (nBlkSizeY >> nLogyRatioUV_super) * _dst_pitch_arr[P]; // pitch is short granularity

  for (int by = td._y_beg; by < td._y_end; ++by)
  {
    int xx = 0; // index

    /* error ?
    if (bDiagOvlp)
    {
      if ((by % 2) != 0) xx += ((nBlkSizeX / 2) >> nLogxRatioUV_super);
    }*/

    // prefetch source full row in linear lines reading
    for (int iH = 0; iH < (nBlkSizeY >> nLogyRatioUV_super); ++iH)
    {
      HWprefetch_T1((char*)pSrcCur + _src_pitch_arr[0] * iH, nBlkX * (nBlkSizeX >> nLogxRatioUV_super));
    }

    for (int bx = 0; bx < nBlkX; ++bx)
    {
      int i = by * nBlkX + bx;
      const BYTE *ref_data_ptr_arr[MAX_TEMP_RAD * 2]; // vs: const uint8_t *pointers[radius * 2]; // Moved by the degrain function. 
      int pitch_arr[MAX_TEMP_RAD * 2];
      int weight_arr[1 + MAX_TEMP_RAD * 2]; // 0th is special. vs:int WSrc, WRefs[radius * 2];

      for (int k = 0; k < _trad * 2; ++k)
      {
        if (!bMVsAddProc)
        {
          (this->*use_block_uv_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][P],
            pSrcCur,
            xx << pixelsize_super_shift, // the pointer increment inside knows that xx later here is incremented with nBlkSize and not nBlkSize>>_xRatioUV
                // todo: copy from MDegrainX. Here we shift, and incement with nBlkSize>>_xRatioUV
            _src_pitch_arr[P],
            bx,
            by,
//            pMVsPlanesArrays[k]
            pMVsWorkPlanesArrays[k]
            ); // vs: extra nLogPel, plane, xSubUV, ySubUV, thSAD
        }
        else
        {
          (this->*use_block_uv_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][P],
            pSrcCur,
            xx << pixelsize_super_shift, // the pointer increment inside knows that xx later here is incremented with nBlkSize and not nBlkSize>>_xRatioUV
                // todo: copy from MDegrainX. Here we shift, and incement with nBlkSize>>_xRatioUV
            _src_pitch_arr[P],
            bx,
            by,
            (const VECTOR*)pFilteredMVsPlanesArrays[k]
            ); // vs: extra nLogPel, plane, xSubUV, ySubUV, thSAD
        }
      }

      if (dn_mm == DN_MM_BLOCKS)
      {
        int iDN_MM_Weight = 255 - pDNMask[by * dnmask_pitch + bx]; // invert mask - 255 is zero refs weight - no denoise 
        apply_dn_mask_weights(weight_arr, _trad, iDN_MM_Weight);
      }

      norm_weights(weight_arr, _trad); // normaliseWeights<radius>(WSrc, WRefs);

      // chroma
/*      _degrainchroma_ptr(
        pDstCur + (xx << pixelsize_output_shift),
        pDstCur + (xx << pixelsize_super_shift) + _lsb_offset_arr[P], _dst_pitch_arr[P],
        pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[P],
        ref_data_ptr_arr, pitch_arr, weight_arr, _trad
      );
      */
      if (MPBNumIt == 0 || !isMVsStable(pMVsWorkPlanesArrays, i, weight_arr))
        _degrainchroma_ptr(
          pDstCur + (xx << pixelsize_output_shift),
          pDstCur + (xx << pixelsize_super_shift) + _lsb_offset_arr[P], _dst_pitch_arr[P],
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[P],
          ref_data_ptr_arr, pitch_arr, weight_arr, _trad
        );
      else
      {
        MPB_SP(pDstCur + (xx << pixelsize_output_shift),
          pDstCur + (xx << pixelsize_super_shift) + _lsb_offset_arr[P], _dst_pitch_arr[P],
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[P],
          ref_data_ptr_arr, pitch_arr, weight_arr, (nBlkSizeX >> nLogxRatioUV_super), (nBlkSizeY >> nLogxRatioUV_super), true, i);
      }

      //if (nLogxRatioUV != nLogxRatioUV_super) // orphaned if. chroma processing failed between 2.7.1-2.7.20
      //xx += nBlkSizeX; // blksize of Y plane, that's why there is xx >> xRatioUVlog above
      xx += (nBlkSizeX >> nLogxRatioUV_super); // xx: indexing offset

      if (bx == nBlkX - 1 && _covered_width < nWidth) // right non-covered region
      {
        // chroma
        if (_out16_flag) {
          // copy 8 bit source to 16bit target
          plane_copy_8_to_16_c(
            pDstCur + ((_covered_width >> nLogxRatioUV_super) << pixelsize_output_shift), _dst_pitch_arr[P],
            pSrcCur + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[P],
            (nWidth - _covered_width) >> nLogxRatioUV_super/* real row_size */, rowsize /* bad name. it's height = nBlkSizeY >> nLogyRatioUV_super*/
          );
        }
        else {
          BitBlt(
            pDstCur + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _dst_pitch_arr[P],
            pSrcCur + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[P],
            ((nWidth - _covered_width) >> nLogxRatioUV_super) << pixelsize_super_shift /* real row_size */, rowsize /* bad name. it's height = nBlkSizeY >> nLogyRatioUV_super*/
          );
        }
      }
    } // for bx

    pDstCur += effective_nDstPitch;
    pSrcCur += effective_nSrcPitch;

    if (by == nBlkY - 1 && _covered_height < nHeight) // bottom uncovered region
    {
      // chroma
      if (_out16_flag) {
        // copy 8 bit source to 16bit target
        plane_copy_8_to_16_c(
          pDstCur, _dst_pitch_arr[P],
          pSrcCur, _src_pitch_arr[P],
          nWidth >> nLogxRatioUV_super, (nHeight - _covered_height) >> nLogyRatioUV_super /* height */
        );
      }
      else {
        BitBlt(
          pDstCur, _dst_pitch_arr[P],
          pSrcCur, _src_pitch_arr[P],
          (nWidth >> nLogxRatioUV_super) << pixelsize_super_shift, (nHeight - _covered_height) >> nLogyRatioUV_super /* height */
        );
      }
    }
  } // for by

}



template <int P>
void	MDegrainN::process_chroma_overlap_slice(Slicer::TaskData &td)
{
  assert(&td != 0);

  if (nOverlapY == 0
    || (td._y_beg == 0 && td._y_end == nBlkY))
  {
    process_chroma_overlap_slice <P>(td._y_beg, td._y_end);
  }

  else
  {
    assert(td._y_end - td._y_beg >= 2);

    process_chroma_overlap_slice <P>(td._y_beg, td._y_end - 1);

    const conc::AioAdd <int> inc_ftor(+1);

    const int cnt_top = conc::AtomicIntOp::exec_new(
      _boundary_cnt_arr[td._y_beg],
      inc_ftor
    );
    if (td._y_beg > 0 && cnt_top == 2)
    {
      process_chroma_overlap_slice <P>(td._y_beg - 1, td._y_beg);
    }

    int				cnt_bot = 2;
    if (td._y_end < nBlkY)
    {
      cnt_bot = conc::AtomicIntOp::exec_new(
        _boundary_cnt_arr[td._y_end],
        inc_ftor
      );
    }
    if (cnt_bot == 2)
    {
      process_chroma_overlap_slice <P>(td._y_end - 1, td._y_end);
    }
  }
}



template <int P>
void	MDegrainN::process_chroma_overlap_slice(int y_beg, int y_end)
{
  TmpBlock       tmp_block;

  const int rowsize = (nBlkSizeY - nOverlapY) >> nLogyRatioUV_super; // bad name. it's height really
  const BYTE *pSrcCur = _src_ptr_arr[P] + y_beg * rowsize * _src_pitch_arr[P];

  uint16_t *pDstShort = (_dst_short.empty()) ? 0 : &_dst_short[0] + y_beg * rowsize * _dst_short_pitch;
  int *pDstInt = (_dst_int.empty()) ? 0 : &_dst_int[0] + y_beg * rowsize * _dst_int_pitch;
  const int tmpPitch = nBlkSizeX;
  assert(tmpPitch <= TmpBlock::MAX_SIZE);

  int effective_nSrcPitch = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super) * _src_pitch_arr[P]; // pitch is byte granularity
  int effective_dstShortPitch = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super) * _dst_short_pitch; // pitch is short granularity
  int effective_dstIntPitch = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super) * _dst_int_pitch; // pitch is int granularity

  for (int by = y_beg; by < y_end; ++by)
  {
    // indexing overlap windows weighting table: top=0 middle=3 bottom=6
    /*
    0 = Top Left    1 = Top Middle    2 = Top Right
    3 = Middle Left 4 = Middle Middle 5 = Middle Right
    6 = Bottom Left 7 = Bottom Middle 8 = Bottom Right
    */

    int wby = (by == 0) ? 0 * 3 : (by == nBlkY - 1) ? 2 * 3 : 1 * 3; // 0 for very first, 2*3 for very last, 1*3 for all others in the middle
    int xx = 0; // logical offset. Mul by 2 for pixelsize_super==2. Don't mul for indexing int* array

    int ibxLast = nBlkX;
    if (bDiagOvlp)
    {
      if ((by % 2) != 0)
      {
        xx += ((nBlkSizeX / 2) >> nLogxRatioUV_super); // shift src start at odd lines
        ibxLast = nBlkX - 1; // not process last block/MV in odd rows
      }
    }

    // prefetch source full row in linear lines reading
    for (int iH = 0; iH < (nBlkSizeY >> nLogyRatioUV_super); ++iH)
    {
      HWprefetch_T1((char*)pSrcCur + _src_pitch_arr[0] * iH, nBlkX * (nBlkSizeX >> nLogxRatioUV_super));
    }

    for (int bx = 0; bx < ibxLast; ++bx)
    {
      // select window
      int wbx;
      // indexing overlap windows weighting table: left=+0 middle=+1 rightmost=+2
      if (bDiagOvlp && ((by % 2) != 0))
        wbx = 1; // all rows diagonally half blocksize shifted are internal
      else
        wbx = (bx == 0) ? 0 : (bx == nBlkX - 1) ? 2 : 1; // 0 for very first, 2 for very last, 1 for all others in the middle

      short *winOverUV = _overwins_uv->GetWindow(wby + wbx);

      int i = by * nBlkX + bx;
      const BYTE *ref_data_ptr_arr[MAX_TEMP_RAD * 2];
      int pitch_arr[MAX_TEMP_RAD * 2];
      int weight_arr[1 + MAX_TEMP_RAD * 2]; // 0th is special

      for (int k = 0; k < _trad * 2; ++k)
      {
        if (!bMVsAddProc)
        {
          (this->*use_block_uv_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][P],
            pSrcCur,
            xx << pixelsize_super_shift, // the pointer increment inside knows that xx later here is incremented with nBlkSize and not nBlkSize>>_xRatioUV
                // todo: copy from MDegrainX. Here we shift, and incement with nBlkSize>>_xRatioUV
            _src_pitch_arr[P],
            bx,
            by,
//            pMVsPlanesArrays[k]
            pMVsWorkPlanesArrays[k]
            ); // vs: extra nLogPel, plane, xSubUV, ySubUV, thSAD
        }
        else
        {
          (this->*use_block_uv_func)(
            ref_data_ptr_arr[k],
            pitch_arr[k],
            weight_arr[k + 1],
            _usable_flag_arr[k],
            _mv_clip_arr[k],
            i,
            _planes_ptr[k][P],
            pSrcCur,
            xx << pixelsize_super_shift, // the pointer increment inside knows that xx later here is incremented with nBlkSize and not nBlkSize>>_xRatioUV
                // todo: copy from MDegrainX. Here we shift, and incement with nBlkSize>>_xRatioUV
            _src_pitch_arr[P],
            bx,
            by,
            (const VECTOR*)pFilteredMVsPlanesArrays[k]
            ); // vs: extra nLogPel, plane, xSubUV, ySubUV, thSAD
        }
      }

      if (dn_mm == DN_MM_BLOCKS)
      {
        int iDN_MM_Weight = 255 - pDNMask[by * dnmask_pitch + bx]; // invert mask - 255 is zero refs weight - no denoise 
        apply_dn_mask_weights(weight_arr, _trad, iDN_MM_Weight);
      }

      norm_weights(weight_arr, _trad); // 0th + 1..MAX_TEMP_RAD*2

      // chroma
      // here we don't pass pixelsize, because _degrainchroma_ptr points already to the uint16_t version
      // if the clip was 16 bit one
/*      _degrainchroma_ptr(
        &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
        pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[P],
        ref_data_ptr_arr, pitch_arr, weight_arr, _trad
      );*/
      if (MPBNumIt == 0 || !isMVsStable(pMVsWorkPlanesArrays, i, weight_arr))
      {
        _degrainchroma_ptr(
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[P],
          ref_data_ptr_arr, pitch_arr, weight_arr, _trad
        );
      }
      else
      {
        MPB_SP(&tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch << pixelsize_output_shift,
          pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[P],
          ref_data_ptr_arr, pitch_arr, weight_arr, (nBlkSizeX >> nLogxRatioUV_super), (nBlkSizeY >> nLogxRatioUV_super), true, i);
      }

      if (_lsb_flag)
      {
        _overschroma_lsb_ptr(
          pDstInt + xx, _dst_int_pitch,
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super
        );
      }
      else if (_out16_flag)
      {
        // cast to match the prototype
        _overschroma16_ptr(
          (uint16_t*)(pDstInt + xx), _dst_int_pitch,
          &tmp_block._d[0], tmpPitch << pixelsize_output_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else if (pixelsize_super == 1)
      {
        _overschroma_ptr(
          pDstShort + xx, _dst_short_pitch,
          &tmp_block._d[0], tmpPitch,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      } else if (pixelsize_super == 2)
      {
        _overschroma16_ptr(
          (uint16_t*)(pDstInt + xx), _dst_int_pitch, 
          &tmp_block._d[0], tmpPitch << pixelsize_super_shift, 
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }
      else // if (pixelsize_super == 4)
      {
        _overschroma32_ptr(
          (uint16_t*)(pDstInt + xx), _dst_int_pitch,
          &tmp_block._d[0], tmpPitch << pixelsize_super_shift,
          winOverUV, nBlkSizeX >> nLogxRatioUV_super);
      }

      xx += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super); // no pixelsize here

    } // for bx

    pSrcCur += effective_nSrcPitch; // pitch is byte granularity
    pDstShort += effective_dstShortPitch; // pitch is short granularity
    pDstInt += effective_dstIntPitch; // pitch is int granularity
  } // for by

}

#define ClipBlxBly \
if (blx < iMinBlx) blx = iMinBlx; \
if (bly < iMinBly) bly = iMinBly; \
if (blx > iMaxBlx) blx = iMaxBlx; \
if (bly > iMaxBly) bly = iMaxBly; 

#define Getblxbly \
if (!bDiagOvlp)\
 blx = ibx * (nBlkSizeX - nOverlapX) * nPel + pMVsArray[i].x;\
else\
{\
 if ((iby % 2) != 0)\
   blx = (ibx * nBlkSizeX + nBlkSizeX / 2) * nPel + pMVsArray[i].x;\
  else\
   blx = (ibx * nBlkSizeX) * nPel + pMVsArray[i].x;\
}\
bly = iby * (nBlkSizeY - nOverlapY) * nPel + pMVsArray[i].y;


MV_FORCEINLINE void	MDegrainN::use_block_y(
  const BYTE * &p, int &np, int &wref, bool usable_flag, const MvClipInfo &c_info,
  int i, const MVPlane *plane_ptr, const BYTE *src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
)
{
  if (usable_flag)
  {
    int blx;
    int bly;

/*    if (!bDiagOvlp)
    {
      blx = ibx * (nBlkSizeX - nOverlapX) * nPel + pMVsArray[i].x;
    }
    else
    {
      if ((iby % 2) != 0)
      {
        blx = (ibx * nBlkSizeX + nBlkSizeX / 2) * nPel + pMVsArray[i].x;
      }
      else
      {
        blx = (ibx * nBlkSizeX) * nPel + pMVsArray[i].x;
      }
    }
    bly = iby * (nBlkSizeY - nOverlapY) * nPel + pMVsArray[i].y;
    */
    Getblxbly

  // temp check - DX12_ME return invalid vectors sometime
     ClipBlxBly
    
     if (/*nPel != 1 && */nUseSubShift != 0)
     {
       MVPlane* p_plane = (MVPlane*)plane_ptr;
       p = p_plane->GetPointerSubShift(blx, bly, np);

#ifdef _DEBUG
 // FOR subshifting debug         
       const BYTE* pold = plane_ptr->GetPointer(blx, bly);
       int np_old = plane_ptr->GetPitch();

       int or0 = pold[0];
       int or1 = pold[1 * np_old];
       int or2 = pold[2 * np_old];
       int or3 = pold[3 * np_old];
       int or4 = pold[4 * np_old];
       int or5 = pold[5 * np_old];
       int or6 = pold[6 * np_old];
       int or7 = pold[7 * np_old];

       if (pixelsize == 1)
       {
         for (int x = 0; x < nBlkSizeX; x++)
         {
           for (int y = 0; y < nBlkSizeY; y++)
           {
             int isample = p[y * np + x];
             int isample_old = pold[y * np_old + x];

             if (abs(isample - isample_old) > 3)
             {
               int idbr = 0;
             }
           }
         }
       }
       else if (pixelsize == 2)
       {
         for (int x = 0; x < nBlkSizeX; x++)
         {
           for (int y = 0; y < nBlkSizeY; y++)
           {
             BYTE* puc = (BYTE*)p + y * np + x;
             unsigned short* pus = (unsigned short*)puc;

             BYTE* pucold = (BYTE*)pold + y * np_old + x;
             unsigned short* pusold = (unsigned short*)pucold;

             int isample = *pus;
             int isample_old = *pusold;

             if (abs(isample - isample_old) > 3)
             {
               int idbr = 0;
             }
           }
         }

       }
#endif     
     }
     else
     {
       p = plane_ptr->GetPointer(blx, bly);
       np = plane_ptr->GetPitch();
     }
    sad_t block_sad = pMVsArray[i].sad;

    wref = DegrainWeightN(c_info._thsad, c_info._thsad_sq, block_sad, _wpow);

  }
  else
  {
    p = src_ptr + xx;
    np = src_pitch;
    wref = 0;
  }
}

MV_FORCEINLINE void MDegrainN::use_block_yuv(const BYTE*& pY, int& npY, const BYTE*& pUV1, int& npUV1, const BYTE*& pUV2, int& npUV2, int& wref, int& wrefUV, bool usable_flag,
  const MvClipInfo& c_info, int i, const MVPlane* plane_ptrY, const BYTE* src_ptrY, const MVPlane* plane_ptrUV1, const BYTE* src_ptrUV1, const MVPlane* plane_ptrUV2, const BYTE* src_ptrUV2,
  int xx, int xx_uv, int src_pitchY, int src_pitchUV1, int src_pitchUV2, int ibx, int iby, const VECTOR* pMVsArray)
{
  if (usable_flag)
  {
    int blx;
    int bly;

    Getblxbly

    // temp check - DX12_ME return invalid vectors sometime
    ClipBlxBly

    if (/*nPel != 1 && */nUseSubShift != 0)
    {
      MVPlane* p_planeY = (MVPlane*)plane_ptrY;
      pY = p_planeY->GetPointerSubShift(blx, bly, npY);

      MVPlane* p_planeUV1 = (MVPlane*)plane_ptrUV1;

      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pUV1 = p_planeUV1->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npUV1);
//      pUV1 = p_planeUV1->GetPointerSubShiftUV(blx, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);

/*      pUV1 = p_planeUV1->GetPointerSubShiftUV(0, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);
      pUV1 = p_planeUV1->GetPointerSubShiftUV(1, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);
      pUV1 = p_planeUV1->GetPointerSubShiftUV(2, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);
      pUV1 = p_planeUV1->GetPointerSubShiftUV(3, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);
      pUV1 = p_planeUV1->GetPointerSubShiftUV(4, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);
  */    

      MVPlane* p_planeUV2 = (MVPlane*)plane_ptrUV2;
      pUV2 = p_planeUV2->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npUV2);
//      pUV2 = p_planeUV2->GetPointerSubShiftUV(blx, bly, npUV2, nLogxRatioUV_super, nLogyRatioUV_super);

    }
    else
    {
      pY = plane_ptrY->GetPointer(blx, bly);
      npY = plane_ptrY->GetPitch();

      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pUV1 = plane_ptrUV1->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npUV1 = plane_ptrUV1->GetPitch();

      pUV2 = plane_ptrUV2->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npUV2 = plane_ptrUV2->GetPitch();

    }

    sad_t block_sad = pMVsArray[i].sad;

    if ((fadjSADzeromv != 1.0f) || (fadjSADcohmv != 1.0f))
    {
      // pull SAD at static areas 
      if ((pMVsArray[i].x == 0) && (pMVsArray[i].y == 0))
      {
        block_sad = (sad_t)((float)block_sad * fadjSADzeromv);
      }
      else
      {
        if (ithCohMV >= 0) // skip long calc if ithCohV<0
        {
          // pull SAD at common motion blocks 
          int x_cur = pMVsArray[i].x, y_cur = pMVsArray[i].y;
          // upper block
          VECTOR v_upper, v_left, v_right, v_lower;
          int i_upper = i - nBlkX;
          if (i_upper < 0) i_upper = 0;
          v_upper = pMVsArray[i_upper];

          int i_left = i - 1;
          if (i_left < 0) i_left = 0;
          v_left = pMVsArray[i_left];

          int i_right = i + 1;
          if (i_right > nBlkX* nBlkY) i_right = nBlkX * nBlkY;
          v_right = pMVsArray[i_right];

          int i_lower = i + nBlkX;
          if (i_lower > nBlkX* nBlkY) i_lower = i;
          v_lower = pMVsArray[i_lower];

          int iabs_dc_x = SADABS(v_upper.x - x_cur) + SADABS(v_left.x - x_cur) + SADABS(v_right.x - x_cur) + SADABS(v_lower.x - x_cur);
          int iabs_dc_y = SADABS(v_upper.y - y_cur) + SADABS(v_left.y - y_cur) + SADABS(v_right.y - y_cur) + SADABS(v_lower.y - y_cur);

          if ((iabs_dc_x + iabs_dc_y) <= ithCohMV)
          {
            block_sad = (sad_t)((float)block_sad * fadjSADcohmv);
          }
        }
      }
    }

    wref = DegrainWeightN(c_info._thsad, c_info._thsad_sq, block_sad, _wpow);

    if (bthLC_diff) wrefUV = DegrainWeightN(c_info._thsadc, c_info._thsadc_sq, block_sad, _wpow);
  }
  else
  {
    pY = src_ptrY + xx;
    npY = src_pitchY;

    pUV1 = src_ptrUV1 + xx_uv;
    npUV1 = src_pitchUV1;

    pUV2 = src_ptrUV2 + xx_uv;
    npUV2 = src_pitchUV2;

    wref = 0;
    wrefUV = 0;
  }
}


MV_FORCEINLINE void MDegrainN::use_block_yuv_mel(const BYTE*& pY, int& npY, const BYTE*& pUV1, int& npUV1, const BYTE*& pUV2, int& npUV2, bool usable_flag,
  const MvClipInfo& c_info, int i, const MVPlane* plane_ptrY, const BYTE* src_ptrY, const MVPlane* plane_ptrUV1, const BYTE* src_ptrUV1, const MVPlane* plane_ptrUV2, const BYTE* src_ptrUV2,
  int xx, int xx_uv, int src_pitchY, int src_pitchUV1, int src_pitchUV2, int ibx, int iby, const VECTOR* pMVsArray)
{
  if (usable_flag) // motion compensate YUV 3 planes and initial DM check
  {
    int blx;
    int bly;

    Getblxbly

    // temp check - DX12_ME return invalid vectors sometime
    ClipBlxBly

    if (nUseSubShift != 0)
    {
      MVPlane* p_planeY = (MVPlane*)plane_ptrY;
      pY = p_planeY->GetPointerSubShift(blx, bly, npY);

      MVPlane* p_planeUV1 = (MVPlane*)plane_ptrUV1;

      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pUV1 = p_planeUV1->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npUV1);
      //      pUV1 = p_planeUV1->GetPointerSubShiftUV(blx, bly, npUV1, nLogxRatioUV_super, nLogyRatioUV_super);

      MVPlane* p_planeUV2 = (MVPlane*)plane_ptrUV2;
      pUV2 = p_planeUV2->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npUV2);
      //      pUV2 = p_planeUV2->GetPointerSubShiftUV(blx, bly, npUV2, nLogxRatioUV_super, nLogyRatioUV_super);

    }
    else
    {
      pY = plane_ptrY->GetPointer(blx, bly);
      npY = plane_ptrY->GetPitch();

      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pUV1 = plane_ptrUV1->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npUV1 = plane_ptrUV1->GetPitch();

      pUV2 = plane_ptrUV2->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npUV2 = plane_ptrUV2->GetPitch();

    }

    // check initial dismetric - if above thSAD - replace with current src block
    int idm_chroma = 0;
    if (TTH_chroma)
    {
      idm_chroma = ScaleSadChroma(DM_TTH_Chroma->GetDisMetric(src_ptrUV1 + xx, src_pitchUV1, pUV1, npUV1)
        + DM_TTH_Chroma->GetDisMetric(src_ptrUV2 + xx, src_pitchUV2, pUV2, npUV2), _mv_clip_arr[0]._clip_sptr->chromaSADScale);
    }
    int idm_luma = DM_TTH_Luma->GetDisMetric(src_ptrY + xx, src_pitchY, pY, npY);
    int idm = idm_luma + idm_chroma;

    if (idm > c_info._thsad) // use hard thresholding, set to current frame block
    {
      pY = src_ptrY + xx;
      npY = src_pitchY;

      pUV1 = src_ptrUV1 + xx_uv;
      npUV1 = src_pitchUV1;

      pUV2 = src_ptrUV2 + xx_uv;
      npUV2 = src_pitchUV2;
    }

  }
  else // set to current frame block
  {
    pY = src_ptrY + xx;
    npY = src_pitchY;

    pUV1 = src_ptrUV1 + xx_uv;
    npUV1 = src_pitchUV1;

    pUV2 = src_ptrUV2 + xx_uv;
    npUV2 = src_pitchUV2;
  }
}


MV_FORCEINLINE void	MDegrainN::use_block_y_thSADzeromv_thSADcohmv(
  const BYTE*& p, int& np, int& wref, bool usable_flag, const MvClipInfo& c_info,
  int i, const MVPlane* plane_ptr, const BYTE* src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
)
{
  if (usable_flag)
  {
    int blx;
    int bly;

    Getblxbly

    // temp check - DX12_ME return invalid vectors sometime
    ClipBlxBly

    if (nPel != 1 && nUseSubShift != 0)
    {
      MVPlane* p_plane = (MVPlane*)plane_ptr;
      p = p_plane->GetPointerSubShift(blx, bly, np);

    }
    else
    {
      p = plane_ptr->GetPointer(blx, bly);
      np = plane_ptr->GetPitch();
    }

    sad_t block_sad = pMVsArray[i].sad;

    // pull SAD at static areas 
    if ((pMVsArray[i].x == 0) && (pMVsArray[i].y == 0))
    {
      block_sad = (sad_t)((float)block_sad * fadjSADzeromv);
    }
    else
    {
      if (ithCohMV >= 0) // skip long calc if ithCohV<0
      {
        // pull SAD at common motion blocks 
        int x_cur = pMVsArray[i].x, y_cur = pMVsArray[i].y;
        // upper block
        VECTOR v_upper, v_left, v_right, v_lower;
        int i_upper = i - nBlkX;
        if (i_upper < 0) i_upper = 0;
        v_upper = pMVsArray[i_upper];

        int i_left = i - 1;
        if (i_left < 0) i_left = 0;
        v_left = pMVsArray[i_left];

        int i_right = i + 1;
        if (i_right > nBlkX* nBlkY) i_right = nBlkX * nBlkY;
        v_right = pMVsArray[i_right];

        int i_lower = i + nBlkX;
        if (i_lower > nBlkX* nBlkY) i_lower = i;
        v_lower = pMVsArray[i_lower];

        int iabs_dc_x = SADABS(v_upper.x - x_cur) + SADABS(v_left.x - x_cur) + SADABS(v_right.x - x_cur) + SADABS(v_lower.x - x_cur);
        int iabs_dc_y = SADABS(v_upper.y - y_cur) + SADABS(v_left.y - y_cur) + SADABS(v_right.y - y_cur) + SADABS(v_lower.y - y_cur);

        if ((iabs_dc_x + iabs_dc_y) <= ithCohMV)
        {
          block_sad = (sad_t)((float)block_sad * fadjSADcohmv);
        }
      }
    }

    wref = DegrainWeightN(c_info._thsad, c_info._thsad_sq, block_sad, _wpow);
  }
  else
  {
    p = src_ptr + xx;
    np = src_pitch;
    wref = 0;
  }
}

MV_FORCEINLINE void	MDegrainN::use_block_uv(
  const BYTE * &p, int &np, int &wref, bool usable_flag, const MvClipInfo &c_info,
  int i, const MVPlane *plane_ptr, const BYTE *src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
)
{
  if (usable_flag)
  {
    int blx;
    int bly;

    Getblxbly

     // temp check - DX12_ME return invalid vectors sometime
     ClipBlxBly

     if (/*nPel != 1 && */nUseSubShift != 0)
     {
       MVPlane* p_plane = (MVPlane*)plane_ptr;
       if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
       if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
       p = p_plane->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, np);
//       p = p_plane->GetPointerSubShiftUV(blx, bly, np, nLogxRatioUV_super, nLogyRatioUV_super);
     }
     else
     {
       if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
       if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
       p = plane_ptr->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
       np = plane_ptr->GetPitch();
     }

    sad_t block_sad = pMVsArray[i].sad;

    wref = DegrainWeightN(c_info._thsadc, c_info._thsadc_sq, block_sad, _wpow);
  }
  else
  {
    // just to have a valid data pointer, will not count, weight is zero
    p = src_ptr + xx; // done: kill  >> nLogxRatioUV_super from here and put it in the caller like in MDegrainX
    np = src_pitch;
    wref = 0;
  }
}

MV_FORCEINLINE void	MDegrainN::use_block_uv_thSADzeromv_thSADcohmv(
  const BYTE*& p, int& np, int& wref, bool usable_flag, const MvClipInfo& c_info,
  int i, const MVPlane* plane_ptr, const BYTE* src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
)
{
  if (usable_flag)
  {
    int blx;
    int bly;

    Getblxbly

    // temp check - DX12_ME return invalid vectors sometime
    ClipBlxBly

    if (/*nPel != 1 && */nUseSubShift != 0)
    {
      MVPlane* p_plane = (MVPlane*)plane_ptr;
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      p = p_plane->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, np);
//      p = p_plane->GetPointerSubShiftUV(blx, bly, np, nLogxRatioUV_super, nLogyRatioUV_super);
    }
    else
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      p = plane_ptr->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      np = plane_ptr->GetPitch();
    }

    sad_t block_sad = pMVsArray[i].sad;

    // pull SAD at static areas 
    if ((pMVsArray[i].x == 0) && (pMVsArray[i].y == 0))
    {
      block_sad = (sad_t)((float)block_sad * fadjSADzeromv);
    }
    else
    {
      if (ithCohMV >= 0) // skip long calc if ithCohV<0
      {
        // pull SAD at common motion blocks 
        int x_cur = pMVsArray[i].x, y_cur = pMVsArray[i].y;
        // upper block
        VECTOR v_upper, v_left, v_right, v_lower;
        int i_upper = i - nBlkX;
        if (i_upper < 0) i_upper = 0;
        v_upper = pMVsArray[i_upper];

        int i_left = i - 1;
        if (i_left < 0) i_left = 0;
        v_left = pMVsArray[i_left];

        int i_right = i + 1;
        if (i_right > nBlkX* nBlkY) i_right = nBlkX * nBlkY;
        v_right = pMVsArray[i_right];

        int i_lower = i + nBlkX;
        if (i_lower > nBlkX* nBlkY) i_lower = i;
        v_lower = pMVsArray[i_lower];

        int iabs_dc_x = SADABS(v_upper.x - x_cur) + SADABS(v_left.x - x_cur) + SADABS(v_right.x - x_cur) + SADABS(v_lower.x - x_cur);
        int iabs_dc_y = SADABS(v_upper.y - y_cur) + SADABS(v_left.y - y_cur) + SADABS(v_right.y - y_cur) + SADABS(v_lower.y - y_cur);

        if ((iabs_dc_x + iabs_dc_y) <= ithCohMV)
        {
          block_sad = (sad_t)((float)block_sad * fadjSADcohmv);
        }
      }
    }

    wref = DegrainWeightN(c_info._thsadc, c_info._thsadc_sq, block_sad, _wpow);
  }
  else
  {
    // just to have a valid data pointer, will not count, weight is zero
    p = src_ptr + xx; // done: kill  >> nLogxRatioUV_super from here and put it in the caller like in MDegrainX
    np = src_pitch;
    wref = 0;
  }
}

void MDegrainN::norm_weights(int wref_arr[], int trad)
{
  const int nbr_frames = trad * 2 + 1;

  const int one = 1 << DEGRAIN_WEIGHT_BITS; // 8 bit, 256

  wref_arr[0] = one;
  int wsum = 1;
  for (int k = 0; k < nbr_frames; ++k)
  {
    wsum += wref_arr[k];
  }

  // normalize weights to 256
  int wsrc = one;
  for (int k = 1; k < nbr_frames; ++k)
  {
    const int norm = wref_arr[k] * one / wsum;
    wref_arr[k] = norm;
    wsrc -= norm;
  }
  wref_arr[0] = wsrc;
}

MV_FORCEINLINE void MDegrainN::norm_weights_all(int wref_arr[], int trad)
{
  const int nbr_frames = trad * 2 + 1;

  const int one = 1 << DEGRAIN_WEIGHT_BITS; // 8 bit, 256

  int wsum = 1;
  for (int k = 0; k < nbr_frames; ++k)
  {
    wsum += wref_arr[k];
  }

  // normalize weights to 256
  int wsrc = one;
  for (int k = 1; k < nbr_frames; ++k)
  {
    const int norm = wref_arr[k] * one / wsum;
    wref_arr[k] = norm;
    wsrc -= norm;
  }
  wref_arr[0] = wsrc;
}

MV_FORCEINLINE void MDegrainN::apply_dn_mask_weights(int wref_arr[], int trad, int iDN_MM_weight)
{
  const int nbr_frames = trad * 2 + 1;

  for (int k = 1; k < nbr_frames; ++k)
  {
    const int old_weight = wref_arr[k];
    wref_arr[k] = (old_weight * iDN_MM_weight) >> 8; // integer div 256 - the max 255 will be reduced
  }

  // safe return 1 weight for no-degrain
  if (iDN_MM_weight == 0)
    wref_arr[0] = 1 << DEGRAIN_WEIGHT_BITS;
}


MV_FORCEINLINE int DegrainWeightN(int thSAD, double thSAD_pow, int blockSAD, int wpow)
{
  // Returning directly prevents a divide by 0 if thSAD == blockSAD == 0.
  // keep integer comparison for speed
  if (thSAD <= blockSAD)
    return 0;

  if (wpow > 6) return (int)(1 << DEGRAIN_WEIGHT_BITS); // if 7  - equal weights version - fast return, max speed

//  double blockSAD_pow = blockSAD;
  float blockSAD_pow = blockSAD;

  for (int i = 0; i < wpow - 1; i++)
  {
    blockSAD_pow *= blockSAD;
  }
  /*
  if (CPU_SSE2) // test if single precicion and approximate reciprocal is enough
  {
    float fthSAD_pow = (float)thSAD_pow;
    __m128 xmm_divisor = _mm_cvt_si2ss(xmm_divisor, (fthSAD_pow + blockSAD_pow));
    __m128 xmm_res = _mm_cvt_si2ss(xmm_res, (fthSAD_pow - blockSAD_pow));
    xmm_divisor = _mm_rcp_ss(xmm_divisor);

    __m128 xmm_dwb = _mm_cvt_si2ss(xmm_dwb, (1 << DEGRAIN_WEIGHT_BITS));

    xmm_res = _mm_mul_ss(xmm_res, xmm_divisor);
    xmm_res = _mm_mul_ss(xmm_res, xmm_dwb);

    return _mm_cvt_ss2si(xmm_res);
  }
  */
  // float is approximately only 24 bit precise, use double
  return (int)((double)(1 << DEGRAIN_WEIGHT_BITS) * (thSAD_pow - blockSAD_pow) / (thSAD_pow + blockSAD_pow));

}


float MDegrainN::fSinc(float x)
{
  x = fabsf(x);

  if (x > 0.000001f)
  {
    return sinf(x) / x;
  }
  else return 1.0f;
}

void MDegrainN::FilterMVs(void) 
{
  for (int by = 0; by < nBlkY; by++)
  {
    for (int bx = 0; bx < nBlkX; bx++)
    {
      int i = by * nBlkX + bx;
      FilterBlkMVs(i, bx, by);
    } // bx
  } // by

}

// single block processing FilterMVs to allow to use cached subshifted block
MV_FORCEINLINE void MDegrainN::FilterBlkMVs(int i, int bx, int by)
{
  VECTOR filteredp2fvectors[(MAX_TEMP_RAD * 2) + 1];
  VECTOR filteredp2fvectors2[(MAX_TEMP_RAD * 2) + 1];

  VECTOR p2fvectors[(MAX_TEMP_RAD * 2) + 1];
  // convert +1, -1, +2, -2, +3, -3 ... to
// -3, -2, -1, 0, +1, +2, +3 timed sequence
  for (int k = 0; k < _trad; ++k)
  {
    p2fvectors[k] = pMVsWorkPlanesArrays[(_trad - k - 1) * 2 + 1][i];
  }

  p2fvectors[_trad].x = 0; // zero trad - source block itself
  p2fvectors[_trad].y = 0;
  p2fvectors[_trad].sad = 0;

  for (int k = 1; k < _trad + 1; ++k)
  {
    p2fvectors[k + _trad] = pMVsWorkPlanesArrays[(k - 1) * 2][i];
  }

  if (iMVMedF > 0) // Median-like temporal filtering enabled
  {
    if (fMVLPFCutoff < 1.0f || fMVLPFGauss > 0.0f) // dual pass filtering
    {
      ProcessMVMedF(&p2fvectors[0], &filteredp2fvectors2[0]);
    }
    else // single pass filtering
      ProcessMVMedF(&p2fvectors[0], &filteredp2fvectors[0]);
  }

  if (fMVLPFCutoff < 1.0f || fMVLPFGauss > 0.0f)
  {
    if (iMVMedF > 0) // Median-like temporal filtering enabled
    {
      ProcessMVLPF(&filteredp2fvectors2[0], &filteredp2fvectors[0]);
    }
    else // single pass filtering
      ProcessMVLPF(&p2fvectors[0], &filteredp2fvectors[0]);
  }

  // final copy output
  VECTOR vLPFed, vOrig;

  for (int k = 0; k < _trad; ++k)
  {
    // recheck SAD:

    vLPFed = filteredp2fvectors[k];
    int idx_mvto = (_trad - k - 1) * 2 + 1;

    if (vLPFed.sad != veryBigSAD)
    {
      if (iNEW_DMFlags == 0)
        vLPFed.sad = CheckSAD(bx, by, idx_mvto, vLPFed.x, vLPFed.y);
      else
        vLPFed.sad = GetDM(bx, by, idx_mvto, vLPFed.x, vLPFed.y);
    }
    // else - block invalidated - do not recheck sad again

    vOrig = pMVsWorkPlanesArrays[(_trad - k - 1) * 2 + 1][i];
    if ((abs(vLPFed.x - vOrig.x) <= ithMVLPFCorr) && (abs(vLPFed.y - vOrig.y) <= ithMVLPFCorr) && (vLPFed.sad < _mv_clip_arr[idx_mvto]._thsad))
    {
      vLPFed.sad = (int)((float)vLPFed.sad * fadjSADLPFedmv); // make some boost of weight for filtered because they typically have worse SAD
      pFilteredMVsPlanesArrays[(_trad - k - 1) * 2 + 1][i] = vLPFed;
    }
    else // place original vector
    {
      if (vLPFed.sad != veryBigSAD)
        pFilteredMVsPlanesArrays[(_trad - k - 1) * 2 + 1][i] = vOrig;
      else
      {
        pFilteredMVsPlanesArrays[(_trad - k - 1) * 2 + 1][i].x = vOrig.x;
        pFilteredMVsPlanesArrays[(_trad - k - 1) * 2 + 1][i].y = vOrig.y;
        pFilteredMVsPlanesArrays[(_trad - k - 1) * 2 + 1][i].sad = veryBigSAD; // invalidate block
      }
    }

    if ((vLPFed.sad > _mv_clip_arr[idx_mvto]._thsad) && (iMVF_fm == 1))
      pFilteredMVsPlanesArrays[(_trad - k - 1) * 2 + 1][i].sad = veryBigSAD; // invalidate block
  }

  for (int k = 1; k < _trad + 1; ++k)
  {
    // recheck SAD
    vLPFed = filteredp2fvectors[k + _trad];
    int idx_mvto = (k - 1) * 2;

    if (vLPFed.sad != veryBigSAD)
    {
      if (iNEW_DMFlags == 0)
        vLPFed.sad = CheckSAD(bx, by, idx_mvto, vLPFed.x, vLPFed.y);
      else
        vLPFed.sad = GetDM(bx, by, idx_mvto, vLPFed.x, vLPFed.y);
    }
    //else - block invalidated - do not recheck sad

    vOrig = pMVsWorkPlanesArrays[(k - 1) * 2][i];
    if ((abs(vLPFed.x - vOrig.x) <= ithMVLPFCorr) && (abs(vLPFed.y - vOrig.y) <= ithMVLPFCorr) && (vLPFed.sad < _mv_clip_arr[idx_mvto]._thsad))
    {
      vLPFed.sad = (int)((float)vLPFed.sad * fadjSADLPFedmv); // make some boost of weight for filtered because they typically have worse SAD
      pFilteredMVsPlanesArrays[(k - 1) * 2][i] = vLPFed;
    }
    else
    {
      if (vLPFed.sad != veryBigSAD)
        pFilteredMVsPlanesArrays[(k - 1) * 2][i] = vOrig;
      else
      {
        pFilteredMVsPlanesArrays[(k - 1) * 2][i].x = vOrig.x;
        pFilteredMVsPlanesArrays[(k - 1) * 2][i].y = vOrig.y;
        pFilteredMVsPlanesArrays[(k - 1) * 2][i].sad = veryBigSAD; // invalidate block
      }
    }

    if ((vLPFed.sad > _mv_clip_arr[idx_mvto]._thsad) && (iMVF_fm == 1))
      pFilteredMVsPlanesArrays[(k - 1) * 2][i].sad = veryBigSAD; // invalidate block

  }
}

MV_FORCEINLINE void MDegrainN::ProcessMVLPF(VECTOR* pVin, VECTOR* pVout)
{
  // perform lpf of all good vectors in tr-scope
  for (int pos = 0; pos < (_trad * 2 + 1); pos++)
  {
    float fSumX = 0.0f;
    float fSumY = 0.0f;
    for (int kpos = 0; kpos < MVLPFKERNELSIZE; kpos++)
    {
      int src_pos = pos + kpos - MVLPFKERNELSIZE / 2;
      if (src_pos < 0) src_pos = 0;
      if (src_pos > _trad * 2) src_pos = (_trad * 2); // total valid samples in vector of VECTORs is _trad*2+1
      fSumX += pVin[src_pos].x * fMVLPFKernel[kpos];
      fSumY += pVin[src_pos].y * fMVLPFKernel[kpos];
    }

    pVout[pos].x = (int)(fSumX);
    pVout[pos].y = (int)(fSumY);
    pVout[pos].sad = pVin[pos].sad;
  }

}

MV_FORCEINLINE void MDegrainN::ProcessMVMedF(VECTOR* pVin, VECTOR* pVout)
{
  VECTOR MedF_vect[(MAX_TEMP_RAD * 2) + 1];

  for (int pos = 0; pos < (_trad * 2 + 1); pos++) 
  {
    VECTOR vOut = pVin[pos];

    if ((pos >= iMVMedF) && (pos < (_trad * 2 + 1) - iMVMedF)) // iMVMedF - temporal radius 1,2,3,.. 
    {
      // fill temporal vector of VECTORs for median filtering of single step
      for (int kpos = 0; kpos < (iMVMedF * 2 + 1); kpos++)
      {
        int src_pos = pos + kpos - iMVMedF;
        MedF_vect[kpos] = pVin[src_pos];
      }

      switch (iMVMedF_cm)
      {
        case 0:
          MVMedF_xy(&MedF_vect[0], &vOut);
          break;
        case 1:
          MVMedF_vl(&MedF_vect[0], &vOut);
          break;
        case 2:
          MVMedF_vad(&MedF_vect[0], &vOut);
          break;
        case 3:
          MVMedF_mg(&MedF_vect[0], &vOut);
          break;
        case 4:
          MVMedF_IQM(&MedF_vect[0], &vOut);
          break;
      }
    }
    else // non-processed edges
    {
      if (iMVMedF_em == 1) vOut.sad = veryBigSAD; // invalidate block - check for this in FilterX() functions to skip SAD re-check
    }

    pVout[pos] = vOut;
  }
}

// Median-like MVs filtering with separated x,y coordinates of vectors
MV_FORCEINLINE void MDegrainN::MVMedF_xy(VECTOR* pVin, VECTOR* pVout)
{
  // process dual coords in scalar C ?
  int iMaxMVlength = std::max(nWidth, nHeight) * 2 * nPel; // hope it is enough ?
  int MaxSumDM = (_trad * 2 + 1) * iMaxMVlength;

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow_x = MaxSumDM;
  int sum_minrow_y = MaxSumDM;
  int i_idx_minrow_x = 0;
  int i_idx_minrow_y = 0;

  for (int dmt_row = 0; dmt_row < (iMVMedF * 2 + 1); dmt_row++)
  {
    int sum_row_x = 0;
    int sum_row_y = 0;

    for (int dmt_col = 0; dmt_col < (iMVMedF * 2 + 1); dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      sum_row_x += std::abs(pVin[dmt_row].x - pVin[dmt_col].x);
      sum_row_y += std::abs(pVin[dmt_row].y - pVin[dmt_col].y);
    }

    if (sum_row_x < sum_minrow_x)
    {
      sum_minrow_x = sum_row_x;
      i_idx_minrow_x = dmt_row;
    }

    if (sum_row_y < sum_minrow_y)
    {
      sum_minrow_y = sum_row_y;
      i_idx_minrow_y = dmt_row;
    }

  }

  pVout[0].x = pVin[i_idx_minrow_x].x;
  pVout[0].y = pVin[i_idx_minrow_y].y;
  pVout[0].sad = pVin[iMVMedF].sad; // pick central (current sample ?), it is not used later

}

// Median-like MVs filtering using single difference vector length dismetric
MV_FORCEINLINE void MDegrainN::MVMedF_vl(VECTOR* pVin, VECTOR* pVout)
{
  // process dual coords in scalar C ?
  int iMaxMVlength = std::max(nWidth, nHeight) * 2 * nPel; // hope it is enough ?
  int MaxSumDM = (_trad * 2 + 1) * iMaxMVlength * (_trad * 2 + 1) * iMaxMVlength; // squared length ?

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow = MaxSumDM;
  int i_idx_minrow = 0;

  for (int dmt_row = 0; dmt_row < (iMVMedF * 2 + 1); dmt_row++)
  {
    int sum_row = 0;

    for (int dmt_col = 0; dmt_col < (iMVMedF * 2 + 1); dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      //difference vector squared length
      int idv_sq_l = (pVin[dmt_row].x - pVin[dmt_col].x) * (pVin[dmt_row].x - pVin[dmt_col].x) + (pVin[dmt_row].y - pVin[dmt_col].y) * (pVin[dmt_row].y - pVin[dmt_col].y);

      sum_row += idv_sq_l;
    }

    if (sum_row < sum_minrow)
    {
      sum_minrow = sum_row;
      i_idx_minrow = dmt_row;
    }

  }

  pVout[0].x = pVin[i_idx_minrow].x;
  pVout[0].y = pVin[i_idx_minrow].y;
  pVout[0].sad = pVin[i_idx_minrow].sad;
  
}

MV_FORCEINLINE void MDegrainN::MVMedF_vad(VECTOR* pVin, VECTOR* pVout)
{
  // process dual coords in scalar C ?
  const int iMaxAngDiff = 3; // hope it is enough ? 
  int MaxSumDM = iMVMedF * iMaxAngDiff;

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  float sum_minrow = (float)MaxSumDM;
  int i_idx_minrow = 0;

  for (int dmt_row = 0; dmt_row < (iMVMedF * 2 + 1); dmt_row++)
  {
    float sum_row = 0;

    for (int dmt_col = 0; dmt_col < (iMVMedF * 2 + 1); dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      //difference vector angle
      sum_row += fDiffAngleVect(pVin[dmt_row].x, pVin[dmt_row].y, pVin[dmt_col].x, pVin[dmt_col].y);
    }

    if (sum_row < sum_minrow)
    {
      sum_minrow = sum_row;
      i_idx_minrow = dmt_row;
    }

  }

  pVout[0].x = pVin[i_idx_minrow].x;
  pVout[0].y = pVin[i_idx_minrow].y;
  pVout[0].sad = pVin[i_idx_minrow].sad;

}

MV_FORCEINLINE void MDegrainN::MVMedF_mg(VECTOR* pVin, VECTOR* pVout)
{
  const int test_steps_dx[] = { -1, 0, 1, 0 };
  const int test_steps_dy[] = { 0, 1, 0, -1 };

  VECTOR_XY vGMedian;
  int iMinDist = 0;

  // need to estimate max radius of vectors area ?
  int iStep = 4 * nPel; // 16 max for pel=4, need to be lower at high levels ?

  // first estimation - center of gravity
  int iMeanX = 0;
  int iMeanY = 0;
  for (int i = 0; i < (iMVMedF * 2 + 1); i++)
  {
    iMeanX += pVin[i].x;
    iMeanY += pVin[i].y;
  }

  vGMedian.x = (iMeanX + ((iMVMedF * 2 + 1) >> 1)) / (iMVMedF * 2 + 1);
  vGMedian.y = (iMeanY + ((iMVMedF * 2 + 1) >> 1)) / (iMVMedF * 2 + 1);

  // init iMinDist with first estimate
  for (int i = 0; i < (iMVMedF * 2 + 1); i++)
  {
    iMinDist += (pVin[i].x - vGMedian.x) * (pVin[i].x - vGMedian.x) + (pVin[i].y - vGMedian.y) * (pVin[i].y - vGMedian.y);
  }

  if (iMinDist > 0)
  {
    while (iStep > 0)
    {
      bool bDone = false;
      for (int i = 0; i < 4; ++i)
      {
        VECTOR_XY vToCheck;
        vToCheck.x = vGMedian.x + iStep * test_steps_dx[i];
        vToCheck.y = vGMedian.y + iStep * test_steps_dy[i];

        int iCheckedSum = 0;
        for (int j = 0; j < (iMVMedF * 2 + 1); j++)
        {
          iCheckedSum += (pVin[j].x - vToCheck.x) * (pVin[j].x - vToCheck.x) + (pVin[j].y - vToCheck.y) * (pVin[j].y - vToCheck.y);
        }

        if (iCheckedSum < iMinDist)
        {
          iMinDist = iCheckedSum;
          vGMedian = vToCheck;

          bDone = true;
          break;
        }
      }

      if (!bDone)
        iStep /= 2;
    }
  }// if iMinDist > 0

  pVout[0].x = vGMedian.x;
  pVout[0].y = vGMedian.y;
  pVout[0].sad = 0; 

}

MV_FORCEINLINE void MDegrainN::MVMedF_IQM(VECTOR* pVin, VECTOR* pVout)
{
  int vX[MAX_TEMP_RAD * 2 + 1];
  int vY[MAX_TEMP_RAD * 2 + 1];

  // copy to temp vectors
  for (int i = 0; i < (iMVMedF * 2 + 1); i++)
  {
    vX[i] = pVin[i].x;
    vY[i] = pVin[i].y;
  }

  // make ordering sort
  std::sort(vX, vX + (iMVMedF * 2 + 1));
  std::sort(vY, vY + (iMVMedF * 2 + 1));

  if ((iMVMedF * 2 + 1) < 4) // 3 possible ?
  {
    pVout[0].x = vX[1];
    pVout[0].y = vY[1];
    pVout[0].sad = pVin[1].sad;
  }
  else
  {
    int qStart = ((iMVMedF * 2 + 1) + 1) / 4; // do we want bias here ?
    int qEnd = (iMVMedF * 2 + 1) - (((iMVMedF * 2 + 1) + 1) / 4);

    int iXmean = 0;
    int iYmean = 0;
    for (int i = qStart; i < qEnd; i++)
    {
      iXmean += vX[i];
      iYmean += vY[i];
    }
    int iBias = (qEnd - qStart) / 2;

    iXmean = (iXmean + iBias) / (qEnd - qStart);
    iYmean = (iYmean + iBias) / (qEnd - qStart);

    pVout[0].x = iXmean;
    pVout[0].y = iYmean;
  }

  pVout[0].sad = 0; 

}

MV_FORCEINLINE void MDegrainN::PrefetchMVs(int i)
{
  if ((i % 5) == 0) // do not prefetch each block - the 12bytes VECTOR sit about 5 times in the 64byte cache line 
  {
    if (!bMVsAddProc)
    {
      for (int k = 0; k < _trad * 2; ++k)
      {
//        const VECTOR* pMVsArrayPref = pMVsPlanesArrays[k];
        const VECTOR* pMVsArrayPref = pMVsWorkPlanesArrays[k];
        _mm_prefetch(const_cast<const CHAR*>(reinterpret_cast<const CHAR*>(&pMVsArrayPref[i + 5])), _MM_HINT_T0);
      }
    }
    else
    {
      for (int k = 0; k < _trad * 2; ++k)
      {
        const VECTOR* pMVsArrayPref = pFilteredMVsPlanesArrays[k];
        _mm_prefetch(const_cast<const CHAR*>(reinterpret_cast<const CHAR*>(&pMVsArrayPref[i + 5])), _MM_HINT_T0);
      }
    }
  }
}

MV_FORCEINLINE void MDegrainN::MemZoneSetUV(uint16_t* pDstShortUV, int* pDstIntUV)
{
  if (_lsb_flag || pixelsize_output > 1)
  {
    MemZoneSet(
      reinterpret_cast <unsigned char*> (pDstIntUV), 0,
      (_covered_width * sizeof(int)) >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
      0, 0, _dst_int_pitch * sizeof(int)
    );
  }
  else
  {
    MemZoneSet(
      reinterpret_cast <unsigned char*> (pDstShortUV), 0,
      (_covered_width * sizeof(short)) >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
      0, 0, _dst_short_pitch * sizeof(short)
    );
  }
}

MV_FORCEINLINE void MDegrainN::MemZoneSetY(uint16_t* pDstShort, int* pDstInt)
{
  if (_lsb_flag || pixelsize_output > 1)
  {
    MemZoneSet(
      reinterpret_cast <unsigned char*> (pDstInt), 0,
      _covered_width * sizeof(int), _covered_height, 0, 0, _dst_int_pitch * sizeof(int)
    );
  }
  else
  {
    MemZoneSet(
      reinterpret_cast <unsigned char*> (pDstShort), 0,
      _covered_width * sizeof(short), _covered_height, 0, 0, _dst_short_pitch * sizeof(short)
    );
  }
}

MV_FORCEINLINE void MDegrainN::post_overlap_chroma_plane(int P, uint16_t* pDstShort, int* pDstInt)
{
  if (_lsb_flag)
  {
    Short2BytesLsb(
      _dst_ptr_arr[P],
      _dst_ptr_arr[P] + _lsb_offset_arr[P], // 8 bit only
      _dst_pitch_arr[P],
      pDstInt, _dst_int_pitch,
      _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
    );
  }
  else if (_out16_flag)
  {
    if ((_cpuFlags & CPU_SSE4) != 0)
    {
      Short2Bytes_Int32toWord16_sse4(
        (uint16_t*)_dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstInt, _dst_int_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
        bits_per_pixel_output
      );
    }
    else
      Short2Bytes_Int32toWord16(
      (uint16_t*)_dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstInt, _dst_int_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
        bits_per_pixel_output
      );
  }
  else if (pixelsize_super == 1)
  {
    if ((_cpuFlags & CPUF_AVX2) != 0)
    {
      Short2Bytes_avx2(
        _dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstShort, _dst_short_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
      );
    }
    else if ((_cpuFlags & CPUF_SSE2) != 0)
    {
      Short2Bytes_sse2(
        _dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstShort, _dst_short_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
      );
    }
    else
    {
      Short2Bytes(
        _dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstShort, _dst_short_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
      );
    }
  }
  else if (pixelsize_super == 2)
  {
    if ((_cpuFlags & CPU_SSE4) != 0)
    {
      Short2Bytes_Int32toWord16_sse4(
        (uint16_t*)_dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstInt, _dst_int_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
        bits_per_pixel_super
      );
    }
    else
      Short2Bytes_Int32toWord16(
      (uint16_t*)_dst_ptr_arr[P], _dst_pitch_arr[P],
        pDstInt, _dst_int_pitch,
        _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super,
        bits_per_pixel_super
      );
  }
  else if (pixelsize_super == 4)
  {
    Short2Bytes_FloatInInt32ArrayToFloat(
      (float*)_dst_ptr_arr[P], _dst_pitch_arr[P],
      (float*)pDstInt, _dst_int_pitch,
      _covered_width >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
    );
  }

  if (_covered_width < nWidth)
  {
    if (_out16_flag) {
      // copy 8 bit source to 16bit target
      plane_copy_8_to_16_c(_dst_ptr_arr[P] + ((_covered_width >> nLogxRatioUV_super) << pixelsize_output_shift), _dst_pitch_arr[P],
        _src_ptr_arr[P] + (_covered_width >> nLogxRatioUV_super), _src_pitch_arr[P],
        (nWidth - _covered_width) >> nLogxRatioUV_super, _covered_height >> nLogyRatioUV_super
      );
    }
    else {
      BitBlt(
        _dst_ptr_arr[P] + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _dst_pitch_arr[P],
        _src_ptr_arr[P] + ((_covered_width >> nLogxRatioUV_super) << pixelsize_super_shift), _src_pitch_arr[P],
        ((nWidth - _covered_width) >> nLogxRatioUV_super) << pixelsize_super_shift, _covered_height >> nLogyRatioUV_super
      );
    }
  }
  if (_covered_height < nHeight) // bottom noncovered region
  {
    if (_out16_flag) {
      // copy 8 bit source to 16bit target
      plane_copy_8_to_16_c(_dst_ptr_arr[P] + ((_dst_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _dst_pitch_arr[P],
        _src_ptr_arr[P] + ((_src_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _src_pitch_arr[P],
        nWidth >> nLogxRatioUV_super, ((nHeight - _covered_height) >> nLogyRatioUV_super)
      );
    }
    else {
      BitBlt(
        _dst_ptr_arr[P] + ((_dst_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _dst_pitch_arr[P],
        _src_ptr_arr[P] + ((_src_pitch_arr[P] * _covered_height) >> nLogyRatioUV_super), _src_pitch_arr[P],
        (nWidth >> nLogxRatioUV_super) << pixelsize_super_shift, ((nHeight - _covered_height) >> nLogyRatioUV_super)
      );
    }
  }
}

MV_FORCEINLINE void MDegrainN::post_overlap_luma_plane(void)
{
  // fixme: SSE versions from ShortToBytes family like in MDegrain3
  if (_lsb_flag)
  {
    Short2BytesLsb(
      _dst_ptr_arr[0],
      _dst_ptr_arr[0] + _lsb_offset_arr[0],
      _dst_pitch_arr[0],
      &_dst_int[0], _dst_int_pitch,
      _covered_width, _covered_height
    );
  }
  else if (_out16_flag)
  {
    if ((_cpuFlags & CPU_SSE4) != 0)
    {
      Short2Bytes_Int32toWord16_sse4(
        (uint16_t*)_dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_int[0], _dst_int_pitch,
        _covered_width, _covered_height,
        bits_per_pixel_output
      );
    }
    else
      Short2Bytes_Int32toWord16(
      (uint16_t*)_dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_int[0], _dst_int_pitch,
        _covered_width, _covered_height,
        bits_per_pixel_output
      );
  }
  else if (pixelsize_super == 1)
  {
    if ((_cpuFlags & CPUF_AVX2) != 0)
    {
      Short2Bytes_avx2(
        _dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_short[0], _dst_short_pitch,
        _covered_width, _covered_height
      );
    }
    else if ((_cpuFlags & CPUF_SSE2) != 0)
    {
      Short2Bytes_sse2(
        _dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_short[0], _dst_short_pitch,
        _covered_width, _covered_height
      );
    }
    else
    {
      Short2Bytes(
        _dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_short[0], _dst_short_pitch,
        _covered_width, _covered_height
      );
    }
  }
  else if (pixelsize_super == 2)
  {
    if ((_cpuFlags & CPU_SSE4) != 0)
    {
      Short2Bytes_Int32toWord16_sse4(
        (uint16_t*)_dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_int[0], _dst_int_pitch,
        _covered_width, _covered_height,
        bits_per_pixel_super
      );
    }
    else
      Short2Bytes_Int32toWord16(
        (uint16_t*)_dst_ptr_arr[0], _dst_pitch_arr[0],
        &_dst_int[0], _dst_int_pitch,
        _covered_width, _covered_height,
        bits_per_pixel_super
      );
  }
  else if (pixelsize_super == 4)
  {
    Short2Bytes_FloatInInt32ArrayToFloat(
      (float*)_dst_ptr_arr[0], _dst_pitch_arr[0],
      (float*)&_dst_int[0], _dst_int_pitch,
      _covered_width, _covered_height
    );
  }
  if (_covered_width < nWidth)
  {
    if (_out16_flag) {
      // copy 8 bit source to 16bit target
      plane_copy_8_to_16_c(_dst_ptr_arr[0] + (_covered_width << pixelsize_output_shift), _dst_pitch_arr[0],
        _src_ptr_arr[0] + _covered_width, _src_pitch_arr[0],
        nWidth - _covered_width, _covered_height
      );
    }
    else {
      BitBlt(
        _dst_ptr_arr[0] + (_covered_width << pixelsize_super_shift), _dst_pitch_arr[0],
        _src_ptr_arr[0] + (_covered_width << pixelsize_super_shift), _src_pitch_arr[0],
        (nWidth - _covered_width) << pixelsize_super_shift, _covered_height
      );
    }
  }
  if (_covered_height < nHeight) // bottom noncovered region
  {
    if (_out16_flag) {
      // copy 8 bit source to 16bit target
      plane_copy_8_to_16_c(_dst_ptr_arr[0] + _covered_height * _dst_pitch_arr[0], _dst_pitch_arr[0],
        _src_ptr_arr[0] + _covered_height * _src_pitch_arr[0], _src_pitch_arr[0],
        nWidth, nHeight - _covered_height
      );
    }
    else {
      BitBlt(
        _dst_ptr_arr[0] + _covered_height * _dst_pitch_arr[0], _dst_pitch_arr[0],
        _src_ptr_arr[0] + _covered_height * _src_pitch_arr[0], _src_pitch_arr[0],
        nWidth << pixelsize_super_shift, nHeight - _covered_height
      );
    }
  }

}

MV_FORCEINLINE void MDegrainN::nlimit_luma(void)
{
  // limit is 0-255 relative, for any bit depth
  float realLimit;
  if (pixelsize_output <= 2)
    realLimit = _nlimit * (1 << (bits_per_pixel_output - 8));
  else
    realLimit = _nlimit / 255.0f;
  LimitFunction(_dst_ptr_arr[0], _dst_pitch_arr[0],
    _src_ptr_arr[0], _src_pitch_arr[0],
    nWidth, nHeight,
    realLimit
  );
}

MV_FORCEINLINE void MDegrainN::nlimit_chroma(int P)
{
    // limit is 0-255 relative, for any bit depth
  float realLimit;
  if (pixelsize_output <= 2)
    realLimit = _nlimitc * (1 << (bits_per_pixel_output - 8));
  else
    realLimit = (float)_nlimitc / 255.0f;
  LimitFunction(_dst_ptr_arr[P], _dst_pitch_arr[P],
    _src_ptr_arr[P], _src_pitch_arr[P],
    nWidth >> nLogxRatioUV_super, nHeight >> nLogyRatioUV_super,
    realLimit
  );
}

void MDegrainN::InterpolateOverlap_4x(VECTOR* pInterpolatedMVs, const VECTOR* pInputMVs, int idx)
{
//  VECTOR* pInp = (VECTOR*)pInputMVs;

  // use linear interpolate 2x of x first
  int i;
  int bxInp = 0;
  int byInp = 0;
  int j;

  for (int by = 0; by < nBlkY; by += 2) // output blkY
  {
    bxInp = 0;
    for (int bx = 0; bx < nBlkX; bx++) // output blkX
    {
      i = by * nBlkX + bx;
      j = byInp * nInputBlkX + bxInp;
      // original MVs in each 2nd MV
      if ((bx % 2) == 0)
      {
        if (iInterpolateOverlap == 1)
        {
          // re-check SAD always
          pInterpolatedMVs[i].x = pInputMVs[j].x;
          pInterpolatedMVs[i].y = pInputMVs[j].y;
          if (iNEW_DMFlags == 0)
            pInterpolatedMVs[i].sad = CheckSAD(bx, by, idx, pInputMVs[j].x, pInputMVs[j].y);
          else
            pInterpolatedMVs[i].sad = GetDM(bx, by, idx, pInputMVs[j].x, pInputMVs[j].y);
        }
        else
          pInterpolatedMVs[i] = pInputMVs[j];
      }
      else
      {
        const int blx = (pInputMVs[j].x + pInputMVs[j + 1].x) / 2;
        const int bly = (pInputMVs[j].y + pInputMVs[j + 1].y) / 2;
        pInterpolatedMVs[i].x = blx;
        pInterpolatedMVs[i].y = bly;
        // update SAD
        if (iInterpolateOverlap == 1)
        {
          if (iNEW_DMFlags == 0)
            pInterpolatedMVs[i].sad = CheckSAD(bx, by, idx, blx, bly); // better quality - slower
          else
            pInterpolatedMVs[i].sad = GetDM(bx, by, idx, blx, bly); // better quality - slower
        }
        else
          pInterpolatedMVs[i].sad = (pInputMVs[j].sad + pInputMVs[j + 1].sad) / 2; // faster mode

        bxInp++;
      }
    }// for bx

    byInp++;

  }	// for by

  // linear interpolate 2x by y of output array
  byInp = 0;

  for (int by = 1; by < nBlkY - 1; by += 2) // output blkY
  {
    for (int bx = 0; bx < nBlkX; bx++) // output blkX
    {
      i = by * nBlkX + bx;
      j = byInp * nBlkX + bx;

      const int blx = (pInterpolatedMVs[j].x + pInterpolatedMVs[j + nBlkX * 2].x) / 2;
      const int bly = (pInterpolatedMVs[j].y + pInterpolatedMVs[j + nBlkX * 2].y) / 2;
      pInterpolatedMVs[i].x = blx;
      pInterpolatedMVs[i].y = bly;
      // update SAD
      if (iInterpolateOverlap == 1)
      {
        if (iNEW_DMFlags == 0)
          pInterpolatedMVs[i].sad = CheckSAD(bx, by, idx, blx, bly); // better quality - slower
        else
          pInterpolatedMVs[i].sad = GetDM(bx, by, idx, blx, bly); // better quality - slower
      }
      else
        pInterpolatedMVs[i].sad = (pInterpolatedMVs[j].sad + pInterpolatedMVs[j + nBlkX * 2].sad) / 2; // faster mode

    }// for by

    byInp += 2;

  }	// for by

}

void MDegrainN::InterpolateOverlap_2x(VECTOR* pInterpolatedMVs, const VECTOR* pInputMVs, int idx)
{
//  VECTOR* pInp = (VECTOR*)pInputMVs;

  // use linear interpolate 2x of x first
  int i;
  int bxInp = 0;
  int byInp = 0;
  int j;

  // linear interpolate 2x by y of output array using average of 4 neibour MVs
  byInp = 0;

  for (int by = 0; by < nBlkY; by ++) // output blkY, each line
  {
    for (int bx = 0; bx < nBlkX; bx++) // output blkX
    {
      i = by * nBlkX + bx;
      j = byInp * nBlkX + bx;

      if ((by % 2) == 0) // even lines 0,2,4,.. simply copy MV
      {
        if (iInterpolateOverlap == 3)
        {
          // re-check SAD always
          pInterpolatedMVs[i].x = pInputMVs[j].x;
          pInterpolatedMVs[i].y = pInputMVs[j].y;
          if (iNEW_DMFlags == 0)
            pInterpolatedMVs[i].sad = CheckSAD(bx, by, idx, pInputMVs[j].x, pInputMVs[j].y);
          else
            pInterpolatedMVs[i].sad = GetDM(bx, by, idx, pInputMVs[j].x, pInputMVs[j].y);
        }
        else
          pInterpolatedMVs[i] = pInputMVs[j];
        continue;
      }

      // odd lines - write interpolated MV of 4 neibour
      // skip last odd line
      if (byInp == nInputBlkY - 1) continue;

      // last block in the interpolated row
      if (bx == nInputBlkX - 1)
      {
        const int blx = (pInputMVs[j].x + pInputMVs[j + nInputBlkX].x ) / 2;
        const int bly = (pInputMVs[j].y + pInputMVs[j + nInputBlkX].y ) / 2;
        pInterpolatedMVs[i].x = blx;
        pInterpolatedMVs[i].y = bly;
        // update SAD
        if (iInterpolateOverlap == 3)
        {
          if (iNEW_DMFlags == 0)
            pInterpolatedMVs[i].sad = CheckSAD(bx, by, idx, blx, bly); // better quality - slower
          else
            pInterpolatedMVs[i].sad = GetDM(bx, by, idx, blx, bly); // better quality - slower
        }
        else // == 4
          pInterpolatedMVs[i].sad = (pInputMVs[j].sad + pInputMVs[j + nInputBlkX].sad ) / 2; // faster mode

        continue;
      }

      const int blx = (pInputMVs[j].x + pInputMVs[j + 1].x + pInputMVs[j + nInputBlkX].x + pInputMVs[j + nInputBlkX + 1].x) / 4;
      const int bly = (pInputMVs[j].y + pInputMVs[j + 1].y + pInputMVs[j + nInputBlkX].y + pInputMVs[j + nInputBlkX + 1].y) / 4;
      pInterpolatedMVs[i].x = blx;
      pInterpolatedMVs[i].y = bly;
      // update SAD
      if (iInterpolateOverlap == 3)
      {
        if (iNEW_DMFlags == 0)
          pInterpolatedMVs[i].sad = CheckSAD(bx, by, idx, blx, bly); // better quality - slower
        else
          pInterpolatedMVs[i].sad = GetDM(bx, by, idx, blx, bly); // better quality - slower
      }
      else // == 4
        pInterpolatedMVs[i].sad = (pInputMVs[j].sad + pInputMVs[j + 1].sad + pInputMVs[j + nInputBlkX].sad + pInputMVs[j + nInputBlkX + 1].sad) / 4; // faster mode

    }// for by

    if ((by % 2) != 0) byInp ++;

  }	// for by

}


MV_FORCEINLINE sad_t MDegrainN::CheckSAD(int bx_src, int by_src, int ref_idx, int dx_ref, int dy_ref)
{
  sad_t sad_out; 

  if (!_usable_flag_arr[ref_idx]) // nothing to process
  {
    return veryBigSAD;
  }
  
  const int  rowsize = nBlkSizeY - nOverlapY; // num of lines in row of blocks = block height - overlap ?
  const BYTE* pSrcCur = _src_ptr_arr[0];
  const BYTE* pSrcCurU = _src_ptr_arr[1];
  const BYTE* pSrcCurV = _src_ptr_arr[2];

  pSrcCur += by_src * (rowsize * _src_pitch_arr[0]);

  const int effective_nSrcPitch = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _src_pitch_arr[1]; // pitch is byte granularity, from 1st chroma plane

  pSrcCurU += by_src * (effective_nSrcPitch);
  pSrcCurV += by_src * (effective_nSrcPitch);

  const int xx = bx_src * (nBlkSizeX - nOverlapX); // xx: indexing offset, - overlap ?
  const int xx_uv = bx_src * ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super); // xx_uv: indexing offset

  bool bChroma = (_nsupermodeyuv & UPLANE) && (_nsupermodeyuv & VPLANE); // chroma present in super clip ?
// scaleCSAD in the MVclip props
  int chromaSADscale = _mv_clip_arr[0]._clip_sptr->chromaSADScale; // from 1st ?

  const uint8_t* pRef;
  int npitchRef;

  int blx = bx_src * (nBlkSizeX - nOverlapX) * nPel + dx_ref;
  int bly = by_src * (nBlkSizeY - nOverlapY) * nPel + dy_ref;

  ClipBlxBly

  if (nPel != 1 && nUseSubShift != 0)
  {
    pRef = _planes_ptr[ref_idx][0]->GetPointerSubShift(blx, bly, npitchRef);
  }
  else
  {
    pRef = _planes_ptr[ref_idx][0]->GetPointer(blx, bly);
    npitchRef = _planes_ptr[ref_idx][0]->GetPitch();
  }

  sad_t sad_chroma = 0;

  if (bChroma)
  {
    const uint8_t* pRefU;
    const uint8_t* pRefV;
    int npitchRefU, npitchRefV;

    if (/*nPel != 1 && */nUseSubShift != 0)
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pRefU = _planes_ptr[ref_idx][1]->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npitchRefU);
      pRefV = _planes_ptr[ref_idx][2]->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npitchRefV);
//      pRefU = _planes_ptr[ref_idx][1]->GetPointerSubShiftUV(blx, bly, npitchRefU, nLogxRatioUV_super, nLogyRatioUV_super);
//      pRefV = _planes_ptr[ref_idx][2]->GetPointerSubShiftUV(blx, bly, npitchRefV, nLogxRatioUV_super, nLogyRatioUV_super);
    }
    else
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pRefU = _planes_ptr[ref_idx][1]->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npitchRefU = _planes_ptr[ref_idx][1]->GetPitch();
      pRefV = _planes_ptr[ref_idx][2]->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npitchRefV = _planes_ptr[ref_idx][2]->GetPitch();
    }

    sad_chroma = ScaleSadChroma(SADCHROMA(pSrcCurU + (xx_uv << pixelsize_super_shift), _src_pitch_arr[1], pRefU, npitchRefU)
      + SADCHROMA(pSrcCurV + (xx_uv << pixelsize_super_shift), _src_pitch_arr[2], pRefV, npitchRefV), chromaSADscale);

    sad_t luma_sad = SAD(pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0], pRef, npitchRef);

    sad_out = luma_sad + sad_chroma;

  }
  else
  {
    sad_out = SAD(pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0], pRef, npitchRef);
  }

  return sad_out;
}

MV_FORCEINLINE sad_t MDegrainN::GetDM(int bx_src, int by_src, int ref_idx, int dx_ref, int dy_ref)
{
  sad_t dm_out;

  if (!_usable_flag_arr[ref_idx]) // nothing to process
  {
    return veryBigSAD;
  }

  const int  rowsize = nBlkSizeY - nOverlapY; // num of lines in row of blocks = block height - overlap ?
  const BYTE* pSrcCur = _src_ptr_arr[0];
  const BYTE* pSrcCurU = _src_ptr_arr[1];
  const BYTE* pSrcCurV = _src_ptr_arr[2];

  pSrcCur += by_src * (rowsize * _src_pitch_arr[0]);

  const int effective_nSrcPitch = ((nBlkSizeY - nOverlapY) >> nLogyRatioUV_super)* _src_pitch_arr[1]; // pitch is byte granularity, from 1st chroma plane

  pSrcCurU += by_src * (effective_nSrcPitch);
  pSrcCurV += by_src * (effective_nSrcPitch);

  const int xx = bx_src * (nBlkSizeX - nOverlapX); // xx: indexing offset, - overlap ?
  const int xx_uv = bx_src * ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super); // xx_uv: indexing offset

  bool bChroma = (_nsupermodeyuv & UPLANE) && (_nsupermodeyuv & VPLANE); // chroma present in super clip ?
// scaleCSAD in the MVclip props
  int chromaSADscale = _mv_clip_arr[0]._clip_sptr->chromaSADScale; // from 1st ?

  const uint8_t* pRef;
  int npitchRef;

  int blx = bx_src * (nBlkSizeX - nOverlapX) * nPel + dx_ref;
  int bly = by_src * (nBlkSizeY - nOverlapY) * nPel + dy_ref;

  ClipBlxBly

    if (nPel != 1 && nUseSubShift != 0)
    {
      pRef = _planes_ptr[ref_idx][0]->GetPointerSubShift(blx, bly, npitchRef);
    }
    else
    {
      pRef = _planes_ptr[ref_idx][0]->GetPointer(blx, bly);
      npitchRef = _planes_ptr[ref_idx][0]->GetPitch();
    }

  sad_t dm_chroma = 0;

  if (bChroma)
  {
    const uint8_t* pRefU;
    const uint8_t* pRefV;
    int npitchRefU, npitchRefV;

    if (/*nPel != 1 && */nUseSubShift != 0)
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pRefU = _planes_ptr[ref_idx][1]->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npitchRefU);
      pRefV = _planes_ptr[ref_idx][2]->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npitchRefV);
      //      pRefU = _planes_ptr[ref_idx][1]->GetPointerSubShiftUV(blx, bly, npitchRefU, nLogxRatioUV_super, nLogyRatioUV_super);
      //      pRefV = _planes_ptr[ref_idx][2]->GetPointerSubShiftUV(blx, bly, npitchRefV, nLogxRatioUV_super, nLogyRatioUV_super);
    }
    else
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pRefU = _planes_ptr[ref_idx][1]->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npitchRefU = _planes_ptr[ref_idx][1]->GetPitch();
      pRefV = _planes_ptr[ref_idx][2]->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npitchRefV = _planes_ptr[ref_idx][2]->GetPitch();
    }

    dm_chroma = ScaleSadChroma(DM_NEW_Chroma->GetDisMetric(pSrcCurU + (xx_uv << pixelsize_super_shift), _src_pitch_arr[1], pRefU, npitchRefU)
      + DM_NEW_Chroma->GetDisMetric(pSrcCurV + (xx_uv << pixelsize_super_shift), _src_pitch_arr[2], pRefV, npitchRefV), chromaSADscale);

    sad_t luma_dm = DM_NEW_Luma->GetDisMetric(pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0], pRef, npitchRef);

    dm_out = luma_dm + dm_chroma;

  }
  else
  {
    dm_out = DM_NEW_Luma->GetDisMetric(pSrcCur + (xx << pixelsize_super_shift), _src_pitch_arr[0], pRef, npitchRef);
  }

  return dm_out;
}



MV_FORCEINLINE void MDegrainN::ProcessRSMVdata(void)
{
  int iFailedMVs = 0;

  for (int k = 0; k < _trad * 2; ++k)
  {
    VECTOR* fwMVs = (VECTOR*)_mv_clip_arr[k]._clip_sptr->GetpMVsArray(0);
    VECTOR* bwMVs = (VECTOR*)_mv_clip_arr[k]._cliprs_sptr->GetpMVsArray(0);
 
    for (int by = 0; by < nInputBlkY; by++) // not interpolated overlap count
    {
      for (int bx = 0; bx < nInputBlkX; bx++) // not interpolated overlap count
      {
        int i = by * nBlkX + bx;

        VECTOR fwMV = fwMVs[i];

        // check SAD - if it > thSAD - skip it
        if (fwMV.sad > _mv_clip_arr[k]._thsad) continue;

        int blx = bx * (nBlkSizeX - nOverlapX) * nPel + fwMV.x;
        int bly = by * (nBlkSizeY - nOverlapY) * nPel + fwMV.y;

        ClipBlxBly

        int bx_bw = blx / (nBlkSizeX * nPel);
        int by_bw = bly / (nBlkSizeY * nPel);

        // do number clipping required also ???

        VECTOR bwMV = bwMVs[by_bw * nBlkX + bx_bw];

        int iLengthSQ_FWMV = fwMV.x * fwMV.x + fwMV.y * fwMV.y;
        int iLengthSQ_BWMV = bwMV.x * bwMV.x + bwMV.y * bwMV.y;

        if (abs(iLengthSQ_BWMV - iLengthSQ_FWMV) > thFWBWmvpos) 
        {
          // fail SAD of bad fw block
          fwMVs[i].sad = veryBigSAD;
          iFailedMVs++;
        }
        
      } // bx
    } // by
  } // k

  float fPrcFailedMVs = (float)iFailedMVs / float(nInputBlkX * nInputBlkY);
}

MV_FORCEINLINE int MDegrainN::AlignBlockWeights(const BYTE* pRef[], int Pitch[], const BYTE* pCurr, int iCurrPitch, int Wall[], int iBlkWidth, int iBlkHeight, bool bChroma, int iBlkNum)
{
  //first count number of non-zero weights, zero is current block weight, 1,2 is +-1frame and so on
  int iNumNZBlocks = 1; // we have at least one non-zero - the source itself ?
  const int iBlockSizeMem = iBlkWidth * iBlkHeight * pixelsize;
  const int iBlocksPitch = iBlkWidth * pixelsize;
  int iNumAlignedBlocks = 0;
  int W_sub[MAX_TEMP_RAD * 2 + 1];

  sad_t sad_array_sub[(MAX_TEMP_RAD * 2 + 1)];
  sad_t sad_array_add[(MAX_TEMP_RAD * 2 + 1)];

  // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
  for (int k = 0; k < _trad * 2; k++)
  {
    pRef[k] -= Pitch[k] * iBlkHeight;
  }

  for (int n = 1; n < (_trad * 2 + 1); n++)
  {
    if (Wall[n] != 0) iNumNZBlocks++;
  }

  if (iNumNZBlocks < 3) // current and at least 2 refs are non-zero weighted
  {
    // nothing to process - return 0 to stop proc
    return 0;
  }
  else
  {
    // fill initial max values
    for (int k = 0; k < (_trad * 2 + 1); k++)
    {
      sad_array_sub[k] = veryBigSAD;
      sad_array_add[k] = veryBigSAD;
    }

    sad_t stAVG_sub_SAD = 0;
    sad_t stAVG_add_SAD = 0;
    int iNumAVG = 0;

    // process current block too
    // subtracted
    if (!MPB_PartBlend) // use subtraction
    {
      if (_cpuFlags & CPUF_SSE2)
      {
        SubtractBlock_uint8_sse2(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
          pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
        //      SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
        //        pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
              // test - compare with real partial blend
/*
#ifdef _DEBUG
        for (int i = 0; i < (_trad * 2 + 1); i++)
          W_sub[i] = Wall[i];
        W_sub[0] = 0; // zero weight of current block
        norm_weights_all(W_sub, _trad);
        TmpBlock       tmp_block;
        const int tmpPitch = nBlkSizeX;
        _degrainluma_ptr(
          &tmp_block._d[0], tmp_block._lsb_ptr, tmpPitch,
          pCurr, iCurrPitch,
          pRef, Pitch, W_sub, _trad);

        int idifsamples = 0;
        for (int x = 0; x < nBlkSizeX; x++)
        {
          for (int y = 0; y < nBlkSizeY; y++)
          {
            int isample_part_blend = tmp_block._d[y * tmpPitch + x];
            int isample_subtr = (pMPBTempBlocks + (iBlockSizeMem * (1)))[y * iBlocksPitch + x];

            if (abs(isample_part_blend - isample_subtr) > 1)
            {
              int idbr = 0;
              idifsamples++;
            }
          }
        }

        sad_t difsad = SAD(&tmp_block._d[0], tmpPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);

        float fdif = (float)idifsamples / (float)(nBlkSizeX * nBlkSizeY);
        int idbr2 = int(fdif);
#endif
*/
      }
      else
      {
        SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
          pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
      }
    }
    else // use real partial blending
    {
      for (int i = 0; i < (_trad * 2 + 1); i++)
        W_sub[i] = Wall[i];
      W_sub[0] = 0; // zero weight of current block
      norm_weights_all(W_sub, _trad);

      if (!bChroma)
        _degrainluma_ptr(
          pMPBTempBlocks + (iBlockSizeMem * (1)), 0, iBlocksPitch,
          pCurr, iCurrPitch,
          pRef, Pitch, W_sub, _trad);
      else
        _degrainchroma_ptr(
          pMPBTempBlocks + (iBlockSizeMem * (1)), 0, iBlocksPitch,
          pCurr, iCurrPitch,
          pRef, Pitch, W_sub, _trad);

      // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
      for (int k = 0; k < _trad * 2; k++)
      {
        pRef[k] -= Pitch[k] * iBlkHeight;
      }
    }

    if (!bChroma)
//      sad_array_sub[0] = SAD(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
//      sad_t tmp_sad = SAD(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
      sad_array_sub[0] = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
    else
//      sad_array_sub[0] = SADCHROMA(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
        sad_array_sub[0] = DM_Chroma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
    stAVG_sub_SAD += sad_array_sub[0];

    // calc SAD of full blended vs refs
    if (!bChroma)
//      sad_array_add[0] = SAD(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
        sad_array_add[0] = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
    else
//      sad_array_add[0] = SADCHROMA(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
        sad_array_add[0] = DM_Chroma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
    stAVG_add_SAD += sad_array_add[0];

    iNumAVG++;

    // ref blocks
    for (int n = 1; n < (_trad * 2 + 1); n++)
    {
      if (Wall[n] != 0)
      {
        if (!MPB_PartBlend) // create subrtracted versions of full blended block (faster)
        {
          if (_cpuFlags & CPUF_SSE2)
          {
            SubtractBlock_uint8_sse2(pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch,
              pMPBTempBlocks, iBlocksPitch, (uint8_t*)pRef[n - 1], Pitch[n - 1], Wall[n], iBlkWidth, iBlkHeight);
          }
          else
          {
            SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch,
              pMPBTempBlocks, iBlocksPitch, (uint8_t*)pRef[n - 1], Pitch[n - 1], Wall[n], iBlkWidth, iBlkHeight);
          }
        }
        else // real partial blend (slower)
        {
          for (int i = 0; i < (_trad * 2 + 1); i++)
            W_sub[i] = Wall[i];
          W_sub[n] = 0; // zero weight of n-th block
          norm_weights_all(W_sub, _trad);

          if (!bChroma)
            _degrainluma_ptr(
              pMPBTempBlocks + (iBlockSizeMem * (n + 1)), 0, iBlocksPitch,
              pCurr, iCurrPitch,
              pRef, Pitch, W_sub, _trad);
          else
            _degrainchroma_ptr(
              pMPBTempBlocks + (iBlockSizeMem * (n + 1)), 0, iBlocksPitch,
              pCurr, iCurrPitch,
              pRef, Pitch, W_sub, _trad);

          // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
          for (int k = 0; k < _trad * 2; k++)
          {
            pRef[k] -= Pitch[k] * iBlkHeight;
          }

        }
        //calc SAD of full blended block vs subtracted
        if (!bChroma)
          //sad_array_sub[n] = SAD(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
          sad_array_sub[n] = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
        else
//          sad_array_sub[n] = SADCHROMA(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
          sad_array_sub[n] = DM_Chroma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
        stAVG_sub_SAD += sad_array_sub[n];

        // calc SAD of full blended vs refs
        if (!bChroma)
//          sad_array_add[n] = SAD(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
            sad_array_add[n] = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
        else
//          sad_array_add[n] = SADCHROMA(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
            sad_array_add[n] = DM_Chroma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
        stAVG_add_SAD += sad_array_add[n];

        iNumAVG++;
      }

    }

    // find average SAD
    stAVG_add_SAD /= iNumAVG;
    stAVG_sub_SAD /= iNumAVG;

    //check if SAD of curr block too differs from average
    for (int n = 0; n < (_trad * 2 + 1); n++)
    {
      if (Wall[n] != 0) // check only processed above blocks
      {
        // check MV length ?
        if (n != 0) // skip current block
        {
          VECTOR vCurr = pMVsWorkPlanesArrays[n - 1][iBlkNum]; // n from current block ?
          int iLengthvCurr_sq = ((vCurr.x * vCurr.x) + (vCurr.y * vCurr.y)); // >> lognPel ?
          if (iLengthvCurr_sq > MPB_MVlth)
          {
            continue; // skip weight adjustment for too long MVs ?
          }
        }

        if ((stAVG_sub_SAD - sad_array_sub[n] > MPBthSub) && (sad_array_add[n] - stAVG_add_SAD > MPBthAdd)) // block worse than avg ?
        {
          Wall[n] = (int)((float)Wall[n] * MPB_SPC_sub); // decrease weight
          iNumAlignedBlocks++;
          continue; // do not check for addition if already adjusted to lower weight ?
        }

        if ((sad_array_sub[n] - stAVG_sub_SAD > MPBthSub) && (stAVG_add_SAD - sad_array_add[n] > MPBthAdd)) // block beter than avg ?
        {
          int iNewW = (int)((float)Wall[n] * MPB_SPC_add); // increase weight 
          if (iNewW > 255) iNewW = 255;
          Wall[n] = iNewW;
          iNumAlignedBlocks++;
        }
      }
    }

  }

  if (iNumAlignedBlocks != 0)
  {
    norm_weights_all(Wall, _trad);
  }

  return iNumAlignedBlocks; // counter of weight-adjusted blocks, 0 if none
}


MV_FORCEINLINE int MDegrainN::AlignBlockWeightsLC(const BYTE* pRef[], int Pitch[],
  const BYTE* pRefUV1[], int PitchUV1[],
  const BYTE* pRefUV2[], int PitchUV2[],
  const BYTE* pCurr, const int iCurrPitch,
  const BYTE* pCurrUV1, const int iCurrPitchUV1,
  const BYTE* pCurrUV2, const int iCurrPitchUV2,
  int Wall[], const int iBlkWidth, const int iBlkHeight,
  const int iBlkWidthC, const int iBlkHeightC, const int chromaSADscale,
  int iBlkNum
)
{
  //first count number of non-zero weights, zero is current block weight, 1,2 is +-1frame and so on
  int iNumNZBlocks = 1; // we have at least one non-zero - the source itself ?
  const int iBlockSizeMem = iBlkWidth * iBlkHeight * pixelsize;
  const int iBlockSizeMemUV = iBlkWidthC * iBlkHeightC * pixelsize;
  const int iBlocksPitch = iBlkWidth * pixelsize;
  const int iBlocksPitchUV = iBlkWidthC * pixelsize;
  int iNumAlignedBlocks = 0;
  int W_sub[MAX_TEMP_RAD * 2 + 1];

  sad_t sad_array_sub[(MAX_TEMP_RAD * 2 + 1)];
  sad_t sad_array_add[(MAX_TEMP_RAD * 2 + 1)];

  // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
  for (int k = 0; k < _trad * 2; k++)
  {
    pRef[k] -= Pitch[k] * iBlkHeight;
    if ((MPBchroma & 0x1) != 0)
    {
      pRefUV1[k] -= PitchUV1[k] * iBlkHeightC;
      pRefUV2[k] -= PitchUV2[k] * iBlkHeightC;
    }
  }

  for (int n = 1; n < (_trad * 2 + 1); n++)
  {
    if (Wall[n] != 0) iNumNZBlocks++;
  }

  if (iNumNZBlocks < 3) // current and at least 2 refs are non-zero weighted
  {
    // nothing to process - return 0 to stop proc
    return 0;
  }
  else
  {
    // fill initial max values
    for (int k = 0; k < (_trad * 2 + 1); k++)
    {
      sad_array_sub[k] = veryBigSAD;
      sad_array_add[k] = veryBigSAD;
    }

    sad_t stAVG_sub_SAD = 0;
    sad_t stAVG_add_SAD = 0;
    int iNumAVG = 0;

    sad_t sad_chroma;
    sad_t luma_sad;

    // process current block too
    // subtracted
    if (!MPB_PartBlend) // use subtraction
    {
      if (_cpuFlags & CPUF_SSE2)
      {
        SubtractBlock_uint8_sse2(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
          pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
        if ((MPBchroma & 0x1) != 0)
        {
          SubtractBlock_uint8_sse2(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pCurrUV1, iCurrPitchUV1, Wall[0], iBlkWidthC, iBlkHeightC);
          SubtractBlock_uint8_sse2(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pCurrUV2, iCurrPitchUV2, Wall[0], iBlkWidthC, iBlkHeightC);
        }
      }
      else
      {
        SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
          pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
        if ((MPBchroma & 0x1) != 0)
        {
          SubtractBlock_C_uint8(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pCurrUV1, iCurrPitchUV1, Wall[0], iBlkWidthC, iBlkHeightC);
          SubtractBlock_C_uint8(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pCurrUV2, iCurrPitchUV2, Wall[0], iBlkWidthC, iBlkHeightC);
        }
      }
    }
    else // use real partial blending
    {
      for (int i = 0; i < (_trad * 2 + 1); i++)
        W_sub[i] = Wall[i];
      W_sub[0] = 0; // zero weight of current block

      norm_weights_all(W_sub, _trad);

      _degrainluma_ptr(
        pMPBTempBlocks + (iBlockSizeMem * (1)), 0, iBlocksPitch,
        pCurr, iCurrPitch,
        pRef, Pitch, W_sub, _trad);
      if ((MPBchroma & 0x1) != 0)
      {
        _degrainchroma_ptr(
          pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), 0, iBlocksPitchUV,
          pCurrUV1, iCurrPitchUV1,
          pRefUV1, PitchUV1, W_sub, _trad);

        _degrainchroma_ptr(
          pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), 0, iBlocksPitchUV,
          pCurrUV2, iCurrPitchUV2,
          pRefUV2, PitchUV2, W_sub, _trad);
      }

      // always rewind pRef pointers after full blending
      for (int k = 0; k < _trad * 2; k++)
      {
        pRef[k] -= Pitch[k] * iBlkHeight;
        if ((MPBchroma & 0x1) != 0)
        {
          pRefUV1[k] -= PitchUV1[k] * iBlkHeightC;
          pRefUV2[k] -= PitchUV2[k] * iBlkHeightC;
        }
      }

    }

/*    luma_sad = SAD(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
    sad_chroma = ScaleSadChroma(SADCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV)
          + SADCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV), chromaSADscale);
*/
    luma_sad = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
    if ((MPBchroma & 0x1) != 0)
    {
      sad_chroma = ScaleSadChroma(DM_Chroma->GetDisMetric(pMPBTempBlocksUV1, iBlocksPitchUV, pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV)
        + DM_Chroma->GetDisMetric(pMPBTempBlocksUV2, iBlocksPitchUV, pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV), chromaSADscale);
    }
    else
      sad_chroma = 0;
    sad_array_sub[0] = luma_sad + sad_chroma;
    stAVG_sub_SAD += sad_array_sub[0];

    // calc SAD of full blended vs refs
/*    luma_sad = SAD(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
    sad_chroma = ScaleSadChroma(SADCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pCurrUV1, iCurrPitchUV1)
      + SADCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pCurrUV2, iCurrPitchUV2), chromaSADscale);*/
    luma_sad = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
    if ((MPBchroma & 0x1) != 0)
    {
      sad_chroma = ScaleSadChroma(DM_Chroma->GetDisMetric(pMPBTempBlocksUV1, iBlocksPitchUV, pCurrUV1, iCurrPitchUV1)
        + DM_Chroma->GetDisMetric(pMPBTempBlocksUV2, iBlocksPitchUV, pCurrUV2, iCurrPitchUV2), chromaSADscale);
    }
    else
      sad_chroma = 0;
    sad_array_add[0] = luma_sad + sad_chroma;
    stAVG_add_SAD += sad_array_add[0];

    iNumAVG++;

    // ref blocks
    for (int n = 1; n < (_trad * 2 + 1); n++)
    {
      if (Wall[n] != 0)
      {
        if (!MPB_PartBlend)
        {
          // create subrtracted versions of full blended block (faster)
          if (_cpuFlags & CPUF_SSE2)
          {
            SubtractBlock_uint8_sse2(pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch,
              pMPBTempBlocks, iBlocksPitch, (uint8_t*)pRef[n - 1], Pitch[n - 1], Wall[n], iBlkWidth, iBlkHeight);
            if ((MPBchroma & 0x1) != 0)
            {
              SubtractBlock_uint8_sse2(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pRefUV1[n - 1], PitchUV1[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
              SubtractBlock_uint8_sse2(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pRefUV2[n - 1], PitchUV2[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
            }
          }
          else
          {
            SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch,
              pMPBTempBlocks, iBlocksPitch, (uint8_t*)pRef[n - 1], Pitch[n - 1], Wall[n], iBlkWidth, iBlkHeight);
            if ((MPBchroma & 0x1) != 0)
            {
              SubtractBlock_C_uint8(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pRefUV1[n - 1], PitchUV1[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
              SubtractBlock_C_uint8(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pRefUV2[n - 1], PitchUV2[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
            }
          }
        }
        else // real partial blending (slower)
        {
          for (int i = 0; i < (_trad * 2 + 1); i++)
            W_sub[i] = Wall[i];
          W_sub[n] = 0; // zero weight of n-th block
          norm_weights_all(W_sub, _trad);

          _degrainluma_ptr(
            pMPBTempBlocks + (iBlockSizeMem * (n + 1)), 0, iBlocksPitch,
            pCurr, iCurrPitch,
            pRef, Pitch, W_sub, _trad);
          if ((MPBchroma & 0x1) != 0)
          {
            _degrainchroma_ptr(
              pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), 0, iBlocksPitchUV,
              pCurrUV1, iCurrPitchUV1,
              pRefUV1, PitchUV1, W_sub, _trad);

            _degrainchroma_ptr(
              pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), 0, iBlocksPitchUV,
              pCurrUV2, iCurrPitchUV2,
              pRefUV2, PitchUV2, W_sub, _trad);
          }
          // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
          for (int k = 0; k < _trad * 2; k++)
          {
            pRef[k] -= Pitch[k] * iBlkHeight;
            if ((MPBchroma & 0x1) != 0)
            {
              pRefUV1[k] -= PitchUV1[k] * iBlkHeightC;
              pRefUV2[k] -= PitchUV2[k] * iBlkHeightC;
            }
          }
        }

        //calc SAD of full blended block vs subtracted
/*        luma_sad = SAD(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
        sad_chroma = ScaleSadChroma(SADCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV)
          + SADCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV), chromaSADscale);*/
        luma_sad = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
        if ((MPBchroma & 0x1) != 0)
        {
          sad_chroma = ScaleSadChroma(DM_Chroma->GetDisMetric(pMPBTempBlocksUV1, iBlocksPitchUV, pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV)
            + DM_Chroma->GetDisMetric(pMPBTempBlocksUV2, iBlocksPitchUV, pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV), chromaSADscale);
        }
        else
          sad_chroma = 0;
        sad_array_sub[n] = luma_sad + sad_chroma;
        stAVG_sub_SAD += sad_array_sub[n];

        // calc SAD of full blended vs refs
/*        luma_sad = SAD(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
        sad_chroma = ScaleSadChroma(SADCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pRefUV1[n - 1], PitchUV1[n - 1])
          + SADCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pRefUV2[n - 1], PitchUV2[n - 1]), chromaSADscale);*/
        luma_sad = DM_Luma->GetDisMetric(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
        if ((MPBchroma & 0x1) != 0)
        {
          sad_chroma = ScaleSadChroma(DM_Chroma->GetDisMetric(pMPBTempBlocksUV1, iBlocksPitchUV, pRefUV1[n - 1], PitchUV1[n - 1])
            + DM_Chroma->GetDisMetric(pMPBTempBlocksUV2, iBlocksPitchUV, pRefUV2[n - 1], PitchUV2[n - 1]), chromaSADscale);
        }
        else
          sad_chroma = 0;
        sad_array_add[n] = luma_sad + sad_chroma;
        stAVG_add_SAD += sad_array_add[n];

        iNumAVG++;
      }

    }

    // find average SAD
    stAVG_add_SAD /= iNumAVG;
    stAVG_sub_SAD /= iNumAVG;

    //check if SAD of curr block too differs from average
    for (int n = 0; n < (_trad * 2 + 1); n++)
    {
      if (Wall[n] != 0) // check only processed above blocks
      {
        // check MV length ?
        if (n != 0) // skip current block
        {
          VECTOR vCurr = pMVsWorkPlanesArrays[n - 1][iBlkNum]; // n from current block ?
          int iLengthvCurr_sq = ((vCurr.x * vCurr.x) + (vCurr.y * vCurr.y)); // >> lognPel ?
          if (iLengthvCurr_sq > MPB_MVlth)
          {
            continue; // skip weight adjustment for too long MVs ?
          }
        }
//        if (stAVG_sub_SAD - sad_array_sub[n] > MPBthSub)
        if ((stAVG_sub_SAD - sad_array_sub[n] > MPBthSub) && (sad_array_add[n] - stAVG_add_SAD > MPBthAdd)) // block worse than avg ?
        {
          Wall[n] = (int)((float)Wall[n] * MPB_SPC_sub); // decrease weight
          iNumAlignedBlocks++;
          continue; // do not check for addition if already adjusted to lower weight ?
        }

//        if (stAVG_add_SAD - sad_array_add[n] > MPBthAdd)
        if ((sad_array_sub[n] - stAVG_sub_SAD > MPBthSub) && (stAVG_add_SAD - sad_array_add[n] > MPBthAdd)) // block beter than avg ?
        {
          int iNewW = (int)((float)Wall[n] * MPB_SPC_add); // increase weight 
          if (iNewW > 255) iNewW = 255;
          Wall[n] = iNewW;
          iNumAlignedBlocks++;
        }
      }
    }

  }

  if (iNumAlignedBlocks != 0)
  {
    norm_weights_all(Wall, _trad);
  }

  return iNumAlignedBlocks; // counter of weight-adjusted blocks, 0 if none
}

MV_FORCEINLINE int MDegrainN::AlignBlockWeightsLC_CV(const BYTE* pRef[], int Pitch[],
  const BYTE* pRefUV1[], int PitchUV1[],
  const BYTE* pRefUV2[], int PitchUV2[],
  const BYTE* pCurr, const int iCurrPitch,
  const BYTE* pCurrUV1, const int iCurrPitchUV1,
  const BYTE* pCurrUV2, const int iCurrPitchUV2,
  int Wall[], const int iBlkWidth, const int iBlkHeight,
  const int iBlkWidthC, const int iBlkHeightC, const int chromaSADscale,
  int iBlkNum
)
{
  //first count number of non-zero weights, zero is current block weight, 1,2 is +-1frame and so on
  int iNumNZBlocks = 1; // we have at least one non-zero - the source itself ?
  const int iBlockSizeMem = iBlkWidth * iBlkHeight * pixelsize;
  const int iBlockSizeMemUV = iBlkWidthC * iBlkHeightC * pixelsize;
  const int iBlocksPitch = iBlkWidth * pixelsize;
  const int iBlocksPitchUV = iBlkWidthC * pixelsize;
  int iNumAlignedBlocks = 0;
  int W_sub[MAX_TEMP_RAD * 2 + 1];

  float cv_array_sub[(MAX_TEMP_RAD * 2 + 1)];
  float cv_array_add[(MAX_TEMP_RAD * 2 + 1)];

  // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
  for (int k = 0; k < _trad * 2; k++)
  {
    pRef[k] -= Pitch[k] * iBlkHeight;
    if ((MPBchroma & 0x1) != 0)
    {
      pRefUV1[k] -= PitchUV1[k] * iBlkHeightC;
      pRefUV2[k] -= PitchUV2[k] * iBlkHeightC;
    }
  }

  for (int n = 1; n < (_trad * 2 + 1); n++)
  {
    if (Wall[n] != 0) iNumNZBlocks++;
  }

  if (iNumNZBlocks < 3) // current and at least 2 refs are non-zero weighted
  {
    // nothing to process - return 0 to stop proc
    return 0;
  }
  else
  {
    float fAVG_sub_CV = 0;
    float fAVG_add_CV = 0;

    int iNumAVG = 0;

    float chroma_cv;
    float luma_cv;

    // process current block too
    // subtracted
    if (!MPB_PartBlend) // use subtraction
    {
      if (_cpuFlags & CPUF_SSE2)
      {
        SubtractBlock_uint8_sse2(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
          pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
        if ((MPBchroma & 0x1) != 0)
        {
          SubtractBlock_uint8_sse2(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pCurrUV1, iCurrPitchUV1, Wall[0], iBlkWidthC, iBlkHeightC);
          SubtractBlock_uint8_sse2(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pCurrUV2, iCurrPitchUV2, Wall[0], iBlkWidthC, iBlkHeightC);
        }
      }
      else
      {
        SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch,
          pMPBTempBlocks, iBlocksPitch, (uint8_t*)pCurr, iCurrPitch, Wall[0], iBlkWidth, iBlkHeight);
        if ((MPBchroma & 0x1) != 0)
        {
          SubtractBlock_C_uint8(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pCurrUV1, iCurrPitchUV1, Wall[0], iBlkWidthC, iBlkHeightC);
          SubtractBlock_C_uint8(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV,
            pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pCurrUV2, iCurrPitchUV2, Wall[0], iBlkWidthC, iBlkHeightC);
        }
      }
    }
    else // use real partial blending
    {
      for (int i = 0; i < (_trad * 2 + 1); i++)
        W_sub[i] = Wall[i];
      W_sub[0] = 0; // zero weight of current block

      norm_weights_all(W_sub, _trad);

      _degrainluma_ptr(
        pMPBTempBlocks + (iBlockSizeMem * (1)), 0, iBlocksPitch,
        pCurr, iCurrPitch,
        pRef, Pitch, W_sub, _trad);
      if ((MPBchroma & 0x1) != 0)
      {
        _degrainchroma_ptr(
          pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), 0, iBlocksPitchUV,
          pCurrUV1, iCurrPitchUV1,
          pRefUV1, PitchUV1, W_sub, _trad);

        _degrainchroma_ptr(
          pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), 0, iBlocksPitchUV,
          pCurrUV2, iCurrPitchUV2,
          pRefUV2, PitchUV2, W_sub, _trad);
      }

      // always rewind pRef pointers after full blending
      for (int k = 0; k < _trad * 2; k++)
      {
        pRef[k] -= Pitch[k] * iBlkHeight;
        if ((MPBchroma & 0x1) != 0)
        {
          pRefUV1[k] -= PitchUV1[k] * iBlkHeightC;
          pRefUV2[k] -= PitchUV2[k] * iBlkHeightC;
        }
      }

    }

    luma_cv = COVAR(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (1)), iBlocksPitch);
    if ((MPBchroma & 0x1) != 0)
    {
      chroma_cv = ScaleSadChroma(COVARCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pMPBTempBlocksUV1 + (iBlockSizeMemUV * (1)), iBlocksPitchUV)
        + COVARCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pMPBTempBlocksUV2 + (iBlockSizeMemUV * (1)), iBlocksPitchUV), chromaSADscale);
    }
    else
    {
      chroma_cv = 0;
    }
    cv_array_sub[0] = luma_cv + chroma_cv;
    fAVG_sub_CV += cv_array_sub[0];

    // calc covarince of full blended vs refs
    luma_cv = COVAR(pMPBTempBlocks, iBlocksPitch, pCurr, iCurrPitch);
    if ((MPBchroma & 0x1) != 0)
    {
      chroma_cv = ScaleSadChroma(COVARCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pCurrUV1, iCurrPitchUV1)
        + COVARCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pCurrUV2, iCurrPitchUV2), chromaSADscale);
    }
    else
    {
      chroma_cv = 0;
    }
    cv_array_add[0] = luma_cv + chroma_cv;
    fAVG_add_CV += cv_array_add[0];

    iNumAVG++;

    // ref blocks
    for (int n = 1; n < (_trad * 2 + 1); n++)
    {
      if (Wall[n] != 0)
      {
        if (!MPB_PartBlend)
        {
          // create subrtracted versions of full blended block (faster)
          if (_cpuFlags & CPUF_SSE2)
          {
            SubtractBlock_uint8_sse2(pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch,
              pMPBTempBlocks, iBlocksPitch, (uint8_t*)pRef[n - 1], Pitch[n - 1], Wall[n], iBlkWidth, iBlkHeight);
            if ((MPBchroma & 0x1) != 0)
            {
              SubtractBlock_uint8_sse2(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pRefUV1[n - 1], PitchUV1[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
              SubtractBlock_uint8_sse2(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pRefUV2[n - 1], PitchUV2[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
            }
          }
          else
          {
            SubtractBlock_C_uint8(pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch,
              pMPBTempBlocks, iBlocksPitch, (uint8_t*)pRef[n - 1], Pitch[n - 1], Wall[n], iBlkWidth, iBlkHeight);
            if ((MPBchroma & 0x1) != 0)
            {
              SubtractBlock_C_uint8(pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV1, iBlocksPitchUV, (uint8_t*)pRefUV1[n - 1], PitchUV1[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
              SubtractBlock_C_uint8(pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV,
                pMPBTempBlocksUV2, iBlocksPitchUV, (uint8_t*)pRefUV2[n - 1], PitchUV2[n - 1], Wall[n], iBlkWidthC, iBlkHeightC);
            }
          }
        }
        else // real partial blending (slower)
        {
          for (int i = 0; i < (_trad * 2 + 1); i++)
            W_sub[i] = Wall[i];
          W_sub[n] = 0; // zero weight of n-th block
          norm_weights_all(W_sub, _trad);

          _degrainluma_ptr(
            pMPBTempBlocks + (iBlockSizeMem * (n + 1)), 0, iBlocksPitch,
            pCurr, iCurrPitch,
            pRef, Pitch, W_sub, _trad);
          if ((MPBchroma & 0x1) != 0)
          {
            _degrainchroma_ptr(
              pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), 0, iBlocksPitchUV,
              pCurrUV1, iCurrPitchUV1,
              pRefUV1, PitchUV1, W_sub, _trad);

            _degrainchroma_ptr(
              pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), 0, iBlocksPitchUV,
              pCurrUV2, iCurrPitchUV2,
              pRefUV2, PitchUV2, W_sub, _trad);
          }
          // always rewind pRef pointers after full blending, hope after zeroing weight of block it will never put back ?
          for (int k = 0; k < _trad * 2; k++)
          {
            pRef[k] -= Pitch[k] * iBlkHeight;
            if ((MPBchroma & 0x1) != 0)
            {
              pRefUV1[k] -= PitchUV1[k] * iBlkHeightC;
              pRefUV2[k] -= PitchUV2[k] * iBlkHeightC;
            }
          }
        }

        //calc covariance of full blended block vs subtracted
        luma_cv = COVAR(pMPBTempBlocks, iBlocksPitch, pMPBTempBlocks + (iBlockSizeMem * (n + 1)), iBlocksPitch);
        if ((MPBchroma & 0x1) != 0)
        {
          chroma_cv = ScaleSadChroma(COVARCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pMPBTempBlocksUV1 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV)
            + COVARCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pMPBTempBlocksUV2 + (iBlockSizeMemUV * (n + 1)), iBlocksPitchUV), chromaSADscale);
        }
        else
        {
          chroma_cv = 0;
        }
        cv_array_sub[n] = luma_cv + chroma_cv;
        fAVG_sub_CV += cv_array_sub[n];

        // calc covariance of full blended vs refs
        luma_cv = COVAR(pMPBTempBlocks, iBlocksPitch, pRef[n - 1], Pitch[n - 1]);
        if ((MPBchroma & 0x1) != 0)
        {
           chroma_cv = ScaleSadChroma(COVARCHROMA(pMPBTempBlocksUV1, iBlocksPitchUV, pRefUV1[n - 1], PitchUV1[n - 1])
            + COVARCHROMA(pMPBTempBlocksUV2, iBlocksPitchUV, pRefUV2[n - 1], PitchUV2[n - 1]), chromaSADscale);
        }
        else
        {
          chroma_cv = 0;
        }

        cv_array_add[n] = luma_cv + chroma_cv;
        fAVG_add_CV += cv_array_add[n];

        iNumAVG++;
      }

    }

    // find average covariance
    fAVG_add_CV /= iNumAVG;
    fAVG_sub_CV /= iNumAVG;

    //check if CV of curr block too differs from average
    for (int n = 0; n < (_trad * 2 + 1); n++)
    {
      if (Wall[n] != 0) // check only processed above blocks
      {
        // check MV length ?
        if (n != 0) // skip current block
        {
          VECTOR vCurr = pMVsWorkPlanesArrays[n - 1][iBlkNum]; // n from current block ?
          int iLengthvCurr_sq = ((vCurr.x * vCurr.x) + (vCurr.y * vCurr.y)); // >> lognPel ?
          if (iLengthvCurr_sq > MPB_MVlth)
          {
            continue; // skip weight adjustment for too long MVs ?
          }
        }
        if (n != 0) // do not decrease weight of current block ?
        {
          if ((cv_array_sub[n] - fAVG_sub_CV > MPBthSub) && (fAVG_add_CV - cv_array_add[n] > MPBthAdd)) // block worse than avg ?
          {
            Wall[n] = (int)((float)Wall[n] * MPB_SPC_sub); // decrease weight
            iNumAlignedBlocks++;
            continue; // do not check for addition if already adjusted to lower weight ?
          }
        }

        if ((fAVG_sub_CV - cv_array_sub[n] > MPBthAdd) && (cv_array_add[n] - fAVG_add_CV > MPBthAdd)) // block better than avg ?
        {
          int iNewW = (int)((float)Wall[n] * MPB_SPC_add); // increase weight 
          if (iNewW > 255) iNewW = 255;
          Wall[n] = iNewW;
          iNumAlignedBlocks++;
          continue; // do not check for addition if already adjusted to lower weight ?
        }
        
      }
    }

  }

  if (iNumAlignedBlocks != 0)
  {
    norm_weights_all(Wall, _trad);
  }

  return iNumAlignedBlocks; // counter of weight-adjusted blocks, 0 if none
}


MV_FORCEINLINE void MDegrainN::CopyBlock(uint8_t* pDst, int iDstPitch, uint8_t* pSrc, int iBlkWidth, int iBlkHeight)
{
  for (int iLine = 0; iLine < iBlkHeight; iLine++)
  {
    memcpy(pDst, pSrc, iBlkWidth*pixelsize);
    pDst += iDstPitch;
    pSrc += iBlkWidth * pixelsize;
  }
}

MV_FORCEINLINE bool MDegrainN::isMVsStable(VECTOR** pMVsPlanesArrays, int iNumBlock, int wref_arr[])
{
  VECTOR blockMVs[(MAX_TEMP_RAD * 2) + 1];
  int iNumMVs2Proc = 0;

  VECTOR** pMVSPlanesArrays_working;

  if (mvmultivs != 0)
  {
    pMVSPlanesArrays_working = (VECTOR**)pMVsPlanesArraysVS;
  }
  else
    pMVSPlanesArrays_working = pMVsPlanesArrays;

  VECTOR blockMVs_sq[(MAX_TEMP_RAD * 2) + 1];
  // convert +1, -1, +2, -2, +3, -3 ... to
// -3, -2, -1, 0, +1, +2, +3 timed sequence
  for (int k = 0; k < _trad; ++k)
  {
    blockMVs_sq[k] = pMVSPlanesArrays_working[(_trad - k - 1) * 2 + 1][iNumBlock];
  }

  blockMVs_sq[_trad].x = 0; // zero trad - source block itself
  blockMVs_sq[_trad].y = 0;
  blockMVs_sq[_trad].sad = 0;

  for (int k = 1; k < _trad + 1; ++k)
  {
    blockMVs_sq[k + _trad] = pMVSPlanesArrays_working[(k - 1) * 2][iNumBlock];
  }

  // velocity Vs (N-1)
  int iVx[(MAX_TEMP_RAD * 2) + 1];
  int iVy[(MAX_TEMP_RAD * 2) + 1];

  //acceleration (N-2)
  int iAx[(MAX_TEMP_RAD * 2) + 1];
  int iAy[(MAX_TEMP_RAD * 2) + 1];

  //delta angles (relative angle velocity per interframe time ?)
  float fDDA[(MAX_TEMP_RAD * 2) + 1];

  // velocity X, Y, linear and vectors rotational
  for (int n = 0; n < (_trad * 2); n++)
  {
    VECTOR v1 = blockMVs_sq[n];
    VECTOR v2 = blockMVs_sq[n+1];
    iVx[n] = v2.x - v1.x;
    iVy[n] = v2.y - v1.y;

    fDDA[n] = fDiffAngleVect(v1.x, v1.y, v2.x, v2.y);
  }

  // acceleration X, Y 
  for (int n = 0; n < (_trad * 2) - 1; n++)
  {
    iAx[n] = iVx[n + 1] - iVx[n];
    iAy[n] = iVy[n + 1] - iVy[n];
  }

  // calc max sum A
/*  int iMaxAsq = 0;
  for (int n = 0; n < (_trad * 2) - 1; n++)
  {
    int Asq = iAx[n] * iAx[n] + iAy[n] * iAy[n];
    if (Asq > iMaxAsq) iMaxAsq = Asq;
  }
  */
  int iSumAsq = 0;
  float fSumDDA = 0.0f;
  for (int n = 0; n < (_trad * 2) - 1; n++)
  {
    int Asq = iAx[n] * iAx[n] + iAy[n] * iAy[n];
    iSumAsq += Asq;
    fSumDDA += fDDA[n];
  }

  int iTotalDif = (int)((float)iSumAsq * fSumDDA);

  if (iTotalDif > MPB_thIVS)
    return false;

  return true;


  /*
  //calc max length of MVs
  int iMaxLength_sq = 0;
  for (int n = 0; n < (_trad * 2 + 1); n++)
  {
    VECTOR v1 = blockMVs_sq[n];

    int iLength_sq = (v1.x ) * (v1.x ) + (v1.y ) * (v1.y);
    if (iLength_sq > iMaxLength_sq) iMaxLength_sq = iLength_sq;

  }

  if (iMaxLength_sq > MPB_thIVS)
    return false;

  return true;
  */

  /*
  //calc avg dif length of MVs
  int iTotalDifLength_sq = 0;
  for (int n = 0; n < (_trad * 2); n++)
  {
    VECTOR v1 = blockMVs_sq[n];
    VECTOR v2 = blockMVs_sq[n + 1];

    int iDifLength_sq = (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y);
    iTotalDifLength_sq += iDifLength_sq;
  }

  int iAvgLength_sq = iTotalDifLength_sq / (_trad * 2);

  if (iAvgLength_sq > MPB_thIVS)
    return false;

  return true;

  */
  /*

  for (int k = 0; k < _trad * 2; ++k)
  {
    if (wref_arr[k + 1] != 0)
    {
      blockMVs[iNumMVs2Proc] = pMVsPlanesArrays[k][iNumBlock];
      iNumMVs2Proc++;
    }
  }

  if (iNumMVs2Proc < 2) return false;

  //calc avg angle between series of MVs ?

  //calc avg dif length of MVs
  int iTotalDifLength_sq = 0;
  for (int n = 0; n < iNumMVs2Proc - 1; n++)
  {
    VECTOR v1 = blockMVs[n];
    VECTOR v2 = blockMVs[n + 1];

    int iDifLength_sq = (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y);
    iTotalDifLength_sq += iDifLength_sq;
  }

  int iAvgLength_sq = iTotalDifLength_sq / (iNumMVs2Proc - 1);

  if (iAvgLength_sq > MPB_thIVS)
    return false;

  return true;
  */
}

MV_FORCEINLINE void MDegrainN::MPB_SP(
  BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  const BYTE* pRef[], int Pitch[],
  int Wall[], const int iBlkWidth, const int iBlkHeight,
  bool bChroma, int iBlkNum
)
{
  int adjWarr[1 + MAX_TEMP_RAD * 2]; 
  int startWarr[1 + MAX_TEMP_RAD * 2]; 

  if (showIVSmask) // may be not best place but less places in text to place
  {
    for (int h = 0; h < iBlkHeight; h++)
    {
      for (int w = 0; w < iBlkWidth; w++)
      {
        pDst[h * nDstPitch + w] = 0;
      }
    }
    return;
  }

  // start check from equal weights (equal to wpow=7 ?), only use max MPGtgtTR number of init non zero weights
  int iNumUsedWeights = MPBtgtTR * 2;

  // if MPB_SPC_add > 10 - skip iterative MPB, output only trimmed Wall
  if (MPB_SPC_add > 10.0f)
  {
    for (int i = 0; i < iNumUsedWeights + 1; i++)
    {
      adjWarr[i] = Wall[i];
    }

    for (int i = iNumUsedWeights + 1; i < (_trad * 2) + 1; i++)
    {
      adjWarr[i] = 0;
    }

    norm_weights_all(adjWarr, MPBtgtTR);

    // make final blend to output
    if (!bChroma)
    {
      _degrainluma_ptr(
        pDst, pDstLsb, nDstPitch,
        pSrc, nSrcPitch,
        pRef, Pitch, adjWarr, MPBtgtTR
      );
    }
    else
    {
      _degrainchroma_ptr(
        pDst, pDstLsb, nDstPitch,
        pSrc, nSrcPitch,
        pRef, Pitch, adjWarr, MPBtgtTR
      );
    }

    return;
  }

  // also set full weight to current block
  adjWarr[0] = 1 << DEGRAIN_WEIGHT_BITS;

  for (int i = 1; i < (_trad * 2) + 1; i++)
  {
    if ((Wall[i] != 0) && (iNumUsedWeights != 0))
    {
      {
        adjWarr[i] = 1 << DEGRAIN_WEIGHT_BITS;
        iNumUsedWeights--;
      }
    }
    else
      adjWarr[i] = 0;
  }

  norm_weights_all(adjWarr, _trad);

  // prepare target
  if (!bChroma)
  {
    _degrainluma_ptr(
      pMPBTempBlocks, 0, (iBlkWidth * pixelsize),
      pSrc, nSrcPitch,
      pRef, Pitch, adjWarr, _trad
    );
  }
  else
  {
    _degrainchroma_ptr(
      pMPBTempBlocks, 0, (iBlkWidth * pixelsize),
      pSrc, nSrcPitch,
      pRef, Pitch, adjWarr, _trad
    );
  }

  // set equal weights to all possible blocks in tr-scope with SAD > thSAD
// also set full weight to current block
  adjWarr[0] = 1 << DEGRAIN_WEIGHT_BITS;

  for (int i = 1; i < (_trad * 2) + 1; i++)
  {
    if (Wall[i] != 0)
    {
      {
        adjWarr[i] = 1 << DEGRAIN_WEIGHT_BITS;
      }
    }
    else
      adjWarr[i] = 0;
  }

  norm_weights_all(adjWarr, _trad);

  // copy start weights to start weights array
  for (int i = 0; i < (_trad * 2) + 1; i++)
  {
    startWarr[i] = adjWarr[i];
  }

  int iNumItCurr = MPBNumIt;
  do
  {
    int iNumAlignedBlocks = AlignBlockWeights(
      pRef, Pitch,
      pSrc, nSrcPitch,
      adjWarr, iBlkWidth, iBlkHeight, bChroma, iBlkNum
    );

    iNumItCurr--;

    if ((iNumAlignedBlocks == 0) || (iNumItCurr == 0))
    {
      break;
    }

    // each iteration blend
    if (!bChroma)
    {
      _degrainluma_ptr(
        pMPBTempBlocks, 0, (iBlkWidth * pixelsize),
        pSrc, nSrcPitch,
        pRef, Pitch, adjWarr, _trad
      );
    }
    else
    {
      _degrainchroma_ptr(
        pMPBTempBlocks, 0, (iBlkWidth * pixelsize),
        pSrc, nSrcPitch,
        pRef, Pitch, adjWarr, _trad
      );
    }

  } while (1);

  // adjust Wall weights proportionally to found weights
  for (int i = 0; i < (_trad * 2) + 1; i++)
  {
    if (startWarr[i] != 0)
    {
      float fRatio = (float)adjWarr[i] / (float)startWarr[i];
      int weight = Wall[i];
      weight = (int)((float)weight * fRatio + 0.5f);
      Wall[i] = weight;
    }
  }

  norm_weights_all(Wall, _trad);

  // make final blend to output
  if (!bChroma)
  {
    _degrainluma_ptr(
      pDst, pDstLsb, nDstPitch,
      pSrc, nSrcPitch,
      pRef, Pitch, Wall, _trad
    );
  }
  else
  {
    _degrainchroma_ptr(
      pDst, pDstLsb, nDstPitch,
      pSrc, nSrcPitch,
      pRef, Pitch, Wall, _trad
    );
  }

}

MV_FORCEINLINE void MDegrainN::MPB_LC(
  BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
  const BYTE* pSrc, int nSrcPitch,
  const BYTE* pRef[], int Pitch[],
  BYTE* pDstUV1, BYTE* pDstLsbUV1, int nDstPitchUV1,
  const BYTE* pSrcUV1, int nSrcPitchUV1,
  const BYTE* pRefUV1[], int PitchUV1[],
  BYTE* pDstUV2, BYTE* pDstLsbUV2, int nDstPitchUV2,
  const BYTE* pSrcUV2, int nSrcPitchUV2,
  const BYTE* pRefUV2[], int PitchUV2[],
  int Wall[], int WallC[], const int iBlkWidth, const int iBlkHeight,
  const int iBlkWidthC, const int iBlkHeightC, const int chromaSADscale,
  int iBlkNum
)
{
  // TEMP DEBUG !!!
//  pRef[1] ++; // shift ptr second ref block 1 to the right !!!

  if (showIVSmask) // may be not best place but less places in text to place
  {
    for (int h = 0; h < iBlkHeight; h++)
    {
      for (int w = 0; w < iBlkWidth; w++)
      {
        pDst[h * nDstPitch + w] = 0;
      }
    }
    return;
  }

  int adjWarr[1 + MAX_TEMP_RAD * 2];
  int startWarr[1 + MAX_TEMP_RAD * 2];

  // start check from equal weights (equal to wpow=7 ?), only use max MPGtgtTR number of init non zero weights
  int iNumUsedWeights = MPBtgtTR * 2;

  // if MPB_SPC_add > 10 - skip iterative MPB, output only trimmed Wall
  if (MPB_SPC_add > 10.0f)
  {
    for (int i = 0; i < iNumUsedWeights + 1; i++)
    {
      adjWarr[i] = Wall[i];
    }

    for (int i = iNumUsedWeights + 1; i < (_trad * 2) + 1; i++)
    {
        adjWarr[i] = 0;
    }

    norm_weights_all(adjWarr, MPBtgtTR);

    // make final blend to output
    _degrainluma_ptr(
      pDst, pDstLsb, nDstPitch,
      pSrc, nSrcPitch,
      pRef, Pitch, adjWarr, MPBtgtTR
    );

    // chroma first plane
    _degrainchroma_ptr(
      pDstUV1, pDstLsbUV1, nDstPitchUV1,
      pSrcUV1, nSrcPitchUV1,
      pRefUV1, PitchUV1, adjWarr, MPBtgtTR
    );

    // chroma second plane
    _degrainchroma_ptr(
      pDstUV2, pDstLsbUV2, nDstPitchUV2,
      pSrcUV2, nSrcPitchUV2,
      pRefUV2, PitchUV2, adjWarr, MPBtgtTR
    );

    return;
  }

  iNumUsedWeights = MPBtgtTR * 2;

  // also set full weight to current block
  adjWarr[0] = 1 << DEGRAIN_WEIGHT_BITS;

  for (int i = 1; i < (_trad * 2) + 1; i++)
  {
    if ((Wall[i] != 0) && (iNumUsedWeights != 0))
    {
      {
        adjWarr[i] = 1 << DEGRAIN_WEIGHT_BITS;
        iNumUsedWeights--;
      }
    }
    else
      adjWarr[i] = 0;
  }

  norm_weights_all(adjWarr, _trad);

  // prepare target
  _degrainluma_ptr(
    pMPBTempBlocks, 0, (iBlkWidth * pixelsize),
    pSrc, nSrcPitch,
    pRef, Pitch, adjWarr, _trad
  );

  if ((MPBchroma & 0x1) != 0)
  {
    _degrainchroma_ptr(
      pMPBTempBlocksUV1, 0, (iBlkWidthC * pixelsize),
      pSrcUV1, nSrcPitchUV1,
      pRefUV1, PitchUV1, adjWarr, _trad
    );

    _degrainchroma_ptr(
      pMPBTempBlocksUV2, 0, (iBlkWidthC * pixelsize),
      pSrcUV2, nSrcPitchUV2,
      pRefUV2, PitchUV2, adjWarr, _trad
    );
  }

  // set equal weights to all possible blocks in tr-scope with SAD > thSAD
  // also set full weight to current block
  adjWarr[0] = 1 << DEGRAIN_WEIGHT_BITS;

  for (int i = 1; i < (_trad * 2) + 1; i++)
  {
    if (Wall[i] != 0)
    {
      {
        adjWarr[i] = 1 << DEGRAIN_WEIGHT_BITS;
      }
    }
    else
      adjWarr[i] = 0;
  }

  norm_weights_all(adjWarr, _trad);

  // copy start weights to start weights array
  for (int i = 0; i < (_trad * 2) + 1; i++)
  {
    startWarr[i] = adjWarr[i];
  }

  int iNumItCurr = MPBNumIt;
  do
  {
    int iNumAlignedBlocks;

    if (MPB_DMFlags & MEF_COVAR)
    {
      iNumAlignedBlocks = AlignBlockWeightsLC_CV(
        pRef, Pitch,
        pRefUV1, PitchUV1,
        pRefUV2, PitchUV2,
        pSrc, nSrcPitch,
        pSrcUV1, nSrcPitchUV1,
        pSrcUV2, nSrcPitchUV2,
        adjWarr, iBlkWidth, iBlkHeight,
        iBlkWidthC, iBlkHeightC,
        chromaSADscale, iBlkNum
      );
    }
    else
    {
      iNumAlignedBlocks = AlignBlockWeightsLC(
        pRef, Pitch,
        pRefUV1, PitchUV1,
        pRefUV2, PitchUV2,
        pSrc, nSrcPitch,
        pSrcUV1, nSrcPitchUV1,
        pSrcUV2, nSrcPitchUV2,
        adjWarr, iBlkWidth, iBlkHeight,
        iBlkWidthC, iBlkHeightC,
        chromaSADscale, iBlkNum
      );

    }


    iNumItCurr--;

    if ((iNumAlignedBlocks == 0) || (iNumItCurr == 0))
    {
      break;
    }

    // each iteration blend
    _degrainluma_ptr(
      pMPBTempBlocks, 0, (iBlkWidth * pixelsize),
      pSrc, nSrcPitch,
      pRef, Pitch, adjWarr, _trad
    );

    if ((MPBchroma & 0x1) != 0)
    {
      _degrainchroma_ptr(
        pMPBTempBlocksUV1, 0, (iBlkWidthC * pixelsize),
        pSrcUV1, nSrcPitchUV1,
        pRefUV1, PitchUV1, adjWarr, _trad
      );

      _degrainchroma_ptr(
        pMPBTempBlocksUV2, 0, (iBlkWidthC * pixelsize),
        pSrcUV2, nSrcPitchUV2,
        pRefUV2, PitchUV2, adjWarr, _trad
      );
    }

  } while (1);

  // adjust Wall weights proportionally to found weights
  for (int i = 0; i < (_trad * 2) + 1; i++)
  {
    if (startWarr[i] != 0)
    {
      float fRatio = (float)adjWarr[i] / (float)startWarr[i];
      int weight = Wall[i];
      weight = (int)((float)weight * fRatio + 0.5f);
      Wall[i] = weight;
    }
  }

  norm_weights_all(Wall, _trad);

  // make final blend to output
  _degrainluma_ptr(
    pDst, pDstLsb, nDstPitch,
    pSrc, nSrcPitch,
    pRef, Pitch, Wall, _trad
  );

  // chroma first plane
  _degrainchroma_ptr(
    pDstUV1, pDstLsbUV1, nDstPitchUV1,
    pSrcUV1, nSrcPitchUV1,
    pRefUV1, PitchUV1, WallC, _trad
  );

  // chroma second plane
  _degrainchroma_ptr(
    pDstUV2, pDstLsbUV2, nDstPitchUV2,
    pSrcUV2, nSrcPitchUV2,
    pRefUV2, PitchUV2, WallC, _trad
  );
}

MV_FORCEINLINE int MDegrainN::abs_frame_offset(int index)
{
  assert(index >= 0);
  assert(index <= _trad * 2);
  // 0 is current frame,
  // convert 1,2,3,4 to
  // -1, +1, -2, +2,...
  const int iSign = (index & 0x1) ? -1 : 1;

  return ((index + 1) >> 1) * iSign;

}

MV_FORCEINLINE void MDegrainN::MEL_LC(
  BYTE* pDstCur, int iDstPitch,
  const BYTE* pSrcCur,
  BYTE* pDstCurUV1, int iDstUV1Pitch,
  const BYTE* pSrcCurUV1,
  BYTE* pDstCurUV2, int iDstUV2Pitch,
  const BYTE* pSrcCurUV2,
  int xx, int xx_uv, int ibx, int iby, int iBlkNum)
{
  BYTE* pYmem = pMELmemY + iBlkNum * nBlkSizeX * nBlkSizeY * pixelsize;
  BYTE* pUV1mem = pMELmemUV1 + iBlkNum * (nBlkSizeX >> nLogxRatioUV_super) * (nBlkSizeY >> nLogyRatioUV_super) * pixelsize;
  BYTE* pUV2mem = pMELmemUV2 + iBlkNum * (nBlkSizeX >> nLogxRatioUV_super)* (nBlkSizeY >> nLogyRatioUV_super) * pixelsize;

  int Ymem_pitch = nBlkSizeY * pixelsize;
  int UV1mem_pitch = (nBlkSizeY >> nLogyRatioUV_super)* pixelsize;
  int UV2mem_pitch = (nBlkSizeY >> nLogyRatioUV_super)* pixelsize;

  DM_cache* dmc = DM_cache_arr[iBlkNum];
#ifdef _DEBUG
  iDM_cache_hits = 0;
#endif

  const int rowsizeUV = nBlkSizeY >> nLogyRatioUV_super; // bad name. it's height really
  const int rowwidthUV = nBlkSizeX >> nLogxRatioUV_super; // bad name. it's width really

  int DM_table[MAX_TEMP_RAD * 2 + 1][MAX_TEMP_RAD * 2 + 1];

  const BYTE* ref_data_ptr_arr[MAX_TEMP_RAD * 2] = { 0 };
  int pitch_arr[MAX_TEMP_RAD * 2] = { 0 };
  
  const BYTE* ref_data_ptr_arrUV1[MAX_TEMP_RAD * 2] = { 0 }; // vs: const uint8_t *pointers[radius * 2]; // Moved by the degrain function.
  const BYTE* ref_data_ptr_arrUV2[MAX_TEMP_RAD * 2] = { 0 }; // vs: const uint8_t *pointers[radius * 2]; // Moved by the degrain function. 
  int pitch_arrUV1[MAX_TEMP_RAD * 2] = { 0 };
  int pitch_arrUV2[MAX_TEMP_RAD * 2] = { 0 };

  if (bMVsAddProc)
  {
    FilterBlkMVs(iBlkNum, ibx, iby);

    for (int k = 0; k < _trad * 2; ++k)
    {
      use_block_yuv_mel(
        ref_data_ptr_arr[k],
        pitch_arr[k],
        ref_data_ptr_arrUV1[k],
        pitch_arrUV1[k],
        ref_data_ptr_arrUV2[k],
        pitch_arrUV2[k],
        _usable_flag_arr[k],
        _mv_clip_arr[k],
        iBlkNum,
        _planes_ptr[k][0],
        pSrcCur,
        _planes_ptr[k][1],
        pSrcCurUV1,
        _planes_ptr[k][2],
        pSrcCurUV2,
        xx << pixelsize_super_shift,
        xx_uv << pixelsize_super_shift,
        _src_pitch_arr[0],
        _src_pitch_arr[1],
        _src_pitch_arr[2],
        ibx,
        iby,
        (const VECTOR*)pFilteredMVsPlanesArrays[k]
      );
    }
  }
  else
  {
    for (int k = 0; k < _trad * 2; ++k)
    {
      use_block_yuv_mel(
        ref_data_ptr_arr[k],
        pitch_arr[k],
        ref_data_ptr_arrUV1[k],
        pitch_arrUV1[k],
        ref_data_ptr_arrUV2[k],
        pitch_arrUV2[k],
        _usable_flag_arr[k],
        _mv_clip_arr[k],
        iBlkNum,
        _planes_ptr[k][0],
        pSrcCur,
        _planes_ptr[k][1],
        pSrcCurUV1,
        _planes_ptr[k][2],
        pSrcCurUV2,
        xx << pixelsize_super_shift,
        xx_uv << pixelsize_super_shift,
        _src_pitch_arr[0],
        _src_pitch_arr[1],
        _src_pitch_arr[2],
        ibx,
        iby,
        //            pMVsPlanesArrays[k]
        pMVsWorkPlanesArrays[k]
      );
    }
  }

  for (int dmt_row = 0; dmt_row < (_trad * 2 + 1); dmt_row++)
  {
    for (int dmt_col = 0; dmt_col < (_trad * 2 + 1); dmt_col++)
    {
      if (dmt_row == dmt_col)
      {
        DM_table[dmt_row][dmt_col] = 0; // block with itself
        continue;
      }

      // calc table triangle first (performance optimization)
      if ( dmt_col > dmt_row)
      {
        continue;
      }

      // 0 is current src block ? 1,2,3,4,... -1, +1, -2, +2, blocks in total tr-pool ?
      const BYTE* row_data_ptr;
      const BYTE* row_data_ptrUV1;
      const BYTE* row_data_ptrUV2;
      int row_pitch;
      int row_pitch_UV1;
      int row_pitch_UV2;

      const BYTE* col_data_ptr;
      const BYTE* col_data_ptrUV1;
      const BYTE* col_data_ptrUV2;
      int col_pitch;
      int col_pitch_UV1;
      int col_pitch_UV2;


      if (dmt_row == 0) // src block
      {
        row_data_ptr = pSrcCur + (xx << pixelsize_super_shift);
        row_pitch = _src_pitch_arr[0];

        row_data_ptrUV1 = pSrcCurUV1 + (xx_uv << pixelsize_super_shift);
        row_pitch_UV1 = _src_pitch_arr[1];

        row_data_ptrUV2 = pSrcCurUV2 + (xx_uv << pixelsize_super_shift);
        row_pitch_UV2 = _src_pitch_arr[2];
      }
      else // ref block
      {
        row_data_ptr = ref_data_ptr_arr[dmt_row - 1];
        row_pitch = pitch_arr[dmt_row - 1];

        row_data_ptrUV1 = ref_data_ptr_arrUV1[dmt_row - 1];
        row_pitch_UV1 = pitch_arrUV1[dmt_row - 1];

        row_data_ptrUV2 = ref_data_ptr_arrUV2[dmt_row - 1];
        row_pitch_UV2 = pitch_arrUV2[dmt_row - 1];
      }

      if (dmt_col == 0) // src block
      {
        col_data_ptr = pSrcCur + (xx << pixelsize_super_shift);
        col_pitch = _src_pitch_arr[0];

        col_data_ptrUV1 = pSrcCurUV1 + (xx_uv << pixelsize_super_shift);
        col_pitch_UV1 = _src_pitch_arr[1];

        col_data_ptrUV2 = pSrcCurUV2 + (xx_uv << pixelsize_super_shift);
        col_pitch_UV2 = _src_pitch_arr[2];
      }
      else // ref block
      {
        col_data_ptr = ref_data_ptr_arr[dmt_col - 1];
        col_pitch = pitch_arr[dmt_col - 1];

        col_data_ptrUV1 = ref_data_ptr_arrUV1[dmt_col - 1];
        col_pitch_UV1 = pitch_arrUV1[dmt_col - 1];

        col_data_ptrUV2 = ref_data_ptr_arrUV2[dmt_col - 1];
        col_pitch_UV2 = pitch_arrUV2[dmt_col - 1];
      }

#if 0 // cache usage temporarily excluded from release - need cache fetching (management) method improuvement, esa search too slow
      // check cached DM:
      int iFr0 = iFrameNumRequested + abs_frame_offset(dmt_row);
      int iFr1 = iFrameNumRequested + abs_frame_offset(dmt_col);

      int iDM;

      // DM-cached process
      if (!dmc->Get(iFr0, iFr1, &iDM)) // frame pair DM not yet cached
      {
        // calculate relative dismetric of dmt_row with dmt_col blocks
        int idm_chroma = 0;
        if (TTH_chroma)
        {
          idm_chroma = ScaleSadChroma(DM_TTH_Chroma->GetDisMetric(row_data_ptrUV1, row_pitch_UV1, col_data_ptrUV1, col_pitch_UV1)
            + DM_TTH_Chroma->GetDisMetric(row_data_ptrUV2, row_pitch_UV2, col_data_ptrUV2, col_pitch_UV2), _mv_clip_arr[0]._clip_sptr->chromaSADScale);
        }
        int idm_luma = DM_TTH_Luma->GetDisMetric(row_data_ptr, row_pitch, col_data_ptr, col_pitch);
        iDM = idm_luma + idm_chroma;

        // also push new value to cache
        dmc->PushNew(iFr0, iFr1, iDM);
      }
      else
      {
#ifdef _DEBUG
        iDM_cache_hits++;
#endif
      }

        DM_table[dmt_row][dmt_col] = iDM;
#endif

//#if 0
        // no-DM cache
        int idm_chroma = 0;
        if (TTH_chroma)
        {
          idm_chroma = ScaleSadChroma(DM_TTH_Chroma->GetDisMetric(row_data_ptrUV1, row_pitch_UV1, col_data_ptrUV1, col_pitch_UV1)
            + DM_TTH_Chroma->GetDisMetric(row_data_ptrUV2, row_pitch_UV2, col_data_ptrUV2, col_pitch_UV2), _mv_clip_arr[0]._clip_sptr->chromaSADScale);
        }
        int idm_luma = DM_TTH_Luma->GetDisMetric(row_data_ptr, row_pitch, col_data_ptr, col_pitch);

        DM_table[dmt_row][dmt_col] = idm_luma + idm_chroma;
//#endif
    }
  }

  // restore full table each row
  for (int dmt_row = 0; dmt_row < (_trad * 2 + 1); dmt_row++)
  {
    for (int dmt_col = 0; dmt_col < (_trad * 2 + 1); dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // block with itself
        continue;
      }

      if (dmt_col > dmt_row)
      {
        DM_table[dmt_row][dmt_col] = DM_table[dmt_col][dmt_row];
      }
    }
  }

  // find lowest sum of row in DM_table ?

  int SumRows[(MAX_TEMP_RAD * 2) + 1] = { 0 };

  for (int dmt_row = 0; dmt_row < (_trad * 2 + 1); dmt_row++)
  {
    int i_sum_row = 0;
    for (int dmt_col = 0; dmt_col < (_trad * 2 + 1); dmt_col++)
    {
      i_sum_row += DM_table[dmt_row][dmt_col];
    }

    SumRows[dmt_row] = i_sum_row;
  }

  int i_sum_minrow = SumRows[0];
  int i_idx_minrow = 0;

  for (int dmt_row = 0; dmt_row < (_trad * 2 + 1); dmt_row++)
  {
    if (SumRows[dmt_row] < i_sum_minrow)
    {
      i_sum_minrow = SumRows[dmt_row];
      i_idx_minrow = dmt_row;
    }
  }

  // set block of idx_minrow as output block
  const BYTE* best_data_ptr;
  const BYTE* best_data_ptrUV1;
  const BYTE* best_data_ptrUV2;
  int best_pitch;
  int best_pitch_UV1;
  int best_pitch_UV2;

  if (i_idx_minrow == 0) // src block
  {
    best_data_ptr = pSrcCur + (xx << pixelsize_super_shift);
    best_pitch = _src_pitch_arr[0];

    best_data_ptrUV1 = pSrcCurUV1 + (xx_uv << pixelsize_super_shift);
    best_pitch_UV1 = _src_pitch_arr[1];

    best_data_ptrUV2 = pSrcCurUV2 + (xx_uv << pixelsize_super_shift);
    best_pitch_UV2 = _src_pitch_arr[2];
  }
  else // ref block
  {
    best_data_ptr = ref_data_ptr_arr[i_idx_minrow - 1];
    best_pitch = pitch_arr[i_idx_minrow - 1];

    best_data_ptrUV1 = ref_data_ptr_arrUV1[i_idx_minrow - 1];
    best_pitch_UV1 = pitch_arrUV1[i_idx_minrow - 1];

    best_data_ptrUV2 = ref_data_ptr_arrUV2[i_idx_minrow - 1];
    best_pitch_UV2 = pitch_arrUV2[i_idx_minrow - 1];
#ifdef _DEBUG
    iMEL_non_zero_blocks++;
#endif

  }

  // IIR - check if memory block is still good
  int idm_chroma = 0;
  if (TTH_chroma)
  {
    idm_chroma = ScaleSadChroma(DM_TTH_Chroma->GetDisMetric(best_data_ptrUV1, best_pitch_UV1, pUV1mem, UV1mem_pitch)
      + DM_TTH_Chroma->GetDisMetric(best_data_ptrUV2, best_pitch_UV1, pUV2mem, UV2mem_pitch), _mv_clip_arr[0]._clip_sptr->chromaSADScale);
  }
  int idm_luma = DM_TTH_Luma->GetDisMetric(best_data_ptr, best_pitch, pYmem, Ymem_pitch);
  int idm_mem = idm_chroma + idm_luma;

  if (idm_mem < TTH_thUPD && (i_sum_minrow >= pMELmemYSum[iby*nBlkX + ibx]))
  {
    //mem still good - output mem block
    best_data_ptr = pYmem;
    best_pitch = Ymem_pitch;

    best_data_ptrUV1 = pUV1mem;
    best_pitch_UV1 = UV1mem_pitch;

    best_data_ptrUV2 = pUV2mem;
    best_pitch_UV2 = UV2mem_pitch;
#ifdef _DEBUG
    iMEL_mem_hits++;
#endif
  }
  else // mem no good - update mem
  {
    // luma
    BitBlt(pYmem, Ymem_pitch,
      best_data_ptr, best_pitch, nBlkSizeX, nBlkSizeY);

    // chroma1
    BitBlt(pUV1mem, UV1mem_pitch,
      best_data_ptrUV1, best_pitch_UV1,
      rowwidthUV, rowsizeUV);

    // chroma1
    BitBlt(pUV2mem, UV2mem_pitch,
      best_data_ptrUV2, best_pitch_UV2,
      rowwidthUV, rowsizeUV);

    // update sum memory with lowest sum
    pMELmemYSum[iby * nBlkX + ibx] = i_sum_minrow;

    // use best_* ptrs for output
#ifdef _DEBUG
    iMEL_mem_updates++;
#endif
  }

  
  // copy block of best_ ptrs as output block
  // luma
  if (_out16_flag) {
    // copy 8 bit source to 16bit target
    plane_copy_8_to_16_c(
      pDstCur, iDstPitch, // is +xx << pix_out_sh valid for out16 ? same below
      best_data_ptr, best_pitch, nBlkSizeX, nBlkSizeY);
  }
  else
  {
    BitBlt(pDstCur, iDstPitch,
      best_data_ptr, best_pitch, nBlkSizeX, nBlkSizeY);
  }

  // chroma 1
  if (_out16_flag) {
    // copy 8 bit source to 16bit target
    plane_copy_8_to_16_c(
      pDstCurUV1, iDstUV1Pitch,
      best_data_ptrUV1, best_pitch_UV1,
      rowwidthUV, rowsizeUV
    );
  }
  else {
    BitBlt(pDstCurUV1, iDstUV1Pitch,
      best_data_ptrUV1, best_pitch_UV1,
      rowwidthUV, rowsizeUV);
  }

  // chroma 2
  if (_out16_flag) {
    // copy 8 bit source to 16bit target
    plane_copy_8_to_16_c(
      pDstCurUV2, iDstUV2Pitch,
      best_data_ptrUV2, best_pitch_UV2,
      rowwidthUV, rowsizeUV
    );
  }
  else {
    BitBlt(
      pDstCurUV2, iDstUV2Pitch,
      best_data_ptrUV2, best_pitch_UV2,
      rowwidthUV, rowsizeUV);
  }
  
}

MV_FORCEINLINE void MDegrainN::CalcAutothSADs(void)
{
  sad_t curr_thSAD = thSAD_param_norm;
  sad_t curr_thSAD2 = thSAD2_param_norm;
  sad_t curr_thSADC = thSADC_param_norm;
  sad_t curr_thSADC2 = thSADC2_param_norm;

  int k = 0;
  bool bNearFramesUsable = false;
  if (_usable_flag_arr[0]) // try +1 frame
  {
    k = 0;
    bNearFramesUsable = true;
  }
  else if (_usable_flag_arr[1]) // try -1 frame
  {
    k = 1;
    bNearFramesUsable = true;
  }

  if (bNearFramesUsable)
  {
     VECTOR *pMV = pMVsWorkPlanesArrays[k];

     int iSumSADs = 0;
     int iCntSADs = 0;

     for (int i = 0; i < nBlkCount; i++)
     {
       if (pMV[i].sad < thSCD1)
       {
         iSumSADs += pMV[i].sad;
         iCntSADs++;
       }
     }

     float fMeanBelowthSCD1_SAD = 0;

     if (iCntSADs > 0)
     {
       fMeanBelowthSCD1_SAD = (float)iSumSADs / (float)iCntSADs;

       float fthSAD = fMeanBelowthSCD1_SAD * thSADA_a + thSADA_b;

       curr_thSAD = (int)(fthSAD);
       curr_thSAD2 = (int)(fthSAD * fthSAD12_ratio);

       curr_thSADC = (int)(fthSAD * fthSAD_LC_ratio);
       curr_thSADC2 = (int)(fthSAD * fthSAD_LC_ratio * fthSADC12_ratio);

     }
  }

  for (int i = 0; i < _trad * 2; ++i)
  {
    MvClipInfo& c_info = _mv_clip_arr[i];

    // Computes the SAD thresholds for this source frame, a cosine-shaped
    // smooth transition between thsad(c) and thsad(c)2.
    const int		d = i / 2 + 1;
    c_info._thsad = ClipFnc::interpolate_thsad(curr_thSAD, curr_thSAD2, d, _trad);
    c_info._thsadc = ClipFnc::interpolate_thsad(curr_thSADC, curr_thSADC2, d, _trad);

    c_info._thsad_sq = double(c_info._thsad);
    c_info._thsadc_sq = double(c_info._thsadc);
    for (int j = 0; j < _wpow - 1; j++)
    {
      c_info._thsad_sq *= double(c_info._thsad);
      c_info._thsadc_sq *= double(c_info._thsadc);
    }
  }
}

MV_FORCEINLINE void MDegrainN::DegrainBlendBlock_LC(
  BYTE* pDst, BYTE* pDstLsb, int iDstPitch,
  const BYTE* pSrc,
  BYTE* pDstUV1, BYTE* pDstLsbUV1, int iDstPitchUV1,
  const BYTE* pSrcUV1,
  BYTE* pDstUV2, BYTE* pDstLsbUV2, int iDstPitchUV2,
  const BYTE* pSrcUV2,
  int iBlkNum, int ibx, int iby, int xx, int xx_uv
)
{
  // ToDo: use BlockArea class later !
  BYTE* pYmem = pMELmemY + iBlkNum * nBlkSizeX * nBlkSizeY * pixelsize;
  BYTE* pUV1mem = pMELmemUV1 + iBlkNum * (nBlkSizeX >> nLogxRatioUV_super)* (nBlkSizeY >> nLogyRatioUV_super)* pixelsize;
  BYTE* pUV2mem = pMELmemUV2 + iBlkNum * (nBlkSizeX >> nLogxRatioUV_super)* (nBlkSizeY >> nLogyRatioUV_super)* pixelsize;
  int Ymem_pitch = nBlkSizeY * pixelsize;
  int UV1mem_pitch = (nBlkSizeY >> nLogyRatioUV_super)* pixelsize;
  int UV2mem_pitch = (nBlkSizeY >> nLogyRatioUV_super)* pixelsize;
  const int rowwidthUV = nBlkSizeX >> nLogxRatioUV_super; // bad name. it's width really
  const int rowsizeUV = nBlkSizeY >> nLogyRatioUV_super; // bad name. it's height really

  const BYTE* ref_data_ptr_arr[MAX_TEMP_RAD * 2];
  int pitch_arr[MAX_TEMP_RAD * 2];
  int weight_arr[1 + MAX_TEMP_RAD * 2];
  int weight_arrUV[1 + MAX_TEMP_RAD * 2];

  const BYTE* ref_data_ptr_arrUV1[MAX_TEMP_RAD * 2]; // vs: const uint8_t *pointers[radius * 2]; // Moved by the degrain function.
  const BYTE* ref_data_ptr_arrUV2[MAX_TEMP_RAD * 2]; // vs: const uint8_t *pointers[radius * 2]; // Moved by the degrain function. 
  int pitch_arrUV1[MAX_TEMP_RAD * 2];
  int pitch_arrUV2[MAX_TEMP_RAD * 2];

  if (bMVsAddProc)
  {
    FilterBlkMVs(iBlkNum, ibx, iby);

    for (int k = 0; k < _trad * 2; ++k)
    {
      use_block_yuv(
        ref_data_ptr_arr[k],
        pitch_arr[k],
        ref_data_ptr_arrUV1[k],
        pitch_arrUV1[k],
        ref_data_ptr_arrUV2[k],
        pitch_arrUV2[k],
        weight_arr[k + 1],
        weight_arrUV[k + 1],
        _usable_flag_arr[k],
        _mv_clip_arr[k],
        iBlkNum,
        _planes_ptr[k][0],
        pSrc,
        _planes_ptr[k][1],
        pSrcUV1,
        _planes_ptr[k][2],
        pSrcUV2,
        xx << pixelsize_super_shift,
        xx_uv << pixelsize_super_shift,
        _src_pitch_arr[0],
        _src_pitch_arr[1],
        _src_pitch_arr[2],
        ibx,
        iby,
        (const VECTOR*)pFilteredMVsPlanesArrays[k]
      );
    }
  }
  else
  {
    for (int k = 0; k < _trad * 2; ++k)
    {
      use_block_yuv(
        ref_data_ptr_arr[k],
        pitch_arr[k],
        ref_data_ptr_arrUV1[k],
        pitch_arrUV1[k],
        ref_data_ptr_arrUV2[k],
        pitch_arrUV2[k],
        weight_arr[k + 1],
        weight_arrUV[k + 1],
        _usable_flag_arr[k],
        _mv_clip_arr[k],
        iBlkNum,
        _planes_ptr[k][0],
        pSrc,
        _planes_ptr[k][1],
        pSrcUV1,
        _planes_ptr[k][2],
        pSrcUV2,
        xx << pixelsize_super_shift,
        xx_uv << pixelsize_super_shift,
        _src_pitch_arr[0],
        _src_pitch_arr[1],
        _src_pitch_arr[2],
        ibx,
        iby,
        //            pMVsPlanesArrays[k]
        pMVsWorkPlanesArrays[k]
      );
    }
  }

  if (dn_mm == DN_MM_BLOCKS)
  {
    int iDN_MM_Weight = 255 - pDNMask[iby * dnmask_pitch + ibx]; // invert mask - 255 is zero refs weight - no denoise 
    apply_dn_mask_weights(weight_arr, _trad, iDN_MM_Weight);

    if (bthLC_diff) apply_dn_mask_weights(weight_arrUV, _trad, iDN_MM_Weight);
  }

  norm_weights(weight_arr, _trad);
  if (bthLC_diff) norm_weights(weight_arrUV, _trad);

  int* pChromaWA;
  if (bthLC_diff)
  pChromaWA = &weight_arrUV[0];
  else
  pChromaWA = &weight_arr[0];

  // luma
  if (MPBNumIt == 0 || !isMVsStable(pMVsWorkPlanesArrays, iBlkNum, weight_arr))
  {
    _degrainluma_ptr(
      pDst,
      pDstLsb, iDstPitch,
      pSrc + (xx << pixelsize_super_shift), _src_pitch_arr[0],
      ref_data_ptr_arr, pitch_arr, weight_arr, _trad
    );

    // chroma first plane
    _degrainchroma_ptr(
      pDstUV1,
      pDstLsbUV1, iDstPitchUV1,
      pSrcUV1 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[1],
      ref_data_ptr_arrUV1, pitch_arrUV1, pChromaWA, _trad
    );

    // chroma second plane
    _degrainchroma_ptr(
      pDstUV2,
      pDstLsbUV2, iDstPitchUV2,
      pSrcUV2 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[2],
      ref_data_ptr_arrUV2, pitch_arrUV2, pChromaWA, _trad
    );

    // Add TTH here too, use output buffer as temporal holder of current denoised block
    if (TTH_thUPD > 0)
    {
      // IIR - check if memory block is still good
      int idm_chroma = 0;
      if (TTH_chroma)
      {
        idm_chroma = ScaleSadChroma(DM_TTH_Chroma->GetDisMetric(pDstUV1, iDstPitchUV1, pUV1mem, UV1mem_pitch)
          + DM_TTH_Chroma->GetDisMetric(pDstUV2, iDstPitchUV2, pUV2mem, UV2mem_pitch), _mv_clip_arr[0]._clip_sptr->chromaSADScale);
      }
      int idm_luma = DM_TTH_Luma->GetDisMetric(pDst, iDstPitch, pYmem, Ymem_pitch);
      int idm_mem = idm_chroma + idm_luma;

      if (idm_mem < TTH_thUPD)
      {
        //mem still good - output mem block
        // luma
        BitBlt(pDst, iDstPitch, pYmem, Ymem_pitch, nBlkSizeX, nBlkSizeY);
        // chroma1
        BitBlt(pDstUV1, iDstPitchUV1, pUV1mem, UV1mem_pitch, rowwidthUV, rowsizeUV);
        // chroma2
        BitBlt(pDstUV2, iDstPitchUV2, pUV2mem, UV2mem_pitch, rowwidthUV, rowsizeUV);

  #ifdef _DEBUG
        iMEL_mem_hits++;
  #endif
      }
      else // mem no good - update mem
      {
        // luma
        BitBlt(pYmem, Ymem_pitch, pDst, iDstPitch, nBlkSizeX, nBlkSizeY);
        // chroma1
        BitBlt(pUV1mem, UV1mem_pitch, pDstUV1, iDstPitchUV1, rowwidthUV, rowsizeUV);
        // chroma2
        BitBlt(pUV2mem, UV2mem_pitch, pDstUV2, iDstPitchUV2, rowwidthUV, rowsizeUV);
  #ifdef _DEBUG
        iMEL_mem_updates++;
  #endif
      }// if (idm_mem < TTH_thUPD)

    } // if (TTH_thUPD > 0)

  } // No MPB
  else
  {
    /*
    MPB_LC(
      pDst + (xx << pixelsize_output_shift),
      pDst + _lsb_offset_arr[0] + (xx << pixelsize_super_shift), _dst_pitch_arr[0],
      pSrc + (xx << pixelsize_super_shift), _src_pitch_arr[0],
      ref_data_ptr_arr, pitch_arr,
      pDstUV1 + (xx_uv << pixelsize_output_shift),
      pDstUV1 + (xx_uv << pixelsize_super_shift) + _lsb_offset_arr[1], _dst_pitch_arr[1],
      pSrcUV1 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[1],
      ref_data_ptr_arrUV1, pitch_arrUV1,
      pDstUV2 + (xx_uv << pixelsize_output_shift),
      pDstUV2 + (xx_uv << pixelsize_super_shift) + _lsb_offset_arr[2], _dst_pitch_arr[2],
      pSrcUV2 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[2],
      ref_data_ptr_arrUV2, pitch_arrUV2,
      weight_arr, pChromaWA,
      nBlkSizeX, nBlkSizeY, nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super,
      _mv_clip_arr[0]._clip_sptr->chromaSADScale, iBlkNum
    );
    */
    MPB_LC(
      pDst,
      pDstLsb, iDstPitch,
      pSrc + (xx << pixelsize_super_shift), _src_pitch_arr[0],
      ref_data_ptr_arr, pitch_arr,
      pDstUV1,
      pDstLsbUV1, iDstPitchUV1,
      pSrcUV1 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[1],
      ref_data_ptr_arrUV1, pitch_arrUV1,
      pDstUV2,
      pDstLsbUV2, iDstPitchUV2,
      pSrcUV2 + (xx_uv << pixelsize_super_shift), _src_pitch_arr[2],
      ref_data_ptr_arrUV2, pitch_arrUV2,
      weight_arr, pChromaWA,
      nBlkSizeX, nBlkSizeY, nBlkSizeX >> nLogxRatioUV_super, nBlkSizeY >> nLogyRatioUV_super,
      _mv_clip_arr[0]._clip_sptr->chromaSADScale, iBlkNum
    );

  }
}

// Multi-generation MVs refining, called after first normal blend using current working MVs
MV_FORCEINLINE void MDegrainN::MGR_LC(
  BYTE* pDst, BYTE* pDstLsb, int iDstPitch,
  const BYTE* pSrc,
  BYTE* pDstUV1, BYTE* pDstLsbUV1, int iDstPitchUV1,
  const BYTE* pSrcUV1,
  BYTE* pDstUV2, BYTE* pDstLsbUV2, int iDstPitchUV2,
  const BYTE* pSrcUV2,
  int iBlkNum, int ibx, int iby, int xx, int xx_uv
)
{
  int iNumItCurr = iMGR;

  do
  {
    // Make refined MVs search
    RefineMVs(
      pDst, pDstLsb, iDstPitch,
      pSrc,
      pDstUV1, pDstLsbUV1, iDstPitchUV1,
      pSrcUV1,
      pDstUV2, pDstLsbUV2, iDstPitchUV2,
      pSrcUV2,
      iBlkNum, ibx, iby, xx, xx_uv);

    // Make new degrain blend
    DegrainBlendBlock_LC(
      pDst, pDstLsb, iDstPitch,
      pSrc,
      pDstUV1, pDstLsbUV1, iDstPitchUV1,
      pSrcUV1,
      pDstUV2, pDstLsbUV2, iDstPitchUV2,
      pSrcUV2,
      iBlkNum, ibx, iby, xx, xx_uv);

    iNumItCurr--;

  } while (iNumItCurr != 0);

}

MV_FORCEINLINE void MDegrainN::RefineMVs(
  BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
  const BYTE* pSrc,
  BYTE* pDstUV1, BYTE* pDstLsbUV1, int nDstPitchUV1,
  const BYTE* pSrcUV1,
  BYTE* pDstUV2, BYTE* pDstLsbUV2, int nDstPitchUV2,
  const BYTE* pSrcUV2,
  int iBlkNum, int ibx, int iby, int xx, int xx_uv
)
{
  // search of ref frames around current working (or source MVs) as a predictors using current Dst blended block as source
  TEMPORAL_MVS Predictors[2];// currently 2 predictors - source InputMVs and filtered (WorkingMVs)
  TEMPORAL_MVS Refined;

  // gather predictors
  // input MVs
  for (int k = 0; k < _trad * 2; k++)
  {
    Predictors[0].vMVs[k] = pMVsWorkPlanesArrays[k][iBlkNum];
  }

  //filtered (if present)
  if (bMVsAddProc)
  {
    for (int k = 0; k < _trad * 2; k++)
    {
      Predictors[1].vMVs[k] = pFilteredMVsPlanesArrays[k][iBlkNum];
    }
  }

  for (int k = 0; k < _trad * 2; k++)
  {
    RefineMV(
      pDst, pDstLsb, nDstPitch,
      pDstUV1, pDstLsbUV1, nDstPitchUV1,
      pDstUV2, pDstLsbUV2, nDstPitchUV2,
      Predictors[0].vMVs[k], Predictors[1].vMVs[k], &Refined.vMVs[k], k,
      ibx, iby
    );
  }

  // write back refined MVs to input
  for (int k = 0; k < _trad * 2; k++)
  {
    pMVsWorkPlanesArrays[k][iBlkNum] = Refined.vMVs[k];
  }
}

MV_FORCEINLINE void MDegrainN::RefineMV(
  BYTE* pDst, BYTE* pDstLsb, int iDstPitch,
  BYTE* pDstUV1, BYTE* pDstLsbUV1, int iDstPitchUV1,
  BYTE* pDstUV2, BYTE* pDstLsbUV2, int iDstPitchUV2,
  VECTOR Predictor0, VECTOR Predictor1, VECTOR* Refined, int k,
  int ibx, int iby
)
{

  if (!_usable_flag_arr[k]) // nothing to process
  {
    Refined->x = 0;
    Refined->y = 0;
    Refined->sad = veryBigSAD;
    return;
  }

  Refined->sad = veryBigSAD;

/*  // ESA refine around P0
  for (int dy = (Predictor0.y - iMGR_sr); dy <= (Predictor0.y + iMGR_sr); dy++)
  {
    for (int dx = (Predictor0.x - iMGR_sr); dx <= (Predictor0.x + iMGR_sr); dx++)
    {
      sad_t currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, ibx, iby, k, dx, dy);
      if (currSAD < Refined->sad)
      {
        Refined->x = dx;
        Refined->y = dy;
        Refined->sad = currSAD;
      }
    }
  }

  // predictor 1 if present
  if (bMVsAddProc)
  {
    // ESA refine around P1
    for (int dy = (Predictor1.y - iMGR_sr); dy <= (Predictor1.y + iMGR_sr); dy++)
    {
      for (int dx = (Predictor1.x - iMGR_sr); dx <= (Predictor1.x + iMGR_sr); dx++)
      {
        sad_t currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, ibx, iby, k, dx, dy);
        if (currSAD < Refined->sad)
        {
          Refined->x = dx;
          Refined->y = dy;
          Refined->sad = currSAD;
        }
      }
    }
  } // if bMVsAddProc

  */

  //Predictor0
  //       ExhaustiveSearch(nSearchParam);
  for (int cur_rad = 1; cur_rad <= iMGR_sr; cur_rad++)// region is same as exhaustive, but ordered by radius (from near to far)
  {
    ExpandingSearch(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, ibx, iby, k, cur_rad, 1, Predictor0.x, Predictor0.y, Refined);
  }

  //Predictor1
  //       ExhaustiveSearch(nSearchParam);
  for (int cur_rad = 1; cur_rad <= iMGR_sr; cur_rad++)// region is same as exhaustive, but ordered by radius (from near to far)
  {
    ExpandingSearch(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, ibx, iby, k, cur_rad, 1, Predictor1.x, Predictor1.y, Refined);
  }


}

void MDegrainN::ExpandingSearch(
  BYTE* pDst, int iDstPitch,
  BYTE* pDstUV1, int iDstPitchUV1,
  BYTE* pDstUV2, int iDstPitchUV2,
  int bx_src, int by_src, // numbers of blocks
  int ref_idx,
  int r, int s, int mvx, int mvy, VECTOR* Refined) // diameter = 2*r + 1, step=s
{ // part of true exhaustive search (thin expanding square) around mvx, mvy
  int i, j;
  sad_t currSAD;

  Refined->x = 0;
  Refined->y = 0;
  Refined->sad = veryBigSAD;

  for (i = -r + s; i < r; i += s) // without corners! - v2.1
  {
//    CheckMV<pixel_t>(workarea, mvx + i, mvy - r);
//    CheckMV<pixel_t>(workarea, mvx + i, mvy + r);
    currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx + i, mvy - r);
    if (currSAD < Refined->sad)
    {
      Refined->x = mvx + i;
      Refined->y = mvx - r;
      Refined->sad = currSAD;
    }

    currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx + i, mvy + r);
    if (currSAD < Refined->sad)
    {
      Refined->x = mvx + i;
      Refined->y = mvx + r;
      Refined->sad = currSAD;
    }
  }

  for (j = -r + s; j < r; j += s)
  {
//    CheckMV<pixel_t>(workarea, mvx - r, mvy + j);
//    CheckMV<pixel_t>(workarea, mvx + r, mvy + j);
    currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx - r, mvy + j);
    if (currSAD < Refined->sad)
    {
      Refined->x = mvx - r;
      Refined->y = mvx + j;
      Refined->sad = currSAD;
    }

    currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx + r, mvy + j);
    if (currSAD < Refined->sad)
    {
      Refined->x = mvx + r;
      Refined->y = mvx + j;
      Refined->sad = currSAD;
    }
  }

  // then corners - they are more far from cenrer
//  CheckMV<pixel_t>(workarea, mvx - r, mvy - r);
  currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx - r, mvy - r);
  if (currSAD < Refined->sad)
  {
    Refined->x = mvx - r;
    Refined->y = mvx - r;
    Refined->sad = currSAD;
  }
//  CheckMV<pixel_t>(workarea, mvx - r, mvy + r);
  currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx - r, mvy + r);
  if (currSAD < Refined->sad)
  {
    Refined->x = mvx - r;
    Refined->y = mvx + r;
    Refined->sad = currSAD;
  }

//  CheckMV<pixel_t>(workarea, mvx + r, mvy - r);
  currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx + r, mvy - r);
  if (currSAD < Refined->sad)
  {
    Refined->x = mvx + r;
    Refined->y = mvx - r;
    Refined->sad = currSAD;
  }

//  CheckMV<pixel_t>(workarea, mvx + r, mvy + r);
  currSAD = GetSAD(pDst, iDstPitch, pDstUV1, iDstPitchUV1, pDstUV2, iDstPitchUV2, bx_src, by_src, ref_idx, mvx + r, mvy + r);
  if (currSAD < Refined->sad)
  {
    Refined->x = mvx + r;
    Refined->y = mvx + r;
    Refined->sad = currSAD;
  }

}

MV_FORCEINLINE sad_t MDegrainN::GetSAD(
  BYTE* pSrc, int iSrcPitch,
  BYTE* pSrcUV1, int iSrcPitchUV1,
  BYTE* pSrcUV2, int iSrcPitchUV2,
  int bx_src, int by_src, // numbers of blocks
  int ref_idx, int dx_ref, int dy_ref)
{
  sad_t sad_out;

  if (!_usable_flag_arr[ref_idx]) // nothing to process
  {
    return veryBigSAD;
  }

  const int xx = bx_src * (nBlkSizeX - nOverlapX); // xx: indexing offset, - overlap ?
  const int xx_uv = bx_src * ((nBlkSizeX - nOverlapX) >> nLogxRatioUV_super); // xx_uv: indexing offset

  bool bChroma = (_nsupermodeyuv & UPLANE) && (_nsupermodeyuv & VPLANE); // chroma present in super clip ?
// scaleCSAD in the MVclip props
  int chromaSADscale = _mv_clip_arr[0]._clip_sptr->chromaSADScale; // from 1st ?

  const uint8_t* pRef;
  int npitchRef;

  int blx = bx_src * (nBlkSizeX - nOverlapX) * nPel + dx_ref;
  int bly = by_src * (nBlkSizeY - nOverlapY) * nPel + dy_ref;

  ClipBlxBly

    if (nPel != 1 && nUseSubShift != 0)
    {
      pRef = _planes_ptr[ref_idx][0]->GetPointerSubShift(blx, bly, npitchRef);
    }
    else
    {
      pRef = _planes_ptr[ref_idx][0]->GetPointer(blx, bly);
      npitchRef = _planes_ptr[ref_idx][0]->GetPitch();
    }

  sad_t sad_chroma = 0;

  if (bChroma)
  {
    const uint8_t* pRefU;
    const uint8_t* pRefV;
    int npitchRefU, npitchRefV;

    if (/*nPel != 1 && */nUseSubShift != 0)
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pRefU = _planes_ptr[ref_idx][1]->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npitchRefU);
      pRefV = _planes_ptr[ref_idx][2]->GetPointerSubShift(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super, npitchRefV);
      //      pRefU = _planes_ptr[ref_idx][1]->GetPointerSubShiftUV(blx, bly, npitchRefU, nLogxRatioUV_super, nLogyRatioUV_super);
      //      pRefV = _planes_ptr[ref_idx][2]->GetPointerSubShiftUV(blx, bly, npitchRefV, nLogxRatioUV_super, nLogyRatioUV_super);
    }
    else
    {
      if (nLogxRatioUV_super == 1) blx++; // add bias for integer division for 4:2:x formats
      if (nLogyRatioUV_super == 1) bly++; // add bias for integer division for 4:2:x formats
      pRefU = _planes_ptr[ref_idx][1]->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npitchRefU = _planes_ptr[ref_idx][1]->GetPitch();
      pRefV = _planes_ptr[ref_idx][2]->GetPointer(blx >> nLogxRatioUV_super, bly >> nLogyRatioUV_super);
      npitchRefV = _planes_ptr[ref_idx][2]->GetPitch();
    }

    sad_chroma = ScaleSadChroma(SADCHROMA(pSrcUV1, iSrcPitchUV1, pRefU, npitchRefU)
      + SADCHROMA(pSrcUV1, iSrcPitchUV1, pRefV, npitchRefV), chromaSADscale);

    sad_t luma_sad = SAD(pSrc, iSrcPitch, pRef, npitchRef);

    sad_out = luma_sad + sad_chroma;

  }
  else
  {
    sad_out = SAD(pSrc, iSrcPitch, pRef, npitchRef);
  }

  return sad_out;
}
