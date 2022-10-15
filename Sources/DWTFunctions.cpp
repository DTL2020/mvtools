#include "SSIMFunctions.h"
#include "overlap.h"
#include <map>
#include <tuple>
#include <stdint.h>
#include "def.h"
#include <immintrin.h>

// one level decomposition, output up to 4 subbands to pointers pA, pV, pH, pD
template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static void DWT_C(const uint8_t* pSrc, int nSrcPitch, void* pA, void* pV, void* pH, void* pD)
{
#define A pSrc[(iVP * 2) * nSrcPitch + iHP * 2]
#define B pSrc[(iVP * 2) * nSrcPitch + iHP * 2 + 1]
#define C pSrc[(((iVP * 2) + 1) * nSrcPitch) + iHP * 2 + 0]
#define D pSrc[(((iVP * 2) + 1) * nSrcPitch) + iHP * 2 + 1]

  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float >::type target_t_dwt;

  target_t_dwt* pDstA = reinterpret_cast<target_t_dwt*>(pA);
  target_t_dwt* pDstV = reinterpret_cast<target_t_dwt*>(pV);
  target_t_dwt* pDstH = reinterpret_cast<target_t_dwt*>(pH);
  target_t_dwt* pDstD = reinterpret_cast<target_t_dwt*>(pD);

  const int iNumHpos = nBlkWidth / 2;
  const int iNumVpos = nBlkHeight / 2;

  // calculate only required subbands if pointer to output != 0

  if (pDstA != 0)
  {
    for (int iVP = 0; iVP < iNumVpos; iVP++)
    {
      for (int iHP = 0; iHP < iNumHpos; iHP++)
      {
//        *pDstA = (pSrc[(iVP * nSrcPitch + iHP) * 2] + pSrc[(iVP * nSrcPitch + iHP) * 2 + 1] + pSrc[((iVP + 1) * nSrcPitch + iHP) * 2] + pSrc[((iVP + 1) * nSrcPitch + (iHP + 1)) * 2]) / 4;
        *pDstA = ((target_t_dwt)(A + B + C + D) / (target_t_dwt)4);
        pDstA++;
      }
    }
  }

  if (pDstV != 0)
  {
    for (int iVP = 0; iVP < iNumVpos; iVP++)
    {
      for (int iHP = 0; iHP < iNumHpos; iHP++)
      {
        *pDstV = ((target_t_dwt)(B + D - A - C) / (target_t_dwt)4);
        pDstV++;
      }
    }
  }

  if (pDstH != 0)
  {
    for (int iVP = 0; iVP < iNumVpos; iVP++)
    {
      for (int iHP = 0; iHP < iNumHpos; iHP++)
      {
        *pDstH = ((target_t_dwt)(C + D - A - B) / (target_t_dwt)4);
        pDstH++;
      }
    }
  }

  if (pDstD != 0)
  {
    for (int iVP = 0; iVP < iNumVpos; iVP++)
    {
      for (int iHP = 0; iHP < iNumHpos; iHP++)
      {
        *pDstD = ((target_t_dwt)(B + C - A - D) / (target_t_dwt)4);
        pDstD++;
      }
    }
  }
}



DWT2DFunction* get_dwt_function(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, DWT2DFunction*> func_sad;
#define MAKE_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = DWT_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = DWT_C<x, y, uint16_t>;
  // match with CopyCode.cpp and Overlap.cpp, and luma (variance.cpp) list
  MAKE_FN(64, 64)
    MAKE_FN(32, 32)
    MAKE_FN(16, 16)
    MAKE_FN(8, 8)
    MAKE_FN(4, 4)
#undef MAKE_FN

  // AVX2
//  func_sad[make_tuple(16, 16, 8, USE_AVX2)] = mvt_ssim_full_16x16_8_avx2; 
//  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_l_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_l_4x4_8_avx2;

  DWT2DFunction* result = nullptr;
  arch_t archlist[] = { USE_AVX2, USE_AVX, USE_SSE41, USE_SSE2, NO_SIMD };
  int index = 0;
  while (result == nullptr) {
    arch_t current_arch_try = archlist[index++];
    if (current_arch_try > arch) continue;
    result = func_sad[make_tuple(BlockX, BlockY, bits_per_pixel, current_arch_try)];

    if (result == nullptr && current_arch_try == NO_SIMD) {
      break;
    }
  }
  // secondary (e.g. if no 10 bit specific found) search bits_per_pixel_2
  index = 0;
  while (result == nullptr) {
    arch_t current_arch_try = archlist[index++];
    if (current_arch_try > arch) continue;
    if (result == nullptr && current_arch_try == NO_SIMD) {
      result = func_sad[make_tuple(BlockX, BlockY, bits_per_pixel_2, NO_SIMD)];
    }
    else {
      result = func_sad[make_tuple(BlockX, BlockY, bits_per_pixel_2, current_arch_try)];
    }

    if (result == nullptr && current_arch_try == NO_SIMD) {
      break;
    }
  }

  return result;
}

