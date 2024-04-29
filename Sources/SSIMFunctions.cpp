#include "SSIMFunctions.h"
#include "overlap.h"
#include <map>
#include <tuple>
#include <stdint.h>
#include "def.h"
#include <immintrin.h>

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float SSIM_FULL_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
  const float k1 = 0.01f;
  constexpr int L = (sizeof(pixel_t) < 2) ? 256 : 65536;
  const float c1 = (k1 * L) * (k1 * L);

  const float k2 = 0.03f;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;

  const pixel_t* pWorkSrc = reinterpret_cast<const pixel_t*>(pSrc);
  const pixel_t* pWorkRef = reinterpret_cast<const pixel_t*>(pRef);

  unsigned int suX = 0;
  unsigned int suY = 0;
  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      suX += pWorkSrc[x];
      suY += pWorkRef[x];
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  const int iN = nBlkHeight * nBlkWidth;
  const int isuX = (int)((float)suX / (float)(iN)+0.5f);
  const int isuY = (int)((float)suY / (float)(iN)+0.5f);

  float fuX = (float)suX / (float)(iN);
  float fuY = (float)suY / (float)(iN);

  float fSSIM_L = (2.0f * fuX * fuY + c1) / (fuX * fuX + fuY * fuY + c1);

  // reset ptrs
  pWorkSrc -= nSrcPitch * nBlkHeight;
  pWorkRef -= nRefPitch * nBlkHeight;

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      fsX += (float)((pWorkSrc[x] - isuX) * (pWorkSrc[x] - isuX));
      fsY += (float)((pWorkRef[x] - isuY) * (pWorkRef[x] - isuY));
      fsXY += (float)((pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY));
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  const float contr = (2.0f * fsX * fsY + c2) / (fsX * fsX + fsY * fsY + c2);
  const float structure = (fsXY + c3) / (sqrtf(fsX * fsY) + c3);

  return fSSIM_L * contr * structure;
}

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float SSIM_CS_C(const uint8_t *pSrc, int nSrcPitch,const uint8_t *pRef,  int nRefPitch)
{
  const float k2 = 0.03f;
  constexpr int L = (sizeof(pixel_t) < 2) ? 256 : 65536;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;

  const pixel_t* pWorkSrc = reinterpret_cast<const pixel_t*>(pSrc);
  const pixel_t* pWorkRef = reinterpret_cast<const pixel_t*>(pRef);

  unsigned int suX = 0;
  unsigned int suY = 0;
  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      suX += pWorkSrc[x];
      suY += pWorkRef[x];
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  const int iN = nBlkHeight * nBlkWidth;
  const int isuX = (int)((float)suX / (float)(iN) + 0.5f);
  const int isuY = (int)((float)suY / (float)(iN) + 0.5f);

  // reset ptrs
  pWorkSrc -= nSrcPitch * nBlkHeight;
  pWorkRef -= nRefPitch * nBlkHeight;

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      fsX += (float)((pWorkSrc[x] - isuX) * (pWorkSrc[x] - isuX));
      fsY += (float)((pWorkRef[x] - isuY) * (pWorkRef[x] - isuY));
      fsXY += (float)((pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY));
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  const float contr = (2.0f * fsX * fsY + c2) / (fsX * fsX + fsY * fsY + c2);
  const float structure = (fsXY + c3) / (sqrtf(fsX * fsY) + c3);

  return contr * structure;
}

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float SSIM_S_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch) // structure only
{
  const float k2 = 0.03f;
  constexpr int L = (sizeof(pixel_t) < 2) ? 256 : 65536;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;

  const pixel_t* pWorkSrc = reinterpret_cast<const pixel_t*>(pSrc);
  const pixel_t* pWorkRef = reinterpret_cast<const pixel_t*>(pRef);

  unsigned int suX = 0;
  unsigned int suY = 0;
  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      suX += pWorkSrc[x];
      suY += pWorkRef[x];
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  const int iN = nBlkHeight * nBlkWidth;
  const int isuX = (int)((float)suX / (float)(iN)+0.5f);
  const int isuY = (int)((float)suY / (float)(iN)+0.5f);

  // reset ptrs
  pWorkSrc -= nSrcPitch * nBlkHeight;
  pWorkRef -= nRefPitch * nBlkHeight;

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      fsX += (float)((pWorkSrc[x] - isuX) * (pWorkSrc[x] - isuX));
      fsY += (float)((pWorkRef[x] - isuY) * (pWorkRef[x] - isuY));
      fsXY += (float)((pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY));
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  const float structure = (fsXY + c3) / (sqrtf(fsX * fsY) + c3);

  return structure;
}


template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float SSIM_L_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef,  int nRefPitch)
{
  const float k1 = 0.01f;
  constexpr int L = (sizeof(pixel_t) < 2) ? 256 : 65536;
  const float c1 = (k1 * L) * (k1 * L);

  unsigned int uX = 0;
  unsigned int uY = 0;
  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      uX += reinterpret_cast<const pixel_t*>(pSrc)[x];
      uY += reinterpret_cast<const pixel_t*>(pRef)[x];
    }
    pSrc += nSrcPitch;
    pRef += nRefPitch;
  }

  const int iN = nBlkHeight * nBlkWidth;
  float fuX = (float)uX / (float)(iN);
  float fuY = (float)uY / (float)(iN);

  float fSSIM_L = (2.0f * fuX * fuY + c1) / (fuX * fuX + fuY * fuY + c1);

  return fSSIM_L;
}


SSIMFunction* get_ssim_function_l(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, SSIMFunction*> func_sad;
#define MAKE_SSIM_L_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = SSIM_L_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = SSIM_L_C<x, y, uint16_t>;
  // match with CopyCode.cpp and Overlap.cpp, and luma (variance.cpp) list
  MAKE_SSIM_L_FN(64, 64)
    MAKE_SSIM_L_FN(64, 48)
    MAKE_SSIM_L_FN(64, 32)
    MAKE_SSIM_L_FN(64, 16)
    MAKE_SSIM_L_FN(48, 64)
    MAKE_SSIM_L_FN(48, 48)
    MAKE_SSIM_L_FN(48, 24)
    MAKE_SSIM_L_FN(48, 12)
    MAKE_SSIM_L_FN(32, 64)
    MAKE_SSIM_L_FN(32, 32)
    MAKE_SSIM_L_FN(32, 24)
    MAKE_SSIM_L_FN(32, 16)
    MAKE_SSIM_L_FN(32, 8)
    MAKE_SSIM_L_FN(24, 48)
    MAKE_SSIM_L_FN(24, 32)
    MAKE_SSIM_L_FN(24, 24)
    MAKE_SSIM_L_FN(24, 12)
    MAKE_SSIM_L_FN(24, 6)
    MAKE_SSIM_L_FN(16, 64)
    MAKE_SSIM_L_FN(16, 32)
    MAKE_SSIM_L_FN(16, 16)
    MAKE_SSIM_L_FN(16, 12)
    MAKE_SSIM_L_FN(16, 8)
    MAKE_SSIM_L_FN(16, 4)
    MAKE_SSIM_L_FN(16, 2)
    MAKE_SSIM_L_FN(16, 1)
    MAKE_SSIM_L_FN(12, 48)
    MAKE_SSIM_L_FN(12, 24)
    MAKE_SSIM_L_FN(12, 16)
    MAKE_SSIM_L_FN(12, 12)
    MAKE_SSIM_L_FN(12, 6)
    MAKE_SSIM_L_FN(12, 3)
    MAKE_SSIM_L_FN(8, 32)
    MAKE_SSIM_L_FN(8, 16)
    MAKE_SSIM_L_FN(8, 8)
    MAKE_SSIM_L_FN(8, 4)
    MAKE_SSIM_L_FN(8, 2)
    MAKE_SSIM_L_FN(8, 1)
    MAKE_SSIM_L_FN(6, 24)
    MAKE_SSIM_L_FN(6, 12)
    MAKE_SSIM_L_FN(6, 6)
    MAKE_SSIM_L_FN(6, 3)
    MAKE_SSIM_L_FN(4, 8)
    MAKE_SSIM_L_FN(4, 4)
    MAKE_SSIM_L_FN(4, 2)
    MAKE_SSIM_L_FN(4, 1)
    MAKE_SSIM_L_FN(3, 6)
    MAKE_SSIM_L_FN(3, 3)
    MAKE_SSIM_L_FN(2, 4)
    MAKE_SSIM_L_FN(2, 2)
    MAKE_SSIM_L_FN(2, 1)
#undef MAKE_SSIM_L_FN

  // AVX2
//  func_sad[make_tuple(16, 16, 8, USE_AVX2)] = mvt_ssim_full_16x16_8_avx2; 
  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_l_8x8_8_avx2;
  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_l_4x4_8_avx2;

    SSIMFunction* result = nullptr;
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

SSIMFunction* get_ssim_function_cs(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, SSIMFunction*> func_sad;
#define MAKE_SSIM_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = SSIM_CS_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = SSIM_CS_C<x, y, uint16_t>;
  // match with CopyCode.cpp and Overlap.cpp, and luma (variance.cpp) list
  MAKE_SSIM_FN(64, 64)
    MAKE_SSIM_FN(64, 48)
    MAKE_SSIM_FN(64, 32)
    MAKE_SSIM_FN(64, 16)
    MAKE_SSIM_FN(48, 64)
    MAKE_SSIM_FN(48, 48)
    MAKE_SSIM_FN(48, 24)
    MAKE_SSIM_FN(48, 12)
    MAKE_SSIM_FN(32, 64)
    MAKE_SSIM_FN(32, 32)
    MAKE_SSIM_FN(32, 24)
    MAKE_SSIM_FN(32, 16)
    MAKE_SSIM_FN(32, 8)
    MAKE_SSIM_FN(24, 48)
    MAKE_SSIM_FN(24, 32)
    MAKE_SSIM_FN(24, 24)
    MAKE_SSIM_FN(24, 12)
    MAKE_SSIM_FN(24, 6)
    MAKE_SSIM_FN(16, 64)
    MAKE_SSIM_FN(16, 32)
    MAKE_SSIM_FN(16, 16)
    MAKE_SSIM_FN(16, 12)
    MAKE_SSIM_FN(16, 8)
    MAKE_SSIM_FN(16, 4)
    MAKE_SSIM_FN(16, 2)
    MAKE_SSIM_FN(16, 1)
    MAKE_SSIM_FN(12, 48)
    MAKE_SSIM_FN(12, 24)
    MAKE_SSIM_FN(12, 16)
    MAKE_SSIM_FN(12, 12)
    MAKE_SSIM_FN(12, 6)
    MAKE_SSIM_FN(12, 3)
    MAKE_SSIM_FN(8, 32)
    MAKE_SSIM_FN(8, 16)
    MAKE_SSIM_FN(8, 8)
    MAKE_SSIM_FN(8, 4)
    MAKE_SSIM_FN(8, 2)
    MAKE_SSIM_FN(8, 1)
    MAKE_SSIM_FN(6, 24)
    MAKE_SSIM_FN(6, 12)
    MAKE_SSIM_FN(6, 6)
    MAKE_SSIM_FN(6, 3)
    MAKE_SSIM_FN(4, 8)
    MAKE_SSIM_FN(4, 4)
    MAKE_SSIM_FN(4, 2)
    MAKE_SSIM_FN(4, 1)
    MAKE_SSIM_FN(3, 6)
    MAKE_SSIM_FN(3, 3)
    MAKE_SSIM_FN(2, 4)
    MAKE_SSIM_FN(2, 2)
    MAKE_SSIM_FN(2, 1)
#undef MAKE_SSIM_FN

    // AVX2
//  func_sad[make_tuple(16, 16, 8, USE_AVX2)] = mvt_ssim_full_16x16_8_avx2; 
  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_cs_8x8_8_avx2;
  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_cs_4x4_8_avx2;

    SSIMFunction* result = nullptr;
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


SSIMFunction* get_ssim_function_s(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, SSIMFunction*> func_sad;
#define MAKE_SSIM_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = SSIM_S_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = SSIM_S_C<x, y, uint16_t>;
  // match with CopyCode.cpp and Overlap.cpp, and luma (variance.cpp) list
  MAKE_SSIM_FN(64, 64)
    MAKE_SSIM_FN(64, 48)
    MAKE_SSIM_FN(64, 32)
    MAKE_SSIM_FN(64, 16)
    MAKE_SSIM_FN(48, 64)
    MAKE_SSIM_FN(48, 48)
    MAKE_SSIM_FN(48, 24)
    MAKE_SSIM_FN(48, 12)
    MAKE_SSIM_FN(32, 64)
    MAKE_SSIM_FN(32, 32)
    MAKE_SSIM_FN(32, 24)
    MAKE_SSIM_FN(32, 16)
    MAKE_SSIM_FN(32, 8)
    MAKE_SSIM_FN(24, 48)
    MAKE_SSIM_FN(24, 32)
    MAKE_SSIM_FN(24, 24)
    MAKE_SSIM_FN(24, 12)
    MAKE_SSIM_FN(24, 6)
    MAKE_SSIM_FN(16, 64)
    MAKE_SSIM_FN(16, 32)
    MAKE_SSIM_FN(16, 16)
    MAKE_SSIM_FN(16, 12)
    MAKE_SSIM_FN(16, 8)
    MAKE_SSIM_FN(16, 4)
    MAKE_SSIM_FN(16, 2)
    MAKE_SSIM_FN(16, 1)
    MAKE_SSIM_FN(12, 48)
    MAKE_SSIM_FN(12, 24)
    MAKE_SSIM_FN(12, 16)
    MAKE_SSIM_FN(12, 12)
    MAKE_SSIM_FN(12, 6)
    MAKE_SSIM_FN(12, 3)
    MAKE_SSIM_FN(8, 32)
    MAKE_SSIM_FN(8, 16)
    MAKE_SSIM_FN(8, 8)
    MAKE_SSIM_FN(8, 4)
    MAKE_SSIM_FN(8, 2)
    MAKE_SSIM_FN(8, 1)
    MAKE_SSIM_FN(6, 24)
    MAKE_SSIM_FN(6, 12)
    MAKE_SSIM_FN(6, 6)
    MAKE_SSIM_FN(6, 3)
    MAKE_SSIM_FN(4, 8)
    MAKE_SSIM_FN(4, 4)
    MAKE_SSIM_FN(4, 2)
    MAKE_SSIM_FN(4, 1)
    MAKE_SSIM_FN(3, 6)
    MAKE_SSIM_FN(3, 3)
    MAKE_SSIM_FN(2, 4)
    MAKE_SSIM_FN(2, 2)
    MAKE_SSIM_FN(2, 1)
#undef MAKE_SSIM_FN

    // AVX2
//func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_s_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_s_4x4_8_avx2;

  SSIMFunction* result = nullptr;
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


SSIMFunction* get_ssim_function_full(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, SSIMFunction*> func_sad;
#define MAKE_SSIM_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = SSIM_FULL_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = SSIM_FULL_C<x, y, uint16_t>;
  // match with CopyCode.cpp and Overlap.cpp, and luma (variance.cpp) list
  MAKE_SSIM_FN(64, 64)
    MAKE_SSIM_FN(64, 48)
    MAKE_SSIM_FN(64, 32)
    MAKE_SSIM_FN(64, 16)
    MAKE_SSIM_FN(48, 64)
    MAKE_SSIM_FN(48, 48)
    MAKE_SSIM_FN(48, 24)
    MAKE_SSIM_FN(48, 12)
    MAKE_SSIM_FN(32, 64)
    MAKE_SSIM_FN(32, 32)
    MAKE_SSIM_FN(32, 24)
    MAKE_SSIM_FN(32, 16)
    MAKE_SSIM_FN(32, 8)
    MAKE_SSIM_FN(24, 48)
    MAKE_SSIM_FN(24, 32)
    MAKE_SSIM_FN(24, 24)
    MAKE_SSIM_FN(24, 12)
    MAKE_SSIM_FN(24, 6)
    MAKE_SSIM_FN(16, 64)
    MAKE_SSIM_FN(16, 32)
    MAKE_SSIM_FN(16, 16)
    MAKE_SSIM_FN(16, 12)
    MAKE_SSIM_FN(16, 8)
    MAKE_SSIM_FN(16, 4)
    MAKE_SSIM_FN(16, 2)
    MAKE_SSIM_FN(16, 1)
    MAKE_SSIM_FN(12, 48)
    MAKE_SSIM_FN(12, 24)
    MAKE_SSIM_FN(12, 16)
    MAKE_SSIM_FN(12, 12)
    MAKE_SSIM_FN(12, 6)
    MAKE_SSIM_FN(12, 3)
    MAKE_SSIM_FN(8, 32)
    MAKE_SSIM_FN(8, 16)
    MAKE_SSIM_FN(8, 8)
    MAKE_SSIM_FN(8, 4)
    MAKE_SSIM_FN(8, 2)
    MAKE_SSIM_FN(8, 1)
    MAKE_SSIM_FN(6, 24)
    MAKE_SSIM_FN(6, 12)
    MAKE_SSIM_FN(6, 6)
    MAKE_SSIM_FN(6, 3)
    MAKE_SSIM_FN(4, 8)
    MAKE_SSIM_FN(4, 4)
    MAKE_SSIM_FN(4, 2)
    MAKE_SSIM_FN(4, 1)
    MAKE_SSIM_FN(3, 6)
    MAKE_SSIM_FN(3, 3)
    MAKE_SSIM_FN(2, 4)
    MAKE_SSIM_FN(2, 2)
    MAKE_SSIM_FN(2, 1)
#undef MAKE_SSIM_FN

   // AVX2
//  func_sad[make_tuple(16, 16, 8, USE_AVX2)] = mvt_ssim_full_16x16_8_avx2; 
  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_full_8x8_8_avx2;
  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_full_4x4_8_avx2;

    SSIMFunction* result = nullptr;
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

float mvt_ssim_full_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 8 // is it local for function ?
#define BLK_Y 8

  const float k1 = 0.01f;
  const int L = BLK_X * BLK_Y;
  const float c1 = (k1 * L) * (k1 * L);

  const float k2 = 0.03f;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;
  const int iN = BLK_X * BLK_Y;

  uint8_t* pWorkSrc = (uint8_t*)pSrc;
  uint8_t* pWorkRef = (uint8_t*)pRef;

  unsigned int suX = 0;
  unsigned int suY = 0;

  __m256i ymm_suX = _mm256_setzero_si256();
  __m256i ymm_suY = _mm256_setzero_si256();
  __m256i ymm_zero = _mm256_setzero_si256();

  for (int y = 0; y < BLK_Y / 2; y++) // process 2 rows
  {
    __m256i ymm_2row_X = _mm256_set_epi64x(0, *(__int64*)(pWorkSrc + nSrcPitch), 0, *(__int64*)(pWorkSrc + 0));
    __m256i ymm_2row_Y = _mm256_set_epi64x(0, *(__int64*)(pWorkRef + nRefPitch), 0, *(__int64*)(pWorkRef + 0));

    ymm_2row_X = _mm256_unpacklo_epi8(ymm_2row_X, ymm_zero);
    ymm_2row_Y = _mm256_unpacklo_epi8(ymm_2row_Y, ymm_zero);

    ymm_suX = _mm256_add_epi16(ymm_suX, ymm_2row_X);
    ymm_suY = _mm256_add_epi16(ymm_suY, ymm_2row_Y);

    pWorkSrc += nSrcPitch * 2;
    pWorkRef += nRefPitch * 2;
  }

  ymm_suX = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suX, ymm_suX, 1), ymm_suX); // 8x16 16320+16320 max - still 16bit unsigned OK
  ymm_suY = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suY, ymm_suY, 1), ymm_suY); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 4x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 4x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 2x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 2x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 1x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 1x16

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX)) & 0xFFFF;
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY)) & 0xFFFF;

#ifdef _DEBUG 
  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  for (int y = 0; y < BLK_Y; y++) // process 2 rows
  {
    for (int x = 0; x < BLK_X; x++)
    {
      suX += pWorkSrc[x];
      suY += pWorkRef[x];
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  if (suX != iAVX2_sumX)
  {
    int idbr = 0;
  }

  if (suY != iAVX2_sumY)
  {
    int idbr = 0;
  }
#endif
  const int isuX = (int)((float)iAVX2_sumX / (float)(iN)+0.5f);
  const int isuY = (int)((float)iAVX2_sumY / (float)(iN)+0.5f);

  float fSSIM_L = (float)(2 * isuX * isuY + (int)c1) / (float)(isuX * isuX + isuY * isuY + (int)c1);

  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  __m256i ymm_sX = _mm256_setzero_si256();
  __m256i ymm_sY = _mm256_setzero_si256();
  __m256i ymm_sXY = _mm256_setzero_si256();

  __m256i ymm_isuX = _mm256_set1_epi32(isuX);
  __m256i ymm_isuY = _mm256_set1_epi32(isuY);

  for (int y = 0; y < BLK_Y; y++)
  {
    __m256i ymm_rowX = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkSrc + 4), 0, 0, 0, *(int*)(pWorkSrc + 0));
    __m256i ymm_rowY = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkRef + 4), 0, 0, 0, *(int*)(pWorkRef + 0));

    ymm_rowX = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_rowX, ymm_zero), ymm_zero);
    ymm_rowY = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_rowY, ymm_zero), ymm_zero);

    __m256i ymm_difX = _mm256_sub_epi32(ymm_rowX, ymm_isuX);
    __m256i ymm_difY = _mm256_sub_epi32(ymm_rowY, ymm_isuY);

    ymm_sXY = _mm256_add_epi32(ymm_sXY, _mm256_mullo_epi32(ymm_difX, ymm_difY));
    ymm_sX = _mm256_add_epi32(ymm_sX, _mm256_mullo_epi32(ymm_difX, ymm_difX));
    ymm_sY = _mm256_add_epi32(ymm_sY, _mm256_mullo_epi32(ymm_difY, ymm_difY));

    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  ymm_sX = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sX, ymm_sX, 1), ymm_sX); // 4 x 32
  ymm_sY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sY, ymm_sY, 1), ymm_sY); // 4 x 32
  ymm_sXY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sXY, ymm_sXY, 1), ymm_sXY); // 4 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 2 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 2 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 2 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 1 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 1 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 1 x 32

  int iAVX2_sX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sX));
  int iAVX2_sY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sY));
  int iAVX2_sXY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sXY));


#ifdef _DEBUG
  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (int y = 0; y < 8; y++)
  {
    for (int x = 0; x < 8; x++)
    {
      fsX += (float)((pWorkSrc[x] - isuX) * (pWorkSrc[x] - isuX));
      fsY += (float)((pWorkRef[x] - isuY) * (pWorkRef[x] - isuY));
      fsXY += (float)((pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY));
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  if (fsX != (float)iAVX2_sX)
  {
    int idbr = 0;
  }

  if (fsY != (float)iAVX2_sY)
  {
    int idbr = 0;
  }

  if (fsXY != (float)iAVX2_sXY)
  {
    int idbr = 0;
  }

#endif

  const float fsX_AVX2 = (float)iAVX2_sX;
  const float fsY_AVX2 = (float)iAVX2_sY;
  const float fsXY_AVX2 = (float)iAVX2_sXY;

  const float contr_avx2 = (2.0f * fsX_AVX2 * fsY_AVX2 + c2) / ( fsX_AVX2 * fsX_AVX2 + fsY_AVX2 * fsY_AVX2 + c2);
  const float structure_avx2 = (fsXY_AVX2 + c3) / (sqrtf(fsX_AVX2 * fsY_AVX2) + c3);

#ifdef _DEBUG
  const float contr = (2.0f * fsX * fsY + c2) / (fsX * fsX + fsY * fsY + c2);
  const float structure = (fsXY + c3) / (sqrtf(fsX * fsY) + c3);


  if (contr != contr_avx2)
  {
    int idbr = 0;
  }

  if (structure != structure_avx2)
  {
    int idbr = 0;
  }

#endif

  return fSSIM_L * contr_avx2 * structure_avx2;

}

float mvt_ssim_cs_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 8 // is it local for function ?
#define BLK_Y 8

  const float k1 = 0.01f;
  const int L = BLK_X * BLK_Y;

  const float k2 = 0.03f;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;
  const int iN = BLK_X * BLK_Y;

  uint8_t* pWorkSrc = (uint8_t*)pSrc;
  uint8_t* pWorkRef = (uint8_t*)pRef;

  __m256i ymm_suX = _mm256_setzero_si256();
  __m256i ymm_suY = _mm256_setzero_si256();
  __m256i ymm_zero = _mm256_setzero_si256();

  for (int y = 0; y < BLK_Y / 2; y++) // process 2 rows
  {
    __m256i ymm_2row_X = _mm256_set_epi64x(0, *(__int64*)(pWorkSrc + nSrcPitch), 0, *(__int64*)(pWorkSrc + 0));
    __m256i ymm_2row_Y = _mm256_set_epi64x(0, *(__int64*)(pWorkRef + nRefPitch), 0, *(__int64*)(pWorkRef + 0));

    ymm_2row_X = _mm256_unpacklo_epi8(ymm_2row_X, ymm_zero);
    ymm_2row_Y = _mm256_unpacklo_epi8(ymm_2row_Y, ymm_zero);

    ymm_suX = _mm256_add_epi16(ymm_suX, ymm_2row_X);
    ymm_suY = _mm256_add_epi16(ymm_suY, ymm_2row_Y);

    pWorkSrc += nSrcPitch * 2;
    pWorkRef += nRefPitch * 2;
  }

  ymm_suX = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suX, ymm_suX, 1), ymm_suX); // 8x16 16320+16320 max - still 16bit unsigned OK
  ymm_suY = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suY, ymm_suY, 1), ymm_suY); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 4x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 4x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 2x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 2x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 1x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 1x16

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX)) & 0xFFFF;
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY)) & 0xFFFF;
  
  const int isuX = (int)((float)iAVX2_sumX / (float)(iN)+0.5f);
  const int isuY = (int)((float)iAVX2_sumY / (float)(iN)+0.5f);

  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  __m256i ymm_sX = _mm256_setzero_si256();
  __m256i ymm_sY = _mm256_setzero_si256();
  __m256i ymm_sXY = _mm256_setzero_si256();

  __m256i ymm_isuX = _mm256_set1_epi32(isuX);
  __m256i ymm_isuY = _mm256_set1_epi32(isuY);

  for (int y = 0; y < BLK_Y; y++)
  {
    __m256i ymm_rowX = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkSrc + 4), 0, 0, 0, *(int*)(pWorkSrc + 0));
    __m256i ymm_rowY = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkRef + 4), 0, 0, 0, *(int*)(pWorkRef + 0));

    ymm_rowX = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_rowX, ymm_zero), ymm_zero);
    ymm_rowY = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_rowY, ymm_zero), ymm_zero);

    __m256i ymm_difX = _mm256_sub_epi32(ymm_rowX, ymm_isuX);
    __m256i ymm_difY = _mm256_sub_epi32(ymm_rowY, ymm_isuY);

    ymm_sXY = _mm256_add_epi32(ymm_sXY, _mm256_mullo_epi32(ymm_difX, ymm_difY));
    ymm_sX = _mm256_add_epi32(ymm_sX, _mm256_mullo_epi32(ymm_difX, ymm_difX));
    ymm_sY = _mm256_add_epi32(ymm_sY, _mm256_mullo_epi32(ymm_difY, ymm_difY));

    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  ymm_sX = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sX, ymm_sX, 1), ymm_sX); // 4 x 32
  ymm_sY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sY, ymm_sY, 1), ymm_sY); // 4 x 32
  ymm_sXY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sXY, ymm_sXY, 1), ymm_sXY); // 4 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 2 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 2 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 2 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 1 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 1 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 1 x 32

  int iAVX2_sX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sX));
  int iAVX2_sY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sY));
  int iAVX2_sXY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sXY));

  const float fsX_AVX2 = (float)iAVX2_sX;
  const float fsY_AVX2 = (float)iAVX2_sY;
  const float fsXY_AVX2 = (float)iAVX2_sXY;

  const float contr_avx2 = (2.0f * fsX_AVX2 * fsY_AVX2 + c2) / (fsX_AVX2 * fsX_AVX2 + fsY_AVX2 * fsY_AVX2 + c2);
  const float structure_avx2 = (fsXY_AVX2 + c3) / (sqrtf(fsX_AVX2 * fsY_AVX2) + c3);

  return contr_avx2 * structure_avx2;

}

float mvt_ssim_l_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 8 // is it local for function ?
#define BLK_Y 8

  const float k1 = 0.01f;
  const int L = BLK_X * BLK_Y;
  const float c1 = (k1 * L) * (k1 * L);

  const int iN = BLK_X * BLK_Y;

  uint8_t* pWorkSrc = (uint8_t*)pSrc;
  uint8_t* pWorkRef = (uint8_t*)pRef;

  __m256i ymm_suX = _mm256_setzero_si256();
  __m256i ymm_suY = _mm256_setzero_si256();
  __m256i ymm_zero = _mm256_setzero_si256();

  for (int y = 0; y < BLK_Y / 2; y++) // process 2 rows
  {
    __m256i ymm_2row_X = _mm256_set_epi64x(0, *(__int64*)(pWorkSrc + nSrcPitch), 0, *(__int64*)(pWorkSrc + 0));
    __m256i ymm_2row_Y = _mm256_set_epi64x(0, *(__int64*)(pWorkRef + nRefPitch), 0, *(__int64*)(pWorkRef + 0));

    ymm_2row_X = _mm256_unpacklo_epi8(ymm_2row_X, ymm_zero);
    ymm_2row_Y = _mm256_unpacklo_epi8(ymm_2row_Y, ymm_zero);

    ymm_suX = _mm256_add_epi16(ymm_suX, ymm_2row_X);
    ymm_suY = _mm256_add_epi16(ymm_suY, ymm_2row_Y);

    pWorkSrc += nSrcPitch * 2;
    pWorkRef += nRefPitch * 2;
  }

  ymm_suX = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suX, ymm_suX, 1), ymm_suX); // 8x16 16320+16320 max - still 16bit unsigned OK
  ymm_suY = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suY, ymm_suY, 1), ymm_suY); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 4x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 4x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 2x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 2x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 1x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 1x16

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX)) & 0xFFFF;
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY)) & 0xFFFF;

  const int isuX = (int)((float)iAVX2_sumX / (float)(iN)+0.5f);
  const int isuY = (int)((float)iAVX2_sumY / (float)(iN)+0.5f);

  float fSSIM_L = (float)(2 * isuX * isuY + (int)c1) / (float)(isuX * isuX + isuY * isuY + (int)c1);

  return fSSIM_L;

}

float mvt_ssim_full_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 4 // is it local for function ?
#define BLK_Y 4

  const float k1 = 0.01f;
  const int L = BLK_X * BLK_Y;
  const float c1 = (k1 * L) * (k1 * L);

  const float k2 = 0.03f;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;
  const int iN = BLK_X * BLK_Y;

  uint8_t* pWorkSrc = (uint8_t*)pSrc;
  uint8_t* pWorkRef = (uint8_t*)pRef;

  unsigned int suX = 0;
  unsigned int suY = 0;

  __m256i ymm_zero = _mm256_setzero_si256();

  __m256i ymm_4row_X = _mm256_set_epi32(0, 0, *(int*)(pWorkSrc + nSrcPitch * 3), *(int*)(pWorkSrc + nSrcPitch * 2), 0, 0, *(int*)(pWorkSrc + nSrcPitch * 1), *(int*)(pWorkSrc + 0));
  __m256i ymm_4row_Y = _mm256_set_epi32(0, 0, *(int*)(pWorkRef + nRefPitch * 3), *(int*)(pWorkRef + nRefPitch * 2), 0, 0, *(int*)(pWorkRef + nRefPitch * 1), *(int*)(pWorkRef + 0));

  __m256i ymm_suX = _mm256_unpacklo_epi8(ymm_4row_X, ymm_zero);
  __m256i ymm_suY = _mm256_unpacklo_epi8(ymm_4row_Y, ymm_zero);

  ymm_suX = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suX, ymm_suX, 1), ymm_suX); // 8x16 16320+16320 max - still 16bit unsigned OK
  ymm_suY = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suY, ymm_suY, 1), ymm_suY); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 4x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 4x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 2x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 2x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 1x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 1x16

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX)) & 0xFFFF;
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY)) & 0xFFFF;

#ifdef _DEBUG 
  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  for (int y = 0; y < BLK_Y; y++) // process 2 rows
  {
    for (int x = 0; x < BLK_X; x++)
    {
      suX += pWorkSrc[x];
      suY += pWorkRef[x];
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  if (suX != iAVX2_sumX)
  {
    int idbr = 0;
  }

  if (suY != iAVX2_sumY)
  {
    int idbr = 0;
  }
#endif
  const int isuX = (int)((float)iAVX2_sumX / (float)(iN)+0.5f);
  const int isuY = (int)((float)iAVX2_sumY / (float)(iN)+0.5f);

  float fSSIM_L = (float)(2 * isuX * isuY + (int)c1) / (float)(isuX * isuX + isuY * isuY + (int)c1);

  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  __m256i ymm_sX = _mm256_setzero_si256();
  __m256i ymm_sY = _mm256_setzero_si256();
  __m256i ymm_sXY = _mm256_setzero_si256();

  __m256i ymm_isuX = _mm256_set1_epi32(isuX);
  __m256i ymm_isuY = _mm256_set1_epi32(isuY);

  for (int y = 0; y < BLK_Y/2; y++)
  {
    __m256i ymm_2row_X = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkSrc + nSrcPitch * 1), 0, 0, 0, *(int*)(pWorkSrc + 0));
    __m256i ymm_2row_Y = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkRef + nRefPitch * 1), 0, 0, 0, *(int*)(pWorkRef + 0));

    ymm_2row_X = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_2row_X, ymm_zero), ymm_zero);
    ymm_2row_Y = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_2row_Y, ymm_zero), ymm_zero);

    __m256i ymm_difX = _mm256_sub_epi32(ymm_2row_X, ymm_isuX);
    __m256i ymm_difY = _mm256_sub_epi32(ymm_2row_Y, ymm_isuY);

    ymm_sXY = _mm256_add_epi32(ymm_sXY, _mm256_mullo_epi32(ymm_difX, ymm_difY));
    ymm_sX = _mm256_add_epi32(ymm_sX, _mm256_mullo_epi32(ymm_difX, ymm_difX));
    ymm_sY = _mm256_add_epi32(ymm_sY, _mm256_mullo_epi32(ymm_difY, ymm_difY));

    pWorkSrc += (nSrcPitch * 2);
    pWorkRef += (nRefPitch * 2);
  }

  ymm_sX = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sX, ymm_sX, 1), ymm_sX); // 4 x 32
  ymm_sY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sY, ymm_sY, 1), ymm_sY); // 4 x 32
  ymm_sXY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sXY, ymm_sXY, 1), ymm_sXY); // 4 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 2 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 2 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 2 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 1 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 1 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 1 x 32

  int iAVX2_sX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sX));
  int iAVX2_sY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sY));
  int iAVX2_sXY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sXY));


#ifdef _DEBUG
  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (int y = 0; y < BLK_X; y++)
  {
    for (int x = 0; x < BLK_Y; x++)
    {
      fsX += (float)((pWorkSrc[x] - isuX) * (pWorkSrc[x] - isuX));
      fsY += (float)((pWorkRef[x] - isuY) * (pWorkRef[x] - isuY));
      fsXY += (float)((pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY));
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  if (fsX != (float)iAVX2_sX)
  {
    int idbr = 0;
  }

  if (fsY != (float)iAVX2_sY)
  {
    int idbr = 0;
  }

  if (fsXY != (float)iAVX2_sXY)
  {
    int idbr = 0;
  }

#endif

  const float fsX_AVX2 = (float)iAVX2_sX;
  const float fsY_AVX2 = (float)iAVX2_sY;
  const float fsXY_AVX2 = (float)iAVX2_sXY;

  const float contr_avx2 = (2.0f * fsX_AVX2 * fsY_AVX2 + c2) / (fsX_AVX2 * fsX_AVX2 + fsY_AVX2 * fsY_AVX2 + c2);
  const float structure_avx2 = (fsXY_AVX2 + c3) / (sqrtf(fsX_AVX2 * fsY_AVX2) + c3);

#ifdef _DEBUG
  const float contr = (2.0f * fsX * fsY + c2) / (fsX * fsX + fsY * fsY + c2);
  const float structure = (fsXY + c3) / (sqrtf(fsX * fsY) + c3);

  if (contr != contr_avx2)
  {
    int idbr = 0;
  }

  if (structure != structure_avx2)
  {
    int idbr = 0;
  }

#endif

  return fSSIM_L * contr_avx2 * structure_avx2;

}

float mvt_ssim_cs_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 4 // is it local for function ?
#define BLK_Y 4

  const float k1 = 0.01f;
  const int L = BLK_X * BLK_Y;
  const float c1 = (k1 * L) * (k1 * L);

  const float k2 = 0.03f;
  const float c2 = (k2 * L) * (k2 * L);
  const float c3 = c2 / 2.0f;
  const int iN = BLK_X * BLK_Y;

  uint8_t* pWorkSrc = (uint8_t*)pSrc;
  uint8_t* pWorkRef = (uint8_t*)pRef;

  unsigned int suX = 0;
  unsigned int suY = 0;

  __m256i ymm_zero = _mm256_setzero_si256();

  __m256i ymm_4row_X = _mm256_set_epi32(0, 0, *(int*)(pWorkSrc + nSrcPitch * 3), *(int*)(pWorkSrc + nSrcPitch * 2), 0, 0, *(int*)(pWorkSrc + nSrcPitch * 1), *(int*)(pWorkSrc + 0));
  __m256i ymm_4row_Y = _mm256_set_epi32(0, 0, *(int*)(pWorkRef + nRefPitch * 3), *(int*)(pWorkRef + nRefPitch * 2), 0, 0, *(int*)(pWorkRef + nRefPitch * 1), *(int*)(pWorkRef + 0));

  __m256i ymm_suX = _mm256_unpacklo_epi8(ymm_4row_X, ymm_zero);;
  __m256i ymm_suY = _mm256_unpacklo_epi8(ymm_4row_Y, ymm_zero);;

  ymm_suX = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suX, ymm_suX, 1), ymm_suX); // 8x16 16320+16320 max - still 16bit unsigned OK
  ymm_suY = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suY, ymm_suY, 1), ymm_suY); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 4x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 4x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 2x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 2x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 1x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 1x16

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX)) & 0xFFFF;
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY)) & 0xFFFF;

  const int isuX = (int)((float)iAVX2_sumX / (float)(iN)+0.5f);
  const int isuY = (int)((float)iAVX2_sumY / (float)(iN)+0.5f);

  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

  __m256i ymm_sX = _mm256_setzero_si256();
  __m256i ymm_sY = _mm256_setzero_si256();
  __m256i ymm_sXY = _mm256_setzero_si256();

  __m256i ymm_isuX = _mm256_set1_epi32(isuX);
  __m256i ymm_isuY = _mm256_set1_epi32(isuY);

  for (int y = 0; y < BLK_Y / 2; y++)
  {
    __m256i ymm_2row_X = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkSrc + nSrcPitch * 1), 0, 0, 0, *(int*)(pWorkSrc + 0));
    __m256i ymm_2row_Y = _mm256_set_epi32(0, 0, 0, *(int*)(pWorkRef + nRefPitch * 1), 0, 0, 0, *(int*)(pWorkRef + 0));

    ymm_2row_X = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_2row_X, ymm_zero), ymm_zero);
    ymm_2row_Y = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(ymm_2row_Y, ymm_zero), ymm_zero);

    __m256i ymm_difX = _mm256_sub_epi32(ymm_2row_X, ymm_isuX);
    __m256i ymm_difY = _mm256_sub_epi32(ymm_2row_Y, ymm_isuY);

    ymm_sXY = _mm256_add_epi32(ymm_sXY, _mm256_mullo_epi32(ymm_difX, ymm_difY));
    ymm_sX = _mm256_add_epi32(ymm_sX, _mm256_mullo_epi32(ymm_difX, ymm_difX));
    ymm_sY = _mm256_add_epi32(ymm_sY, _mm256_mullo_epi32(ymm_difY, ymm_difY));

    pWorkSrc += (nSrcPitch * 2);
    pWorkRef += (nRefPitch * 2);
  }

  ymm_sX = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sX, ymm_sX, 1), ymm_sX); // 4 x 32
  ymm_sY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sY, ymm_sY, 1), ymm_sY); // 4 x 32
  ymm_sXY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sXY, ymm_sXY, 1), ymm_sXY); // 4 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 2 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 2 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 2 x 32

  ymm_sX = _mm256_hadd_epi32(ymm_sX, ymm_sX); // 1 x 32
  ymm_sY = _mm256_hadd_epi32(ymm_sY, ymm_sY); // 1 x 32 
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 1 x 32

  int iAVX2_sX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sX));
  int iAVX2_sY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sY));
  int iAVX2_sXY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sXY));

  const float fsX_AVX2 = (float)iAVX2_sX;
  const float fsY_AVX2 = (float)iAVX2_sY;
  const float fsXY_AVX2 = (float)iAVX2_sXY;

  const float contr_avx2 = (2.0f * fsX_AVX2 * fsY_AVX2 + c2) / (fsX_AVX2 * fsX_AVX2 + fsY_AVX2 * fsY_AVX2 + c2);
  const float structure_avx2 = (fsXY_AVX2 + c3) / (sqrtf(fsX_AVX2 * fsY_AVX2) + c3);
  return contr_avx2 * structure_avx2;

}

float mvt_ssim_l_4x4_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 4 // is it local for function ?
#define BLK_Y 4

  const float k1 = 0.01f;
  const int L = BLK_X * BLK_Y;
  const float c1 = (k1 * L) * (k1 * L);

  const int iN = BLK_X * BLK_Y;

  uint8_t* pWorkSrc = (uint8_t*)pSrc;
  uint8_t* pWorkRef = (uint8_t*)pRef;

  unsigned int suX = 0;
  unsigned int suY = 0;

  __m256i ymm_zero = _mm256_setzero_si256();

  __m256i ymm_4row_X = _mm256_set_epi32(0, 0, *(int*)(pWorkSrc + nSrcPitch * 3), *(int*)(pWorkSrc + nSrcPitch * 2), 0, 0, *(int*)(pWorkSrc + nSrcPitch * 1), *(int*)(pWorkSrc + 0));
  __m256i ymm_4row_Y = _mm256_set_epi32(0, 0, *(int*)(pWorkRef + nRefPitch * 3), *(int*)(pWorkRef + nRefPitch * 2), 0, 0, *(int*)(pWorkRef + nRefPitch * 1), *(int*)(pWorkRef + 0));

  __m256i ymm_suX = _mm256_unpacklo_epi8(ymm_4row_X, ymm_zero);;
  __m256i ymm_suY = _mm256_unpacklo_epi8(ymm_4row_Y, ymm_zero);;

  ymm_suX = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suX, ymm_suX, 1), ymm_suX); // 8x16 16320+16320 max - still 16bit unsigned OK
  ymm_suY = _mm256_hadd_epi16(_mm256_permute2x128_si256(ymm_suY, ymm_suY, 1), ymm_suY); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 4x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 4x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 2x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 2x16

  ymm_suX = _mm256_hadd_epi16(ymm_suX, ymm_suX); // 1x16 
  ymm_suY = _mm256_hadd_epi16(ymm_suY, ymm_suY); // 1x16

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX)) & 0xFFFF;
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY)) & 0xFFFF;

  const int isuX = (int)((float)iAVX2_sumX / (float)(iN)+0.5f);
  const int isuY = (int)((float)iAVX2_sumY / (float)(iN)+0.5f);

  float fSSIM_L = (float)(2 * isuX * isuY + (int)c1) / (float)(isuX * isuX + isuY * isuY + (int)c1);

  return fSSIM_L;

}
