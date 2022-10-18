#include "SADCOVARFunctions.h"
#include <map>
#include <tuple>
#include <stdint.h>
#include "def.h"
#include <immintrin.h>

MV_FORCEINLINE unsigned int SADABS(int x) { return (x < 0) ? -x : x; }

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float SADCOVAR_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch, float* pfCov)
{

  //pRef++; // TEMP DEBUG !!!
  const pixel_t* pWorkSrc = reinterpret_cast<const pixel_t*>(pSrc);
  const pixel_t* pWorkRef = reinterpret_cast<const pixel_t*>(pRef);

  unsigned int sad = 0; // int is probably enough for 32x32

  unsigned int suX = 0;
  unsigned int suY = 0;

  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      sad += SADABS((pWorkSrc)[x] - (pWorkRef)[x]);
      suX += pWorkSrc[x];
      suY += pWorkRef[x];
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  // reset ptrs
  pWorkSrc -= nSrcPitch * nBlkHeight;
  pWorkRef -= nRefPitch * nBlkHeight;

  int iN = nBlkWidth * nBlkHeight;
  // todo: replace with bitshift for blksz 4x4 8x8 16x16 32x32 64x64
  const int isuX = (int)((float)suX / (float)(iN)+0.5f);
  const int isuY = (int)((float)suY / (float)(iN)+0.5f);

  float fsXY = 0.0f;

  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
    {
      fsXY += (float)((pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY)); 
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  *pfCov = fsXY;

  return (float)sad;
}

SADCOVARFunction* get_sadcovar_function(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, SADCOVARFunction*> func_sad;
#define MAKE_SSIM_L_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = SADCOVAR_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = SADCOVAR_C<x, y, uint16_t>;
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
//  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_l_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_l_4x4_8_avx2;

    SADCOVARFunction* result = nullptr;
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

