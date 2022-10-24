#include "COVARFunctions.h"
#include <map>
#include <tuple>
#include <stdint.h>
#include "def.h"
#include <immintrin.h>

MV_FORCEINLINE unsigned int SADABS(int x) { return (x < 0) ? -x : x; }

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float COVAR_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
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
      fsXY += (pWorkSrc[x] - isuX) * (pWorkRef[x] - isuY);
    }
    pWorkSrc += nSrcPitch;
    pWorkRef += nRefPitch;
  }

  return fsXY;
}

COVARFunction* get_covar_function(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, COVARFunction*> func_sad;
#define MAKE_SSIM_L_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = COVAR_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = COVAR_C<x, y, uint16_t>;
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
  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_covar_8x8_8_avx2; // covar only
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_l_4x4_8_avx2;

    COVARFunction* result = nullptr;
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


float mvt_covar_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
#define BLK_X 4 // is it local for function ?
#define BLK_Y 4

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

  pWorkSrc = (uint8_t*)pSrc;
  pWorkRef = (uint8_t*)pRef;

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

    pWorkSrc += (nSrcPitch * 2);
    pWorkRef += (nRefPitch * 2);
  }

  ymm_sXY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_sXY, ymm_sXY, 1), ymm_sXY); // 4 x 32
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 2 x 32
  ymm_sXY = _mm256_hadd_epi32(ymm_sXY, ymm_sXY); // 1 x 32

  int iAVX2_sXY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_sXY));

  return (float)iAVX2_sXY; 
}
