#include "VIFFunctions.h"
#include "overlap.h"
#include <map>
#include <tuple>
#include <stdint.h>
#include "def.h"
#include <immintrin.h>

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float VIF_DWT_FULL_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch, DWT2DFunction* pDWT2D)
{
  typedef typename std::conditional < sizeof(pixel_t) <= 2, DWT_DECOMP_INT, DWT_DECOMP_FLOAT >::type target_t_dwt;
  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float >::type dwt_t;
  constexpr dwt_t rounder = (sizeof(pixel_t) <= 2) ? 0 : 0.5f;

  target_t_dwt Src_DWT;
  target_t_dwt Ref_DWT;

  pDWT2D(pSrc, nSrcPitch, &Src_DWT.a, &Src_DWT.v, &Src_DWT.h, &Src_DWT.d);
  pDWT2D(pRef, nRefPitch, &Ref_DWT.a, &Ref_DWT.v, &Ref_DWT.h, &Ref_DWT.d);

  float Xe[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float Ye[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];

  float fMu = 0.45f; // H edges weight
  float fLa = 0.45f; // V edges weight
  float fPsi = 0.1f; // Diagonal edges weight

  const int iNumHpos = nBlkWidth / 2;
  const int iNumVpos = nBlkHeight / 2;

  // calculate only required subbands if pointer to output != 0

  int idxMax = (iNumHpos * iNumVpos); // treat as 1D vectors
  int idx;

  for(idx = 0; idx < idxMax; idx++)
  {
     Xe[idx] = sqrtf(fMu * (Src_DWT.v[idx] * Src_DWT.v[idx]) + fLa * (Src_DWT.h[idx] * Src_DWT.h[idx]) + fPsi * (Src_DWT.d[idx] * Src_DWT.d[idx]));
     Ye[idx] = sqrtf(fMu * (Ref_DWT.v[idx] * Ref_DWT.v[idx]) + fLa * (Ref_DWT.h[idx] * Ref_DWT.h[idx]) + fPsi * (Ref_DWT.d[idx] * Ref_DWT.d[idx]));
  }

  // VIF_A
  float fSigm_sq_N = 5; // some default ?

  int suX = 0;
  int suY = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suX += Src_DWT.a[idx];
    suY += Ref_DWT.a[idx];
  }

  const int iN = (nBlkHeight * nBlkWidth) / 4;
  const dwt_t isuX = (dwt_t)(((float)suX / (float)(iN)) + rounder);
  const dwt_t isuY = (dwt_t)(((float)suY / (float)(iN)) + rounder);

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (idx = 0; idx < idxMax; idx++)
  {
    fsX += (float)((Src_DWT.a[idx] - isuX) * (Src_DWT.a[idx] - isuX)); // squared
    fsY += (float)((Ref_DWT.a[idx] - isuY) * (Ref_DWT.a[idx] - isuY)); // squared
    fsXY += (float)((Src_DWT.a[idx] - isuX) * (Ref_DWT.a[idx] - isuY));
  }

  fsX = fsX / (float)iN;
  fsY = fsY / (float)iN;
  fsXY = fsXY / (float)iN;

  float fEps = 1e-20;
  float fg = fsXY / (fsX + fEps);
  float fsV = fsY - fg * fsXY;
  float fVIFa = logf(1 + (fg * fsX) / (fsV + fSigm_sq_N)) / logf(1 + fsX / fSigm_sq_N);

  // VIF_E
  float suXe = 0;
  float suYe = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suXe += Xe[idx];
    suYe += Ye[idx];
  }

  const float fsuXe = suXe / (float)(iN);
  const float fsuYe = suYe / (float)(iN);

  float fsXe = 0.0f;
  float fsYe = 0.0f;
  float fsXYe = 0.0f;

  for (idx = 0; idx < idxMax; idx++)
  {
    fsXe += (Xe[idx] - fsuXe) * (Xe[idx] - fsuXe); // squared
    fsYe += (Ye[idx] - fsuYe) * (Ye[idx] - fsuYe); // squared
    fsXYe += (Xe[idx] - fsuXe) * (Ye[idx] - fsuYe);
  }

  fsXe = fsXe / (float)iN;
  fsYe = fsYe / (float)iN;
  fsXYe = fsXYe / (float)iN;

  float fge = fsXYe / (fsXe + fEps);
  float fsVe = fsYe - fge * fsXYe;
  float fVIFe = logf(1 + (fge * fsXe) / (fsVe + fSigm_sq_N)) / logf(1 + fsXe / fSigm_sq_N);

  float fAWeight = 0.85f;
  
  float fVIF = fVIFa * fAWeight + (1.0f - fAWeight) * fVIFe;

  if (fVIF > 5.0f)
  {
    int idbr = 0;
    fVIF = 5.0f;
  }

  if (fVIF < -5.0f)
  {
    int idbr = 0;
    fVIF = -5.0f;
  }

  return fVIF;


}

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float VIF_DWT_A_C(const uint8_t *pSrc, int nSrcPitch,const uint8_t *pRef,  int nRefPitch, DWT2DFunction* pDWT2D)
{
  typedef typename std::conditional < sizeof(pixel_t) <= 2, DWT_DECOMP_INT, DWT_DECOMP_FLOAT >::type target_t_dwt;
  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float >::type dwt_t;
  constexpr dwt_t rounder = (sizeof(pixel_t) <= 2) ? 0 : 0.5f;

  target_t_dwt Src_DWT;
  target_t_dwt Ref_DWT;

  pDWT2D(pSrc, nSrcPitch, &Src_DWT.a, 0, 0, 0);
  pDWT2D(pRef, nRefPitch, &Ref_DWT.a, 0, 0, 0);

  float Xe[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float Ye[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];


  float fMu = 0.45f; // H edges weight
  float fLa = 0.45f; // V edges weight
  float fPsi = 0.1f; // Diagonal edges weight

  const int iNumHpos = nBlkWidth / 2;
  const int iNumVpos = nBlkHeight / 2;

  // calculate only required subbands if pointer to output != 0

  int idxMax = (iNumHpos * iNumVpos); // treat as 1D vectors
  int idx;

  for (idx = 0; idx < idxMax; idx++)
  {
    Xe[idx] = sqrtf(fMu * (Src_DWT.v[idx] * Src_DWT.v[idx]) + fLa * (Src_DWT.h[idx] * Src_DWT.h[idx]) + fPsi * (Src_DWT.d[idx] * Src_DWT.d[idx]));
    Ye[idx] = sqrtf(fMu * (Ref_DWT.v[idx] * Ref_DWT.v[idx]) + fLa * (Ref_DWT.h[idx] * Ref_DWT.h[idx]) + fPsi * (Ref_DWT.d[idx] * Ref_DWT.d[idx]));
  }

  // VIF_A
  float fSigm_sq_N = 5; // some default ?

  int suX = 0;
  int suY = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suX += Src_DWT.a[idx];
    suY += Ref_DWT.a[idx];
  }

  const int iN = (nBlkHeight * nBlkWidth) / 4;
  const dwt_t isuX = (dwt_t)(((float)suX / (float)(iN)) + rounder);
  const dwt_t isuY = (dwt_t)(((float)suY / (float)(iN)) + rounder);

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

  for (idx = 0; idx < idxMax; idx++)
  {
    fsX += (float)((Src_DWT.a[idx] - isuX) * (Src_DWT.a[idx] - isuX)); // squared
    fsY += (float)((Ref_DWT.a[idx] - isuY) * (Ref_DWT.a[idx] - isuY)); // squared
    fsXY += (float)((Src_DWT.a[idx] - isuX) * (Ref_DWT.a[idx] - isuY));
  }

  fsX = fsX / (float)iN;
  fsY = fsY / (float)iN;
  fsXY = fsXY / (float)iN;

  float fEps = 1e-20f;
  float fg = fsXY / (fsX + fEps);
  float fsV = fsY - fg * fsXY;
  float fVIFa = logf(1 + (fg * fsX) / (fsV + fSigm_sq_N)) / logf(1 + fsX / fSigm_sq_N);

 
  if (fVIFa > 5.0f)
  {
    int idbr = 0;
    fVIFa = 5.0f;
  }

  if (fVIFa < -5.0f)
  {
    int idbr = 0;
    fVIFa = -5.0f;
  }

  return fVIFa;
}

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static float VIF_DWT_E_C(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef,  int nRefPitch, DWT2DFunction* pDWT2D)
{
  typedef typename std::conditional < sizeof(pixel_t) <= 2, DWT_DECOMP_INT, DWT_DECOMP_FLOAT >::type target_t_dwt;
  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float >::type dwt_t;

  target_t_dwt Src_DWT;
  target_t_dwt Ref_DWT;

  pDWT2D(pSrc, nSrcPitch, 0, &Src_DWT.v, &Src_DWT.h, &Src_DWT.d);
  pDWT2D(pRef, nRefPitch, 0, &Ref_DWT.v, &Ref_DWT.h, &Ref_DWT.d);

  float Xe[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float Ye[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];


  float fMu = 0.45f; // H edges weight
  float fLa = 0.45f; // V edges weight
  float fPsi = 0.1f; // Diagonal edges weight

  const int iNumHpos = nBlkWidth / 2;
  const int iNumVpos = nBlkHeight / 2;

  // calculate only required subbands if pointer to output != 0

  int idxMax = (iNumHpos * iNumVpos); // treat as 1D vectors
  int idx;

  for (idx = 0; idx < idxMax; idx++)
  {
    Xe[idx] = sqrtf(fMu * (Src_DWT.v[idx] * Src_DWT.v[idx]) + fLa * (Src_DWT.h[idx] * Src_DWT.h[idx]) + fPsi * (Src_DWT.d[idx] * Src_DWT.d[idx]));
    Ye[idx] = sqrtf(fMu * (Ref_DWT.v[idx] * Ref_DWT.v[idx]) + fLa * (Ref_DWT.h[idx] * Ref_DWT.h[idx]) + fPsi * (Ref_DWT.d[idx] * Ref_DWT.d[idx]));
  }

  // VIF_A
  float fSigm_sq_N = 5; // some default ?
  const int iN = (nBlkHeight * nBlkWidth) / 4;

  // VIF_E
  float suXe = 0;
  float suYe = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suXe += Xe[idx];
    suYe += Ye[idx];
  }

  const float fsuXe = suXe / (float)(iN);
  const float fsuYe = suYe / (float)(iN);

  float fsXe = 0.0f;
  float fsYe = 0.0f;
  float fsXYe = 0.0f;

  for (idx = 0; idx < idxMax; idx++)
  {
    fsXe += (Xe[idx] - fsuXe) * (Xe[idx] - fsuXe); // squared
    fsYe += (Ye[idx] - fsuYe) * (Ye[idx] - fsuYe); // squared
    fsXYe += (Xe[idx] - fsuXe) * (Ye[idx] - fsuYe);
  }

  fsXe = fsXe / (float)iN;
  fsYe = fsYe / (float)iN;
  fsXYe = fsXYe / (float)iN;

  float fEps = 1e-20f;
  float fge = fsXYe / (fsXe + fEps);
  float fsVe = fsYe - fge * fsXYe;
  float fVIFe = logf(1 + (fge * fsXe) / (fsVe + fSigm_sq_N)) / logf(1 + fsXe / fSigm_sq_N);

  if (fVIFe > 5.0f)
  {
    int idbr = 0;
    fVIFe = 5.0f;
  }

  if (fVIFe < -5.0f)
  {
    int idbr = 0;
    fVIFe = -5.0f;
  }

  return fVIFe;
}


VIFFunction* get_vif_function_a(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, VIFFunction*> func_sad;
#define MAKE_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = VIF_DWT_A_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = VIF_DWT_A_C<x, y, uint16_t>;
  // match with CopyCode.cpp and Overlap.cpp, and luma (variance.cpp) list
  MAKE_FN(64, 64)
    MAKE_FN(32, 32)
    MAKE_FN(16, 16)
    MAKE_FN(8, 8)
    MAKE_FN(4, 4)
#undef MAKE_FN

  // AVX2
//  func_sad[make_tuple(16, 16, 8, USE_AVX2)] = mvt_vif_full_16x16_8_avx2; 
//  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_vif_l_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_vif_l_4x4_8_avx2;

    VIFFunction* result = nullptr;
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

VIFFunction* get_vif_function_e(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, VIFFunction*> func_sad;
#define MAKE_SSIM_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = VIF_DWT_E_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = VIF_DWT_E_C<x, y, uint16_t>;
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
//  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_cs_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_cs_4x4_8_avx2;

    VIFFunction* result = nullptr;
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

VIFFunction* get_vif_function_full(int BlockX, int BlockY, int bits_per_pixel, arch_t arch)
{
  using std::make_tuple;

  int bits_per_pixel_2 = bits_per_pixel;
  if (bits_per_pixel >= 10 && bits_per_pixel < 16)
    bits_per_pixel_2 = 16; // if no 10-bit specific found, secondary find: 16

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, arch_t>, VIFFunction*> func_sad;
#define MAKE_SSIM_FN(x, y) func_sad[make_tuple(x, y, 8, NO_SIMD)] = VIF_DWT_FULL_C<x, y, uint8_t>; \
func_sad[make_tuple(x, y, 16, NO_SIMD)] = VIF_DWT_FULL_C<x, y, uint16_t>;
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
  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_vif_full_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_full_4x4_8_avx2;

    VIFFunction* result = nullptr;
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

float mvt_vif_full_8x8_8_avx2(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch, DWT2DFunction* pDWT2D)
{

  DWT_DECOMP_INT Src_DWT;
  DWT_DECOMP_INT Ref_DWT;

  pDWT2D(pSrc, nSrcPitch, &Src_DWT.a, &Src_DWT.v, &Src_DWT.h, &Src_DWT.d);
  pDWT2D(pRef, nRefPitch, &Ref_DWT.a, &Ref_DWT.v, &Ref_DWT.h, &Ref_DWT.d);

  float Xe[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float Ye[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
   
  float fMu = 0.45f; // H edges weight
  float fLa = 0.45f; // V edges weight
  float fPsi = 0.1f; // Diagonal edges weight

  __m256 ymm_fMu = _mm256_set1_ps(fMu);
  __m256 ymm_fLa = _mm256_set1_ps(fLa);
  __m256 ymm_fPsi = _mm256_set1_ps(fPsi);

  __m256 ymm_r_01sumf;
  __m256 ymm_r_23sumf;

  __m256i ymm_r_01 = _mm256_loadu_si256((const __m256i*)&Src_DWT.v[0]); // load 8 samples of DWT 4+4 2 rows
  __m256i ymm_r_23 = _mm256_loadu_si256((const __m256i*)&Src_DWT.v[8]);

  __m256 ymm_r_01tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_01, ymm_r_01));
  __m256 ymm_r_23tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_23, ymm_r_23));

  ymm_r_01sumf = _mm256_mul_ps(ymm_r_01tmpf, ymm_fMu);
  ymm_r_23sumf = _mm256_mul_ps(ymm_r_23tmpf, ymm_fMu);

  ymm_r_01 = _mm256_loadu_si256((const __m256i*)&Src_DWT.h[0]);
  ymm_r_23 = _mm256_loadu_si256((const __m256i*)&Src_DWT.h[8]);

  ymm_r_01tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_01, ymm_r_01));
  ymm_r_23tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_23, ymm_r_23));

  ymm_r_01sumf = _mm256_add_ps(ymm_r_01sumf, _mm256_mul_ps(ymm_r_01tmpf, ymm_fLa));
  ymm_r_23sumf = _mm256_add_ps(ymm_r_23sumf, _mm256_mul_ps(ymm_r_23tmpf, ymm_fLa));

  ymm_r_01 = _mm256_loadu_si256((const __m256i*)&Src_DWT.d[0]);
  ymm_r_23 = _mm256_loadu_si256((const __m256i*)&Src_DWT.d[8]);

  ymm_r_01tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_01, ymm_r_01));
  ymm_r_23tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_23, ymm_r_23));

  ymm_r_01sumf = _mm256_add_ps(ymm_r_01sumf, _mm256_mul_ps(ymm_r_01tmpf, ymm_fPsi));
  ymm_r_23sumf = _mm256_add_ps(ymm_r_23sumf, _mm256_mul_ps(ymm_r_23tmpf, ymm_fPsi));

  ymm_r_01sumf = _mm256_rsqrt_ps(ymm_r_01sumf);
  ymm_r_23sumf = _mm256_rsqrt_ps(ymm_r_23sumf);

  ymm_r_01sumf = _mm256_rcp_ps(ymm_r_01sumf);
  ymm_r_23sumf = _mm256_rcp_ps(ymm_r_23sumf);

  _mm256_storeu_ps(&Xe[0], ymm_r_01sumf);
  _mm256_storeu_ps(&Xe[8], ymm_r_23sumf);

  ymm_r_01 = _mm256_loadu_si256((const __m256i*)&Ref_DWT.v[0]);
  ymm_r_23 = _mm256_loadu_si256((const __m256i*)&Ref_DWT.v[8]);

  ymm_r_01tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_01, ymm_r_01));
  ymm_r_23tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_23, ymm_r_23));

  ymm_r_01sumf = _mm256_mul_ps(ymm_r_01tmpf, ymm_fMu);
  ymm_r_23sumf = _mm256_mul_ps(ymm_r_23tmpf, ymm_fMu);

  ymm_r_01 = _mm256_loadu_si256((const __m256i*)&Ref_DWT.h[0]);
  ymm_r_23 = _mm256_loadu_si256((const __m256i*)&Ref_DWT.h[8]);

  ymm_r_01tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_01, ymm_r_01));
  ymm_r_23tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_23, ymm_r_23));

  ymm_r_01sumf = _mm256_add_ps(ymm_r_01sumf, _mm256_mul_ps(ymm_r_01tmpf, ymm_fLa));
  ymm_r_23sumf = _mm256_add_ps(ymm_r_23sumf, _mm256_mul_ps(ymm_r_23tmpf, ymm_fLa));

  ymm_r_01 = _mm256_loadu_si256((const __m256i*)&Ref_DWT.d[0]);
  ymm_r_23 = _mm256_loadu_si256((const __m256i*)&Ref_DWT.d[8]);

  ymm_r_01tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_01, ymm_r_01));
  ymm_r_23tmpf = _mm256_cvtepi32_ps(_mm256_mullo_epi32(ymm_r_23, ymm_r_23));

  ymm_r_01sumf = _mm256_add_ps(ymm_r_01sumf, _mm256_mul_ps(ymm_r_01tmpf, ymm_fPsi));
  ymm_r_23sumf = _mm256_add_ps(ymm_r_23sumf, _mm256_mul_ps(ymm_r_23tmpf, ymm_fPsi));

  ymm_r_01sumf = _mm256_rsqrt_ps(ymm_r_01sumf);
  ymm_r_23sumf = _mm256_rsqrt_ps(ymm_r_23sumf);

  ymm_r_01sumf = _mm256_rcp_ps(ymm_r_01sumf);
  ymm_r_23sumf = _mm256_rcp_ps(ymm_r_23sumf);

  _mm256_storeu_ps(&Ye[0], ymm_r_01sumf);
  _mm256_storeu_ps(&Ye[8], ymm_r_23sumf);

#ifdef _DEBUG
  int idxMax = 16;//(iNumHpos=8 * iNumVpos=8 /4); // treat as 1D vectors
  int idx;

  float Xe_tmp[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];
  float Ye_tmp[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE / 4];

  for (idx = 0; idx < idxMax; idx++)
  {
    Xe_tmp[idx] = sqrtf(fMu * (Src_DWT.v[idx] * Src_DWT.v[idx]) + fLa * (Src_DWT.h[idx] * Src_DWT.h[idx]) + fPsi * (Src_DWT.d[idx] * Src_DWT.d[idx]));
    Ye_tmp[idx] = sqrtf(fMu * (Ref_DWT.v[idx] * Ref_DWT.v[idx]) + fLa * (Ref_DWT.h[idx] * Ref_DWT.h[idx]) + fPsi * (Ref_DWT.d[idx] * Ref_DWT.d[idx]));

    if (fabsf(Xe[idx] - Xe_tmp[idx]) > 0.1f)
    {
      int idbr = 0;
    }

    if (fabsf(Ye[idx] - Ye_tmp[idx]) > 0.1f)
    {
      int idbr = 0;
    }
  }

#endif

  // VIF_A
  float fSigm_sq_N = 5; // some default ?

  __m256i ymm_X_r01 = _mm256_loadu_si256((const __m256i*) & Src_DWT.a[0]);
  __m256i ymm_X_r23 = _mm256_loadu_si256((const __m256i*) & Src_DWT.a[8]);
  __m256i ymm_Y_r01 = _mm256_loadu_si256((const __m256i*) & Ref_DWT.a[0]);
  __m256i ymm_Y_r23 = _mm256_loadu_si256((const __m256i*) & Ref_DWT.a[8]);

  ymm_X_r01 = _mm256_add_epi32(ymm_X_r01, ymm_X_r23);
  ymm_Y_r01 = _mm256_add_epi32(ymm_Y_r01, ymm_Y_r23);

  __m256i ymm_suX = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_X_r01, ymm_X_r01, 1), ymm_X_r01); // 8x16 16320+16320 max - still 16bit unsigned OK
  __m256i ymm_suY = _mm256_hadd_epi32(_mm256_permute2x128_si256(ymm_Y_r01, ymm_Y_r01, 1), ymm_Y_r01); // 8x16

  // sum of 2 rows in low 128bit now
  ymm_suX = _mm256_hadd_epi32(ymm_suX, ymm_suX); 
  ymm_suY = _mm256_hadd_epi32(ymm_suY, ymm_suY); 

  ymm_suX = _mm256_hadd_epi32(ymm_suX, ymm_suX);  
  ymm_suY = _mm256_hadd_epi32(ymm_suY, ymm_suY); 

  int iAVX2_sumX = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suX));
  int iAVX2_sumY = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm_suY));

#ifdef _DEBUG
  int suX = 0;
  int suY = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suX += Src_DWT.a[idx];
    suY += Ref_DWT.a[idx];
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

//  const int iN = (64) / 4; // 8x8
  const float rcpiN = 1.0f / 16.0f;
//  const int isuX = (int)(((float)suX / (float)(iN)) + 0.5f);
//  const int isuY = (int)(((float)suY / (float)(iN)) + 0.5f);
  const int isuX = (int)(((float)iAVX2_sumX * rcpiN) + 0.5f);
  const int isuY = (int)(((float)iAVX2_sumY * rcpiN) + 0.5f);

  __m256i ymm_sX = _mm256_setzero_si256();
  __m256i ymm_sY = _mm256_setzero_si256();
  __m256i ymm_sXY = _mm256_setzero_si256();

  __m256i ymm_isuX = _mm256_set1_epi32(isuX);
  __m256i ymm_isuY = _mm256_set1_epi32(isuY);

  ymm_X_r01 = _mm256_loadu_si256((const __m256i*) & Src_DWT.a[0]);
  ymm_X_r23 = _mm256_loadu_si256((const __m256i*) & Src_DWT.a[8]);
  ymm_Y_r01 = _mm256_loadu_si256((const __m256i*) & Ref_DWT.a[0]);
  ymm_Y_r23 = _mm256_loadu_si256((const __m256i*) & Ref_DWT.a[8]);

  __m256i ymm_difX_r01 = _mm256_sub_epi32(ymm_X_r01, ymm_isuX);
  __m256i ymm_difX_r23 = _mm256_sub_epi32(ymm_X_r23, ymm_isuX);
  __m256i ymm_difY_r01 = _mm256_sub_epi32(ymm_Y_r01, ymm_isuY);
  __m256i ymm_difY_r23 = _mm256_sub_epi32(ymm_Y_r23, ymm_isuY);

  ymm_sXY = _mm256_mullo_epi32(ymm_difX_r01, ymm_difY_r01);
  ymm_sX = _mm256_mullo_epi32(ymm_difX_r01, ymm_difX_r01);
  ymm_sY = _mm256_mullo_epi32(ymm_difY_r01, ymm_difY_r01);

  ymm_sXY = _mm256_add_epi32(ymm_sXY, _mm256_mullo_epi32(ymm_difX_r23, ymm_difY_r23));
  ymm_sX = _mm256_add_epi32(ymm_sX, _mm256_mullo_epi32(ymm_difX_r23, ymm_difX_r23));
  ymm_sY = _mm256_add_epi32(ymm_sY, _mm256_mullo_epi32(ymm_difY_r23, ymm_difY_r23));

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

  float fsX = 0.0f;
  float fsY = 0.0f;
  float fsXY = 0.0f;

#ifdef _DEBUG

  for (idx = 0; idx < idxMax; idx++)
  {
    fsX += (float)((Src_DWT.a[idx] - isuX) * (Src_DWT.a[idx] - isuX)); // squared
    fsY += (float)((Ref_DWT.a[idx] - isuY) * (Ref_DWT.a[idx] - isuY)); // squared
    fsXY += (float)((Src_DWT.a[idx] - isuX) * (Ref_DWT.a[idx] - isuY));
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
/*  fsX = fsX / (float)iN;
  fsY = fsY / (float)iN;
  fsXY = fsXY / (float)iN;
  */
/*  fsX = fsX * rcpiN; // /16 may be bitshift ? need test
  fsY = fsY * rcpiN;
  fsXY = fsXY * rcpiN;*/
  fsX = (float)iAVX2_sX * rcpiN;
  fsY = (float)iAVX2_sY * rcpiN;
  fsXY = (float)iAVX2_sXY * rcpiN;

  float fEps = 1e-20;
  float fg = fsXY / (fsX + fEps);
  float fsV = fsY - fg * fsXY;
  float fVIFa = logf(1 + (fg * fsX) / (fsV + fSigm_sq_N)) / logf(1 + fsX / fSigm_sq_N);

  // VIF_E
/*  float suXe = 0;
  float suYe = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suXe += Xe[idx];
    suYe += Ye[idx];
  }
  */

  __m256 ymm_X_r01f = _mm256_loadu_ps((const float*) & Xe[0]);
  __m256 ymm_X_r23f = _mm256_loadu_ps((const float*) & Xe[8]);
  __m256 ymm_Y_r01f = _mm256_loadu_ps((const float*) & Ye[0]);
  __m256 ymm_Y_r23f = _mm256_loadu_ps((const float*) & Ye[8]);

  ymm_X_r01f = _mm256_add_ps(ymm_X_r01f, ymm_X_r23f);
  ymm_Y_r01f = _mm256_add_ps(ymm_Y_r01f, ymm_Y_r23f);

  __m256 ymm_suXf = _mm256_hadd_ps(_mm256_permute2f128_ps(ymm_X_r01f, ymm_X_r01f, 1), ymm_X_r01f);
  __m256 ymm_suYf = _mm256_hadd_ps(_mm256_permute2f128_ps(ymm_Y_r01f, ymm_Y_r01f, 1), ymm_Y_r01f);

  // sum of 2 rows in low 128bit now
  ymm_suXf = _mm256_hadd_ps(ymm_suXf, ymm_suXf);
  ymm_suYf = _mm256_hadd_ps(ymm_suYf, ymm_suYf);

  ymm_suXf = _mm256_hadd_ps(ymm_suXf, ymm_suXf);
  ymm_suYf = _mm256_hadd_ps(ymm_suYf, ymm_suYf);

  float fAVX2_sumXe = _mm_cvtss_f32(_mm256_castps256_ps128(ymm_suXf));
  float fAVX2_sumYe = _mm_cvtss_f32(_mm256_castps256_ps128(ymm_suYf));

#ifdef _DEBUG
  float suXe = 0;
  float suYe = 0;
  for (idx = 0; idx < idxMax; idx++)
  {
    suXe += Xe[idx];
    suYe += Ye[idx];
  }

  if (fabsf(suXe - fAVX2_sumXe) > 0.5f)
  {
    int idbr = 0;
  }

  if (fabsf(suYe - fAVX2_sumYe) > 0.5f)
  {
    int idbr = 0;
  }
#endif

//  const float fsuXe = suXe * rcpiN;
//  const float fsuYe = suYe * rcpiN;

  const float fsuXe = fAVX2_sumXe * rcpiN;
  const float fsuYe = fAVX2_sumYe * rcpiN;

  __m256 ymm_sXf = _mm256_setzero_ps();
  __m256 ymm_sYf = _mm256_setzero_ps();
  __m256 ymm_sXYf = _mm256_setzero_ps();

  __m256 ymm_isuXf = _mm256_set1_ps(fsuXe);
  __m256 ymm_isuYf = _mm256_set1_ps(fsuYe);

  ymm_X_r01f = _mm256_loadu_ps((const float*) & Xe[0]);
  ymm_X_r23f = _mm256_loadu_ps((const float*) & Xe[8]);
  ymm_Y_r01f = _mm256_loadu_ps((const float*) & Ye[0]);
  ymm_Y_r23f = _mm256_loadu_ps((const float*) & Ye[8]);

  __m256 ymm_difX_r01f = _mm256_sub_ps(ymm_X_r01f, ymm_isuXf);
  __m256 ymm_difX_r23f = _mm256_sub_ps(ymm_X_r23f, ymm_isuXf);
  __m256 ymm_difY_r01f = _mm256_sub_ps(ymm_Y_r01f, ymm_isuYf);
  __m256 ymm_difY_r23f = _mm256_sub_ps(ymm_Y_r23f, ymm_isuYf);

  ymm_sXYf = _mm256_mul_ps(ymm_difX_r01f, ymm_difY_r01f);
  ymm_sXf = _mm256_mul_ps(ymm_difX_r01f, ymm_difX_r01f);
  ymm_sYf = _mm256_mul_ps(ymm_difY_r01f, ymm_difY_r01f);

  ymm_sXYf = _mm256_add_ps(ymm_sXYf, _mm256_mul_ps(ymm_difX_r23f, ymm_difY_r23f));
  ymm_sXf = _mm256_add_ps(ymm_sXf, _mm256_mul_ps(ymm_difX_r23f, ymm_difX_r23f));
  ymm_sYf = _mm256_add_ps(ymm_sYf, _mm256_mul_ps(ymm_difY_r23f, ymm_difY_r23f));

  ymm_sXf = _mm256_hadd_ps(_mm256_permute2f128_ps(ymm_sXf, ymm_sXf, 1), ymm_sXf); // 4 x 32
  ymm_sYf = _mm256_hadd_ps(_mm256_permute2f128_ps(ymm_sYf, ymm_sYf, 1), ymm_sYf); // 4 x 32
  ymm_sXYf = _mm256_hadd_ps(_mm256_permute2f128_ps(ymm_sXYf, ymm_sXYf, 1), ymm_sXYf); // 4 x 32

  ymm_sXf = _mm256_hadd_ps(ymm_sXf, ymm_sXf); // 2 x 32
  ymm_sYf = _mm256_hadd_ps(ymm_sYf, ymm_sYf); // 2 x 32 
  ymm_sXYf = _mm256_hadd_ps(ymm_sXYf, ymm_sXYf); // 2 x 32

  ymm_sXf = _mm256_hadd_ps(ymm_sXf, ymm_sXf); // 1 x 32
  ymm_sYf = _mm256_hadd_ps(ymm_sYf, ymm_sYf); // 1 x 32 
  ymm_sXYf = _mm256_hadd_ps(ymm_sXYf, ymm_sXYf); // 1 x 32

  float fAVX2_sXe = _mm_cvtss_f32(_mm256_castps256_ps128(ymm_sXf));
  float fAVX2_sYe = _mm_cvtss_f32(_mm256_castps256_ps128(ymm_sYf));
  float fAVX2_sXYe = _mm_cvtss_f32(_mm256_castps256_ps128(ymm_sXYf));

  float fsXe = 0.0f;
  float fsYe = 0.0f;
  float fsXYe = 0.0f;

#ifdef _DEBUG

  for (idx = 0; idx < idxMax; idx++)
  {
    fsXe += (Xe[idx] - fsuXe) * (Xe[idx] - fsuXe); // squared
    fsYe += (Ye[idx] - fsuYe) * (Ye[idx] - fsuYe); // squared
    fsXYe += (Xe[idx] - fsuXe) * (Ye[idx] - fsuYe);
  }

  if (fabsf(fsXe - fAVX2_sXe) > 0.5f)
  {
    int idbr = 0;
  }

  if (fabsf(fsYe - fAVX2_sYe) > 0.5f)
  {
    int idbr = 0;
  }

  if (fabsf(fsXYe - fAVX2_sXYe) > 0.5f)
  {
    int idbr = 0;
  }


#endif
/*  fsXe = fsXe * rcpiN;
  fsYe = fsYe * rcpiN;
  fsXYe = fsXYe * rcpiN;
  */
  fsXe = fAVX2_sXe * rcpiN;
  fsYe = fAVX2_sYe * rcpiN;
  fsXYe = fAVX2_sXYe * rcpiN;

  float fge = fsXYe / (fsXe + fEps);
  float fsVe = fsYe - fge * fsXYe;
  float fVIFe = logf(1 + (fge * fsXe) / (fsVe + fSigm_sq_N)) / logf(1 + fsXe / fSigm_sq_N);

  float fAWeight = 0.85f;

  float fVIF = fVIFa * fAWeight + (1.0f - fAWeight) * fVIFe;

  if (fVIF > 5.0f)
  {
    int idbr = 0;
    fVIF = 5.0f;
  }

  if (fVIF < -5.0f)
  {
    int idbr = 0;
    fVIF = -5.0f;
  }

  return fVIF;


}
