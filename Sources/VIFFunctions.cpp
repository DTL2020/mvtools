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
  float fVIFa = log10f(1 + (fg * fsX) / (fsV + fSigm_sq_N)) / log10f(1 + fsX / fSigm_sq_N);

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
  float fVIFe = log10f(1 + (fge * fsXe) / (fsVe + fSigm_sq_N)) / log10f(1 + fsXe / fSigm_sq_N);

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
  float fVIFa = log10f(1 + (fg * fsX) / (fsV + fSigm_sq_N)) / log10f(1 + fsX / fSigm_sq_N);

 
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
  constexpr dwt_t rounder = (sizeof(pixel_t) <= 2) ? 0 : 0.5f;

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
  float fVIFe = log10f(1 + (fge * fsXe) / (fsVe + fSigm_sq_N)) / log10f(1 + fsXe / fSigm_sq_N);

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
//  func_sad[make_tuple(16, 16, 8, USE_AVX2)] = mvt_ssim_full_16x16_8_avx2; 
//  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_l_8x8_8_avx2;
//  func_sad[make_tuple(4, 4, 8, USE_AVX2)] = mvt_ssim_l_4x4_8_avx2;

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
//  func_sad[make_tuple(8, 8, 8, USE_AVX2)] = mvt_ssim_full_8x8_8_avx2;
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
