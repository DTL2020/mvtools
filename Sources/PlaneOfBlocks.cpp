// Author: Manao
// Copyright(c)2006 A.G.Balakhnin aka Fizick - global motion, overlap,  mode, refineMVs
// See legal notice in Copying.txt for more information

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



#include "AnaFlags.h"
#include "AvstpWrapper.h"
#include "commonfunctions.h"
#include "DCTClass.h"
#include "DCTFactory.h"
#include "debugprintf.h"
#include "FakePlaneOfBlocks.h"
#include "MVClip.h"
#include "MVFrame.h"
#include "MVPlane.h"
#include "PlaneOfBlocks.h"
#include "Padding.h"
#include "profile.h"

#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <stdint.h>
#include <map>
#include <tuple>


static unsigned int SadDummy(const uint8_t *, int , const uint8_t *, int )
{
  return 0;
}

PlaneOfBlocks::PlaneOfBlocks(int _nBlkX, int _nBlkY, int _nBlkSizeX, int _nBlkSizeY, int _nPel, int _nLevel, int _nFlags, int _nOverlapX, int _nOverlapY,
  int _xRatioUV, int _yRatioUV, int _pixelsize, int _bits_per_pixel,
  conc::ObjPool <DCTClass> *dct_pool_ptr,
  bool mt_flag, int _chromaSADscale, int _optSearchOption, float _scaleCSADfine, int _iUseSubShift, int _DMFlags,
  IScriptEnvironment* env)
  : nBlkX(_nBlkX)
  , nBlkY(_nBlkY)
  , nBlkSizeX(_nBlkSizeX)
  , nBlkSizeY(_nBlkSizeY)
  , nSqrtBlkSize2D((int)(std::sqrt((float)_nBlkSizeX * _nBlkSizeY) + 0.5f)) // precalc for DCT 2.7.38-
  , nBlkCount(_nBlkX * _nBlkY)
  , nPel(_nPel)
  , nLogPel(ilog2(_nPel))	// nLogPel=0 for nPel=1, 1 for nPel=2, 2 for nPel=4, i.e. (x*nPel) = (x<<nLogPel)
  , nScale(iexp2(_nLevel))
  , nLogScale(_nLevel)
  , nFlags(_nFlags)
  , nOverlapX(_nOverlapX)
  , nOverlapY(_nOverlapY)
  , xRatioUV(_xRatioUV) // PF
  , nLogxRatioUV(ilog2(_xRatioUV))
  , yRatioUV(_yRatioUV)
  , nLogyRatioUV(ilog2(_yRatioUV))
  , pixelsize(_pixelsize) // PF
  , pixelsize_shift(ilog2(pixelsize)) // 161201
  , bits_per_pixel(_bits_per_pixel) // PF
  , _mt_flag(mt_flag)
  , chromaSADscale(_chromaSADscale)
  , optSearchOption(_optSearchOption)
  , scaleCSADfine(_scaleCSADfine)
  , iUseSubShift(_iUseSubShift)
  , SAD(0)
  , LUMA(0)
//  , VAR(0)
  , BLITLUMA(0)
  , BLITCHROMA(0)
  , SADCHROMA(0)
  , SATD(0)
//  , vectors(nBlkCount)
//  , vectors(nBlkCount, env)
  , vectors(_nBlkX* _nBlkY, env)
  , smallestPlane((_nFlags & MOTION_SMALLEST_PLANE) != 0)
  , isse((_nFlags & MOTION_USE_ISSE) != 0)
  , chroma((_nFlags & MOTION_USE_CHROMA_MOTION) != 0)
  , dctpitch(AlignNumber(_nBlkSizeX, 16) * _pixelsize)
  , _dct_pool_ptr(dct_pool_ptr)
  , freqArray()
  , verybigSAD(3 * _nBlkSizeX * _nBlkSizeY * (pixelsize == 4 ? 1 : (1 << bits_per_pixel))) // * 256, pixelsize==2 -> 65536. Float:1
  , dctmode(0)
  , _workarea_fact(nBlkSizeX, nBlkSizeY, dctpitch, nLogxRatioUV, nLogyRatioUV, pixelsize, bits_per_pixel)
  , _workarea_pool()
  , _gvect_estim_ptr(0)
  , _gvect_result_count(0)
{
  _workarea_pool.set_factory(_workarea_fact);

  // half must be more than max vector length, which is (framewidth + Padding) * nPel
  freqArray[0].resize(8192 * _nPel * 2);
  freqArray[1].resize(8192 * _nPel * 2);
  // for nFlags, we use CPU_xxxx constants instead of Avisynth's CPUF_xxx values, because there are extra bits here
  sse2 = (bool)(nFlags & CPU_SSE2); // no tricks for really old processors. If SSE2 is reported, use it
  sse41 = (bool)(nFlags & CPU_SSE4);
  avx = (bool)(nFlags & CPU_AVX);
  avx2 = (bool)(nFlags & CPU_AVX2);
  avx512 = (bool)(nFlags & CPU_AVX512);
//  bool ssd = (bool)(nFlags & MOTION_USE_SSD);
//  bool satd = (bool)(nFlags & MOTION_USE_SATD);

  // New experiment from 2.7.18.22: keep LumaSAD:chromaSAD ratio to 4:2

  //      luma SAD : chroma SAD
  // YV12 4:(1+1) = 4:2 (this 4:2 is the new standard from 2.7.18.22 even for YV24)
  // YV16 4:(2+2) = 4:4
  // YV24 4:(4+4) = 4:8

  // that means that nSCD1 should be normalize not by subsampling but with user's chromaSADscale

  //                          YV12  YV16   YV24
  // nLogXRatioUV              1      1     0   
  // nLogYRatioUV              1      0     0   
  // effective_chromaSADscales: (shift right chromaSAD)
  // chromaSADscale=0  ->      0      1     2  // default. YV12:no change. YV24: chroma SAD is divided by 4 (shift right 2)
  //               =1  ->     -1      0     1  // YV12: shift right -1 (=left 1, =*2) YV24: divide by 2 (shift right 1)
  //               =2  ->     -2     -1     0  // YV12: shift right -2 (=left 2, =*4) YV24: no change
  effective_chromaSADscale = (2 - (nLogxRatioUV + nLogyRatioUV));
  effective_chromaSADscale -= chromaSADscale; // user parameter to have larger magnitude for chroma SAD
                                              // effective effective_chromaSADscale can be -2..2.
                                              // when chromaSADscale is zero (default), effective_chromaSADscale is 0..2
  // effective_chromaSADscale = 0; pre-2.7.18.22 format specific chroma SAD weight
  //	ssd=false;
  //	satd=false;

  //	globalMVPredictor.x = zeroMV.x;
  //	globalMVPredictor.y = zeroMV.y;
  //	globalMVPredictor.sad = zeroMV.sad;

  memset(&vectors[0], 0, vectors.size() * sizeof(vectors[0]));

  // function's pointers 
  // Sad_C: SadFunction.cpp
  // Var_c: Variance.h   PF nowhere used!!!
  // Luma_c: Variance.h 
  // Copy_C: CopyCode


  SATD = SadDummy; //for now disable SATD if default functions are used

                     // in overlaps.h
                     // OverlapsLsbFunction
                     // OverlapsFunction
                     // in M(V)DegrainX: DenoiseXFunction
  arch_t arch;
  if (isse && avx512)
    arch = USE_AVX512;
  else if (isse && avx2)
    arch = USE_AVX2;
  else if (isse && avx)
    arch = USE_AVX;
  else if (isse && sse41)
    arch = USE_SSE41;
  else if (isse && sse2)
    arch = USE_SSE2;
  else
    arch = NO_SIMD;

  SAD = get_sad_function(nBlkSizeX, nBlkSizeY, bits_per_pixel, arch);
  SADCHROMA = get_sad_function(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, arch);

  DM_Luma = new DisMetric(nBlkSizeX, nBlkSizeY, bits_per_pixel, pixelsize, arch, _DMFlags);
  DM_Chroma = new DisMetric(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits_per_pixel, pixelsize, arch, _DMFlags);

  BLITLUMA = get_copy_function(nBlkSizeX, nBlkSizeY, pixelsize, arch);
  BLITCHROMA = get_copy_function(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, pixelsize, arch);
  //VAR        = get_var_function(nBlkSizeX/xRatioUV, nBlkSizeY/yRatioUV, pixelsize, arch); // variance.h PF: no VAR
  LUMA = get_luma_function(nBlkSizeX, nBlkSizeY, pixelsize, arch); // variance.h
  SATD = get_satd_function(nBlkSizeX, nBlkSizeY, pixelsize, arch); // P.F. 2.7.0.22d SATD made live
  if (SATD == nullptr)
    SATD = SadDummy;
  if (chroma) {
    if (BLITCHROMA == nullptr) {
      // we don't have env ptr here
      env->ThrowError("MVTools: no BLITCHROMA function for block size %dx%d", nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV);
    }
    if (SADCHROMA == nullptr) {
      // we don't have env ptr here
      env->ThrowError("MVTools: no SADCHROMA function for block size %dx%d", nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV);
    }
  }
  if (!chroma)
  {
    SADCHROMA = SadDummy;
  }

  // DTL's new test 2.7.46
  for (auto &fn : ExhaustiveSearchFunctions)
    fn = nullptr;
  // Get additional test functions only when optSearchOption is not 0.
  // MAnalyze and MRecalculate has now an optsearchoption parameter.
  if (optSearchOption != 0) {

    // fill function array multiple functions, because nSearchParam can change during search
    // block sizes and chroma remain the same

    // not implemented ones are nullptr
    if ((nBlkSizeX == 8 || nBlkSizeX == 16) && (nBlkSizeY == 8 || nBlkSizeY == 16) && pixelsize == 1 && !chroma) {
      for (int iSearchParam = 0; iSearchParam <= MAX_SUPPORTED_EXH_SEARCHPARAM; iSearchParam++)
        ExhaustiveSearchFunctions[iSearchParam] = get_ExhaustiveSearchFunction(nBlkSizeX, nBlkSizeY, iSearchParam, bits_per_pixel, arch);
    }
  }

  if (optSearchOption == 2 && (/*arch != USE_AVX2 ||*/ nPel != 1)) // do not see at Rocket Lake ???
  {
    env->ThrowError("optSearchOption=2 require AVX2 or more CPU and pel=1");
  }
  

  // for debug:
  //         SAD = x264_pixel_sad_4x4_mmx2;
  //         VAR = Var_C<8>;
  //         LUMA = Luma_C<8>;
  //         BLITLUMA = Copy_C<16,16>;
  //		 BLITCHROMA = Copy_C<8,8>; // idem
  //		 SADCHROMA = x264_pixel_sad_8x8_mmx2;

#ifdef ALLOW_DCT
  if (_dct_pool_ptr != 0)
  {
    DCTFactory &	dct_fact =
      dynamic_cast <DCTFactory &> (_dct_pool_ptr->use_factory());
    dctmode = dct_fact.get_dctmode();

    // Preallocate DCT objects using FFTW, to make sure they are allocated
    // during the plug-in construction stage, to avoid as possible
    // concurrency with FFTW instantiations from other plug-ins.
    if (dct_fact.use_fftw())
    {
      const int		nbr_threads =
        (_mt_flag)
        ? AvstpWrapper::use_instance().get_nbr_threads()
        : 1;

      std::vector <DCTClass *>	dct_stack;
      dct_stack.reserve(nbr_threads);
      bool				err_flag = false;
      for (int dct_cnt = 0; dct_cnt < nbr_threads && !err_flag; ++dct_cnt)
      {
        DCTClass *			dct_ptr = _dct_pool_ptr->take_obj();
        if (dct_ptr == 0)
        {
          err_flag = true;
        }
        else
        {
          dct_stack.push_back(dct_ptr);
        }
      }
      while (!dct_stack.empty())
      {
        DCTClass *			dct_ptr = dct_stack.back();
        _dct_pool_ptr->return_obj(*dct_ptr);
        dct_stack.pop_back();
      }
      if (err_flag)
      {
        throw std::runtime_error(
          "MVTools: error while trying to allocate DCT objects using FFTW."
        );
      }
    }
  }
#endif
}



PlaneOfBlocks::~PlaneOfBlocks()
{
  // Nothing
}


void PlaneOfBlocks::SearchMVs(
  MVFrame *_pSrcFrame, MVFrame *_pRefFrame,
  SearchType st, int stp, int lambda, sad_t lsad, int pnew,
  int plevel, int flags, sad_t *out, const VECTOR * globalMVec,
  short *outfilebuf, int fieldShift, sad_t * pmeanLumaChange,
  int divideExtra, int _pzero, int _pglobal, sad_t _badSAD, int _badrange, bool meander, int *vecPrev, bool _tryMany,
  int optPredictorType
)
{
  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  // Frame- and plane-related data preparation

  zeroMVfieldShifted.x = 0;
  zeroMVfieldShifted.y = fieldShift;
  zeroMVfieldShifted.sad = 0; // vs
#ifdef ALLOW_DCT
  // pMeanLumaChange is scaled by bits_per_pixel
  // bit we keep this factor in the ~16 range
  dctweight16 = std::min((sad_t)16, (abs(*pmeanLumaChange) >> (bits_per_pixel-8)) / (nBlkSizeX*nBlkSizeY)); //equal dct and spatial weights for meanLumaChange=8 (empirical)
#endif	// ALLOW_DCT

  badSAD = _badSAD;
  badrange = _badrange;
  _glob_mv_pred_def.x = globalMVec->x * nPel;	// v1.8.2
  _glob_mv_pred_def.y = globalMVec->y * nPel + fieldShift;
  _glob_mv_pred_def.sad = globalMVec->sad;

  //	int nOutPitchY = nBlkX * (nBlkSizeX - nOverlapX) + nOverlapX;
  //	int nOutPitchUV = (nBlkX * (nBlkSizeX - nOverlapX) + nOverlapX) / 2; // xRatioUV=2
  //	char debugbuf[128];
  //	wsprintf(debugbuf,"MVCOMP1: nOutPitchUV=%d, nOverlap=%d, nBlkX=%d, nBlkSize=%d",nOutPitchUV, nOverlap, nBlkX, nBlkSize);
  //	OutputDebugString(debugbuf);

    // write the plane's header
  WriteHeaderToArray(out);

  nFlags |= flags;

  pSrcFrame = _pSrcFrame;
  pRefFrame = _pRefFrame;

#if (ALIGN_SOURCEBLOCK > 1)
  nSrcPitch_plane[0] = pSrcFrame->GetPlane(YPLANE)->GetPitch();
  if (chroma)
  {
    nSrcPitch_plane[1] = pSrcFrame->GetPlane(UPLANE)->GetPitch();
    nSrcPitch_plane[2] = pSrcFrame->GetPlane(VPLANE)->GetPitch();
  }
  nSrcPitch[0] = pixelsize * nBlkSizeX;
  nSrcPitch[1] = pixelsize * nBlkSizeX / xRatioUV; // PF xRatio instead of /2: after 2.7.0.22c;
  nSrcPitch[2] = pixelsize * nBlkSizeX / xRatioUV;
  for (int i = 0; i < 3; i++) {
    nSrcPitch[i] = AlignNumber(nSrcPitch[i], ALIGN_SOURCEBLOCK); // e.g. align reference block pitch to mod16 e.g. at blksize 24
  }
#else	// ALIGN_SOURCEBLOCK
  nSrcPitch[0] = pSrcFrame->GetPlane(YPLANE)->GetPitch();
  if (chroma)
  {
    nSrcPitch[1] = pSrcFrame->GetPlane(UPLANE)->GetPitch();
    nSrcPitch[2] = pSrcFrame->GetPlane(VPLANE)->GetPitch();
  }
#endif	// ALIGN_SOURCEBLOCK
  nRefPitch[0] = pRefFrame->GetPlane(YPLANE)->GetPitch();
  if (chroma)
  {
    nRefPitch[1] = pRefFrame->GetPlane(UPLANE)->GetPitch();
    nRefPitch[2] = pRefFrame->GetPlane(VPLANE)->GetPitch();
  }

  if (iUseSubShift > 0) // send blocksize to MVPlanes
  {
    pRefFrame->GetPlane(YPLANE)->SetBlockSize(nBlkSizeX, nBlkSizeY);

    if (chroma)
    {
      pRefFrame->GetPlane(UPLANE)->SetBlockSize(nBlkSizeX >> nLogxRatioUV, nBlkSizeY >> nLogxRatioUV);
      pRefFrame->GetPlane(VPLANE)->SetBlockSize(nBlkSizeX >> nLogxRatioUV, nBlkSizeY >> nLogxRatioUV);
    }
  }

  searchType = st;		// ( nLogScale == 0 ) ? st : EXHAUSTIVE;
  nSearchParam = stp;	// *nPel;	// v1.8.2 - redesigned in v1.8.5

  _lambda_level = lambda / (nPel * nPel);
  if (plevel == 1)
  {
    _lambda_level *= nScale;	// scale lambda - Fizick
  }
  else if (plevel == 2)
  {
    _lambda_level *= nScale*nScale;
  }

  temporal = (vecPrev != 0);
  if (vecPrev)
  {
    vecPrev += 1; // Just skips the header
  }

  penaltyZero = _pzero;
  pglobal = _pglobal;
  badcount = 0;
  tryMany = _tryMany;
  planeSAD = 0;
  sumLumaChange = 0;

  _out = out;
  _outfilebuf = outfilebuf;
  if (_outfilebuf != NULL)
  {
    int ibr = 0;
  }
  _vecPrev = vecPrev;
  _meander_flag = meander;
  _pnew = pnew;
  _lsad = lsad;
  _predictorType = optPredictorType; // v2.7.46

  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

  penaltyNew = _pnew; // penalty for new vector
  LSAD = _lsad;    // SAD limit for lambda using

  Slicer			slicer(_mt_flag); // fixme: mt bug
  if (bits_per_pixel == 8)
  {
    if (optSearchOption == 2)
    {
      slicer.start(nBlkY, *this, &PlaneOfBlocks::search_mv_slice_SO2<uint8_t>, 4);
    }
    else
    if (optSearchOption == 3)
    {
      slicer.start(nBlkY, *this, &PlaneOfBlocks::search_mv_slice_SO3<uint8_t>, 4); // AVX2 multi-block
    }
    else
    if (optSearchOption == 4)
    {
      slicer.start(nBlkY, *this, &PlaneOfBlocks::search_mv_slice_SO4<uint8_t>, 4); // AVX512 multi-block
    }
    else
    {
      slicer.start(nBlkY, *this, &PlaneOfBlocks::search_mv_slice<uint8_t>, 4);
    }
  }
  else
    slicer.start(nBlkY, *this, &PlaneOfBlocks::search_mv_slice<uint16_t>, 4);

  slicer.wait();

  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

  if (smallestPlane)
  {
    *pmeanLumaChange = (sad_t)(sumLumaChange / nBlkCount); // for all finer planes
  }

}



void PlaneOfBlocks::RecalculateMVs(
  MVClip & mvClip, MVFrame *_pSrcFrame, MVFrame *_pRefFrame,
  SearchType st, int stp, int lambda, sad_t lsad, int pnew,
  int flags, int *out,
  short *outfilebuf, int fieldShift, sad_t thSAD, int divideExtra, int smooth, bool meander,
  int optPredictorType
)
{
  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  // Frame- and plane-related data preparation

  zeroMVfieldShifted.x = 0;
  zeroMVfieldShifted.y = fieldShift;
  zeroMVfieldShifted.sad = 0; // vs
#ifdef ALLOW_DCT
  dctweight16 = 8;//min(16,abs(*pmeanLumaChange)/(nBlkSizeX*nBlkSizeY)); //equal dct and spatial weights for meanLumaChange=8 (empirical)
#endif	// ALLOW_DCT

  // Actually the global predictor is not used in RecalculateMVs().
  _glob_mv_pred_def.x = 0;
  _glob_mv_pred_def.y = fieldShift;
  _glob_mv_pred_def.sad = 9999999;

  //	int nOutPitchY = nBlkX * (nBlkSizeX - nOverlapX) + nOverlapX;
  //	int nOutPitchUV = (nBlkX * (nBlkSizeX - nOverlapX) + nOverlapX) / 2; // xRatioUV=2
  //	char debugbuf[128];
  //	wsprintf(debugbuf,"MVCOMP1: nOutPitchUV=%d, nOverlap=%d, nBlkX=%d, nBlkSize=%d",nOutPitchUV, nOverlap, nBlkX, nBlkSize);
  //	OutputDebugString(debugbuf);

    // write the plane's header
  WriteHeaderToArray(out);

  nFlags |= flags;

  pSrcFrame = _pSrcFrame;
  pRefFrame = _pRefFrame;

#if (ALIGN_SOURCEBLOCK > 1)
  nSrcPitch_plane[0] = pSrcFrame->GetPlane(YPLANE)->GetPitch();
  if (chroma)
  {
    nSrcPitch_plane[1] = pSrcFrame->GetPlane(UPLANE)->GetPitch();
    nSrcPitch_plane[2] = pSrcFrame->GetPlane(VPLANE)->GetPitch();
  }
  nSrcPitch[0] = pixelsize * nBlkSizeX;
  nSrcPitch[1] = pixelsize * nBlkSizeX / xRatioUV; // PF after 2.7.0.22c
  nSrcPitch[2] = pixelsize * nBlkSizeX / xRatioUV; // PF after 2.7.0.22c
  for (int i = 0; i < 3; i++) {
      nSrcPitch[i] = AlignNumber(nSrcPitch[i], ALIGN_SOURCEBLOCK); // e.g. align reference block pitch to mod16 e.g. at blksize 24
  }
#else	// ALIGN_SOURCEBLOCK
  nSrcPitch[0] = pSrcFrame->GetPlane(YPLANE)->GetPitch();
  if (chroma)
  {
    nSrcPitch[1] = pSrcFrame->GetPlane(UPLANE)->GetPitch();
    nSrcPitch[2] = pSrcFrame->GetPlane(VPLANE)->GetPitch();
  }
#endif	// ALIGN_SOURCEBLOCK
  nRefPitch[0] = pRefFrame->GetPlane(YPLANE)->GetPitch();
  if (chroma)
  {
    nRefPitch[1] = pRefFrame->GetPlane(UPLANE)->GetPitch();
    nRefPitch[2] = pRefFrame->GetPlane(VPLANE)->GetPitch();
  }

  searchType = st;
  nSearchParam = stp;//*nPel; // v1.8.2 - redesigned in v1.8.5

  _lambda_level = lambda / (nPel * nPel);
  //	if (plevel==1)
  //	{
  //		_lambda_level *= nScale;// scale lambda - Fizick
  //	}
  //	else if (plevel==2)
  //	{
  //		_lambda_level *= nScale*nScale;
  //	}

  planeSAD = 0;
  sumLumaChange = 0;

  _out = out;
  _outfilebuf = outfilebuf;
  _meander_flag = meander;
  _predictorType = optPredictorType; // 2.7.46
  _pnew = pnew;
  _lsad = lsad;
  _mv_clip_ptr = &mvClip;
  _smooth = smooth;
  _thSAD = thSAD;

  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  // fixme: consider disabling internal mt, it's giving inconsistent results when used
  Slicer			slicer(_mt_flag);
  if(pixelsize==1)
    slicer.start(nBlkY, *this, &PlaneOfBlocks::recalculate_mv_slice<uint8_t>, 4);
  else
    slicer.start(nBlkY, *this, &PlaneOfBlocks::recalculate_mv_slice<uint16_t>, 4);
  slicer.wait();

  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
}


template<typename safe_sad_t, typename smallOverlapSafeSad_t>
//void PlaneOfBlocks::InterpolatePrediction(const PlaneOfBlocks &pob)
void PlaneOfBlocks::InterpolatePrediction(PlaneOfBlocks& pob)
{
  int normFactor = 3 - nLogPel + pob.nLogPel;
  int mulFactor = (normFactor < 0) ? -normFactor : 0;
  normFactor = (normFactor < 0) ? 0 : normFactor;
  int normov = (nBlkSizeX - nOverlapX) * (nBlkSizeY - nOverlapY);
  int aoddx = (nBlkSizeX * 3 - nOverlapX * 2);
  int aevenx = (nBlkSizeX * 3 - nOverlapX * 4);
  int aoddy = (nBlkSizeY * 3 - nOverlapY * 2);
  int aeveny = (nBlkSizeY * 3 - nOverlapY * 4);
  // note: overlapping is still (v2.5.7) not processed properly
  // PF todo make faster

  // 2.7.19.22 max safe: BlkX*BlkY: sqrt(2147483647 / 3 / 255) = 1675 ,(2147483647 = 0x7FFFFFFF)
  //bool isSafeBlkSizeFor8bits = (nBlkSizeX*nBlkSizeY) < 1675;

  // 2.7.35: 
  // the limit was too small for smallOverlap case e.g. BlkSizeX=32 BlkSizeY=32 OverLapX=0 OverLapY=4
  // Worst case (approximately) ax1 * ay1: (nBlkSizeX*3 * nBlkSizeY*3)
  // 32 bit usability for 8 bits (10+bits are always using bigsad_t):
  // (evenOrOdd_xMax * evenOrOdd_yMax) * SadMax                              *4 < 0x7FFFFFFF
  //    *4: four components before /normov
  // (nBlkSizeX*3 * nBlkSizeY*3) * SadMax                                    *4 < 0x7FFFFFFF
  //    SadMax: 3 full planes (worst case 4:4:4 luma and chroma)
  // (nBlkSizeX*3 * nBlkSizeY*3) * (nBlkSizeX * nBlkSizeY * 255 * 3plane)    *4 < 0x7FFFFFFF
  // (nBlkSizeX*nBlkSizeY)^2 *9*255*3 * 4 < 0x7FFFFFFF
  // nBlkSizeX*nBlkSizeY < sqrt(...) 
  // nBlkSizeX*nBlkSizeY < 279.24 -> 280
  // => smallOverlapSafeSad_t needs to be bigsad_t when (nBlkSizeX*nBlkSizeY) >= 280

  // safe_sad_t: 16 bit worst case: 16 * sad_max: 16 * 3x32x32x65536 = 4+5+5+16 > 2^31 over limit
  //             in case of BlockSize > 32, e.g. 128x128x65536 is even more: 7+7+16=30 bits
  //             generally use big_sad_t for 10+ bits


  bool bNoOverlap = (nOverlapX == 0 && nOverlapY == 0);
  bool bSmallOverlap = nOverlapX <= (nBlkSizeX >> 1) && nOverlapY <= (nBlkSizeY >> 1);

  int iout_x, iout_y, iout_sad;

  for (int l = 0, index = 0; l < nBlkY; l++)
  {
    for (int k = 0; k < nBlkX; k++, index++)
    {
      VECTOR v1, v2, v3, v4;
      int i = k;
      int j = l;
      if (i >= 2 * pob.nBlkX)
      {
        i = 2 * pob.nBlkX - 1;
      }
      if (j >= 2 * pob.nBlkY)
      {
        j = 2 * pob.nBlkY - 1;
      }
      int offy = -1 + 2 * (j % 2);
      int offx = -1 + 2 * (i % 2);
      int iper2 = i / 2;
      int jper2 = j / 2;

      if ((i == 0) || (i >= 2 * pob.nBlkX - 1))
      {
        if ((j == 0) || (j >= 2 * pob.nBlkY - 1))
        {
          v1 = v2 = v3 = v4 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
        }
        else
        {
          v1 = v2 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
          v3 = v4 = pob.vectors[iper2 + (jper2 + offy) * pob.nBlkX];
        }
      }
      else if ((j == 0) || (j >= 2 * pob.nBlkY - 1))
      {
        v1 = v2 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
        v3 = v4 = pob.vectors[iper2 + offx + (jper2)*pob.nBlkX];
      }
      else
      {
        v1 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
        v2 = pob.vectors[iper2 + offx + (jper2)*pob.nBlkX];
        v3 = pob.vectors[iper2 + (jper2 + offy) * pob.nBlkX];
        v4 = pob.vectors[iper2 + offx + (jper2 + offy) * pob.nBlkX];
      }

      safe_sad_t tmp_sad;

      if (bNoOverlap)
      {
        iout_x = 9 * v1.x + 3 * v2.x + 3 * v3.x + v4.x;
        iout_y = 9 * v1.y + 3 * v2.y + 3 * v3.y + v4.y;
        tmp_sad = 9 * (safe_sad_t)v1.sad + 3 * (safe_sad_t)v2.sad + 3 * (safe_sad_t)v3.sad + (safe_sad_t)v4.sad + 8;

      }
      else if (bSmallOverlap) // corrected in v1.4.11
      {
        int	ax1 = (offx > 0) ? aoddx : aevenx;
        int ax2 = (nBlkSizeX - nOverlapX) * 4 - ax1;
        int ay1 = (offy > 0) ? aoddy : aeveny;
        int ay2 = (nBlkSizeY - nOverlapY) * 4 - ay1;
        int a11 = ax1 * ay1, a12 = ax1 * ay2, a21 = ax2 * ay1, a22 = ax2 * ay2;
        iout_x = (a11 * v1.x + a21 * v2.x + a12 * v3.x + a22 * v4.x) / normov;
        iout_y = (a11 * v1.y + a21 * v2.y + a12 * v3.y + a22 * v4.y) / normov;
        // generic safe_sad_t is not always safe for the next calculations
        tmp_sad = (safe_sad_t)(((smallOverlapSafeSad_t)a11 * v1.sad + (smallOverlapSafeSad_t)a21 * v2.sad + (smallOverlapSafeSad_t)a12 * v3.sad + (smallOverlapSafeSad_t)a22 * v4.sad) / normov);
#if 0
        if (tmp_sad < 0)
          _RPT1(0, "Vector and SAD Interpolate Problem: possible SAD overflow %d\n", (sad_t)tmp_sad);
#endif
      }
      else // large overlap. Weights are not quite correct but let it be
      {
        iout_x = (v1.x + v2.x + v3.x + v4.x) << 2;
        iout_y = (v1.y + v2.y + v3.y + v4.y) << 2;
        tmp_sad = ((safe_sad_t)v1.sad + v2.sad + v3.sad + v4.sad + 2) << 2;
      }

      iout_x = (iout_x >> normFactor) << mulFactor;
      iout_y = (iout_y >> normFactor) << mulFactor;
      iout_sad = (sad_t)(tmp_sad >> 4);

      // non-temporal store require better arrangement to 64bytes - to do.
      vectors[index].x = iout_x;
      vectors[index].y = iout_y;
      vectors[index].sad = iout_sad;

#if 0
      if (vectors[index].sad < 0)
        _RPT1(0, "Vector and SAD Interpolate Problem: possible SAD overflow: %d\n", vectors[index].sad);
#endif
    }	// for k < nBlkX
  }	// for l < nBlkY

}

/*
// instantiate
template void PlaneOfBlocks::InterpolatePrediction<sad_t, sad_t>(const PlaneOfBlocks &pob);
template void PlaneOfBlocks::InterpolatePrediction<sad_t, bigsad_t>(const PlaneOfBlocks &pob);
template void PlaneOfBlocks::InterpolatePrediction<bigsad_t, bigsad_t>(const PlaneOfBlocks &pob);
*/
// instantiate
template void PlaneOfBlocks::InterpolatePrediction<sad_t, sad_t>(PlaneOfBlocks& pob);
template void PlaneOfBlocks::InterpolatePrediction<sad_t, bigsad_t>(PlaneOfBlocks& pob);
template void PlaneOfBlocks::InterpolatePrediction<bigsad_t, bigsad_t>(PlaneOfBlocks& pob);

template<typename safe_sad_t, typename smallOverlapSafeSad_t>
void PlaneOfBlocks::InterpolatePrediction_sse(PlaneOfBlocks& pob)
{
  int normFactor = 3 - nLogPel + pob.nLogPel;
  int mulFactor = (normFactor < 0) ? -normFactor : 0;
  normFactor = (normFactor < 0) ? 0 : normFactor;
  int normov = (nBlkSizeX - nOverlapX) * (nBlkSizeY - nOverlapY);
  __m128 xmm_normov_rcp = _mm_rcp_ps(_mm_set1_ps((float)(normov)));

  int aoddx = (nBlkSizeX * 3 - nOverlapX * 2);
  int aevenx = (nBlkSizeX * 3 - nOverlapX * 4);
  int aoddy = (nBlkSizeY * 3 - nOverlapY * 2);
  int aeveny = (nBlkSizeY * 3 - nOverlapY * 4);
  // note: overlapping is still (v2.5.7) not processed properly
  // PF todo make faster

  // 2.7.19.22 max safe: BlkX*BlkY: sqrt(2147483647 / 3 / 255) = 1675 ,(2147483647 = 0x7FFFFFFF)
  //bool isSafeBlkSizeFor8bits = (nBlkSizeX*nBlkSizeY) < 1675;

  // 2.7.35: 
  // the limit was too small for smallOverlap case e.g. BlkSizeX=32 BlkSizeY=32 OverLapX=0 OverLapY=4
  // Worst case (approximately) ax1 * ay1: (nBlkSizeX*3 * nBlkSizeY*3)
  // 32 bit usability for 8 bits (10+bits are always using bigsad_t):
  // (evenOrOdd_xMax * evenOrOdd_yMax) * SadMax                              *4 < 0x7FFFFFFF
  //    *4: four components before /normov
  // (nBlkSizeX*3 * nBlkSizeY*3) * SadMax                                    *4 < 0x7FFFFFFF
  //    SadMax: 3 full planes (worst case 4:4:4 luma and chroma)
  // (nBlkSizeX*3 * nBlkSizeY*3) * (nBlkSizeX * nBlkSizeY * 255 * 3plane)    *4 < 0x7FFFFFFF
  // (nBlkSizeX*nBlkSizeY)^2 *9*255*3 * 4 < 0x7FFFFFFF
  // nBlkSizeX*nBlkSizeY < sqrt(...) 
  // nBlkSizeX*nBlkSizeY < 279.24 -> 280
  // => smallOverlapSafeSad_t needs to be bigsad_t when (nBlkSizeX*nBlkSizeY) >= 280

  // safe_sad_t: 16 bit worst case: 16 * sad_max: 16 * 3x32x32x65536 = 4+5+5+16 > 2^31 over limit
  //             in case of BlockSize > 32, e.g. 128x128x65536 is even more: 7+7+16=30 bits
  //             generally use big_sad_t for 10+ bits


  bool bNoOverlap = (nOverlapX == 0 && nOverlapY == 0);
  bool bSmallOverlap = nOverlapX <= (nBlkSizeX >> 1) && nOverlapY <= (nBlkSizeY >> 1);

  int iout_x, iout_y, iout_sad;

  // prefetch all vectors array
  int iSize_of_vectors = pob.vectors.size() * sizeof(vectors[0]);
  for (int i = 0; i < iSize_of_vectors; i += CACHE_LINE_SIZE)
  {
    _mm_prefetch(const_cast<const CHAR*>(reinterpret_cast<const CHAR*>(&pob.vectors[0] + i)), _MM_HINT_T0);
  }

  for (int l = 0, index = 0; l < nBlkY; l++)
  {
    for (int k = 0; k < nBlkX; k++, index++)
    {
      VECTOR v1, v2, v3, v4;
      int i = k;
      int j = l;
      if (i >= 2 * pob.nBlkX)
      {
        i = 2 * pob.nBlkX - 1;
      }
      if (j >= 2 * pob.nBlkY)
      {
        j = 2 * pob.nBlkY - 1;
      }
      int offy = -1 + 2 * (j % 2);
      int offx = -1 + 2 * (i % 2);
      int iper2 = i / 2;
      int jper2 = j / 2;

      if ((i == 0) || (i >= 2 * pob.nBlkX - 1))
      {
        if ((j == 0) || (j >= 2 * pob.nBlkY - 1))
        {
          v1 = v2 = v3 = v4 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
        }
        else
        {
          v1 = v2 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
          v3 = v4 = pob.vectors[iper2 + (jper2 + offy) * pob.nBlkX];
        }
      }
      else if ((j == 0) || (j >= 2 * pob.nBlkY - 1))
      {
        v1 = v2 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
        v3 = v4 = pob.vectors[iper2 + offx + (jper2)*pob.nBlkX];
      }
      else
      {
        v1 = pob.vectors[iper2 + (jper2)*pob.nBlkX];
        v2 = pob.vectors[iper2 + offx + (jper2)*pob.nBlkX];
        v3 = pob.vectors[iper2 + (jper2 + offy) * pob.nBlkX];
        v4 = pob.vectors[iper2 + offx + (jper2 + offy) * pob.nBlkX];
      }

      safe_sad_t tmp_sad;

      if (bNoOverlap)
      {
        iout_x = 9 * v1.x + 3 * v2.x + 3 * v3.x + v4.x;
        iout_y = 9 * v1.y + 3 * v2.y + 3 * v3.y + v4.y;
        tmp_sad = 9 * (safe_sad_t)v1.sad + 3 * (safe_sad_t)v2.sad + 3 * (safe_sad_t)v3.sad + (safe_sad_t)v4.sad + 8;

      }
      else if (bSmallOverlap) // corrected in v1.4.11
      {
        int	ax1 = (offx > 0) ? aoddx : aevenx;
        int ax2 = (nBlkSizeX - nOverlapX) * 4 - ax1;
        int ay1 = (offy > 0) ? aoddy : aeveny;
        int ay2 = (nBlkSizeY - nOverlapY) * 4 - ay1;
        int a11 = ax1 * ay1, a12 = ax1 * ay2, a21 = ax2 * ay1, a22 = ax2 * ay2;

//        iout_x = (a11 * v1.x + a21 * v2.x + a12 * v3.x + a22 * v4.x) / normov;
//        iout_y = (a11 * v1.y + a21 * v2.y + a12 * v3.y + a22 * v4.y) / normov; 
        __m128i xmm_x = _mm_set_epi32(v4.x, v3.x, v2.x, v1.x);
        __m128i xmm_y = _mm_set_epi32(v4.y, v3.y, v2.y, v1.y);
        __m128i xmm_a = _mm_set_epi32(a22, a12, a21, a11);

        xmm_x = _mm_mullo_epi32(xmm_x, xmm_a); // 32bit is enough ?
        xmm_y = _mm_mullo_epi32(xmm_y, xmm_a); // 32bit is enough ?

        xmm_x = _mm_hadd_epi32(xmm_x, xmm_x);
        xmm_y = _mm_hadd_epi32(xmm_y, xmm_y);

        xmm_x = _mm_add_epi32(xmm_x, _mm_srli_si128(xmm_x, 4));
        xmm_y = _mm_add_epi32(xmm_y, _mm_srli_si128(xmm_y, 4));

        iout_x = _mm_cvtss_si32(_mm_mul_ss(_mm_cvtepi32_ps(xmm_x), xmm_normov_rcp));
        iout_y = _mm_cvtss_si32(_mm_mul_ss(_mm_cvtepi32_ps(xmm_y), xmm_normov_rcp));
        
        // generic safe_sad_t is not always safe for the next calculations
//        tmp_sad = (safe_sad_t)(((smallOverlapSafeSad_t)a11 * v1.sad + (smallOverlapSafeSad_t)a21 * v2.sad + (smallOverlapSafeSad_t)a12 * v3.sad + (smallOverlapSafeSad_t)a22 * v4.sad) / normov);
        __m128i xmm_sad = _mm_set_epi32(v4.sad, v3.sad, v2.sad, v1.sad);
        xmm_sad = _mm_mullo_epi32(xmm_sad, xmm_a); // 32bit is enough ?
        xmm_sad = _mm_hadd_epi32(xmm_sad, xmm_sad);
        xmm_sad = _mm_add_epi32(xmm_sad, _mm_srli_si128(xmm_sad, 4));

        tmp_sad = _mm_cvtss_si32(_mm_mul_ss(_mm_cvtepi32_ps(xmm_sad), xmm_normov_rcp)); // for 32bit sad only ?

#if defined _DEBUG
        if (tmp_sad < 0)
          _RPT1(0, "Vector and SAD Interpolate Problem: possible SAD overflow %d\n", (sad_t)tmp_sad);
#endif
      }
      else // large overlap. Weights are not quite correct but let it be
      {
        iout_x = (v1.x + v2.x + v3.x + v4.x) << 2;
        iout_y = (v1.y + v2.y + v3.y + v4.y) << 2;
        tmp_sad = ((safe_sad_t)v1.sad + v2.sad + v3.sad + v4.sad + 2) << 2;
      }

      iout_x = (iout_x >> normFactor) << mulFactor;
      iout_y = (iout_y >> normFactor) << mulFactor;
      iout_sad = (sad_t)(tmp_sad >> 4);

      // non-temporal store require better arrangement to 64bytes - to do.
      vectors[index].x = iout_x;
      vectors[index].y = iout_y;
      vectors[index].sad = iout_sad;

#if 0
      if (vectors[index].sad < 0)
        _RPT1(0, "Vector and SAD Interpolate Problem: possible SAD overflow: %d\n", vectors[index].sad);
#endif
    }	// for k < nBlkX
  }	// for l < nBlkY

}

template void PlaneOfBlocks::InterpolatePrediction_sse<sad_t, sad_t>(PlaneOfBlocks& pob);
template void PlaneOfBlocks::InterpolatePrediction_sse<sad_t, bigsad_t>(PlaneOfBlocks& pob);
template void PlaneOfBlocks::InterpolatePrediction_sse<bigsad_t, bigsad_t>(PlaneOfBlocks& pob);


void PlaneOfBlocks::WriteHeaderToArray(int *array)
{
  array[0] = nBlkCount * N_PER_BLOCK + 1;
}



int PlaneOfBlocks::WriteDefaultToArray(int *array, int divideMode)
{
  array[0] = nBlkCount * N_PER_BLOCK + 1;
  //	int verybigSAD = nBlkSizeX*nBlkSizeY*256*  bits_per_pixel_factor;
  for (int i = 0; i < nBlkCount*N_PER_BLOCK; i += N_PER_BLOCK)
  {
    array[i + 1] = 0;
    array[i + 2] = 0;
    array[i + 3] = verybigSAD; // float or int!!
    //*(sad_t *)(&array[i + 3]) = verybigSAD; // float or int!!
  }

  if (nLogScale == 0)
  {
    array += array[0];
    if (divideMode)
    {
      // reserve space for divided subblocks extra level
      array[0] = nBlkCount * N_PER_BLOCK * 4 + 1; // 4 subblocks
      for (int i = 0; i < nBlkCount * 4 * N_PER_BLOCK; i += N_PER_BLOCK)
      {
        array[i + 1] = 0;
        array[i + 2] = 0;
        array[i + 3] = verybigSAD; // float or int!!
        //*(sad_t *)(&array[i + 3]) = verybigSAD; // float or int
      }
      array += array[0];
    }
  }
  return GetArraySize(divideMode);
}



int PlaneOfBlocks::GetArraySize(int divideMode)
{
  int size = 0;
  size += 1;              // mb data size storage
  size += nBlkCount * N_PER_BLOCK;  // vectors, sad, luma src, luma ref, var

  if (nLogScale == 0)
  {
    if (divideMode)
    {
      size += 1 + nBlkCount * N_PER_BLOCK * 4; // reserve space for divided subblocks extra level
    }
  }

  return size;
}

template<typename pixel_t>
void PlaneOfBlocks::FetchPredictors(WorkingArea &workarea)
{
  // Left (or right) predictor
  if ((workarea.blkScanDir == 1 && workarea.blkx > 0) || (workarea.blkScanDir == -1 && workarea.blkx < nBlkX - 1))
  {
    workarea.predictors[1] = ClipMV(workarea, vectors[workarea.blkIdx - workarea.blkScanDir]);
  }
  else
  {
    workarea.predictors[1] = ClipMV(workarea, zeroMVfieldShifted); // v1.11.1 - values instead of pointer
  }
  // fixme note:
  // MAnalyze mt-inconsistency reason #1
  // this is _not_ internal mt friendly, since here up or bottom predictors
  // are omitted for top/bottom data. Not non-mt case this happens only for
  // the most top and bottom blocks.
  // In vertically sliced multithreaded case it happens an _each_ top/bottom of the sliced block
  const bool isTop = workarea.blky == workarea.blky_beg;
  const bool isBottom = workarea.blky == workarea.blky_end - 1;
    // Up predictor
  if (!isTop)
  {
    workarea.predictors[2] = ClipMV(workarea, vectors[workarea.blkIdx - nBlkX]);
  }
  else
  {
    workarea.predictors[2] = ClipMV(workarea, zeroMVfieldShifted);
  }
    // Original problem: random, small, rare, mostly irreproducible differences between multiple encodings.
    // In all, I spent at least a week on the problem during a half year, losing hope
    // and restarting again four times. Nasty bug it was.
    // !smallestPlane: use bottom right only if a coarser level exists or else we get random
    // crap from a previous frame.
  // bottom-right predictor (from coarse level)
  if (!isBottom && 
    !smallestPlane && // v2.7.44
    ((workarea.blkScanDir == 1 && workarea.blkx < nBlkX - 1) || (workarea.blkScanDir == -1 && workarea.blkx > 0)))
  {
    workarea.predictors[3] = ClipMV(workarea, vectors[workarea.blkIdx + nBlkX + workarea.blkScanDir]);
  }
  // Up-right predictor
  else if (!isTop && ((workarea.blkScanDir == 1 && workarea.blkx < nBlkX - 1) || (workarea.blkScanDir == -1 && workarea.blkx > 0)))
  {
    workarea.predictors[3] = ClipMV(workarea, vectors[workarea.blkIdx - nBlkX + workarea.blkScanDir]);
  }
  else
  {
    workarea.predictors[3] = ClipMV(workarea, zeroMVfieldShifted);
  }

  // Median predictor
  if (!isTop) // replaced 1 by 0 - Fizick
  {
    workarea.predictors[0].x = Median(workarea.predictors[1].x, workarea.predictors[2].x, workarea.predictors[3].x);
    workarea.predictors[0].y = Median(workarea.predictors[1].y, workarea.predictors[2].y, workarea.predictors[3].y);
    //		workarea.predictors[0].sad = Median(workarea.predictors[1].sad, workarea.predictors[2].sad, workarea.predictors[3].sad);
        // but it is not true median vector (x and y may be mixed) and not its sad ?!
        // we really do not know SAD, here is more safe estimation especially for phaseshift method - v1.6.0
    workarea.predictors[0].sad = std::max(workarea.predictors[1].sad, std::max(workarea.predictors[2].sad, workarea.predictors[3].sad));
  }
  else
  {
    //		workarea.predictors[0].x = (workarea.predictors[1].x + workarea.predictors[2].x + workarea.predictors[3].x);
    //		workarea.predictors[0].y = (workarea.predictors[1].y + workarea.predictors[2].y + workarea.predictors[3].y);
    //		workarea.predictors[0].sad = (workarea.predictors[1].sad + workarea.predictors[2].sad + workarea.predictors[3].sad);
        // but for top line we have only left workarea.predictor[1] - v1.6.0
    workarea.predictors[0].x = workarea.predictors[1].x;
    workarea.predictors[0].y = workarea.predictors[1].y;
    workarea.predictors[0].sad = workarea.predictors[1].sad;
  }

   // if there are no other planes, predictor is the median
  if (smallestPlane)
  {
    workarea.predictor = workarea.predictors[0];
  }
  /*
    else
    {
      if ( workarea.predictors[0].sad < workarea.predictor.sad )// disabled by Fizick (hierarchy only!)
      {
        workarea.predictors[4] = workarea.predictor;
        workarea.predictor = workarea.predictors[0];
        workarea.predictors[0] = workarea.predictors[4];
      }
    }
  */
  //	if ( workarea.predictor.sad > LSAD ) { workarea.nLambda = 0; } // generalized (was LSAD=400) by Fizick
  
  // v2.7.11.32:
  typedef bigsad_t safe_sad_t;
  // for large block sizes int32 overflows during calculation even for 8 bits, so we always use 64 bit bigsad_t intermediate here
  // (some calculations for truemotion=true)
  // blksize  lambda                LSAD   LSAD (renormalized)        lambda*LSAD
  // 8x8      1000*(8*8)/64=1000    1200   1200*8x8/64<<0=1200          1 200 000
  // 16x16    1000*(16*16)/64=4000  1200   1200*16x16/64<<0=4800       19 200 000
  // 24x24    1000*(24*24)/64=9000  1200   1200*24x24/64<<0=10800      97 200 000
  // 32x32    1000*(32*32)/64=16000 1200   1200*32x32/64<<0=19200     307 200 000
  //          other level:   128000                         19200   2 457 600 000 (int32 overflow!)
  // 48x48    1000*(48*48)/64=36000 1200   1200*48x48/64<<0=43200   1 555 200 000 still OK
  // 64x64    1000*(64*64)/64=64000 1200   1200*64x64/64<<0=76800   4 915 200 000 (int32 overflow!)
  safe_sad_t divisor = (safe_sad_t)LSAD + (workarea.predictor.sad >> 1);
  workarea.nLambda = (int)(workarea.nLambda
                     *(safe_sad_t)LSAD
                     / divisor
                     * LSAD
                     / divisor);
  // replaced hard threshold by soft in v1.10.2 by Fizick (a liitle complex expression to avoid overflow)
  //	int a = LSAD/(LSAD + (workarea.predictor.sad>>1));
  //	workarea.nLambda = workarea.nLambda*a*a;
}

template<typename pixel_t>
void PlaneOfBlocks::FetchMorePredictors(WorkingArea& workarea)
{
  VECTOR toMedian[MAX_MEDIAN_PREDICTORS]; // to be more SIMD friendly we not need SAD for this computing
  VECTOR toClip;

  int iPredIdx = (temporal) ? 5 : 4; // place additional predictors after temporal (if present)

  // _predictorType of -1 : 3x3 median predictor, workarea.predictors[5]

  for (int dy = -1; dy < 2; dy++)
  {
    for (int dx = -1; dx < 2; dx++)
    {
      int iGetIdx;
      // scan linearly from left to right and from top to bottom
      // check for lower than zero or over max index
      iGetIdx = workarea.blkIdx + dy * nBlkX + dx;

      if ((iGetIdx < 0) || (iGetIdx > nBlkCount - 1))
      {
        toMedian[(dy + 1) * 3 + (dx + 1)] = zeroMVfieldShifted;
      }
      else
      {
        toMedian[(dy + 1) * 3 + (dx + 1)] = vectors[iGetIdx];
      }
    }
  }

  GetMedianXY(&toMedian[0], &toClip, 3*3);

  workarea.predictors[iPredIdx] = ClipMV(workarea, toClip);

  if (_predictorType >= -1) return;

  // _predictorType of -2 : 5x5 median predictor
  for (int dy = -2; dy < 3; dy++)
  {
    for (int dx = -2; dx < 3; dx++)
    {
      int iGetIdx;
      // scan linearly from left to right and from top to bottom
      // check for lower than zero or over max index
      iGetIdx = workarea.blkIdx + dy * nBlkX + dx;

      if ((iGetIdx < 0) || (iGetIdx > nBlkCount - 1))
      {
        toMedian[(dy + 2) * 5 + (dx + 2)] = zeroMVfieldShifted;
      }
      else
      {
        toMedian[(dy + 2) * 5 + (dx + 2)] = vectors[iGetIdx];
      }
    }
  }

  GetMedianXY(&toMedian[0], &toClip, 5 * 5);

  workarea.predictors[iPredIdx + 1] = ClipMV(workarea, toClip);

  if (_predictorType >= -2) return;

  // _predictorType of -3 : 7x7 median predictor
  for (int dy = -3; dy < 4; dy++)
  {
    for (int dx = -3; dx < 4; dx++)
    {
      int iGetIdx;
      // scan linearly from left to right and from top to bottom
      // check for lower than zero or over max index
      iGetIdx = workarea.blkIdx + dy * nBlkX + dx;

      if ((iGetIdx < 0) || (iGetIdx > nBlkCount - 1))
      {
        toMedian[(dy + 3) * 7 + (dx + 3)] = zeroMVfieldShifted;
      }
      else
      {
        toMedian[(dy + 3) * 7 + (dx + 3)] = vectors[iGetIdx];
      }
    }
  }

  GetMedianXY(&toMedian[0], &toClip, 7 * 7);

  workarea.predictors[iPredIdx + 2] = ClipMV(workarea, toClip);


}

MV_FORCEINLINE void PlaneOfBlocks::GetMedianXY(VECTOR* toMedian, VECTOR *vOut, int iNumMVs)
{
  // process dual coords in scalar C ?
  const int iMaxMVlength = std::max(nBlkX * nBlkSizeX, nBlkY * nBlkSizeY) * 2 * nPel; // hope it is enough ? todo: make global constant ?
  int MaxSumDM = iNumMVs * iMaxMVlength;

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow_x = MaxSumDM;
  int sum_minrow_y = MaxSumDM;
  int i_idx_minrow_x = 0;
  int i_idx_minrow_y = 0;

  for (int dmt_row = 0; dmt_row < iNumMVs; dmt_row++)
  {
    int sum_row_x = 0;
    int sum_row_y = 0;

    for (int dmt_col = 0; dmt_col < iNumMVs; dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      sum_row_x += std::abs(toMedian[dmt_row].x - toMedian[dmt_col].x);
      sum_row_y += std::abs(toMedian[dmt_row].y - toMedian[dmt_col].y);
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

  vOut[0].x = toMedian[i_idx_minrow_x].x;
  vOut[0].y = toMedian[i_idx_minrow_y].y;
  vOut[0].sad = std::max(toMedian[i_idx_minrow_x].sad, toMedian[i_idx_minrow_y].sad); // do we need SAD of predictor anywhere later ?

}

template<typename pixel_t>
MV_FORCEINLINE void PlaneOfBlocks::FetchPredictors_sse41(WorkingArea& workarea)
{
  VECTOR v1; VECTOR v2; VECTOR v3;

  // Gathering vectors first

  // Left (or right) predictor
  if ((workarea.blkScanDir == 1 && workarea.blkx > 0) || (workarea.blkScanDir == -1 && workarea.blkx < nBlkX - 1))
  {
    v1 = vectors[workarea.blkIdx - workarea.blkScanDir];
  }
  else
  {
    v1 = zeroMVfieldShifted;
  }
  // fixme note:
  // MAnalyze mt-inconsistency reason #1
  // this is _not_ internal mt friendly, since here up or bottom predictors
  // are omitted for top/bottom data. Not non-mt case this happens only for
  // the most top and bottom blocks.
  // In vertically sliced multithreaded case it happens an _each_ top/bottom of the sliced block */
  const bool isTop = workarea.blky == workarea.blky_beg;
  const bool isBottom = workarea.blky == workarea.blky_end - 1;
  // Up predictor
  if (!isTop)
  {
    v2 = vectors[workarea.blkIdx - nBlkX];
  }
  else
  {
    v2 = zeroMVfieldShifted;
  }
  // Original problem: random, small, rare, mostly irreproducible differences between multiple encodings.
  // In all, I spent at least a week on the problem during a half year, losing hope
  // and restarting again four times. Nasty bug it was.
  // !smallestPlane: use bottom right only if a coarser level exists or else we get random
  // crap from a previous frame.
// bottom-right predictor (from coarse level)
  if (!isBottom &&
    !smallestPlane && // v2.7.44
    ((workarea.blkScanDir == 1 && workarea.blkx < nBlkX - 1) || (workarea.blkScanDir == -1 && workarea.blkx > 0)))
  {
    v3 = vectors[workarea.blkIdx + nBlkX + workarea.blkScanDir];
  }
  // Up-right predictor
  else if (!isTop && ((workarea.blkScanDir == 1 && workarea.blkx < nBlkX - 1) || (workarea.blkScanDir == -1 && workarea.blkx > 0)))
  {
    v3 = vectors[workarea.blkIdx - nBlkX + workarea.blkScanDir];
  }
  else
  {
    v3 = zeroMVfieldShifted;
  }

  // Copy SADs
  workarea.predictors[1].sad = v1.sad;
  workarea.predictors[2].sad = v2.sad;
  workarea.predictors[3].sad = v3.sad;

  // ClipMV x,y
  __m128i xmm0_x, xmm1_y;

#ifdef _DEBUG
  xmm0_x = _mm_setzero_si128(); // no need to clear in release - overwritten
  xmm1_y = _mm_setzero_si128();
#endif

  // SSE 4.1 !
  xmm0_x = _mm_cvtsi32_si128(v1.x);
  xmm0_x = _mm_insert_epi32(xmm0_x, v2.x, 1);  // SSE 4.1 !
  xmm0_x = _mm_insert_epi32(xmm0_x, v3.x, 2);

  xmm1_y = _mm_cvtsi32_si128(v1.y);
  xmm1_y = _mm_insert_epi32(xmm1_y, v2.y, 1); // SSE 4.1 !
  xmm1_y = _mm_insert_epi32(xmm1_y, v3.y, 2);


  __m128i xmm2_DxMin = _mm_set1_epi32(workarea.nDxMin); // for AVX2 builds - will be vpbroadcastd - enable AVX2 in C++ compiler !
  __m128i xmm3_DxMax = _mm_set1_epi32(workarea.nDxMax - 1);
  __m128i xmm4_DyMin = _mm_set1_epi32(workarea.nDyMin);
  __m128i xmm5_DyMax = _mm_set1_epi32(workarea.nDyMax - 1); 

  xmm0_x = _mm_max_epi32(xmm0_x, xmm2_DxMin); // SSE 4.1 !!
  xmm1_y = _mm_max_epi32(xmm1_y, xmm4_DyMin);

  xmm0_x = _mm_min_epi32(xmm0_x, xmm3_DxMax);
  xmm1_y = _mm_min_epi32(xmm1_y, xmm5_DyMax); // no < and >= and -1 (?) for this version

  workarea.predictors[1].x = _mm_cvtsi128_si32(xmm0_x);
  workarea.predictors[2].x = _mm_extract_epi32(xmm0_x, 1);
  workarea.predictors[3].x = _mm_extract_epi32(xmm0_x, 2);

  workarea.predictors[1].y = _mm_cvtsi128_si32(xmm1_y);
  workarea.predictors[2].y = _mm_extract_epi32(xmm1_y, 1);
  workarea.predictors[3].y = _mm_extract_epi32(xmm1_y, 2);

  // Median predictor
  if (!isTop) // replaced 1 by 0 - Fizick
  {
//   workarea.predictors[0].x = Median(workarea.predictors[1].x, workarea.predictors[2].x, workarea.predictors[3].x);
//   workarea.predictors[0].y = Median(workarea.predictors[1].y, workarea.predictors[2].y, workarea.predictors[3].y);
    // Median as of   a + b + c - imax(a, imax(b, c)) - imin(c, imin(a, b)) looks correct.

    /*  __m128i xmm0_a = _mm_set_epi32(0, 0, workarea.predictors[1].y, workarea.predictors[1].x);
    __m128i xmm1_b = _mm_set_epi32(0, 0, workarea.predictors[2].y, workarea.predictors[2].x);
    __m128i xmm2_c = _mm_set_epi32(0, 0, workarea.predictors[3].y, workarea.predictors[3].x);*/
    
    __m128i xmm0_a = _mm_loadl_epi64((__m128i*) & workarea.predictors[1].x);
    __m128i xmm1_b = _mm_loadl_epi64((__m128i*) & workarea.predictors[2].x);
    __m128i xmm2_c = _mm_loadl_epi64((__m128i*) & workarea.predictors[3].x);

    __m128i xmm3_sum = _mm_add_epi32(_mm_add_epi32(xmm0_a, xmm1_b), xmm2_c);
    __m128i xmm4_min = _mm_min_epi32(_mm_min_epi32(xmm0_a, xmm1_b), xmm2_c);
    __m128i xmm5_max = _mm_max_epi32(_mm_max_epi32(xmm0_a, xmm1_b), xmm2_c);

    xmm3_sum = _mm_sub_epi32(_mm_sub_epi32(xmm3_sum, xmm5_max), xmm4_min);

    _mm_storel_epi64((__m128i*) & workarea.predictors[0].x, xmm3_sum);

/*    workarea.predictors[0].x = _mm_cvtsi128_si32(xmm3_sum);
    workarea.predictors[0].y = _mm_extract_epi32(xmm3_sum, 1);*/

    //		workarea.predictors[0].sad = Median(workarea.predictors[1].sad, workarea.predictors[2].sad, workarea.predictors[3].sad);
        // but it is not true median vector (x and y may be mixed) and not its sad ?!
        // we really do not know SAD, here is more safe estimation especially for phaseshift method - v1.6.0
//    workarea.predictors[0].sad = std::max(workarea.predictors[1].sad, std::max(workarea.predictors[2].sad, workarea.predictors[3].sad));
    __m128i xmm6_sad1, xmm7_sad2, xmm8_sad3;

    xmm6_sad1 = _mm_cvtsi32_si128(v1.sad);
    xmm7_sad2 = _mm_cvtsi32_si128(v2.sad);
    xmm8_sad3 = _mm_cvtsi32_si128(v3.sad);

    xmm6_sad1 = _mm_max_epi32(_mm_max_epi32(xmm6_sad1, xmm7_sad2), xmm8_sad3);

    workarea.predictors[0].sad = _mm_cvtsi128_si32(xmm6_sad1);
  }
  else
  {
    //		workarea.predictors[0].x = (workarea.predictors[1].x + workarea.predictors[2].x + workarea.predictors[3].x);
    //		workarea.predictors[0].y = (workarea.predictors[1].y + workarea.predictors[2].y + workarea.predictors[3].y);
    //		workarea.predictors[0].sad = (workarea.predictors[1].sad + workarea.predictors[2].sad + workarea.predictors[3].sad);
        // but for top line we have only left workarea.predictor[1] - v1.6.0
    workarea.predictors[0].x = workarea.predictors[1].x;
    workarea.predictors[0].y = workarea.predictors[1].y;
    workarea.predictors[0].sad = workarea.predictors[1].sad;
  }

  // if there are no other planes, predictor is the median
  if (smallestPlane)
  {
    workarea.predictor = workarea.predictors[0];
  }
  /*
    else
    {
      if ( workarea.predictors[0].sad < workarea.predictor.sad )// disabled by Fizick (hierarchy only!)
      {
        workarea.predictors[4] = workarea.predictor;
        workarea.predictor = workarea.predictors[0];
        workarea.predictors[0] = workarea.predictors[4];
      }
    }
  */
  //	if ( workarea.predictor.sad > LSAD ) { workarea.nLambda = 0; } // generalized (was LSAD=400) by Fizick

  // v2.7.11.32:
  typedef bigsad_t safe_sad_t;
  // for large block sizes int32 overflows during calculation even for 8 bits, so we always use 64 bit bigsad_t intermediate here
  // (some calculations for truemotion=true)
  // blksize  lambda                LSAD   LSAD (renormalized)        lambda*LSAD
  // 8x8      1000*(8*8)/64=1000    1200   1200*8x8/64<<0=1200          1 200 000
  // 16x16    1000*(16*16)/64=4000  1200   1200*16x16/64<<0=4800       19 200 000
  // 24x24    1000*(24*24)/64=9000  1200   1200*24x24/64<<0=10800      97 200 000
  // 32x32    1000*(32*32)/64=16000 1200   1200*32x32/64<<0=19200     307 200 000
  //          other level:   128000                         19200   2 457 600 000 (int32 overflow!)
  // 48x48    1000*(48*48)/64=36000 1200   1200*48x48/64<<0=43200   1 555 200 000 still OK
  // 64x64    1000*(64*64)/64=64000 1200   1200*64x64/64<<0=76800   4 915 200 000 (int32 overflow!)
  safe_sad_t divisor = (safe_sad_t)LSAD + (workarea.predictor.sad >> 1);

  __m128 xmm_divisor = _mm_setzero_ps();
  __m128 xmm_LSAD = _mm_cvt_si2ss(xmm_LSAD, LSAD);
  if (sizeof(safe_sad_t) == 4) // safe_sad_t may be 64bit ?
  {
    xmm_divisor = _mm_cvt_si2ss(xmm_divisor, divisor);
  }
  else
  {
    xmm_divisor = _mm_cvtsi64_ss(xmm_divisor, divisor);
  }

  xmm_LSAD = _mm_mul_ss(xmm_LSAD, xmm_LSAD);
  xmm_divisor = _mm_mul_ss(xmm_divisor, xmm_divisor);
  xmm_divisor = _mm_rcp_ss(xmm_divisor);

  xmm_LSAD = _mm_mul_ss(xmm_divisor, xmm_LSAD);

  __m128 xmm_nLambda = _mm_cvt_si2ss(xmm_nLambda, workarea.nLambda);

  xmm_nLambda = _mm_mul_ss(xmm_nLambda, xmm_LSAD);
  workarea.nLambda = _mm_cvt_ss2si(xmm_nLambda);
    
  // workarea.nLambda = (int)(workarea.nLambda * (safe_sad_t)LSAD / divisor * LSAD / divisor); correct ?

   // replaced hard threshold by soft in v1.10.2 by Fizick (a liitle complex expression to avoid overflow)
   //	int a = LSAD/(LSAD + (workarea.predictor.sad>>1));
   //	workarea.nLambda = workarea.nLambda*a*a;
}

template<typename pixel_t>
MV_FORCEINLINE void PlaneOfBlocks::FetchPredictors_sse41_intraframe(WorkingArea& workarea)
{
  VECTOR v1; VECTOR v2; VECTOR v3;

  // Gathering vectors first
  v1 = vectors[workarea.blkIdx - workarea.blkScanDir];
  // fixme note:
  // MAnalyze mt-inconsistency reason #1
  // this is _not_ internal mt friendly, since here up or bottom predictors
  // are omitted for top/bottom data. Not non-mt case this happens only for
  // the most top and bottom blocks.
  // In vertically sliced multithreaded case it happens an _each_ top/bottom of the sliced block */
  v2 = vectors[workarea.blkIdx - nBlkX];

  // Original problem: random, small, rare, mostly irreproducible differences between multiple encodings.
// In all, I spent at least a week on the problem during a half year, losing hope
// and restarting again four times. Nasty bug it was.
// !smallestPlane: use bottom right only if a coarser level exists or else we get random
// crap from a previous frame.
// bottom-right predictor (from coarse level)
  if (!smallestPlane)
  {
    v3 = vectors[workarea.blkIdx + nBlkX + workarea.blkScanDir];
  }
  // Up-right predictor
  else
  {
    v3 = vectors[workarea.blkIdx - nBlkX + workarea.blkScanDir];
  }

  // Copy SADs
  workarea.predictors[1].sad = v1.sad;
  workarea.predictors[2].sad = v2.sad;
  workarea.predictors[3].sad = v3.sad;

  // ClipMV x,y
  __m128i xmm0_x, xmm1_y;
  __m128i xmm6_sad1, xmm7_sad2, xmm8_sad3;
#ifdef _DEBUG
  xmm0_x = _mm_setzero_si128(); // no need to clear in release - overwritten
  xmm1_y = _mm_setzero_si128();
#endif

  // SSE 4.1 !
  xmm0_x = _mm_cvtsi32_si128(v1.x);
  xmm0_x = _mm_insert_epi32(xmm0_x, v2.x, 1);  // SSE 4.1 !
  xmm0_x = _mm_insert_epi32(xmm0_x, v3.x, 2);

  xmm1_y = _mm_cvtsi32_si128(v1.y);
  xmm1_y = _mm_insert_epi32(xmm1_y, v2.y, 1); // SSE 4.1 !
  xmm1_y = _mm_insert_epi32(xmm1_y, v3.y, 2);

  xmm6_sad1 = _mm_cvtsi32_si128(v1.sad);
  xmm7_sad2 = _mm_cvtsi32_si128(v2.sad);
  xmm8_sad3 = _mm_cvtsi32_si128(v3.sad);

  __m128i xmm2_DxMin = _mm_set1_epi32(workarea.nDxMin); // for AVX2 builds - will be vpbroadcastd - enable AVX2 in C++ compiler !
  __m128i xmm3_DxMax = _mm_set1_epi32(workarea.nDxMax - 1);
  __m128i xmm4_DyMin = _mm_set1_epi32(workarea.nDyMin);
  __m128i xmm5_DyMax = _mm_set1_epi32(workarea.nDyMax - 1);

  xmm0_x = _mm_max_epi32(xmm0_x, xmm2_DxMin); // SSE 4.1 !!
  xmm1_y = _mm_max_epi32(xmm1_y, xmm4_DyMin);

  xmm0_x = _mm_min_epi32(xmm0_x, xmm3_DxMax);
  xmm1_y = _mm_min_epi32(xmm1_y, xmm5_DyMax); // no < and >= and -1 (?) for this version

  xmm6_sad1 = _mm_max_epi32(xmm6_sad1, xmm7_sad2);
  xmm6_sad1 = _mm_max_epi32(xmm6_sad1, xmm8_sad3);

  workarea.predictors[1].x = _mm_cvtsi128_si32(xmm0_x);
  workarea.predictors[2].x = _mm_extract_epi32(xmm0_x, 1);
  workarea.predictors[3].x = _mm_extract_epi32(xmm0_x, 2);

  workarea.predictors[1].y = _mm_cvtsi128_si32(xmm1_y);
  workarea.predictors[2].y = _mm_extract_epi32(xmm1_y, 1);
  workarea.predictors[3].y = _mm_extract_epi32(xmm1_y, 2);

  //   workarea.predictors[0].x = Median(workarea.predictors[1].x, workarea.predictors[2].x, workarea.predictors[3].x);
  //   workarea.predictors[0].y = Median(workarea.predictors[1].y, workarea.predictors[2].y, workarea.predictors[3].y);
      // Median as of   a + b + c - imax(a, imax(b, c)) - imin(c, imin(a, b)) looks correct.

  __m128i xmm0_a = _mm_loadl_epi64((__m128i*) & workarea.predictors[1].x);
  __m128i xmm1_b = _mm_loadl_epi64((__m128i*) & workarea.predictors[2].x);
  __m128i xmm2_c = _mm_loadl_epi64((__m128i*) & workarea.predictors[3].x);

  __m128i xmm3_sum = _mm_add_epi32(_mm_add_epi32(xmm0_a, xmm1_b), xmm2_c);
  __m128i xmm4_min = _mm_min_epi32(_mm_min_epi32(xmm0_a, xmm1_b), xmm2_c);
  __m128i xmm5_max = _mm_max_epi32(_mm_max_epi32(xmm0_a, xmm1_b), xmm2_c);

  xmm3_sum = _mm_sub_epi32(_mm_sub_epi32(xmm3_sum, xmm5_max), xmm4_min);

  _mm_storel_epi64((__m128i*) & workarea.predictors[0].x, xmm3_sum);

  //		workarea.predictors[0].sad = Median(workarea.predictors[1].sad, workarea.predictors[2].sad, workarea.predictors[3].sad);
      // but it is not true median vector (x and y may be mixed) and not its sad ?!
      // we really do not know SAD, here is more safe estimation especially for phaseshift method - v1.6.0
//    workarea.predictors[0].sad = std::max(workarea.predictors[1].sad, std::max(workarea.predictors[2].sad, workarea.predictors[3].sad));
  workarea.predictors[0].sad = _mm_cvtsi128_si32(xmm6_sad1);

  // if there are no other planes, predictor is the median
  if (smallestPlane)
  {
    workarea.predictor = workarea.predictors[0];
  }
  /*
    else
    {
      if ( workarea.predictors[0].sad < workarea.predictor.sad )// disabled by Fizick (hierarchy only!)
      {
        workarea.predictors[4] = workarea.predictor;
        workarea.predictor = workarea.predictors[0];
        workarea.predictors[0] = workarea.predictors[4];
      }
    }
  */
  //	if ( workarea.predictor.sad > LSAD ) { workarea.nLambda = 0; } // generalized (was LSAD=400) by Fizick

  // v2.7.11.32:
  typedef bigsad_t safe_sad_t;
  // for large block sizes int32 overflows during calculation even for 8 bits, so we always use 64 bit bigsad_t intermediate here
  // (some calculations for truemotion=true)
  // blksize  lambda                LSAD   LSAD (renormalized)        lambda*LSAD
  // 8x8      1000*(8*8)/64=1000    1200   1200*8x8/64<<0=1200          1 200 000
  // 16x16    1000*(16*16)/64=4000  1200   1200*16x16/64<<0=4800       19 200 000
  // 24x24    1000*(24*24)/64=9000  1200   1200*24x24/64<<0=10800      97 200 000
  // 32x32    1000*(32*32)/64=16000 1200   1200*32x32/64<<0=19200     307 200 000
  //          other level:   128000                         19200   2 457 600 000 (int32 overflow!)
  // 48x48    1000*(48*48)/64=36000 1200   1200*48x48/64<<0=43200   1 555 200 000 still OK
  // 64x64    1000*(64*64)/64=64000 1200   1200*64x64/64<<0=76800   4 915 200 000 (int32 overflow!)
  safe_sad_t divisor = (safe_sad_t)LSAD + (workarea.predictor.sad >> 1);

  __m128 xmm_divisor = _mm_setzero_ps();
  __m128 xmm_LSAD = _mm_cvt_si2ss(xmm_LSAD, LSAD);
  if (sizeof(safe_sad_t) == 4) // safe_sad_t may be 64bit ?
  {
    xmm_divisor = _mm_cvt_si2ss(xmm_divisor, divisor);
  }
  else
  {
    xmm_divisor = _mm_cvtsi64_ss(xmm_divisor, divisor);
  }

  xmm_LSAD = _mm_mul_ss(xmm_LSAD, xmm_LSAD);
  xmm_divisor = _mm_mul_ss(xmm_divisor, xmm_divisor);
  xmm_divisor = _mm_rcp_ss(xmm_divisor);

  xmm_LSAD = _mm_mul_ss(xmm_divisor, xmm_LSAD);

  __m128 xmm_nLambda = _mm_cvt_si2ss(xmm_nLambda, workarea.nLambda);

  xmm_nLambda = _mm_mul_ss(xmm_nLambda, xmm_LSAD);
  workarea.nLambda = _mm_cvt_ss2si(xmm_nLambda);

  // workarea.nLambda = (int)(workarea.nLambda * (safe_sad_t)LSAD / divisor * LSAD / divisor); correct ?

   // replaced hard threshold by soft in v1.10.2 by Fizick (a liitle complex expression to avoid overflow)
   //	int a = LSAD/(LSAD + (workarea.predictor.sad>>1));
   //	workarea.nLambda = workarea.nLambda*a*a;
}

template<typename pixel_t>
void PlaneOfBlocks::FetchPredictors_avx2_intraframe(WorkingArea& workarea) // linker do not see it in the _avx2 file ???
{
  VECTOR v1; VECTOR v2; VECTOR v3;

  // Gathering vectors first
  v1 = vectors[workarea.blkIdx - workarea.blkScanDir];
  // fixme note:
  // MAnalyze mt-inconsistency reason #1
  // this is _not_ internal mt friendly, since here up or bottom predictors
  // are omitted for top/bottom data. Not non-mt case this happens only for
  // the most top and bottom blocks.
  // In vertically sliced multithreaded case it happens an _each_ top/bottom of the sliced block */
  v2 = vectors[workarea.blkIdx - nBlkX];

  // Original problem: random, small, rare, mostly irreproducible differences between multiple encodings.
// In all, I spent at least a week on the problem during a half year, losing hope
// and restarting again four times. Nasty bug it was.
// !smallestPlane: use bottom right only if a coarser level exists or else we get random
// crap from a previous frame.
// bottom-right predictor (from coarse level)
  if (!smallestPlane)
  {
    v3 = vectors[workarea.blkIdx + nBlkX + workarea.blkScanDir];
  }
  // Up-right predictor
  else
  {
    v3 = vectors[workarea.blkIdx - nBlkX + workarea.blkScanDir];
  }

  // Copy SADs
  workarea.predictors[1].sad = v1.sad;
  workarea.predictors[2].sad = v2.sad;
  workarea.predictors[3].sad = v3.sad;

  // ClipMV x,y of v1,v2,v3
  __m256i ymm0_3yx = _mm256_set_epi32(0, 0, v3.y, v3.x, v2.y, v2.x, v1.y, v1.x); // compiler decide how to ?

  ymm0_3yx = _mm256_min_epi32(ymm0_3yx, _mm256_broadcastq_epi64(_mm_set_epi32(0, 0, workarea.nDyMax - 1, workarea.nDxMax - 1)));
  ymm0_3yx = _mm256_max_epi32(ymm0_3yx, _mm256_broadcastq_epi64(_mm_set_epi32(0, 0, workarea.nDyMin, workarea.nDxMin)));
  /*
  workarea.predictors[1].x = _mm256_extract_epi32(ymm0_3xy, 1);
  workarea.predictors[1].y = _mm256_extract_epi32(ymm0_3xy, 0);

  workarea.predictors[2].x = _mm256_extract_epi32(ymm0_3xy, 3);
  workarea.predictors[2].y = _mm256_extract_epi32(ymm0_3xy, 2);

  workarea.predictors[3].x = _mm256_extract_epi32(ymm0_3xy, 5);
  workarea.predictors[3].y = _mm256_extract_epi32(ymm0_3xy, 4);
  */
  _mm_storel_epi64((__m128i*) & workarea.predictors[1].x, _mm256_castsi256_si128(ymm0_3yx));
  _mm_storel_epi64((__m128i*) & workarea.predictors[2].x, _mm256_castsi256_si128(_mm256_srli_si256(ymm0_3yx, 8))); // shift high 64 to low in low 128 of ymm
  _mm_storel_epi64((__m128i*) & workarea.predictors[3].x, _mm256_castsi256_si128(_mm256_permute4x64_epi64(ymm0_3yx, 14))); // shift high 128 to low part of ymm

  //   workarea.predictors[0].x = Median(workarea.predictors[1].x, workarea.predictors[2].x, workarea.predictors[3].x);
  //   workarea.predictors[0].y = Median(workarea.predictors[1].y, workarea.predictors[2].y, workarea.predictors[3].y);
      // Median as of   a + b + c - imax(a, imax(b, c)) - imin(c, imin(a, b)) looks correct.

  __m128i xmm0_a = _mm_loadl_epi64((__m128i*) & workarea.predictors[1].x);
  __m128i xmm1_b = _mm_loadl_epi64((__m128i*) & workarea.predictors[2].x);
  __m128i xmm2_c = _mm_loadl_epi64((__m128i*) & workarea.predictors[3].x);

  __m128i xmm3_sum = _mm_add_epi32(_mm_add_epi32(xmm0_a, xmm1_b), xmm2_c);
  __m128i xmm4_min = _mm_min_epi32(_mm_min_epi32(xmm0_a, xmm1_b), xmm2_c);
  __m128i xmm5_max = _mm_max_epi32(_mm_max_epi32(xmm0_a, xmm1_b), xmm2_c);

  xmm3_sum = _mm_sub_epi32(_mm_sub_epi32(xmm3_sum, xmm5_max), xmm4_min);

  _mm_storel_epi64((__m128i*) & workarea.predictors[0].x, xmm3_sum);

  //		workarea.predictors[0].sad = Median(workarea.predictors[1].sad, workarea.predictors[2].sad, workarea.predictors[3].sad);
      // but it is not true median vector (x and y may be mixed) and not its sad ?!
      // we really do not know SAD, here is more safe estimation especially for phaseshift method - v1.6.0
//    workarea.predictors[0].sad = std::max(workarea.predictors[1].sad, std::max(workarea.predictors[2].sad, workarea.predictors[3].sad));
  __m128i xmm6_sad1, xmm7_sad2, xmm8_sad3;
  xmm6_sad1 = _mm_cvtsi32_si128(v1.sad);
  xmm7_sad2 = _mm_cvtsi32_si128(v2.sad);
  xmm8_sad3 = _mm_cvtsi32_si128(v3.sad);

  xmm6_sad1 = _mm_max_epi32(_mm_max_epi32(xmm6_sad1, xmm7_sad2), xmm8_sad3);

  workarea.predictors[0].sad = _mm_cvtsi128_si32(xmm6_sad1);

  // if there are no other planes, predictor is the median
  if (smallestPlane)
  {
    workarea.predictor = workarea.predictors[0];
  }
  /*
    else
    {
      if ( workarea.predictors[0].sad < workarea.predictor.sad )// disabled by Fizick (hierarchy only!)
      {
        workarea.predictors[4] = workarea.predictor;
        workarea.predictor = workarea.predictors[0];
        workarea.predictors[0] = workarea.predictors[4];
      }
    }
  */
  //	if ( workarea.predictor.sad > LSAD ) { workarea.nLambda = 0; } // generalized (was LSAD=400) by Fizick

  // v2.7.11.32:
  typedef bigsad_t safe_sad_t;
  // for large block sizes int32 overflows during calculation even for 8 bits, so we always use 64 bit bigsad_t intermediate here
  // (some calculations for truemotion=true)
  // blksize  lambda                LSAD   LSAD (renormalized)        lambda*LSAD
  // 8x8      1000*(8*8)/64=1000    1200   1200*8x8/64<<0=1200          1 200 000
  // 16x16    1000*(16*16)/64=4000  1200   1200*16x16/64<<0=4800       19 200 000
  // 24x24    1000*(24*24)/64=9000  1200   1200*24x24/64<<0=10800      97 200 000
  // 32x32    1000*(32*32)/64=16000 1200   1200*32x32/64<<0=19200     307 200 000
  //          other level:   128000                         19200   2 457 600 000 (int32 overflow!)
  // 48x48    1000*(48*48)/64=36000 1200   1200*48x48/64<<0=43200   1 555 200 000 still OK
  // 64x64    1000*(64*64)/64=64000 1200   1200*64x64/64<<0=76800   4 915 200 000 (int32 overflow!)
  safe_sad_t divisor = (safe_sad_t)LSAD + (workarea.predictor.sad >> 1);

  __m128 xmm_divisor = _mm_setzero_ps();
  __m128 xmm_LSAD = _mm_cvt_si2ss(xmm_LSAD, LSAD);
  if (sizeof(safe_sad_t) == 4) // safe_sad_t may be 64bit ?
  {
    xmm_divisor = _mm_cvt_si2ss(xmm_divisor, divisor);
  }
  else
  {
    xmm_divisor = _mm_cvtsi64_ss(xmm_divisor, divisor);
  }

  xmm_LSAD = _mm_mul_ss(xmm_LSAD, xmm_LSAD);
  xmm_divisor = _mm_mul_ss(xmm_divisor, xmm_divisor);
  xmm_divisor = _mm_rcp_ss(xmm_divisor);

  xmm_LSAD = _mm_mul_ss(xmm_divisor, xmm_LSAD);

  __m128 xmm_nLambda = _mm_cvt_si2ss(xmm_nLambda, workarea.nLambda);

  xmm_nLambda = _mm_mul_ss(xmm_nLambda, xmm_LSAD);
  workarea.nLambda = _mm_cvt_ss2si(xmm_nLambda);

  // workarea.nLambda = (int)(workarea.nLambda * (safe_sad_t)LSAD / divisor * LSAD / divisor); correct ?

   // replaced hard threshold by soft in v1.10.2 by Fizick (a liitle complex expression to avoid overflow)
   //	int a = LSAD/(LSAD + (workarea.predictor.sad>>1));
   //	workarea.nLambda = workarea.nLambda*a*a;

  _mm256_zeroupper();

}



template<typename pixel_t>
void PlaneOfBlocks::Refine(WorkingArea &workarea)
{
  // then, we refine, according to the search type
  switch (searchType) {
  case ONETIME:
    for (int i = nSearchParam; i > 0; i /= 2)
    {
      OneTimeSearch<pixel_t>(workarea, i);
    }
    break;
  case NSTEP:
    NStepSearch<pixel_t>(workarea, nSearchParam);
    break;
  case LOGARITHMIC:
    for (int i = nSearchParam; i > 0; i /= 2)
    {
      DiamondSearch<pixel_t>(workarea, i);
    }
    break;
  case EXHAUSTIVE: {

    //		ExhaustiveSearch(nSearchParam);
    int mvx = workarea.bestMV.x;
    int mvy = workarea.bestMV.y;

    // DTL TEST
    // one-pass Exa search by DTL
    // only for 8 bit, nPel==1, nSearchParam == 1 and 2 and 4 && nBlkSizeX == 8 && nBlkSizeY == 8 && !chroma
    // c or avx2. See function dispatcher
    if (0 != optSearchOption && nPel == 1 && avx2) { // keep compatibility - new addition - only for pel=1 now !! other will cause buggy x,y,sad output, only avx2 and later
      if (nSearchParam <= MAX_SUPPORTED_EXH_SEARCHPARAM) {
        // nSearchParam can change during the algorithm so we are choosing from prefilled function pointer table
        auto ExhaustiveSearchFunction = ExhaustiveSearchFunctions[nSearchParam];
        if (nullptr != ExhaustiveSearchFunction) {
          (this->*ExhaustiveSearchFunction)(workarea, mvx, mvy);
          break;
        }
      }
    }
    
    for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
    {
      ExpandingSearch<pixel_t>(workarea, i, 1, mvx, mvy);
    }
  }
                   break;

                   //	if ( searchType & SQUARE )
                   //	{
                   //		SquareSearch();
                   //	}
  case HEX2SEARCH:
    Hex2Search<pixel_t>(workarea, nSearchParam);
    break;
  case UMHSEARCH:
    UMHSearch<pixel_t>(workarea, nSearchParam, workarea.bestMV.x, workarea.bestMV.y);
    break;
  case HSEARCH:
  {
    int mvx = workarea.bestMV.x;
    int mvy = workarea.bestMV.y;
    for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
    {
      CheckMV<pixel_t>(workarea, mvx - i, mvy);
      CheckMV<pixel_t>(workarea, mvx + i, mvy);
    }
  }
  break;
  case VSEARCH:
  {
    int mvx = workarea.bestMV.x;
    int mvy = workarea.bestMV.y;
    for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
    {
      CheckMV<pixel_t>(workarea, mvx, mvy - i);
      CheckMV<pixel_t>(workarea, mvx, mvy + i);
    }
  }
  break;
  }
}

MV_FORCEINLINE bool PlaneOfBlocks::IsVectorChecked(uint64_t xy) // 2.7.46
{
  int i;
  for (i = 0; i < iNumCheckedVectors; i++)
  {
    if (checked_mv_vectors[i] == xy) return true;
  }

  // record it to checked
  checked_mv_vectors[iNumCheckedVectors] = xy;
  iNumCheckedVectors++;

  return false;
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch(WorkingArea& workarea)
{
  if (_predictorType < 0)
  {
    FetchMorePredictors<pixel_t>(workarea);
  }

  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
  if (sse41 && optSearchOption > 0)
  {
    FetchPredictors_sse41<pixel_t>(workarea);
  }
  else
    FetchPredictors<pixel_t>(workarea);

  iNumCheckedVectors = 0;

  sad_t sad;
  sad_t cost;
  sad_t saduv;

#ifdef ALLOW_DCT
  if (dctmode != 0) // DCT method (luma only - currently use normal spatial SAD chroma)
  {
    // make dct of source block
    if (dctmode <= 4) //don't do the slow dct conversion if SATD used
    {
      workarea.DCT->DCTBytes2D(workarea.pSrc[0], nSrcPitch[0], &workarea.dctSrc[0], dctpitch);
      // later, workarea.dctSrc is used as a reference block
    }
  }
  if (dctmode >= 3) // most use it and it should be fast anyway //if (dctmode == 3 || dctmode == 4) // check it
  {
    workarea.srcLuma = LUMA(workarea.pSrc[0], nSrcPitch[0]);
  }
#endif	// ALLOW_DCT

  // We treat zero alone
  // Do we bias zero with not taking into account distorsion ?
  workarea.bestMV.x = zeroMVfieldShifted.x;
  workarea.bestMV.y = zeroMVfieldShifted.y;
/*  saduv = (chroma) ?
    ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, 0, 0), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, 0, 0), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0; */
  saduv = (chroma) ?
    ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, 0, 0), nRefPitch[1])
      + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, 0, 0), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
  sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, 0, zeroMVfieldShifted.y));
  sad += saduv;
  workarea.bestMV.sad = sad;
  workarea.nMinCost = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2

  checked_mv_vectors[iNumCheckedVectors] = 0;
  iNumCheckedVectors++;

  VECTOR bestMVMany[MAX_PREDICTOR+3];
  int nMinCostMany[MAX_PREDICTOR+3];

  for (int i = 0; i < 8; i++) nMinCostMany[i] = verybigSAD + 1; // init trymany with verybig value for skipped by already checked vectors points !

  if (tryMany)
  {
    //  refine around zero
    Refine<pixel_t>(workarea);
    bestMVMany[0] = workarea.bestMV;    // save bestMV
    nMinCostMany[0] = workarea.nMinCost;
  }

  // Global MV predictor  - added by Fizick
  workarea.globalMVPredictor = ClipMV(workarea, workarea.globalMVPredictor);

  //	if ( workarea.IsVectorOK(workarea.globalMVPredictor.x, workarea.globalMVPredictor.y ) )
  {
    if (!IsVectorChecked((uint64_t)workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32)))
    {
/*      saduv = (chroma) ?
        ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[1])
          + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      saduv = (chroma) ?
        ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[1])
          + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
      sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y));
      sad += saduv;
      cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

      if (cost < workarea.nMinCost || tryMany)
      {
        workarea.bestMV.x = workarea.globalMVPredictor.x;
        workarea.bestMV.y = workarea.globalMVPredictor.y;
        workarea.bestMV.sad = sad;
        workarea.nMinCost = cost;
      }
      if (tryMany)
      {
        // refine around global
        Refine<pixel_t>(workarea);    // reset bestMV
        bestMVMany[1] = workarea.bestMV;    // save bestMV
        nMinCostMany[1] = workarea.nMinCost;
      }
    }
    //	}
    //	Then, the predictor :
    //	if (   (( workarea.predictor.x != zeroMVfieldShifted.x ) || ( workarea.predictor.y != zeroMVfieldShifted.y ))
    //	    && (( workarea.predictor.x != workarea.globalMVPredictor.x ) || ( workarea.predictor.y != workarea.globalMVPredictor.y )))
    //	{
    if (!IsVectorChecked((uint64_t)workarea.predictor.x | ((uint64_t)workarea.predictor.y << 32)))
    {
/*      saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
        + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
                + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
      sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
      sad += saduv;
      cost = sad;

      if (cost < workarea.nMinCost || tryMany)
      {
        workarea.bestMV.x = workarea.predictor.x;
        workarea.bestMV.y = workarea.predictor.y;
        workarea.bestMV.sad = sad;
        workarea.nMinCost = cost;
      }
    }

  }

  if (tryMany)
  {
    // refine around predictor
    Refine<pixel_t>(workarea);    // reset bestMV
    bestMVMany[2] = workarea.bestMV;    // save bestMV
    nMinCostMany[2] = workarea.nMinCost;
  }

  // then all the other predictors
  int npred = (temporal) ? 5 : 4;

  // add number of possible more median predictors
  npred += std::abs(_predictorType); // maybe not best but convert -1..-3 to 1..3 additional predictors to check

  for (int i = 0; i < npred; i++)
  {
    if (tryMany)
    {
      workarea.nMinCost = verybigSAD + 1;
    }

    if (!IsVectorChecked((uint64_t)workarea.predictors[i].x | ((uint64_t)workarea.predictors[i].y << 32)))
    {
      CheckMV0<pixel_t>(workarea, workarea.predictors[i].x, workarea.predictors[i].y);
    }

    if (tryMany)
    {
      // refine around predictor
      Refine<pixel_t>(workarea);    // reset bestMV
      bestMVMany[i + 3] = workarea.bestMV;    // save bestMV
      nMinCostMany[i + 3] = workarea.nMinCost;
    }
  }	// for i


  if (tryMany)
  {
    // select best of multi best
    workarea.nMinCost = verybigSAD + 1;
    for (int i = 0; i < npred + 3; i++)
    {
      if (nMinCostMany[i] < workarea.nMinCost)
      {
        workarea.bestMV = bestMVMany[i];
        workarea.nMinCost = nMinCostMany[i];
      }
    }
  }
  else
  {
    // then, we refine, according to the search type
    Refine<pixel_t>(workarea);
  }
  sad_t foundSAD = workarea.bestMV.sad;

  const int		BADCOUNT_LIMIT = 16;

  // fixme note:
  // MAnalyze mt-inconsistency reason #2
  // 'badcount' can be increased in different order when multithreaded
  // (processing vertically sliced vector data parallel)
  // so the expression in the condition below can be different for each run
  // depending on the order the parallel tasks increase badcount

  // bad vector, try wide search
  if (workarea.blkIdx > 1 + workarea.blky_beg * nBlkX
    && foundSAD > (badSAD + badSAD*badcount / BADCOUNT_LIMIT))
  {
    // with some soft limit (BADCOUNT_LIMIT) of bad cured vectors (time consumed)
    ++badcount;

    DebugPrintf(
      "bad  blk=%d x=%d y=%d sad=%d mean=%d iter=%d",
      workarea.blkIdx,
      workarea.bestMV.x,
      workarea.bestMV.y,
      workarea.bestMV.sad,
      workarea.planeSAD / (workarea.blkIdx - workarea.blky_beg * nBlkX),
      workarea.iter
    );
    /*
        int mvx0 = workarea.bestMV.x; // store for comparing
        int mvy0 = workarea.bestMV.y;
        int msad0 = workarea.bestMV.sad;
        int mcost0 = workarea.nMinCost;
    */
    if (badrange > 0) // UMH
    {

      //			UMHSearch(badrange*nPel, workarea.bestMV.x, workarea.bestMV.y);

      //			if (workarea.bestMV.sad > foundSAD/2)
      {
        // rathe good is not found, lets try around zero
//				UMHSearch(workarea, badSADRadius, abs(mvx0)%4 - 2, abs(mvy0)%4 - 2);
        UMHSearch<pixel_t>(workarea, badrange*nPel, 0, 0);
      }
    }

    else if (badrange < 0) // ESA
    {
      /*
            workarea.bestMV.x = mvx0; // restore  for comparing
            workarea.bestMV.y = mvy0;
            workarea.bestMV.sad = msad0;
            workarea.nMinCost = mcost0;

            int mvx = workarea.bestMV.x; // store to not move the search center!
            int mvy = workarea.bestMV.y;
            int msad = workarea.bestMV.sad;

            for ( int i = 1; i < -badrange*nPel; i+=nPel )// at radius
            {
              ExpandingSearch(i, nPel, mvx, mvy);
              if (workarea.bestMV.sad < foundSAD/4)
              {
                break; // stop search
              }
            }

            if (workarea.bestMV.sad > foundSAD/2 && abs(mvx)+abs(mvy) > badSADRadius/2)
            {
              // rathe good is not found, lets try around zero
              mvx = 0; // store to not move the search center!
              mvy = 0;
      */
      for (int i = 1; i < -badrange*nPel; i += nPel)// at radius
      {
        ExpandingSearch<pixel_t>(workarea, i, nPel, 0, 0);
        if (workarea.bestMV.sad < foundSAD / 4)
        {
          break; // stop search if rathe good is found
        }
      }	// for i
    }	// badrange < 0

    int mvx = workarea.bestMV.x; // refine in small area
    int mvy = workarea.bestMV.y;
    for (int i = 1; i < nPel; i++)// small radius
    {
      ExpandingSearch<pixel_t>(workarea, i, 1, mvx, mvy);
    }
    DebugPrintf("best blk=%d x=%d y=%d sad=%d iter=%d", workarea.blkIdx, workarea.bestMV.x, workarea.bestMV.y, workarea.bestMV.sad, workarea.iter);
  }	// bad vector, try wide search

  // we store the result
  vectors[workarea.blkIdx].x = workarea.bestMV.x;
  vectors[workarea.blkIdx].y = workarea.bestMV.y;
  vectors[workarea.blkIdx].sad = workarea.bestMV.sad;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

// DTL test
template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_no_pred(WorkingArea& workarea) // no new predictors - only interpolated from previous iteration
{
    typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

    sad_t sad;

    if (smallestPlane)
    {
      workarea.bestMV = zeroMV;
      workarea.nMinCost = verybigSAD + 1;
    }
    else
    {
      workarea.bestMV = workarea.predictor;
      sad = workarea.predictor.sad;
      workarea.nMinCost = (sad * 2) + ((penaltyNew * (safe_sad_t)sad) >> 8); // *2 - typically sad from previous level is lower about 2 times. depend on noise/spectrum ?  
    }

    // then, we refine, according to the search type
    Refine<pixel_t>(workarea);

    // we store the result
    vectors[workarea.blkIdx].x = workarea.bestMV.x;
    vectors[workarea.blkIdx].y = workarea.bestMV.y;
    vectors[workarea.blkIdx].sad = workarea.bestMV.sad;

    workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

// DTL test
template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_no_refine(WorkingArea& workarea) // no refine - only predictor check
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  sad_t sad;

  if (smallestPlane) // never get here - normal use is sequence of params with 'real' search like optPredictorsType="3,x" where x < 3.
  {
    workarea.bestMV = zeroMV;
    workarea.nMinCost = verybigSAD + 1;
  }
  else
  {
    workarea.bestMV = workarea.predictor; // already ClipMV() processed in the search_mv_slice
    // only recalculate sad for interpolated predictor to be compatible with old/typical thSAD setting in MDegrain
      sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.bestMV.x, workarea.bestMV.y));
/*      sad_t saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.bestMV.x, workarea.bestMV.y), nRefPitch[1])
        + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.bestMV.x, workarea.bestMV.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      sad_t saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.bestMV.x, workarea.bestMV.y), nRefPitch[1])
                + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.bestMV.x, workarea.bestMV.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;

      workarea.bestMV.sad = sad + saduv;
  }

  // only recalculate sad for interpolated predictor to be compatible with old/typical thSAD setting in MDegrain
//  CheckMV0<pixel_t>(workarea, workarea.predictor.x, workarea.predictor.y);

  // we store the result
  vectors[workarea.blkIdx].x = workarea.bestMV.x;
  vectors[workarea.blkIdx].y = workarea.bestMV.y;
  vectors[workarea.blkIdx].sad = workarea.bestMV.sad;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}


// DTL test
template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_glob_med_pred(WorkingArea& workarea)
{
    typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
    if (sse41 && optSearchOption > 0)
    {
      FetchPredictors_sse41<pixel_t>(workarea);
    }
    else
      FetchPredictors<pixel_t>(workarea);

    iNumCheckedVectors = 0;
    
    sad_t sad;
    sad_t saduv;
    sad_t cost;

    int iRefPitchY = 0;
    int iRefPitchU = 0;
    int iRefPitchV = 0;

    // We treat zero alone
    // Do we bias zero with not taking into account distorsion ?
    workarea.bestMV.x = zeroMVfieldShifted.x;
    workarea.bestMV.y = zeroMVfieldShifted.y;

    VECTOR bestMVMany[3]; // zero, global, median predictor only
    int nMinCostMany[3];

    for (int i = 0; i < 3; i++) nMinCostMany[i] = verybigSAD + 1; // init with verybig values to prevent bug with skipped already checked positions !

    if (iUseSubShift == 0)
    {
/*      saduv = (chroma) ?
        ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, 0, 0), nRefPitch[1])
          + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, 0, 0), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      saduv = (chroma) ?
        ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, 0, 0), nRefPitch[1])
          + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, 0, 0), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
      sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, 0, zeroMVfieldShifted.y));
    }
    else
    {
      if (chroma)
      {
        const unsigned char* ptrRefU = GetRefBlockUSubShifted(workarea, 0, 0, iRefPitchU);
        const unsigned char* ptrRefV = GetRefBlockVSubShifted(workarea, 0, 0, iRefPitchV);
/*        saduv = ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
          + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);*/
        saduv = ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
          + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);
      }
      else
        saduv = 0;
    
      const unsigned char* ptrRef = GetRefBlockSubShifted(workarea, 0, zeroMVfieldShifted.y, iRefPitchY);
//      sad = SAD(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
      sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
//      sad = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlockSubShifted(workarea, 0, zeroMVfieldShifted.y, iRefPitchY), iRefPitchY);
    }

    sad += saduv;
    workarea.bestMV.sad = sad;
    workarea.nMinCost = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2

    checked_mv_vectors[iNumCheckedVectors] = 0;
    iNumCheckedVectors++;

    if (tryMany)
    {
      //  refine around zero
      Refine<pixel_t>(workarea);
      bestMVMany[0] = workarea.bestMV;    // save bestMV
      nMinCostMany[0] = workarea.nMinCost;
    }

   // Global MV predictor  - added by Fizick
    workarea.globalMVPredictor = ClipMV(workarea, workarea.globalMVPredictor);

    if (!IsVectorChecked(workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32)))
    {
      if (iUseSubShift == 0)
      {
/*        saduv = (chroma) ?
          ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[1])
            + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
        saduv = (chroma) ?
          ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[1])
            + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;

        sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y));
      }
      else
      {
        if (chroma)
        {
          const unsigned char* ptrRefU = GetRefBlockUSubShifted(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y, iRefPitchU);
          const unsigned char* ptrRefV = GetRefBlockVSubShifted(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y, iRefPitchV);
/*          saduv = ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
            + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);*/
          saduv = ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
            + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);

        }
        else
          saduv = 0;

        const unsigned char* ptrRef = GetRefBlockSubShifted(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y, iRefPitchY);
//        sad = SAD(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
        sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
      }
        sad += saduv;
        sad_t cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

        if (cost < workarea.nMinCost || tryMany)
        {
          workarea.bestMV.x = workarea.globalMVPredictor.x;
          workarea.bestMV.y = workarea.globalMVPredictor.y;
          workarea.bestMV.sad = sad;
          workarea.nMinCost = cost;
        }

        if (tryMany)
        {
          // refine around global
          Refine<pixel_t>(workarea);    // reset bestMV
          bestMVMany[1] = workarea.bestMV;    // save bestMV
          nMinCostMany[1] = workarea.nMinCost;
        }
    }

    //	}
    //	Then, the predictor :
    //	if (   (( workarea.predictor.x != zeroMVfieldShifted.x ) || ( workarea.predictor.y != zeroMVfieldShifted.y ))
    //	    && (( workarea.predictor.x != workarea.globalMVPredictor.x ) || ( workarea.predictor.y != workarea.globalMVPredictor.y )))
    //	{
    if (!IsVectorChecked((uint64_t)workarea.predictor.x | ((uint64_t)workarea.predictor.y << 32)))
    {
      if (iUseSubShift == 0)
      {
        /*
        saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
          + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
        saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
          + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
        sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
      }
      else
      {
        if (chroma)
        {
          const unsigned char* ptrRefU = GetRefBlockUSubShifted(workarea, workarea.predictor.x, workarea.predictor.y, iRefPitchU);
          const unsigned char* ptrRefV = GetRefBlockVSubShifted(workarea, workarea.predictor.x, workarea.predictor.y, iRefPitchV);
/*          saduv = ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
            + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);*/
          saduv = ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
            + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);
        }
        else
          saduv = 0;

        const unsigned char* ptrRef = GetRefBlockSubShifted(workarea, workarea.predictor.x, workarea.predictor.y, iRefPitchY);
//        sad = SAD(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
        sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);

      }
        sad += saduv;
        cost = sad;

        if (cost < workarea.nMinCost || tryMany)
        {
          workarea.bestMV.x = workarea.predictor.x;
          workarea.bestMV.y = workarea.predictor.y;
          workarea.bestMV.sad = sad;
          workarea.nMinCost = cost;
        }

        if (tryMany)
        {
          // refine around median
          Refine<pixel_t>(workarea);    // reset bestMV
          bestMVMany[2] = workarea.bestMV;    // save bestMV
          nMinCostMany[2] = workarea.nMinCost;
        }
    }

    if (tryMany)
    {
      // select best of multi best
      workarea.nMinCost = verybigSAD + 1;
      for (int i = 0; i < 3; i++)
      {
        if (nMinCostMany[i] < workarea.nMinCost)
        {
          workarea.bestMV = bestMVMany[i];
          workarea.nMinCost = nMinCostMany[i];
        }
      }
    }
    else
    {
      // then, we refine, according to the search type
      Refine<pixel_t>(workarea);
    }

#ifdef RETURN_PREV_LEVEL_SAD_AT_LEVEL_0
    // special feature, disable in the standard release !!!
    if (nSearchParam == 1) // finest level 0 - test for better denoise 
    {
      workarea.bestMV.sad = workarea.predictor.sad; // use previous level sad
    }
#endif  

    // we store the result
    vectors[workarea.blkIdx].x = workarea.bestMV.x;
    vectors[workarea.blkIdx].y = workarea.bestMV.y;
    vectors[workarea.blkIdx].sad = workarea.bestMV.sad;

    workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

// DTL test
template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO2(WorkingArea& workarea)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  if (workarea.bIntraframe)
  {
    FetchPredictors_avx2_intraframe<pixel_t>(workarea); // faster
  }
  else
  {
    FetchPredictors_sse41<pixel_t>(workarea);
  }


  sad_t sad;
  sad_t cost;

  // We treat zero alone
  // Do we bias zero with not taking into account distorsion ?
  workarea.bestMV.x = zeroMVfieldShifted.x;
  workarea.bestMV.y = zeroMVfieldShifted.y;
  sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, 0, zeroMVfieldShifted.y));
  workarea.bestMV.sad = sad;
  workarea.nMinCost = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2

  iNumCheckedVectors = 0;
  checked_mv_vectors[iNumCheckedVectors] = 0;
  iNumCheckedVectors++;

  /*    if (!IsVectorChecked(workarea.predictors[i].x | (workarea.predictors[i].y << 32)))
      {
        CheckMV0<pixel_t>(workarea, workarea.predictors[i].x, workarea.predictors[i].y);

        checked_mv_vectors[iNumCheckedVectors] = workarea.predictors[i].x | (workarea.predictors[i].y << 32);
        iNumCheckedVectors++;
      }
      */


  // Global MV predictor  - added by Fizick
  workarea.globalMVPredictor = ClipMV_SO2(workarea, workarea.globalMVPredictor);

  if (!IsVectorChecked((uint64_t)workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32)))
  {
    sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y));
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = workarea.globalMVPredictor.x;
      workarea.bestMV.y = workarea.globalMVPredictor.y;
      workarea.bestMV.sad = sad;
      workarea.nMinCost = cost;
    }
  }
  //	}
  //	Then, the predictor :
  //	if (   (( workarea.predictor.x != zeroMVfieldShifted.x ) || ( workarea.predictor.y != zeroMVfieldShifted.y ))
  //	    && (( workarea.predictor.x != workarea.globalMVPredictor.x ) || ( workarea.predictor.y != workarea.globalMVPredictor.y )))
  //	{
  if (!IsVectorChecked((uint64_t)workarea.predictor.x | ((uint64_t)workarea.predictor.y << 32)))
  {
    sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
    cost = sad;

    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = workarea.predictor.x;
      workarea.bestMV.y = workarea.predictor.y;
      workarea.bestMV.sad = sad;
      workarea.nMinCost = cost;
    }
  }
  // then all the other predictors
  // compute checks on motion distortion first and skip MV if above cost:

  __m256i ymm2_yx_predictors = _mm256_set_epi32(workarea.predictors[3].y, workarea.predictors[3].x, workarea.predictors[2].y, workarea.predictors[2].x, \
    workarea.predictors[1].y, workarea.predictors[1].x, workarea.predictors[0].y, workarea.predictors[0].x);
//  __m256i ymm3_predictor = _mm256_set_epi32(workarea.predictor.y, workarea.predictor.x, workarea.predictor.y, workarea.predictor.x, \
//    workarea.predictor.y, workarea.predictor.x, workarea.predictor.y, workarea.predictor.x);
  __m256i ymm3_predictor = _mm256_broadcastq_epi64(_mm_set_epi32(0, 0, workarea.predictor.y, workarea.predictor.x)); // hope movq + vpbroadcast

  __m256i ymm_d1d2 = _mm256_sub_epi32(ymm3_predictor, ymm2_yx_predictors);
  ymm_d1d2 = _mm256_add_epi32(_mm256_mullo_epi32(ymm_d1d2, ymm_d1d2), _mm256_srli_si256(ymm_d1d2, 4));

  __m256i ymm_dist = _mm256_permutevar8x32_epi32(ymm_d1d2, _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0));
  __m128i xmm_nLambda = _mm_set1_epi32(workarea.nLambda);
  __m128i  xmm0_cost = _mm_srli_epi32(_mm_mullo_epi32(xmm_nLambda, _mm256_castsi256_si128(ymm_dist)), 8);
  __m128i xmm_mask = _mm_cmplt_epi32(xmm0_cost, _mm_set1_epi32(workarea.nMinCost));
    int iMask = _mm_movemask_epi8(xmm_mask);
    _mm256_zeroupper(); // need ?

    // if ((iMask & 0x1111) == 0x1111) - use 4-predictors CheckMV0_avx2() - to do.
  // vectors were clipped in FetchPredictors - no new IsVectorOK() check ?
  if ((iMask & 0x1) != 0)
  {
    if (!IsVectorChecked((uint64_t)workarea.predictors[0].x | ((uint64_t)workarea.predictors[0].y << 32)))
    {
      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[0].x, workarea.predictors[0].y, _mm_extract_epi32(xmm0_cost, 0));
    }
  }
  if ((iMask & 0x10) != 0)
  {
    if (!IsVectorChecked((uint64_t)workarea.predictors[1].x | ((uint64_t)workarea.predictors[1].y << 32)))
    {
      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[1].x, workarea.predictors[1].y, _mm_extract_epi32(xmm0_cost, 1));
    }
  }
  if ((iMask & 0x100) != 0)
  {
    if (!IsVectorChecked((uint64_t)workarea.predictors[2].x | ((uint64_t)workarea.predictors[2].y << 32)))
    {
      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[2].x, workarea.predictors[2].y, _mm_extract_epi32(xmm0_cost, 2));
    }
  }
  if ((iMask & 0x1000) != 0)
  {
    if (!IsVectorChecked((uint64_t)workarea.predictors[3].x | ((uint64_t)workarea.predictors[3].y << 32)))
    {
      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[3].x, workarea.predictors[3].y, _mm_extract_epi32(xmm0_cost, 3));
    }
  }
  /*
  CheckMV0<pixel_t>(workarea, workarea.predictors[0].x, workarea.predictors[0].y);
  CheckMV0<pixel_t>(workarea, workarea.predictors[1].x, workarea.predictors[1].y);
  CheckMV0<pixel_t>(workarea, workarea.predictors[2].x, workarea.predictors[2].y);
  CheckMV0<pixel_t>(workarea, workarea.predictors[3].x, workarea.predictors[3].y);
  */

  // then, we refine, 
  // sp = 1 for level=0 (finest) sp = 2 for other levels
  (this->*ExhaustiveSearch_SO2)(workarea, workarea.bestMV.x, workarea.bestMV.y);

  // we store the result
  vectors[workarea.blkIdx] = workarea.bestMV;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO2_glob_med_pred(WorkingArea& workarea)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  if (workarea.bIntraframe)
  {
//    FetchPredictors_sse41_intraframe<pixel_t>(workarea);
    FetchPredictors_avx2_intraframe<pixel_t>(workarea); // faster
  }
  else
  {
    FetchPredictors_sse41<pixel_t>(workarea);
  }

  sad_t sad;

  workarea.bestMV = zeroMV;
  workarea.nMinCost = verybigSAD + 1;

/*  // We treat zero alone
  // Do we bias zero with not taking into account distorsion ?
  workarea.bestMV.x = zeroMVfieldShifted.x;
  workarea.bestMV.y = zeroMVfieldShifted.y;
  sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, 0, zeroMVfieldShifted.y));
  workarea.bestMV.sad = sad;
  workarea.nMinCost = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2
  */
 // Global MV predictor  - added by Fizick
  workarea.globalMVPredictor = ClipMV_SO2(workarea, workarea.globalMVPredictor);

  sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y));
  sad_t cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

  iNumCheckedVectors = 0;
  checked_mv_vectors[iNumCheckedVectors] = workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32);
  iNumCheckedVectors++;
  
  if (cost < workarea.nMinCost)
  {
    workarea.bestMV.x = workarea.globalMVPredictor.x;
    workarea.bestMV.y = workarea.globalMVPredictor.y;
    workarea.bestMV.sad = sad;
    workarea.nMinCost = cost;
  }
  //	}
  //	Then, the predictor :
  //	if (   (( workarea.predictor.x != zeroMVfieldShifted.x ) || ( workarea.predictor.y != zeroMVfieldShifted.y ))
  //	    && (( workarea.predictor.x != workarea.globalMVPredictor.x ) || ( workarea.predictor.y != workarea.globalMVPredictor.y )))
  //	{
  if (!IsVectorChecked((uint64_t)workarea.predictor.x | ((uint64_t)workarea.predictor.y << 32)))
  {
    sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
    cost = sad;

    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = workarea.predictor.x;
      workarea.bestMV.y = workarea.predictor.y;
      workarea.bestMV.sad = sad;
      workarea.nMinCost = cost;
    }
  }
  
  // then, we refine, 
  // sp = 1 for level=0 (finest) sp = 2 for other levels
  (this->*ExhaustiveSearch_SO2)(workarea, workarea.bestMV.x, workarea.bestMV.y);

#ifdef RETURN_PREV_LEVEL_SAD_AT_LEVEL_0
  // special feature, disable in the standard release !!!
  if (nSearchParam == 1) // finest level 0 - test for better denoise 
  {
    workarea.bestMV.sad = workarea.predictor.sad; // use previous level sad
  }
#endif  

  // we store the result
  vectors[workarea.blkIdx] = workarea.bestMV;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO2_no_pred(WorkingArea& workarea)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  sad_t sad;

  if (smallestPlane)
  {
    workarea.bestMV = zeroMV;
    workarea.nMinCost = verybigSAD + 1;
  }
  else
  {
    workarea.bestMV = workarea.predictor;
    sad = workarea.predictor.sad;
    workarea.nMinCost = (sad * 2) + ((penaltyNew * (safe_sad_t)sad) >> 8); // *2 - typically sad from previous level is lower about 2 times. depend on noise/spectrum ?
  }

  // then, we refine, 
  // sp = 1 for level=0 (finest) sp = 2 for other levels
  (this->*ExhaustiveSearch_SO2)(workarea, workarea.bestMV.x, workarea.bestMV.y);

  // we store the result
  vectors[workarea.blkIdx] = workarea.bestMV;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO2_no_refine(WorkingArea& workarea)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  sad_t sad;

  if (smallestPlane) // do not use it with levels=1 - it will not denoise at all.
  {
    workarea.bestMV = zeroMV;
    workarea.nMinCost = verybigSAD + 1;
  }
  else
  {
    workarea.bestMV = workarea.predictor;
    workarea.bestMV.sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.bestMV.x, workarea.bestMV.y));
  }

  // we store the result
  vectors[workarea.blkIdx] = workarea.bestMV;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}


MV_FORCEINLINE bool PlaneOfBlocks::IsVectorsCoherent(VECTOR_XY* vectors_coh_check, int cnt)
{
  VECTOR_XY v1;

  v1.x = vectors_coh_check[0].x;
  v1.y = vectors_coh_check[0].y;

  for (int i = 1; i < cnt; i++)
  {
    if ((vectors_coh_check[i].x != v1.x) || (vectors_coh_check[i].y != v1.y))
      return false;
  }

  return true;
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO3_no_pred(WorkingArea& workarea, int* pBlkData)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  sad_t sad;

  VECTOR_XY vectors_coh_check[MAX_MULTI_BLOCKS_8x8_AVX2];

  if (smallestPlane)
  {
    workarea.bestMV = zeroMV;
    workarea.nMinCost = verybigSAD + 1;
  }
  else
  {
    workarea.bestMV = workarea.predictor;
    sad = workarea.predictor.sad;
    workarea.nMinCost = (sad * 2) + ((penaltyNew * (safe_sad_t)sad) >> 8); // *2 - typically sad from previous level is lower about 2 times. depend on noise/spectrum ?
  }

  // check 4 blocks prev level predictor coherency
  vectors_coh_check[0].x = workarea.predictor.x;
  vectors_coh_check[0].y = workarea.predictor.y;

  for (int i = 1; i < MAX_MULTI_BLOCKS_8x8_AVX2; i++)
  {
    VECTOR predictor_next = ClipMV_SO2(workarea, vectors[workarea.blkIdx + i]); // need update dy/dx max min to 4 blocks advance ?
    vectors_coh_check[i].x = predictor_next.x;
    vectors_coh_check[i].y = predictor_next.y;
  }
  
  if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX2))
  {
    // level 0 only here
    ExhaustiveSearch8x8_uint8_4Blks_Z_np1_sp1_avx2(workarea, workarea.bestMV.x, workarea.bestMV.y, pBlkData); // + center (zero shift pos)
  }
  else // predictors for next 3 blocks not coherent - perform standart per block search
  {
    ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(workarea, workarea.bestMV.x, workarea.bestMV.y);

    for (int iBlkNum = 1; iBlkNum < 4; iBlkNum++)
    {
      if (smallestPlane)
      {
        workarea.bestMV = zeroMV;
        workarea.nMinCost = verybigSAD + 1;
      }
      else
      {
        workarea.bestMV = ClipMV_SO2(workarea, vectors[workarea.blkIdx + iBlkNum]);
        sad = workarea.predictor.sad;
        workarea.nMinCost = (sad * 2) + ((penaltyNew * (safe_sad_t)sad) >> 8); // *2 - typically sad from previous level is lower about 2 times. depend on noise/spectrum ?
      }

      ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(workarea, workarea.bestMV.x, workarea.bestMV.y);

      pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 0] = workarea.bestMV.x;
      pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 1] = workarea.bestMV.y;
      pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 2] = workarea.bestMV.sad;

    }
  }
  // we store the result
//  vectors[workarea.blkIdx] = workarea.bestMV; - no need to store back because no analyse local level predictors in this type of search
  // stored internally in Exa_search()

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO4_no_pred(WorkingArea& workarea, int* pBlkData)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  sad_t sad;

  VECTOR_XY vectors_coh_check[MAX_MULTI_BLOCKS_8x8_AVX512];

  if (smallestPlane)
  {
    workarea.bestMV = zeroMV;
    workarea.nMinCost = verybigSAD + 1;
  }
  else
  {
    workarea.bestMV = workarea.predictor;
    sad = workarea.predictor.sad;
    workarea.nMinCost = (sad * 2) + ((penaltyNew * (safe_sad_t)sad) >> 8); // *2 - typically sad from previous level is lower about 2 times. depend on noise/spectrum ?
  }

  // check 16 blocks prev level predictor coherency
  vectors_coh_check[0].x = workarea.predictor.x;
  vectors_coh_check[0].y = workarea.predictor.y;

  for (int i = 1; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
  {
    VECTOR predictor_next = ClipMV_SO2(workarea, vectors[workarea.blkIdx + i]); // need update dy/dx max min to 4 blocks advance ?
    vectors_coh_check[i].x = predictor_next.x;
    vectors_coh_check[i].y = predictor_next.y;
  }

  if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX512))
  {
    // level 0 only here
    ExhaustiveSearch8x8_uint8_16Blks_Z_np1_sp1_avx512(workarea, workarea.bestMV.x, workarea.bestMV.y, pBlkData); // + center (zero shift pos)
  }
  else // predictors for next 15 blocks not coherent - perform standart per block search, todo: make additional 4-blocks groups analysys and try 4-blocks AVX2 search
  {
    ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512(workarea, workarea.bestMV.x, workarea.bestMV.y);

    for (int iBlkNum = 1; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX512; iBlkNum++)
    {
      if (smallestPlane)
      {
        workarea.bestMV = zeroMV;
        workarea.nMinCost = verybigSAD + 1;
      }
      else
      {
        workarea.bestMV = ClipMV_SO2(workarea, vectors[workarea.blkIdx + iBlkNum]);
        sad = workarea.predictor.sad;
        workarea.nMinCost = (sad * 2) + ((penaltyNew * (safe_sad_t)sad) >> 8); // *2 - typically sad from previous level is lower about 2 times. depend on noise/spectrum ?
      }

      ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512(workarea, workarea.bestMV.x, workarea.bestMV.y);

      pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 0] = workarea.bestMV.x;
      pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 1] = workarea.bestMV.y;
      pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 2] = workarea.bestMV.sad;

    }
  }
  // we store the result
//  vectors[workarea.blkIdx] = workarea.bestMV; - no need to store back because no analyse local level predictors in this type of search
  // stored internally in Exa_search()

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}


template<typename pixel_t>
void PlaneOfBlocks::DiamondSearch(WorkingArea &workarea, int length)
{
  // The meaning of the directions are the following :
  //		* 1 means right
  //		* 2 means left
  //		* 4 means down
  //		* 8 means up
  // So 1 + 4 means down right, and so on...

  int dx;
  int dy;

  // We begin by making no assumption on which direction to search.
  int direction = 15;

  int lastDirection;

  while (direction > 0)
  {
    dx = workarea.bestMV.x;
    dy = workarea.bestMV.y;
    lastDirection = direction;
    direction = 0;

    // First, we look the directions that were hinted by the previous step
    // of the algorithm. If we find one, we add it to the set of directions
    // we'll test next
    if (lastDirection & 1) CheckMV2<pixel_t>(workarea, dx + length, dy, &direction, 1);
    if (lastDirection & 2) CheckMV2<pixel_t>(workarea, dx - length, dy, &direction, 2);
    if (lastDirection & 4) CheckMV2<pixel_t>(workarea, dx, dy + length, &direction, 4);
    if (lastDirection & 8) CheckMV2<pixel_t>(workarea, dx, dy - length, &direction, 8);

    // If one of the directions improves the SAD, we make further tests
    // on the diagonals
    if (direction)
    {
      lastDirection = direction;
      dx = workarea.bestMV.x;
      dy = workarea.bestMV.y;

      if (lastDirection & 3)
      {
        CheckMV2<pixel_t>(workarea, dx, dy + length, &direction, 4);
        CheckMV2<pixel_t>(workarea, dx, dy - length, &direction, 8);
      }
      else
      {
        CheckMV2<pixel_t>(workarea, dx + length, dy, &direction, 1);
        CheckMV2<pixel_t>(workarea, dx - length, dy, &direction, 2);
      }
    }

    // If not, we do not stop here. We infer from the last direction the
    // diagonals to be checked, because we might be lucky.
    else
    {
      switch (lastDirection)
      {
      case 1:
        CheckMV2<pixel_t>(workarea, dx + length, dy + length, &direction, 1 + 4);
        CheckMV2<pixel_t>(workarea, dx + length, dy - length, &direction, 1 + 8);
        break;
      case 2:
        CheckMV2<pixel_t>(workarea, dx - length, dy + length, &direction, 2 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy - length, &direction, 2 + 8);
        break;
      case 4:
        CheckMV2<pixel_t>(workarea, dx + length, dy + length, &direction, 1 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy + length, &direction, 2 + 4);
        break;
      case 8:
        CheckMV2<pixel_t>(workarea, dx + length, dy - length, &direction, 1 + 8);
        CheckMV2<pixel_t>(workarea, dx - length, dy - length, &direction, 2 + 8);
        break;
      case 1 + 4:
        CheckMV2<pixel_t>(workarea, dx + length, dy + length, &direction, 1 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy + length, &direction, 2 + 4);
        CheckMV2<pixel_t>(workarea, dx + length, dy - length, &direction, 1 + 8);
        break;
      case 2 + 4:
        CheckMV2<pixel_t>(workarea, dx + length, dy + length, &direction, 1 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy + length, &direction, 2 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy - length, &direction, 2 + 8);
        break;
      case 1 + 8:
        CheckMV2<pixel_t>(workarea, dx + length, dy + length, &direction, 1 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy - length, &direction, 2 + 8);
        CheckMV2<pixel_t>(workarea, dx + length, dy - length, &direction, 1 + 8);
        break;
      case 2 + 8:
        CheckMV2<pixel_t>(workarea, dx - length, dy - length, &direction, 2 + 8);
        CheckMV2<pixel_t>(workarea, dx - length, dy + length, &direction, 2 + 4);
        CheckMV2<pixel_t>(workarea, dx + length, dy - length, &direction, 1 + 8);
        break;
      default:
        // Even the default case may happen, in the first step of the
        // algorithm for example.
        CheckMV2<pixel_t>(workarea, dx + length, dy + length, &direction, 1 + 4);
        CheckMV2<pixel_t>(workarea, dx - length, dy + length, &direction, 2 + 4);
        CheckMV2<pixel_t>(workarea, dx + length, dy - length, &direction, 1 + 8);
        CheckMV2<pixel_t>(workarea, dx - length, dy - length, &direction, 2 + 8);
        break;
      }
    }	// if ! direction
  }	// while direction > 0
}



/*
void PlaneOfBlocks::SquareSearch(WorkingArea &workarea)
{
  ExhaustiveSearch(workarea, 1);
}



void PlaneOfBlocks::ExhaustiveSearch(WorkingArea &workarea, int s)// diameter = 2*s - 1
{
  int i, j;
  VECTOR mv = workarea.bestMV;

  for ( i = -s + 1; i < 0; i++ )
    for ( j = -s + 1; j < s; j++ )
      CheckMV(workarea, mv.x + i, mv.y + j);

  for ( i = 1; i < s; i++ )
    for ( j = -s + 1; j < s; j++ )
      CheckMV(workarea, mv.x + i, mv.y + j);

  for ( j = -s + 1; j < 0; j++ )
    CheckMV(workarea, mv.x, mv.y + j);

  for ( j = 1; j < s; j++ )
    CheckMV(workarea, mv.x, mv.y + j);

}
*/



template<typename pixel_t>
void PlaneOfBlocks::NStepSearch(WorkingArea &workarea, int stp)
{
  int dx, dy;
  int length = stp;
  while (length > 0)
  {
    dx = workarea.bestMV.x;
    dy = workarea.bestMV.y;

    CheckMV<pixel_t>(workarea, dx + length, dy + length);
    CheckMV<pixel_t>(workarea, dx + length, dy);
    CheckMV<pixel_t>(workarea, dx + length, dy - length);
    CheckMV<pixel_t>(workarea, dx, dy - length);
    CheckMV<pixel_t>(workarea, dx, dy + length);
    CheckMV<pixel_t>(workarea, dx - length, dy + length);
    CheckMV<pixel_t>(workarea, dx - length, dy);
    CheckMV<pixel_t>(workarea, dx - length, dy - length);

    length--;
  }
}




template<typename pixel_t>
void PlaneOfBlocks::OneTimeSearch(WorkingArea &workarea, int length)
{
  int direction = 0;
  int dx = workarea.bestMV.x;
  int dy = workarea.bestMV.y;

  CheckMV2<pixel_t>(workarea, dx - length, dy, &direction, 2);
  CheckMV2<pixel_t>(workarea, dx + length, dy, &direction, 1);

  if (direction == 1)
  {
    while (direction)
    {
      direction = 0;
      dx += length;
      CheckMV2<pixel_t>(workarea, dx + length, dy, &direction, 1);
    }
  }
  else if (direction == 2)
  {
    while (direction)
    {
      direction = 0;
      dx -= length;
      CheckMV2<pixel_t>(workarea, dx - length, dy, &direction, 1);
    }
  }

  CheckMV2<pixel_t>(workarea, dx, dy - length, &direction, 2);
  CheckMV2<pixel_t>(workarea, dx, dy + length, &direction, 1);

  if (direction == 1)
  {
    while (direction)
    {
      direction = 0;
      dy += length;
      CheckMV2<pixel_t>(workarea, dx, dy + length, &direction, 1);
    }
  }
  else if (direction == 2)
  {
    while (direction)
    {
      direction = 0;
      dy -= length;
      CheckMV2<pixel_t>(workarea, dx, dy - length, &direction, 1);
    }
  }
}



template<typename pixel_t>
void PlaneOfBlocks::ExpandingSearch(WorkingArea &workarea, int r, int s, int mvx, int mvy) // diameter = 2*r + 1, step=s
{ // part of true enhaustive search (thin expanding square) around mvx, mvy
  int i, j;
  //	VECTOR mv = workarea.bestMV; // bug: it was pointer assignent, not values, so iterative! - v2.1
    // sides of square without corners
  for (i = -r + s; i < r; i += s) // without corners! - v2.1
  {
    CheckMV<pixel_t>(workarea, mvx + i, mvy - r);
    CheckMV<pixel_t>(workarea, mvx + i, mvy + r);
  }

  for (j = -r + s; j < r; j += s)
  {
    CheckMV<pixel_t>(workarea, mvx - r, mvy + j);
    CheckMV<pixel_t>(workarea, mvx + r, mvy + j);
  }

  // then corners - they are more far from cenrer
  CheckMV<pixel_t>(workarea, mvx - r, mvy - r);
  CheckMV<pixel_t>(workarea, mvx - r, mvy + r);
  CheckMV<pixel_t>(workarea, mvx + r, mvy - r);
  CheckMV<pixel_t>(workarea, mvx + r, mvy + r);
}



/* (x-1)%6 */
static const int mod6m1[8] = { 5,0,1,2,3,4,5,0 };
/* radius 2 hexagon. repeated entries are to avoid having to compute mod6 every time. */
static const int hex2[8][2] = { {-1,-2}, {-2,0}, {-1,2}, {1,2}, {2,0}, {1,-2}, {-1,-2}, {-2,0} };

template<typename pixel_t>
void PlaneOfBlocks::Hex2Search(WorkingArea &workarea, int i_me_range)
{
  // adopted from x264
  int dir = -2;
  int bmx = workarea.bestMV.x;
  int bmy = workarea.bestMV.y;

  if (i_me_range > 1)
  {
    /* hexagon */
//		COST_MV_X3_DIR( -2,0, -1, 2,  1, 2, costs   );
//		COST_MV_X3_DIR(  2,0,  1,-2, -1,-2, costs+3 );
//		COPY2_IF_LT( bcost, costs[0], dir, 0 );
//		COPY2_IF_LT( bcost, costs[1], dir, 1 );
//		COPY2_IF_LT( bcost, costs[2], dir, 2 );
//		COPY2_IF_LT( bcost, costs[3], dir, 3 );
//		COPY2_IF_LT( bcost, costs[4], dir, 4 );
//		COPY2_IF_LT( bcost, costs[5], dir, 5 );
    CheckMVdir<pixel_t>(workarea, bmx - 2, bmy, &dir, 0);
    CheckMVdir<pixel_t>(workarea, bmx - 1, bmy + 2, &dir, 1);
    CheckMVdir<pixel_t>(workarea, bmx + 1, bmy + 2, &dir, 2);
    CheckMVdir<pixel_t>(workarea, bmx + 2, bmy, &dir, 3);
    CheckMVdir<pixel_t>(workarea, bmx + 1, bmy - 2, &dir, 4);
    CheckMVdir<pixel_t>(workarea, bmx - 1, bmy - 2, &dir, 5);


    if (dir != -2)
    {
      bmx += hex2[dir + 1][0];
      bmy += hex2[dir + 1][1];
      /* half hexagon, not overlapping the previous iteration */
      for (int i = 1; i < i_me_range / 2 && workarea.IsVectorOK(bmx, bmy); i++)
      {
        const int odir = mod6m1[dir + 1];
        //				COST_MV_X3_DIR (hex2[odir+0][0], hex2[odir+0][1],
        //				                hex2[odir+1][0], hex2[odir+1][1],
        //				                hex2[odir+2][0], hex2[odir+2][1],
        //				                costs);

        dir = -2;
        //				COPY2_IF_LT( bcost, costs[0], dir, odir-1 );
        //				COPY2_IF_LT( bcost, costs[1], dir, odir   );
        //				COPY2_IF_LT( bcost, costs[2], dir, odir+1 );

        CheckMVdir<pixel_t>(workarea, bmx + hex2[odir + 0][0], bmy + hex2[odir + 0][1], &dir, odir - 1);
        CheckMVdir<pixel_t>(workarea, bmx + hex2[odir + 1][0], bmy + hex2[odir + 1][1], &dir, odir);
        CheckMVdir<pixel_t>(workarea, bmx + hex2[odir + 2][0], bmy + hex2[odir + 2][1], &dir, odir + 1);
        if (dir == -2)
        {
          break;
        }
        bmx += hex2[dir + 1][0];
        bmy += hex2[dir + 1][1];
      }
    }

    workarea.bestMV.x = bmx;
    workarea.bestMV.y = bmy;
  }

  // square refine
//	omx = bmx; omy = bmy;
//	COST_MV_X4(  0,-1,  0,1, -1,0, 1,0 );
//	COST_MV_X4( -1,-1, -1,1, 1,-1, 1,1 );
  ExpandingSearch<pixel_t>(workarea, 1, 1, bmx, bmy);
}


template<typename pixel_t>
void PlaneOfBlocks::CrossSearch(WorkingArea &workarea, int start, int x_max, int y_max, int mvx, int mvy)
{
  // part of umh  search
  for (int i = start; i < x_max; i += 2)
  {
    CheckMV<pixel_t>(workarea, mvx - i, mvy);
    CheckMV<pixel_t>(workarea, mvx + i, mvy);
  }

  for (int j = start; j < y_max; j += 2)
  {
    CheckMV<pixel_t>(workarea, mvx, mvy + j);
    CheckMV<pixel_t>(workarea, mvx, mvy + j);
  }
}

#if 0 // x265
// this part comes as a sample from the x265 project, to find any similarities
// and study if star search method can be applied.
int MotionEstimate::motionEstimate(ReferencePlanes *ref,
const MV &       mvmin,
const MV &       mvmax,
const MV &       qmvp,
int              numCandidates,
const MV *       mvc,
int              merange,
MV &             outQMv,
uint32_t         maxSlices,
pixel *          srcReferencePlane)
{
ALIGN_VAR_16(int, costs[16]);
if (ctuAddr >= 0)
blockOffset = ref->reconPic->getLumaAddr(ctuAddr, absPartIdx) - ref->reconPic->getLumaAddr(0);
intptr_t stride = ref->lumaStride;
pixel* fenc = fencPUYuv.m_buf[0];
pixel* fref = srcReferencePlane == 0 ? ref->fpelPlane[0] + blockOffset : srcReferencePlane + blockOffset;

setMVP(qmvp);

MV qmvmin = mvmin.toQPel();
MV qmvmax = mvmax.toQPel();

/* The term cost used here means satd/sad values for that particular search.
* The costs used in ME integer search only includes the SAD cost of motion
* residual and sqrtLambda times MVD bits.  The subpel refine steps use SATD
* cost of residual and sqrtLambda * MVD bits.  Mode decision will be based
* on video distortion cost (SSE/PSNR) plus lambda times all signaling bits
* (mode + MVD bits). */

// measure SAD cost at clipped QPEL MVP
MV pmv = qmvp.clipped(qmvmin, qmvmax);
MV bestpre = pmv;
int bprecost;

if (ref->isLowres)
bprecost = ref->lowresQPelCost(fenc, blockOffset, pmv, sad);
else
bprecost = subpelCompare(ref, pmv, sad);

/* re-measure full pel rounded MVP with SAD as search start point */
MV bmv = pmv.roundToFPel();
int bcost = bprecost;
if (pmv.isSubpel())
bcost = sad(fenc, FENC_STRIDE, fref + bmv.x + bmv.y * stride, stride) + mvcost(bmv << 2);

// measure SAD cost at MV(0) if MVP is not zero
if (pmv.notZero())
{
  int cost = sad(fenc, FENC_STRIDE, fref, stride) + mvcost(MV(0, 0));
  if (cost < bcost)
  {
    bcost = cost;
    bmv = 0;
    bmv.y = X265_MAX(X265_MIN(0, mvmax.y), mvmin.y);
  }
}

X265_CHECK(!(ref->isLowres && numCandidates), "lowres motion candidates not allowed\n")
// measure SAD cost at each QPEL motion vector candidate
for (int i = 0; i < numCandidates; i++)
{
  MV m = mvc[i].clipped(qmvmin, qmvmax);
  if (m.notZero() & (m != pmv ? 1 : 0) & (m != bestpre ? 1 : 0)) // check already measured
  {
    int cost = subpelCompare(ref, m, sad) + mvcost(m);
    if (cost < bprecost)
    {
      bprecost = cost;
      bestpre = m;
    }
  }
}

pmv = pmv.roundToFPel();
MV omv = bmv;  // current search origin or starting point

switch (searchMethod)
{
case X265_DIA_SEARCH:
{
  /* diamond search, radius 1 */
  bcost <<= 4;
  int i = merange;
  do
  {
    COST_MV_X4_DIR(0, -1, 0, 1, -1, 0, 1, 0, costs);
    if ((bmv.y - 1 >= mvmin.y) & (bmv.y - 1 <= mvmax.y))
      COPY1_IF_LT(bcost, (costs[0] << 4) + 1);
    if ((bmv.y + 1 >= mvmin.y) & (bmv.y + 1 <= mvmax.y))
      COPY1_IF_LT(bcost, (costs[1] << 4) + 3);
    COPY1_IF_LT(bcost, (costs[2] << 4) + 4);
    COPY1_IF_LT(bcost, (costs[3] << 4) + 12);
    if (!(bcost & 15))
      break;
    bmv.x -= (bcost << 28) >> 30;
    bmv.y -= (bcost << 30) >> 30;
    bcost &= ~15;
  } while (--i && bmv.checkRange(mvmin, mvmax));
  bcost >>= 4;
  break;
}

case X265_HEX_SEARCH:
{
me_hex2:
  /* hexagon search, radius 2 */
#if 0
  for (int i = 0; i < merange / 2; i++)
  {
    omv = bmv;
    COST_MV(omv.x - 2, omv.y);
    COST_MV(omv.x - 1, omv.y + 2);
    COST_MV(omv.x + 1, omv.y + 2);
    COST_MV(omv.x + 2, omv.y);
    COST_MV(omv.x + 1, omv.y - 2);
    COST_MV(omv.x - 1, omv.y - 2);
    if (omv == bmv)
      break;
    if (!bmv.checkRange(mvmin, mvmax))
      break;
  }

#else // if 0
  /* equivalent to the above, but eliminates duplicate candidates */
  COST_MV_X3_DIR(-2, 0, -1, 2, 1, 2, costs);
  bcost <<= 3;
  if ((bmv.y >= mvmin.y) & (bmv.y <= mvmax.y))
    COPY1_IF_LT(bcost, (costs[0] << 3) + 2);
  if ((bmv.y + 2 >= mvmin.y) & (bmv.y + 2 <= mvmax.y))
  {
    COPY1_IF_LT(bcost, (costs[1] << 3) + 3);
    COPY1_IF_LT(bcost, (costs[2] << 3) + 4);
  }

  COST_MV_X3_DIR(2, 0, 1, -2, -1, -2, costs);
  if ((bmv.y >= mvmin.y) & (bmv.y <= mvmax.y))
    COPY1_IF_LT(bcost, (costs[0] << 3) + 5);
  if ((bmv.y - 2 >= mvmin.y) & (bmv.y - 2 <= mvmax.y))
  {
    COPY1_IF_LT(bcost, (costs[1] << 3) + 6);
    COPY1_IF_LT(bcost, (costs[2] << 3) + 7);
  }

  if (bcost & 7)
  {
    int dir = (bcost & 7) - 2;

    if ((bmv.y + hex2[dir + 1].y >= mvmin.y) & (bmv.y + hex2[dir + 1].y <= mvmax.y))
    {
      bmv += hex2[dir + 1];

      /* half hexagon, not overlapping the previous iteration */
      for (int i = (merange >> 1) - 1; i > 0 && bmv.checkRange(mvmin, mvmax); i--)
      {
        COST_MV_X3_DIR(hex2[dir + 0].x, hex2[dir + 0].y,
          hex2[dir + 1].x, hex2[dir + 1].y,
          hex2[dir + 2].x, hex2[dir + 2].y,
          costs);
        bcost &= ~7;

        if ((bmv.y + hex2[dir + 0].y >= mvmin.y) & (bmv.y + hex2[dir + 0].y <= mvmax.y))
          COPY1_IF_LT(bcost, (costs[0] << 3) + 1);

        if ((bmv.y + hex2[dir + 1].y >= mvmin.y) & (bmv.y + hex2[dir + 1].y <= mvmax.y))
          COPY1_IF_LT(bcost, (costs[1] << 3) + 2);

        if ((bmv.y + hex2[dir + 2].y >= mvmin.y) & (bmv.y + hex2[dir + 2].y <= mvmax.y))
          COPY1_IF_LT(bcost, (costs[2] << 3) + 3);

        if (!(bcost & 7))
          break;

        dir += (bcost & 7) - 2;
        dir = mod6m1[dir + 1];
        bmv += hex2[dir + 1];
      }
    } // if ((bmv.y + hex2[dir + 1].y >= mvmin.y) & (bmv.y + hex2[dir + 1].y <= mvmax.y))
  }
  bcost >>= 3;
#endif // if 0

  /* square refine */
  int dir = 0;
  COST_MV_X4_DIR(0, -1, 0, 1, -1, 0, 1, 0, costs);
  if ((bmv.y - 1 >= mvmin.y) & (bmv.y - 1 <= mvmax.y))
    COPY2_IF_LT(bcost, costs[0], dir, 1);
  if ((bmv.y + 1 >= mvmin.y) & (bmv.y + 1 <= mvmax.y))
    COPY2_IF_LT(bcost, costs[1], dir, 2);
  COPY2_IF_LT(bcost, costs[2], dir, 3);
  COPY2_IF_LT(bcost, costs[3], dir, 4);
  COST_MV_X4_DIR(-1, -1, -1, 1, 1, -1, 1, 1, costs);
  if ((bmv.y - 1 >= mvmin.y) & (bmv.y - 1 <= mvmax.y))
    COPY2_IF_LT(bcost, costs[0], dir, 5);
  if ((bmv.y + 1 >= mvmin.y) & (bmv.y + 1 <= mvmax.y))
    COPY2_IF_LT(bcost, costs[1], dir, 6);
  if ((bmv.y - 1 >= mvmin.y) & (bmv.y - 1 <= mvmax.y))
    COPY2_IF_LT(bcost, costs[2], dir, 7);
  if ((bmv.y + 1 >= mvmin.y) & (bmv.y + 1 <= mvmax.y))
    COPY2_IF_LT(bcost, costs[3], dir, 8);
  bmv += square1[dir];
  break;
}

case X265_UMH_SEARCH:
{
  int ucost1, ucost2;
  int16_t cross_start = 1;

  /* refine predictors */
  omv = bmv;
  ucost1 = bcost;
  X265_CHECK(((pmv.y >= mvmin.y) & (pmv.y <= mvmax.y)), "pmv outside of search range!");
  DIA1_ITER(pmv.x, pmv.y);
  if (pmv.notZero())
    DIA1_ITER(0, 0);

  ucost2 = bcost;
  if (bmv.notZero() && bmv != pmv)
    DIA1_ITER(bmv.x, bmv.y);
  if (bcost == ucost2)
    cross_start = 3;

  /* Early Termination */
  omv = bmv;
  if (bcost == ucost2 && SAD_THRESH(2000))
  {
    COST_MV_X4(0, -2, -1, -1, 1, -1, -2, 0);
    COST_MV_X4(2, 0, -1, 1, 1, 1, 0, 2);
    if (bcost == ucost1 && SAD_THRESH(500))
      break;
    if (bcost == ucost2)
    {
      int16_t range = (int16_t)(merange >> 1) | 1;
      CROSS(3, range, range);
      COST_MV_X4(-1, -2, 1, -2, -2, -1, 2, -1);
      COST_MV_X4(-2, 1, 2, 1, -1, 2, 1, 2);
      if (bcost == ucost2)
        break;
      cross_start = range + 2;
    }
  }

  // TODO: Need to study x264's logic for building mvc list to understand why they
  //       have special cases here for 16x16, and whether they apply to HEVC CTU

  // adaptive search range based on mvc variability
  if (numCandidates)
  {
    /* range multipliers based on casual inspection of some statistics of
    * average distance between current predictor and final mv found by ESA.
    * these have not been tuned much by actual encoding. */
    static const uint8_t range_mul[4][4] =
    {
      { 3, 3, 4, 4 },
      { 3, 4, 4, 4 },
      { 4, 4, 4, 5 },
      { 4, 4, 5, 6 },
    };

    int mvd;
    int sad_ctx, mvd_ctx;
    int denom = 1;

    if (numCandidates == 1)
    {
      if (LUMA_64x64 == partEnum)
        /* mvc is probably the same as mvp, so the difference isn't meaningful.
        * but prediction usually isn't too bad, so just use medium range */
        mvd = 25;
      else
        mvd = abs(qmvp.x - mvc[0].x) + abs(qmvp.y - mvc[0].y);
    }
    else
    {
      /* calculate the degree of agreement between predictors. */

      /* in 64x64, mvc includes all the neighbors used to make mvp,
      * so don't count mvp separately. */

      denom = numCandidates - 1;
      mvd = 0;
      if (partEnum != LUMA_64x64)
      {
        mvd = abs(qmvp.x - mvc[0].x) + abs(qmvp.y - mvc[0].y);
        denom++;
      }
      mvd += predictorDifference(mvc, numCandidates);
    }

    sad_ctx = SAD_THRESH(1000) ? 0
      : SAD_THRESH(2000) ? 1
      : SAD_THRESH(4000) ? 2 : 3;
    mvd_ctx = mvd < 10 * denom ? 0
      : mvd < 20 * denom ? 1
      : mvd < 40 * denom ? 2 : 3;

    merange = (merange * range_mul[mvd_ctx][sad_ctx]) >> 2;
  }

  /* FIXME if the above DIA2/OCT2/CROSS found a new mv, it has not updated omx/omy.
  * we are still centered on the same place as the DIA2. is this desirable? */
  CROSS(cross_start, merange, merange >> 1);
  COST_MV_X4(-2, -2, -2, 2, 2, -2, 2, 2);

  /* hexagon grid */
  omv = bmv;
  const uint16_t *p_cost_omvx = m_cost_mvx + omv.x * 4;
  const uint16_t *p_cost_omvy = m_cost_mvy + omv.y * 4;
  uint16_t i = 1;
  do
  {
    if (4 * i > X265_MIN4(mvmax.x - omv.x, omv.x - mvmin.x,
      mvmax.y - omv.y, omv.y - mvmin.y))
    {
      for (int j = 0; j < 16; j++)
      {
        MV mv = omv + (hex4[j] * i);
        if (mv.checkRange(mvmin, mvmax))
          COST_MV(mv.x, mv.y);
      }
    }
    else
    {
      int16_t dir = 0;
      pixel *fref_base = fref + omv.x + (omv.y - 4 * i) * stride;
      size_t dy = (size_t)i * stride;
#define SADS(k, x0, y0, x1, y1, x2, y2, x3, y3) \
    sad_x4(fenc, \
           fref_base x0 * i + (y0 - 2 * k + 4) * dy, \
           fref_base x1 * i + (y1 - 2 * k + 4) * dy, \
           fref_base x2 * i + (y2 - 2 * k + 4) * dy, \
           fref_base x3 * i + (y3 - 2 * k + 4) * dy, \
           stride, costs + 4 * k); \
    fref_base += 2 * dy;
#define ADD_MVCOST(k, x, y) costs[k] += p_cost_omvx[x * 4 * i] + p_cost_omvy[y * 4 * i]
#define MIN_MV(k, dx, dy)     if ((omv.y + (dy) >= mvmin.y) & (omv.y + (dy) <= mvmax.y)) { COPY2_IF_LT(bcost, costs[k], dir, dx * 16 + (dy & 15)) }

      SADS(0, +0, -4, +0, +4, -2, -3, +2, -3);
      SADS(1, -4, -2, +4, -2, -4, -1, +4, -1);
      SADS(2, -4, +0, +4, +0, -4, +1, +4, +1);
      SADS(3, -4, +2, +4, +2, -2, +3, +2, +3);
      ADD_MVCOST(0, 0, -4);
      ADD_MVCOST(1, 0, 4);
      ADD_MVCOST(2, -2, -3);
      ADD_MVCOST(3, 2, -3);
      ADD_MVCOST(4, -4, -2);
      ADD_MVCOST(5, 4, -2);
      ADD_MVCOST(6, -4, -1);
      ADD_MVCOST(7, 4, -1);
      ADD_MVCOST(8, -4, 0);
      ADD_MVCOST(9, 4, 0);
      ADD_MVCOST(10, -4, 1);
      ADD_MVCOST(11, 4, 1);
      ADD_MVCOST(12, -4, 2);
      ADD_MVCOST(13, 4, 2);
      ADD_MVCOST(14, -2, 3);
      ADD_MVCOST(15, 2, 3);
      MIN_MV(0, 0, -4);
      MIN_MV(1, 0, 4);
      MIN_MV(2, -2, -3);
      MIN_MV(3, 2, -3);
      MIN_MV(4, -4, -2);
      MIN_MV(5, 4, -2);
      MIN_MV(6, -4, -1);
      MIN_MV(7, 4, -1);
      MIN_MV(8, -4, 0);
      MIN_MV(9, 4, 0);
      MIN_MV(10, -4, 1);
      MIN_MV(11, 4, 1);
      MIN_MV(12, -4, 2);
      MIN_MV(13, 4, 2);
      MIN_MV(14, -2, 3);
      MIN_MV(15, 2, 3);
#undef SADS
#undef ADD_MVCOST
#undef MIN_MV
      if (dir)
      {
        bmv.x = omv.x + i * (dir >> 4);
        bmv.y = omv.y + i * ((dir << 28) >> 28);
      }
    }
  } while (++i <= merange >> 2);
  if (bmv.checkRange(mvmin, mvmax))
    goto me_hex2;
  break;
}

case X265_STAR_SEARCH: // Adapted from HM ME
{
  int bPointNr = 0;
  int bDistance = 0;

  const int EarlyExitIters = 3;
  StarPatternSearch(ref, mvmin, mvmax, bmv, bcost, bPointNr, bDistance, EarlyExitIters, merange);
  if (bDistance == 1)
  {
    // if best distance was only 1, check two missing points.  If no new point is found, stop
    if (bPointNr)
    {
      /* For a given direction 1 to 8, check nearest two outer X pixels
      X   X
      X 1 2 3 X
      4 * 5
      X 6 7 8 X
      X   X
      */
      int saved = bcost;
      const MV mv1 = bmv + offsets[(bPointNr - 1) * 2];
      const MV mv2 = bmv + offsets[(bPointNr - 1) * 2 + 1];
      if (mv1.checkRange(mvmin, mvmax))
      {
        COST_MV(mv1.x, mv1.y);
      }
      if (mv2.checkRange(mvmin, mvmax))
      {
        COST_MV(mv2.x, mv2.y);
      }
      if (bcost == saved)
        break;
    }
    else
      break;
  }

  const int RasterDistance = 5;
  if (bDistance > RasterDistance)
  {
    // raster search refinement if original search distance was too big
    MV tmv;
    for (tmv.y = mvmin.y; tmv.y <= mvmax.y; tmv.y += RasterDistance)
    {
      for (tmv.x = mvmin.x; tmv.x <= mvmax.x; tmv.x += RasterDistance)
      {
        if (tmv.x + (RasterDistance * 3) <= mvmax.x)
        {
          pixel *pix_base = fref + tmv.y * stride + tmv.x;
          sad_x4(fenc,
            pix_base,
            pix_base + RasterDistance,
            pix_base + RasterDistance * 2,
            pix_base + RasterDistance * 3,
            stride, costs);
          costs[0] += mvcost(tmv << 2);
          COPY2_IF_LT(bcost, costs[0], bmv, tmv);
          tmv.x += RasterDistance;
          costs[1] += mvcost(tmv << 2);
          COPY2_IF_LT(bcost, costs[1], bmv, tmv);
          tmv.x += RasterDistance;
          costs[2] += mvcost(tmv << 2);
          COPY2_IF_LT(bcost, costs[2], bmv, tmv);
          tmv.x += RasterDistance;
          costs[3] += mvcost(tmv << 3);
          COPY2_IF_LT(bcost, costs[3], bmv, tmv);
        }
        else
          COST_MV(tmv.x, tmv.y);
      }
    }
  }

  while (bDistance > 0)
  {
    // center a new search around current best
    bDistance = 0;
    bPointNr = 0;
    const int MaxIters = 32;
    StarPatternSearch(ref, mvmin, mvmax, bmv, bcost, bPointNr, bDistance, MaxIters, merange);

    if (bDistance == 1)
    {
      if (!bPointNr)
        break;

      /* For a given direction 1 to 8, check nearest 2 outer X pixels
      X   X
      X 1 2 3 X
      4 * 5
      X 6 7 8 X
      X   X
      */
      const MV mv1 = bmv + offsets[(bPointNr - 1) * 2];
      const MV mv2 = bmv + offsets[(bPointNr - 1) * 2 + 1];
      if (mv1.checkRange(mvmin, mvmax))
      {
        COST_MV(mv1.x, mv1.y);
      }
      if (mv2.checkRange(mvmin, mvmax))
      {
        COST_MV(mv2.x, mv2.y);
      }
      break;
    }
  }

  break;
}

case X265_SEA:
{
  // Successive Elimination Algorithm
  const int16_t minX = X265_MAX(omv.x - (int16_t)merange, mvmin.x);
  const int16_t minY = X265_MAX(omv.y - (int16_t)merange, mvmin.y);
  const int16_t maxX = X265_MIN(omv.x + (int16_t)merange, mvmax.x);
  const int16_t maxY = X265_MIN(omv.y + (int16_t)merange, mvmax.y);
  const uint16_t *p_cost_mvx = m_cost_mvx - qmvp.x;
  const uint16_t *p_cost_mvy = m_cost_mvy - qmvp.y;
  int16_t* meScratchBuffer = NULL;
  int scratchSize = merange * 2 + 4;
  if (scratchSize)
  {
    meScratchBuffer = X265_MALLOC(int16_t, scratchSize);
    memset(meScratchBuffer, 0, sizeof(int16_t)* scratchSize);
  }

  /* SEA is fastest in multiples of 4 */
  int meRangeWidth = (maxX - minX + 3) & ~3;
  int w = 0, h = 0;                    // Width and height of the PU
  ALIGN_VAR_32(pixel, zero[64 * FENC_STRIDE]) = { 0 };
  ALIGN_VAR_32(int, encDC[4]);
  uint16_t *fpelCostMvX = m_fpelMvCosts[-qmvp.x & 3] + (-qmvp.x >> 2);
  sizesFromPartition(partEnum, &w, &h);
  int deltaX = (w <= 8) ? (w) : (w >> 1);
  int deltaY = (h <= 8) ? (h) : (h >> 1);

  /* Check if very small rectangular blocks which cannot be sub-divided anymore */
  bool smallRectPartition = partEnum == LUMA_4x4 || partEnum == LUMA_16x12 ||
    partEnum == LUMA_12x16 || partEnum == LUMA_16x4 || partEnum == LUMA_4x16;
  /* Check if vertical partition */
  bool verticalRect = partEnum == LUMA_32x64 || partEnum == LUMA_16x32 || partEnum == LUMA_8x16 ||
    partEnum == LUMA_4x8;
  /* Check if horizontal partition */
  bool horizontalRect = partEnum == LUMA_64x32 || partEnum == LUMA_32x16 || partEnum == LUMA_16x8 ||
    partEnum == LUMA_8x4;
  /* Check if assymetric vertical partition */
  bool assymetricVertical = partEnum == LUMA_12x16 || partEnum == LUMA_4x16 || partEnum == LUMA_24x32 ||
    partEnum == LUMA_8x32 || partEnum == LUMA_48x64 || partEnum == LUMA_16x64;
  /* Check if assymetric horizontal partition */
  bool assymetricHorizontal = partEnum == LUMA_16x12 || partEnum == LUMA_16x4 || partEnum == LUMA_32x24 ||
    partEnum == LUMA_32x8 || partEnum == LUMA_64x48 || partEnum == LUMA_64x16;

  int tempPartEnum = 0;

  /* If a vertical rectangular partition, it is horizontally split into two, for ads_x2() */
  if (verticalRect)
    tempPartEnum = partitionFromSizes(w, h >> 1);
  /* If a horizontal rectangular partition, it is vertically split into two, for ads_x2() */
  else if (horizontalRect)
    tempPartEnum = partitionFromSizes(w >> 1, h);
  /* We have integral planes introduced to account for assymetric partitions.
  * Hence all assymetric partitions except those which cannot be split into legal sizes,
  * are split into four for ads_x4() */
  else if (assymetricVertical || assymetricHorizontal)
    tempPartEnum = smallRectPartition ? partEnum : partitionFromSizes(w >> 1, h >> 1);
  /* General case: Square partitions. All partitions with width > 8 are split into four
  * for ads_x4(), for 4x4 and 8x8 we do ads_x1() */
  else
    tempPartEnum = (w <= 8) ? partEnum : partitionFromSizes(w >> 1, h >> 1);

  /* Successive elimination by comparing DC before a full SAD,
  * because sum(abs(diff)) >= abs(diff(sum)). */
  primitives.pu[tempPartEnum].sad_x4(zero,
    fenc,
    fenc + deltaX,
    fenc + deltaY * FENC_STRIDE,
    fenc + deltaX + deltaY * FENC_STRIDE,
    FENC_STRIDE,
    encDC);

  /* Assigning appropriate integral plane */
  uint32_t *sumsBase = NULL;
  switch (deltaX)
  {
  case 32: if (deltaY % 24 == 0)
    sumsBase = integral[1];
           else if (deltaY == 8)
             sumsBase = integral[2];
           else
             sumsBase = integral[0];
    break;
  case 24: sumsBase = integral[3];
    break;
  case 16: if (deltaY % 12 == 0)
    sumsBase = integral[5];
           else if (deltaY == 4)
             sumsBase = integral[6];
           else
             sumsBase = integral[4];
    break;
  case 12: sumsBase = integral[7];
    break;
  case 8: if (deltaY == 32)
    sumsBase = integral[8];
          else
            sumsBase = integral[9];
    break;
  case 4: if (deltaY == 16)
    sumsBase = integral[10];
          else
            sumsBase = integral[11];
    break;
  default: sumsBase = integral[11];
    break;
  }

  if (partEnum == LUMA_64x64 || partEnum == LUMA_32x32 || partEnum == LUMA_16x16 ||
    partEnum == LUMA_32x64 || partEnum == LUMA_16x32 || partEnum == LUMA_8x16 ||
    partEnum == LUMA_4x8 || partEnum == LUMA_12x16 || partEnum == LUMA_4x16 ||
    partEnum == LUMA_24x32 || partEnum == LUMA_8x32 || partEnum == LUMA_48x64 ||
    partEnum == LUMA_16x64)
    deltaY *= (int)stride;

  if (verticalRect)
    encDC[1] = encDC[2];

  if (horizontalRect)
    deltaY = deltaX;

  /* ADS and SAD */
  MV tmv;
  for (tmv.y = minY; tmv.y <= maxY; tmv.y++)
  {
    int i, xn;
    int ycost = p_cost_mvy[tmv.y] << 2;
    if (bcost <= ycost)
      continue;
    bcost -= ycost;

    /* ADS_4 for 16x16, 32x32, 64x64, 24x32, 32x24, 48x64, 64x48, 32x8, 8x32, 64x16, 16x64 partitions
    * ADS_1 for 4x4, 8x8, 16x4, 4x16, 16x12, 12x16 partitions
    * ADS_2 for all other rectangular partitions */
    xn = ads(encDC,
      sumsBase + minX + tmv.y * stride,
      deltaY,
      fpelCostMvX + minX,
      meScratchBuffer,
      meRangeWidth,
      bcost);

    for (i = 0; i < xn - 2; i += 3)
      COST_MV_X3_ABS(minX + meScratchBuffer[i], tmv.y,
        minX + meScratchBuffer[i + 1], tmv.y,
        minX + meScratchBuffer[i + 2], tmv.y);

    bcost += ycost;
    for (; i < xn; i++)
      COST_MV(minX + meScratchBuffer[i], tmv.y);
  }
  if (meScratchBuffer)
    x265_free(meScratchBuffer);
  break;
}

case X265_FULL_SEARCH:
{
  // dead slow exhaustive search, but at least it uses sad_x4()
  MV tmv;
  for (tmv.y = mvmin.y; tmv.y <= mvmax.y; tmv.y++)
  {
    for (tmv.x = mvmin.x; tmv.x <= mvmax.x; tmv.x++)
    {
      if (tmv.x + 3 <= mvmax.x)
      {
        pixel *pix_base = fref + tmv.y * stride + tmv.x;
        sad_x4(fenc,
          pix_base,
          pix_base + 1,
          pix_base + 2,
          pix_base + 3,
          stride, costs);
        costs[0] += mvcost(tmv << 2);
        COPY2_IF_LT(bcost, costs[0], bmv, tmv);
        tmv.x++;
        costs[1] += mvcost(tmv << 2);
        COPY2_IF_LT(bcost, costs[1], bmv, tmv);
        tmv.x++;
        costs[2] += mvcost(tmv << 2);
        COPY2_IF_LT(bcost, costs[2], bmv, tmv);
        tmv.x++;
        costs[3] += mvcost(tmv << 2);
        COPY2_IF_LT(bcost, costs[3], bmv, tmv);
      }
      else
        COST_MV(tmv.x, tmv.y);
    }
  }

  break;
}

default:
  X265_CHECK(0, "invalid motion estimate mode\n");
  break;
}

if (bprecost < bcost)
{
  bmv = bestpre;
  bcost = bprecost;
}
else
bmv = bmv.toQPel(); // promote search bmv to qpel

const SubpelWorkload& wl = workload[this->subpelRefine];

// check mv range for slice bound
if ((maxSlices > 1) & ((bmv.y < qmvmin.y) | (bmv.y > qmvmax.y)))
{
  bmv.y = x265_min(x265_max(bmv.y, qmvmin.y), qmvmax.y);
  bcost = subpelCompare(ref, bmv, satd) + mvcost(bmv);
}

if (!bcost)
{
  /* if there was zero residual at the clipped MVP, we can skip subpel
  * refine, but we do need to include the mvcost in the returned cost */
  bcost = mvcost(bmv);
}
else if (ref->isLowres)
{
  int bdir = 0;
  for (int i = 1; i <= wl.hpel_dirs; i++)
  {
    MV qmv = bmv + square1[i] * 2;

    /* skip invalid range */
    if ((qmv.y < qmvmin.y) | (qmv.y > qmvmax.y))
      continue;

    int cost = ref->lowresQPelCost(fenc, blockOffset, qmv, sad) + mvcost(qmv);
    COPY2_IF_LT(bcost, cost, bdir, i);
  }

  bmv += square1[bdir] * 2;
  bcost = ref->lowresQPelCost(fenc, blockOffset, bmv, satd) + mvcost(bmv);

  bdir = 0;
  for (int i = 1; i <= wl.qpel_dirs; i++)
  {
    MV qmv = bmv + square1[i];

    /* skip invalid range */
    if ((qmv.y < qmvmin.y) | (qmv.y > qmvmax.y))
      continue;

    int cost = ref->lowresQPelCost(fenc, blockOffset, qmv, satd) + mvcost(qmv);
    COPY2_IF_LT(bcost, cost, bdir, i);
  }

  bmv += square1[bdir];
}
else
{
  pixelcmp_t hpelcomp;

  if (wl.hpel_satd)
  {
    bcost = subpelCompare(ref, bmv, satd) + mvcost(bmv);
    hpelcomp = satd;
  }
  else
    hpelcomp = sad;

  for (int iter = 0; iter < wl.hpel_iters; iter++)
  {
    int bdir = 0;
    for (int i = 1; i <= wl.hpel_dirs; i++)
    {
      MV qmv = bmv + square1[i] * 2;

      // check mv range for slice bound
      if ((qmv.y < qmvmin.y) | (qmv.y > qmvmax.y))
        continue;

      int cost = subpelCompare(ref, qmv, hpelcomp) + mvcost(qmv);
      COPY2_IF_LT(bcost, cost, bdir, i);
    }

    if (bdir)
      bmv += square1[bdir] * 2;
    else
      break;
  }

  /* if HPEL search used SAD, remeasure with SATD before QPEL */
  if (!wl.hpel_satd)
    bcost = subpelCompare(ref, bmv, satd) + mvcost(bmv);

  for (int iter = 0; iter < wl.qpel_iters; iter++)
  {
    int bdir = 0;
    for (int i = 1; i <= wl.qpel_dirs; i++)
    {
      MV qmv = bmv + square1[i];

      // check mv range for slice bound
      if ((qmv.y < qmvmin.y) | (qmv.y > qmvmax.y))
        continue;

      int cost = subpelCompare(ref, qmv, satd) + mvcost(qmv);
      COPY2_IF_LT(bcost, cost, bdir, i);
    }

    if (bdir)
      bmv += square1[bdir];
    else
      break;
  }
}

// check mv range for slice bound
X265_CHECK(((bmv.y >= qmvmin.y) & (bmv.y <= qmvmax.y)), "mv beyond range!");

x265_emms();
outQMv = bmv;
return bcost;
}
#endif // 0 x265

template<typename pixel_t>
void PlaneOfBlocks::UMHSearch(WorkingArea &workarea, int i_me_range, int omx, int omy) // radius
{
  // Uneven-cross Multi-Hexagon-grid Search (see x264)
  /* hexagon grid */

//	int omx = workarea.bestMV.x;
//	int omy = workarea.bestMV.y;
  // my mod: do not shift the center after Cross
  CrossSearch<pixel_t>(workarea, 1, i_me_range, i_me_range, omx, omy);

  int i = 1;
  do
  {
  /*   -4 -2  0  2  4
 -4           x
 -3        x     x
 -2     x           x
 -1     x           x
  0     x           x
  1     x           x
  2     x           x
  3        x     x
  4           x
  */
    static const int hex4[16][2] =
    {
      {-4, 2}, {-4, 1}, {-4, 0}, {-4,-1}, {-4,-2},
      { 4,-2}, { 4,-1}, { 4, 0}, { 4, 1}, { 4, 2},
      { 2, 3}, { 0, 4}, {-2, 3},
      {-2,-3}, { 0,-4}, { 2,-3},
    };

    for (int j = 0; j < 16; j++)
    {
      int mx = omx + hex4[j][0] * i;
      int my = omy + hex4[j][1] * i;
      CheckMV<pixel_t>(workarea, mx, my);
    }
  } while (++i <= i_me_range / 4);

  //	if( bmy <= mv_y_max )
  // {
  //		goto me_hex2;
  //	}

  Hex2Search<pixel_t>(workarea, i_me_range);
}



//----------------------------------------------------------------



// estimate global motion from current plane vectors data for using on next plane - added by Fizick
// on input globalMVec is prev estimation
// on output globalMVec is doubled for next scale plane using
//
// use very simple but robust method
// more advanced method (like MVDepan) can be implemented later
void PlaneOfBlocks::EstimateGlobalMVDoubled(VECTOR *globalMVec, Slicer &slicer)
{
  assert(globalMVec != 0);
  assert(&slicer != 0);

  // compute of x and y part parallelly
  _gvect_result_count = 2;
  // when x and y is ready, they decrease it by 1, if reaches zero, both are ready
  _gvect_estim_ptr = globalMVec;
  // 'height' == 2 but here it means the number of compute tasks
  // Two tasks internally: y=0 for finding maxx, y=1 is for finding maxy
  slicer.start(2, *this, &PlaneOfBlocks::estimate_global_mv_doubled_slice);
}



void	PlaneOfBlocks::estimate_global_mv_doubled_slice(Slicer::TaskData &td)
{
  bool				both_done_flag = false;
  for (int y = td._y_beg; y < td._y_end; ++y) // 0..0, 1..1 or 0..1 (no mt)
  {
    std::vector <int> &	freq_arr = freqArray[y];

    const int      freqSize = int(freq_arr.size());
    memset(&freq_arr[0], 0, freqSize * sizeof(freq_arr[0])); // reset

    int            indmin = freqSize - 1;
    int            indmax = 0;

    // find most frequent x
    if (y == 0)
    {
      for (int i = 0; i < nBlkCount; i++)
      {
        int ind = (freqSize >> 1) + vectors[i].x;
        if (ind >= 0 && ind < freqSize)
        {
          ++freq_arr[ind];
          if (ind > indmax)
          {
            indmax = ind;
          }
          if (ind < indmin)
          {
            indmin = ind;
          }
        }
      }
    }

    // find most frequent y
    else
    {
      for (int i = 0; i < nBlkCount; i++)
      {
        int ind = (freqSize >> 1) + vectors[i].y;
        if (ind >= 0 && ind < freqSize)
        {
          ++freq_arr[ind];
          if (ind > indmax)
          {
            indmax = ind;
          }
          if (ind < indmin)
          {
            indmin = ind;
          }
        }
      }	// i < nBlkCount
    }

    int count = freq_arr[indmin];
    int index = indmin;
    for (int i = indmin + 1; i <= indmax; i++)
    {
      if (freq_arr[i] > count)
      {
        count = freq_arr[i];
        index = i;
      }
    }

    // most frequent value
    // y is either 0 or 1, their computation can be parallelized
    if(y == 0)
      _gvect_estim_ptr->x = index - (freqSize >> 1);
    else
      _gvect_estim_ptr->y = index - (freqSize >> 1);

    const int new_count = --_gvect_result_count;
    both_done_flag = (new_count == 0);
  }

  if (!both_done_flag)
    return;

  // iteration to increase precision
  if (both_done_flag)
  {
    int medianx = _gvect_estim_ptr->x;
    int mediany = _gvect_estim_ptr->y;
    int meanvx = 0;
    int meanvy = 0;
    int num = 0;
    for (int i = 0; i < nBlkCount; i++)
    {
      if (abs(vectors[i].x - medianx) < 6
        && abs(vectors[i].y - mediany) < 6)
      {
        meanvx += vectors[i].x;
        meanvy += vectors[i].y;
        num += 1;
      }
    }

    // output vectors must be doubled for next (finer) scale level
    if (num > 0)
    {
      _gvect_estim_ptr->x = 2 * meanvx / num;
      _gvect_estim_ptr->y = 2 * meanvy / num;
    }
    else
    {
      _gvect_estim_ptr->x = 2 * medianx;
      _gvect_estim_ptr->y = 2 * mediany;
    }
  }

  //	char debugbuf[100];
  //	sprintf(debugbuf,"MVAnalyse: nx=%d ny=%d next global vx=%d vy=%d", nBlkX, nBlkY, globalMVec->x, globalMVec->y);
  //	OutputDebugString(debugbuf);
}



//----------------------------------------------------------------

template<typename pixel_t>
sad_t PlaneOfBlocks::LumaSADx(WorkingArea &workarea, const unsigned char *pRef0)
{
  sad_t sad;
  sad_t refLuma;
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
  switch (dctmode)
  {
  case 1: // dct SAD
  {
    workarea.DCT->DCTBytes2D(pRef0, nRefPitch[0], &workarea.dctRef[0], dctpitch);
    const pixel_t src_DC = reinterpret_cast<pixel_t *>(&workarea.dctSrc[0])[0];
    const pixel_t ref_DC = reinterpret_cast<pixel_t *>(&workarea.dctRef[0])[0];
/*    sad = ((safe_sad_t)SAD(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch) +
      // correct reduced DC component: *3: because DC component was normalized by an additional 1/4 factor
      abs(src_DC - ref_DC) * 3)
      * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
      / 2;*/
      sad = ((safe_sad_t)DM_Luma->GetDisMetric(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch) +
            // correct reduced DC component: *3: because DC component was normalized by an additional 1/4 factor
            abs(src_DC - ref_DC) * 3)
            * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
            / 2;
    break;
  }
  case 2: //  globally (lumaChange) weighted spatial and DCT
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (dctweight16 > 0)
    {
      workarea.DCT->DCTBytes2D(pRef0, nRefPitch[0], &workarea.dctRef[0], dctpitch);
      const pixel_t src_DC = reinterpret_cast<pixel_t *>(&workarea.dctSrc[0])[0];
      const pixel_t ref_DC = reinterpret_cast<pixel_t *>(&workarea.dctRef[0])[0];
/*      sad_t dctsad = ((safe_sad_t)SAD(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch) +
        // correct reduced DC component: *3: because DC component was normalized by an additional 1/4 factor
        abs(src_DC - ref_DC) * 3)
        * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
        / 2;*/
      sad_t dctsad = ((safe_sad_t)DM_Luma->GetDisMetric(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch) +
          // correct reduced DC component: *3: because DC component was normalized by an additional 1/4 factor
          abs(src_DC - ref_DC) * 3)
          * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
          / 2;

      sad = (sad*(16 - dctweight16) + dctsad*dctweight16) / 16;
    }
    break;
  case 3: // per block adaptive switched from spatial to equal mixed SAD (faster)
    refLuma = LUMA(pRef0, nRefPitch[0]);
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (abs((int)workarea.srcLuma - (int)refLuma) > ((int)workarea.srcLuma + (int)refLuma) >> 5)
    {
      workarea.DCT->DCTBytes2D(pRef0, nRefPitch[0], &workarea.dctRef[0], dctpitch);
/*      sad_t dctsad = (safe_sad_t)SAD(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch)
        * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
        / 2;*/
      sad_t dctsad = (safe_sad_t)DM_Luma->GetDisMetric(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch)
          * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
          / 2;

      sad = sad / 2 + dctsad / 2;
    }
    break;
  case 4: //  per block adaptive switched from spatial to mixed SAD with more weight of DCT (best?)
    refLuma = LUMA(pRef0, nRefPitch[0]);
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (abs((int)workarea.srcLuma - (int)refLuma) > ((int)workarea.srcLuma + (int)refLuma) >> 5)
    {
      workarea.DCT->DCTBytes2D(pRef0, nRefPitch[0], &workarea.dctRef[0], dctpitch);
/*      sad_t dctsad = (safe_sad_t)SAD(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch)
        * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
        / 2;*/
      sad_t dctsad = (safe_sad_t)DM_Luma->GetDisMetric(&workarea.dctSrc[0], dctpitch, &workarea.dctRef[0], dctpitch)
          * nSqrtBlkSize2D // instead of nBlkSizeX, sqrt(nBlkSizeX * nBlkSizeY)
          / 2;

      sad = sad / 4 + dctsad / 2 + dctsad / 4;
    }
    break;
  case 5: // dct SAD (SATD)
    sad = SATD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    // buggy? PF QTGMC(dct=5). 20160816 No! SATD function was linked to Dummy, did nothing. Made live again from 2.7.0.22d
    break;
  case 6: //  globally (lumaChange) weighted spatial and DCT (better estimate)
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (dctweight16 > 0)
    {
      sad_t dctsad = SATD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
      sad = ((safe_sad_t)sad*(16 - dctweight16) + dctsad*dctweight16) / 16;
    }
    break;
  case 7: // per block adaptive switched from spatial to equal mixed SAD (faster?)
    refLuma = LUMA(pRef0, nRefPitch[0]);
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (abs((int)workarea.srcLuma - (int)refLuma) > (workarea.srcLuma + refLuma) >> 5)
    {
      sad_t dctsad = SATD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
      sad = sad / 2 + dctsad / 2;
    }
    break;
  case 8: //  per block adaptive switched from spatial to mixed SAD with more weight of DCT (faster?)
    refLuma = LUMA(pRef0, nRefPitch[0]);
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (abs((int)workarea.srcLuma - (int)refLuma) > (workarea.srcLuma + refLuma) >> 5)
    {
      sad_t dctsad = SATD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
      sad = sad / 4 + dctsad / 2 + dctsad / 4;
    }
    break;
  case 9: //  globally (lumaChange) weighted spatial and DCT (better estimate, only half weight on SATD)
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (dctweight16 > 1)
    {
      int dctweighthalf = dctweight16 / 2;
      sad_t dctsad = SATD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
      sad = ((safe_sad_t)sad*(16 - dctweighthalf) + dctsad*dctweighthalf) / 16;
    }
    break;
  case 10: // per block adaptive switched from spatial to mixed SAD, weighted to SAD (faster)
    refLuma = LUMA(pRef0, nRefPitch[0]);
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    if (abs((int)workarea.srcLuma - (int)refLuma) > ((int)workarea.srcLuma + (int)refLuma) >> 4)
    {
      sad_t dctsad = SATD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
      sad = sad / 2 + dctsad / 4 + sad / 4;
    }
    break;
  default:
//    sad = SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
    sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
  }
  return sad;
}

template<typename pixel_t>
MV_FORCEINLINE sad_t	PlaneOfBlocks::LumaSAD(WorkingArea &workarea, const unsigned char *pRef0)
{
#ifdef MOTION_DEBUG
  workarea.iter++;
#endif
#ifdef ALLOW_DCT
  // made simple SAD more prominent (~1% faster) while keeping DCT support (TSchniede)
//  return !dctmode ? SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]) : LumaSADx<pixel_t>(workarea, pRef0);
  return !dctmode ? DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]) : LumaSADx<pixel_t>(workarea, pRef0);
#else
//  return SAD(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
  return DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
#endif
}


/* check if the vector (vx, vy) is better than the best vector found so far without penalty new - renamed in v.2.11*/
template<typename pixel_t>
MV_FORCEINLINE void	PlaneOfBlocks::CheckMV0(WorkingArea &workarea, int vx, int vy)
{		//here the chance for default values are high especially for zeroMVfieldShifted (on left/top border)
  if (
#ifdef ONLY_CHECK_NONDEFAULT_MV
  ((vx != 0) || (vy != zeroMVfieldShifted.y)) &&
    ((vx != workarea.predictor.x) || (vy != workarea.predictor.y)) &&
    ((vx != workarea.globalMVPredictor.x) || (vy != workarea.globalMVPredictor.y)) &&
#endif
    workarea.IsVectorOK(vx, vy))
  {
#if 0
    sad_t saduv = (chroma) ? ScaleSadChroma(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale) : 0;
    sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    sad += saduv;
    sad_t cost = sad + workarea.MotionDistorsion(vx, vy);
    //		int cost = sad + sad*workarea.MotionDistorsion(vx, vy)/(nBlkSizeX*nBlkSizeY*4);
    //		if (sad>bigSAD) { DebugPrintf("%d %d %d %d %d %d", workarea.blkIdx, vx, vy, workarea.nMinCost, cost, sad);}
    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = vx;
      workarea.bestMV.y = vy;
      workarea.nMinCost = cost;
      workarea.bestMV.sad = sad;
    }
#else
    // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
    sad_t cost=workarea.MotionDistorsion<pixel_t>(vx, vy);
    if(cost>=workarea.nMinCost) return;

    sad_t sad=LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    cost+=sad;
    if(cost>=workarea.nMinCost) return;

/*    sad_t saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
    sad_t saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;

    cost += saduv;
    if(cost>=workarea.nMinCost) return;

    workarea.bestMV.x = vx;
    workarea.bestMV.y = vy;
    workarea.nMinCost = cost;
    workarea.bestMV.sad = sad+saduv;

#endif
  }
}

/* check if the vector (vx, vy) is better than the best vector found so far without penalty new - renamed in v.2.11*/
template<typename pixel_t>
MV_FORCEINLINE void	PlaneOfBlocks::CheckMV0_SO2(WorkingArea& workarea, int vx, int vy, sad_t cost)
{		//here the chance for default values are high especially for zeroMVfieldShifted (on left/top border)
  if (
#ifdef ONLY_CHECK_NONDEFAULT_MV
  ((vx != 0) || (vy != zeroMVfieldShifted.y)) &&
    ((vx != workarea.predictor.x) || (vy != workarea.predictor.y)) &&
    ((vx != workarea.globalMVPredictor.x) || (vy != workarea.globalMVPredictor.y)) &&
#endif
    1)
  {
    // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
//    sad_t cost = workarea.MotionDistorsion<pixel_t>(vx, vy); - no check for motiondistortion - made already in PseudoEPZSeach_SO2()
//    if (cost >= workarea.nMinCost) return;

    sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    cost += sad;
    if (cost >= workarea.nMinCost) return;

    workarea.bestMV.x = vx;
    workarea.bestMV.y = vy;
    workarea.nMinCost = cost;
    workarea.bestMV.sad = sad;

  }
}

/* check if the vector (vx, vy) is better than the best vector found so far */
template<typename pixel_t>
MV_FORCEINLINE void	PlaneOfBlocks::CheckMV(WorkingArea &workarea, int vx, int vy)
{		//here the chance for default values are high especially for zeroMVfieldShifted (on left/top border)
  if (
#ifdef ONLY_CHECK_NONDEFAULT_MV
  ((vx != 0) || (vy != zeroMVfieldShifted.y)) &&
    ((vx != workarea.predictor.x) || (vy != workarea.predictor.y)) &&
    ((vx != workarea.globalMVPredictor.x) || (vy != workarea.globalMVPredictor.y)) &&
#endif
    workarea.IsVectorOK(vx, vy))
  {
#if 0
    sad_t saduv =
      !(chroma) ? 0 :
      ScaleSadChroma(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale);
    sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    sad += saduv;
    sad_t cost = sad + workarea.MotionDistorsion(vx, vy) + ((penaltyNew*(bigsad_t)sad) >> 8); //v2
//		int cost = sad + sad*workarea.MotionDistorsion(vx, vy)/(nBlkSizeX*nBlkSizeY*4);
//		if (sad>bigSAD) { DebugPrintf("%d %d %d %d %d %d", workarea.blkIdx, vx, vy, workarea.nMinCost, cost, sad);}
    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = vx;
      workarea.bestMV.y = vy;
      workarea.nMinCost = cost;
      workarea.bestMV.sad = sad;
    }
#else
    // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
    sad_t cost=workarea.MotionDistorsion<pixel_t>(vx, vy);
    if(cost>=workarea.nMinCost) return;

    typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

    sad_t sad; //=LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    if (iUseSubShift == 0)
    {
      sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    }
    else
    {
      int iRefPitchY = 0;
      const unsigned char* ptrRef = GetRefBlockSubShifted(workarea, vx, vy, iRefPitchY);
//      sad = SAD(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
      sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
    }

    cost += sad + ((penaltyNew*(safe_sad_t)sad) >> 8);
    if(cost>=workarea.nMinCost) return;

    sad_t saduv; /* = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0; */
    if (iUseSubShift == 0)
    {
/*      saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
        + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
        + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;

    }
    else
    {
      if (chroma)
      {
        int iRefPitchU = 0;
        int iRefPitchV = 0;
        const unsigned char* ptrRefU = GetRefBlockUSubShifted(workarea, vx, vy, iRefPitchU);
        const unsigned char* ptrRefV = GetRefBlockVSubShifted(workarea, vx, vy, iRefPitchV);
/*        saduv = ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
          + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);*/
        saduv = ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
          + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);

      }
      else
        saduv = 0;
    }
    cost += saduv + ((penaltyNew*(safe_sad_t)saduv) >> 8);
    if(cost>=workarea.nMinCost) return;

    workarea.bestMV.x = vx;
    workarea.bestMV.y = vy;
    workarea.nMinCost = cost;
    workarea.bestMV.sad = sad+saduv;
#endif
  }
}

/* check if the vector (vx, vy) is better, and update dir accordingly */
template<typename pixel_t>
MV_FORCEINLINE void	PlaneOfBlocks::CheckMV2(WorkingArea &workarea, int vx, int vy, int *dir, int val)
{
  if (
#ifdef ONLY_CHECK_NONDEFAULT_MV
  ((vx != 0) || (vy != zeroMVfieldShifted.y)) &&
    ((vx != workarea.predictor.x) || (vy != workarea.predictor.y)) &&
    ((vx != workarea.globalMVPredictor.x) || (vy != workarea.globalMVPredictor.y)) &&
#endif
    workarea.IsVectorOK(vx, vy))
  {
#if 0
    sad_t saduv =
      !(chroma) ? 0 :
      ScaleSadChroma(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale);
    sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    sad += saduv;
    sad_t cost = sad + workarea.MotionDistorsion(vx, vy) + ((penaltyNew*(bigsad_t)sad) >> 8); // v1.5.8
//		if (sad > LSAD/4) DebugPrintf("%d %d %d %d %d %d %d", workarea.blkIdx, vx, vy, val, workarea.nMinCost, cost, sad);
//		int cost = sad + sad*workarea.MotionDistorsion(vx, vy)/(nBlkSizeX*nBlkSizeY*4) + ((penaltyNew*sad)>>8); // v1.5.8
    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = vx;
      workarea.bestMV.y = vy;
      workarea.bestMV.sad = sad;
      workarea.nMinCost = cost;
      *dir = val;
    }
#else
    // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
    sad_t cost=workarea.MotionDistorsion<pixel_t>(vx, vy);
    if(cost>=workarea.nMinCost) return;

    typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

    sad_t sad;

    if (iUseSubShift == 0)
    {
      sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    }
    else
    {
      int iRefPitchY = 0;
      const unsigned char* ptrRef = GetRefBlockSubShifted(workarea, vx, vy, iRefPitchY);
//      sad = SAD(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
      sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
    }

    cost += sad + ((penaltyNew*(safe_sad_t)sad) >> 8);
    if(cost>=workarea.nMinCost) return;

    sad_t saduv;

    if (iUseSubShift == 0)
    {
/*      saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
        + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
        + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;

    }
    else
    {
      if (chroma)
      {
        int iRefPitchU = 0;
        int iRefPitchV = 0;
        const unsigned char* ptrRefU = GetRefBlockUSubShifted(workarea, vx, vy, iRefPitchU);
        const unsigned char* ptrRefV = GetRefBlockVSubShifted(workarea, vx, vy, iRefPitchV);
/*        saduv = ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
          + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);*/
        saduv = ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
          + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);
      }
      else
        saduv = 0;
    }

    cost += saduv+((penaltyNew*(safe_sad_t)saduv) >> 8);
    if(cost>=workarea.nMinCost) return;

    workarea.bestMV.x = vx;
    workarea.bestMV.y = vy;
    workarea.nMinCost = cost;
    workarea.bestMV.sad = sad + saduv;
    *dir = val;
#endif
  }
}

/* check if the vector (vx, vy) is better, and update dir accordingly, but not workarea.bestMV.x, y */
template<typename pixel_t>
MV_FORCEINLINE void	PlaneOfBlocks::CheckMVdir(WorkingArea &workarea, int vx, int vy, int *dir, int val)
{
  if (
#ifdef ONLY_CHECK_NONDEFAULT_MV
  ((vx != 0) || (vy != zeroMVfieldShifted.y)) &&
    ((vx != workarea.predictor.x) || (vy != workarea.predictor.y)) &&
    ((vx != workarea.globalMVPredictor.x) || (vy != workarea.globalMVPredictor.y)) &&
#endif
    workarea.IsVectorOK(vx, vy))
  {
#if 0
    sad_t saduv = (chroma) ? ScaleSadChroma(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale) : 0;
    sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    sad += saduv;
    sad_t cost = sad + workarea.MotionDistorsion(vx, vy) + ((penaltyNew*(bigsad_t)sad) >> 8); // v1.5.8
//		if (sad > LSAD/4) DebugPrintf("%d %d %d %d %d %d %d", workarea.blkIdx, vx, vy, val, workarea.nMinCost, cost, sad);
//		int cost = sad + sad*workarea.MotionDistorsion(vx, vy)/(nBlkSizeX*nBlkSizeY*4) + ((penaltyNew*sad)>>8); // v1.5.8
    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.sad = sad;
      workarea.nMinCost = cost;
      *dir = val;
    }
#else
    // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
    sad_t cost=workarea.MotionDistorsion<pixel_t>(vx, vy);
    if(cost>=workarea.nMinCost) return;

    typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

    sad_t sad=LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, vx, vy));
    cost += sad + ((penaltyNew*(safe_sad_t)sad) >> 8);
    if(cost>=workarea.nMinCost) return;

/*    sad_t saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
    sad_t saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, vx, vy), nRefPitch[1])
      + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, vx, vy), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
    cost += saduv+((penaltyNew*(safe_sad_t)saduv) >> 8);
    if(cost>=workarea.nMinCost) return;

    workarea.nMinCost = cost;
    workarea.bestMV.sad = sad + saduv;
    *dir = val;
#endif
  }
}

/* clip a vector to the horizontal boundaries */
MV_FORCEINLINE int	PlaneOfBlocks::ClipMVx(WorkingArea &workarea, int vx)
{
  //	return imin(workarea.nDxMax - 1, imax(workarea.nDxMin, vx));
  if (vx < workarea.nDxMin) return workarea.nDxMin;
  else if (vx >= workarea.nDxMax) return workarea.nDxMax - 1;
  else return vx;
}

/* clip a vector to the vertical boundaries */
MV_FORCEINLINE int	PlaneOfBlocks::ClipMVy(WorkingArea &workarea, int vy)
{
  //	return imin(workarea.nDyMax - 1, imax(workarea.nDyMin, vy));
  if (vy < workarea.nDyMin) return workarea.nDyMin;
  else if (vy >= workarea.nDyMax) return workarea.nDyMax - 1;
  else return vy;
}

/* clip a vector to the search boundaries */
MV_FORCEINLINE VECTOR	PlaneOfBlocks::ClipMV(WorkingArea& workarea, VECTOR v)
{
  VECTOR v2;

  if (sse41 && optSearchOption > 0)
  {
    __m128i xmm0_yx = _mm_loadl_epi64((__m128i*)&v.x); // check is it faster ??

    xmm0_yx = _mm_min_epi32(xmm0_yx, _mm_set_epi32(0, 0, workarea.nDyMax - 1, workarea.nDxMax - 1));
    xmm0_yx = _mm_max_epi32(xmm0_yx, _mm_set_epi32(0, 0, workarea.nDyMin, workarea.nDxMin));

    _mm_storel_epi64((__m128i*) & v2.x, xmm0_yx);

  }
  else
  {
    v2.x = ClipMVx(workarea, v.x);
    v2.y = ClipMVy(workarea, v.y);
  }

  v2.sad = v.sad;

  return v2;
}

MV_FORCEINLINE VECTOR	PlaneOfBlocks::ClipMV_SO2(WorkingArea& workarea, VECTOR v)
{
  VECTOR v2;
/*  __m128i xmm0_yx = _mm_set_epi32(0, 0, v.y, v.x); // check is it faster ??

  xmm0_yx = _mm_min_epi32(xmm0_yx, _mm_set_epi32(0, 0, workarea.nDyMax - 1, workarea.nDxMax - 1));
  xmm0_yx = _mm_max_epi32(xmm0_yx, _mm_set_epi32(0, 0, workarea.nDyMin, workarea.nDxMin));

  v2.x = _mm_cvtsi128_si32(xmm0_yx);
  v2.y = _mm_extract_epi32(xmm0_yx, 1);*/

  __m128i xmm0_yx = _mm_loadl_epi64((__m128i*) & v.x); // check is it faster ??

  xmm0_yx = _mm_min_epi32(xmm0_yx, _mm_set_epi32(0, 0, workarea.nDyMax - 1, workarea.nDxMax - 1));
  xmm0_yx = _mm_max_epi32(xmm0_yx, _mm_set_epi32(0, 0, workarea.nDyMin, workarea.nDxMin));

  _mm_storel_epi64((__m128i*) & v2.x, xmm0_yx);

  v2.sad = v.sad;

  return v2;
}


/* find the median between a, b and c */
MV_FORCEINLINE int	PlaneOfBlocks::Median(int a, int b, int c)
{
  //	return a + b + c - imax(a, imax(b, c)) - imin(c, imin(a, b));
  if (a < b)
  {
    if (b < c) return b;
    else if (a < c) return c;
    else return a;
  }
  else {
    if (a < c) return a;
    else if (b < c) return c;
    else return b;
  }
}

/* computes square distance between two vectors */
#if 0
// not used
MV_FORCEINLINE unsigned int	PlaneOfBlocks::SquareDifferenceNorm(const VECTOR& v1, const VECTOR& v2)
{
  return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y);
}
#endif
/* computes square distance between two vectors */
MV_FORCEINLINE unsigned int	PlaneOfBlocks::SquareDifferenceNorm(const VECTOR& v1, const int v2x, const int v2y)
{
  const int d1 = (v1.x - v2x);
  const int d2 = (v1.y - v2y);
  return d1 * d1 + d2 * d2;
}

/* check if an index is inside the block's min and max indexes */
MV_FORCEINLINE bool	PlaneOfBlocks::IsInFrame(int i)
{
  return ((i >= 0) && (i < nBlkCount));
}



template<typename pixel_t>
void	PlaneOfBlocks::search_mv_slice(Slicer::TaskData &td)
{
  assert(&td != 0);

  short *outfilebuf = _outfilebuf;

  WorkingArea &	workarea = *(_workarea_pool.take_obj());
  assert(&workarea != 0);

  workarea.blky_beg = td._y_beg;
  workarea.blky_end = td._y_end;

  workarea.DCT = 0;
#ifdef ALLOW_DCT
  if (_dct_pool_ptr != 0)
  {
    workarea.DCT = _dct_pool_ptr->take_obj();
  }
#endif	// ALLOW_DCT

  int *pBlkData = _out + 1 + workarea.blky_beg * nBlkX*N_PER_BLOCK;
  if (outfilebuf != NULL)
  {
    outfilebuf += workarea.blky_beg * nBlkX * 4;// 4 short word per block
    // short vx, short vy, uint32_t sad
  }

  workarea.y[0] = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  workarea.y[0] += workarea.blky_beg * (nBlkSizeY - nOverlapY);

  // fixme: use if(chroma) like in recalculate
  if (pSrcFrame->GetMode() & UPLANE)
  {
    workarea.y[1] = pSrcFrame->GetPlane(UPLANE)->GetVPadding();
    workarea.y[1] += workarea.blky_beg * ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
  }
  if (pSrcFrame->GetMode() & VPLANE)
  {
    workarea.y[2] = pSrcFrame->GetPlane(VPLANE)->GetVPadding();
    workarea.y[2] += workarea.blky_beg * ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
  }

  workarea.planeSAD = 0; // for debug, plus fixme outer planeSAD is not used
  workarea.sumLumaChange = 0;

  int nBlkSizeX_Ovr[3] = { (nBlkSizeX - nOverlapX), (nBlkSizeX - nOverlapX) >> nLogxRatioUV, (nBlkSizeX - nOverlapX) >> nLogxRatioUV };
  int nBlkSizeY_Ovr[3] = { (nBlkSizeY - nOverlapY), (nBlkSizeY - nOverlapY) >> nLogyRatioUV, (nBlkSizeY - nOverlapY) >> nLogyRatioUV };

  for (workarea.blky = workarea.blky_beg; workarea.blky < workarea.blky_end; workarea.blky++)
  {
    workarea.blkScanDir = (workarea.blky % 2 == 0 || !_meander_flag) ? 1 : -1;
    // meander (alternate) scan blocks (even row left to right, odd row right to left)
    int blkxStart = (workarea.blky % 2 == 0 || !_meander_flag) ? 0 : nBlkX - 1;
    if (workarea.blkScanDir == 1) // start with leftmost block
    {
      workarea.x[0] = pSrcFrame->GetPlane(YPLANE)->GetHPadding();
      if (chroma)
      {
        workarea.x[1] = pSrcFrame->GetPlane(UPLANE)->GetHPadding();
        workarea.x[2] = pSrcFrame->GetPlane(VPLANE)->GetHPadding();
      }
    }
    else // start with rightmost block, but it is already set at prev row
    {
      workarea.x[0] = pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nBlkSizeX_Ovr[0]*(nBlkX - 1);
      if (chroma)
      {
        workarea.x[1] = pSrcFrame->GetPlane(UPLANE)->GetHPadding() + nBlkSizeX_Ovr[1]*(nBlkX - 1);
        workarea.x[2] = pSrcFrame->GetPlane(VPLANE)->GetHPadding() + nBlkSizeX_Ovr[2]*(nBlkX - 1);
      }
    }

    if (_predictorType != 4)
    {
      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
      {
        workarea.blkx = blkxStart + iblkx * workarea.blkScanDir;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;
        workarea.iter = 0;
        //			DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
        PROFILE_START(MOTION_PROFILE_ME);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)

        // fixme: why recalc is resetting only outside, why, maybe recalc is not using that at all?
        workarea.globalMVPredictor = _glob_mv_pred_def;

#if (ALIGN_SOURCEBLOCK > 1)
        //store the pitch
        const BYTE* pY = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        //create aligned copy
        BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], pY, nSrcPitch_plane[0]);
        //set the to the aligned copy
        workarea.pSrc[0] = workarea.pSrc_temp[0];
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
          workarea.pSrc[1] = workarea.pSrc_temp[1];
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
          BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
          workarea.pSrc[2] = workarea.pSrc_temp[2];
        }
#else	// ALIGN_SOURCEBLOCK
        workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        }
#endif	// ALIGN_SOURCEBLOCK

        // fixme note:
        // MAnalyze mt-inconsistency reason #3
        // this is _not_ internal mt friendly
        // because workarea.nLambda is set to 0 differently:
        // In vertically sliced multithreaded case it happens an _each_ top of the sliced block
        // In non-mt: only for the most top blocks

        if (workarea.blky == workarea.blky_beg)
        {
          workarea.nLambda = 0;
        }
        else
        {
          workarea.nLambda = _lambda_level;
        }

        // fixme:
        // not exacly nice, but works
        // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
        penaltyNew = _pnew; // penalty for new vector
        LSAD = _lsad;    // SAD limit for lambda using
        // may be they must be scaled by nPel ?

        // decreased padding of coarse levels
        int nHPaddingScaled = pSrcFrame->GetPlane(YPLANE)->GetHPadding() >> nLogScale;
        int nVPaddingScaled = pSrcFrame->GetPlane(YPLANE)->GetVPadding() >> nLogScale;
        /* computes search boundaries */
        if (iUseSubShift == 0)
        {
          workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
          workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
          workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
          workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
        }
        else
        {
          int iKS_sh_d2 = ((SHIFTKERNELSIZE / 2) + 2); // +2 is to prevent run out of buffer for UV planes
          workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled - iKS_sh_d2);
          workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled - iKS_sh_d2);
          workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled - iKS_sh_d2);
          workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled - iKS_sh_d2);
        }

        /* search the mv */
        workarea.predictor = ClipMV(workarea, vectors[workarea.blkIdx]);
        if (temporal)
        {
          workarea.predictors[4] = ClipMV(workarea, *reinterpret_cast<VECTOR*>(&_vecPrev[workarea.blkIdx * N_PER_BLOCK])); // temporal predictor
        }
        else
        {
          workarea.predictors[4] = ClipMV(workarea, zeroMV);
        }

//        if (optSearchOption == 5 || optSearchOption == 6) // only calc sad for x,y from DX12_ME
        if (optSearchOption == 6) // only calc sad for x,y from DX12_ME, SO=5 is shader SAD now
        {
          sad_t sad = 0;
          sad_t saduv = 0;

          if (iUseSubShift == 0)
          {
            sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
/*            saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
              + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
            saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
              + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
          }
          else
          {
            if (chroma)
            {
              int iRefPitchU = 0;
              int iRefPitchV = 0;
              const unsigned char* ptrRefU = GetRefBlockUSubShifted(workarea, workarea.predictor.x, workarea.predictor.y, iRefPitchU);
              const unsigned char* ptrRefV = GetRefBlockVSubShifted(workarea, workarea.predictor.x, workarea.predictor.y, iRefPitchV);
/*              saduv = ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
                + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);*/
              saduv = ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], ptrRefU, iRefPitchU)
                + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], ptrRefV, iRefPitchV), effective_chromaSADscale, scaleCSADfine);

            }
            int iRefPitchY = 0;
            const unsigned char* ptrRef = GetRefBlockSubShifted(workarea, workarea.predictor.x, workarea.predictor.y, iRefPitchY);
//            sad = SAD(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
            sad = DM_Luma->GetDisMetric(workarea.pSrc[0], nSrcPitch[0], ptrRef, iRefPitchY);
          }
          workarea.bestMV = workarea.predictor; // clip outside - no need in MDegrain 
          workarea.bestMV.sad = sad + saduv;
          
        }
        else 
        {
          // Possible point of placement selection of 'predictors control'
          if (_predictorType <= 0)
            PseudoEPZSearch<pixel_t>(workarea); // all predictors (original)
          else if (_predictorType == 1) // DTL: partial predictors
            PseudoEPZSearch_glob_med_pred<pixel_t>(workarea);
          else if (_predictorType == 2) // DTL: no predictiors
            PseudoEPZSearch_no_pred<pixel_t>(workarea);
          else // DTL: no refine (at level = 0 typically)
            PseudoEPZSearch_no_refine<pixel_t>(workarea);
        }

        // workarea.bestMV = zeroMV; // debug

        if (outfilebuf != NULL) // write vector to outfile
        {
          outfilebuf[workarea.blkx * 4 + 0] = workarea.bestMV.x;
          outfilebuf[workarea.blkx * 4 + 1] = workarea.bestMV.y;
          outfilebuf[workarea.blkx * 4 + 2] = (*(uint32_t*)(&workarea.bestMV.sad) & 0x0000ffff); // low word
          outfilebuf[workarea.blkx * 4 + 3] = (*(uint32_t*)(&workarea.bestMV.sad) >> 16);     // high word, usually null
        }

        /* write the results */
        pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);

        PROFILE_STOP(MOTION_PROFILE_ME);


        if (smallestPlane) // do we need it with DX12_ME ??? 
        {
          /*
          int64_t i64_1 = 0;
          int64_t i64_2 = 0;
          int32_t i32 = 0;
          unsigned int a1 = 200;
          unsigned int a2 = 201;

          i64_1 += a1 - a2; // 0x00000000 FFFFFFFF   !!!!!
          i64_2 = i64_2 + a1 - a2; // 0xFFFFFFFF FFFFFFFF O.K.!
          i32 += a1 - a2; // 0xFFFFFFFF
          */

          // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
          // LUMA diff can be negative! we should cast from uint32_t
          // 64 bit cast or else: int64_t += uint32t - uint32_t results in int64_t += (uint32_t)(uint32t - uint32_t)
          // which is baaaad 0x00000000 FFFFFFFF instead of 0xFFFFFFFF FFFFFFFF

          // 161204 todo check: why is it not abs(lumadiff)?
          typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
          workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
        }

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        }
      }	// for iblkx
    }// if predictorType != 4
    else
    {
      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
      {
        workarea.blkx = iblkx; 
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;

        workarea.bestMV = vectors[workarea.blkIdx]; // may be not safe somewhere in MDegrain without clipping ??

        if (outfilebuf != NULL) // write vector to outfile
        {
          outfilebuf[workarea.blkx * 4 + 0] = workarea.bestMV.x;
          outfilebuf[workarea.blkx * 4 + 1] = workarea.bestMV.y;
          outfilebuf[workarea.blkx * 4 + 2] = (*(uint32_t*)(&workarea.bestMV.sad) & 0x0000ffff); // low word
          outfilebuf[workarea.blkx * 4 + 3] = (*(uint32_t*)(&workarea.bestMV.sad) >> 16);     // high word, usually null
        }

        /* write the results */
        pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);

      }	// for iblkx

    } // opt_predictorType=4

    pBlkData += nBlkX*N_PER_BLOCK;
    if (outfilebuf != NULL) // write vector to outfile
    {
      outfilebuf += nBlkX * 4;// 4 short word per block
    }

    workarea.y[0] += nBlkSizeY_Ovr[0];
    workarea.y[1] += nBlkSizeY_Ovr[1];
    workarea.y[2] += nBlkSizeY_Ovr[2];
  }	// for workarea.blky

  planeSAD += workarea.planeSAD; // for debug, plus fixme outer planeSAD is not used
  sumLumaChange += workarea.sumLumaChange;

  if (isse)
  {
#ifndef _M_X64
    _mm_empty();
#endif
  }

#ifdef ALLOW_DCT
  if (_dct_pool_ptr != 0)
  {
    _dct_pool_ptr->return_obj(*(workarea.DCT));
    workarea.DCT = 0;
  }
#endif

  _workarea_pool.return_obj(workarea);
} // search_mv_slice

template<typename pixel_t>
void	PlaneOfBlocks::search_mv_slice_SO2(Slicer::TaskData& td)
{
  assert(&td != 0);

  bool bInterframeH, bInterframeV;

  short* outfilebuf = _outfilebuf;

  WorkingArea& workarea = *(_workarea_pool.take_obj());
  assert(&workarea != 0);

  workarea.blky_beg = td._y_beg;
  workarea.blky_end = td._y_end;

  workarea.DCT = 0;

  if (nSearchParam == 1)
  {
    if (nBlkSizeX == 8 && nBlkSizeY == 8)
    {
      if (USE_AVX512)
      {
        ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512;
      }
      else
      {
        ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2;
      }
    }
    else if (nBlkSizeX == 16 && nBlkSizeY == 16)
    {
      if (USE_AVX512)
      {
        ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch16x16_uint8_SO2_np1_sp1_avx512;
      }
      else
      {
        ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch16x16_uint8_SO2_np1_sp1_avx2;
      }
    }
  }
  else // sp = 2 at all levels except finest
  {
    if (nBlkSizeX == 8 && nBlkSizeY == 8)
    {
      ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp2_avx2;
    }
    else if (nBlkSizeX == 16 && nBlkSizeY == 16)
    {
      ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch16x16_uint8_SO2_np1_sp2_avx2;
    }
  }
  /*
  if (_predictorType == 0) // this selector method do not work not now - need to found why ???
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2;
  }
  else if (_predictorType == 1)
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2_glob_med_pred;
  }
   else if (_predictorType == 2)
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2_no_pred;
  }
  */

  const int iY_H_Padding = pSrcFrame->GetPlane(YPLANE)->GetHPadding();
  const int iY_V_Padding = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  const int iY_Ext_Width = pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth();
  const int iY_Ext_Height = pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight();
  const int iY_Height = pSrcFrame->GetPlane(YPLANE)->GetHeight();

  int* pBlkData = _out + 1 + workarea.blky_beg * nBlkX * N_PER_BLOCK;
  if (outfilebuf != NULL)
  {
    outfilebuf += workarea.blky_beg * nBlkX * 4;// 4 short word per block
    // short vx, short vy, uint32_t sad
  }

  workarea.y[0] = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  workarea.y[0] += workarea.blky_beg * (nBlkSizeY - nOverlapY);

  workarea.planeSAD = 0; // for debug, plus fixme outer planeSAD is not used
  workarea.sumLumaChange = 0;

  int nBlkSizeX_Ovr[3] = { (nBlkSizeX - nOverlapX), (nBlkSizeX - nOverlapX) >> nLogxRatioUV, (nBlkSizeX - nOverlapX) >> nLogxRatioUV };
  int nBlkSizeY_Ovr[3] = { (nBlkSizeY - nOverlapY), (nBlkSizeY - nOverlapY) >> nLogyRatioUV, (nBlkSizeY - nOverlapY) >> nLogyRatioUV };

  for (workarea.blky = workarea.blky_beg; workarea.blky < workarea.blky_end; workarea.blky++) 
  {
    bInterframeV = ((workarea.blky != workarea.blky_beg) && (workarea.blky != workarea.blky_end - 1));

    workarea.blkScanDir = (workarea.blky % 2 == 0 || !_meander_flag) ? 1 : -1;
    // meander (alternate) scan blocks (even row left to right, odd row right to left)
    int blkxStart = (workarea.blky % 2 == 0 || !_meander_flag) ? 0 : nBlkX - 1;
    if (workarea.blkScanDir == 1) // start with leftmost block
    {
      workarea.x[0] = iY_H_Padding; //  pSrcFrame->GetPlane(YPLANE)->GetHPadding();
    }
    else // start with rightmost block, but it is already set at prev row
    {
      workarea.x[0] = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ + nBlkSizeX_Ovr[0] * (nBlkX - 1);
    }

    if(_predictorType != 4)
    {
      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
      {
        bInterframeH = ((iblkx > 0) && (iblkx < (nBlkX - 1)));

        workarea.blkx = blkxStart + iblkx * workarea.blkScanDir;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;
        workarea.iter = 0;
        //			DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
        PROFILE_START(MOTION_PROFILE_ME);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)

        // fixme: why recalc is resetting only outside, why, maybe recalc is not using that at all?
        workarea.globalMVPredictor = _glob_mv_pred_def;

#if (ALIGN_SOURCEBLOCK > 1)
        //store the pitch
        const BYTE* pY = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        //create aligned copy
        BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], pY, nSrcPitch_plane[0]);
        //set the to the aligned copy
        workarea.pSrc[0] = workarea.pSrc_temp[0];
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
          workarea.pSrc[1] = workarea.pSrc_temp[1];
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
          BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
          workarea.pSrc[2] = workarea.pSrc_temp[2];
        }
#else	// ALIGN_SOURCEBLOCK
        workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        }
#endif	// ALIGN_SOURCEBLOCK

        // fixme note:
        // MAnalyze mt-inconsistency reason #3
        // this is _not_ internal mt friendly
        // because workarea.nLambda is set to 0 differently:
        // In vertically sliced multithreaded case it happens an _each_ top of the sliced block
        // In non-mt: only for the most top blocks

        if (workarea.blky == workarea.blky_beg)
        {
          workarea.nLambda = 0;
        }
        else
        {
          workarea.nLambda = _lambda_level;
        }

        // fixme:
        // not exacly nice, but works
        // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
        penaltyNew = _pnew; // penalty for new vector
        LSAD = _lsad;    // SAD limit for lambda using
        // may be they must be scaled by nPel ?

        // decreased padding of coarse levels
        int nHPaddingScaled = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ >> nLogScale;
        int nVPaddingScaled = iY_V_Padding /*pSrcFrame->GetPlane(YPLANE)->GetVPadding()*/ >> nLogScale;
        /* computes search boundaries */
  /*    workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
        workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);*/
        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - nBlkSizeX - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMax = nPel * (iY_Ext_Height - workarea.y[0] - nBlkSizeY - iY_V_Padding + nVPaddingScaled - nSearchParam);
        workarea.nDxMin = -nPel * (workarea.x[0] - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - iY_V_Padding + nVPaddingScaled - nSearchParam); // if (- nSearchParam) not helps - need to think more.

        /* search the mv */
        workarea.predictor = ClipMV_SO2(workarea, vectors[workarea.blkIdx]);

        workarea.predictors[4] = ClipMV_SO2(workarea, zeroMV);

        workarea.bIntraframe = bInterframeH && bInterframeV;

        if (_predictorType == 0)
        {
          if (nBlkSizeX == 8 && nBlkSizeY == 8)
            PseudoEPZSearch_optSO2_8x8_avx2<pixel_t>(workarea); // all predictors (original) 8x8 avx2, also predictor type 1
          else
            PseudoEPZSearch_optSO2<pixel_t>(workarea); // all predictors (original), other block sizes
        }
        else if (_predictorType == 1)
        {
          if (nBlkSizeX == 8 && nBlkSizeY == 8)
            PseudoEPZSearch_optSO2_8x8_avx2<pixel_t>(workarea); // all predictors (original) 8x8 avx2, also predictor type 1
          else
            PseudoEPZSearch_optSO2_glob_med_pred<pixel_t>(workarea); // other block sizes
        }
        else if (_predictorType == 2)// _predictorType == 2
          PseudoEPZSearch_optSO2_no_pred<pixel_t>(workarea);
        else // _predictorType == 3
          PseudoEPZSearch_optSO2_no_refine<pixel_t>(workarea);
          
  //      (this->*Sel_Pseudo_EPZ_search_SO2)(workarea); // still not works - maybe possible to fix ?

        // workarea.bestMV = zeroMV; // debug

        /* write the results */
        pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);

        PROFILE_STOP(MOTION_PROFILE_ME);

        if (smallestPlane)
        {
          /*
          int64_t i64_1 = 0;
          int64_t i64_2 = 0;
          int32_t i32 = 0;
          unsigned int a1 = 200;
          unsigned int a2 = 201;

          i64_1 += a1 - a2; // 0x00000000 FFFFFFFF   !!!!!
          i64_2 = i64_2 + a1 - a2; // 0xFFFFFFFF FFFFFFFF O.K.!
          i32 += a1 - a2; // 0xFFFFFFFF
          */

          // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
          // LUMA diff can be negative! we should cast from uint32_t
          // 64 bit cast or else: int64_t += uint32t - uint32_t results in int64_t += (uint32_t)(uint32t - uint32_t)
          // which is baaaad 0x00000000 FFFFFFFF instead of 0xFFFFFFFF FFFFFFFF

          // 161204 todo check: why is it not abs(lumadiff)?
          typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
          workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
        }

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        }
      }	// for iblkx
    }// if predictorType != 4
    else
    {
      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
      {
        workarea.blkx = iblkx;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;

        workarea.bestMV = vectors[workarea.blkIdx]; // may be not safe somewhere in MDegrain without clipping ??

        if (outfilebuf != NULL) // write vector to outfile
        {
          outfilebuf[workarea.blkx * 4 + 0] = workarea.bestMV.x;
          outfilebuf[workarea.blkx * 4 + 1] = workarea.bestMV.y;
          outfilebuf[workarea.blkx * 4 + 2] = (*(uint32_t*)(&workarea.bestMV.sad) & 0x0000ffff); // low word
          outfilebuf[workarea.blkx * 4 + 3] = (*(uint32_t*)(&workarea.bestMV.sad) >> 16);     // high word, usually null
        }

        /* write the results */
        pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);

      }	// for iblkx

    } // opt_predictorType=4


    pBlkData += nBlkX * N_PER_BLOCK;

    workarea.y[0] += nBlkSizeY_Ovr[0];
    workarea.y[1] += nBlkSizeY_Ovr[1];
    workarea.y[2] += nBlkSizeY_Ovr[2];
  }	// for workarea.blky

  planeSAD += workarea.planeSAD; // for debug, plus fixme outer planeSAD is not used
  sumLumaChange += workarea.sumLumaChange;

  if (isse)
  {
#ifndef _M_X64
    _mm_empty();
#endif
  }

  _workarea_pool.return_obj(workarea);
} // search_mv_slice_SO2

template<typename pixel_t>
void	PlaneOfBlocks::search_mv_slice_SO3(Slicer::TaskData& td) // multi-blocks search AVX2
{
//#define NUM_BLOCKS_PER_SEARCH 4

  assert(&td != 0);

  bool bInterframeH, bInterframeV;

  short* outfilebuf = _outfilebuf;

  WorkingArea& workarea = *(_workarea_pool.take_obj());
  assert(&workarea != 0);

  workarea.blky_beg = td._y_beg;
  workarea.blky_end = td._y_end;

  workarea.DCT = 0;

  if (nSearchParam == 1)
  {
//    ExhaustiveSearch8x8_avx2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2;
  }
  else // sp = 2 at all levels except finest
  {
    ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp2_avx2;
  }
  /*
  if (_predictorType == 0) // this selector method do not work not now - need to found why ???
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2;
  }
  else if (_predictorType == 1)
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2_glob_med_pred;
  }
   else if (_predictorType == 2)
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2_no_pred;
  }
  */

  const int iY_H_Padding = pSrcFrame->GetPlane(YPLANE)->GetHPadding();
  const int iY_V_Padding = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  const int iY_Ext_Width = pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth();
  const int iY_Ext_Height = pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight();
  const int iY_Height = pSrcFrame->GetPlane(YPLANE)->GetHeight();

  int* pBlkData = _out + 1 + workarea.blky_beg * nBlkX * N_PER_BLOCK;
  if (outfilebuf != NULL)
  {
    outfilebuf += workarea.blky_beg * nBlkX * 4;// 4 short word per block
    // short vx, short vy, uint32_t sad
  }

  workarea.y[0] = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  workarea.y[0] += workarea.blky_beg * (nBlkSizeY - nOverlapY);

  workarea.planeSAD = 0; // for debug, plus fixme outer planeSAD is not used
  workarea.sumLumaChange = 0;

  int nBlkSizeX_Ovr[3] = { (nBlkSizeX - nOverlapX), (nBlkSizeX - nOverlapX) >> nLogxRatioUV, (nBlkSizeX - nOverlapX) >> nLogxRatioUV };
  int nBlkSizeY_Ovr[3] = { (nBlkSizeY - nOverlapY), (nBlkSizeY - nOverlapY) >> nLogyRatioUV, (nBlkSizeY - nOverlapY) >> nLogyRatioUV };

  for (workarea.blky = workarea.blky_beg; workarea.blky < workarea.blky_end; workarea.blky++)
  {
    bInterframeV = ((workarea.blky != workarea.blky_beg) && (workarea.blky != workarea.blky_end - 1));

    workarea.blkScanDir = (workarea.blky % 2 == 0 || !_meander_flag) ? 1 : -1;
    // meander (alternate) scan blocks (even row left to right, odd row right to left)
    int blkxStart = (workarea.blky % 2 == 0 || !_meander_flag) ? 0 : nBlkX - 1;
    if (workarea.blkScanDir == 1) // start with leftmost block
    {
      workarea.x[0] = iY_H_Padding; //  pSrcFrame->GetPlane(YPLANE)->GetHPadding();
    }
    else // start with rightmost block, but it is already set at prev row
    {
      workarea.x[0] = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ + nBlkSizeX_Ovr[0] * (nBlkX - 1);
    }

    if (nSearchParam == 2) // std search
//    if (nSearchParam == 1) // 4bl search
    {
      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
//      for (int iblkx = 0; iblkx < nBlkX; iblkx += MAX_MULTI_BLOCKS_8x8_AVX2)
      {
        bInterframeH = ((iblkx > 0) && (iblkx < (nBlkX - 1)));

        workarea.blkx = blkxStart + iblkx * workarea.blkScanDir;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;
        workarea.iter = 0;
        //			DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
        PROFILE_START(MOTION_PROFILE_ME);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)

        // fixme: why recalc is resetting only outside, why, maybe recalc is not using that at all?
        workarea.globalMVPredictor = _glob_mv_pred_def;

#if (ALIGN_SOURCEBLOCK > 1)
        //store the pitch
        const BYTE* pY = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        //create aligned copy
        BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], pY, nSrcPitch_plane[0]);
        //set the to the aligned copy
        workarea.pSrc[0] = workarea.pSrc_temp[0];
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
          workarea.pSrc[1] = workarea.pSrc_temp[1];
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
          BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
          workarea.pSrc[2] = workarea.pSrc_temp[2];
        }
#else	// ALIGN_SOURCEBLOCK
        workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        }
#endif	// ALIGN_SOURCEBLOCK

        // fixme note:
        // MAnalyze mt-inconsistency reason #3
        // this is _not_ internal mt friendly
        // because workarea.nLambda is set to 0 differently:
        // In vertically sliced multithreaded case it happens an _each_ top of the sliced block
        // In non-mt: only for the most top blocks

        if (workarea.blky == workarea.blky_beg)
        {
          workarea.nLambda = 0;
        }
        else
        {
          workarea.nLambda = _lambda_level;
        }

        // fixme:
        // not exacly nice, but works
        // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
        penaltyNew = _pnew; // penalty for new vector
        LSAD = _lsad;    // SAD limit for lambda using
        // may be they must be scaled by nPel ?

        // decreased padding of coarse levels
        int nHPaddingScaled = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ >> nLogScale;
        int nVPaddingScaled = iY_V_Padding /*pSrcFrame->GetPlane(YPLANE)->GetVPadding()*/ >> nLogScale;
        /* computes search boundaries */
  /*    workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
        workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);*/

        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - nBlkSizeX - iY_H_Padding + nHPaddingScaled);
//        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - (nBlkSizeX * MAX_MULTI_BLOCKS_8x8_AVX2) - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMax = nPel * (iY_Ext_Height - workarea.y[0] - nBlkSizeY - iY_V_Padding + nVPaddingScaled - nSearchParam);
        workarea.nDxMin = -nPel * (workarea.x[0] - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - iY_V_Padding + nVPaddingScaled - nSearchParam); // if (- nSearchParam) not helps - need to think more.

        /* search the mv */
        workarea.predictor = ClipMV_SO2(workarea, vectors[workarea.blkIdx]);

        workarea.predictors[4] = ClipMV_SO2(workarea, zeroMV);

        workarea.bIntraframe = bInterframeH && bInterframeV;

        if (_predictorType == 0)
          PseudoEPZSearch_optSO2<pixel_t>(workarea); // all predictors (original)
        else if (_predictorType == 1)
          PseudoEPZSearch_optSO2_glob_med_pred<pixel_t>(workarea);
        else // _predictorType == 2
          PseudoEPZSearch_optSO2_no_pred<pixel_t>(workarea);

        
/*        if (_predictorType == 1)
          PseudoEPZSearch_optSO3_glob_pred_avx2<pixel_t>(workarea, pBlkData);
        else // PT=2
          PseudoEPZSearch_optSO3_no_pred<pixel_t>(workarea, pBlkData);
          */
        // workarea.bestMV = zeroMV; // debug

              /* write the results */
 /*       pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);
        */
        PROFILE_STOP(MOTION_PROFILE_ME);

        if (smallestPlane)
        {
          /*
          int64_t i64_1 = 0;
          int64_t i64_2 = 0;
          int32_t i32 = 0;
          unsigned int a1 = 200;
          unsigned int a2 = 201;

          i64_1 += a1 - a2; // 0x00000000 FFFFFFFF   !!!!!
          i64_2 = i64_2 + a1 - a2; // 0xFFFFFFFF FFFFFFFF O.K.!
          i32 += a1 - a2; // 0xFFFFFFFF
          */

          // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
          // LUMA diff can be negative! we should cast from uint32_t
          // 64 bit cast or else: int64_t += uint32t - uint32_t results in int64_t += (uint32_t)(uint32t - uint32_t)
          // which is baaaad 0x00000000 FFFFFFFF instead of 0xFFFFFFFF FFFFFFFF

          // 161204 todo check: why is it not abs(lumadiff)?
          typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
          workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
        }

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        }
/*        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        } */
      }	// for iblkx
    } // if nSearchparam==2 - level 1 and more
    else // if nsearchparam == 1 - level 0, 4blocks search
    {

      for (int iblkx = 0; iblkx < nBlkX; iblkx += MAX_MULTI_BLOCKS_8x8_AVX2)
      {
        bInterframeH = ((iblkx > 0) && (iblkx < (nBlkX - 1)));

        workarea.blkx = blkxStart + iblkx * workarea.blkScanDir;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;
        workarea.iter = 0;

        // prefetch is useful for speed but prefetch advance distance (and count and cache level ?) need to be selected after series of tests (at different HW systems ?)
#define PREFETCH_ADVANCE_SOURCE_AVX2 3

        int x_next = workarea.x[0] + PREFETCH_ADVANCE_SOURCE_AVX2 * (MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[0] * workarea.blkScanDir);
        const uint8_t* pSrcNext = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(x_next, workarea.y[0]);

        for (int i = 0; i < nBlkSizeY; i++)
        {
          _mm_prefetch(const_cast<const CHAR*>(reinterpret_cast<const CHAR*>(pSrcNext + nSrcPitch[0] * i)), _MM_HINT_T0);
        }

        x_next += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[0] * workarea.blkScanDir;
        pSrcNext = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(x_next, workarea.y[0]);

        for (int i = 0; i < nBlkSizeY; i++)
        {
          _mm_prefetch(const_cast<const CHAR*>(reinterpret_cast<const CHAR*>(pSrcNext + nSrcPitch[0] * i)), _MM_HINT_T0);
        }
       

        //			DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
        PROFILE_START(MOTION_PROFILE_ME);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)

        // fixme: why recalc is resetting only outside, why, maybe recalc is not using that at all?
        workarea.globalMVPredictor = _glob_mv_pred_def;

  #if (ALIGN_SOURCEBLOCK > 1)
        //store the pitch
        const BYTE* pY = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        //create aligned copy
        BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], pY, nSrcPitch_plane[0]);
        //set the to the aligned copy
        workarea.pSrc[0] = workarea.pSrc_temp[0];
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
          workarea.pSrc[1] = workarea.pSrc_temp[1];
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
          BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
          workarea.pSrc[2] = workarea.pSrc_temp[2];
        }
  #else	// ALIGN_SOURCEBLOCK
        workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        }
  #endif	// ALIGN_SOURCEBLOCK

        // fixme note:
        // MAnalyze mt-inconsistency reason #3
        // this is _not_ internal mt friendly
        // because workarea.nLambda is set to 0 differently:
        // In vertically sliced multithreaded case it happens an _each_ top of the sliced block
        // In non-mt: only for the most top blocks

        if (workarea.blky == workarea.blky_beg)
        {
          workarea.nLambda = 0;
        }
        else
        {
          workarea.nLambda = _lambda_level;
        }

        // fixme:
        // not exacly nice, but works
        // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
        penaltyNew = _pnew; // penalty for new vector
        LSAD = _lsad;    // SAD limit for lambda using
        // may be they must be scaled by nPel ?

        // decreased padding of coarse levels
        int nHPaddingScaled = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ >> nLogScale;
        int nVPaddingScaled = iY_V_Padding /*pSrcFrame->GetPlane(YPLANE)->GetVPadding()*/ >> nLogScale;
        /* computes search boundaries */
  /*    workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
        workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);*/
        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - (nBlkSizeX * MAX_MULTI_BLOCKS_8x8_AVX2) - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMax = nPel * (iY_Ext_Height - workarea.y[0] - nBlkSizeY - iY_V_Padding + nVPaddingScaled - nSearchParam);
        workarea.nDxMin = -nPel * (workarea.x[0] - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - iY_V_Padding + nVPaddingScaled - nSearchParam); // if (- nSearchParam) not helps - need to think more.

        /* search the mv */
        workarea.predictor = ClipMV_SO2(workarea, vectors[workarea.blkIdx]);

        workarea.predictors[4] = ClipMV_SO2(workarea, zeroMV);

        workarea.bIntraframe = bInterframeH && bInterframeV;
/*
        if (_predictorType == 0)
          PseudoEPZSearch_optSO2<pixel_t>(workarea); // all predictors (original)
        else if (_predictorType == 1)
          PseudoEPZSearch_optSO2_glob_med_pred<pixel_t>(workarea);
        else // _predictorType == 2
          PseudoEPZSearch_optSO2_no_pred<pixel_t>(workarea);
        //      (this->*Sel_Pseudo_EPZ_search_SO2)(workarea); // still not works - maybe possible to fix ? */
        if (_predictorType == 1)
          PseudoEPZSearch_optSO3_glob_pred_avx2<pixel_t>(workarea, pBlkData);
        else // PT=2
          PseudoEPZSearch_optSO3_no_pred<pixel_t>(workarea, pBlkData);

        // workarea.bestMV = zeroMV; // debug

              /* write the results */
/*        pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);
        */
        // 4 results written internally in Exa_search_4Blks()

        PROFILE_STOP(MOTION_PROFILE_ME);

        if (smallestPlane)
        {
          /*
          int64_t i64_1 = 0;
          int64_t i64_2 = 0;
          int32_t i32 = 0;
          unsigned int a1 = 200;
          unsigned int a2 = 201;

          i64_1 += a1 - a2; // 0x00000000 FFFFFFFF   !!!!!
          i64_2 = i64_2 + a1 - a2; // 0xFFFFFFFF FFFFFFFF O.K.!
          i32 += a1 - a2; // 0xFFFFFFFF
          */

          // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
          // LUMA diff can be negative! we should cast from uint32_t
          // 64 bit cast or else: int64_t += uint32t - uint32_t results in int64_t += (uint32_t)(uint32t - uint32_t)
          // which is baaaad 0x00000000 FFFFFFFF instead of 0xFFFFFFFF FFFFFFFF

          // 161204 todo check: why is it not abs(lumadiff)?
          typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
          workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
        }

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        }
      }	// for iblkx
    }

    pBlkData += nBlkX * N_PER_BLOCK;

    workarea.y[0] += nBlkSizeY_Ovr[0];
    workarea.y[1] += nBlkSizeY_Ovr[1];
    workarea.y[2] += nBlkSizeY_Ovr[2];
  }	// for workarea.blky

  planeSAD += workarea.planeSAD; // for debug, plus fixme outer planeSAD is not used
  sumLumaChange += workarea.sumLumaChange;

  if (isse)
  {
#ifndef _M_X64
    _mm_empty();
#endif
  }

  _workarea_pool.return_obj(workarea);
} // search_mv_slice_SO3

template<typename pixel_t>
void	PlaneOfBlocks::search_mv_slice_SO4(Slicer::TaskData& td) // multi-blocks search AVX512
{


  assert(&td != 0);

  bool bInterframeH, bInterframeV;

  short* outfilebuf = _outfilebuf;

  WorkingArea& workarea = *(_workarea_pool.take_obj());
  assert(&workarea != 0);

  workarea.blky_beg = td._y_beg;
  workarea.blky_end = td._y_end;

  workarea.DCT = 0;

  if (nSearchParam == 1)
  {
    //    ExhaustiveSearch8x8_avx2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2;
  }
  else // sp = 2 at all levels except finest
  {
    ExhaustiveSearch_SO2 = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_SO2_np1_sp2_avx2;
  }
  /*
  if (_predictorType == 0) // this selector method do not work not now - need to found why ???
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2;
  }
  else if (_predictorType == 1)
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2_glob_med_pred;
  }
   else if (_predictorType == 2)
  {
    Sel_Pseudo_EPZ_search_SO2 = &PlaneOfBlocks::PseudoEPZSearch_optSO2_no_pred;
  }
  */

  const int iY_H_Padding = pSrcFrame->GetPlane(YPLANE)->GetHPadding();
  const int iY_V_Padding = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  const int iY_Ext_Width = pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth();
  const int iY_Ext_Height = pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight();
  const int iY_Height = pSrcFrame->GetPlane(YPLANE)->GetHeight();

  int* pBlkData = _out + 1 + workarea.blky_beg * nBlkX * N_PER_BLOCK;
  if (outfilebuf != NULL)
  {
    outfilebuf += workarea.blky_beg * nBlkX * 4;// 4 short word per block
    // short vx, short vy, uint32_t sad
  }

  workarea.y[0] = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  workarea.y[0] += workarea.blky_beg * (nBlkSizeY - nOverlapY);

  workarea.planeSAD = 0; // for debug, plus fixme outer planeSAD is not used
  workarea.sumLumaChange = 0;

  int nBlkSizeX_Ovr[3] = { (nBlkSizeX - nOverlapX), (nBlkSizeX - nOverlapX) >> nLogxRatioUV, (nBlkSizeX - nOverlapX) >> nLogxRatioUV };
  int nBlkSizeY_Ovr[3] = { (nBlkSizeY - nOverlapY), (nBlkSizeY - nOverlapY) >> nLogyRatioUV, (nBlkSizeY - nOverlapY) >> nLogyRatioUV };

  for (workarea.blky = workarea.blky_beg; workarea.blky < workarea.blky_end; workarea.blky++)
  {
    bInterframeV = ((workarea.blky != workarea.blky_beg) && (workarea.blky != workarea.blky_end - 1));

    workarea.blkScanDir = (workarea.blky % 2 == 0 || !_meander_flag) ? 1 : -1;
    // meander (alternate) scan blocks (even row left to right, odd row right to left)
    int blkxStart = (workarea.blky % 2 == 0 || !_meander_flag) ? 0 : nBlkX - 1;
    if (workarea.blkScanDir == 1) // start with leftmost block
    {
      workarea.x[0] = iY_H_Padding; //  pSrcFrame->GetPlane(YPLANE)->GetHPadding();
    }
    else // start with rightmost block, but it is already set at prev row
    {
      workarea.x[0] = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ + nBlkSizeX_Ovr[0] * (nBlkX - 1);
    }

    if (nSearchParam == 2) // std search
//    if (nSearchParam == 1) // 4bl search
    {
      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
        //      for (int iblkx = 0; iblkx < nBlkX; iblkx += MAX_MULTI_BLOCKS_8x8_AVX2)
      {
        bInterframeH = ((iblkx > 0) && (iblkx < (nBlkX - 1)));

        workarea.blkx = blkxStart + iblkx * workarea.blkScanDir;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;
        workarea.iter = 0;
        //			DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
        PROFILE_START(MOTION_PROFILE_ME);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)

        // fixme: why recalc is resetting only outside, why, maybe recalc is not using that at all?
        workarea.globalMVPredictor = _glob_mv_pred_def;

#if (ALIGN_SOURCEBLOCK > 1)
        //store the pitch
        const BYTE* pY = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        //create aligned copy
        BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], pY, nSrcPitch_plane[0]);
        //set the to the aligned copy
        workarea.pSrc[0] = workarea.pSrc_temp[0];
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
          workarea.pSrc[1] = workarea.pSrc_temp[1];
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
          BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
          workarea.pSrc[2] = workarea.pSrc_temp[2];
        }
#else	// ALIGN_SOURCEBLOCK
        workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        }
#endif	// ALIGN_SOURCEBLOCK

        // fixme note:
        // MAnalyze mt-inconsistency reason #3
        // this is _not_ internal mt friendly
        // because workarea.nLambda is set to 0 differently:
        // In vertically sliced multithreaded case it happens an _each_ top of the sliced block
        // In non-mt: only for the most top blocks

        if (workarea.blky == workarea.blky_beg)
        {
          workarea.nLambda = 0;
        }
        else
        {
          workarea.nLambda = _lambda_level;
        }

        // fixme:
        // not exacly nice, but works
        // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
        penaltyNew = _pnew; // penalty for new vector
        LSAD = _lsad;    // SAD limit for lambda using
        // may be they must be scaled by nPel ?

        // decreased padding of coarse levels
        int nHPaddingScaled = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ >> nLogScale;
        int nVPaddingScaled = iY_V_Padding /*pSrcFrame->GetPlane(YPLANE)->GetVPadding()*/ >> nLogScale;
        /* computes search boundaries */
  /*    workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
        workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);*/

        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - nBlkSizeX - iY_H_Padding + nHPaddingScaled);
        //        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - (nBlkSizeX * MAX_MULTI_BLOCKS_8x8_AVX2) - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMax = nPel * (iY_Ext_Height - workarea.y[0] - nBlkSizeY - iY_V_Padding + nVPaddingScaled - nSearchParam);
        workarea.nDxMin = -nPel * (workarea.x[0] - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - iY_V_Padding + nVPaddingScaled - nSearchParam); // if (- nSearchParam) not helps - need to think more.

        /* search the mv */
        workarea.predictor = ClipMV_SO2(workarea, vectors[workarea.blkIdx]);

        workarea.predictors[4] = ClipMV_SO2(workarea, zeroMV);

        workarea.bIntraframe = bInterframeH && bInterframeV;

        if (_predictorType == 0)
          PseudoEPZSearch_optSO2<pixel_t>(workarea); // all predictors (original)
        else if (_predictorType == 1)
          PseudoEPZSearch_optSO2_glob_med_pred<pixel_t>(workarea);
        else // _predictorType == 2
          PseudoEPZSearch_optSO2_no_pred<pixel_t>(workarea);


        /*        if (_predictorType == 1)
                  PseudoEPZSearch_optSO3_glob_pred_avx2<pixel_t>(workarea, pBlkData);
                else // PT=2
                  PseudoEPZSearch_optSO3_no_pred<pixel_t>(workarea, pBlkData);
                  */
                  // workarea.bestMV = zeroMV; // debug

                        /* write the results */
           /*       pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
                  pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
                  pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);
                  */
        PROFILE_STOP(MOTION_PROFILE_ME);

        if (smallestPlane)
        {
          /*
          int64_t i64_1 = 0;
          int64_t i64_2 = 0;
          int32_t i32 = 0;
          unsigned int a1 = 200;
          unsigned int a2 = 201;

          i64_1 += a1 - a2; // 0x00000000 FFFFFFFF   !!!!!
          i64_2 = i64_2 + a1 - a2; // 0xFFFFFFFF FFFFFFFF O.K.!
          i32 += a1 - a2; // 0xFFFFFFFF
          */

          // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
          // LUMA diff can be negative! we should cast from uint32_t
          // 64 bit cast or else: int64_t += uint32t - uint32_t results in int64_t += (uint32_t)(uint32t - uint32_t)
          // which is baaaad 0x00000000 FFFFFFFF instead of 0xFFFFFFFF FFFFFFFF

          // 161204 todo check: why is it not abs(lumadiff)?
          typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
          workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
        }

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        }
        /*        if (iblkx < nBlkX - 1)
                {
                  workarea.x[0] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[0] * workarea.blkScanDir;
                  workarea.x[1] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[1] * workarea.blkScanDir;
                  workarea.x[2] += MAX_MULTI_BLOCKS_8x8_AVX2 * nBlkSizeX_Ovr[2] * workarea.blkScanDir;
                } */
      }	// for iblkx
    } // if nSearchparam==2 - level 1 and more
    else // if nsearchparam == 1 - level 0, 16blocks search
    {

      for (int iblkx = 0; iblkx < nBlkX; iblkx += MAX_MULTI_BLOCKS_8x8_AVX512)
      {
        bInterframeH = ((iblkx > 0) && (iblkx < (nBlkX - 1)));

        workarea.blkx = blkxStart + iblkx * workarea.blkScanDir;
        workarea.blkIdx = workarea.blky * nBlkX + workarea.blkx;
        workarea.iter = 0;
        //			DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
        PROFILE_START(MOTION_PROFILE_ME);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)

        // fixme: why recalc is resetting only outside, why, maybe recalc is not using that at all?
        workarea.globalMVPredictor = _glob_mv_pred_def;

#if (ALIGN_SOURCEBLOCK > 1)
        //store the pitch
        const BYTE* pY = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        //create aligned copy
        BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], pY, nSrcPitch_plane[0]);
        //set the to the aligned copy
        workarea.pSrc[0] = workarea.pSrc_temp[0];
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
          workarea.pSrc[1] = workarea.pSrc_temp[1];
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
          BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
          workarea.pSrc[2] = workarea.pSrc_temp[2];
        }
#else	// ALIGN_SOURCEBLOCK
        workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
        if (chroma)
        {
          workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
          workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        }
#endif	// ALIGN_SOURCEBLOCK

        // fixme note:
        // MAnalyze mt-inconsistency reason #3
        // this is _not_ internal mt friendly
        // because workarea.nLambda is set to 0 differently:
        // In vertically sliced multithreaded case it happens an _each_ top of the sliced block
        // In non-mt: only for the most top blocks

        if (workarea.blky == workarea.blky_beg)
        {
          workarea.nLambda = 0;
        }
        else
        {
          workarea.nLambda = _lambda_level;
        }

        // fixme:
        // not exacly nice, but works
        // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
        penaltyNew = _pnew; // penalty for new vector
        LSAD = _lsad;    // SAD limit for lambda using
        // may be they must be scaled by nPel ?

        // decreased padding of coarse levels
        int nHPaddingScaled = iY_H_Padding /*pSrcFrame->GetPlane(YPLANE)->GetHPadding()*/ >> nLogScale;
        int nVPaddingScaled = iY_V_Padding /*pSrcFrame->GetPlane(YPLANE)->GetVPadding()*/ >> nLogScale;
        /* computes search boundaries */
  /*    workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);
        workarea.nDxMin = -nPel * (workarea.x[0] - pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - pSrcFrame->GetPlane(YPLANE)->GetVPadding() + nVPaddingScaled);*/
        workarea.nDxMax = nPel * (iY_Ext_Width - workarea.x[0] - (nBlkSizeX * MAX_MULTI_BLOCKS_8x8_AVX512) - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMax = nPel * (iY_Ext_Height - workarea.y[0] - nBlkSizeY - iY_V_Padding + nVPaddingScaled - nSearchParam);
        workarea.nDxMin = -nPel * (workarea.x[0] - iY_H_Padding + nHPaddingScaled);
        workarea.nDyMin = -nPel * (workarea.y[0] - iY_V_Padding + nVPaddingScaled - nSearchParam); // if (- nSearchParam) not helps - need to think more.

        /* search the mv */
        workarea.predictor = ClipMV_SO2(workarea, vectors[workarea.blkIdx]);

        workarea.predictors[4] = ClipMV_SO2(workarea, zeroMV);

        workarea.bIntraframe = bInterframeH && bInterframeV;
        /*
                if (_predictorType == 0)
                  PseudoEPZSearch_optSO2<pixel_t>(workarea); // all predictors (original)
                else if (_predictorType == 1)
                  PseudoEPZSearch_optSO2_glob_med_pred<pixel_t>(workarea);
                else // _predictorType == 2
                  PseudoEPZSearch_optSO2_no_pred<pixel_t>(workarea);
                //      (this->*Sel_Pseudo_EPZ_search_SO2)(workarea); // still not works - maybe possible to fix ? */
        if (_predictorType == 1)
          PseudoEPZSearch_optSO4_glob_pred_avx512<pixel_t>(workarea, pBlkData);
        else // PT=2
          PseudoEPZSearch_optSO4_no_pred<pixel_t>(workarea, pBlkData);

        // workarea.bestMV = zeroMV; // debug

              /* write the results */
/*        pBlkData[workarea.blkx * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[workarea.blkx * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[workarea.blkx * N_PER_BLOCK + 2] = *(uint32_t*)(&workarea.bestMV.sad);
        */
        // 4 results written internally in Exa_search_4Blks()

        PROFILE_STOP(MOTION_PROFILE_ME);

        if (smallestPlane)
        {
          /*
          int64_t i64_1 = 0;
          int64_t i64_2 = 0;
          int32_t i32 = 0;
          unsigned int a1 = 200;
          unsigned int a2 = 201;

          i64_1 += a1 - a2; // 0x00000000 FFFFFFFF   !!!!!
          i64_2 = i64_2 + a1 - a2; // 0xFFFFFFFF FFFFFFFF O.K.!
          i32 += a1 - a2; // 0xFFFFFFFF
          */

          // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
          // LUMA diff can be negative! we should cast from uint32_t
          // 64 bit cast or else: int64_t += uint32t - uint32_t results in int64_t += (uint32_t)(uint32t - uint32_t)
          // which is baaaad 0x00000000 FFFFFFFF instead of 0xFFFFFFFF FFFFFFFF

          // 161204 todo check: why is it not abs(lumadiff)?
          typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
          workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
        }

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          workarea.x[0] += MAX_MULTI_BLOCKS_8x8_AVX512 * nBlkSizeX_Ovr[0] * workarea.blkScanDir;
          workarea.x[1] += MAX_MULTI_BLOCKS_8x8_AVX512 * nBlkSizeX_Ovr[1] * workarea.blkScanDir;
          workarea.x[2] += MAX_MULTI_BLOCKS_8x8_AVX512 * nBlkSizeX_Ovr[2] * workarea.blkScanDir;
        }
      }	// for iblkx
    }

    pBlkData += nBlkX * N_PER_BLOCK;

    workarea.y[0] += nBlkSizeY_Ovr[0];
    workarea.y[1] += nBlkSizeY_Ovr[1];
    workarea.y[2] += nBlkSizeY_Ovr[2];
  }	// for workarea.blky

  planeSAD += workarea.planeSAD; // for debug, plus fixme outer planeSAD is not used
  sumLumaChange += workarea.sumLumaChange;

  if (isse)
  {
#ifndef _M_X64
    _mm_empty();
#endif
  }

  _workarea_pool.return_obj(workarea);
} // search_mv_slice_SO4



template<typename pixel_t>
void	PlaneOfBlocks::recalculate_mv_slice(Slicer::TaskData &td)
{
  assert(&td != 0);

  short *outfilebuf = _outfilebuf;

  WorkingArea &	workarea = *(_workarea_pool.take_obj());
  assert(&workarea != 0);

  workarea.blky_beg = td._y_beg;
  workarea.blky_end = td._y_end;

  workarea.DCT = 0;
#ifdef ALLOW_DCT
  if (_dct_pool_ptr != 0)
  {
    workarea.DCT = _dct_pool_ptr->take_obj();
  }
#endif	// ALLOW_DCT
  // fixme: why here? search_mv is resetting it for inside each block scan
  // inside for (int iblkx = 0; iblkx < nBlkX; iblkx++)
  workarea.globalMVPredictor = _glob_mv_pred_def;

  int *pBlkData = _out + 1 + workarea.blky_beg * nBlkX*N_PER_BLOCK;
  if (outfilebuf != NULL)
  {
    outfilebuf += workarea.blky_beg * nBlkX * 4;// 4 short word per block
  }

  workarea.y[0] = pSrcFrame->GetPlane(YPLANE)->GetVPadding();
  workarea.y[0] += workarea.blky_beg * (nBlkSizeY - nOverlapY);

  if (chroma)
  {
      workarea.y[1] = pSrcFrame->GetPlane(UPLANE)->GetVPadding();
      workarea.y[2] = pSrcFrame->GetPlane(VPLANE)->GetVPadding();
      workarea.y[1] += workarea.blky_beg * ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
      workarea.y[2] += workarea.blky_beg * ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
  }

  workarea.planeSAD = 0; // for debug, plus fixme outer planeSAD is not used
  workarea.sumLumaChange = 0;

  // get old vectors plane
  const FakePlaneOfBlocks &plane = _mv_clip_ptr->GetPlane(0);
  int nBlkXold = plane.GetReducedWidth();
  int nBlkYold = plane.GetReducedHeight();
  int nBlkSizeXold = plane.GetBlockSizeX();
  int nBlkSizeYold = plane.GetBlockSizeY();
  int nOverlapXold = plane.GetOverlapX();
  int nOverlapYold = plane.GetOverlapY();
  int nStepXold = nBlkSizeXold - nOverlapXold;
  int nStepYold = nBlkSizeYold - nOverlapYold;
  int nPelold = plane.GetPel();
  int nLogPelold = ilog2(nPelold);

  int nBlkSizeXoldMulYold = nBlkSizeXold * nBlkSizeYold;
  int nBlkSizeXMulY = nBlkSizeX *nBlkSizeY;

  int nBlkSizeX_Ovr[3] = { (nBlkSizeX - nOverlapX), (nBlkSizeX - nOverlapX) >> nLogxRatioUV, (nBlkSizeX - nOverlapX) >> nLogxRatioUV };
  int nBlkSizeY_Ovr[3] = { (nBlkSizeY - nOverlapY), (nBlkSizeY - nOverlapY) >> nLogyRatioUV, (nBlkSizeY - nOverlapY) >> nLogyRatioUV };

  // 2.7.19.22
  // 32 bit safe: max_sad * (nBlkSizeX*nBlkSizeY) < 0x7FFFFFFF -> (3_planes*nBlkSizeX*nBlkSizeY*max_pixel_value) * (nBlkSizeX*nBlkSizeY) < 0x7FFFFFFF
  // 8 bit: nBlkSizeX*nBlkSizeY < sqrt(0x7FFFFFFF / 3 / 255), that is < sqrt(1675), above approx 40x40 is not OK even in 8 bits
  bool safeBlockAreaFor32bitCalc = nBlkSizeXMulY < 1675;

  // Functions using float must not be used here
  for (workarea.blky = workarea.blky_beg; workarea.blky < workarea.blky_end; workarea.blky++)
  {
    workarea.blkScanDir = (workarea.blky % 2 == 0 || !_meander_flag) ? 1 : -1;
    // meander (alternate) scan blocks (even row left to right, odd row right to left)
    int blkxStart = (workarea.blky % 2 == 0 || !_meander_flag) ? 0 : nBlkX - 1;
    if (workarea.blkScanDir == 1) // start with leftmost block
    {
      workarea.x[0] = pSrcFrame->GetPlane(YPLANE)->GetHPadding();
      if (chroma)
      {
        workarea.x[1] = pSrcFrame->GetPlane(UPLANE)->GetHPadding();
        workarea.x[2] = pSrcFrame->GetPlane(VPLANE)->GetHPadding();
      }
    }
    else // start with rightmost block, but it is already set at prev row
    {
      workarea.x[0] = pSrcFrame->GetPlane(YPLANE)->GetHPadding() + nBlkSizeX_Ovr[0]*(nBlkX - 1);
      if (chroma)
      {
        workarea.x[1] = pSrcFrame->GetPlane(UPLANE)->GetHPadding() + nBlkSizeX_Ovr[1]*(nBlkX - 1);
        workarea.x[2] = pSrcFrame->GetPlane(VPLANE)->GetHPadding() + nBlkSizeX_Ovr[2]*(nBlkX - 1);
      }
    }

    for (int iblkx = 0; iblkx < nBlkX; iblkx++)
    {
      workarea.blkx = blkxStart + iblkx*workarea.blkScanDir;
      workarea.blkIdx = workarea.blky*nBlkX + workarea.blkx;
      //		DebugPrintf("BlkIdx = %d \n", workarea.blkIdx);
      PROFILE_START(MOTION_PROFILE_ME);

#if (ALIGN_SOURCEBLOCK > 1)
      //store the pitch
      workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
      //create aligned copy
      BLITLUMA(workarea.pSrc_temp[0], nSrcPitch[0], workarea.pSrc[0], nSrcPitch_plane[0]);
      //set the to the aligned copy
      workarea.pSrc[0] = workarea.pSrc_temp[0];
      if (chroma)
      {
        workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
        workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
        BLITCHROMA(workarea.pSrc_temp[1], nSrcPitch[1], workarea.pSrc[1], nSrcPitch_plane[1]);
        BLITCHROMA(workarea.pSrc_temp[2], nSrcPitch[2], workarea.pSrc[2], nSrcPitch_plane[2]);
        workarea.pSrc[1] = workarea.pSrc_temp[1];
        workarea.pSrc[2] = workarea.pSrc_temp[2];
      }
#else	// ALIGN_SOURCEBLOCK
      workarea.pSrc[0] = pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(workarea.x[0], workarea.y[0]);
      if (chroma)
      {
        workarea.pSrc[1] = pSrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(workarea.x[1], workarea.y[1]);
        workarea.pSrc[2] = pSrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(workarea.x[2], workarea.y[2]);
      }
#endif	// ALIGN_SOURCEBLOCK

      if (workarea.blky == workarea.blky_beg)
      {
        workarea.nLambda = 0;
      }
      else
      {
        workarea.nLambda = _lambda_level;
      }

      // fixme:
      // not exacly nice, but works
      // different threads are writing, but the are the same always and come from parameters _pnew, _lsad
      penaltyNew = _pnew; // penalty for new vector
      LSAD = _lsad;    // SAD limit for lambda using
      // may be they must be scaled by nPel ?

      /* computes search boundaries */
      workarea.nDxMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedWidth() - workarea.x[0] - nBlkSizeX);
      workarea.nDyMax = nPel * (pSrcFrame->GetPlane(YPLANE)->GetExtendedHeight() - workarea.y[0] - nBlkSizeY);
      workarea.nDxMin = -nPel * workarea.x[0];
      workarea.nDyMin = -nPel * workarea.y[0];

      // get and interplolate old vectors
      int centerX = nBlkSizeX / 2 + (nBlkSizeX - nOverlapX)*workarea.blkx; // center of new block
      int blkxold = (centerX - nBlkSizeXold / 2) / nStepXold; // centerXold less or equal to new
      int centerY = nBlkSizeY / 2 + (nBlkSizeY - nOverlapY)*workarea.blky;
      int blkyold = (centerY - nBlkSizeYold / 2) / nStepYold;

      int deltaX = std::max(0, centerX - (nBlkSizeXold / 2 + nStepXold*blkxold)); // distance from old to new
      int deltaY = std::max(0, centerY - (nBlkSizeYold / 2 + nStepYold*blkyold));

      int blkxold1 = std::min(nBlkXold - 1, std::max(0, blkxold));
      int blkxold2 = std::min(nBlkXold - 1, std::max(0, blkxold + 1));
      int blkyold1 = std::min(nBlkYold - 1, std::max(0, blkyold));
      int blkyold2 = std::min(nBlkYold - 1, std::max(0, blkyold + 1));

      VECTOR vectorOld; // interpolated or nearest

      typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

      if (_smooth == 1) // interpolate
      {
        VECTOR vectorOld1 = _mv_clip_ptr->GetBlock(0, blkxold1 + blkyold1*nBlkXold).GetMV(); // 4 old nearest vectors (may coinside)
        VECTOR vectorOld2 = _mv_clip_ptr->GetBlock(0, blkxold2 + blkyold1*nBlkXold).GetMV();
        VECTOR vectorOld3 = _mv_clip_ptr->GetBlock(0, blkxold1 + blkyold2*nBlkXold).GetMV();
        VECTOR vectorOld4 = _mv_clip_ptr->GetBlock(0, blkxold2 + blkyold2*nBlkXold).GetMV();

        // interpolate
        int vector1_x = vectorOld1.x*nStepXold + deltaX*(vectorOld2.x - vectorOld1.x); // scaled by nStepXold to skip slow division
        int vector1_y = vectorOld1.y*nStepXold + deltaX*(vectorOld2.y - vectorOld1.y);
        safe_sad_t vector1_sad = (safe_sad_t)vectorOld1.sad*nStepXold + deltaX*((safe_sad_t)vectorOld2.sad - vectorOld1.sad);

        int vector2_x = vectorOld3.x*nStepXold + deltaX*(vectorOld4.x - vectorOld3.x);
        int vector2_y = vectorOld3.y*nStepXold + deltaX*(vectorOld4.y - vectorOld3.y);
        safe_sad_t vector2_sad = (safe_sad_t)vectorOld3.sad*nStepXold + deltaX*((safe_sad_t)vectorOld4.sad - vectorOld3.sad);

        vectorOld.x = (vector1_x + deltaY*(vector2_x - vector1_x) / nStepYold) / nStepXold;
        vectorOld.y = (vector1_y + deltaY*(vector2_y - vector1_y) / nStepYold) / nStepXold;
        vectorOld.sad = (sad_t)((vector1_sad + deltaY*(vector2_sad - vector1_sad) / nStepYold) / nStepXold);
      }

      else // nearest
      {
        if (deltaX * 2 < nStepXold && deltaY * 2 < nStepYold)
        {
          vectorOld = _mv_clip_ptr->GetBlock(0, blkxold1 + blkyold1*nBlkXold).GetMV();
        }
        else if (deltaX * 2 >= nStepXold && deltaY * 2 < nStepYold)
        {
          vectorOld = _mv_clip_ptr->GetBlock(0, blkxold2 + blkyold1*nBlkXold).GetMV();
        }
        else if (deltaX * 2 < nStepXold && deltaY * 2 >= nStepYold)
        {
          vectorOld = _mv_clip_ptr->GetBlock(0, blkxold1 + blkyold2*nBlkXold).GetMV();
        }
        else //(deltaX*2>=nStepXold && deltaY*2>=nStepYold )
        {
          vectorOld = _mv_clip_ptr->GetBlock(0, blkxold2 + blkyold2*nBlkXold).GetMV();
        }
      }

      // scale vector to new nPel
      vectorOld.x = (vectorOld.x << nLogPel) >> nLogPelold;
      vectorOld.y = (vectorOld.y << nLogPel) >> nLogPelold;

      workarea.predictor = ClipMV(workarea, vectorOld); // predictor
      if(safeBlockAreaFor32bitCalc && sizeof(pixel_t)==1)
        workarea.predictor.sad = (sad_t)((safe_sad_t)vectorOld.sad * nBlkSizeXMulY / nBlkSizeXoldMulYold); // normalized to new block size
      else // 16 bit or unsafe blocksize
        workarea.predictor.sad = (sad_t)((bigsad_t)vectorOld.sad * nBlkSizeXMulY / nBlkSizeXoldMulYold); // normalized to new block size

//			workarea.bestMV = workarea.predictor; // by pointer?
      workarea.bestMV.x = workarea.predictor.x;
      workarea.bestMV.y = workarea.predictor.y;
      workarea.bestMV.sad = workarea.predictor.sad;

      // update SAD
#ifdef ALLOW_DCT
      if (dctmode != 0) // DCT method (luma only - currently use normal spatial SAD chroma)
      {
        // make dct of source block
        if (dctmode <= 4) //don't do the slow dct conversion if SATD used
        {
          workarea.DCT->DCTBytes2D(workarea.pSrc[0], nSrcPitch[0], &workarea.dctSrc[0], dctpitch);
        }
      }
      if (dctmode >= 3) // most use it and it should be fast anyway //if (dctmode == 3 || dctmode == 4) // check it
      {
        workarea.srcLuma = LUMA(workarea.pSrc[0], nSrcPitch[0]);
      }
#endif	// ALLOW_DCT

/*      sad_t saduv = (chroma) ? ScaleSadChroma_f(SADCHROMA(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
        + SADCHROMA(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;*/
      sad_t saduv = (chroma) ? ScaleSadChroma_f(DM_Chroma->GetDisMetric(workarea.pSrc[1], nSrcPitch[1], GetRefBlockU(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[1])
        + DM_Chroma->GetDisMetric(workarea.pSrc[2], nSrcPitch[2], GetRefBlockV(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[2]), effective_chromaSADscale, scaleCSADfine) : 0;
      sad_t sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
      sad += saduv;
      workarea.bestMV.sad = sad;
      workarea.nMinCost = sad;

      if (workarea.bestMV.sad > _thSAD)// if old interpolated vector is bad
      {
        //				CheckMV(vectorOld1.x, vectorOld1.y);
        //				CheckMV(vectorOld2.x, vectorOld2.y);
        //				CheckMV(vectorOld3.x, vectorOld3.y);
        //				CheckMV(vectorOld4.x, vectorOld4.y);
                // then, we refine, according to the search type

        // todo PF: consider switch and not bitfield searchType
        if (searchType & ONETIME)
        {
          for (int i = nSearchParam; i > 0; i /= 2)
          {
            OneTimeSearch<pixel_t>(workarea, i);
          }
        }

        if (searchType & NSTEP)
        {
          NStepSearch<pixel_t>(workarea, nSearchParam);
        }

        if (searchType & LOGARITHMIC)
        {
          for (int i = nSearchParam; i > 0; i /= 2)
          {
            DiamondSearch<pixel_t>(workarea, i);
          }
        }

        if (searchType & EXHAUSTIVE)
        {
          //       ExhaustiveSearch(nSearchParam);
          int mvx = workarea.bestMV.x;
          int mvy = workarea.bestMV.y;
          for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
          {
            ExpandingSearch<pixel_t>(workarea, i, 1, mvx, mvy);
          }
        }

        if (searchType & HEX2SEARCH)
        {
          Hex2Search<pixel_t>(workarea, nSearchParam);
        }

        if (searchType & UMHSEARCH)
        {
          UMHSearch<pixel_t>(workarea, nSearchParam, workarea.bestMV.x, workarea.bestMV.y);
        }

        if (searchType & HSEARCH)
        {
          int mvx = workarea.bestMV.x;
          int mvy = workarea.bestMV.y;
          for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
          {
            CheckMV<pixel_t>(workarea, mvx - i, mvy);
            CheckMV<pixel_t>(workarea, mvx + i, mvy);
          }
        }

        if (searchType & VSEARCH)
        {
          int mvx = workarea.bestMV.x;
          int mvy = workarea.bestMV.y;
          for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
          {
            CheckMV<pixel_t>(workarea, mvx, mvy - i);
            CheckMV<pixel_t>(workarea, mvx, mvy + i);
          }
        }
      }	// if bestMV.sad > thSAD

      // we store the result
      vectors[workarea.blkIdx].x = workarea.bestMV.x;
      vectors[workarea.blkIdx].y = workarea.bestMV.y;
      vectors[workarea.blkIdx].sad = workarea.bestMV.sad;

      if (outfilebuf != NULL) // write vector to outfile
      {
        outfilebuf[workarea.blkx * 4 + 0] = workarea.bestMV.x;
        outfilebuf[workarea.blkx * 4 + 1] = workarea.bestMV.y;
        outfilebuf[workarea.blkx * 4 + 2] = (workarea.bestMV.sad & 0x0000ffff); // low word
        outfilebuf[workarea.blkx * 4 + 3] = (workarea.bestMV.sad >> 16);     // high word, usually null
      }

      /* write the results */
      pBlkData[workarea.blkx*N_PER_BLOCK + 0] = workarea.bestMV.x;
      pBlkData[workarea.blkx*N_PER_BLOCK + 1] = workarea.bestMV.y;
      pBlkData[workarea.blkx*N_PER_BLOCK + 2] = workarea.bestMV.sad;


      PROFILE_STOP(MOTION_PROFILE_ME);

      if (smallestPlane)
      {
        // int64_t += uint32_t - uint32_t is not ok, if diff would be negative
        // 161204 todo check: why is it not abs(lumadiff)?
        workarea.sumLumaChange += (safe_sad_t)LUMA(GetRefBlock(workarea, 0, 0), nRefPitch[0]) - (safe_sad_t)LUMA(workarea.pSrc[0], nSrcPitch[0]);
      }

      if (iblkx < nBlkX - 1)
      {
        workarea.x[0] += nBlkSizeX_Ovr[0]*workarea.blkScanDir;
        workarea.x[1] += nBlkSizeX_Ovr[1]*workarea.blkScanDir;
        workarea.x[2] += nBlkSizeX_Ovr[2]*workarea.blkScanDir;
      }
    }	// for workarea.blkx

    pBlkData += nBlkX*N_PER_BLOCK;
    if (outfilebuf != NULL) // write vector to outfile
    {
      outfilebuf += nBlkX * 4;// 4 short word per block
    }

    workarea.y[0] += nBlkSizeY_Ovr[0];
    workarea.y[1] += nBlkSizeY_Ovr[1];
    workarea.y[2] += nBlkSizeY_Ovr[2];
  }	// for workarea.blky

  planeSAD += workarea.planeSAD; // for debug, plus fixme outer planeSAD is not used
  sumLumaChange += workarea.sumLumaChange;

  if (isse)
  {
#ifndef _M_X64
    _mm_empty();
#endif
  }

#ifdef ALLOW_DCT
  if (_dct_pool_ptr != 0)
  {
    _dct_pool_ptr->return_obj(*(workarea.DCT));
    workarea.DCT = 0;
  }
#endif

  _workarea_pool.return_obj(workarea);
} // recalculate_mv_slice



// -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -



PlaneOfBlocks::WorkingArea::WorkingArea(int nBlkSizeX, int nBlkSizeY, int dctpitch, int nLogxRatioUV, int nLogyRatioUV, int _pixelsize, int _bits_per_pixel)
  : dctSrc(nBlkSizeY*dctpitch) // dctpitch is pixelsize aware
  , DCT(0)
  , dctRef(nBlkSizeY* dctpitch)
  , pixelsize(_pixelsize)
  , bits_per_pixel(_bits_per_pixel)
{
#if (ALIGN_SOURCEBLOCK > 1)
  int xPitch = AlignNumber(nBlkSizeX*pixelsize, ALIGN_SOURCEBLOCK);  // for memory allocation pixelsize needed
  int xPitchUV = AlignNumber((nBlkSizeX*pixelsize) >> nLogxRatioUV, ALIGN_SOURCEBLOCK);
  int blocksize = xPitch*nBlkSizeY;
  int UVblocksize = xPitchUV * (nBlkSizeY >> nLogyRatioUV); // >> nx >> ny
  int sizeAlignedBlock = blocksize + 2 * UVblocksize;
  // int sizeAlignedBlock=blocksize+(ALIGN_SOURCEBLOCK-(blocksize%ALIGN_SOURCEBLOCK))+
  //                         2*((blocksize/2)>>nLogyRatioUV)+(ALIGN_SOURCEBLOCK-(((blocksize/2)/yRatioUV)%ALIGN_SOURCEBLOCK)); // why >>Logy then /y?
  pSrc_temp_base.resize(sizeAlignedBlock);
  pSrc_temp[0] = &pSrc_temp_base[0];
  pSrc_temp[1] = (uint8_t *)(pSrc_temp[0] + blocksize);
  pSrc_temp[2] = (uint8_t *)(pSrc_temp[1] + UVblocksize);
#endif	// ALIGN_SOURCEBLOCK
}



PlaneOfBlocks::WorkingArea::~WorkingArea()
{
  // Nothing
}


/* computes the cost of a vector (vx, vy) */
template<typename pixel_t>
MV_FORCEINLINE sad_t PlaneOfBlocks::WorkingArea::MotionDistorsion(int vx, int vy) const
{
  int dist = SquareDifferenceNorm(predictor, vx, vy);
  if constexpr(sizeof(pixel_t) == 1)
  {
#if 0
    // 20181018 PF: hope it'll not overflow, could not produce such case
    // left here for future tests
    if (dist != 0) {
      if (nLambda > std::numeric_limits<int>::max() / dist)
      {
        int x = 0;
        _RPT1(0, "Lambda is over! %d\r\n", nLambda);
      }
    }
#endif
    return (nLambda * dist) >> 8; // 8 bit: faster
  }
  else
    return (nLambda * dist) >> (16 - bits_per_pixel) /*8*/; // PF scaling because it appears as a sad addition 
    // nLambda itself is bit-depth independent, but blocksize-dependent. 
    // For historical reasons this parameter is not normalized, 
    // Caller have to scale it properly by the actual block sizes, like MAnalyze does it when truemotion=true: 1000*blocksize*blocksizeV/64.
    // Because of the bit-depth scaling above the parameter is at least bit-depth independent.
    // To have lambda blocksize independent, it should be scaled in MvAnalyze and MvRecalculate but it kills compatibility.
}

/* computes the length cost of a vector (vx, vy) */
//MV_FORCEINLINE int LengthPenalty(int vx, int vy)
//{
//	return ( (vx*vx + vy*vy)*nLambdaLen) >> 8;
//}



PlaneOfBlocks::WorkingAreaFactory::WorkingAreaFactory(int nBlkSizeX, int nBlkSizeY, int dctpitch, int nLogxRatioUV, int nLogyRatioUV, int pixelsize, int bits_per_pixel)
  : _blk_size_x(nBlkSizeX)
  , _blk_size_y(nBlkSizeY)
  , _dctpitch(dctpitch)
  , _x_ratio_uv_log(nLogxRatioUV)
  , _y_ratio_uv_log(nLogyRatioUV)
  , _pixelsize(pixelsize)
  , _bits_per_pixel(bits_per_pixel)
{
  // Nothing
}



PlaneOfBlocks::WorkingArea *PlaneOfBlocks::WorkingAreaFactory::do_create()
{
  return (new WorkingArea(
    _blk_size_x,
    _blk_size_y,
    _dctpitch,
    _x_ratio_uv_log,
    _y_ratio_uv_log,
    _pixelsize,
    _bits_per_pixel
  ));
}


PlaneOfBlocks::ExhaustiveSearchFunction_t PlaneOfBlocks::get_ExhaustiveSearchFunction(int BlockX, int BlockY, int SearchParam, int _bits_per_pixel, arch_t arch)
{

  // BlkSizeX, BlkSizeY, bits_per_pixel, arch_t
  std::map<std::tuple<int, int, int, int, arch_t>, ExhaustiveSearchFunction_t> func_fn;

  //TODO: add nPel here too ? It need separate functions for nPel=1 and other nPel.

  // SearchParam 1 or 2 or 4 is supported at the moment
  func_fn[std::make_tuple(8, 8, 1, 8, USE_AVX512)] = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp1_avx512;
  func_fn[std::make_tuple(8, 8, 1, 8, USE_AVX2)] = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp1_avx2;
  func_fn[std::make_tuple(8, 8, 1, 8, NO_SIMD)] = &PlaneOfBlocks::ExhaustiveSearch_uint8_sp1_c;
  func_fn[std::make_tuple(16, 16, 1, 8, USE_AVX512)] = &PlaneOfBlocks::ExhaustiveSearch16x16_uint8_np1_sp1_avx512;
  func_fn[std::make_tuple(16, 16, 1, 8, USE_AVX2)] = &PlaneOfBlocks::ExhaustiveSearch16x16_uint8_np1_sp1_avx2;
  func_fn[std::make_tuple(16, 16, 1, 8, NO_SIMD)] = &PlaneOfBlocks::ExhaustiveSearch_uint8_sp1_c;
  func_fn[std::make_tuple(8, 8, 2, 8, USE_AVX2)] = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp2_avx2;
  func_fn[std::make_tuple(8, 8, 2, 8, NO_SIMD)] = &PlaneOfBlocks::ExhaustiveSearch_uint8_sp2_c;
  func_fn[std::make_tuple(16, 16, 2, 8, USE_AVX2)] = &PlaneOfBlocks::ExhaustiveSearch16x16_uint8_np1_sp2_avx2;
  func_fn[std::make_tuple(16, 16, 2, 8, NO_SIMD)] = &PlaneOfBlocks::ExhaustiveSearch_uint8_sp2_c;
  func_fn[std::make_tuple(8, 8, 3, 8, USE_AVX2)] = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp3_avx2;
  func_fn[std::make_tuple(8, 8, 3, 8, NO_SIMD)] = &PlaneOfBlocks::ExhaustiveSearch_uint8_sp3_c;
  func_fn[std::make_tuple(8, 8, 4, 8, USE_AVX2)] = &PlaneOfBlocks::ExhaustiveSearch8x8_uint8_np1_sp4_avx2;
  func_fn[std::make_tuple(8, 8, 4, 8, NO_SIMD)] = &PlaneOfBlocks::ExhaustiveSearch_uint8_sp4_c;

  ExhaustiveSearchFunction_t result = nullptr;
  arch_t archlist[] = { USE_AVX512, USE_AVX2, USE_AVX, USE_SSE41, USE_SSE2, NO_SIMD };
  int index = 0;
  while (result == nullptr) {
    arch_t current_arch_try = archlist[index++];
    if (current_arch_try > arch) continue;
    result = func_fn[std::make_tuple(BlockX, BlockY, SearchParam, _bits_per_pixel, current_arch_try)];

    if (result == nullptr && current_arch_try == NO_SIMD) {
      break;
    }
  }

  return result;
}


void PlaneOfBlocks::ExhaustiveSearch_uint8_sp4_c(WorkingArea& workarea, int mvx, int mvy) // exa search radius 4
{
  // debug check !
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 4, mvy - 4))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 3, mvy + 4)) // 8 positions only so -4..+3.
  {
    return;
  }
  unsigned short minsad = 65535;
  int x_minsad = 0;
  int y_minsad = 0;
  for (int y = 3; y > -5; y--) // reversed scan to match _mm_minpos() logic ?
  {
    for (int x = 4; x > -5; x--)
    {
      int sad = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlock(workarea, mvx + x, mvy + y), nRefPitch[0]);
      if (sad < minsad)
      {
        minsad = sad;
        x_minsad = x;
        y_minsad = y;
      }
    }
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost) return;

  workarea.bestMV.x = mvx + x_minsad;
  workarea.bestMV.y = mvy + y_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

}

// Dispatcher for DTL tests
void PlaneOfBlocks::ExhaustiveSearch_uint8_sp3_c(WorkingArea& workarea, int mvx, int mvy) // exa search radius 3,  works for any nPel !.
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

  unsigned short minsad = 65535;
  int x_minsad = 0;
  int y_minsad = 0;
  for (int y = 3; y > -4; y--)
  {
    for (int x = 3; x > -4; x--)
    {
      int sad = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlock(workarea, mvx + x, mvy + y), nRefPitch[0]);
      if (sad < minsad)
      {
        minsad = sad;
        x_minsad = x;
        y_minsad = y;
      }
    }
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost) return;

  workarea.bestMV.x = mvx + x_minsad;
  workarea.bestMV.y = mvy + y_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

}


void PlaneOfBlocks::ExhaustiveSearch_uint8_sp2_c(WorkingArea& workarea, int mvx, int mvy) // exa search radius 2,  works for any nPel !.
{
  // debug check !
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 2, mvy - 2))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 2, mvy + 2))
  {
    return;
  }

  unsigned short minsad = 65535;
  int x_minsad = 0;
  int y_minsad = 0;
  for (int y = 2; y > -3; y--)
  {
    for (int x = 2; x > -3; x--)
    {
      int sad = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlock(workarea, mvx + x, mvy + y), nRefPitch[0]);
      if (sad < minsad)
      {
        minsad = sad;
        x_minsad = x;
        y_minsad = y;
      }
    }
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost) return;

  workarea.bestMV.x = mvx + x_minsad;
  workarea.bestMV.y = mvy + y_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

}

void PlaneOfBlocks::ExhaustiveSearch_uint8_sp1_c(WorkingArea& workarea, int mvx, int mvy) // exa search radius 1, works for any nPel !, and any block size
{
  // debug check !
  // idea - may be not 4 checks are required - only upper left corner (starting addresses of buffer) and lower right (to not over-run atfer end of buffer - need check/test)
  if (!workarea.IsVectorOK(mvx - 1, mvy - 1))
  {
    return;
  }
  if (!workarea.IsVectorOK(mvx + 1, mvy + 1))
  {
    return;
  }

  unsigned short minsad = 65535;
  int x_minsad = 0;
  int y_minsad = 0;
  for (int y = 1; y > -2; y--)
  {
    for (int x = 1; x > -2; x--)
    {
      int sad = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlock(workarea, mvx + x, mvy + y), nRefPitch[0]);
      if (sad < minsad)
      {
        minsad = sad;
        x_minsad = x;
        y_minsad = y;
      }
    }
  }

  sad_t cost = minsad + ((penaltyNew * minsad) >> 8);
  if (cost >= workarea.nMinCost) return;

  workarea.bestMV.x = mvx + x_minsad;
  workarea.bestMV.y = mvy + y_minsad;
  workarea.nMinCost = cost;
  workarea.bestMV.sad = minsad;

}

// MV_Vector
template<class T>
MVVector<T>::MVVector()
{
  my_size = 0;
  buffer = 0;
  size_bytes = 0;
}

template<class T>
MVVector<T>::~MVVector()
{
  my_size = 0;
  buffer = 0;
  size_bytes = 0;
#ifdef _WIN32
  if (buffer != 0)
    VirtualFree(buffer, 0, MEM_RELEASE);
#else
  delete[]buffer;
#endif
}


template<class T>
MVVector<T>::MVVector(size_t size, IScriptEnvironment* env)
{
  my_size = size;
#ifdef _WIN32
  DWORD error;
  // large pages - allocate if possible, if error - silent fallback to standard 4kB pages
  SIZE_T stLPGranularity = GetLargePageMinimum();
  SIZE_T NumLPUnits = ((size * sizeof(VECTOR)) / stLPGranularity) + 1;
  SIZE_T stSizeToAlloc = NumLPUnits * stLPGranularity;
  buffer = (T*)VirtualAlloc(0, stSizeToAlloc, MEM_LARGE_PAGES | MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  error = GetLastError();

  size_bytes = stSizeToAlloc;

  if (buffer == 0)
  {
    // may be enable in debug ?
//    env->ThrowError("MVVector LargePages alloc error. While allocating %d pages of %d size, GetLastError returned: %d\n", NumLPUnits, stLPGranularity, error);
    // standart 4kB pages
    buffer = (T*)VirtualAlloc(0, size * sizeof(VECTOR), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    error = GetLastError();

    size_bytes = size * sizeof(VECTOR);

    if (buffer == 0)
    {
      // major error - need to break
      env->ThrowError("MVVector VirtualAlloc 4 kB pages alloc error. GetLastError returned: %d\n", error);
    }
  }

#else //Linux ?
  buffer = new T[size];
#endif
}

template<class T>
size_t MVVector<T>::size()const
{
  return my_size;
}

template<class T>
T& MVVector<T>::operator[](size_t index) // need to add something for 'const' to make working 'const' version of InterpolatePredictors ?
{
  return buffer[index];
}

// linker do not see it in _avx2 file ???
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

// test of AVX2 IsVectorChecked - for future inlining 
/*MV_FORCEINLINE bool PlaneOfBlocks::IsVectorChecked_avx2(uint32_t xy)
{
  const __m256i ymm_shift_4 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);

  __m256i ymm_to_check = _mm256_broadcastd_epi32(_mm_cvtsi32_si128(xy));

  int iCmp_res = _mm256_movemask_epi8(_mm256_cmpeq_epi32(ymm_checked_mv_vectors, ymm_to_check));
  if (iCmp_res != 0)
  {
    return true;
  }

  // add new xy
  ymm_checked_mv_vectors = _mm256_permutevar8x32_epi32(ymm_checked_mv_vectors, ymm_shift_4);
  ymm_checked_mv_vectors = _mm256_blend_epi32(ymm_checked_mv_vectors, ymm_to_check, 1);

}*/



template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO2_8x8_avx2(WorkingArea& workarea) // + predictorType=1
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;

  __m128i xmm8_Ref0, xmm9_Ref1, xmm10_Ref2, xmm11_Ref3, xmm12_Ref4, xmm13_Ref5, xmm14_Ref6, xmm15_Ref7;

  const uint8_t* pucCurr = (uint8_t*)workarea.pSrc[0];

  uint8_t* pucRef;

  const __m128i xmm0_Src0 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 0));
  const __m128i xmm1_Src1 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 1));
  const __m128i xmm2_Src2 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 2));
  const __m128i xmm3_Src3 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 3));
  const __m128i xmm4_Src4 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 4));
  const __m128i xmm5_Src5 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 5));
  const __m128i xmm6_Src6 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 6));
  const __m128i xmm7_Src7 = _mm_loadl_epi64((__m128i*)(pucCurr + nSrcPitch[0] * 7));

  if (workarea.bIntraframe)
  {
    FetchPredictors_avx2_intraframe<pixel_t>(workarea); // faster
  }
  else
  {
    FetchPredictors_sse41<pixel_t>(workarea);
  }
  
  sad_t sad;
  sad_t cost;

  // We treat zero alone
  // Do we bias zero with not taking into account distorsion ?
  workarea.bestMV.x = zeroMVfieldShifted.x;
  workarea.bestMV.y = zeroMVfieldShifted.y;
//    sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, 0, zeroMVfieldShifted.y));
  pucRef = (uint8_t*)GetRefBlock(workarea, 0, zeroMVfieldShifted.y);

  sad_block_8x8
  sad = _mm_cvtsi128_si32(xmm8_Ref0);

  workarea.bestMV.sad = sad;
  workarea.nMinCost = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2

  iNumCheckedVectors = 0;
  checked_mv_vectors[iNumCheckedVectors] = 0;
  iNumCheckedVectors++;

  // Global MV predictor  - added by Fizick
  workarea.globalMVPredictor = ClipMV_SO2(workarea, workarea.globalMVPredictor);

  if (!IsVectorChecked((uint64_t)workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32)))
  {
    //    sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y));
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y);

    sad_block_8x8
      sad = _mm_cvtsi128_si32(xmm8_Ref0);

    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = workarea.globalMVPredictor.x;
      workarea.bestMV.y = workarea.globalMVPredictor.y;
      workarea.bestMV.sad = sad;
      workarea.nMinCost = cost;
    }
  }
  //	}
  //	Then, the predictor :
  //	if (   (( workarea.predictor.x != zeroMVfieldShifted.x ) || ( workarea.predictor.y != zeroMVfieldShifted.y ))
  //	    && (( workarea.predictor.x != workarea.globalMVPredictor.x ) || ( workarea.predictor.y != workarea.globalMVPredictor.y )))
  //	{
  if (!IsVectorChecked((uint64_t)workarea.predictor.x | ((uint64_t)workarea.predictor.y << 32)))
  {
    //    sad = LumaSAD<pixel_t>(workarea, GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y));
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y);

    sad_block_8x8
      cost = _mm_cvtsi128_si32(xmm8_Ref0);

    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x = workarea.predictor.x;
      workarea.bestMV.y = workarea.predictor.y;
      workarea.bestMV.sad = cost;
      workarea.nMinCost = cost;
    }
  }
  // then all the other predictors
  // compute checks on motion distortion first and skip MV if above cost:
  if (_predictorType == 0) // all predictors, combine PT=0 and PT=1 in one function, some minor performnce penalty ?
  {
    // MotionDistortion SIMD calculation for all 4 predictors
    __m256i ymm2_yx_predictors = _mm256_set_epi32(workarea.predictors[3].y, workarea.predictors[3].x, workarea.predictors[2].y, workarea.predictors[2].x, \
      workarea.predictors[1].y, workarea.predictors[1].x, workarea.predictors[0].y, workarea.predictors[0].x);
    __m256i ymm3_predictor = _mm256_broadcastq_epi64(_mm_set_epi32(0, 0, workarea.predictor.y, workarea.predictor.x)); // hope movq + vpbroadcast

    __m256i ymm_d1d2 = _mm256_sub_epi32(ymm3_predictor, ymm2_yx_predictors);
    ymm_d1d2 = _mm256_add_epi32(_mm256_mullo_epi32(ymm_d1d2, ymm_d1d2), _mm256_srli_si256(ymm_d1d2, 4));

    __m256i ymm_dist = _mm256_permutevar8x32_epi32(ymm_d1d2, _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0));
    __m128i xmm_nLambda = _mm_set1_epi32(workarea.nLambda);
    __m128i  xmm0_cost = _mm_srli_epi32(_mm_mullo_epi32(xmm_nLambda, _mm256_castsi256_si128(ymm_dist)), 8);
    __m128i xmm_mask = _mm_cmplt_epi32(xmm0_cost, _mm_set1_epi32(workarea.nMinCost));
    int iMask = _mm_movemask_epi8(xmm_mask);
    _mm256_zeroupper(); // need ?

    // if ((iMask & 0x1111) == 0x1111) - use 4-predictors CheckMV0_avx2() - to do.
    // vectors were clipped in FetchPredictors - no new IsVectorOK() check ?
    if ((iMask & 0x1) != 0)
    {
      if (!IsVectorChecked((uint64_t)workarea.predictors[0].x | ((uint64_t)workarea.predictors[0].y << 32)))
      {
        //      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[0].x, workarea.predictors[0].y, _mm_extract_epi32(xmm0_cost, 0));
        cost = _mm_extract_epi32(xmm0_cost, 0);

        pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictors[0].x, workarea.predictors[0].y);
        sad_block_8x8
          sad = _mm_cvtsi128_si32(xmm8_Ref0);

        cost += sad;

        if (cost < workarea.nMinCost)
        {
          workarea.bestMV.x = workarea.predictors[0].x;
          workarea.bestMV.y = workarea.predictors[0].y;
          workarea.nMinCost = cost;
          workarea.bestMV.sad = sad;
        }
      }
    }
    if ((iMask & 0x10) != 0)
    {
      if (!IsVectorChecked((uint64_t)workarea.predictors[1].x | ((uint64_t)workarea.predictors[1].y << 32)))
      {
        //      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[1].x, workarea.predictors[1].y, _mm_extract_epi32(xmm0_cost, 1));
        cost = _mm_extract_epi32(xmm0_cost, 1);

        pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictors[1].x, workarea.predictors[1].y);
        sad_block_8x8
          sad = _mm_cvtsi128_si32(xmm8_Ref0);

        cost += sad;

        if (cost < workarea.nMinCost)
        {
          workarea.bestMV.x = workarea.predictors[1].x;
          workarea.bestMV.y = workarea.predictors[1].y;
          workarea.nMinCost = cost;
          workarea.bestMV.sad = sad;
        }

      }
    }
    if ((iMask & 0x100) != 0)
    {
      if (!IsVectorChecked((uint64_t)workarea.predictors[2].x | ((uint64_t)workarea.predictors[2].y << 32)))
      {
        //      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[2].x, workarea.predictors[2].y, _mm_extract_epi32(xmm0_cost, 2));
        cost = _mm_extract_epi32(xmm0_cost, 2);

        pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictors[2].x, workarea.predictors[2].y);
        sad_block_8x8
          sad = _mm_cvtsi128_si32(xmm8_Ref0);

        cost += sad;

        if (cost < workarea.nMinCost)
        {
          workarea.bestMV.x = workarea.predictors[2].x;
          workarea.bestMV.y = workarea.predictors[2].y;
          workarea.nMinCost = cost;
          workarea.bestMV.sad = sad;
        }

      }
    }
    if ((iMask & 0x1000) != 0)
    {
      if (!IsVectorChecked((uint64_t)workarea.predictors[3].x | ((uint64_t)workarea.predictors[3].y << 32)))
      {
        //      CheckMV0_SO2<pixel_t>(workarea, workarea.predictors[3].x, workarea.predictors[3].y, _mm_extract_epi32(xmm0_cost, 3));
        cost = _mm_extract_epi32(xmm0_cost, 3);

        pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictors[3].x, workarea.predictors[3].y);
        sad_block_8x8
          sad = _mm_cvtsi128_si32(xmm8_Ref0);

        cost += sad;

        if (cost < workarea.nMinCost)
        {
          workarea.bestMV.x = workarea.predictors[3].x;
          workarea.bestMV.y = workarea.predictors[3].y;
          workarea.nMinCost = cost;
          workarea.bestMV.sad = sad;
        }

      }
    }
  } // if (_optPredictorType == 0)

  // then, we refine, 
  // sp = 1 for level=0 (finest) sp = 2 for other levels
//  (this->*ExhaustiveSearch_SO2)(workarea, workarea.bestMV.x, workarea.bestMV.y);
 // const __m256i ymm4_Src_01, ymm5_Src_23, ymm6_Src_45, ymm7_Src_67;
  const __m256i ymm4_Src_01 = _mm256_set_m128i(xmm1_Src1, xmm0_Src0);
  const __m256i ymm5_Src_23 = _mm256_set_m128i(xmm3_Src3, xmm2_Src2);
  const __m256i ymm6_Src_45 = _mm256_set_m128i(xmm5_Src5, xmm4_Src4);
  const __m256i ymm7_Src_67 = _mm256_set_m128i(xmm7_Src7, xmm6_Src6);


  __m256i ymm0_Ref_01, ymm1_Ref_23, ymm2_Ref_45, ymm3_Ref_67; // require buf padding to allow 16bytes reads to xmm
  __m256i ymm10_sads_r0, ymm11_sads_r1, ymm12_sads_r2, ymm13_sads_r3, ymm14_sads_r4;

  const __m256i ymm13_all_ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());

  __m256i ymm_block_ress;

  if (nSearchParam == 1)
  {
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.bestMV.x - 1, workarea.bestMV.y - 1); // upper left corner

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
    if (cost < workarea.nMinCost)
    {
      workarea.bestMV.x += dx_minsad;
      workarea.bestMV.y += dy_minsad;
      workarea.nMinCost = cost;
      workarea.bestMV.sad = minsad;
    }

  }
  else if (nSearchParam == 2)
  {
    // sp2 here
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.bestMV.x - 2, workarea.bestMV.y - 2); // upper left corner

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
    if (cost < workarea.nMinCost)
    {

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

      workarea.bestMV.x += dx_minsad;
      workarea.bestMV.y += dy_minsad;
      workarea.nMinCost = cost;
      workarea.bestMV.sad = minsad;

    }
  }

  // we store the result
  vectors[workarea.blkIdx] = workarea.bestMV;

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used

  _mm256_zeroupper(); // may be not needed ?
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO3_glob_pred_avx2(WorkingArea& workarea, int* pBlkData)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
  sad_t sad, cost;
  VECTOR_XY vectors_coh_check[MAX_MULTI_BLOCKS_8x8_AVX2];

  __m256i ymm8_r1, ymm9_r2, ymm10_r3, ymm11_r4, ymm12_r5, ymm13_r6, ymm14_r7, ymm15_r8;
  /*
    ymm8_r1 = _mm256_sad_epu8(ymm0_src_r1, *(__m256i*)(pucRef + nRefPitch[0] * 0)); \
  ymm9_r2 = _mm256_sad_epu8(ymm1_src_r2, *(__m256i*)(pucRef + nRefPitch[0] * 1)); \
  ymm10_r3 = _mm256_sad_epu8(ymm2_src_r3, *(__m256i*)(pucRef + nRefPitch[0] * 2)); \
  ymm11_r4 = _mm256_sad_epu8(ymm3_src_r4, *(__m256i*)(pucRef + nRefPitch[0] * 3)); \
  ymm12_r5 = _mm256_sad_epu8(ymm4_src_r5, *(__m256i*)(pucRef + nRefPitch[0] * 4)); \
  ymm13_r6 = _mm256_sad_epu8(ymm5_src_r6, *(__m256i*)(pucRef + nRefPitch[0] * 5)); \
  ymm14_r7 = _mm256_sad_epu8(ymm6_src_r7, *(__m256i*)(pucRef + nRefPitch[0] * 6)); \
  ymm15_r8 = _mm256_sad_epu8(ymm7_src_r8, *(__m256i*)(pucRef + nRefPitch[0] * 7)); \

  */

#define Sad4blocks_8x8 \
  ymm8_r1 = _mm256_sad_epu8(workarea.ymm0_src_r1, *(__m256i*)(pucRef + nRefPitch[0] * 0)); \
  ymm9_r2 = _mm256_sad_epu8(workarea.ymm1_src_r2, *(__m256i*)(pucRef + nRefPitch[0] * 1)); \
  ymm10_r3 = _mm256_sad_epu8(workarea.ymm2_src_r3, *(__m256i*)(pucRef + nRefPitch[0] * 2)); \
  ymm11_r4 = _mm256_sad_epu8(workarea.ymm3_src_r4, *(__m256i*)(pucRef + nRefPitch[0] * 3)); \
  ymm12_r5 = _mm256_sad_epu8(workarea.ymm4_src_r5, *(__m256i*)(pucRef + nRefPitch[0] * 4)); \
  ymm13_r6 = _mm256_sad_epu8(workarea.ymm5_src_r6, *(__m256i*)(pucRef + nRefPitch[0] * 5)); \
  ymm14_r7 = _mm256_sad_epu8(workarea.ymm6_src_r7, *(__m256i*)(pucRef + nRefPitch[0] * 6)); \
  ymm15_r8 = _mm256_sad_epu8(workarea.ymm7_src_r8, *(__m256i*)(pucRef + nRefPitch[0] * 7)); \
\
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm9_r2); \
  ymm10_r3 = _mm256_adds_epu16(ymm10_r3, ymm11_r4); \
  ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm13_r6); \
  ymm14_r7 = _mm256_adds_epu16(ymm14_r7, ymm15_r8); \
 \
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm10_r3); \
  ymm12_r5 = _mm256_adds_epu16(ymm12_r5, ymm14_r7); \
\
  ymm8_r1 = _mm256_adds_epu16(ymm8_r1, ymm12_r5);

  // load src blocks
  const uint8_t* pucCurr = (uint8_t*)workarea.pSrc[0];

  uint8_t* pucRef;
  // 4 blocks at once proc, pel=1
/*  __m256i ymm0_src_r1 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 0));
  __m256i	ymm1_src_r2 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 1));
  __m256i	ymm2_src_r3 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 2));
  __m256i	ymm3_src_r4 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 3));
  __m256i	ymm4_src_r5 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 4));
  __m256i	ymm5_src_r6 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 5));
  __m256i	ymm6_src_r7 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 6));
  __m256i	ymm7_src_r8 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 7));*/
  workarea.ymm0_src_r1 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 0));
  workarea.ymm1_src_r2 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 1));
  workarea.ymm2_src_r3 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 2));
  workarea.ymm3_src_r4 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 3));
  workarea.ymm4_src_r5 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 4));
  workarea.ymm5_src_r6 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 5));
  workarea.ymm6_src_r7 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 6));
  workarea.ymm7_src_r8 = _mm256_loadu_si256((__m256i*)(pucCurr + nSrcPitch[0] * 7));


  // check zero and global predictor as 4block AVX2 proc because they are guaranteed to be coherent
  pucRef = (uint8_t*)GetRefBlock(workarea, 0, 0);

  for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX2; i++)
  {
    workarea.bestMV_multi[i] = zeroMV;
  }

  Sad4blocks_8x8 // 4 results in ymm8_r1
  sad = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm8_r1));
  workarea.bestMV_multi[0].sad = sad;
  workarea.nMinCost_multi[0] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm8_r1, 2);
  workarea.bestMV_multi[1].sad = sad;
  workarea.nMinCost_multi[1] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm8_r1, 4);
  workarea.bestMV_multi[2].sad = sad;
  workarea.nMinCost_multi[2] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm8_r1, 6);
  workarea.bestMV_multi[3].sad = sad;
  workarea.nMinCost_multi[3] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  iNumCheckedVectors = 0;
  checked_mv_vectors[iNumCheckedVectors] = 0;
  iNumCheckedVectors++;

  workarea.globalMVPredictor = ClipMV_SO2(workarea, workarea.globalMVPredictor);
  if (!IsVectorChecked((uint64_t)workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32)))
  {
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y);

    Sad4blocks_8x8 // 4 results in ymm8_r1
    sad = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm8_r1));
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[0])
    {
      workarea.bestMV_multi[0].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[0].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[0].sad = sad;
      workarea.nMinCost_multi[0] = cost;
    }

    sad = _mm256_extract_epi32(ymm8_r1, 2);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[1])
    {
      workarea.bestMV_multi[1].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[1].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[1].sad = sad;
      workarea.nMinCost_multi[1] = cost;
    }

    sad = _mm256_extract_epi32(ymm8_r1, 4);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[2])
    {
      workarea.bestMV_multi[2].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[2].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[2].sad = sad;
      workarea.nMinCost_multi[2] = cost;
    }

    sad = _mm256_extract_epi32(ymm8_r1, 6);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[3])
    {
      workarea.bestMV_multi[3].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[3].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[3].sad = sad;
      workarea.nMinCost_multi[3] = cost;
    }
  }

  // check 4 blocks prev level predictor coherency
  vectors_coh_check[0].x = workarea.predictor.x;
  vectors_coh_check[0].y = workarea.predictor.y;

  for (int i = 1; i < MAX_MULTI_BLOCKS_8x8_AVX2; i++)
  {
    VECTOR predictor_next = ClipMV_SO2(workarea, vectors[workarea.blkIdx + i]); // need update dy/dx max min to 4 blocks advance ?
    vectors_coh_check[i].x = predictor_next.x;
    vectors_coh_check[i].y = predictor_next.y;
  }

  if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX2))
  {
    // 4 blocks predictors of previous level are coherent - try to check if it is better zero and global checked positions
    // with 4 blocks check
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y);

    Sad4blocks_8x8 // 4 results in ymm8_r1
    cost = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm8_r1));

    if (cost < workarea.nMinCost_multi[0])
    {
      workarea.bestMV_multi[0].x = workarea.predictor.x;
      workarea.bestMV_multi[0].y = workarea.predictor.y;
      workarea.bestMV_multi[0].sad = cost;
      workarea.nMinCost_multi[0] = cost;
    }

    cost = _mm256_extract_epi32(ymm8_r1, 2);
    if (cost < workarea.nMinCost_multi[1])
    {
      workarea.bestMV_multi[1].x = workarea.predictor.x;
      workarea.bestMV_multi[1].y = workarea.predictor.y;
      workarea.bestMV_multi[1].sad = cost;
      workarea.nMinCost_multi[1] = cost;
    }

    cost = _mm256_extract_epi32(ymm8_r1, 4);
    if (cost < workarea.nMinCost_multi[2])
    {
      workarea.bestMV_multi[2].x = workarea.predictor.x;
      workarea.bestMV_multi[2].y = workarea.predictor.y;
      workarea.bestMV_multi[2].sad = cost;
      workarea.nMinCost_multi[2] = cost;
    }

    cost = _mm256_extract_epi32(ymm8_r1, 6);
    if (cost < workarea.nMinCost_multi[3])
    {
      workarea.bestMV_multi[3].x = workarea.predictor.x;
      workarea.bestMV_multi[3].y = workarea.predictor.y;
      workarea.bestMV_multi[3].sad = cost;
      workarea.nMinCost_multi[3] = cost;
    }

    // check coherency of best checked vectors again
    for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX2; i++)
    {
      vectors_coh_check[i].x = workarea.bestMV_multi[i].x;
      vectors_coh_check[i].y = workarea.bestMV_multi[i].y;
    }

    if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX2))
    {
      workarea.bestMV.sad = workarea.bestMV_multi[0].sad;
      pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = workarea.bestMV_multi[1].sad;
      pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = workarea.bestMV_multi[2].sad;
      pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = workarea.bestMV_multi[3].sad;

      ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2(workarea, workarea.bestMV_multi[0].x, workarea.bestMV_multi[0].y, pBlkData);
    }
    else // predictors cheking results are not coherent
    {
      for (int iBlkNum = 0; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX2; iBlkNum++)
      {
        workarea.bestMV.x = workarea.bestMV_multi[iBlkNum].x;
        workarea.bestMV.y = workarea.bestMV_multi[iBlkNum].y;
        workarea.bestMV.sad = workarea.bestMV_multi[iBlkNum].sad;
        workarea.nMinCost = workarea.nMinCost_multi[iBlkNum];

        ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(workarea, workarea.bestMV_multi[iBlkNum].x, workarea.bestMV_multi[iBlkNum].y);

        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 2] = workarea.bestMV.sad;

        workarea.pSrc[0] += nSrcPitch[0]; // advance src block pointer
      }
    }

  }
  else // predictors for next 3 blocks not coherent - try to check predictors
  {
    // using already checked zero and global results

    // first block
    // check first predictor
    cost = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[0]); // may be AVX2 sad with loaded src??

    if (cost < workarea.nMinCost_multi[0])
    {
      workarea.bestMV_multi[0].x = workarea.predictor.x;
      workarea.bestMV_multi[0].y = workarea.predictor.y;
      workarea.bestMV_multi[0].sad = cost;
      workarea.nMinCost_multi[0] = cost;
    }

    for (int iBlkNum = 1; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX2; iBlkNum++)
    {
      VECTOR predictor_next = ClipMV_SO2(workarea, vectors[workarea.blkIdx + iBlkNum]);
      cost = SAD(workarea.pSrc[0] + nSrcPitch[0] * iBlkNum, nSrcPitch[0], GetRefBlock(workarea, predictor_next.x, predictor_next.y), nRefPitch[0]); // may be AVX2 sad with loaded src ? ?

      if (cost < workarea.nMinCost_multi[iBlkNum])
      {
        workarea.bestMV_multi[iBlkNum].x = predictor_next.x;
        workarea.bestMV_multi[iBlkNum].y = predictor_next.y;
        workarea.bestMV_multi[iBlkNum].sad = cost;
        workarea.nMinCost_multi[iBlkNum] = cost;
      }

    }

    // check coherency of best checked vectors again
    for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX2; i++)
    {
      vectors_coh_check[i].x = workarea.bestMV_multi[i].x;
      vectors_coh_check[i].y = workarea.bestMV_multi[i].y;
    }

    if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX2))
    {
      workarea.bestMV.sad = workarea.bestMV_multi[0].sad;
      pBlkData[(workarea.blkx + 1) * N_PER_BLOCK + 2] = workarea.bestMV_multi[1].sad;
      pBlkData[(workarea.blkx + 2) * N_PER_BLOCK + 2] = workarea.bestMV_multi[2].sad;
      pBlkData[(workarea.blkx + 3) * N_PER_BLOCK + 2] = workarea.bestMV_multi[3].sad;

      ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2(workarea, workarea.bestMV_multi[0].x, workarea.bestMV_multi[0].y, pBlkData); // reuse loaded ref ?

    }
    else
    {
      for (int iBlkNum = 0; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX2; iBlkNum++)
      {
        workarea.bestMV.x = workarea.bestMV_multi[iBlkNum].x;
        workarea.bestMV.y = workarea.bestMV_multi[iBlkNum].y;
        workarea.bestMV.sad = workarea.bestMV_multi[iBlkNum].sad;
        workarea.nMinCost = workarea.nMinCost_multi[iBlkNum];

        ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(workarea, workarea.bestMV_multi[iBlkNum].x, workarea.bestMV_multi[iBlkNum].y);

        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 2] = workarea.bestMV.sad;

        workarea.pSrc[0] += nSrcPitch[0]; // advance src block pointer
      }
    }
  }
  // we store the result
//  vectors[workarea.blkIdx] = workarea.bestMV; - no need to store back because no analyse local level predictors in this type of search
  // stored internally in Exa_search()

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}

template<typename pixel_t>
void PlaneOfBlocks::PseudoEPZSearch_optSO4_glob_pred_avx512(WorkingArea& workarea, int* pBlkData)
{
  typedef typename std::conditional < sizeof(pixel_t) == 1, sad_t, bigsad_t >::type safe_sad_t;
  sad_t sad, cost;
  VECTOR_XY vectors_coh_check[MAX_MULTI_BLOCKS_8x8_AVX512];

  __m512i zmm16_r1_b0007, zmm18_r2_b0007, zmm20_r3_b0007, zmm22_r4_b0007, zmm24_r5_b0007, zmm26_r6_b0007, zmm28_r7_b0007, zmm30_r8_b0007;
  __m512i	zmm17_r1_b0815, zmm19_r2_b0815, zmm21_r3_b0815, zmm23_r4_b0815, zmm25_r5_b0815, zmm27_r6_b0815, zmm29_r7_b0815, zmm31_r8_b0815;

  __m256i ymm_0003, ymm_0407, ymm_0811, ymm_1215;

#define SAD_16blocks8x8_xy0 /*AVX512*/\
/* calc sads src with ref */ \
zmm16_r1_b0007 = _mm512_sad_epu8(workarea.zmm0_Src_r1_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 0)); \
zmm17_r1_b0815 = _mm512_sad_epu8(workarea.zmm1_Src_r1_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 0 + 64)); \
zmm18_r2_b0007 = _mm512_sad_epu8(workarea.zmm2_Src_r2_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 1)); \
zmm19_r2_b0815 = _mm512_sad_epu8(workarea.zmm3_Src_r2_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 1 + 64)); \
zmm20_r3_b0007 = _mm512_sad_epu8(workarea.zmm4_Src_r3_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 2)); \
zmm21_r3_b0815 = _mm512_sad_epu8(workarea.zmm5_Src_r3_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 2 + 64)); \
zmm22_r4_b0007 = _mm512_sad_epu8(workarea.zmm6_Src_r4_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 3)); \
zmm23_r4_b0815 = _mm512_sad_epu8(workarea.zmm7_Src_r4_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 3 + 64)); \
zmm24_r5_b0007 = _mm512_sad_epu8(workarea.zmm8_Src_r5_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 4)); \
zmm25_r5_b0815 = _mm512_sad_epu8(workarea.zmm9_Src_r5_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 4 + 64)); \
zmm26_r6_b0007 = _mm512_sad_epu8(workarea.zmm10_Src_r6_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 5)); \
zmm27_r6_b0815 = _mm512_sad_epu8(workarea.zmm11_Src_r6_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 5 + 64)); \
zmm28_r7_b0007 = _mm512_sad_epu8(workarea.zmm12_Src_r7_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 6)); \
zmm29_r7_b0815 = _mm512_sad_epu8(workarea.zmm13_Src_r7_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 6 + 64)); \
zmm30_r8_b0007 = _mm512_sad_epu8(workarea.zmm14_Src_r8_b0007, *(__m512i*)(pucRef + nRefPitch[0] * 7)); \
zmm31_r8_b0815 = _mm512_sad_epu8(workarea.zmm15_Src_r8_b0815, *(__m512i*)(pucRef + nRefPitch[0] * 7 + 64)); \
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

  // load src blocks
  const uint8_t* pucCurr = (uint8_t*)workarea.pSrc[0];

  uint8_t* pucRef;
  // 16 blocks at once proc, pel=1
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

  // check zero and global predictor as 16block AVX512 proc because they are guaranteed to be coherent
  pucRef = (uint8_t*)GetRefBlock(workarea, 0, 0);

  for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
  {
    workarea.bestMV_multi[i] = zeroMV;
  }

  SAD_16blocks8x8_xy0
  // results blocks 0..7 - zmm16_r1_b0007, blocks 8..15 - zmm17_r1_b0815

  ymm_0003 = _mm512_extracti64x4_epi64(zmm16_r1_b0007, 0);
  ymm_0407 = _mm512_extracti64x4_epi64(zmm16_r1_b0007, 1);
  ymm_0811 = _mm512_extracti64x4_epi64(zmm17_r1_b0815, 0);
  ymm_1215 = _mm512_extracti64x4_epi64(zmm17_r1_b0815, 1);

  sad = _mm256_extract_epi32(ymm_0003, 0);
  workarea.bestMV_multi[0].sad = sad;
  workarea.nMinCost_multi[0] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0003, 2);
  workarea.bestMV_multi[1].sad = sad;
  workarea.nMinCost_multi[1] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0003, 4);
  workarea.bestMV_multi[2].sad = sad;
  workarea.nMinCost_multi[2] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0003, 6);
  workarea.bestMV_multi[3].sad = sad;
  workarea.nMinCost_multi[3] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0407, 0);
  workarea.bestMV_multi[4].sad = sad;
  workarea.nMinCost_multi[4] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0407, 2);
  workarea.bestMV_multi[5].sad = sad;
  workarea.nMinCost_multi[5] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0407, 4);
  workarea.bestMV_multi[6].sad = sad;
  workarea.nMinCost_multi[6] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0407, 6);
  workarea.bestMV_multi[7].sad = sad;
  workarea.nMinCost_multi[7] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0811, 0);
  workarea.bestMV_multi[8].sad = sad;
  workarea.nMinCost_multi[8] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0811, 2);
  workarea.bestMV_multi[9].sad = sad;
  workarea.nMinCost_multi[9] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0811, 4);
  workarea.bestMV_multi[10].sad = sad;
  workarea.nMinCost_multi[10] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_0811, 6);
  workarea.bestMV_multi[11].sad = sad;
  workarea.nMinCost_multi[11] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_1215, 0);
  workarea.bestMV_multi[12].sad = sad;
  workarea.nMinCost_multi[12] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_1215, 2);
  workarea.bestMV_multi[13].sad = sad;
  workarea.nMinCost_multi[13] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_1215, 4);
  workarea.bestMV_multi[14].sad = sad;
  workarea.nMinCost_multi[14] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  sad = _mm256_extract_epi32(ymm_1215, 6);
  workarea.bestMV_multi[15].sad = sad;
  workarea.nMinCost_multi[15] = sad + ((penaltyZero * (safe_sad_t)sad) >> 8); // v.1.11.0.2*/

  iNumCheckedVectors = 0;
  checked_mv_vectors[iNumCheckedVectors] = 0;
  iNumCheckedVectors++;

  workarea.globalMVPredictor = ClipMV_SO2(workarea, workarea.globalMVPredictor);
  if (!IsVectorChecked((uint64_t)workarea.globalMVPredictor.x | ((uint64_t)workarea.globalMVPredictor.y << 32)))
  {
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.globalMVPredictor.x, workarea.globalMVPredictor.y);

    SAD_16blocks8x8_xy0
      // results blocks 0..7 - zmm16_r1_b0007, blocks 8..15 - zmm17_r1_b0815

    ymm_0003 = _mm512_extracti64x4_epi64(zmm16_r1_b0007, 0);
    ymm_0407 = _mm512_extracti64x4_epi64(zmm16_r1_b0007, 1);
    ymm_0811 = _mm512_extracti64x4_epi64(zmm17_r1_b0815, 0);
    ymm_1215 = _mm512_extracti64x4_epi64(zmm17_r1_b0815, 1);

    // todo: this long scalar processing is not good - need to change to SIMD with re-arranging of workarea.bestMV_multi and workarea.nMinCost_multi to be more SIMD-compatible ?

    sad = _mm256_extract_epi32(ymm_0003, 0);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[0])
    {
      workarea.bestMV_multi[0].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[0].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[0].sad = sad;
      workarea.nMinCost_multi[0] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0003, 2);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[1])
    {
      workarea.bestMV_multi[1].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[1].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[1].sad = sad;
      workarea.nMinCost_multi[1] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0003, 4);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[2])
    {
      workarea.bestMV_multi[2].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[2].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[2].sad = sad;
      workarea.nMinCost_multi[2] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0003, 6);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[3])
    {
      workarea.bestMV_multi[3].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[3].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[3].sad = sad;
      workarea.nMinCost_multi[3] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0407, 0);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[4])
    {
      workarea.bestMV_multi[4].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[4].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[4].sad = sad;
      workarea.nMinCost_multi[4] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0407, 2);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[5])
    {
      workarea.bestMV_multi[5].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[5].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[5].sad = sad;
      workarea.nMinCost_multi[5] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0407, 4);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[6])
    {
      workarea.bestMV_multi[6].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[6].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[6].sad = sad;
      workarea.nMinCost_multi[6] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0407, 6);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[7])
    {
      workarea.bestMV_multi[7].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[7].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[7].sad = sad;
      workarea.nMinCost_multi[7] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0811, 0);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[8])
    {
      workarea.bestMV_multi[8].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[8].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[8].sad = sad;
      workarea.nMinCost_multi[8] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0811, 2);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[9])
    {
      workarea.bestMV_multi[9].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[9].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[9].sad = sad;
      workarea.nMinCost_multi[9] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0811, 4);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[10])
    {
      workarea.bestMV_multi[10].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[10].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[10].sad = sad;
      workarea.nMinCost_multi[10] = cost;
    }

    sad = _mm256_extract_epi32(ymm_0811, 6);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[11])
    {
      workarea.bestMV_multi[11].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[11].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[11].sad = sad;
      workarea.nMinCost_multi[11] = cost;
    }

    sad = _mm256_extract_epi32(ymm_1215, 0);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[12])
    {
      workarea.bestMV_multi[12].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[12].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[12].sad = sad;
      workarea.nMinCost_multi[12] = cost;
    }

    sad = _mm256_extract_epi32(ymm_1215, 2);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[13])
    {
      workarea.bestMV_multi[13].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[13].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[13].sad = sad;
      workarea.nMinCost_multi[13] = cost;
    }

    sad = _mm256_extract_epi32(ymm_1215, 4);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[14])
    {
      workarea.bestMV_multi[14].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[14].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[14].sad = sad;
      workarea.nMinCost_multi[14] = cost;
    }

    sad = _mm256_extract_epi32(ymm_1215, 6);
    cost = sad + ((pglobal * (safe_sad_t)sad) >> 8);

    if (cost < workarea.nMinCost_multi[15])
    {
      workarea.bestMV_multi[15].x = workarea.globalMVPredictor.x;
      workarea.bestMV_multi[15].y = workarea.globalMVPredictor.y;
      workarea.bestMV_multi[15].sad = sad;
      workarea.nMinCost_multi[15] = cost;
    }
  }

  // check 16 blocks prev level predictor coherency
  vectors_coh_check[0].x = workarea.predictor.x;
  vectors_coh_check[0].y = workarea.predictor.y;

  for (int i = 1; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
  {
    VECTOR predictor_next = ClipMV_SO2(workarea, vectors[workarea.blkIdx + i]); // need update dy/dx max min to 4 blocks advance ?
    vectors_coh_check[i].x = predictor_next.x;
    vectors_coh_check[i].y = predictor_next.y;
  }

  if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX512))
  {
    // 16 blocks predictors of previous level are coherent - try to check if it is better zero and global checked positions
    // with 4 blocks check
    pucRef = (uint8_t*)GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y);

    SAD_16blocks8x8_xy0
      // results blocks 0..7 - zmm16_r1_b0007, blocks 8..15 - zmm17_r1_b0815

    ymm_0003 = _mm512_extracti64x4_epi64(zmm16_r1_b0007, 0);
    ymm_0407 = _mm512_extracti64x4_epi64(zmm16_r1_b0007, 1);
    ymm_0811 = _mm512_extracti64x4_epi64(zmm17_r1_b0815, 0);
    ymm_1215 = _mm512_extracti64x4_epi64(zmm17_r1_b0815, 1);

    cost = _mm256_extract_epi32(ymm_0003, 0);
    if (cost < workarea.nMinCost_multi[0])
    {
      workarea.bestMV_multi[0].x = workarea.predictor.x;
      workarea.bestMV_multi[0].y = workarea.predictor.y;
      workarea.bestMV_multi[0].sad = cost;
      workarea.nMinCost_multi[0] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0003, 2);
    if (cost < workarea.nMinCost_multi[1])
    {
      workarea.bestMV_multi[1].x = workarea.predictor.x;
      workarea.bestMV_multi[1].y = workarea.predictor.y;
      workarea.bestMV_multi[1].sad = cost;
      workarea.nMinCost_multi[1] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0003, 4);
    if (cost < workarea.nMinCost_multi[2])
    {
      workarea.bestMV_multi[2].x = workarea.predictor.x;
      workarea.bestMV_multi[2].y = workarea.predictor.y;
      workarea.bestMV_multi[2].sad = cost;
      workarea.nMinCost_multi[2] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0003, 6);
    if (cost < workarea.nMinCost_multi[3])
    {
      workarea.bestMV_multi[3].x = workarea.predictor.x;
      workarea.bestMV_multi[3].y = workarea.predictor.y;
      workarea.bestMV_multi[3].sad = cost;
      workarea.nMinCost_multi[3] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0407, 0);
    if (cost < workarea.nMinCost_multi[4])
    {
      workarea.bestMV_multi[4].x = workarea.predictor.x;
      workarea.bestMV_multi[4].y = workarea.predictor.y;
      workarea.bestMV_multi[4].sad = cost;
      workarea.nMinCost_multi[4] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0407, 2);
    if (cost < workarea.nMinCost_multi[5])
    {
      workarea.bestMV_multi[5].x = workarea.predictor.x;
      workarea.bestMV_multi[5].y = workarea.predictor.y;
      workarea.bestMV_multi[5].sad = cost;
      workarea.nMinCost_multi[5] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0407, 4);
    if (cost < workarea.nMinCost_multi[6])
    {
      workarea.bestMV_multi[6].x = workarea.predictor.x;
      workarea.bestMV_multi[6].y = workarea.predictor.y;
      workarea.bestMV_multi[6].sad = cost;
      workarea.nMinCost_multi[6] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0407, 6);
    if (cost < workarea.nMinCost_multi[7])
    {
      workarea.bestMV_multi[7].x = workarea.predictor.x;
      workarea.bestMV_multi[7].y = workarea.predictor.y;
      workarea.bestMV_multi[7].sad = cost;
      workarea.nMinCost_multi[7] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0811, 0);
    if (cost < workarea.nMinCost_multi[8])
    {
      workarea.bestMV_multi[8].x = workarea.predictor.x;
      workarea.bestMV_multi[8].y = workarea.predictor.y;
      workarea.bestMV_multi[8].sad = cost;
      workarea.nMinCost_multi[8] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0811, 2);
    if (cost < workarea.nMinCost_multi[9])
    {
      workarea.bestMV_multi[9].x = workarea.predictor.x;
      workarea.bestMV_multi[9].y = workarea.predictor.y;
      workarea.bestMV_multi[9].sad = cost;
      workarea.nMinCost_multi[9] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0811, 4);
    if (cost < workarea.nMinCost_multi[10])
    {
      workarea.bestMV_multi[10].x = workarea.predictor.x;
      workarea.bestMV_multi[10].y = workarea.predictor.y;
      workarea.bestMV_multi[10].sad = cost;
      workarea.nMinCost_multi[10] = cost;
    }

    cost = _mm256_extract_epi32(ymm_0811, 6);
    if (cost < workarea.nMinCost_multi[11])
    {
      workarea.bestMV_multi[11].x = workarea.predictor.x;
      workarea.bestMV_multi[11].y = workarea.predictor.y;
      workarea.bestMV_multi[11].sad = cost;
      workarea.nMinCost_multi[11] = cost;
    }

    cost = _mm256_extract_epi32(ymm_1215, 0);
    if (cost < workarea.nMinCost_multi[12])
    {
      workarea.bestMV_multi[12].x = workarea.predictor.x;
      workarea.bestMV_multi[12].y = workarea.predictor.y;
      workarea.bestMV_multi[12].sad = cost;
      workarea.nMinCost_multi[12] = cost;
    }

    cost = _mm256_extract_epi32(ymm_1215, 2);
    if (cost < workarea.nMinCost_multi[13])
    {
      workarea.bestMV_multi[13].x = workarea.predictor.x;
      workarea.bestMV_multi[13].y = workarea.predictor.y;
      workarea.bestMV_multi[13].sad = cost;
      workarea.nMinCost_multi[13] = cost;
    }

    cost = _mm256_extract_epi32(ymm_1215, 4);
    if (cost < workarea.nMinCost_multi[14])
    {
      workarea.bestMV_multi[14].x = workarea.predictor.x;
      workarea.bestMV_multi[14].y = workarea.predictor.y;
      workarea.bestMV_multi[14].sad = cost;
      workarea.nMinCost_multi[14] = cost;
    }

    cost = _mm256_extract_epi32(ymm_1215, 6);
    if (cost < workarea.nMinCost_multi[15])
    {
      workarea.bestMV_multi[15].x = workarea.predictor.x;
      workarea.bestMV_multi[15].y = workarea.predictor.y;
      workarea.bestMV_multi[15].sad = cost;
      workarea.nMinCost_multi[15] = cost;
    }

    // check coherency of best checked vectors again
    for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
    {
      vectors_coh_check[i].x = workarea.bestMV_multi[i].x;
      vectors_coh_check[i].y = workarea.bestMV_multi[i].y;
    }

    if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX512))
    {
      workarea.bestMV.sad = workarea.bestMV_multi[0].sad;
      for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
      {
        pBlkData[(workarea.blkx + i) * N_PER_BLOCK + 2] = workarea.bestMV_multi[i].sad;
      }

      ExhaustiveSearch8x8_uint8_16Blks_np1_sp1_avx512(workarea, workarea.bestMV_multi[0].x, workarea.bestMV_multi[0].y, pBlkData); 
    }
    else // predictors cheking results are not coherent
    {
      //TODO: try to fallback to 4blocks AVX2 search at first

      for (int iBlkNum = 0; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX512; iBlkNum++)
      {
        workarea.bestMV.x = workarea.bestMV_multi[iBlkNum].x;
        workarea.bestMV.y = workarea.bestMV_multi[iBlkNum].y;
        workarea.bestMV.sad = workarea.bestMV_multi[iBlkNum].sad;
        workarea.nMinCost = workarea.nMinCost_multi[iBlkNum];

        ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512(workarea, workarea.bestMV_multi[iBlkNum].x, workarea.bestMV_multi[iBlkNum].y);

        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 2] = workarea.bestMV.sad;

        workarea.pSrc[0] += nSrcPitch[0]; // advance src block pointer
      }
    }

  }
  else // predictors for next 3 blocks not coherent - try to check predictors
  {
    // using already checked zero and global results

    // first block
    // check first predictor
    cost = SAD(workarea.pSrc[0], nSrcPitch[0], GetRefBlock(workarea, workarea.predictor.x, workarea.predictor.y), nRefPitch[0]); // may be AVX2 sad with loaded src??

    if (cost < workarea.nMinCost_multi[0])
    {
      workarea.bestMV_multi[0].x = workarea.predictor.x;
      workarea.bestMV_multi[0].y = workarea.predictor.y;
      workarea.bestMV_multi[0].sad = cost;
      workarea.nMinCost_multi[0] = cost;
    }

    for (int iBlkNum = 1; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX512; iBlkNum++)
    {
      VECTOR predictor_next = ClipMV_SO2(workarea, vectors[workarea.blkIdx + iBlkNum]);
      cost = SAD(workarea.pSrc[0] + nSrcPitch[0] * iBlkNum, nSrcPitch[0], GetRefBlock(workarea, predictor_next.x, predictor_next.y), nRefPitch[0]); // may be AVX2 sad with loaded src ? ?

      if (cost < workarea.nMinCost_multi[iBlkNum])
      {
        workarea.bestMV_multi[iBlkNum].x = predictor_next.x;
        workarea.bestMV_multi[iBlkNum].y = predictor_next.y;
        workarea.bestMV_multi[iBlkNum].sad = cost;
        workarea.nMinCost_multi[iBlkNum] = cost;
      }

    }

    // check coherency of best checked vectors again
    for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
    {
      vectors_coh_check[i].x = workarea.bestMV_multi[i].x;
      vectors_coh_check[i].y = workarea.bestMV_multi[i].y;
    }

    if (IsVectorsCoherent(vectors_coh_check, MAX_MULTI_BLOCKS_8x8_AVX512))
    {
      workarea.bestMV.sad = workarea.bestMV_multi[0].sad;
      for (int i = 0; i < MAX_MULTI_BLOCKS_8x8_AVX512; i++)
      {
        pBlkData[(workarea.blkx + i) * N_PER_BLOCK + 2] = workarea.bestMV_multi[i].sad;
      }

      ExhaustiveSearch8x8_uint8_16Blks_np1_sp1_avx512(workarea, workarea.bestMV_multi[0].x, workarea.bestMV_multi[0].y, pBlkData); // reuse loaded ref ? do src keeped OK in register file ?

    }
    else
    {
      //TODO: try to fallback to 4blocks AVX2 search at first

      for (int iBlkNum = 0; iBlkNum < MAX_MULTI_BLOCKS_8x8_AVX512; iBlkNum++)
      {
        workarea.bestMV.x = workarea.bestMV_multi[iBlkNum].x;
        workarea.bestMV.y = workarea.bestMV_multi[iBlkNum].y;
        workarea.bestMV.sad = workarea.bestMV_multi[iBlkNum].sad;
        workarea.nMinCost = workarea.nMinCost_multi[iBlkNum];

        ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512(workarea, workarea.bestMV_multi[iBlkNum].x, workarea.bestMV_multi[iBlkNum].y);

        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 0] = workarea.bestMV.x;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 1] = workarea.bestMV.y;
        pBlkData[(workarea.blkx + iBlkNum) * N_PER_BLOCK + 2] = workarea.bestMV.sad;

        workarea.pSrc[0] += nSrcPitch[0]; // advance src block pointer
      }
    }
  }
  // we store the result
//  vectors[workarea.blkIdx] = workarea.bestMV; - no need to store back because no analyse local level predictors in this type of search
  // stored internally in Exa_search()

  workarea.planeSAD += workarea.bestMV.sad; // for debug, plus fixme outer planeSAD is not used
}
