// MVTools
// 2004 Manao
// Copyright(c)2006-2009 A.G.Balakhnin aka Fizick - true motion, overlap, YUY2, pelclip, divide, super
// General classe for motion based filters

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

#include "def.h"
#include	"ClipFnc.h"
#include "commonfunctions.h"
#include "cpu.h"
#include "DCTFFTW.h"
#include "DCTINT.h"
#include "MVAnalyse.h"
#include "MVGroupOfFrames.h"
#include "MVSuper.h"
#include "profile.h"
#include "SuperParams64Bits.h"


#if defined _WIN32 && defined DX12_ME
using namespace Microsoft::WRL;
#endif

#include <cmath>
#include <cstdio>
#include <algorithm>

MVAnalyse::MVAnalyse(
  PClip _child, int _blksizex, int _blksizey, int lv, int st, int stp,
  int _pelSearch, bool isb, int lambda, bool chroma, int df, sad_t _lsad,
  int _plevel, bool _global, int _pnew, int _pzero, int _pglobal,
  int _overlapx, int _overlapy, const char* _outfilename, int _dctmode,
  int _divide, int _sadx264, sad_t _badSAD, int _badrange, bool _isse,
  bool _meander, bool temporal_flag, bool _tryMany, bool multi_flag,
  bool mt_flag, int _chromaSADScale, int _optSearchOption, int _optPredictorType, IScriptEnvironment* env
)
  : ::GenericVideoFilter(_child)
  , _srd_arr(1)
  , _vectorfields_aptr()
  , _multi_flag(multi_flag)
  , _temporal_flag(temporal_flag)
  , _mt_flag(mt_flag)
  , _dct_factory_ptr()
  , _dct_pool()
  , _delta_max(0)
  , optSearchOption(_optSearchOption)
  , optPredictorType(_optPredictorType)

{
  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  static int id = 0; _instance_id = id++;
  _RPT1(0, "MvAnalyse.Create id=%d\n", _instance_id);

  if (multi_flag && df < 1)
  {
    env->ThrowError(
      "MAnalyse: cannot use a fixed frame reference "
      "(delta < 1) in multi mode."
    );
  }

  _RPT1(0, "MAnalyze created, isb=%d\n", isb ? 1 : 0);

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  MVAnalysisData &	analysisData = _srd_arr[0]._analysis_data;
  MVAnalysisData &	analysisDataDivided = _srd_arr[0]._analysis_data_divided;

  if (pixelsize == 4)
  {
    env->ThrowError("MAnalyse: Clip with float pixel type is not supported");
  }

  if (!vi.IsYUV() && !vi.IsYUY2()) // YUY2 is also YUV but let's see what is supported
  {
    env->ThrowError("MAnalyse: Clip must be YUV or YUY2");
  }

  if (_optPredictorType < 0 || _optPredictorType > 4)
  {
    env->ThrowError("MAnalyse: parameter 'optPredictorType' must be from 0 to 4");
  }

#ifdef _WIN32
  // large pages priv:
  DWORD error;
  HANDLE hToken = NULL;
  TOKEN_PRIVILEGES tp;
  BOOL result;

  // Enable Lock pages in memory priveledge for the current process
  if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES, &hToken))
  {
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid))
    {
      result = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
      error = GetLastError();

      if (!result || (error != ERROR_SUCCESS))
      {
//        env->ThrowError("LargePages: AdjustTokenPrivileges failed.");
      }
    }
    else
    {
//      env->ThrowError("LargePages: LookupPrivilegeValue failed.");
    }
  }
  else
  {
    // env->ThrowError("LargePages: Can not open process token"); // do not bug user with stop processing errors ???
  }

  // Cleanup
  if (hToken != 0) CloseHandle(hToken);
  hToken = NULL;

#endif


  if (vi.IsY())
    chroma = false; // silent fallback

  // get parameters of super clip - v2.0
  SuperParams64Bits	params;
  memcpy(&params, &child->GetVideoInfo().num_audio_samples, 8);
  const int		nHeight = params.nHeight;
  const int		nSuperHPad = params.nHPad;
  const int		nSuperVPad = params.nVPad;
  const int		nSuperPel = params.nPel;
  const int		nSuperModeYUV = params.nModeYUV;
  const int		nSuperLevels = params.nLevels;

  if (nHeight <= 0
    || nSuperHPad < 0
    || nSuperHPad >= vi.width / 2 // PF: intentional /2
    || nSuperVPad < 0
    || nSuperPel < 1
    || nSuperPel     >  4
    || nSuperModeYUV <  0
    || nSuperModeYUV >  YUVPLANES
    || nSuperLevels < 1)
  {
    env->ThrowError("MAnalyse: wrong super clip (pseudoaudio) parameters");
  }

  analysisData.nWidth = vi.width - nSuperHPad * 2;
  analysisData.nHeight = nHeight;
  analysisData.pixelType = vi.pixel_type;
  if (!vi.IsY()) {
    analysisData.yRatioUV = vi.IsYUY2() ? 1 : (1 << vi.GetPlaneHeightSubsampling(PLANAR_U));
    analysisData.xRatioUV = vi.IsYUY2() ? 2 : (1 << vi.GetPlaneWidthSubsampling(PLANAR_U));
  }
  else {
    analysisData.yRatioUV = 1; // n/a
    analysisData.xRatioUV = 1; // n/a
  }
  analysisData.pixelsize = pixelsize;
  analysisData.bits_per_pixel = bits_per_pixel;

#if defined _WIN32 && defined DX12_ME

  if (optSearchOption == 5 || optSearchOption == 6) // DX12_ME
  {
    if (!vi.IsYV12())
    {
      env->ThrowError("MAnalyse: Unsupported format for DX12_ME, only YV12 supported");
    }

    if ((_overlapx != 0) || (_overlapy != 0))
    {
      env->ThrowError("MAnalyse: Overlap processing currently not supported with DX12_ME");
    }

    int iBlkSize = 8;
    if ((_blksizex == 8) && (_blksizey == 8))
      iBlkSize = 8;
    else
      if ((_blksizex == 16) && (_blksizey == 16))
        iBlkSize = 16;
      else
      {
         env->ThrowError("MAnalyse: Unsupported block size for DX12_ME, only 8x8 and 16x16 supported");
      }

    Init_DX12_ME(env, analysisData.nWidth, analysisData.nHeight, iBlkSize, chroma);
  }

  iUploadedCurrentFrameNum = -1; // is it non-existent frame num ?

  //pNV12FrameData = new uint8_t[vi.width * vi.height * 2]; // NV12 format 4:2:0, really WxH*1.5 sized ? pitch is multiply of 32 ??
  pNV12FrameData = (uint8_t*)VirtualAlloc(0, vi.width * vi.height * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

#endif

  if (_chromaSADScale < -2 || _chromaSADScale>2)
    env->ThrowError(
      "MAnalyze: scaleCSAD must be -2..2"
    );

  analysisData.chromaSADScale = _chromaSADScale;

  pSrcGOF = new MVGroupOfFrames(
    nSuperLevels, analysisData.nWidth, analysisData.nHeight,
    nSuperPel, nSuperHPad, nSuperVPad, nSuperModeYUV,
    _isse, analysisData.xRatioUV, analysisData.yRatioUV, pixelsize, bits_per_pixel, mt_flag
  );
  pRefGOF = new MVGroupOfFrames(
    nSuperLevels, analysisData.nWidth, analysisData.nHeight,
    nSuperPel, nSuperHPad, nSuperVPad, nSuperModeYUV,
    _isse, analysisData.xRatioUV, analysisData.yRatioUV, pixelsize, bits_per_pixel, mt_flag
  );

  analysisData.nBlkSizeX = _blksizex;
  analysisData.nBlkSizeY = _blksizey;
  // same blocksize check in MAnalyze and MRecalculate
  // some blocksizes may not work in 4:2:0 (chroma subsampling division), but o.k. in 4:4:4
  const std::vector< std::pair< int, int > > allowed_blksizes =
  {
    { 64, 64 },{ 64,48 },{ 64,32 },{ 64,16 },
    { 48,64 },{ 48,48 },{ 48,24 },{ 48,12 },
    { 32,64 },{ 32,32 },{ 32,24 },{ 32,16 },{ 32,8 },
    { 24,48 },{ 24,32 },{ 24,24 },{ 24,12 },{ 24,6 },
    { 16,64 },{ 16,32 },{ 16,16 },{ 16,12 },{ 16,8 },{ 16,4 },{ 16,2 },
    { 12,48 },{ 12,24 },{ 12,16 },{ 12,12 },{ 12,6 },{ 12,3 },
    { 8,32 },{ 8,16 },{ 8,8 },{ 8,4 },{ 8,2 },{ 8,1 },
    { 6,24 },{ 6,12 },{ 6,6 },{ 6,3 },
    { 4,8 },{ 4,4 },{ 4,2 },
    { 3,6 },{ 3,3 },
    { 2,4 },{ 2,2 }
  };
  bool found = false;
  for (int i = 0; i < (int)allowed_blksizes.size(); i++) {
    if (analysisData.nBlkSizeX == allowed_blksizes[i].first && analysisData.nBlkSizeY == allowed_blksizes[i].second) {
      found = true;
      break;
    }
  }
  if (!found) {
    env->ThrowError(
      "MAnalyse: Invalid block size: %d x %d", analysisData.nBlkSizeX, analysisData.nBlkSizeY);
  }

  analysisData.nPel = nSuperPel;
  if (analysisData.nPel != 1
    && analysisData.nPel != 2
    && analysisData.nPel != 4)
  {
    env->ThrowError("MAnalyse: pel has to be 1 or 2 or 4");
  }

  analysisData.nDeltaFrame = df;
//	if (analysisData.nDeltaFrame < 1)
//	{
//		analysisData.nDeltaFrame = 1;
//	}

  if (_overlapx < 0 || _overlapx > _blksizex/2
    || _overlapy < 0 || _overlapy > _blksizey/2)
  {
    env->ThrowError("MAnalyse: overlap must be less or equal than half block size");
  }

  if (_overlapx % analysisData.xRatioUV || _overlapy % analysisData.yRatioUV)
  {
    env->ThrowError("MAnalyse: wrong overlap for the colorspace subsampling");
  }

  if (_divide != 0 && (_blksizex < 8 || _blksizey < 8)) // || instead of && 2.5.11.22 green garbage issue
  {
    env->ThrowError(
      "MAnalyse: Block sizes must be 8 or more for divide mode"
    );
  }

  if (_divide != 0
    && ((_overlapx % (2 * analysisData.xRatioUV)) || (_overlapy % (2 * analysisData.yRatioUV))) // PF subsampling-aware
    )
  {
    env->ThrowError("MAnalyse: wrong overlap for the colorspace subsampling for divide mode");
  }

  divideExtra = _divide;

  // include itself, but usually equal to 256 :-)
  headerSize = std::max(int(4 + sizeof(analysisData)), 256);

  analysisData.nOverlapX = _overlapx;
  analysisData.nOverlapY = _overlapy;

  int		nBlkX = (analysisData.nWidth - analysisData.nOverlapX)
    / (analysisData.nBlkSizeX - analysisData.nOverlapX);
  int		nBlkY = (analysisData.nHeight - analysisData.nOverlapY)
    / (analysisData.nBlkSizeY - analysisData.nOverlapY);

  // 2.7.36: fallback to no overlap when either nBlk count is less that 2
  if (nBlkX < 2 || nBlkY < 2) {
    analysisData.nOverlapX = 0;
    analysisData.nOverlapY = 0;
    nBlkX = analysisData.nWidth / analysisData.nBlkSizeX;
    nBlkY = analysisData.nHeight / analysisData.nBlkSizeY;
  }

  analysisData.nBlkX = nBlkX;
  analysisData.nBlkY = nBlkY;

  const int		nWidth_B =
    (analysisData.nBlkSizeX - analysisData.nOverlapX) * nBlkX
    + analysisData.nOverlapX; // covered by blocks
  const int		nHeight_B =
    (analysisData.nBlkSizeY - analysisData.nOverlapY) * nBlkY
    + analysisData.nOverlapY;

  // calculate valid levels
  int				nLevelsMax = 0;
  // at last one block
  while (((nWidth_B >> nLevelsMax) - analysisData.nOverlapX)
    / (analysisData.nBlkSizeX - analysisData.nOverlapX) > 0
    && ((nHeight_B >> nLevelsMax) - analysisData.nOverlapY)
    / (analysisData.nBlkSizeY - analysisData.nOverlapY) > 0)
  {
    ++nLevelsMax;
  }

  analysisData.nLvCount = (lv > 0) ? lv : nLevelsMax + lv;
  if (analysisData.nLvCount > nSuperLevels)
  {
    env->ThrowError(
      "MAnalyse: it is not enough levels  in super clip (%d), "
      "while MAnalyse asks %d", nSuperLevels, analysisData.nLvCount
    );
  }
  if (analysisData.nLvCount < 1
    || analysisData.nLvCount > nLevelsMax)
  {
    env->ThrowError(
      "MAnalyse: non-valid number of levels (%d)", analysisData.nLvCount
    );
  }

  analysisData.isBackward = isb;

  nLambda = lambda;

  // lambda is finally scaled in PlaneOfBlocks::WorkingArea::MotionDistorsion(int vx, int vy) const
  // as return (nLambda * dist) >> (16 - bits_per_pixel)
  // To have it 8x8 normalized, we would use 
  //   nLambda = lambda * ((_blksizex * _blksizey) / (8 * 8)) << (bits_per_pixel-8);  // normalize to 8x8 block size
  // and use 
  //   (nLambda * dist) >> 8  in PlaneOfBlocks::WorkingArea::MotionDistorsion
  // and change default lambda generation in truemotion=true preset
  // But doing this would kill compatibility, there are scripts which are using lambda properly scaled by the block size.

  lsad = _lsad   * (_blksizex * _blksizey) / 64 * (1 << (bits_per_pixel - 8));
  pnew = _pnew;
  plevel = _plevel;
  global = _global;
  pglobal = _pglobal;
  pzero = _pzero;
  badSAD = _badSAD * (_blksizex * _blksizey) / 64 * (1 << (bits_per_pixel - 8));
  badrange = _badrange;
  meander = _meander;
  tryMany = _tryMany;

  if (_dctmode != 0)
  {
    _dct_factory_ptr = std::unique_ptr <DCTFactory>(
      new DCTFactory(_dctmode, _isse, _blksizex, _blksizey, pixelsize, bits_per_pixel, *env)
      );
    _dct_pool.set_factory(*_dct_factory_ptr);
  }

  switch (st)
  {
  case 0:
    searchType = ONETIME;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 1:
    searchType = NSTEP;
    nSearchParam = (stp < 0) ? 0 : stp;
    break;
  case 3:
    searchType = EXHAUSTIVE;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 4:
    searchType = HEX2SEARCH;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 5:
    searchType = UMHSEARCH;
    nSearchParam = (stp < 1) ? 1 : stp; // really min is 4
    break;
  case 6:
    searchType = HSEARCH;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 7:
    searchType = VSEARCH;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 2:
  default:
    searchType = LOGARITHMIC;
    nSearchParam = (stp < 1) ? 1 : stp;
  }

  // not below value of 0 at finest level
  nPelSearch = (_pelSearch <= 0) ? analysisData.nPel : _pelSearch;


  analysisData.nFlags = 0;
  analysisData.nFlags |= (_isse) ? MOTION_USE_ISSE : 0;
  analysisData.nFlags |= (analysisData.isBackward) ? MOTION_IS_BACKWARD : 0;
  analysisData.nFlags |= (chroma) ? MOTION_USE_CHROMA_MOTION : 0;

  analysisData.nFlags |= conv_cpuf_flags_to_cpu(env->GetCPUFlags());
  // cpu flags has different layout that Avisynth's CPUF_xxx layout.
  // Never mix CPU_xxxx and CPUF_xxxx constants!

  nModeYUV = (chroma) ? YUVPLANES : YPLANE;
  if ((nModeYUV & nSuperModeYUV) != nModeYUV)
  {
    env->ThrowError(
      "MAnalyse: super clip does not contain needed color data"
    );
  }

  _vectorfields_aptr = std::unique_ptr <GroupOfPlanes>(new GroupOfPlanes(
    analysisData.nBlkSizeX,
    analysisData.nBlkSizeY,
    analysisData.nLvCount,
    analysisData.nPel,
    analysisData.nFlags,
    analysisData.nOverlapX,
    analysisData.nOverlapY,
    analysisData.nBlkX,
    analysisData.nBlkY,
    analysisData.xRatioUV, // PF
    analysisData.yRatioUV,
    divideExtra,
    analysisData.pixelsize, // PF
    analysisData.bits_per_pixel,
    (_dct_factory_ptr.get() != 0) ? &_dct_pool : 0,
    _mt_flag,
    analysisData.chromaSADScale,
    optSearchOption,
    env
  ));

  analysisData.nMagicKey = MVAnalysisData::MOTION_MAGIC_KEY;
  analysisData.nHPadding = nSuperHPad; // v2.0
  analysisData.nVPadding = nSuperVPad;

  // MVAnalysisData and outfile format version: last update v1.8.1
  analysisData.nVersion = MVAnalysisData::VERSION;
//	DebugPrintf(" MVAnalyseData size= %d",sizeof(analysisData));

  outfilename = _outfilename;
  if (lstrlen(outfilename) > 0)
  {
    outfile = fopen(outfilename, "wb");
    if (outfile == NULL)
    {
      env->ThrowError("MAnalyse: out file can not be created!");
    }
    else
    {
      fwrite(&analysisData, sizeof(analysisData), 1, outfile);
      // short vx, short vy, int SAD = 4 words = 8 bytes per block
      outfilebuf = new short[nBlkX * nBlkY * 4];
    }
  }
  else
  {
    outfile = NULL;
    outfilebuf = NULL;
  }

  // Defines the format of the output vector clip
  // count of 32 bit integers: 2_size_validity+(foreachblock(1_validity+blockCount*3))
  const int		width_bytes = headerSize + _vectorfields_aptr->GetArraySize() * 4;
  ClipFnc::format_vector_clip(
    vi, true, nBlkX, "rgb32", width_bytes, "MAnalyse", env
  );

  if (divideExtra)	//v1.8.1
  {
    memcpy(&analysisDataDivided, &analysisData, sizeof(analysisData));
    analysisDataDivided.nBlkX = analysisData.nBlkX * 2;
    analysisDataDivided.nBlkY = analysisData.nBlkY * 2;
    analysisDataDivided.nBlkSizeX = analysisData.nBlkSizeX / 2;
    analysisDataDivided.nBlkSizeY = analysisData.nBlkSizeY / 2;
    analysisDataDivided.nOverlapX = analysisData.nOverlapX / 2;
    analysisDataDivided.nOverlapY = analysisData.nOverlapY / 2;
    analysisDataDivided.nLvCount = analysisData.nLvCount + 1;
  }

  if (_temporal_flag)
  {
    _srd_arr[0]._vec_prev.resize(_vectorfields_aptr->GetArraySize()); // array for prev vectors
  }
  _srd_arr[0]._vec_prev_frame = -2;

  // From this point, analysisData and analysisDataDivided references will
  // become invalid, because of the _srd_arr.resize(). Don't use them any more.

  if (_multi_flag)
  {
    _delta_max = df;

    _srd_arr.resize(_delta_max * 2);
    for (int delta_index = 0; delta_index < _delta_max; ++delta_index)
    {
      for (int dir_index = 0; dir_index < 2; ++dir_index)
      {
        const int		index = delta_index * 2 + dir_index;
        SrcRefData &	srd = _srd_arr[index];
        srd = _srd_arr[0];

        srd._analysis_data.nDeltaFrame = delta_index + 1;
        srd._analysis_data.isBackward = (dir_index == 0);
        if (srd._analysis_data.isBackward)
        {
          srd._analysis_data.nFlags |= MOTION_IS_BACKWARD;
        }
        else
        {
          srd._analysis_data.nFlags &= ~MOTION_IS_BACKWARD;
        }

        srd._analysis_data_divided.nDeltaFrame = srd._analysis_data.nDeltaFrame;
        srd._analysis_data_divided.isBackward = srd._analysis_data.isBackward;
        srd._analysis_data_divided.nFlags = srd._analysis_data.nFlags;
      }
    }

    vi.num_frames *= _delta_max * 2;
    vi.MulDivFPS(_delta_max * 2, 1);
  }

  // we'll transmit to the processing filters a handle
  // on the analyzing filter itself ( it's own pointer ), in order
  // to activate the right parameters.
  if (divideExtra)	//v1.8.1
  {
#if !defined(MV_64BIT)
    vi.nchannels = reinterpret_cast <uintptr_t> (&_srd_arr[0]._analysis_data_divided);
#else
    uintptr_t p = reinterpret_cast <uintptr_t> (&_srd_arr[0]._analysis_data_divided);
    vi.nchannels = 0x80000000L | (int)(p >> 32);
    vi.sample_type = (int)(p & 0xffffffffUL);
#endif
  }
  else
  {
#if !defined(MV_64BIT)
    vi.nchannels = reinterpret_cast <uintptr_t> (&_srd_arr[0]._analysis_data);
#else
    uintptr_t p = reinterpret_cast <uintptr_t> (&_srd_arr[0]._analysis_data);
    vi.nchannels = 0x80000000L | (int)(p >> 32);
    vi.sample_type = (int)(p & 0xffffffffUL);
#endif
  }

}



MVAnalyse::~MVAnalyse()
{
  if (outfile != NULL)
  {
    fclose(outfile);
    outfile = 0;
    delete[] outfilebuf;
    outfilebuf = 0;
  }

  delete pSrcGOF;
  pSrcGOF = 0;
  delete pRefGOF;
  pRefGOF = 0;
  _RPT1(0, "MAnalyze destroyed %d\n",_instance_id);
#ifdef _WIN32
//  delete pNV12FrameData;
  VirtualFree(pNV12FrameData, 0, MEM_RELEASE);
#endif
}



PVideoFrame __stdcall MVAnalyse::GetFrame(int n, IScriptEnvironment* env)
{
  _RPT2(0, "MAnalyze GetFrame, frame=%d id=%d\n", n, _instance_id);
  const int		ndiv = (_multi_flag) ? _delta_max * 2 : 1;
  const int		nsrc = n / ndiv;
  const int		srd_index = n % ndiv;

  SrcRefData &	srd = _srd_arr[srd_index];

  const int		nbr_src_frames = child->GetVideoInfo().num_frames;
  int				minframe;
  int				maxframe;
  int				nref;
  if (srd._analysis_data.nDeltaFrame > 0)
  {
    const int		offset =
      (srd._analysis_data.isBackward)
      ? srd._analysis_data.nDeltaFrame
      : -srd._analysis_data.nDeltaFrame;
    minframe = std::max(-offset, 0);
    maxframe = nbr_src_frames + std::min(-offset, 0);
    nref = nsrc + offset;
  }
  else // special static mode
  {
    nref = -srd._analysis_data.nDeltaFrame;	// positive fixed frame number
    minframe = 0;
    maxframe = nbr_src_frames;
  }

  PVideoFrame			dst = env->NewVideoFrame(vi); // frameprop inheritance later (if there is source)
  unsigned char *	pDst = dst->GetWritePtr();

  // 0 headersize (max(4+sizeof(analysisData),256)
  // 4: analysysData
  // 256: data 
  // 256: 2_size_validity+(foreachblock(1_validity+blockCount*3))

  // write analysis parameters as a header to frame
  memcpy(pDst, &headerSize, sizeof(int));
  if (divideExtra)
  {
    memcpy(
      pDst + sizeof(int),
      &srd._analysis_data_divided,
      sizeof(srd._analysis_data_divided)
    );
  }
  else
  {
    memcpy(
      pDst + sizeof(int),
      &srd._analysis_data,
      sizeof(srd._analysis_data)
    );
  }
  pDst += headerSize;

  if (nsrc < minframe || nsrc >= maxframe)
  {
    // fill all vectors with invalid data
    _vectorfields_aptr->WriteDefaultToArray(reinterpret_cast <int *> (pDst));
  }

  else
  {
//		DebugPrintf ("MVAnalyse: Get src frame %d",nsrc);
    _RPT3(0, "MAnalyze GetFrame, frame_nsrc=%d nref=%d id=%d\n", nsrc, nref, _instance_id);

    PVideoFrame	src = child->GetFrame(nsrc, env); // v2.0
    // if(has_at_least_v8) env->copyFrameProps(src, dst); // frame property support
    // The result clip is a special MV clip. It does not need to inherit the frame props of source

    load_src_frame(*pSrcGOF, src, srd._analysis_data);

//		DebugPrintf ("MVAnalyse: Get ref frame %d", nref);
//		DebugPrintf ("MVAnalyse frame %i backward=%i", nsrc, srd._analysis_data.isBackward);
    ::PVideoFrame	ref = child->GetFrame(nref, env); // v2.0
    load_src_frame(*pRefGOF, ref, srd._analysis_data);

    const int		fieldShift = ClipFnc::compute_fieldshift(
      child,
      vi.IsFieldBased(),
      srd._analysis_data.nPel,
      nsrc,
      nref
    );

    if (outfile != NULL)
    {
      fwrite(&n, sizeof(int), 1, outfile);	// write frame number
    }

    // temporal predictor dst if prev frame was really prev
    int *			pVecPrevOrNull = 0;
    if (_temporal_flag && srd._vec_prev_frame == nsrc - 1)
    {
      pVecPrevOrNull = &srd._vec_prev[0];
    }

#if defined _WIN32 && defined DX12_ME

    int16_t* pSADReadbackBufferData{}; // temp here 

    if (optSearchOption == 5 || optSearchOption == 6)
    {
      HRESULT hr;
      UINT64 size;

      hr = m_commandAllocatorGraphics->Reset();
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_commandAllocatorGraphics->Reset"
        );
      }

      hr = m_commandAllocatorVideo->Reset();
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_commandAllocatorVideo->Reset"
        );
      }

      hr = m_GraphicsCommandList->Reset(m_commandAllocatorGraphics.Get(), 0);
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_GraphicsCommandList->Reset 1"
        );
      }

      int iWidth;
      int iHeight;

      if (nsrc != iUploadedCurrentFrameNum) // do not upload current source on each src+ref pair
      {
        m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spCurrentResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));

        // todo: make format conversion at time or UpdateSubresources for lesser memory copy
        LoadNV12(pSrcGOF, srd._analysis_data.nFlags & MOTION_USE_CHROMA_MOTION, iWidth, iHeight);

        D3D12_SUBRESOURCE_DATA textureData_current = {};
        textureData_current.pData = pNV12FrameData;
        textureData_current.RowPitch = iWidth;
        textureData_current.SlicePitch = (uint64_t)iWidth * (uint64_t)iHeight + (uint64_t)iWidth * (uint64_t)iHeight / 2;

        size = UpdateSubresources(m_GraphicsCommandList.Get(), spCurrentResource.Get(), spCurrentResourceUpload.Get(), 0, 0, 1, &textureData_current);
        m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spCurrentResource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));

        iUploadedCurrentFrameNum = nsrc;
      } 

      m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spReferenceResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));

      LoadNV12(pRefGOF, srd._analysis_data.nFlags & MOTION_USE_CHROMA_MOTION, iWidth, iHeight);

      D3D12_SUBRESOURCE_DATA textureData_ref = {};
      textureData_ref.pData = pNV12FrameData;
      textureData_ref.RowPitch = iWidth;
      textureData_ref.SlicePitch = (uint64_t)iWidth * (uint64_t)iHeight + (uint64_t)iWidth * (uint64_t)iHeight / 2;

      size = UpdateSubresources(m_GraphicsCommandList.Get(), spReferenceResource.Get(), spReferenceResourceUpload.Get(), 0, 0, 1, &textureData_ref);
      m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spReferenceResource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));

      hr = m_GraphicsCommandList->Close();
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_GraphicsCommandList->Close"
        );
      }

      // Execute Commandlist.
      ID3D12CommandList* ppGraphicsCommandLists[1] = { m_GraphicsCommandList.Get() };
      m_commandQueueGraphics->ExecuteCommandLists(1, ppGraphicsCommandLists);

      // Signal and increment the fence value.
      const UINT64 fence_Graphics = m_fenceValue;
      hr = m_commandQueueGraphics->Signal(m_fence.Get(), fence_Graphics);
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_commandQueue->Signal fence_Graphics"
        );
      }

      m_fenceValue++;

      // Wait until the previous frame is finished.
      if (m_fence->GetCompletedValue() < fence_Graphics)
      {
        hr = m_fence->SetEventOnCompletion(fence_Graphics, m_fenceEventGraphics);
        if (hr != S_OK)
        {
          env->ThrowError(
            "MAnalyse: Error m_fence->SetEventOnCompletion -> EventGraphics"
          );
        }
        WaitForSingleObject(m_fenceEventGraphics, INFINITE);
      }
      
     
      hr = m_VideoEncodeCommandList->Reset(m_commandAllocatorVideo.Get());

      m_VideoEncodeCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spCurrentResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ));
      m_VideoEncodeCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spReferenceResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ));
      m_VideoEncodeCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spResolvedMotionVectors.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE));
      
      const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT outputArgsEM = { spVideoMotionVectorHeap.Get() };

      const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT inputArgsEM = {
          spCurrentResource.Get(),
          0,
          spReferenceResource.Get(),
          0,
          nullptr // pHintMotionVectorHeap
      };
    
      m_VideoEncodeCommandList->EstimateMotion(spVideoMotionEstimator.Get(), &outputArgsEM, &inputArgsEM);

      const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT outputArgsRMVH = {
          spResolvedMotionVectors.Get(),
          {} };

      D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT inputArgsRMVH = {};
      if (optSearchOption == 5)
      {
        inputArgsRMVH = {
            spVideoMotionVectorHeap.Get(),
            (UINT)srd._analysis_data.nWidth,
            (UINT)srd._analysis_data.nHeight
        };
      }
      else // SO==6
      {
        inputArgsRMVH = {
            spVideoMotionVectorHeap.Get(),
            (UINT)srd._analysis_data.nWidth / 2,
            (UINT)srd._analysis_data.nHeight / 2
        };
      }

      m_VideoEncodeCommandList->ResolveMotionVectorHeap(&outputArgsRMVH, &inputArgsRMVH);

      m_VideoEncodeCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spResolvedMotionVectors.Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, D3D12_RESOURCE_STATE_COMMON));
      m_VideoEncodeCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spCurrentResource.Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, D3D12_RESOURCE_STATE_COMMON));
      m_VideoEncodeCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spReferenceResource.Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, D3D12_RESOURCE_STATE_COMMON));
      

      hr = m_VideoEncodeCommandList->Close();
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_VideoEncodeCommandList->Close"
        );
      }

      // Execute Commandlist.
      ID3D12CommandList* ppCommandLists[1] = { m_VideoEncodeCommandList.Get() };
      m_commandQueueVideo->ExecuteCommandLists(1, ppCommandLists);

      // Signal and increment the fence value.
      const UINT64 fence_Video = m_fenceValue;
      hr = m_commandQueueVideo->Signal(m_fence.Get(), fence_Video);
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_commandQueue->Signal fence_Video"
        );
      }

      m_fenceValue++;

      // Wait until the previous frame is finished.
      if (m_fence->GetCompletedValue() < fence_Video)
      {
        hr = m_fence->SetEventOnCompletion(fence_Video, m_fenceEventVideo);
        if (hr != S_OK)
        {
          env->ThrowError(
            "MAnalyse: Error m_fence->SetEventOnCompletion -> EventVideo"
          );
        }
        WaitForSingleObject(m_fenceEventVideo, INFINITE);
      }
      
      
      // copy back to CPU memory
      hr = m_GraphicsCommandList->Reset(m_commandAllocatorGraphics.Get(), 0);

      m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spResolvedMotionVectors.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));

      int iNumBlocksX = srd._analysis_data.GetBlkX();
      int iNumBlocksY = srd._analysis_data.GetBlkY();

      int iRowPitchUA = iNumBlocksX * sizeof(int); // 2x16bit
      int iRowPitch = iRowPitchUA + (256 - (iRowPitchUA % 256)); // must be multiply of 256

      // Get the copy target location
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
      bufferFootprint.Footprint.Width = iNumBlocksX; // num blocks W 
      bufferFootprint.Footprint.Height = iNumBlocksY; // num Blocks H
      bufferFootprint.Footprint.Depth = 1;
      bufferFootprint.Footprint.RowPitch = iRowPitch; // 16+16 * num blocks W and multiply of 256 ?
      bufferFootprint.Footprint.Format = DXGI_FORMAT_R16G16_SINT;

      CD3DX12_TEXTURE_COPY_LOCATION copyDest(spResolvedMotionVectorsReadBack.Get(), bufferFootprint);
      CD3DX12_TEXTURE_COPY_LOCATION copySrc(spResolvedMotionVectors.Get(), 0);

      // Copy the texture
      m_GraphicsCommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

      m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(spResolvedMotionVectors.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));

      hr = m_GraphicsCommandList->Close();
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_GraphicsCommandList->Close readback"
        );
      }
      // Execute Commandlist.
      ID3D12CommandList* ppCopyBackCommandLists[1] = { m_GraphicsCommandList.Get() };
      m_commandQueueGraphics->ExecuteCommandLists(1, ppCopyBackCommandLists);

      // Signal and increment the fence value.
      const UINT64 fence_CopyBack = m_fenceValue;
      hr = m_commandQueueGraphics->Signal(m_fence.Get(), fence_CopyBack);
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_commandQueue->Signal fence_CopyBack"
        );
      }

      m_fenceValue++;

      // Wait until the previous frame is finished.
      if (m_fence->GetCompletedValue() < fence_CopyBack)
      {
        hr = m_fence->SetEventOnCompletion(fence_CopyBack, m_fenceEventCopyBack);
        if (hr != S_OK)
        {
          env->ThrowError(
            "MAnalyse: Error m_fence->SetEventOnCompletion -> EventCopyBack"
          );
        }
        WaitForSingleObject(m_fenceEventCopyBack, INFINITE);
      }

      int16_t* pReadbackBufferData{};

      hr = spResolvedMotionVectorsReadBack->Map(0, nullptr, reinterpret_cast<void**>(&pReadbackBufferData));
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error spResolvedMotionVectorsReadBack->Map"
        );
      }

      // make reading
      int16_t* pSrcMVs = pReadbackBufferData;
      int* piDstMVs = (int*)pDst;

      //  group's size
      piDstMVs[0] = _vectorfields_aptr->GetArraySize();

      // validity : 1 in that case
      piDstMVs[1] = 1;

      piDstMVs+=3; // +1 in search_mv_slice - size of ??

      if (optSearchOption == 5)
      {
        for (int h = 0; h < iNumBlocksY; ++h)
        {
          for (int w = 0; w < iNumBlocksX; ++w)
          {
            piDstMVs[0] = pSrcMVs[w * 2] / 4; // for pel=1, divide qpel by 4
            piDstMVs[1] = pSrcMVs[w * 2 + 1] / 4; // for pel=1, divide qpel by 4

            piDstMVs += 3;
          }
          pSrcMVs += iRowPitch / 2; // pitch in bytes ?
        }
      }
      else // if == 6
      {

      }

/*      // copy to 'vectors' structure of plane 0 for sad calc only
      PlaneOfBlocks* pob = _vectorfields_aptr->GetPlane(0);
      VECTOR *pVectors = &pob->vectors[0];
      int16_t* pSrcMVs = pReadbackBufferData;

#ifdef _DEBUG
      // debug check
      if (iNumBlocksX * iNumBlocksY != pob->vectors.size())
      {
        env->ThrowError(
          "MAnalyse: Error size of vectors buf != number of vectors"
        );
      }

#endif

      if (optSearchOption == 5)
      {
        for (int h = 0; h < iNumBlocksY; ++h)
        {
          for (int w = 0; w < iNumBlocksX; ++w)
          {
            pVectors[0].x = pSrcMVs[w * 2] / 4; // for pel=1, divide qpel by 4
            pVectors[0].y = pSrcMVs[w * 2 + 1] / 4; // for pel=1, divide qpel by 4

            pVectors++;
          }
          pSrcMVs += iRowPitch / 2; // pitch in bytes
        }
      }
      else // if == 6
      {
        for (int h = 0; h < iNumBlocksY; ++h)
        {
          for (int w = 0; w < iNumBlocksX; ++w)
          {
            pVectors[0].x = pSrcMVs[w * 2] / 2; // for pel=1, divide qpel by 4
            pVectors[0].y = pSrcMVs[w * 2 + 1] / 2; // for pel=1, divide qpel by 4

            pVectors++;
          }
          pSrcMVs += iRowPitch / 2; // pitch in bytes
        }
      }
      */


      spResolvedMotionVectorsReadBack->Unmap(0, NULL);

      // calc SADs using loaded resources and D3D12 compute shader
      m_computeAllocator->Reset();
      m_computeCommandList->Reset(m_computeAllocator.Get(), m_computePSO.Get());

      ID3D12DescriptorHeap* pHeaps[] = { m_SRVDescriptorHeap->Heap(), m_samplerDescriptorHeap->Heap() };
      m_computeCommandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

      m_computeCommandList->SetComputeRootSignature(m_computeRootSignature.Get());

//      m_computeCommandList->SetComputeRootConstantBufferView(e_rootParameterCB, m_computeHeap.GpuAddress());
      m_computeCommandList->SetComputeRootConstantBufferView(e_rootParameterCB, sadCBparamsBV.BufferLocation);
      m_computeCommandList->SetComputeRootDescriptorTable(e_rootParameterSampler, m_samplerDescriptorHeap->GetGpuHandle(0));
      m_computeCommandList->SetComputeRootDescriptorTable(e_rootParameterSRV, m_SRVDescriptorHeap->GetGpuHandle(e_iSRV + 0));				
      m_computeCommandList->SetComputeRootDescriptorTable(e_rootParameterUAV, m_SRVDescriptorHeap->GetGpuHandle(e_iUAV + 0)); 

      m_computeCommandList->SetPipelineState(m_computePSO.Get());
      m_computeCommandList->Dispatch(m_ThreadGroupX, m_ThreadGroupY, 1);

      // close and execute the command list
      m_computeCommandList->Close();
      ID3D12CommandList* computeList = m_computeCommandList.Get();
      m_computeCommandQueue->ExecuteCommandLists(1, &computeList);

      const uint64_t fenceCompute = m_fenceValue++;
      m_computeCommandQueue->Signal(m_fence.Get(), fenceCompute);
      if (m_fence->GetCompletedValue() < fenceCompute)								// block until async compute has completed using a fence
      {
        m_fence->SetEventOnCompletion(fenceCompute, m_computeFenceEvent);
        WaitForSingleObject(m_computeFenceEvent, INFINITE);
      }


      // readback computed SAD to CPU memory
      hr = m_GraphicsCommandList->Reset(m_commandAllocatorGraphics.Get(), 0);

      m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SADTexture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));

//      int iNumBlocksX = srd._analysis_data.GetBlkX();
//      int iNumBlocksY = srd._analysis_data.GetBlkY();

      int iSADRowPitchUA = iNumBlocksX * sizeof(short); // 16bit
      int iSADRowPitch = iSADRowPitchUA + (256 - (iSADRowPitchUA % 256)); // must be multiply of 256

      // Get the copy target location
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferSADFootprint = {};
      bufferSADFootprint.Footprint.Width = iNumBlocksX; // num blocks W 
      bufferSADFootprint.Footprint.Height = iNumBlocksY; // num Blocks H
      bufferSADFootprint.Footprint.Depth = 1;
      bufferSADFootprint.Footprint.RowPitch = iSADRowPitch; // 16+16 * num blocks W and multiply of 256 ?
      bufferSADFootprint.Footprint.Format = DXGI_FORMAT_R16_UINT;

      CD3DX12_TEXTURE_COPY_LOCATION copySADDest(spSADReadBack.Get(), bufferSADFootprint);
      CD3DX12_TEXTURE_COPY_LOCATION copySADSrc(m_SADTexture.Get(), 0);

      // Copy the texture
      m_GraphicsCommandList->CopyTextureRegion(&copySADDest, 0, 0, 0, &copySADSrc, nullptr);

      m_GraphicsCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SADTexture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

      hr = m_GraphicsCommandList->Close();
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_GraphicsCommandList->Close readback SAD"
        );
      }
      // Execute Commandlist.
      ID3D12CommandList* ppSADCopyBackCommandLists[1] = { m_GraphicsCommandList.Get() };
      m_commandQueueGraphics->ExecuteCommandLists(1, ppSADCopyBackCommandLists);

      // Signal and increment the fence value.
      const UINT64 fence_SADCopyBack = m_fenceValue;
      hr = m_commandQueueGraphics->Signal(m_fence.Get(), fence_SADCopyBack);
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error m_commandQueue->Signal fence_SADCopyBack"
        );
      }

      m_fenceValue++;

      // Wait until the previous frame is finished.
      if (m_fence->GetCompletedValue() < fence_SADCopyBack)
      {
        hr = m_fence->SetEventOnCompletion(fence_SADCopyBack, m_fenceEventCopyBack);
        if (hr != S_OK)
        {
          env->ThrowError(
            "MAnalyse: Error m_fence->SetEventOnCompletion -> EventCopyBack SAD"
          );
        }
        WaitForSingleObject(m_fenceEventCopyBack, INFINITE);
      }

//      int16_t* pSADReadbackBufferData{};

      hr = spSADReadBack->Map(0, nullptr, reinterpret_cast<void**>(&pSADReadbackBufferData));
      if (hr != S_OK)
      {
        env->ThrowError(
          "MAnalyse: Error spSADReadBack->Map"
        );
      }

      // make reading
      // scatter to 'out' structure of MAnalyse output
      int16_t* pSrcSADs = pSADReadbackBufferData;
      int* piDstSAD = (int*)pDst;

      // skip group's size
      piDstSAD++;

      // skip validity : 1 in that case
      piDstSAD++;

      piDstSAD++; // +1 in search_mv_slice

      piDstSAD += 2; // SAD part

      if (optSearchOption == 5)
      {
        for (int h = 0; h < iNumBlocksY; ++h)
        {
          for (int w = 0; w < iNumBlocksX; ++w)
          {
/*            if (piDstSAD[0] != (int)pSrcSADs[w])
            {
              int idbr = 0;
            }*/
            piDstSAD[0] = (int)pSrcSADs[w];

            piDstSAD += 3;
          }
          pSrcSADs += iSADRowPitch / 2; // pitch in bytes ?
        }
      }
      else // if == 6
      {

      }

      // unmap finally
//      spSADReadBack->Unmap(0, NULL);
 
    }
#endif

//    if ((optSearchOption != 5) && (optSearchOption != 6)) // do not call search from PlaneofBlocks - all done with DX12
    {
      _vectorfields_aptr->SearchMVs(
        pSrcGOF, pRefGOF,
        searchType, nSearchParam, nPelSearch, nLambda, lsad, pnew, plevel,
        global, srd._analysis_data.nFlags, reinterpret_cast<int*>(pDst),
        outfilebuf, fieldShift, pzero, pglobal, badSAD, badrange,
        meander, pVecPrevOrNull, tryMany, optPredictorType
      );
    }

    // compare shader SAD with MAnalyse SAD

    if ((optSearchOption == 5) || (optSearchOption == 6))
    {
      int16_t* pSrcSADs = pSADReadbackBufferData;
      int* piDstSAD = (int*)pDst;

      // skip group's size
      piDstSAD++;

      // skip validity : 1 in that case
      piDstSAD++;

      piDstSAD++; // +1 in search_mv_slice

      piDstSAD += 2; // SAD part

      int iNumBlocksX = srd._analysis_data.GetBlkX();
      int iNumBlocksY = srd._analysis_data.GetBlkY();

      int iSADRowPitchUA = iNumBlocksX * sizeof(short); // 16bit
      int iSADRowPitch = iSADRowPitchUA + (256 - (iSADRowPitchUA % 256)); // must be multiply of 256


      if (optSearchOption == 5)
      {
        for (int h = 0; h < iNumBlocksY; ++h)
        {
          for (int w = 0; w < iNumBlocksX; ++w)
          {
            if (piDstSAD[0] != (int)pSrcSADs[w])
            {
              int idbr = 0;
            }
            piDstSAD+=3;
          }
          pSrcSADs += iSADRowPitch / 2; // pitch in bytes ?
        }
      }
      else // if == 6
      {

      }

      // unmap finally
      spSADReadBack->Unmap(0, NULL);
    }
    
    if (divideExtra)
    {
      // make extra level with divided sublocks with median (not estimated)
      // motion
      _vectorfields_aptr->ExtraDivide(
        reinterpret_cast <int *> (pDst),
        srd._analysis_data.nFlags
      );
    }

//		PROFILE_CUMULATE ();
    if (outfile != NULL)
    {
      fwrite(
        outfilebuf,
        sizeof(short) * 4 * srd._analysis_data.nBlkX
        * srd._analysis_data.nBlkY,
        1,
        outfile
      );
    }
  }

  if (_temporal_flag)
  {
    // store previous vectors for use as predictor in next frame
    memcpy(
      &srd._vec_prev[0],
      reinterpret_cast <int *> (pDst),
      _vectorfields_aptr->GetArraySize()
    );
    srd._vec_prev_frame = nsrc;
  }

  _RPT3(0, "MAnalyze GetFrame END, frame_nsrc=%d nref=%d id=%d\n", nsrc, nref, _instance_id);
  return dst;
}



void	MVAnalyse::load_src_frame(MVGroupOfFrames &gof, ::PVideoFrame &src, const MVAnalysisData &ana_data)
{
  PROFILE_START(MOTION_PROFILE_YUY2CONVERT);
  const unsigned char *	pSrcY;
  const unsigned char *	pSrcU;
  const unsigned char *	pSrcV;
  int				nSrcPitchY;
  int				nSrcPitchUV;
  if ((ana_data.pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2)
  {
    // planar data packed to interleaved format (same as interleved2planar
    // by kassandro) - v2.0.0.5
    pSrcY = src->GetReadPtr();
    pSrcU = pSrcY + src->GetRowSize() / 2;
    pSrcV = pSrcU + src->GetRowSize() / 4;
    nSrcPitchY = src->GetPitch();
    nSrcPitchUV = nSrcPitchY;
  }
  else
  {
    pSrcY = src->GetReadPtr(PLANAR_Y);
    pSrcU = src->GetReadPtr(PLANAR_U);
    pSrcV = src->GetReadPtr(PLANAR_V);
    nSrcPitchY = src->GetPitch(PLANAR_Y);
    nSrcPitchUV = src->GetPitch(PLANAR_U);
  }
  PROFILE_STOP(MOTION_PROFILE_YUY2CONVERT);

  gof.Update(
    nModeYUV,
    (BYTE*)pSrcY, nSrcPitchY,
    (BYTE*)pSrcU, nSrcPitchUV,
    (BYTE*)pSrcV, nSrcPitchUV
  ); // v2.0
}

#if defined _WIN32 && defined DX12_ME

void MVAnalyse::GetHardwareAdapter(
  IDXGIFactory1* pFactory,
  IDXGIAdapter1** ppAdapter,
  bool requestHighPerformanceAdapter)
{
  *ppAdapter = nullptr;

  ComPtr<IDXGIAdapter1> adapter;

  ComPtr<IDXGIFactory6> factory6;
  if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6))))
  {
    for (
      UINT adapterIndex = 0;
      SUCCEEDED(factory6->EnumAdapterByGpuPreference(
        adapterIndex,
        requestHighPerformanceAdapter == true ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED,
        IID_PPV_ARGS(&adapter)));
      ++adapterIndex)
    {
      DXGI_ADAPTER_DESC1 desc;
      adapter->GetDesc1(&desc);

      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
      {
        // Don't select the Basic Render Driver adapter.
        // If you want a software adapter, pass in "/warp" on the command line.
        continue;
      }

      // Check to see whether the adapter supports Direct3D 12, but don't create the
      // actual device yet.
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
      {
        break;
      }
    }
  }

  if (adapter.Get() == nullptr)
  {
    for (UINT adapterIndex = 0; SUCCEEDED(pFactory->EnumAdapters1(adapterIndex, &adapter)); ++adapterIndex)
    {
      DXGI_ADAPTER_DESC1 desc;
      adapter->GetDesc1(&desc);

      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
      {
        // Don't select the Basic Render Driver adapter.
        // If you want a software adapter, pass in "/warp" on the command line.
        continue;
      }

      // Check to see whether the adapter supports Direct3D 12, but don't create the
      // actual device yet.
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
      {
        break;
      }
    }
  }

  *ppAdapter = adapter.Detach();
}

void MVAnalyse::Init_DX12_ME(IScriptEnvironment* env, int nWidth, int nHeight, int iBlkSize, bool bChroma)
{
  // check for hardware D3D12 motion estimator support and init
  UINT dxgiFactoryFlags = 0;
  HRESULT hr;

  if (optSearchOption == 6)
  {
    nWidth /= 2; // nWidth must be divisible to 4 ?
    nHeight /= 2; // nHeight must be divisible to 4 ?
    iBlkSize /= 2; // BlkSize = 16 for MAnalyse, internally = 8 for half size search with qpel presicion
  }


#if defined(_DEBUG)
  // Enable the debug layer (requires the Graphics Tools "optional feature").
  // NOTE: Enabling the debug layer after device creation will invalidate the active device.
  {
//    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
    {
      debugController->EnableDebugLayer();

      // Enable additional debug layers.
      dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
  }
#endif

  hr = CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateDXGIFactory2()"
    );
  }

  GetHardwareAdapter(factory.Get(), &hardwareAdapter);

  if (hardwareAdapter == NULL)
  {
    env->ThrowError(
      "MAnalyse: Error GetHardwareAdapter return NULL"
    );
  }

  hr = D3D12CreateDevice(
    hardwareAdapter.Get(),
    D3D_FEATURE_LEVEL_11_0,
    IID_PPV_ARGS(&m_D3D12device)
  );

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error D3D12CreateDevice()"
    );
  }

  // Describe and create the command queue Video
  D3D12_COMMAND_QUEUE_DESC queueDescVideo = {};
  queueDescVideo.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDescVideo.Type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;

  hr = m_D3D12device->CreateCommandQueue(&queueDescVideo, IID_PPV_ARGS(&m_commandQueueVideo));

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error m_D3D12device->CreateCommandQueueVideo"
    );
  }

  // Describe and create the command queue Graphics
  D3D12_COMMAND_QUEUE_DESC queueDescGraphics = {};
  queueDescGraphics.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDescGraphics.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

  hr = m_D3D12device->CreateCommandQueue(&queueDescGraphics, IID_PPV_ARGS(&m_commandQueueGraphics));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error m_D3D12device->CreateCommandQueueGraphics"
    );
  }


  hr = m_D3D12device->QueryInterface(IID_PPV_ARGS(&dev_D3D12VideoDevice));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error m_D3D12device->QueryInterface(dev_D3D12VideoDevice)"
    );
  }

  D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR MotionEstimatorSupport = { 0u, DXGI_FORMAT_NV12 };
  HRESULT feature_support = dev_D3D12VideoDevice->CheckFeatureSupport(D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR, &MotionEstimatorSupport, sizeof(MotionEstimatorSupport));

  if (feature_support != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CheckFeatureSupport() D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR"
    );
  }

  // check if size supported
  if (nWidth > MotionEstimatorSupport.SizeRange.MaxWidth)
  {
    env->ThrowError(
      "MAnalyse: frame width not supported by DX12_ME, max supported width %d", MotionEstimatorSupport.SizeRange.MaxWidth
    );
  }

  if (nHeight > MotionEstimatorSupport.SizeRange.MaxHeight)
  {
    env->ThrowError(
      "MAnalyse: frame width not supported by DX12_ME, max supported width %d", MotionEstimatorSupport.SizeRange.MaxWidth
    );
  }

  hr = dev_D3D12VideoDevice->QueryInterface(IID_PPV_ARGS(&dev_D3D12VideoDevice1));

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error QueryInterface -> dev_D3D12VideoDevice1"
    );
  }

  D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE ME_BlockSize = D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_8X8;

  if (iBlkSize == 8)
  {
    ME_BlockSize = D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_8X8;
  }
  else
    if (iBlkSize == 16)
    {
      ME_BlockSize = D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16;
    }

  D3D12_VIDEO_MOTION_ESTIMATOR_DESC motionEstimatorDesc = {
  0, //NodeIndex
  DXGI_FORMAT_NV12,
  ME_BlockSize,
  D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL,
  {(UINT)nWidth, (UINT)nHeight, (UINT)nWidth, (UINT)nHeight} // D3D12_VIDEO_SIZE_RANGE
  };

  HRESULT vid_est_result = dev_D3D12VideoDevice1->CreateVideoMotionEstimator(
    &motionEstimatorDesc,
    nullptr,
    IID_PPV_ARGS(&spVideoMotionEstimator));

  if (vid_est_result != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateVideoMotionEstimator()"
    );
  }

  D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC MotionVectorHeapDesc = {
  0, // NodeIndex 
  DXGI_FORMAT_NV12,
  ME_BlockSize,
  D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL,
  {nWidth, nHeight, nWidth, nHeight} // D3D12_VIDEO_SIZE_RANGE
  };

  HRESULT vect_heap_result = dev_D3D12VideoDevice1->CreateVideoMotionVectorHeap(
    &MotionVectorHeapDesc,
    nullptr,
    IID_PPV_ARGS(&spVideoMotionVectorHeap));

  if (vect_heap_result != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateVideoMotionVectorHeap()"
    );
  }

  HRESULT res_motion_vectors_texture = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Tex2D(
      DXGI_FORMAT_R16G16_SINT,
      Align(nWidth, iBlkSize) / iBlkSize, 
      Align(nHeight, iBlkSize) / iBlkSize,
      1, // ArraySize
      1  // MipLevels
    ),
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(&spResolvedMotionVectors));

  if (res_motion_vectors_texture != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spResolvedMotionVectors"
    );
  }

  int iNumBlocksX = nWidth / iBlkSize;
  int iRowPitchUA = iNumBlocksX * sizeof(int); // 2x16bit
  int iMod = iRowPitchUA % 256;
  int iRowPitch = iRowPitchUA + (256 - iMod); // must be multiply of 256

  // Readback resources must be buffers
  D3D12_RESOURCE_DESC bufferDesc = {};
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.Height = 1;
  bufferDesc.Width = (int64_t)iRowPitch * (nHeight / iBlkSize) * 2; // x2 test ??
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufferDesc.MipLevels = 1;
  bufferDesc.SampleDesc.Count = 1;

  // Create a staging texture
  hr = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    nullptr,
    IID_PPV_ARGS(&spResolvedMotionVectorsReadBack));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spResolvedMotionVectorsReadBack"
    );
  }

  // SAD readback buffer
  int iSADRowPitchUA = iNumBlocksX * sizeof(short); // 16bit samples ?
  int iSADMod = iSADRowPitchUA % 256;
  int iSADRowPitch = iSADRowPitchUA + (256 - iMod); // must be multiply of 256

  // Readback resources must be buffers
  D3D12_RESOURCE_DESC bufferSADDesc = {};
  bufferSADDesc.DepthOrArraySize = 1;
  bufferSADDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferSADDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
  bufferSADDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferSADDesc.Height = 1;
  bufferSADDesc.Width = (int64_t)iSADRowPitch * (nHeight / iBlkSize); 
  bufferSADDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufferSADDesc.MipLevels = 1;
  bufferSADDesc.SampleDesc.Count = 1;

  // Create a staging texture
  hr = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
    D3D12_HEAP_FLAG_NONE,
    &bufferSADDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    nullptr,
    IID_PPV_ARGS(&spSADReadBack));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spSADReadBack"
    );
  }
    
  D3D12_RESOURCE_DESC textureDesc = {};
  textureDesc.MipLevels = 1;
  textureDesc.Format = DXGI_FORMAT_NV12;
  textureDesc.Width = nWidth;
  textureDesc.Height = nHeight;
  textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
  textureDesc.DepthOrArraySize = 1;
  textureDesc.SampleDesc.Count = 1;
  textureDesc.SampleDesc.Quality = 0;
  textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

  HRESULT res_current_texture = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &textureDesc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(&spCurrentResource));

  if (res_current_texture != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spCurrentResource"
    );
  }

  // Create the GPU upload buffer.
  // test speed alt way:
  // to make user-defined options: L0/L1 memory pool and WB/WC memory type
  D3D12_HEAP_PROPERTIES hpRUpload = {};
  hpRUpload.Type = D3D12_HEAP_TYPE_CUSTOM;
  hpRUpload.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
//  hpRUpload.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
  hpRUpload.MemoryPoolPreference = D3D12_MEMORY_POOL_L0; // L0 for discrete HWAcc, L1 may be for CPU-integrated ?

  hr = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
//    &hpRUpload,
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Buffer((uint64_t)nWidth*nHeight*sizeof(int)), // size of NV12 format ??
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&spCurrentResourceUpload));

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spCurrentResourceUpload"
    );
  }
  
  HRESULT res_ref_texture = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &textureDesc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(&spReferenceResource));

  if (res_ref_texture != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spReferenceResource"
    );
  }

  // Create the GPU upload buffer.
  hr = m_D3D12device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
//    &hpRUpload,
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Buffer((uint64_t)nWidth * nHeight * sizeof(int)), // size of NV12 format ??
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&spReferenceResourceUpload));

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommittedResource -> spReferenceResourceUpload"
    );
  }

  hr = m_D3D12device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE, IID_PPV_ARGS(&m_commandAllocatorVideo));

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandAllocatorVideo "
    );
  }

  hr = m_D3D12device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocatorGraphics));

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandAllocatorGraphics "
    );
  }

  hr = m_D3D12device->CreateCommandList(
    0,
    D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE,
    m_commandAllocatorVideo.Get(),
    NULL,
    IID_PPV_ARGS(&m_VideoEncodeCommandList)
  );

  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandList -> D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE"
    );
  }

  m_VideoEncodeCommandList->Close();

  hr = m_D3D12device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocatorGraphics.Get(), 0, IID_PPV_ARGS(&m_GraphicsCommandList));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandList -> D3D12_COMMAND_LIST_TYPE_DIRECT"
    );
  }

  m_GraphicsCommandList->Close();


  // compute init
  m_resourceState[0] = m_resourceState[1] = ResourceState_ReadyCompute;

  // create compute fence and event
  m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_computeFenceEvent == nullptr)
  {
    env->ThrowError(
      "MAnalyse: Error createEvent -> m_computeFenceEvent "
    );
  }

  // Initialize resource and descriptor heaps

  // sampler setup
  const D3D12_SAMPLER_DESC s_samplerType[] =
  {
    // MinMagMipPointUVWClamp
    {
        D3D12_FILTER_MIN_MAG_MIP_POINT,                 // Filter mode
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,               // U address clamping
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,               // V address clamping
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,               // W address clamping
        0.0F,                                           // Mip LOD bias
        0,                                              // Max Anisotropy - applies if using ANISOTROPIC filtering only
        D3D12_COMPARISON_FUNC_ALWAYS,                   // Comparison Func - always pass
        { 0.0F, 0.0F, 0.0F, 0.0F },                     // BorderColor float values - used if TEXTURE_ADDRESS_BORDER is set.
        0.0F,                                           // MinLOD
        D3D12_FLOAT32_MAX                               // MaxLOD
    },
  };
  
  {
    m_samplerDescriptorHeap = std::make_unique<DescriptorHeap>(m_D3D12device.Get(),
      D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
      1);
    m_D3D12device->CreateSampler(s_samplerType, m_samplerDescriptorHeap->GetCpuHandle(0));
  }

  m_SRVDescriptorHeap = std::make_unique<DescriptorHeap>(m_D3D12device.Get(), e_iHeapEnd);

  // create SAD texture and views
  const D3D12_HEAP_PROPERTIES defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  const D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16_UINT, nWidth / iBlkSize, nHeight / iBlkSize, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  m_resourceStateSADTexture = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

  hr = m_D3D12device->CreateCommittedResource(
      &defaultHeapProperties,
      D3D12_HEAP_FLAG_NONE,
      &texDesc,
      m_resourceStateSADTexture,
      nullptr,
      IID_PPV_ARGS(&m_SADTexture));
    if (hr != S_OK)
    {
      env->ThrowError(
        "MAnalyse: Error CreateCommandList -> D3D12_COMMAND_LIST_TYPE_DIRECT"
      );
    }


    const uint32_t s_numShaderThreads = 8;		// make sure to update value in shader if this changes

  m_ThreadGroupX = static_cast<uint32_t>(texDesc.Width) / s_numShaderThreads;
  m_ThreadGroupY = texDesc.Height / s_numShaderThreads;

  // create uav
  m_D3D12device->CreateUnorderedAccessView(m_SADTexture.Get(), nullptr, nullptr, m_SRVDescriptorHeap->GetCpuHandle(e_iUAV));

  // create srv
//  m_D3D12device->CreateShaderResourceView(m_SADTexture.Get(), nullptr, m_SRVDescriptorHeap->GetCpuHandle(e_iSRV)); // required ??

  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc_CurrRef_Y = {};
  srv_desc_CurrRef_Y.Format = DXGI_FORMAT_R8_UINT;
  srv_desc_CurrRef_Y.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc_CurrRef_Y.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc_CurrRef_Y.Texture2D.MipLevels = 1;
  srv_desc_CurrRef_Y.Texture2D.MostDetailedMip = 0;
  srv_desc_CurrRef_Y.Texture2D.PlaneSlice = 0;
  srv_desc_CurrRef_Y.Texture2D.ResourceMinLODClamp = 0;

  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc_CurrRef_UV = {};
  srv_desc_CurrRef_UV.Format = DXGI_FORMAT_R8G8_UINT;
  srv_desc_CurrRef_UV.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc_CurrRef_UV.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc_CurrRef_UV.Texture2D.MipLevels = 1;
  srv_desc_CurrRef_UV.Texture2D.MostDetailedMip = 0;
  srv_desc_CurrRef_UV.Texture2D.PlaneSlice = 1;
  srv_desc_CurrRef_UV.Texture2D.ResourceMinLODClamp = 0;

  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc_RMVs = {};
  srv_desc_RMVs.Format = DXGI_FORMAT_R16G16_SINT;
  srv_desc_RMVs.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc_RMVs.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc_RMVs.Texture2D.MipLevels = 1;
  srv_desc_RMVs.Texture2D.MostDetailedMip = 0;
  srv_desc_RMVs.Texture2D.PlaneSlice = 0;
  srv_desc_RMVs.Texture2D.ResourceMinLODClamp = 0;

  m_D3D12device->CreateShaderResourceView(spCurrentResource.Get(), &srv_desc_CurrRef_Y, m_SRVDescriptorHeap->GetCpuHandle(e_iSRV + 0));
  m_D3D12device->CreateShaderResourceView(spCurrentResource.Get(), &srv_desc_CurrRef_UV, m_SRVDescriptorHeap->GetCpuHandle(e_iSRV + 1));

  m_D3D12device->CreateShaderResourceView(spReferenceResource.Get(), &srv_desc_CurrRef_Y, m_SRVDescriptorHeap->GetCpuHandle(e_iSRV + 2));
  m_D3D12device->CreateShaderResourceView(spReferenceResource.Get(), &srv_desc_CurrRef_UV, m_SRVDescriptorHeap->GetCpuHandle(e_iSRV + 3));
  m_D3D12device->CreateShaderResourceView(spResolvedMotionVectors.Get(), &srv_desc_RMVs, m_SRVDescriptorHeap->GetCpuHandle(e_iSRV + 4));

  // load compute shader
  auto computeShaderBlob = DX::ReadData(L"Compute.cso");
  // Define root table layout
  {
    CD3DX12_DESCRIPTOR_RANGE descRange[e_numRootParameters];
    descRange[e_rootParameterSampler].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0); // s0
    descRange[e_rootParameterSRV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0); // t0
    descRange[e_rootParameterUAV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0); // u0

    CD3DX12_ROOT_PARAMETER rootParameters[e_numRootParameters];
    rootParameters[e_rootParameterCB].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);
    rootParameters[e_rootParameterSampler].InitAsDescriptorTable(1, &descRange[e_rootParameterSampler], D3D12_SHADER_VISIBILITY_ALL);
    rootParameters[e_rootParameterSRV].InitAsDescriptorTable(1, &descRange[e_rootParameterSRV], D3D12_SHADER_VISIBILITY_ALL);
    rootParameters[e_rootParameterUAV].InitAsDescriptorTable(1, &descRange[e_rootParameterUAV], D3D12_SHADER_VISIBILITY_ALL);

    CD3DX12_ROOT_SIGNATURE_DESC rootSignature(_countof(rootParameters), rootParameters);

    ComPtr<ID3DBlob> serializedSignature;
    hr = D3D12SerializeRootSignature(&rootSignature, D3D_ROOT_SIGNATURE_VERSION_1, serializedSignature.GetAddressOf(), nullptr);
    if (hr != S_OK)
    {
      env->ThrowError(
        "MAnalyse: Error D3D12SerializeRootSignature"
      );
    }

    // Create the root signature
    hr = m_D3D12device->CreateRootSignature(
        0,
        serializedSignature->GetBufferPointer(),
        serializedSignature->GetBufferSize(),
        IID_PPV_ARGS(&m_computeRootSignature));

    if (hr != S_OK)
    {
      env->ThrowError(
        "MAnalyse: Error D3D12Device -> CreateRootSignature"
      );
    }
  }

  // Create compute pipeline state
  D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
  descComputePSO.pRootSignature = m_computeRootSignature.Get();
  descComputePSO.CS.pShaderBytecode = computeShaderBlob.data();
  descComputePSO.CS.BytecodeLength = computeShaderBlob.size();

  hr = m_D3D12device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSO));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateComputePipelineState"
    );
  }

  // Create compute allocator, command queue and command list
  D3D12_COMMAND_QUEUE_DESC descCommandQueue = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE };
  hr = m_D3D12device->CreateCommandQueue(&descCommandQueue, IID_PPV_ARGS(&m_computeCommandQueue));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandQueue -> D3D12_COMMAND_LIST_TYPE_COMPUTE"
    );
  }

  hr = m_D3D12device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_computeAllocator));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandAllocator -> D3D12_COMMAND_LIST_TYPE_COMPUTE"
    );
  }

  hr = m_D3D12device->CreateCommandList(
    0,
    D3D12_COMMAND_LIST_TYPE_COMPUTE,
    m_computeAllocator.Get(),
    m_computePSO.Get(),
    IID_PPV_ARGS(&m_computeCommandList));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateCommandList -> D3D12_COMMAND_LIST_TYPE_COMPUTE"
    );
  }

  m_computeCommandList->Close();

  // upload sadCSparams
  {
    CD3DX12_HEAP_PROPERTIES uploadCBHeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferCBDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(SAD_CS_PARAMS)); // or 4KB mempage ?

    // Allocate the CB upload heap
    hr = m_D3D12device->CreateCommittedResource(
      &uploadCBHeapProperties,
      D3D12_HEAP_FLAG_NONE,
      &bufferCBDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr,
      IID_PPV_ARGS(&spSADCBResource));
    if (hr != S_OK)
    {
      env->ThrowError(
        "MAnalyse: Error CreateCommittedResource -> spSADCBResource"
      );
    }

    // Get a pointer to the memory
    void* pSADCBMemory = nullptr;
    hr = spSADCBResource->Map(0, nullptr, &pSADCBMemory);
    if (hr != S_OK)
    {
      env->ThrowError(
        "MAnalyse: Error spSADCBResource->Map"
      );
    }

    sadCBparamsBV.BufferLocation = spSADCBResource->GetGPUVirtualAddress();

    SAD_CS_PARAMS* pCBsadCSparams = reinterpret_cast<SAD_CS_PARAMS*> (pSADCBMemory);


    pCBsadCSparams->blockSizeH = iBlkSize;// srd._analysis_data.GetBlkSizeX();
    pCBsadCSparams->blockSizeV = iBlkSize;// srd._analysis_data.GetBlkSizeY();
    pCBsadCSparams->useChroma = bChroma;// (int)((srd._analysis_data.nFlags & MOTION_USE_CHROMA_MOTION) != 0);
    if (optSearchOption == 5)
    {
      pCBsadCSparams->precisionMVs = 2; // bitshift 2= /4
    }
    else // ==6
    {
      pCBsadCSparams->precisionMVs = 1; // bitshift 1 = /2
    }

    spSADCBResource->Unmap(0, nullptr);
  }

  hr = m_D3D12device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error CreateFence"
    );
  }

  m_fenceValue = 1;

  // Create an event handle to use for frame synchronization.
  m_fenceEventGraphics = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_fenceEventGraphics == nullptr)
  {
    env->ThrowError(
      "MAnalyse: Error createEvent -> m_fenceEventGraphics "
    );
  }

  m_fenceEventVideo = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_fenceEventVideo == nullptr)
  {
    env->ThrowError(
      "MAnalyse: Error createEvent -> m_fenceEventVideo "
    );
  }

  m_fenceEventCopyBack = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_fenceEventCopyBack == nullptr)
  {
    env->ThrowError(
      "MAnalyse: Error createEvent -> m_fenceEventCopyBack "
    );
  }

  // wait until CB upload finishes ?
        // Signal and increment the fence value.
  const UINT64 fence_Graphics = m_fenceValue;
  hr = m_computeCommandQueue->Signal(m_fence.Get(), fence_Graphics); // can reuse it ?
  if (hr != S_OK)
  {
    env->ThrowError(
      "MAnalyse: Error m_computeCommandQueue->Signal fence_Graphics"
    );
  }

  m_fenceValue++;

  // Wait until the previous op is finished.
  if (m_fence->GetCompletedValue() < fence_Graphics)
  {
    hr = m_fence->SetEventOnCompletion(fence_Graphics, m_computeFenceEvent);
    if (hr != S_OK)
    {
      env->ThrowError(
        "MAnalyse: Error m_fence->SetEventOnCompletion -> EventGraphics (sadCBparams upload)"
      );
    }
    WaitForSingleObject(m_computeFenceEvent, INFINITE);
  }

}

void MVAnalyse::LoadNV12(MVGroupOfFrames* pGOF, bool bChroma, int& iWidth, int& iHeight)
{
  // convert input src to NV12 buffer for upload
  uint8_t* pDstSrc = pNV12FrameData;

  // copy Y plane 8bit
  MVFrame* SrcFrame;
  if (optSearchOption == 5)
    SrcFrame = pGOF->GetFrame(0); // use 0 - largest plane (original ?? or nPel enlarged ??)
  else// (optSearchOption == 6)
    SrcFrame = pGOF->GetFrame(1); // use 1 - half sized plane (original ?? or nPel enlarged ??)

  int SrcYPitch = SrcFrame->GetPlane(YPLANE)->GetPitch();
  int src_Yx0 = SrcFrame->GetPlane(YPLANE)->GetVPadding();
  int src_Yy0 = SrcFrame->GetPlane(YPLANE)->GetHPadding();
  const BYTE* pY = SrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(src_Yx0, src_Yy0);

  iWidth = SrcFrame->GetPlane(YPLANE)->GetWidth();
  iHeight = SrcFrame->GetPlane(YPLANE)->GetHeight();
  // copy Y lines
  for (int h = 0; h < iHeight; ++h)
  {
    memcpy(pDstSrc, pY, iWidth);
    pDstSrc += iWidth;
    pY += SrcYPitch;
  }

  if (bChroma)
  {
    // interleave U and V planes
    const int src_Ux0 = SrcFrame->GetPlane(UPLANE)->GetHPadding();
    const int src_Uy0 = SrcFrame->GetPlane(UPLANE)->GetVPadding();
    const int src_Upitch = SrcFrame->GetPlane(UPLANE)->GetPitch();

    const int src_Vx0 = SrcFrame->GetPlane(VPLANE)->GetHPadding();
    const int src_Vy0 = SrcFrame->GetPlane(VPLANE)->GetVPadding();
    const int src_Vpitch = SrcFrame->GetPlane(VPLANE)->GetPitch();

    const uint8_t* pUsrc = SrcFrame->GetPlane(UPLANE)->GetAbsolutePelPointer(src_Ux0, src_Uy0);
    const uint8_t* pVsrc = SrcFrame->GetPlane(VPLANE)->GetAbsolutePelPointer(src_Vx0, src_Vy0);

    for (int h = 0; h < iHeight / 2; ++h)
    {
      for (int w = 0; w < iWidth / 2; ++w)
      {
        pDstSrc[w * 2] = pUsrc[w];
        pDstSrc[w * 2 + 1] = pVsrc[w];
      }

      pDstSrc += iWidth;
      pUsrc += src_Upitch;
      pVsrc += src_Vpitch;

    }
  }
  else // no chroma data avaialable - set grey UV
  {
    memset(pDstSrc, 128, iWidth * iHeight / 2);
  }

}



#endif // _WIN32 && DX12_ME
