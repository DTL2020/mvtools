// Make a motion compensate temporal denoiser

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

#ifndef __MV_ANALYSE__
#define __MV_ANALYSE__

#include "conc/ObjPool.h"
#include "DCTFactory.h"
#include "GroupOfPlanes.h"
#include "MVAnalysisData.h"
#include "yuy2planes.h"

#include "avisynth.h"

#include <memory>
#include <vector>

#if defined _WIN32 //&& defined DX_12ME

#include <initguid.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include "d3dx12.h"
#include "d3d12video.h"
#include "DirectXHelpers.h"

#include <string>
#include <wrl.h>
#include <shellapi.h>

using Microsoft::WRL::ComPtr;

#endif

class MVAnalyse
  : public GenericVideoFilter
{
protected:
  bool has_at_least_v8;

  int _instance_id; // debug unique id

  // One instance per Src/Ref combination
  // Multi mode order (bwd/fwd delta): B1, F1, B2, F2, B3, F3...
  // In single mode, only the first element is used.
  class SrcRefData
  {
  public:
    MVAnalysisData _analysis_data;
    MVAnalysisData _analysis_data_divided;

    std::vector<int> _vec_prev;
    int _vec_prev_frame;
  };

  typedef std::vector<SrcRefData> SrcRefArray;

  SrcRefArray _srd_arr;

  /*! \brief Frames of blocks for which motion vectors will be computed */
  std::unique_ptr<GroupOfPlanes> _vectorfields_aptr; // Temporary data, structure initialised once.

  /*! \brief isse optimisations enabled */
  int cpuFlags;

  /*! \brief motion vecteur cost factor */
  int nLambda;

  /*! \brief search type chosen for refinement in the EPZ */
  SearchType searchType;

  /*! \brief additionnal parameter for this search */
  int nSearchParam; // usually search radius

  int nPelSearch; // search radius at finest level

  sad_t lsad; // SAD limit for lambda using - added by Fizick
  int pnew; // penalty to cost for new canditate - added by Fizick
  int plen; // penalty factor (similar to lambda) for vector length - added by Fizick
  int plevel; // penalty factors (lambda, plen) level scaling - added by Fizick
  bool global; // use global motion predictor
  int pglobal; // penalty factor for global motion predictor
  int pzero; // penalty factor for zero vector
  const char* outfilename;// vectors output file
  int divideExtra; // divide blocks on sublocks with median motion
  sad_t badSAD; //  SAD threshold to make more wide search for bad vectors
  int badrange;// range (radius) of wide search
  bool meander; //meander (alternate) scan blocks (even row left to right, odd row right to left
  bool tryMany; // try refine around many predictors
  const bool _multi_flag;
  const bool _temporal_flag;
  const bool _mt_flag;
  // 'opt' beginning until live during tests
  int optSearchOption; // DTL test
  int optPredictorType; // DTL test

  int pixelsize; // PF
  int bits_per_pixel;

  FILE *outfile;
  short * outfilebuf;

  //	YUY2Planes * SrcPlanes;
  //	YUY2Planes * RefPlanes;

  std::unique_ptr<DCTFactory> _dct_factory_ptr; // Not instantiated if not needed
  conc::ObjPool<DCTClass> _dct_pool;

  int headerSize;

  MVGroupOfFrames *pSrcGOF, *pRefGOF; //v2.0. Temporary data, structure initialised once.

  int nModeYUV;

  int _delta_max;

  int iUploadedCurrentFrameNum;

public:

  MVAnalyse(
    PClip _child, int _blksizex, int _blksizey, int lv, int st, int stp,
    int _pelSearch, bool isb, int lambda, bool chroma, int df, sad_t _lsad,
    int _plevel, bool _global, int _pnew, int _pzero, int _pglobal,
    int _overlapx, int _overlapy, const char* _outfilename, int _dctmode,
    int _divide, int _sadx264, sad_t _badSAD, int _badrange, bool _isse,
    bool _meander, bool temporal_flag, bool _tryMany, bool multi_flag,
    bool mt_flag, int _chromaSADScale, int _optSearchOption, int _predictorType, IScriptEnvironment* env);
  ~MVAnalyse();

  ::PVideoFrame __stdcall	GetFrame(int n, ::IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? (_temporal_flag || lstrlen(outfilename)>0 ? MT_SERIALIZED : MT_MULTI_INSTANCE) : 0;
    // adaptive!
    // temporal = true or using output file is not MT-friendly
  }

private:

  void load_src_frame(MVGroupOfFrames &gof, ::PVideoFrame &src, const MVAnalysisData &ana_data);

#if defined _WIN32 && defined DX12_ME
//  ComPtr<ID3D12Device> m_device;
  inline UINT Align(UINT size, UINT alignment)
  {
    return (size + (alignment - 1)) & ~(alignment - 1);
  }

  ComPtr<IDXGIFactory4> factory;
  ComPtr<IDXGIAdapter1> hardwareAdapter;
  ComPtr<ID3D12Device> m_D3D12device;
  ComPtr<ID3D12VideoDevice> dev_D3D12VideoDevice;
  ComPtr<ID3D12VideoDevice1> dev_D3D12VideoDevice1;
  ComPtr<ID3D12VideoMotionEstimator> spVideoMotionEstimator;
  ComPtr<ID3D12VideoMotionVectorHeap> spVideoMotionVectorHeap;
  ComPtr<ID3D12Resource> spResolvedMotionVectors;
  ComPtr<ID3D12Resource> spResolvedMotionVectorsReadBack;
  ComPtr<ID3D12Resource> spCurrentResource;
  ComPtr<ID3D12Resource> spCurrentResourceUpload;
  ComPtr<ID3D12Resource> spReferenceResource;
  ComPtr<ID3D12Resource> spReferenceResourceUpload;
  ComPtr<ID3D12VideoEncodeCommandList> m_VideoEncodeCommandList;
  ComPtr<ID3D12GraphicsCommandList> m_GraphicsCommandList;
  ComPtr<ID3D12CommandAllocator> m_commandAllocatorGraphics;
  ComPtr<ID3D12CommandAllocator> m_commandAllocatorVideo;
  ComPtr<ID3D12CommandQueue> m_commandQueueVideo;
  ComPtr<ID3D12CommandQueue> m_commandQueueGraphics;

  HANDLE m_fenceEventGraphics;
  HANDLE m_fenceEventVideo;
  HANDLE m_fenceEventCopyBack;
  ComPtr<ID3D12Fence> m_fence;
  UINT64 m_fenceValue;

  ComPtr<ID3D12Debug> debugController;


  void GetHardwareAdapter(
    _In_ IDXGIFactory1* pFactory,
    _Outptr_result_maybenull_ IDXGIAdapter1** ppAdapter,
    bool requestHighPerformanceAdapter = false);

  void Init_DX12_ME(IScriptEnvironment* env, int nWidth, int nHeight, int iBlkSize);
#endif 
};

#endif
