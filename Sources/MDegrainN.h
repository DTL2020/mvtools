#ifndef __MV_DEGRAINN__
#define __MV_DEGRAINN__



#include	"conc/AtomicInt.h"
#include "MTSlicer.h"
#include "MVClip.h"
#include "MVFilter.h"
#include	"MVGroupOfFrames.h"
#include "overlap.h"
#include "SharedPtr.h"
#include "yuy2planes.h"
#include "def.h"

#include "MVInterface.h"

#include	<memory>
#include	<vector>

#define CACHE_LINE_SIZE 64
#define MVLPFKERNELSIZE 11 // 10+1 odd number, 10 - just some medium number relative to typical tr and allow to have some variance in slope

class MVPlane;

class MDegrainN
  : public GenericVideoFilter
  , public MVFilter
{

public:

  enum { MAX_TEMP_RAD = 128 };

  MDegrainN(
    ::PClip child, ::PClip super, ::PClip mvmulti, int trad,
    sad_t thsad, sad_t thsadc, int yuvplanes, float nlimit, float nlimitc,
    sad_t nscd1, int nscd2, bool isse_flag, bool planar_flag, bool lsb_flag,
    sad_t thsad2, sad_t thsadc2, bool mt_flag, bool out16_flag, int wpow,
    float adjSADzeromv, float adjSADcohmv, int thCohMV,
    float fMVLPFCutoff, float fMVLPFSlope, float fMVLPFGauss, int thMVLPFCorr, float adjSADLPFedmv,
    int UseSubShift, int InterpolateOverlap, ::PClip _mvmultirs, int _thFWBWmvpos, int _thPostProc1, int _iPP1NumSkip,
    ::IScriptEnvironment* env_ptr
  );
  ~MDegrainN();

  ::PVideoFrame __stdcall GetFrame(int n, ::IScriptEnvironment* env_ptr) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
  }


protected:

private:
  bool has_at_least_v8;

  typedef void (DenoiseNFunction)(
    BYTE *pDst, BYTE *pDstLsb, int nDstPitch,
    const BYTE *pSrc, int nSrcPitch,
    // 2*k = ref backwards, 2*k+1 = ref forwards
    const BYTE *pRef[], int Pitch[],
    // 0 = src, 2*k+1 = ref backwards, 2*k+2 = ref forwards
    int Wall[], int trad
    );

  DenoiseNFunction* get_denoiseN_function(int BlockX, int BlockY, int _bits_per_pixel, bool _lsb_flag, bool _out16_flag, arch_t arch);

  class MvClipInfo
  {
  public:
    SharedPtr <MVClip> _clip_sptr;
    SharedPtr <MVClip> _cliprs_sptr; // separate MVclip reverse search provided
    SharedPtr <MVGroupOfFrames> _gof_sptr;
    sad_t _thsad;
    sad_t _thsadc;
    double _thsad_sq;
    double _thsadc_sq;
  };
  typedef std::vector <MvClipInfo> MvClipArray;

  typedef MTSlicer <MDegrainN> Slicer;

  class TmpBlock
  {
  public:
    enum { MAX_SIZE = MAX_BLOCK_SIZE };
    enum { AREA = MAX_SIZE * MAX_SIZE };
    TmpBlock() : _lsb_ptr(&_d[AREA]) {}
    unsigned char  _d[MAX_SIZE * MAX_SIZE * 2]; // * 2 for 8 bit MSB and LSB parts, or for 1*uint16_t
    unsigned char* _lsb_ptr;// Not allocated, it's just a reference to a part of the _d area
  };

  MV_FORCEINLINE int reorder_ref(int index) const;
  template <int P>
  MV_FORCEINLINE void process_chroma(int plane_mask);

  void process_luma_normal_slice(Slicer::TaskData &td);
  void process_luma_overlap_slice(Slicer::TaskData &td);
  void process_luma_overlap_slice(int y_beg, int y_end);

  template <int P>
  void process_chroma_normal_slice(Slicer::TaskData &td);
  template <int P>
  void process_chroma_overlap_slice(Slicer::TaskData &td);
  template <int P>
  void process_chroma_overlap_slice(int y_beg, int y_end);

  // 2.7.46
  void process_luma_and_chroma_normal_slice(Slicer::TaskData& td); // for faster MVLPF proc, and faster total as single pass YUV proc ?
  void process_luma_and_chroma_overlap_slice(Slicer::TaskData& td);
  void process_luma_and_chroma_overlap_slice(int y_beg, int y_end); // for faster MVLPF proc, and faster total as single pass YUV proc ?

  
  MV_FORCEINLINE void
    use_block_y(
      const BYTE * &p, int &np, int &wref, bool usable_flag, const MvClipInfo &c_info,
      int i, const MVPlane *plane_ptr, const BYTE *src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray 
    );

  MV_FORCEINLINE void
    use_block_y_thSADzeromv_thSADcohmv(
      const BYTE*& p, int& np, int& wref, bool usable_flag, const MvClipInfo& c_info,
      int i, const MVPlane* plane_ptr, const BYTE* src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
    );

  MV_FORCEINLINE void
    use_block_uv(
      const BYTE * &p, int &np, int &wref, bool usable_flag, const MvClipInfo &c_info,
      int i, const MVPlane *plane_ptr, const BYTE *src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
    );

  MV_FORCEINLINE void
    use_block_uv_thSADzeromv_thSADcohmv(
      const BYTE*& p, int& np, int& wref, bool usable_flag, const MvClipInfo& c_info,
      int i, const MVPlane* plane_ptr, const BYTE* src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
  );

  MV_FORCEINLINE void
    use_block_yuv(
      const BYTE*& pY, int& npY, const BYTE*& pUV1, int& npUV1, const BYTE*& pUV2, int& npUV2, int& wref, int& wrefUV, bool usable_flag, const MvClipInfo& c_info,
      int i, const MVPlane* plane_ptrY, const BYTE* src_ptrY, const MVPlane* plane_ptrUV1, const BYTE* src_ptrUV1, const MVPlane* plane_ptrUV2, const BYTE* src_ptrUV2,
      int xx, int xx_uv, int src_pitchY, int src_pitchUV1, int src_pitchUV2, int ibx, int iby, const VECTOR* pMVsArray
    );
  
  void(MDegrainN::* use_block_y_func)(
    const BYTE*& p, int& np, int& wref, bool usable_flag, const MvClipInfo& c_info,
    int i, const MVPlane* plane_ptr, const BYTE* src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
    ); // selector for old and alt proc

  void(MDegrainN::* use_block_uv_func)(
    const BYTE*& p, int& np, int& wref, bool usable_flag, const MvClipInfo& c_info,
    int i, const MVPlane* plane_ptr, const BYTE* src_ptr, int xx, int src_pitch, int ibx, int iby, const VECTOR* pMVsArray
    ); // selector for old and alt proc


  static MV_FORCEINLINE void
    norm_weights(int wref_arr[], int trad);

  void FilterMVs(void);
  MV_FORCEINLINE void FilterBlkMVs(int i, int bx, int by);
  MV_FORCEINLINE void PrefetchMVs(int i);

  MvClipArray _mv_clip_arr;
  
  int _trad;// Temporal radius (nbr frames == _trad * 2 + 1)
  int _yuvplanes;
  float _nlimit;
  float _nlimitc;
  PClip _super;
  int _cpuFlags;
  const bool _planar_flag;
  const bool _lsb_flag;
  const bool _out16_flag;
  const bool _mt_flag;
  int _height_lsb_or_out16_mul;
  //int pixelsize, bits_per_pixel; // in MVFilter
  //int xRatioUV, yRatioUV; // in MVFilter
  int pixelsize_super;
  int bits_per_pixel_super;
  int pixelsize_super_shift;
  int xRatioUV_super, nLogxRatioUV_super; // 2.7.39-
  int yRatioUV_super, nLogyRatioUV_super;

  // 2.7.26
  int pixelsize_output;
  int bits_per_pixel_output;
  int pixelsize_output_shift;

  int _nsupermodeyuv;

  // 2.7.46
  int _wpow;
//  const VECTOR* pMVsPlanesArrays[MAX_TEMP_RAD * 2];
//  const VECTOR* pMVsPlanesArraysRS[MAX_TEMP_RAD * 2]; // reverse search

  float fadjSADzeromv;
  float fadjSADcohmv;
  float fadjSADLPFedmv;
  int ithCohMV;

  float fMVLPFCutoff;
  float fMVLPFSlope;
  float fMVLPFGauss;
  int ithMVLPFCorr;
  bool bMVsAddProc; // bool indicate if additional processing of incoming MVs were performed and read must be from pFilteredMVsPlanesArrays (or even later in the future ?)
  float fMVLPFKernel[MVLPFKERNELSIZE];// 10+1 odd numbered
  VECTOR* pFilteredMVsPlanesArrays[MAX_TEMP_RAD * 2];
  const uint8_t* pFilteredMVsPlanesArrays_a[MAX_TEMP_RAD * 2]; // pointers to aligned memory pages to free
  SADFunction* SAD;              /* function which computes the sad */
  SADFunction* SADCHROMA;

  bool bthLC_diff;

  int iInterpolateOverlap;
  VECTOR* pMVsIntOvlpPlanesArrays[MAX_TEMP_RAD * 2]; // interpolated overlap MVs
  const uint8_t* pMVsIntOvlpPlanesArrays_a[MAX_TEMP_RAD * 2]; // pointers to aligned memory pages to free
  int nInputBlkX;
  int nInputBlkY;
  int nInputBlkCount;
  void InterpolateOverlap_4x(VECTOR* pInterpolatedMVs, const VECTOR* pInputMVs, int idx);
  void InterpolateOverlap_2x(VECTOR* pInterpolatedMVs, const VECTOR* pInputMVs, int idx);
  bool bDiagOvlp;
  VECTOR* pMVsWorkPlanesArrays[MAX_TEMP_RAD * 2]; // curernt working MVs
  VECTOR* pMVsWorkPlanesArraysRS[MAX_TEMP_RAD * 2]; // curernt working MVs reverse search
  sad_t veryBigSAD;
  MV_FORCEINLINE sad_t CheckSAD(int bx_src, int by_src, int ref_idx, int dx_ref, int dy_ref);

  int nUseSubShift;
  int iMaxBlx; // max blx for GetPointer*()
  int iMinBlx; // min blx for GetPointer*()
  int iMaxBly; // max bly for GetPointer*()
  int iMinBly; // min bly for GetPointer*()

  float fSinc(float x);

  PClip mvmultirs;
  MV_FORCEINLINE void ProcessRSMVdata(void);
  int thFWBWmvpos;

  int thPostProc1;
  int iPP1NumSkip;
  uint8_t* pSubtrTempBlocks; // single area to hold temporal single block subtracted blended results, contiguos in memory so may not cause cache aliasing
  MV_FORCEINLINE uint8_t* PostProc1(const BYTE* pRef[], int Pitch[], int Wall[], int iBlkWidth, int iBlkHeight);
  MV_FORCEINLINE int FindBadBlock(const BYTE* pRef[], int Pitch[], int Wall[], int iBlkWidth, int iBlkHeight);
  MV_FORCEINLINE void CopyBlock(uint8_t* pDst, int iDstPitch, uint8_t* pSrc, int iBlkWidth, int iBlkHeight);

  std::unique_ptr <YUY2Planes> _dst_planes;
  std::unique_ptr <YUY2Planes> _src_planes;

  std::unique_ptr <OverlapWindows> _overwins;
  std::unique_ptr <OverlapWindows> _overwins_uv;

  OverlapsFunction *_oversluma_ptr;
  OverlapsFunction *_overschroma_ptr;
  OverlapsFunction *_oversluma16_ptr;
  OverlapsFunction *_overschroma16_ptr;
  OverlapsFunction *_oversluma32_ptr;
  OverlapsFunction *_overschroma32_ptr;
  OverlapsLsbFunction *_oversluma_lsb_ptr;
  OverlapsLsbFunction *_overschroma_lsb_ptr;
  DenoiseNFunction *_degrainluma_ptr;
  DenoiseNFunction *_degrainchroma_ptr;

  LimitFunction_t *LimitFunction;

// -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
// Processing variables

  std::vector <uint16_t> _dst_short;
  int _dst_short_pitch;
  std::vector <int> _dst_int;
  int _dst_int_pitch;

  // 2.7.46 for single pass proc of YUV
  std::vector <uint16_t> _dst_shortUV1;
  std::vector <uint16_t> _dst_shortUV2;
  std::vector <int> _dst_intUV1;
  std::vector <int> _dst_intUV2;
  bool bYUVProc;
  MV_FORCEINLINE void MemZoneSetY(uint16_t* pDstShort, int* pDstInt);
  MV_FORCEINLINE void MemZoneSetUV(uint16_t* pDstShortUV, int* pDstIntUV);
  MV_FORCEINLINE void post_overlap_luma_plane(void);
  
  MV_FORCEINLINE void post_overlap_chroma_plane(int P, uint16_t* pDstShort, int* pDstInt);

  MV_FORCEINLINE void nlimit_luma(void);
  MV_FORCEINLINE void nlimit_chroma(int P);

  bool _usable_flag_arr[MAX_TEMP_RAD * 2];
  MVPlane *_planes_ptr[MAX_TEMP_RAD * 2][3];
  BYTE *_dst_ptr_arr[3];
  const BYTE *_src_ptr_arr[3];
  int _dst_pitch_arr[3];
  int _src_pitch_arr[3];
  int _lsb_offset_arr[3];
  int _covered_width;
  int _covered_height;

  // This array has an nBlkY size. It is used in vertical overlap mode
  // to avoid read/write sync problems when processing is multithreaded.
  // Only elements corresponding to the first row of each sub-plane are
  // actually used. They count how many sub-planes (excepted their last
  // row) have been processed on each side of the boundary. When a counter
  // reaches 2, the boundary row (just above the element position) can be
  // processed safely.
  std::vector <conc::AtomicInt <int> > _boundary_cnt_arr;
};

MV_FORCEINLINE int DegrainWeightN(int thSAD, double thSAD_pow, int blockSAD, int wpow);

MV_FORCEINLINE unsigned int SADABS(int x) { return (x < 0) ? -x : x; }


#endif
