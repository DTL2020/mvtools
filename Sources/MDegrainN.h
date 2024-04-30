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
#include "DisMetric.h"
#include "BlockArea.h"
#include "dm_cache.h"
#include "SADFunctions.h"

#include	<memory>
#include	<vector>

#define CACHE_LINE_SIZE 64
#define MVLPFKERNELSIZE 11 // 10+1 odd number, 10 - just some medium number relative to typical tr and allow to have some variance in slope

enum PMode
{
  PM_BLEND = 0,
  PM_MEL = 1,
};

enum DN_Mask_Mode
{
  DN_MM_NONE = 0,
  DN_MM_BLOCKS = 1,
  DN_MM_SAMPLES = 2,
};

enum PN_Mask_Mode
{
  PM_INPUT = 0x01, // Input MVs
  PM_MVF = 0x02,   // MVF output MVs
};

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
    int UseSubShift, int InterpolateOverlap, ::PClip _mvmultirs, int _thFWBWmvpos,
    int _MPBthSub, int _MPBthAdd, int _MPBNumIt, float _MPB_SPC_sub, float _MPB_SPC_add, bool _MPB_PartBlend,
    int _MPBthIVS, bool _showIVSmask, ::PClip _mvmultivs, int _MPB_DMFlags, int _MPBchroma, int _MPBtgtTR,
    int _MPB_MVlth, int _pmode, int _TTH_DMFlags, int _TTH_thUPD, int _TTH_BAS, bool _TTH_chroma, ::PClip _dnmask,
    float _thSADA_a, float _thSADA_b, int _MVMedF, int _MVMedF_em, int _MVMedF_cm, int _MVF_fm,
    int _MGR, int _MGR_sr, int _MGR_st, int _MGR_pm,
    int _LtComp, int _NEW_DMFlags,
    ::IScriptEnvironment* env_ptr
  );
  ~MDegrainN();

  ::PVideoFrame __stdcall GetFrame(int n, ::IScriptEnvironment* env_ptr) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    //    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    // if any IIR-type processing enabled - set MT_SERIALIZED
    return cachehints == CACHE_GET_MTMODE ? ((TTH_thUPD > 0 ) ? MT_SERIALIZED : MT_MULTI_INSTANCE) : 0;
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
    SharedPtr <MVClip> _clipvs_sptr; // separate MVclip vectors stable check provided
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

  MV_FORCEINLINE void
    use_block_yuv_mel(
      const BYTE*& pY, int& npY, const BYTE*& pUV1, int& npUV1, const BYTE*& pUV2, int& npUV2, bool usable_flag, const MvClipInfo& c_info,
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

  // MVs temporal filtering
  float fMVLPFCutoff;
  float fMVLPFSlope;
  float fMVLPFGauss;
  int ithMVLPFCorr;
  int iMVMedF; // MV Median - like filterting radius, 0 - default disabled
  int iMVMedF_em; // MV Median-like filterting temporal edges processing mode: 0 - use all edge MVs, 1 - skip non-filtered MVs (invalidate SAD)
  int iMVMedF_cm; // MV Median-like filterting temporal coordinates processing mode: 0 - use separated x,y filtering, 1 - use vector length dismetric
  int iMVF_fm; // MVF_fm - MV filtering blocks fail mode: 0 - pass blocks with too bad filtered MVs SADs to blending, 1 - invalidate blocks with too bad filtered MVs SADs (skip from blending)
  bool bMVsAddProc; // bool indicate if additional processing of incoming MVs were performed and read must be from pFilteredMVsPlanesArrays (or even later in the future ?)
  float fMVLPFKernel[MVLPFKERNELSIZE];// 10+1 odd numbered
  MV_FORCEINLINE void ProcessMVLPF(VECTOR* pVin, VECTOR* pVout);
  MV_FORCEINLINE void ProcessMVMedF(VECTOR* pVin, VECTOR* pVout);

  MV_FORCEINLINE void MVMedF_xy(VECTOR* pVin, VECTOR* pVout);
  MV_FORCEINLINE void MVMedF_vl(VECTOR* pVin, VECTOR* pVout);
  MV_FORCEINLINE void MVMedF_vad(VECTOR* pVin, VECTOR* pVout);
  MV_FORCEINLINE void MVMedF_mg(VECTOR* pVin, VECTOR* pVout);
  MV_FORCEINLINE void MVMedF_IQM(VECTOR* pVin, VECTOR* pVout);
  

  VECTOR* pFilteredMVsPlanesArrays[MAX_TEMP_RAD * 2];
  const uint8_t* pFilteredMVsPlanesArrays_a[MAX_TEMP_RAD * 2]; // pointers to aligned memory pages to free
  SADFunction* SAD;              /* function which computes the sad */
  SADFunction* SADCHROMA;
  int iNEW_DMFlags;


  // Multi-generation MVs refining
  int iMGR; // multi-generation MVs refining processing. Integer number of additional refining generations. 0 - disabled.
  int iMGR_sr; // search radius
  int iMGR_st; // search type, 0 - NStepSearch, 1 - Logariphmic/Diamond, 2 - Exhaustive, 3 - Hexagon, 4 - UMH ?
  int iMGR_pm;  // predictors bitmask (1 - input source, 2 - after MVF)

  // Lighting compensation
  int iLtComp; // 0 - disabled, 1 - DC compensation only mode
  uint8_t* pCompRefsBlksY;

  // Single iteration degrain blend (support both normal and overlap blending modes)
  MV_FORCEINLINE void DegrainBlendBlock_LC(
    BYTE* pDst, BYTE* pDstLsb, int iDstPitch,
    const BYTE* pSrc, 
    BYTE* pDstUV1, BYTE* pDstLsbUV1, int iDstPitchUV1,
    const BYTE* pSrcUV1, 
    BYTE* pDstUV2, BYTE* pDstLsbUV2, int iDstPitchUV2,
    const BYTE* pSrcUV2, 
    int iBlkNum, int ibx, int iby, int xx, int xx_uv
  );


  // multi-pass blending luma and chroma planes
  MV_FORCEINLINE void MGR_LC(
    BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
    const BYTE* pSrc,
    BYTE* pDstUV1, BYTE* pDstLsbUV1, int nDstPitchUV1,
    const BYTE* pSrcUV1,
    BYTE* pDstUV2, BYTE* pDstLsbUV2, int nDstPitchUV2,
    const BYTE* pSrcUV2,
    int iBlkNum,int ibx, int iby, int xx, int xx_uv
  );

  MV_FORCEINLINE void RefineMVs(
    BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
    const BYTE* pSrc,
    BYTE* pDstUV1, BYTE* pDstLsbUV1, int nDstPitchUV1,
    const BYTE* pSrcUV1,
    BYTE* pDstUV2, BYTE* pDstLsbUV2, int nDstPitchUV2,
    const BYTE* pSrcUV2,
    int iBlkNum, int ibx, int iby, int xx, int xx_uv
  );

  struct TEMPORAL_MVS
  {
    VECTOR vMVs[MAX_TEMP_RAD * 2];
  };

  MV_FORCEINLINE void RefineMV(
    BYTE* pDst, BYTE* pDstLsb, int iDstPitch,
    BYTE* pDstUV1, BYTE* pDstLsbUV1, int iDstPitchUV1,
    BYTE* pDstUV2, BYTE* pDstLsbUV2, int iDstPitchUV2,
    VECTOR Predictor0, VECTOR Predictor1, VECTOR* Refined, int k,
    int ibx, int iby
  );

  MV_FORCEINLINE void ExpandingSearch(
    BYTE* pDst, int iDstPitch,
    BYTE* pDstUV1, int iDstPitchUV1,
    BYTE* pDstUV2, int iDstPitchUV2,
    int bx_src, int by_src, // numbers of blocks
    int ref_idx,
    int r, int s, int mvx, int mvy, VECTOR* Refined);

  MV_FORCEINLINE sad_t GetSAD(
    BYTE* pSrc, int iSrcPitch,
    BYTE* pSrcUV1, int iSrcPitchUV1,
    BYTE* pSrcUV2, int iSrcPitchUV2,
    int bx_src, int by_src,
    int ref_idx, int dx_ref, int dy_ref);

  DisMetric* DM_Luma;
  DisMetric* DM_Chroma;

  DisMetric* DM_TTH_Luma;
  DisMetric* DM_TTH_Chroma;

  DisMetric* DM_NEW_Luma;
  DisMetric* DM_NEW_Chroma;

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
  const VECTOR* pMVsPlanesArraysVS[MAX_TEMP_RAD * 2]; // curernt working MVs IVS check
  VECTOR* pMVsIntOvlpPlanesArraysVS[MAX_TEMP_RAD * 2]; // interpolated overlap MVs
  const uint8_t* pMVsIntOvlpPlanesArraysVS_a[MAX_TEMP_RAD * 2]; // pointers to aligned memory pages to free
  sad_t veryBigSAD;
  MV_FORCEINLINE sad_t CheckSAD(int bx_src, int by_src, int ref_idx, int dx_ref, int dy_ref);

  MV_FORCEINLINE sad_t GetDM(int bx_src, int by_src, int ref_idx, int dx_ref, int dy_ref);

  int nUseSubShift;
  int iMaxBlx; // max blx for GetPointer*()
  int iMinBlx; // min blx for GetPointer*()
  int iMaxBly; // max bly for GetPointer*()
  int iMinBly; // min bly for GetPointer*()

  float fSinc(float x);

  PClip mvmultirs;
  PClip mvmultivs;
  MV_FORCEINLINE void ProcessRSMVdata(void);
  int thFWBWmvpos;

  PClip dnmask;
  DN_Mask_Mode dn_mm;
  PVideoFrame src_dnmask;
  BYTE* pDNMask;
  int dnmask_pitch;

  static MV_FORCEINLINE void apply_dn_mask_weights(int wref_arr[], int trad, int iDN_MM_weight);


  // Multi Pass Blending
  int MPBthSub;
  int MPBthAdd;
  int MPBNumIt;
  int MPB_DMFlags;
  float MPB_SPC_sub;
  float MPB_SPC_add;
  int MPB_thIVS;
  bool showIVSmask;
  int MPBchroma; // bit 0 - analysis (1 - use, 0 - not use)
  int MPBtgtTR; // MPB target tr for initial bleng
  int MPB_MVlth;
  bool MPB_PartBlend; // false if using faster blocksubtract (may be lower precison/quality), true if use real partial blend with skipped tested block
  uint8_t* pMPBTempBlocks; // single area to hold temporal single block subtracted blended results, contiguos in memory so may not cause cache aliasing
  uint8_t* pMPBTempBlocksUV1; // single area to hold temporal single block subtracted blended results, contiguos in memory so may not cause cache aliasing
  uint8_t* pMPBTempBlocksUV2; // single area to hold temporal single block subtracted blended results, contiguos in memory so may not cause cache aliasing
  COVARFunction* COVAR;              /* function which computes the covariance */
  COVARFunction* COVARCHROMA;

  PMode pmode;
  int TTH_DMFlags;
  int TTH_thUPD;
  int TTH_BAS;
  bool TTH_chroma;
  BlockArea** BA_Yarr;
  BlockArea** BA_UV1arr;
  BlockArea** BA_UV2arr;

  DM_cache** DM_cache_arr;
  int iFrameNumRequested;
  MV_FORCEINLINE int abs_frame_offset(int index);

  uint8_t* pMELmemY;
  uint8_t* pMELmemUV1;
  uint8_t* pMELmemUV2;

  // memory for current stored blocks MEL sums (may be better to make structure with pMELmem, or add to BlockArea class ?)
  int* pMELmemYSum;
  int* pMELmemUV1Sum;
  int* pMELmemUV2Sum;

  // single plane only
  MV_FORCEINLINE int AlignBlockWeights(const BYTE* pRef[], int Pitch[],
    const BYTE* pCurr, int iCurrPitch, int Wall[], int iBlkWidth,
    int iBlkHeight, bool bChroma, int iBlkNum);

  // luma and chroma
  MV_FORCEINLINE int AlignBlockWeightsLC(const BYTE* pRef[], int Pitch[],
    const BYTE* pRefUV1[], int PitchUV1[],
    const BYTE* pRefUV2[], int PitchUV2[],
    const BYTE* pCurr, const int iCurrPitch,
    const BYTE* pCurrUV1, const int iCurrPitchUV1,
    const BYTE* pCurrUV2, const int iCurrPitchUV2,
    int Wall[], const int iBlkWidth, const int iBlkHeight,
    const int iBlkWidthC, const int iBlkHeightC, const int chromaSADscale,
    int iBlkNum
  );

  MV_FORCEINLINE int AlignBlockWeightsLC_CV(const BYTE* pRef[], int Pitch[],
    const BYTE* pRefUV1[], int PitchUV1[],
    const BYTE* pRefUV2[], int PitchUV2[],
    const BYTE* pCurr, const int iCurrPitch,
    const BYTE* pCurrUV1, const int iCurrPitchUV1,
    const BYTE* pCurrUV2, const int iCurrPitchUV2,
    int Wall[], const int iBlkWidth, const int iBlkHeight,
    const int iBlkWidthC, const int iBlkHeightC, const int chromaSADscale,
    int iBlkNum
  );

  //multi-pass blending single plane only
  MV_FORCEINLINE void MPB_SP(
    BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
    const BYTE* pSrc, int nSrcPitch,
    const BYTE* pRef[], int Pitch[],
    int Wall[], const int iBlkWidth, const int iBlkHeight,
    bool bChroma, int iBlkNum
  );

  // multi-pass blending luma and chroma planes
  MV_FORCEINLINE void MPB_LC(
    BYTE* pDst, BYTE* pDstLsb, int nDstPitch,
    const BYTE* pSrc, int nSrcPitch,
    const BYTE* pRef[], int Pitch[],
    BYTE* pDstUV1, BYTE* pDstLsbUV1, int nDstPitchUV1,
    const BYTE* pSrcUV1, int nSrcPitchUV1,
    const BYTE* pRefUV1[], int PitchUV1[],
    BYTE* pDstUV2, BYTE* pDstLsbUV2, int nDstPitchUV2,
    const BYTE* pSrcUV2, int nSrcPitchUV2,
    const BYTE* pRefUV2[], int PitchUV2[],
    int Wall[], int WallC[], const int iBlkWidth, const int iBlkHeight,
    const int iBlkWidthC, const int iBlkHeightC, const int chromaSADscale,
    int iBlkNum
  );


  // MEL select luma and chroma planes
  MV_FORCEINLINE void MEL_LC(
    BYTE* pDstCur, int iDstPitch,
    const BYTE* pSrcCur,
    BYTE* pDstCurUV1, int iDstUV1Pitch,
    const BYTE* pSrcCurUV1,
    BYTE* pDstCurUV2, int iDstUV2Pitch,
    const BYTE* pSrcCurUV2,
    int xx, int xx_uv, int ibx, int iby, int iBlkNum
  );

#ifdef _DEBUG
  //MEL debug stat
  int iMEL_non_zero_blocks;
  int iMEL_mem_hits;
  int iMEL_mem_updates;
  int iDM_cache_hits;
#endif

  MV_FORCEINLINE void CopyBlock(uint8_t* pDst, int iDstPitch, uint8_t* pSrc, int iBlkWidth, int iBlkHeight);
  MV_FORCEINLINE void norm_weights_all(int wref_arr[], int trad);

  MV_FORCEINLINE bool isMVsStable(VECTOR** pMVsPlanesArrays, int iNumBlock, int wref_arr[]);
  //pMVsPlanesArrays may be not work (interfiltered, prefiltered or other but separately provided with most noised and most noise-search Manalyse settings
  //like truemotion=false, low lambda and lsad, zero penalties

  // auto-thSAD
  float thSADA_a; // a-param (multiplier) of auto-thSAD calculation
  float thSADA_b; // b-param (additive) of auto-thSAD calculation
  int thSAD_param_norm;
  int thSAD2_param_norm;
  float fthSAD12_ratio;
  int thSADC_param_norm;
  int thSADC2_param_norm;
  float fthSADC12_ratio;
  float fthSAD_LC_ratio;
  int thSCD1;
  MV_FORCEINLINE void CalcAutothSADs(void);

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


#endif
