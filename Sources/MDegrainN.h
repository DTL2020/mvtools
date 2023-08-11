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

#include	<memory>
#include	<vector>



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
    sad_t thsad2, sad_t thsadc2, bool mt_flag, bool out16_flag,
    float _thSADA_a, float _thSADA_b, ::IScriptEnvironment* env_ptr
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

  MV_FORCEINLINE void
    use_block_y(
      const BYTE * &p, int &np, int &wref, bool usable_flag, const MvClipInfo &c_info,
      int i, const MVPlane *plane_ptr, const BYTE *src_ptr, int xx, int src_pitch
    );
  MV_FORCEINLINE void
    use_block_uv(
      const BYTE * &p, int &np, int &wref, bool usable_flag, const MvClipInfo &c_info,
      int i, const MVPlane *plane_ptr, const BYTE *src_ptr, int xx, int src_pitch
    );

  static MV_FORCEINLINE void
    norm_weights(int wref_arr[], int trad);

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


#endif
