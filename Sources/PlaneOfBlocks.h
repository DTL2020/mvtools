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

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

#ifndef __POBLOCKS__
#define __POBLOCKS__

#include "conc/ObjPool.h"
#include "CopyCode.h"
#include "MTSlicer.h"
#include	"MVInterface.h"	// Required for ALIGN_SOURCEBLOCK
#include "SADFunctions.h"
#include "SearchType.h"
#include "Variance.h"

#if (ALIGN_SOURCEBLOCK > 1)
#include "AllocAlign.h"
#endif	// ALIGN_SOURCEBLOCK

#include	<vector>
#include "avisynth.h"
#include <atomic>

#include "MVFrame.h"
#include "MVPlane.h"
#include "MVPlaneSet.h"


// right now 5 should be enough (TSchniede)
#define MAX_PREDICTOR (20)

#define MAX_MULTI_BLOCKS_8x8_AVX2 4
#define MAX_MULTI_BLOCKS_8x8_AVX512 16

struct VECTOR_XY
{
  int x;
  int y;
};

class DCTClass;
class MVClip;
class MVFrame;

// MVVector
template <class T>
class  MVVector
{
public:

  typedef T* iterator;

  MVVector();
  ~MVVector();
  MVVector(size_t size, IScriptEnvironment* env);
  size_t size() const;

  size_t size_bytes;

  T& operator[](size_t index);

private:
  size_t my_size;
  T* buffer;
};

// v2.5.13.1: This class is currently a bit messy,
// it's being reorganised and reworked for further improvement.

constexpr int MAX_SUPPORTED_EXH_SEARCHPARAM = 4;
// DTL test for new exhaustive functions dispatchers. 4: up to SearchParam==4

class PlaneOfBlocks
{

public:

  typedef	MTSlicer <PlaneOfBlocks>	Slicer;

  PlaneOfBlocks(int _nBlkX, int _nBlkY, int _nBlkSizeX, int _nBlkSizeY, int _nPel, int _nLevel, int _nFlags, int _nOverlapX, int _nOverlapY,
    int _xRatioUV, int _yRatioUV, int _pixelsize, int _bits_per_pixel,
    conc::ObjPool <DCTClass> *dct_pool_ptr,
    bool mt_flag, int _chromaSADscale, int _optSearchOption,
  IScriptEnvironment* env);

  ~PlaneOfBlocks();

  /* search the vectors for the whole plane */
  void SearchMVs(MVFrame *_pSrcFrame, MVFrame *_pRefFrame, SearchType st,
    int stp, int lambda, sad_t lsad, int pnew, int plevel,
    int flags, sad_t *out, const VECTOR *globalMVec, short * outfilebuf, int fieldShiftCur,
    int * meanLumaChange, int divideExtra,
    int _pzero, int _pglobal, sad_t _badSAD, int _badrange, bool meander, int *vecPrev, bool _tryMany,
    int optPredictorType);


  /* plane initialisation */

    /* compute the predictors from the upper plane */
  template<typename safe_sad_t, typename smallOverlapSafeSad_t>
//  void InterpolatePrediction(const PlaneOfBlocks &pob);
  void InterpolatePrediction(PlaneOfBlocks& pob); // temp for MVVector[]


  void WriteHeaderToArray(int *array);
  int WriteDefaultToArray(int *array, int divideExtra);
  int GetArraySize(int divideExtra);
  // not used void FitReferenceIntoArray(MVFrame *_pRefFrame, int *array);
  void EstimateGlobalMVDoubled(VECTOR *globalMVec, Slicer &slicer); // Fizick
  MV_FORCEINLINE int GetnBlkX() { return nBlkX; }
  MV_FORCEINLINE int GetnBlkY() { return nBlkY; }
  MV_FORCEINLINE int GetnBlkSizeX() { return nBlkSizeX; }
  MV_FORCEINLINE int GetnBlkSizeY() { return nBlkSizeY; }

  void RecalculateMVs(MVClip & mvClip, MVFrame *_pSrcFrame, MVFrame *_pRefFrame, SearchType st,
    int stp, int _lambda, sad_t _lSAD, int _pennew,
    int flags, int *out, short * outfilebuf, int fieldShift, sad_t thSAD,
    int _divideExtra, int smooth, bool meander,
    int optPredictorType);


private:

  /* fields set at initialization */

  const int      nBlkX;            /* width in number of blocks */
  const int      nBlkY;            /* height in number of blocks */
  const int      nBlkSizeX;        /* size of a block */
  const int      nBlkSizeY;        /* size of a block */
  const int      nSqrtBlkSize2D;   /* precalc for DCT 2.7.38- */
  const int      nBlkCount;        /* number of blocks in the plane */
  const int      nPel;             /* pel refinement accuracy */
  const int      nLogPel;          /* logarithm of the pel refinement accuracy */
  const int      nScale;           /* scaling factor of the plane */
  const int      nLogScale;        /* logarithm of the scaling factor */
  int            nFlags;           /* additionnal flags */
  const int      nOverlapX;        // overlap size
  const int      nOverlapY;        // overlap size
  const int      xRatioUV;        // PF
  const int      nLogxRatioUV;     // log of xRatioUV (0 for 1 and 1 for 2)
  const int      yRatioUV;
  const int      nLogyRatioUV;     // log of yRatioUV (0 for 1 and 1 for 2)
  const int      pixelsize; // PF
  const int      pixelsize_shift; // log of pixelsize (0,1,2) for shift instead of mul or div
  const int      bits_per_pixel;
  const bool     _mt_flag;         // Allows multithreading
  const int      chromaSADscale;   // PF experimental 2.7.18.22 allow e.g. YV24 chroma to have the same magnitude as for YV12
  int            effective_chromaSADscale;   // PF experimental 2.7.18.22 allow e.g. YV24 chroma to have the same magnitude as for YV12
  const int      optSearchOption; // DTL test != 0: allow

  SADFunction *  SAD;              /* function which computes the sad */
  LUMAFunction * LUMA;             /* function which computes the mean luma */
  //VARFunction *  VAR;              /* function which computes the variance */
  COPYFunction * BLITLUMA;
  COPYFunction * BLITCHROMA;
  SADFunction *  SADCHROMA;
  SADFunction *  SATD;              /* SATD function, (similar to SAD), used as replacement to dct */

  // DTL test
  class WorkingArea; // forward
  using ExhaustiveSearchFunction_t = void(PlaneOfBlocks::*)(WorkingArea& workarea, int mvx, int mvy);

  ExhaustiveSearchFunction_t get_ExhaustiveSearchFunction(int BlockX, int BlockY, int SearchParam, int bits_per_pixel, arch_t arch);
  ExhaustiveSearchFunction_t ExhaustiveSearchFunctions[MAX_SUPPORTED_EXH_SEARCHPARAM + 1]; // the function pointer

  //std::vector <VECTOR>              /* motion vectors of the blocks */
  //  vectors;           /* before the search, contains the hierachal predictor */
  MVVector <VECTOR> vectors;
                       /* after the search, contains the best motion vector */

  bool           smallestPlane;     /* say whether vectors can use predictors from a smaller plane */
  bool           isse;              /* can we use isse asm code */
  bool           chroma;            /* do we do chroma me */

  bool sse2; // make members now to use in SADs
  bool sse41;
  bool avx;
  bool avx2;
  bool avx512;


  int dctpitch;
  conc::ObjPool <DCTClass> *		// Set to 0 if not used
    _dct_pool_ptr;

  std::array <std::vector <int>, 2>
    freqArray; // temporary array for global motion estimaton [x|y][value]

  sad_t verybigSAD;

  /* working fields */

    // Current frame
  MVFrame *pSrcFrame;
  MVFrame *pRefFrame;
  int nSrcPitch[3];
  int nRefPitch[3];
#if (ALIGN_SOURCEBLOCK > 1)
  int nSrcPitch_plane[3];     // stores the pitch of the whole plane for easy access (nSrcPitch in non-aligned mode)
#endif	// ALIGN_SOURCEBLOCK

  VECTOR zeroMVfieldShifted;  // zero motion vector for fieldbased video at finest level pel2

  int dctmode;
  sad_t dctweight16;

  // Current plane
  SearchType searchType;      /* search type used */
  int nSearchParam;           /* additional parameter for this search */
  sad_t LSAD;                   // SAD limit for lambda using - Fizick.
  int penaltyNew;             // cost penalty factor for new candidates
  int penaltyZero;            // cost penalty factor for zero vector
  int pglobal;                // cost penalty factor for global predictor
//	int nLambdaLen;             // penalty factor (lambda) for vector length
  sad_t badSAD;                 // SAD threshold for more wide search
  int badrange;               // wide search radius
  std::atomic <int> badcount;      // number of bad blocks refined
  bool temporal;              // use temporal predictor
  bool tryMany;               // try refine around many predictors

  // PF todo this should be float or double for float format??
  // it is not AtomicInt anymore
  std::atomic <bigsad_t> planeSAD;      // summary SAD of plane
  std::atomic <bigsad_t> sumLumaChange; // luma change sum
  VECTOR _glob_mv_pred_def;
  int _lambda_level;

  // Parameters from SearchMVs() and RecalculateMVs()
  int *_out;
  short *_outfilebuf;
  int *_vecPrev;
  bool _meander_flag;
  int _pnew;
  sad_t _lsad;
  MVClip *	_mv_clip_ptr;
  int _smooth;
  sad_t _thSAD;
  //  const VECTOR zeroMV = {0,0,(sad_t)-1};
  int _predictorType; // 2.7.46

  uint64_t checked_mv_vectors[9]; // 2.7.46
  int iNumCheckedVectors; // 2.7.46

  // Working area
  class WorkingArea
  {
  public:
#if (ALIGN_SOURCEBLOCK > 1)
    typedef	std::vector <uint8_t, aligned_allocator <uint8_t, ALIGN_SOURCEBLOCK> >	TmpDataArray;
#else	// ALIGN_SOURCEBLOCK
    typedef	std::vector <uint8_t>	TmpDataArray;
#endif	// ALIGN_SOURCEBLOCK

    int x[3];                   /* absolute x coordinate of the origin of the block in the reference frame */
    int y[3];                   /* absolute y coordinate of the origin of the block in the reference frame */
    int blkx;                   /* x coordinate in blocks */
    int blky;                   /* y coordinate in blocks */
    int blkIdx;                 /* index of the block */
    int blkScanDir;             // direction of scan (1 is left to right, -1 is right to left)

    DCTClass * DCT;

    VECTOR globalMVPredictor;   // predictor of global motion vector

    bigsad_t planeSAD;          // partial summary SAD of plane
    bigsad_t sumLumaChange;     // partial luma change sum
    int blky_beg;               // First line of blocks to process from this thread
    int blky_end;               // Last line of blocks + 1 to process from this thread

    // Current block
    const uint8_t* pSrc[3];     // the alignment of this array is important for speed for some reason (cacheline?)

    VECTOR bestMV;              /* best vector found so far during the search */
    VECTOR bestMV_multi[MAX_MULTI_BLOCKS_8x8_AVX512]; // 2.7.46
    sad_t nMinCost;               /* minimum cost ( sad + mv cost ) found so far */
    sad_t nMinCost_multi[MAX_MULTI_BLOCKS_8x8_AVX512]; // 2.7.46
    VECTOR predictor;           /* best predictor for the current vector */
    VECTOR predictors[MAX_PREDICTOR];   /* set of predictors for the current block */

    int nDxMin;                 /* minimum x coordinate for the vector */ //need to be in order DxMin, DyMin for ClipMV faster load
    int nDyMin;                 /* minimum y coordinate for the vector */
    int nDxMax;                 /* maximum x corrdinate for the vector */
    int nDyMax;                 /* maximum y coordinate for the vector */

    int nLambda;                /* vector cost factor */
    int iter;                   // MOTION_DEBUG only?
    int srcLuma;

    int pixelsize;
    int bits_per_pixel;

    bool bIntraframe;

    // Data set once
    TmpDataArray dctSrc;
    TmpDataArray dctRef;
#if (ALIGN_SOURCEBLOCK > 1)
    TmpDataArray pSrc_temp_base;// stores base memory pointer to non _base pointer
    uint8_t* pSrc_temp[3];      //for easy WRITE access to temp block
#endif	// ALIGN_SOURCEBLOCK

    WorkingArea(int nBlkSizeX, int nBlkSizeY, int dctpitch, int nLogxRatioUV, int nLogyRatioUV, int pixelsize, int bits_per_pixel);
    virtual			~WorkingArea();

    /* check if a vector is inside search boundaries */
    // Moved here from cpp in order to able to inlined from other modules (gcc error)
    MV_FORCEINLINE bool IsVectorOK(int vx, int vy) const
    {
      return (
        (vx >= nDxMin)
        && (vy >= nDyMin)
        && (vx < nDxMax)
        && (vy < nDyMax)
        );
    }

    template<typename pixel_t>
    sad_t MotionDistorsion(int vx, int vy) const; // this one is better not forceinlined

    // multi blocks processing sources for reusing
    __m256i ymm0_src_r1, ymm1_src_r2, ymm2_src_r3, ymm3_src_r4, ymm4_src_r5, ymm5_src_r6, ymm6_src_r7, ymm7_src_r8;

    __m512i zmm0_Src_r1_b0007, zmm2_Src_r2_b0007, zmm4_Src_r3_b0007, zmm6_Src_r4_b0007, zmm8_Src_r5_b0007, zmm10_Src_r6_b0007, zmm12_Src_r7_b0007, zmm14_Src_r8_b0007;
    __m512i zmm1_Src_r1_b0815, zmm3_Src_r2_b0815, zmm5_Src_r3_b0815, zmm7_Src_r4_b0815, zmm9_Src_r5_b0815, zmm11_Src_r6_b0815, zmm13_Src_r7_b0815, zmm15_Src_r8_b0815;

  };

  class WorkingAreaFactory
    : public conc::ObjFactoryInterface <WorkingArea>
  {
  public:
    WorkingAreaFactory(int nBlkSizeX, int nBlkSizeY, int dctpitch, int nLogxRatioUV, int nLogyRatioUV, int pixelsize, int bits_per_pixel);
  protected:
    // conc::ObjFactoryInterface
    virtual WorkingArea *
      do_create();
  private:
    int _blk_size_x;
    int _blk_size_y;
    int _dctpitch;
    int _x_ratio_uv_log; // PF
    int _y_ratio_uv_log;
    int _pixelsize; // PF
    int _bits_per_pixel;
  };

  typedef	conc::ObjPool <WorkingArea>	WorkingAreaPool;

  WorkingAreaFactory
    _workarea_fact;
  WorkingAreaPool
    _workarea_pool;

  VECTOR *_gvect_estim_ptr;	// Points on the global motion vector estimation result. 0 when not used.
  std::atomic<int> _gvect_result_count;

  /* mv search related functions */

    /* fill the predictors array */
  template<typename pixel_t>
  void FetchPredictors(WorkingArea &workarea);

  template<typename pixel_t>
  MV_FORCEINLINE void FetchPredictors_sse41(WorkingArea &workarea);

  template<typename pixel_t>
  MV_FORCEINLINE void FetchPredictors_sse41_intraframe(WorkingArea& workarea);

  template<typename pixel_t>
  MV_FORCEINLINE void FetchPredictors_avx2_intraframe(WorkingArea& workarea);


  /* performs a diamond search */
  template<typename pixel_t>
  void DiamondSearch(WorkingArea &workarea, int step);

  /* performs a square search */
  //	void SquareSearch(WorkingArea &workarea);

  /* performs an exhaustive search */
  //	void ExhaustiveSearch(WorkingArea &workarea, int radius); // diameter = 2*radius - 1

  /* performs an n-step search */
  template<typename pixel_t>
  void NStepSearch(WorkingArea &workarea, int stp);

  /* performs a one time search */
  template<typename pixel_t>
  void OneTimeSearch(WorkingArea &workarea, int length);

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch(WorkingArea &workarea); // full predictors, slowest, max quality

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO2(WorkingArea& workarea); // full predictors, optSearchOption = 2 set of params

  template<typename pixel_t>
  void PseudoEPZSearch_optSO2_8x8_avx2(WorkingArea& workarea); // full predictors, optSearchOption = 2 set of params, avx2 version of SADs and search
 
  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_glob_med_pred(WorkingArea& workarea); // planes >=2 recommended (optPredictorType=1)

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_no_pred(WorkingArea& workarea); // only interpolated predictor (optPredictorType=2)

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_no_refine(WorkingArea& workarea); // no refining mode - faster (optPredictorType=3)

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO2_glob_med_pred(WorkingArea& workarea); // global and median predictors, optSearchOption = 2 set of params

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO2_no_pred(WorkingArea& workarea); // no predictors, optSearchOption = 2 set of params

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO2_no_refine(WorkingArea& workarea); // no predictors, optSearchOption = 2 optPredictorType = 3 set of params

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO3_no_pred(WorkingArea& workarea, int* pBlkData); // no predictors, multi-block search AVX2

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO3_glob_pred_avx2(WorkingArea& workarea, int* pBlkData); // zero and global predictor, multi-block search AVX2

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO4_no_pred(WorkingArea& workarea, int* pBlkData); // no predictors, multi-block search AVX512

  /* performs an epz search */
  template<typename pixel_t>
  void PseudoEPZSearch_optSO4_glob_pred_avx512(WorkingArea& workarea, int* pBlkData); // zero and global predictor, multi-block search AVX512


  //	void PhaseShiftSearch(int vx, int vy);

  /* performs an exhaustive search */
  template<typename pixel_t>
  void ExpandingSearch(WorkingArea &workarea, int radius, int step, int mvx, int mvy); // diameter = 2*radius + 1

  // DTL test function, 8x8 and 16x16 block, 8 bit only

  // C-versions
  void ExhaustiveSearch_uint8_sp1_c(WorkingArea& workarea, int mvx, int mvy); // for any nPel and blocksize (SAD() is selected for blocksize)
  void ExhaustiveSearch_uint8_sp2_c(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch_uint8_sp3_c(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch_uint8_sp4_c(WorkingArea& workarea, int mvx, int mvy);

  // 8x8 exa search radius 4
  void ExhaustiveSearch8x8_uint8_np1_sp4_avx2(WorkingArea& workarea, int mvx, int mvy);

  // 8x8 exa search radius 1
  void ExhaustiveSearch8x8_uint8_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch8x8_uint8_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch8x8_uint8_SO2_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch8x8_uint8_4Blks_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy, int* pBlkData);
  void ExhaustiveSearch8x8_uint8_4Blks_Z_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy, int* pBlkData); // + zero pos
  void ExhaustiveSearch8x8_uint8_16Blks_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy, int* pBlkData);
  void ExhaustiveSearch8x8_uint8_16Blks_Z_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy, int* pBlkData); // +zero pos
   
  // 8x8 exa search radius 2
  void ExhaustiveSearch8x8_uint8_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy);
  void ExhaustiveSearch8x8_uint8_SO2_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy);
  
  // 8x8 exa search radius 3
  void ExhaustiveSearch8x8_uint8_np1_sp3_avx2(WorkingArea& workarea, int mvx, int mvy);

  // 16x16 exa search radius 1
  void ExhaustiveSearch16x16_uint8_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy); // minsadbw only version
  void ExhaustiveSearch16x16_uint8_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy); 
  void ExhaustiveSearch16x16_uint8_SO2_np1_sp1_avx2(WorkingArea& workarea, int mvx, int mvy); // minsadbw only version
  void ExhaustiveSearch16x16_uint8_SO2_np1_sp1_avx512(WorkingArea& workarea, int mvx, int mvy); // minsadbw only version

  // 16x16 exa search radius 2
  void ExhaustiveSearch16x16_uint8_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy); // minsadbw only version
  void ExhaustiveSearch16x16_uint8_SO2_np1_sp2_avx2(WorkingArea& workarea, int mvx, int mvy); // minsadbw only version


  // END OF DTL test function

  template<typename pixel_t>
  void Hex2Search(WorkingArea &workarea, int i_me_range);
  template<typename pixel_t>
  void CrossSearch(WorkingArea &workarea, int start, int x_max, int y_max, int mvx, int mvy);
  template<typename pixel_t>
  void UMHSearch(WorkingArea &workarea, int i_me_range, int omx, int omy);

  /* inline functions */

  /* fetch the block in the reference frame, which is pointed by the vector (vx, vy) */
  // moved here from cpp in order to able to inline from other (e.g. _avx2) cpps (gcc error)
  MV_FORCEINLINE const uint8_t* GetRefBlock(WorkingArea& workarea, int nVx, int nVy) {
    return
      (nPel == 2) ? pRefFrame->GetPlane(YPLANE)->GetAbsolutePointerPel <1>((workarea.x[0] << 1) + nVx, (workarea.y[0] << 1) + nVy) :
      (nPel == 1) ? pRefFrame->GetPlane(YPLANE)->GetAbsolutePointerPel <0>((workarea.x[0]) + nVx, (workarea.y[0]) + nVy) :
      pRefFrame->GetPlane(YPLANE)->GetAbsolutePointerPel <2>((workarea.x[0] << 2) + nVx, (workarea.y[0] << 2) + nVy);
  }

  MV_FORCEINLINE const uint8_t* GetRefBlockU(WorkingArea& workarea, int nVx, int nVy)
  {
    return
      (nPel == 2) ? pRefFrame->GetPlane(UPLANE)->GetAbsolutePointerPel <1>((workarea.x[1] << 1) + (nVx >> nLogxRatioUV), (workarea.y[1] << 1) + (nVy >> nLogyRatioUV)) :
      (nPel == 1) ? pRefFrame->GetPlane(UPLANE)->GetAbsolutePointerPel <0>((workarea.x[1]) + (nVx >> nLogxRatioUV), (workarea.y[1]) + (nVy >> nLogyRatioUV)) :
      pRefFrame->GetPlane(UPLANE)->GetAbsolutePointerPel <2>((workarea.x[1] << 2) + (nVx >> nLogxRatioUV), (workarea.y[1] << 2) + (nVy >> nLogyRatioUV));
  }

  MV_FORCEINLINE const uint8_t* GetRefBlockV(WorkingArea& workarea, int nVx, int nVy)
  {
    return
      (nPel == 2) ? pRefFrame->GetPlane(VPLANE)->GetAbsolutePointerPel <1>((workarea.x[2] << 1) + (nVx >> nLogxRatioUV), (workarea.y[2] << 1) + (nVy >> nLogyRatioUV)) :
      (nPel == 1) ? pRefFrame->GetPlane(VPLANE)->GetAbsolutePointerPel <0>((workarea.x[2]) + (nVx >> nLogxRatioUV), (workarea.y[2]) + (nVy >> nLogyRatioUV)) :
      pRefFrame->GetPlane(VPLANE)->GetAbsolutePointerPel <2>((workarea.x[2] << 2) + (nVx >> nLogxRatioUV), (workarea.y[2] << 2) + (nVy >> nLogyRatioUV));
  }

  MV_FORCEINLINE const uint8_t* GetSrcBlock(int nX, int nY)
  {
    return pSrcFrame->GetPlane(YPLANE)->GetAbsolutePelPointer(nX, nY);
  }

  //	MV_FORCEINLINE int LengthPenalty(int vx, int vy);
  template<typename pixel_t>
  sad_t LumaSADx(WorkingArea &workarea, const unsigned char *pRef0);
  template<typename pixel_t>
  MV_FORCEINLINE sad_t LumaSAD(WorkingArea &workarea, const unsigned char *pRef0);
  template<typename pixel_t>
  MV_FORCEINLINE void CheckMV0(WorkingArea &workarea, int vx, int vy);
  template<typename pixel_t>
  MV_FORCEINLINE void CheckMV0_SO2(WorkingArea& workarea, int vx, int vy, sad_t cost);
  template<typename pixel_t>
  MV_FORCEINLINE void CheckMV(WorkingArea &workarea, int vx, int vy);
  template<typename pixel_t>
  MV_FORCEINLINE void CheckMV2(WorkingArea &workarea, int vx, int vy, int *dir, int val);
  template<typename pixel_t>
  MV_FORCEINLINE void CheckMVdir(WorkingArea &workarea, int vx, int vy, int *dir, int val);
  MV_FORCEINLINE int ClipMVx(WorkingArea &workarea, int vx);
  MV_FORCEINLINE int ClipMVy(WorkingArea &workarea, int vy);
  MV_FORCEINLINE VECTOR ClipMV(WorkingArea &workarea, VECTOR v);
  MV_FORCEINLINE VECTOR	ClipMV_SO2(WorkingArea& workarea, VECTOR v);
  MV_FORCEINLINE static int Median(int a, int b, int c);
  // MV_FORCEINLINE static unsigned int SquareDifferenceNorm(const VECTOR& v1, const VECTOR& v2); // not used
  MV_FORCEINLINE static unsigned int SquareDifferenceNorm(const VECTOR& v1, const int v2x, const int v2y);
  MV_FORCEINLINE bool IsInFrame(int i);
  MV_FORCEINLINE bool IsVectorChecked(uint64_t xy); // 2.7.46
  MV_FORCEINLINE bool IsVectorsCoherent(VECTOR_XY* vectors_coh_check, int cnt);

  template<typename pixel_t>
  void Refine(WorkingArea &workarea);

  template<typename pixel_t>
  void	search_mv_slice(Slicer::TaskData &td);

  template<typename pixel_t>
  void	search_mv_slice_SO2(Slicer::TaskData& td); // with optSearchOption = 2 set of params

  template<typename pixel_t>
  void	search_mv_slice_SO3(Slicer::TaskData& td); // with optSearchOption = 3 set of params, multi-blocks search AVX2

  template<typename pixel_t>
  void	search_mv_slice_SO4(Slicer::TaskData& td); // with optSearchOption = 4 set of params, multi-blocks search AVX512


  template<typename pixel_t>
  void	recalculate_mv_slice(Slicer::TaskData &td);

  void	estimate_global_mv_doubled_slice(Slicer::TaskData &td);

  void(PlaneOfBlocks::* ExhaustiveSearch_SO2)(WorkingArea& workarea, int mvx, int mvy); // selector for sp1 and sp2

//  void(PlaneOfBlocks::* Sel_Pseudo_EPZ_search_SO2)(WorkingArea& workarea); // selector for optPredictors 0,1

};

#endif
