/*****************************************************************************
        MAverage.h

*Tab=3***********************************************************************/



#if ! defined (MAverage_HEADER_INCLUDED)
#define	MAverage_HEADER_INCLUDED

#if defined (_MSC_VER)
  #pragma once
  #pragma warning (4 : 4250)
#endif

#define MAX_AREAMODE_STEPS 100 // expected enough ? todo: make variable memory allocation


/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include	"def.h"
#include "MVAnalysisData.h"
#include	"types.h"

#include "avisynth.h"
#include "VECTOR.h"

#include <vector>


class MAverage
:	public ::GenericVideoFilter
{

/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

public:
  bool has_at_least_v8;

  explicit			MAverage (std::vector <::PClip> clip_arr, int _mode, IScriptEnvironment *env);
  virtual			~MAverage () {}

  // GenericVideoFilter
  ::PVideoFrame __stdcall
            GetFrame (int n, ::IScriptEnvironment *env_ptr);



/*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

protected:

  MVAnalysisData mVectorsInfo;  // Clip dimensions, block layout etc.
  VECTOR vAMResults[MAX_AREAMODE_STEPS];
  int nbr_clips;
  std::vector <::PClip> m_clip_arr;

  int* pSrcPlanes[MAX_AREAMODE_STEPS];
  VECTOR* pSrcBlocks[MAX_AREAMODE_STEPS];

  int nBlkX;
  int nBlkSizeX;
  int nBlkY;
  int nBlkSizeY;
  int nPel;


/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:

  CHECK_COMPILE_TIME (SizeOfInt, (sizeof (int) == sizeof (int32_t)));

  int iMode;



/*\\\ FORBIDDEN MEMBER FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:

            MAverage ();
            MAverage (const MAverage &other);


            template<typename pixel_t>
            MV_FORCEINLINE void GetModeVECTORxy(VECTOR* toMedian, VECTOR* vOut, int iNumMVs);

            template<typename pixel_t>
            MV_FORCEINLINE void GetModeVECTORvad(VECTOR* toMedian, VECTOR* vOut, int iNumMVs);

            template<typename pixel_t>
            MV_FORCEINLINE void GetModeVECTORvld(VECTOR* toMedian, VECTOR* vOut, int iNumMVs);

            template<typename pixel_t>
            MV_FORCEINLINE void GetMeanVECTORxy(VECTOR* toMedian, VECTOR* vOut, int iNumMVs);

            template<typename pixel_t>
            MV_FORCEINLINE void GetMedianVECTORg(VECTOR* toMedian, VECTOR* vOut, int iNumMVs); // geometric median

            template<typename pixel_t>
            MV_FORCEINLINE void Get_IQM_VECTORxy(VECTOR* toMedian, VECTOR* vOut, int iNumMVs); // IQM for x and y separately

            template<typename pixel_t>
            MV_FORCEINLINE void GetModeVECTORxyda(VECTOR* toMedian, VECTOR* vOut, int iNumMVs);

            template<typename pixel_t>
            MV_FORCEINLINE void GetModeVECTORxydadm(VECTOR* toMedian, VECTOR* vOut, int iNumMVs);

            template<typename pixel_t>
            MV_FORCEINLINE void GetLowestDM(VECTOR* toProc, VECTOR* vOut, int iNumMVs);

};	// class MStoreVect

MV_FORCEINLINE float fDiffAngleVect(int x1, int y1, int x2, int y2);

#endif	// MAverage_HEADER_INCLUDED



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
