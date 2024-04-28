/*****************************************************************************
        MTransform.h

*Tab=3***********************************************************************/



#if ! defined (MTransform_HEADER_INCLUDED)
#define	MTransform_HEADER_INCLUDED

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


class MTransform
:	public ::GenericVideoFilter
{

/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

public:
  bool has_at_least_v8;

  explicit			MTransform (PClip clip, int _mode, IScriptEnvironment *env);
  virtual			~MTransform () {}

  // GenericVideoFilter
  ::PVideoFrame __stdcall
            GetFrame (int n, ::IScriptEnvironment *env_ptr);



/*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

protected:

  MVAnalysisData mVectorsInfo;  // Clip dimensions, block layout etc.
  PClip m_clip;

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

            MTransform ();
            MTransform (const MTransform &other);

            void FlipHorizontal(int* pSrcPlanes, int* pDstPlanes, int* pEnd, ::IScriptEnvironment* env_ptr);


};	// class MTransform

#endif	// MTransform_HEADER_INCLUDED



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
