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


#ifndef __MV_MVPlanes__
#define __MV_MVPlanes__

#include	"MTFlowGraphSimple.h"
#include	"MTFlowGraphSched.h"
#include	"MTSlicer.h"
#include	"types.h"

#include	<cstdio>
#include <stdint.h>
#include "def.h"

#define SHIFTKERNELSIZE 8
#define SHIFTKERNELSIZE_I16 6

class MVPlane
{
public:

   MVPlane(int _nWidth, int _nHeight, int _nPel, int _nHPad, int _nVPad, int _pixelsize, int _bits_per_pixel, int cpuFlags, bool mt_flag);
   ~MVPlane();

   void set_interp (int rfilter, int sharp);
   void Update(uint8_t* pSrc, int _nPitch);
   void ChangePlane(const uint8_t *pNewPlane, int nNewPitch);
   void Pad();
   void refine_start ();
   void refine_wait ();

   template<typename pixel_t>
   void RefineExt(const uint8_t *pSrc2x_8, int nSrc2xPitch, bool isExtPadded); //2.0.08
   
   void reduce_start (MVPlane *pReducedPlane);
  void reduce_wait ();
   void WritePlane(FILE *pFile);

  template <int NPELL2>
  MV_FORCEINLINE const uint8_t* GetAbsolutePointerPel(int nX, int nY) const;

  MV_FORCEINLINE const uint8_t* GetAbsolutePointer(int nX, int nY) const;

  MV_FORCEINLINE const uint8_t *GetPointer(int nX, int nY) const
   {
      return GetAbsolutePointer(nX + nHPaddingPel, nY + nVPaddingPel);
   }

  MV_FORCEINLINE const uint8_t *GetAbsolutePelPointer(int nX, int nY) const
   {
    return pPlane[0] + (nX << pixelsize_shift) + nY * nPitch;
  }

  template <int NPELL2>
  MV_FORCEINLINE const uint8_t* GetPointerPel(int nX, int nY) const
  {
    return GetAbsolutePointerPel <NPELL2>(nX + nHPaddingPel, nY + nVPaddingPel);
  }

  // 2.7.46
  const uint8_t* GetPointerSubShift(int nX, int nY, int iBlockSizeX, int iBlockSizeY, int& nDstPitch)const;
 
  MV_FORCEINLINE int GetPitch() const { return nPitch; }
  MV_FORCEINLINE int GetWidth() const { return nWidth; }
  MV_FORCEINLINE int GetHeight() const { return nHeight; }
  MV_FORCEINLINE int GetExtendedWidth() const { return nExtendedWidth; }
  MV_FORCEINLINE int GetExtendedHeight() const { return nExtendedHeight; }
  MV_FORCEINLINE int GetHPadding() const { return nHPadding; }
  MV_FORCEINLINE int GetVPadding() const { return nVPadding; }
  MV_FORCEINLINE void ResetState() { isRefined = isFilled = isPadded = false; }

private:

  bool _isse;
  bool hasSSE41;
  bool hasAVX2;

  typedef	MTFlowGraphSched <MVPlane, MTFlowGraphSimple <16>, MVPlane, 16>	SchedulerRefine;
  typedef	MTSlicer <MVPlane>	SlicerReduce;

  typedef void (*InterpFncPtr) (
    unsigned char *pDst, const unsigned char *pSrc,
     int nDstPitch, int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel
  );

  typedef void (*ReducePtr) (
    unsigned char *pDst, const unsigned char *pSrc, int nDstPitch, int nSrcPitch,
    int nWidth, int nHeight, int y_beg, int y_end, int cpuFlags
  );

  void	refine_pel2 (SchedulerRefine::TaskData &td);
  void	refine_pel4 (SchedulerRefine::TaskData &td);
  void	reduce_slice (SlicerReduce::TaskData &td);

  uint8_t **pPlane;
  int nWidth;
  int nHeight;
  int nPitch;
  int nHPadding;
  int nVPadding;
  int nOffsetPadding;
  int nHPaddingPel;
  int nVPaddingPel;
  int nExtendedWidth;
  int nExtendedHeight;

  int nPel;
  int pixelsize;
  int pixelsize_shift; // prevent multiplicaion of *pixelsize, use shift
  int bits_per_pixel;

  int nSharp;		// Set only from MSuper, used in Refine()
  int nRfilter;	// Same as above, for ReduceTo()

   int cpuFlags;
  bool _mt_flag;

   bool isPadded;
   bool isRefined;
   bool isFilled;

   uint8_t* pShiftedBlockBuf;
   uint8_t* pShiftedBlockBuf_a;

   InterpFncPtr	_bilin_hor_ptr;
  InterpFncPtr	_bilin_ver_ptr;
  InterpFncPtr	_bilin_dia_ptr;
  InterpFncPtr	_bicubic_hor_ptr;
  InterpFncPtr	_bicubic_ver_ptr;
  InterpFncPtr	_wiener_hor_ptr;
  InterpFncPtr	_wiener_ver_ptr;
  void				(*_average_ptr) (unsigned char*, const unsigned char*, const unsigned char*, int, int, int);

  ReducePtr		_reduce_ptr;

  SchedulerRefine
            _sched_refine;
  MTFlowGraphSimple <16>
            _plan_refine;

  SlicerReduce	_slicer_reduce;
  MVPlane *		_redp_ptr;			// The plane where the reduction is rendered.

  // 2.7.46

  MV_FORCEINLINE float fSinc(float x)
  {
    x = fabsf(x);

    if (x > 0.000001f)
    {
      return sinf(x) / x;
    }
    else return 1.0f;
  }



  void CalcShiftKernel(float* fKernel, float fPelShift, int iKS);

  float fKernelSh_01[SHIFTKERNELSIZE];
  float fKernelSh_10[SHIFTKERNELSIZE];
  float fKernelSh_11[SHIFTKERNELSIZE];

  void CalcShiftKernels(short* fKernel, float fPelShift, int iKS);

  short sKernelSh_01[SHIFTKERNELSIZE];
  short sKernelSh_10[SHIFTKERNELSIZE];
  short sKernelSh_11[SHIFTKERNELSIZE];

  short* sKernelShWI6_01;
  short* sKernelShWI6_10;
  short* sKernelShWI6_11;

  typedef void (*SubShiftFncPtr) (
    unsigned char* pSrc, unsigned char* pDst, int iBlockSizeX, int iBlockSizeY, float* fKernelH, float* fKernelV, int nSrcPitch, int nDstPitch, int iKS
    );

  SubShiftFncPtr	_sub_shift_ptr;

};

template <int NPELL2>
MV_FORCEINLINE const uint8_t* MVPlane::GetAbsolutePointerPel(int nX, int nY) const
{
  enum { MASK = (1 << NPELL2) - 1 };

  int idx = (nX & MASK) | ((nY & MASK) << NPELL2);

  nX >>= NPELL2;
  nY >>= NPELL2;

  return pPlane[idx] + (nX << pixelsize_shift) + nY * nPitch;
}

template <>
MV_FORCEINLINE const uint8_t* MVPlane::GetAbsolutePointerPel <0>(int nX, int nY) const
{
  return pPlane[0] + (nX << pixelsize_shift) + nY * nPitch;
}

MV_FORCEINLINE const uint8_t* MVPlane::GetAbsolutePointer(int nX, int nY) const
{
  if (nPel == 1)
  {
    return GetAbsolutePointerPel <0>(nX, nY);
  }
  else if (nPel == 2)
  {
    return GetAbsolutePointerPel <1>(nX, nY);
  }
  else // nPel == 4
  {
    return GetAbsolutePointerPel <2>(nX, nY);
  }
}



#endif	// __MV_MVPlanes__
