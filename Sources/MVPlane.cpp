// Author: Manao
// Copyright(c)2006 A.G.Balakhnin aka Fizick - bicubic, wiener
// See legal notice in Copying.txt for more information
//
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

/******************************************************************************
*                                                                             *
*  MVPlane : manages a single plane, allowing padding and refining            *
*                                                                             *
******************************************************************************/

/*
About the parallelisation of the refine operation:
We have a lot of dependencies between the planes that kills the parallelism.
A better optimisation would be slicing each plane. However this requires a
significant modification of the assembly code to handle correctly the plane
boundaries.
*/


#include "CopyCode.h"
#include "Interpolation.h"
#include "MVPlane.h"
#include "Padding.h"
#include "MVInterface.h"
#include <stdint.h>
#include <commonfunctions.h>


MVPlane::MVPlane(int _nWidth, int _nHeight, int _nPel, int _nHPad, int _nVPad, int _pixelsize, int _bits_per_pixel, int _cpuFlags, bool mt_flag)
  : pPlane(new uint8_t*[_nPel * _nPel * _pixelsize])
  , nWidth(_nWidth)
  , nHeight(_nHeight)
  , nPitch(0)
  , nHPadding(_nHPad)
  , nVPadding(_nVPad)
  , nOffsetPadding(0)
  , nHPaddingPel(_nHPad * _nPel)
  , nVPaddingPel(_nVPad * _nPel)
  , nExtendedWidth(_nWidth + 2 * _nHPad)
  , nExtendedHeight(_nHeight + 2 * _nVPad)
  , nPel(_nPel)
  , pixelsize(_pixelsize)
  , pixelsize_shift((_pixelsize == 1) ? 0 : (_pixelsize == 2 ? 1 : 2)) // pixelsize 1/2/4: shift 0/1/2
  , bits_per_pixel(_bits_per_pixel)
  , nSharp(2)
  , cpuFlags(_cpuFlags)
  , _mt_flag(mt_flag)
  , isPadded(false)
  , isRefined(false)
  , isFilled(false)
  , _sched_refine(mt_flag)
  , _plan_refine()
  , _slicer_reduce(mt_flag)
  , _redp_ptr(0)
{
  _isse = !!(cpuFlags & CPUF_SSE2);
  hasSSE41 = !!(cpuFlags & CPUF_SSE4_1);
  hasAVX2 = !!(cpuFlags & CPUF_AVX2);

  if (pixelsize == 1) {
    _bilin_hor_ptr = _isse ? HorizontalBilin_sse2<uint8_t> : HorizontalBilin<uint8_t>;
    _bilin_ver_ptr = _isse ? VerticalBilin_sse2<uint8_t> : VerticalBilin<uint8_t>;
    _bilin_dia_ptr = _isse ? DiagonalBilin_sse2<uint8_t, false> : DiagonalBilin<uint8_t>;
    _bicubic_hor_ptr = _isse ? HorizontalBicubic_sse2<uint8_t, false> : HorizontalBicubic<uint8_t>;
    _bicubic_ver_ptr = _isse ? (hasSSE41 ? VerticalBicubic_sse2<uint8_t, true> : VerticalBicubic_sse2<uint8_t, false>) : VerticalBicubic<uint8_t>;
    _wiener_hor_ptr = _isse ? HorizontalWiener_sse2<uint8_t, false> : HorizontalWiener<uint8_t>;
    _wiener_ver_ptr = _isse ? VerticalWiener_sse2<uint8_t, false> : VerticalWiener<uint8_t>;
    _average_ptr = _isse ? Average2_sse2<uint8_t> : Average2<uint8_t>;
    _reduce_ptr = &RB2BilinearFiltered<uint8_t>;

//    if (blocksizeH == 8 && blocksizeV == 8 ) ?? need to pass blksize H,V to constructor ?
      _sub_shift_ptr = SubShiftBlock_C<uint8_t>;
  }
  else if (pixelsize == 2) {
    _bilin_hor_ptr = _isse ? HorizontalBilin_sse2<uint16_t> : HorizontalBilin<uint16_t>;
    _bilin_ver_ptr = _isse ? VerticalBilin_sse2<uint16_t> : VerticalBilin<uint16_t>;
    _bilin_dia_ptr = _isse ? (hasSSE41 ? DiagonalBilin_sse2<uint16_t, true> : DiagonalBilin_sse2<uint16_t, false>) : DiagonalBilin<uint16_t>;
    _bicubic_hor_ptr = _isse ? (hasSSE41 ? HorizontalBicubic_sse2<uint16_t, true> : HorizontalBicubic_sse2<uint16_t, false>) : HorizontalBicubic<uint16_t>;
    _bicubic_ver_ptr = _isse ? (hasSSE41 ? VerticalBicubic_sse2<uint16_t, true> : VerticalBicubic_sse2<uint16_t, false>) :VerticalBicubic<uint16_t>;
    _wiener_hor_ptr = _isse ? (hasSSE41 ? HorizontalWiener_sse2<uint16_t, true> : HorizontalWiener_sse2<uint16_t, false>) : HorizontalWiener<uint16_t>;
    _wiener_ver_ptr = _isse ? (hasSSE41 ? VerticalWiener_sse2<uint16_t, 1> : VerticalWiener_sse2<uint16_t, false>) : VerticalWiener<uint16_t>;
    _average_ptr = _isse ? Average2_sse2<uint16_t> : Average2<uint16_t>;
    _reduce_ptr = &RB2BilinearFiltered<uint16_t>;

    _sub_shift_ptr = SubShiftBlock_C<uint16_t>;
  }
  else {
    _bilin_hor_ptr = HorizontalBilin<float>;
    _bilin_ver_ptr = VerticalBilin<float>;
    _bilin_dia_ptr = DiagonalBilin<float>;
    _bicubic_hor_ptr = HorizontalBicubic<float>;
    _bicubic_ver_ptr = VerticalBicubic<float>;
    _wiener_hor_ptr = HorizontalWiener<float>;
    _wiener_ver_ptr = VerticalWiener<float>;
    _average_ptr = Average2<float>;
    _reduce_ptr = &RB2BilinearFiltered<float>;

    _sub_shift_ptr = SubShiftBlock_C<float>;
  }
  // Nothing

  // 2.7.46
  // prepare subshift kernels
  CalcShiftKernel(fKernelSh_01, 0.25f, SHIFTKERNELSIZE);
  CalcShiftKernel(fKernelSh_10, 0.5f, SHIFTKERNELSIZE);
  CalcShiftKernel(fKernelSh_11, 0.75f, SHIFTKERNELSIZE);

  // 2.7.46
// prepare subshift kernels
  CalcShiftKernels(sKernelSh_01, 0.25f, SHIFTKERNELSIZE_I16);
  CalcShiftKernels(sKernelSh_10, 0.5f, SHIFTKERNELSIZE_I16);
  CalcShiftKernels(sKernelSh_11, 0.75f, SHIFTKERNELSIZE_I16);

//  _sub_shift_ptr = SubShiftBlock_C<uint8_t>;
#ifdef _WIN32
 // to prevent cache set overloading when accessing fpob MVs arrays - add random L2L3_CACHE_LINE_SIZE-bytes sized offset to different allocations
  size_t random = rand();
  random *= RAND_OFFSET_MAX;
  random /= RAND_MAX;
  random *= L2L3_CACHE_LINE_SIZE;

  SIZE_T stSizeToAlloc = (pixelsize * (64*64))+ RAND_OFFSET_MAX * L2L3_CACHE_LINE_SIZE; // 64x64 - max block size ??

  pShiftedBlockBuf_a = (uint8_t*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
  pShiftedBlockBuf = (uint8_t*)(pShiftedBlockBuf_a + random);
#else
  pShiftedBlockBuf = new uint8_t[pixelsize * (64 * 64)]; // allocate in heap ?
#endif
}



MVPlane::~MVPlane()
{
  delete[] pPlane;
  pPlane = 0;

#ifdef _WIN32
  VirtualFree(pShiftedBlockBuf_a, 0, MEM_RELEASE);
#else
  delete[] pShiftedBlockBuf;
#endif
}



void MVPlane::set_interp(int rfilter, int sharp)
{
  nSharp = sharp; // for pel>1
  MV_UNUSED(nRfilter); // not used

  switch (rfilter)
  {
  case 0: _reduce_ptr = (pixelsize == 1) ? &RB2F<uint8_t> : (pixelsize == 2 ? &RB2F<uint16_t> : &RB2F<float>); break;
  case 1: _reduce_ptr = (pixelsize == 1) ? &RB2Filtered<uint8_t> : (pixelsize == 2 ? &RB2Filtered<uint16_t> : &RB2Filtered<float>); break;
  case 2: _reduce_ptr = (pixelsize == 1) ? &RB2BilinearFiltered<uint8_t> : (pixelsize == 2 ? &RB2BilinearFiltered<uint16_t> : &RB2BilinearFiltered<float>); break;
  case 3: _reduce_ptr = (pixelsize == 1) ? &RB2Quadratic<uint8_t> : (pixelsize == 2 ? &RB2Quadratic<uint16_t> : &RB2Quadratic<float>); break;
  case 4: _reduce_ptr = (pixelsize == 1) ? &RB2Cubic<uint8_t> : (pixelsize == 2 ? &RB2Cubic<uint16_t> : &RB2Cubic<float>); break;
  default:
    assert(false);
    break;
  };

  _plan_refine.clear();
  if (nSharp == 0)
  {
    if (nPel == 2)
    {
      _plan_refine.add_dep(0, 1);
      _plan_refine.add_dep(0, 2);
      _plan_refine.add_dep(0, 3);
    }
    else if (nPel == 4)
    {
      _plan_refine.add_dep(0, 2);
      _plan_refine.add_dep(0, 8);
      _plan_refine.add_dep(0, 10);
    }
  }
  else	// nSharp == 1 or 2
  {
    if (nPel == 2)
    {
      _plan_refine.add_dep(0, 1);
      _plan_refine.add_dep(0, 2);
      _plan_refine.add_dep(2, 3);
    }
    else if (nPel == 4)
    {
      _plan_refine.add_dep(0, 2);
      _plan_refine.add_dep(0, 8);
      _plan_refine.add_dep(8, 10);
    }
  }
  if (nPel == 4)
  {
    _plan_refine.add_dep(0, 1);
    _plan_refine.add_dep(2, 1);
    _plan_refine.add_dep(0, 3);
    _plan_refine.add_dep(2, 3);
    _plan_refine.add_dep(0, 4);
    _plan_refine.add_dep(8, 4);
    _plan_refine.add_dep(0, 12);
    _plan_refine.add_dep(8, 12);
    _plan_refine.add_dep(8, 9);
    _plan_refine.add_dep(10, 9);
    _plan_refine.add_dep(8, 11);
    _plan_refine.add_dep(10, 11);
    _plan_refine.add_dep(2, 6);
    _plan_refine.add_dep(10, 6);
    _plan_refine.add_dep(2, 14);
    _plan_refine.add_dep(10, 14);
    _plan_refine.add_dep(4, 5);
    _plan_refine.add_dep(6, 5);
    _plan_refine.add_dep(4, 7);
    _plan_refine.add_dep(6, 7);
    _plan_refine.add_dep(12, 13);
    _plan_refine.add_dep(14, 13);
    _plan_refine.add_dep(12, 15);
    _plan_refine.add_dep(14, 15);
  }
}



void MVPlane::Update(uint8_t* pSrc, int _nPitch) //v2.0
{
  // npitch is pixelsize aware
  nPitch = _nPitch;

  nOffsetPadding = nPitch * nVPadding + (nHPadding << pixelsize_shift);

  for (int i = 0; i < nPel * nPel; i++)
  {
    pPlane[i] = pSrc + i * nPitch * nExtendedHeight;
  }

  ResetState();
}



void MVPlane::ChangePlane(const uint8_t *pNewPlane, int nNewPitch)
{
  if (!isFilled)
  {
    // noffsetPadding is pixelsize aware
    BitBlt(pPlane[0] + nOffsetPadding, nPitch, pNewPlane, nNewPitch, (nWidth << pixelsize_shift), nHeight);
    isFilled = true;
  }
}



void MVPlane::Pad()
{
  // npitch is pixelsize aware
  if (!isPadded)
  {
    if (pixelsize == 1)
      Padding::PadReferenceFrame<uint8_t>(pPlane[0], nPitch, nHPadding, nVPadding, nWidth, nHeight);
    else if (pixelsize == 2)
      Padding::PadReferenceFrame<uint16_t>(pPlane[0], nPitch, nHPadding, nVPadding, nWidth, nHeight);
    else
      Padding::PadReferenceFrame<float>(pPlane[0], nPitch, nHPadding, nVPadding, nWidth, nHeight);
    isPadded = true;
  }
}



void MVPlane::refine_start()
{
  if (!isRefined)
  {
    if (nPel == 2)
    {
      _sched_refine.start(_plan_refine, *this, &MVPlane::refine_pel2);
    }
    else if (nPel == 4)
    {
      _sched_refine.start(_plan_refine, *this, &MVPlane::refine_pel4);
    }
  }
}



void MVPlane::refine_wait()
{
  if (!isRefined)
  {
    if (nPel > 1)
    {
      _sched_refine.wait();
    }

    isRefined = true;
  }
}


// if a non-static template function is in cpp, we have to instantiate it
template void MVPlane::RefineExt<uint8_t>(const uint8_t *pSrc2x_8, int nSrc2xPitch, bool isExtPadded);
template void MVPlane::RefineExt<uint16_t>(const uint8_t *pSrc2x_8, int nSrc2xPitch, bool isExtPadded);
template void MVPlane::RefineExt<float>(const uint8_t *pSrc2x_8, int nSrc2xPitch, bool isExtPadded);

template<typename pixel_t>
void MVPlane::RefineExt(const uint8_t *pSrc2x_8, int nSrc2xPitch, bool isExtPadded) // copy from external upsized clip
{
  const pixel_t *pSrc2x = reinterpret_cast<const pixel_t *>(pSrc2x_8);
  nSrc2xPitch /= sizeof(pixel_t);

  if ((nPel == 2) && (!isRefined))
  {
    // pel clip may be already padded (i.e. is finest clip)
    int offset = isExtPadded ? 0 : nPitch * nVPadding / sizeof(pixel_t) + nHPadding;
    pixel_t* pp1 = reinterpret_cast<pixel_t *>(pPlane[1]) + offset;
    pixel_t* pp2 = reinterpret_cast<pixel_t *>(pPlane[2]) + offset;
    pixel_t* pp3 = reinterpret_cast<pixel_t *>(pPlane[3]) + offset;

    for (int h = 0; h < nHeight; h++) // assembler optimization?
    {
      for (int w = 0; w < nWidth; w++)
      {
        pp1[w] = pSrc2x[(w << 1) + 1];
        pp2[w] = pSrc2x[(w << 1) + nSrc2xPitch];
        pp3[w] = pSrc2x[(w << 1) + nSrc2xPitch + 1];
      }
      pp1 += nPitch / sizeof(pixel_t);
      pp2 += nPitch / sizeof(pixel_t);
      pp3 += nPitch / sizeof(pixel_t);
      pSrc2x += nSrc2xPitch * 2;
    }
    if (!isExtPadded)
    {
      Padding::PadReferenceFrame<pixel_t>(pPlane[1], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[2], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[3], nPitch, nHPadding, nVPadding, nWidth, nHeight);
    }
    isPadded = true;
  }
  else if ((nPel == 4) && (!isRefined))
  {
    // pel clip may be already padded (i.e. is finest clip)
    int offset = isExtPadded ? 0 : nPitch * nVPadding / sizeof(pixel_t) + nHPadding;
    pixel_t* pp1 = reinterpret_cast<pixel_t *>(pPlane[1]) + offset;
    pixel_t* pp2 = reinterpret_cast<pixel_t *>(pPlane[2]) + offset;
    pixel_t* pp3 = reinterpret_cast<pixel_t *>(pPlane[3]) + offset;
    pixel_t* pp4 = reinterpret_cast<pixel_t *>(pPlane[4]) + offset;
    pixel_t* pp5 = reinterpret_cast<pixel_t *>(pPlane[5]) + offset;
    pixel_t* pp6 = reinterpret_cast<pixel_t *>(pPlane[6]) + offset;
    pixel_t* pp7 = reinterpret_cast<pixel_t *>(pPlane[7]) + offset;
    pixel_t* pp8 = reinterpret_cast<pixel_t *>(pPlane[8]) + offset;
    pixel_t* pp9 = reinterpret_cast<pixel_t *>(pPlane[9]) + offset;
    pixel_t* pp10 = reinterpret_cast<pixel_t *>(pPlane[10]) + offset;
    pixel_t* pp11 = reinterpret_cast<pixel_t *>(pPlane[11]) + offset;
    pixel_t* pp12 = reinterpret_cast<pixel_t *>(pPlane[12]) + offset;
    pixel_t* pp13 = reinterpret_cast<pixel_t *>(pPlane[13]) + offset;
    pixel_t* pp14 = reinterpret_cast<pixel_t *>(pPlane[14]) + offset;
    pixel_t* pp15 = reinterpret_cast<pixel_t *>(pPlane[15]) + offset;

    for (int h = 0; h < nHeight; h++) // assembler optimization?
    {
      for (int w = 0; w < nWidth; w++)
      {
        pp1[w] = pSrc2x[(w << 2) + 1];
        pp2[w] = pSrc2x[(w << 2) + 2];
        pp3[w] = pSrc2x[(w << 2) + 3];
        pp4[w] = pSrc2x[(w << 2) + nSrc2xPitch];
        pp5[w] = pSrc2x[(w << 2) + nSrc2xPitch + 1];
        pp6[w] = pSrc2x[(w << 2) + nSrc2xPitch + 2];
        pp7[w] = pSrc2x[(w << 2) + nSrc2xPitch + 3];
        pp8[w] = pSrc2x[(w << 2) + nSrc2xPitch * 2];
        pp9[w] = pSrc2x[(w << 2) + nSrc2xPitch * 2 + 1];
        pp10[w] = pSrc2x[(w << 2) + nSrc2xPitch * 2 + 2];
        pp11[w] = pSrc2x[(w << 2) + nSrc2xPitch * 2 + 3];
        pp12[w] = pSrc2x[(w << 2) + nSrc2xPitch * 3];
        pp13[w] = pSrc2x[(w << 2) + nSrc2xPitch * 3 + 1];
        pp14[w] = pSrc2x[(w << 2) + nSrc2xPitch * 3 + 2];
        pp15[w] = pSrc2x[(w << 2) + nSrc2xPitch * 3 + 3];
      }
      pp1 += nPitch / sizeof(pixel_t);
      pp2 += nPitch / sizeof(pixel_t);
      pp3 += nPitch / sizeof(pixel_t);
      pp4 += nPitch / sizeof(pixel_t);
      pp5 += nPitch / sizeof(pixel_t);
      pp6 += nPitch / sizeof(pixel_t);
      pp7 += nPitch / sizeof(pixel_t);
      pp8 += nPitch / sizeof(pixel_t);
      pp9 += nPitch / sizeof(pixel_t);
      pp10 += nPitch / sizeof(pixel_t);
      pp11 += nPitch / sizeof(pixel_t);
      pp12 += nPitch / sizeof(pixel_t);
      pp13 += nPitch / sizeof(pixel_t);
      pp14 += nPitch / sizeof(pixel_t);
      pp15 += nPitch / sizeof(pixel_t);
      pSrc2x += nSrc2xPitch * 4;
    }
    if (!isExtPadded)
    {
      Padding::PadReferenceFrame<pixel_t>(pPlane[1], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[2], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[3], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[4], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[5], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[6], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[7], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[8], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[9], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[10], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[11], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[12], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[13], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[14], nPitch, nHPadding, nVPadding, nWidth, nHeight);
      Padding::PadReferenceFrame<pixel_t>(pPlane[15], nPitch, nHPadding, nVPadding, nWidth, nHeight);
    }
    isPadded = true;
  }
  isRefined = true;
}



void MVPlane::reduce_start(MVPlane *pReducedPlane)
{
  if (!pReducedPlane->isFilled)
  {
    _redp_ptr = pReducedPlane;
    _slicer_reduce.start(pReducedPlane->nHeight, *this, &MVPlane::reduce_slice, 4);
  }
}



void	MVPlane::reduce_wait()
{
  assert(_redp_ptr != 0);

  if (!_redp_ptr->isFilled)
  {
    _slicer_reduce.wait();

    _redp_ptr->isFilled = true;
    _redp_ptr = 0;
  }
}



void MVPlane::WritePlane(FILE *pFile)
{
  // noffsetPadding is pixelsize aware
  for (int i = 0; i < nHeight; i++)
  {
    fwrite(pPlane[0] + i * nPitch + nOffsetPadding, 1, (nWidth << pixelsize_shift), pFile);
  }
}



void MVPlane::refine_pel2(SchedulerRefine::TaskData &td)
{
  assert(&td != 0);

  switch (td._task_index)
  {
  case 0:  break;	// Nothing on the root node
  case 1:
    switch (nSharp)
    {
    case 0: _bilin_hor_ptr(pPlane[1], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    case 1: _bicubic_hor_ptr(pPlane[1], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    default: _wiener_hor_ptr(pPlane[1], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    }
  case 2:
    switch (nSharp)
    {
    case 0: _bilin_ver_ptr(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    case 1: _bicubic_ver_ptr(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    default: _wiener_ver_ptr(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    }
    break;
  case 3:
    switch (nSharp)
    {
    case 0: _bilin_dia_ptr(pPlane[3], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    case 1: _bicubic_hor_ptr(pPlane[3], pPlane[2], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;	// faster from ready-made horizontal
    default: _wiener_hor_ptr(pPlane[3], pPlane[2], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;	// faster from ready-made horizontal
    }
    break;
  default:
    assert(false);
    break;
  }
}



void MVPlane::refine_pel4(SchedulerRefine::TaskData &td)
{
  assert(&td != 0);

  switch (td._task_index)
  {
  case 0:  break;	// Nothing on the root node
  case 1:  _average_ptr(pPlane[1], pPlane[0], pPlane[2], nPitch, nExtendedWidth, nExtendedHeight); break;
  case 2:
    switch (nSharp)
    {
    case 0: _bilin_hor_ptr(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    case 1: _bicubic_hor_ptr(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    default: _wiener_hor_ptr(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    }
    break;
  case 3:  _average_ptr(pPlane[3], pPlane[0] + 1 * pixelsize, pPlane[2], nPitch, nExtendedWidth - 1, nExtendedHeight); break;
  case 4:  _average_ptr(pPlane[4], pPlane[0], pPlane[8], nPitch, nExtendedWidth, nExtendedHeight); break;
  case 5:  _average_ptr(pPlane[5], pPlane[4], pPlane[6], nPitch, nExtendedWidth, nExtendedHeight); break;
  case 6:  _average_ptr(pPlane[6], pPlane[2], pPlane[10], nPitch, nExtendedWidth, nExtendedHeight); break;
  case 7:  _average_ptr(pPlane[7], pPlane[4] + 1 * pixelsize, pPlane[6], nPitch, nExtendedWidth - 1, nExtendedHeight); break;
  case 8:
    switch (nSharp)
    {
    case 0: _bilin_ver_ptr(pPlane[8], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    case 1: _bicubic_ver_ptr(pPlane[8], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    default: _wiener_ver_ptr(pPlane[8], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    }
    break;
  case 9:  _average_ptr(pPlane[9], pPlane[8], pPlane[10], nPitch, nExtendedWidth, nExtendedHeight); break;
  case 10:
    switch (nSharp)
    {
    case 0: _bilin_dia_ptr(pPlane[10], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;
    case 1: _bicubic_hor_ptr(pPlane[10], pPlane[8], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;	// faster from ready-made horizontal
    default: _wiener_hor_ptr(pPlane[10], pPlane[8], nPitch, nPitch, nExtendedWidth, nExtendedHeight, bits_per_pixel); break;	// faster from ready-made horizontal
    }
    break;
  case 11: _average_ptr(pPlane[11], pPlane[8] + 1*pixelsize, pPlane[10], nPitch, nExtendedWidth - 1, nExtendedHeight); break;
  case 12: _average_ptr(pPlane[12], pPlane[0] + nPitch, pPlane[8], nPitch, nExtendedWidth, nExtendedHeight - 1); break;
  case 13: _average_ptr(pPlane[13], pPlane[12], pPlane[14], nPitch, nExtendedWidth, nExtendedHeight); break;
  case 14: _average_ptr(pPlane[14], pPlane[2] + nPitch, pPlane[10], nPitch, nExtendedWidth, nExtendedHeight - 1); break;
  case 15: _average_ptr(pPlane[15], pPlane[12] + 1 * pixelsize, pPlane[14], nPitch, nExtendedWidth - 1, nExtendedHeight); break;
  default:
    assert(false);
    break;
  }
}



void MVPlane::reduce_slice(SlicerReduce::TaskData &td)
{
  assert(&td != 0);
  assert(_redp_ptr != 0);
  // noffsetPadding is pixelsize aware
  MVPlane &red = *_redp_ptr; // target (smaller dimension)
  _reduce_ptr(
    red.pPlane[0] + red.nOffsetPadding, // shrink to
    pPlane[0] + nOffsetPadding, // shrink from
    red.nPitch, nPitch,
    red.nWidth, red.nHeight, td._y_beg, td._y_end,
    cpuFlags
  );
}

void MVPlane::CalcShiftKernel(float* fKernel, float fPelShift, int iKS)
{
  float fPi = 3.14159265f;
  int iKS_d2 = iKS / 2;

  for (int i = 0; i < iKS; i++)
  {
    float fArg = (float)(i - iKS_d2) * fPi + fPelShift;
    fKernel[i] = fSinc(fArg);

    // Lanczos weighting
    float fArgLz = (float)(i - iKS_d2) * fPi / (float)(iKS_d2);
    fKernel[i] *= fSinc(fArgLz);;
  }

  float fSum = 0.0f;
  for (int i = 0; i < iKS; i++)
  {
    fSum += fKernel[i];
  }

  for (int i = 0; i < iKS; i++)
  {
    fKernel[i] /= fSum;
  }
}

void MVPlane::CalcShiftKernels(short* psKernel, float fPelShift, int iKS)
{
  float fPi = 3.14159265f;
  int iKS_d2 = iKS / 2;
  float fKernel[SHIFTKERNELSIZE];

  for (int i = 0; i < iKS; i++)
  {
    float fArg = ((float)(i - iKS_d2) + fPelShift) * fPi;
    fKernel[i] = fSinc(fArg);

    // Lanczos weighting
    float fArgLz = (float)(i - iKS_d2) * fPi / (float)(iKS_d2);
    fKernel[i] *= fSinc(fArgLz);
  }

  float fSum = 0.0f;
  for (int i = 0; i < iKS; i++)
  {
    fSum += fKernel[i];
  }

  for (int i = 0; i < iKS; i++)
  {
    fKernel[i] /= fSum;
    psKernel[i] = (short)(fKernel[i] * 255); // or 256 ??
  }
}

const uint8_t* MVPlane::GetPointerSubShift(int nX, int nY, int iBlockSizeX, int iBlockSizeY, int& pDstPitch) const
{
  int NPELL2 = nPel >> 1;

  int nfullX = nX + nHPaddingPel;
  int nfullY = nY + nVPaddingPel;

  int iMASK = (1 << NPELL2) - 1;

  int i_dx = (nfullX & iMASK);
  int i_dy = (nfullY & iMASK);

  float* pfKrnH = 0;
  float* pfKrnV = 0;

  short* psKrnH = 0;
  short* psKrnV = 0;

  switch (i_dx)
  {
  case 0:
    pfKrnH = 0;
    psKrnH = 0;
    break;
  case 1:
    pfKrnH = (float*)fKernelSh_01;
    psKrnH = (short*)sKernelSh_01;
    break;
  case 2:
    pfKrnH = (float*)fKernelSh_10;
    psKrnH = (short*)sKernelSh_10;
    break;
  case 3:
    pfKrnH = (float*)fKernelSh_11;
    psKrnH = (short*)sKernelSh_11;
    break;
  }

  switch (i_dy)
  {
  case 0:
    pfKrnV = 0;
    psKrnV = 0;
    break;
  case 1:
    pfKrnV = (float*)fKernelSh_01;
    psKrnV = (short*)sKernelSh_01;
    break;
  case 2:
    pfKrnV = (float*)fKernelSh_10;
    psKrnV = (short*)sKernelSh_10;
    break;
  case 3:
    pfKrnV = (float*)fKernelSh_11;
    psKrnV = (short*)sKernelSh_11;
    break;
  }

  nfullX >>= NPELL2;
  nfullY >>= NPELL2;

  uint8_t* pSrc =(uint8_t*)GetAbsolutePointerPel <0>(nfullX, nfullY);

  if (i_dx == 0 && i_dy == 0) // no sub shift required
  {
    pDstPitch = nPitch;
    return pSrc;
  }

  int nShiftedBufPitch = (iBlockSizeX << pixelsize_shift);

  if (hasAVX2)
  {
    if (iBlockSizeX == 8 && iBlockSizeY == 8 && pixelsize == 1)
    {
    // SubShiftBlock8x8_KS8_uint8_avx2(pSrc, pShiftedBlockBuf, iBlockSizeX, iBlockSizeY, pfKrnH, pfKrnV, nPitch, nShiftedBufPitch, SHIFTKERNELSIZE);
      SubShiftBlock8x8_KS4_i16_uint8_avx2(pSrc, pShiftedBlockBuf, iBlockSizeX, iBlockSizeY, (float*)psKrnH, (float*)psKrnV, nPitch, nShiftedBufPitch, SHIFTKERNELSIZE_I16);
    }
    else if (iBlockSizeX == 4 && iBlockSizeY == 4 && pixelsize == 1)
    {
      SubShiftBlock4x4_KS8_uint8_avx2(pSrc, pShiftedBlockBuf, iBlockSizeX, iBlockSizeY, pfKrnH, pfKrnV, nPitch, nShiftedBufPitch, SHIFTKERNELSIZE);
    }
    else
      _sub_shift_ptr(pSrc, pShiftedBlockBuf, iBlockSizeX, iBlockSizeY, pfKrnH, pfKrnV, nPitch, nShiftedBufPitch, SHIFTKERNELSIZE);
  }
  else
  {
    _sub_shift_ptr(pSrc, pShiftedBlockBuf, iBlockSizeX, iBlockSizeY, pfKrnH, pfKrnV, nPitch, nShiftedBufPitch, SHIFTKERNELSIZE);
  }

  pDstPitch = nShiftedBufPitch;
  return pShiftedBlockBuf;

}
