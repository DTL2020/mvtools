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

#ifndef __GOPLANES__
#define __GOPLANES__



#include "PlaneOfBlocks.h"
#include "avisynth.h"


class MVGroupOfFrames;

class GroupOfPlanes
{
  int            nBlkSizeX;
  int            nBlkSizeY;
  int            nLevelCount;
  int            nPel;
  int            nFlags;
  int            nOverlapX;
  int            nOverlapY;
  int            xRatioUV; // PF 160729
  int            yRatioUV;
    int           chromaSADScale; // P.F. 170504
    int            pixelsize; // PF 160729
    int bits_per_pixel;
  int            divideExtra;
  bool           _mt_flag;
  int            optSearchOption; // DTL test
  float          scaleCSADfine; //DTL test
  int            iUseSubShift; // DTL test
  int            DMFlags; // DTL test
  int            AreaMode; // DTL test (see MAnalyse.h for description) - todo: remove from class and use at Search and Recalc calls
  int            AMDiffSAD; 
  int            AMstep;
  int            AMoffset;
  int            AMpel;
  
  conc::ObjPool <DCTClass> *
                 _dct_pool_ptr;
  PlaneOfBlocks **
                 planes;

public :
  GroupOfPlanes(
    int _nBlkSizeX, int _nBlkSizeY, int _nLevelCount, int _nPel, int _nFlags,
    int _nOverlapX, int _nOverlapY, int _nBlkX, int _nBlkY, int _xRatioUV, int _yRatioUV, int _divideExtra, int _pixelsize, int _bits_per_pixel, 
    conc::ObjPool <DCTClass> *dct_pool_ptr,
    bool mt_flag, int _chromaSADScale, int _optSearchOption, float _scaleCSADfine, int _iUseSubShift, int DMFlags,
    int _AreaMode, int _AMDiffSAD, int _AMstep, int _AMoffset, int _AMpel,
    IScriptEnvironment *env);
  ~GroupOfPlanes ();
  void           SearchMVs (
    MVGroupOfFrames *pSrcGOF, MVGroupOfFrames *pRefGOF,
    SearchType searchType, int nSearchParam, int _PelSearch, int _nLambda,
    sad_t _lsad, int _pnew, int _plevel, bool _global, int flags, int *out,
    short * outfilebuf, int fieldShift, int _pzero, int _pglobal, sad_t badSAD,
    int badrange, bool meander, int *vecPrev, bool tryMany, int optPredictorType, int PTpel,
    int AMflags, int AMavg, int AMpt, SearchType AMst, int AMsp,
    int TMAvg, int MDp, int ScanDir, int MPM);
  void           WriteDefaultToArray (int *array);
  int            GetArraySize ();
  void           ExtraDivide (int *out, int flags);
  void           RecalculateMVs (
    MVClip &mvClip, MVGroupOfFrames *pSrcGOF, MVGroupOfFrames *pRefGOF,
    SearchType _searchType, int _nSearchParam, int _nLambda, sad_t _lsad,
    int _pnew, int flags, int *out, short * outfilebuf, int fieldShift,
    sad_t thSAD, int smooth, bool meander, int optPredictorType, int AreaMode, int AMstep, int AMoffset, float fAMthVSMang, int AMflags, int AMavg,
    bool _global, int pglobal, int pzero);
  PlaneOfBlocks* GetPlane(int iPlane) { return planes[iPlane]; };
};

#endif
