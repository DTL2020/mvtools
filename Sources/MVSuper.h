// MVTools v2

// Copyright(c)2008 A.G.Balakhnin aka Fizick
// Prepare super clip (hierachical levels data) for MVAnalyse

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

#ifndef __MV_SUPER__
#define __MV_SUPER__

#include "commonfunctions.h"
#include "yuy2planes.h"
#include	"avisynth.h"
#include "stdint.h"


MV_FORCEINLINE int PlaneHeightLuma(int src_height, int level, int yRatioUV, int vpad)
{
  int height = src_height;

  for (int i = 1; i <= level; i++)
  {
    //		height = (height/2) - ((height/2) % yRatioUV) ;
    height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV : ((height / yRatioUV) / 2) * yRatioUV;
  }
  return height;
}

MV_FORCEINLINE int PlaneWidthLuma(int src_width, int level, int xRatioUV, int hpad)
{
  int width = src_width;

  for (int i = 1; i <= level; i++)
  {
    //		width = (width/2) - ((width/2) % xRatioUV) ;
    width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV : ((width / xRatioUV) / 2) * xRatioUV;
  }
  return width;
}

// PF: OK no need for xRatioUV here
// returned offset is pixelsize-aware because plane_pitch is in bytes
MV_FORCEINLINE unsigned int PlaneSuperOffset(bool chroma, int src_height, int level, int pel, int vpad, int plane_pitch, int yRatioUV)
{
  // storing subplanes in superframes may be implemented by various ways
  int height = src_height; // luma or chroma

  unsigned int offset;

  if (level == 0)
  {
    offset = 0;
  }
  else
  {
    offset = pel * pel*plane_pitch*(src_height + vpad * 2);

    for (int i = 1; i < level; i++)
    {
      height = chroma ? 
                  PlaneHeightLuma(src_height*yRatioUV, i, yRatioUV, vpad*yRatioUV) / yRatioUV : 
        // PF 20181113: Why is it good to use yRatioUV for luma???
        PlaneHeightLuma(src_height, i, yRatioUV, vpad);

      offset += plane_pitch * (height + vpad * 2);
    }
  }
  return offset;
}


class MVSuper
  : public GenericVideoFilter
{
private:
  bool has_at_least_v8;

protected:

  int            nHPad;
  int            nVPad;
  int            nPel;
  int            nLevels;
  int            sharp;
  int            rfilter; // frame reduce filter mode
  PClip          pelclip; // upsized source clip with doubled frame width and heigth (used for pel=2)
  //bool           isse;   //PF maybe obsolate or debug 160729
  int            cpuFlags;
  bool           planar; //v2.0.0.7 PF maybe obsolate 160729

  int            nWidth;
  int            nHeight;

  int            yRatioUV;
  int            xRatioUV;

  int pixelsize; // PF
  int bits_per_pixel;

  bool           chroma;
  int            pixelType; //PF maybe obsolate 160729 used in YUY decision along with planar flag
  bool           usePelClip;
  int            nSuperWidth;
  int            nSuperHeight;

  MVPlaneSet     nModeYUV;

  YUY2Planes *   SrcPlanes;  //PF maybe obsolate 160729 YUY
//	YUY2Planes *   DstPlanes;
  YUY2Planes *   SrcPelPlanes; //PF maybe obsolate 160729 YUY

  MVGroupOfFrames *
    pSrcGOF;
  bool           isPelClipPadded;

  bool           _mt_flag; // PF maybe 2.6.0.5
  bool           pel_refine; // 2.7.46 - default true, generate subpel buffers for pel > 1 or not (not needed for DX12_ME and internal sub shifting in MDegrainN)

public:

  MVSuper(
    PClip _child, int _hpad, int _vpad, int pel, int _levels, bool _chroma,
    int _sharp, int _rfilter, PClip _pelclip, bool _isse, bool _planar,
    bool mt_flag, bool _pel_refine, IScriptEnvironment* env
  );
  ~MVSuper();

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
  }

};

#endif
