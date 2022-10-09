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

#include "DisMetric.h"

DisMetric::DisMetric(int iBlkSizeX, int iBlkSizeY, int iBPP, int _pixelsize, arch_t _arch, int metric_flags)
{
  nBlkSizeX = iBlkSizeX;
  nBlkSizeY = iBlkSizeY;
  nBPP = iBPP;
  arch = _arch;
  nMetricFlags = metric_flags;
  pixelsize = _pixelsize;

  if (metric_flags & MEF_SAD)
  {
    SAD = get_sad_function(nBlkSizeX, nBlkSizeY, nBPP, arch);
  }

  if (metric_flags & MEF_SSIM_L)
  {
    SSIM_L = get_ssim_function_l(nBlkSizeX, nBlkSizeY, nBPP, arch);
  }

  if (metric_flags & MEF_SSIM_CS)
  {
    SSIM_CS = get_ssim_function_cs(nBlkSizeX, nBlkSizeY, nBPP, arch);
  }

  veryBigSAD = (3 * nBlkSizeX * nBlkSizeY * (pixelsize == 4 ? 1 : (1 << nBPP))); // * 256, pixelsize==2 -> 65536. Float:1

}

DisMetric::~DisMetric()
{
}

int DisMetric::GetDisMetric(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch)
{
  int iRetDisMetric = 0;

  if (nMetricFlags & MEF_SAD)
  {
    iRetDisMetric += SAD(pSrc, nSrcPitch, pRef, nRefPitch);
  }

  if ((nMetricFlags & MEF_SSIM_L) && !(nMetricFlags & MEF_SSIM_CS))
  {
    iRetDisMetric += (int)(((1.0f - SSIM_L(pSrc, nSrcPitch, pRef, nRefPitch)) * (float)(veryBigSAD >> 1)));
  }

  if (nMetricFlags & MEF_SSIM_CS && !(nMetricFlags & MEF_SSIM_L))
  {
    iRetDisMetric += (int)(((1.0f - SSIM_CS(pSrc, nSrcPitch, pRef, nRefPitch)) * (float)(veryBigSAD >> 1))); // SSIM may be low as -1.0f
  }

  if (nMetricFlags & MEF_SSIM_CS && (nMetricFlags & MEF_SSIM_L))
  {
    iRetDisMetric += (int)(((1.0f - SSIM_L(pSrc, nSrcPitch, pRef, nRefPitch) * SSIM_CS(pSrc, nSrcPitch, pRef, nRefPitch)) * (float)(veryBigSAD >> 1))); // SSIM may be low as -1.0f
  }

  // todo: currently sum may be num of metrics * veryBigSAD ! may be norm max value to verybigsad ?

  return iRetDisMetric;

}
