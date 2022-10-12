// Selector-aggregator class for using of single or several
// dissimilarity metrics between blocks

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


#ifndef	__MV_DisMetric__
#define	__MV_DisMetric__

#include <stdint.h>
#include "types.h"
#include "SADFunctions.h"
#include "SSIMFunctions.h"

enum {
  MEF_SAD = 0x01,       // SAD
  MEF_SSIM_L = 0x02,    // SSIM luma
  MEF_SSIM_CS = 0x04,   // SSIM contrast and structure
  MEF_MS_SSIM = 0x08    // MS-SSIM (multi-scale SSIM)
};


class DisMetric
{
  SADFunction* SAD;  /* function which computes the sad */
  SSIMFunction* SSIM_L; /* function which computes the SSIM luma only */
  SSIMFunction* SSIM_CS; /* function which computes the SSIM contrast and structure */
  SSIMFunction* SSIM_FULL; /* function which computes the SSIM full components (luma and contrast and structure) */

  int nBlkSizeX;
  int nBlkSizeY;
  int nBPP;
  int pixelsize;
  arch_t arch;
  int nMetricFlags;
  sad_t maxSAD;

public:
  DisMetric(int iBlkSizeX, int iBlkSizeY, int iBPP, int _pixelsize, arch_t _arch, int metric_flags);
  ~DisMetric();

int GetDisMetric(const uint8_t* pSrc, int nSrcPitch, const uint8_t* pRef, int nRefPitch);
};




#endif	// __MV_DisMetric__

