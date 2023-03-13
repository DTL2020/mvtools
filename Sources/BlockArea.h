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


#ifndef	__MV_BlockArea__
#define	__MV_BlockArea__

#include <stdint.h>
#include "types.h"
#include "CopyCode.h"
#include "VECTOR.h"

class BlockArea
{
  int nBlkSizeX;
  int nBlkSizeY;
  int nPixelSize;
  int nBlockAreaSize; // in number of BlkSizeX at left and right and BlkSizeY at top and bottom
  int nMetricFlags;
  int nPel;
  arch_t arch;
  uint8_t* buff;

public:
  BlockArea(int iBlkSizeX, int iBlkSizeY, int iBlockAreaSize, int iPixelSize, int iPel, arch_t _arch, int metric_flags);
  ~BlockArea();

int Update(const uint8_t* pSrc, int nSrcPitch, uint8_t* pPlaneMinAddr, uint8_t* pPlaneMaxAddr);
int GetDM(const uint8_t* pSrc, int nSrcPitch, VECTOR mv);
bool GetBlock(uint8_t* pDst, int nDstPitch, VECTOR mv);

};

#endif	// __MV_BlockArea__

