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

#include "BlockArea.h"

BlockArea::BlockArea(int iBlkSizeX, int iBlkSizeY, int iBlockAreaSize, int iPixelSize, int iPel, arch_t _arch, int metric_flags)
{
  nBlkSizeX = iBlkSizeX;
  nBlkSizeY = iBlkSizeY;
  nBlockAreaSize = iBlockAreaSize;
  nMetricFlags = metric_flags;
  nPixelSize = iPixelSize;

  nPel = iPel;
  arch = _arch;

  int iBuffSize = (nBlkSizeX + nBlkSizeX * nBlockAreaSize * 2) * (nBlkSizeY + nBlkSizeY * nBlockAreaSize * 2) * nPixelSize;

  buff = new uint8_t[iBuffSize];

}

BlockArea::~BlockArea()
{
  delete buff;
}

int BlockArea::Update(const uint8_t* pSrc, int nSrcPitch, uint8_t* pPlaneMinAddr, uint8_t* pPlaneMaxAddr)
{
  return 0;
};

int BlockArea::GetDM(const uint8_t* pSrc, int nSrcPitch, VECTOR mv)
{
  return 0;
};

bool BlockArea::GetBlock(uint8_t* pDst, int nDstPitch, VECTOR mv)
{
  return false;
};
