// Author: Manao
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



#include "commonfunctions.h"
#include "FakePlaneOfBlocks.h"
#include	"MVInterface.h"
#include <stdlib.h>


FakePlaneOfBlocks::FakePlaneOfBlocks(int sizeX, int sizeY, int lv, int pel, int _nOverlapX, int _nOverlapY, int _nBlkX, int _nBlkY, bool bMVsArrayOnly)
{
   nBlkSizeX = sizeX;
   nBlkSizeY = sizeY;
   nOverlapX = _nOverlapX;
   nOverlapY = _nOverlapY;
   nBlkX = _nBlkX;
   nBlkY = _nBlkY;
  nWidth_Bi = nOverlapX + nBlkX*(nBlkSizeX - nOverlapX);//w;
  nHeight_Bi = nOverlapY + nBlkY*(nBlkSizeY - nOverlapY);//h;
//   nBlkX = (nWidth_Bi - nOverlapX) / (nBlkSizeX - nOverlapX); // without remainder
//   nBlkY = (nHeight_Bi - nOverlapY) / (nBlkSizeY - nOverlapY); //
   nBlkCount = nBlkX * nBlkY;
   nPel = pel;
   bnMVsArrayOnly = bMVsArrayOnly;

#ifdef _WIN32
   // to prevent cache set overloading when accessing fpob MVs arrays - add random L2L3_CACHE_LINE_SIZE-bytes sized offset to different allocations
   size_t random = rand();
   random *= RAND_OFFSET_MAX;
   random /= RAND_MAX;
   random *= L2L3_CACHE_LINE_SIZE;

   SIZE_T stSizeToAlloc = nBlkCount * sizeof(VECTOR) + RAND_OFFSET_MAX * L2L3_CACHE_LINE_SIZE;

   pbMVsArray_a = (BYTE*)VirtualAlloc(0, stSizeToAlloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); // 4KByte page aligned address
   pMVsArray = (VECTOR*)(pbMVsArray_a + random);
#else
   pMVsArray = new VECTOR[nBlkCount]; // allocate in heap ?
#endif


  nLogPel = ilog2(nPel);
  nLogScale = lv;
  nScale = iexp2(nLogScale);

  blocks = new FakeBlockData [nBlkCount];
  for ( int j = 0, blkIdx = 0; j < nBlkY; j++ )
    for ( int i = 0; i < nBlkX; i++, blkIdx++ )
      blocks[blkIdx].Init(i * (nBlkSizeX - nOverlapX), j * (nBlkSizeY - nOverlapY));
}



FakePlaneOfBlocks::~FakePlaneOfBlocks()
{

  delete[] blocks;

#ifdef _WIN32
  VirtualFree(pbMVsArray_a, 0, MEM_RELEASE);
#else
  delete[] pMVsArray;
#endif
}

void FakePlaneOfBlocks::Update(const int *array)
{
  // need to copy, pointer not keeped ?
  if (bnMVsArrayOnly) // for faster MDegrain
  {
    memcpy(pMVsArray, array, nBlkCount * sizeof(VECTOR));
  }
  else // for compatibility with old filters/functions
  {
    array += 0;
    for (int i = 0; i < nBlkCount; i++)
    {
      blocks[i].Update(array);
      array += N_PER_BLOCK;
    }
  }
}

bool FakePlaneOfBlocks::IsSceneChange(sad_t nTh1, int nTh2) const
{
  int sum = 0;
  for (int i = 0; i < nBlkCount; i++)
    //    sum += ( blocks[i].GetSAD() > nTh1 ) ? 1 : 0;
    sum += (pMVsArray[i].sad > nTh1) ? 1 : 0;

  return ( sum > nTh2 );
}
