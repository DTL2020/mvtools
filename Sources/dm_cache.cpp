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

#include "dm_cache.h"
#include <malloc.h>

DM_cache::DM_cache(int _size)
{
  size = _size;
  pBuff = (DM2FRAMES*)malloc(sizeof(DM2FRAMES) * size);

  Invalidate();
}

DM_cache::~DM_cache()
{
  free(pBuff);
}

void DM_cache::Invalidate(void)
{
  for (int i = 0; i < size; i++)
  {
    pBuff[i].bValid = false;
  }
}

bool DM_cache::Get(int iFr0, int iFr1, int *iDM)
{
  // search over all buff to find this frames pair (both forward or backward DM)
  for (int i = 0; i < size; i++)
  {
    if (((pBuff[i].fr0 == iFr0) && (pBuff[i].fr1 == iFr1) || (pBuff[i].fr0 == iFr1) && (pBuff[i].fr1 == iFr0)) && pBuff[i].bValid)
    {
      *iDM = pBuff[i].dm;
      return true;
    }
  }

  return false;
}

void DM_cache::PushNew(int iFr0, int iFr1, int iDM)
{
  // make shift to 1 value to the beginning of list
  // skip first storage
  for (int i = 0; i < size - 1; i++)
  {
    pBuff[i] = pBuff[i + 1]; // do = opertor copy all ?
  }

  // add to the end
  pBuff[size - 1].fr0 = iFr0;
  pBuff[size - 1].fr1 = iFr1;
  pBuff[size - 1].dm = iDM;
  pBuff[size - 1].bValid = true;

}

