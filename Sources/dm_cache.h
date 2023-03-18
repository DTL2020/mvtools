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


#ifndef	__MV_DM_cache__
#define	__MV_DM_cache__

#include <stdint.h>
#include "types.h"

struct DM2FRAMES
{
  int fr0;
  int fr1;
  int dm; // do it enough ? may be sad_t ?
  bool bValid;
};

class DM_cache
{
  int size;
  DM2FRAMES* pBuff;

public:
  DM_cache(int size);
  ~DM_cache();

bool Get(int iFr0, int iFr1, int* iDM);
void PushNew(int iFr0, int iFr1, int iDM);
void Invalidate(void);

};

#endif	// __MV_DM_cache__

