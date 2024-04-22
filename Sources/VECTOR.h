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


#ifndef	__MV_VECTOR__
#define	__MV_VECTOR__


#include "types.h"

#pragma pack (push, 16)

struct VECTOR
{
  union
  {
    struct
    {
      int x;
      int y;
    };
    int coord [2];
  };
  sad_t sad;
};

#pragma pack (pop)

struct VECTOR_XY
{
  int x;
  int y;
};


#endif	// __MV_VECTOR__
