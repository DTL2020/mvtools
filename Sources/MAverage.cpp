/*****************************************************************************
        MAverage.cpp
*Tab=3***********************************************************************/



#if defined (_MSC_VER)
//#pragma warning (1 : 4130 4223 4705 4706)
//#pragma warning (4 : 4355 4786 4800)
#endif



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include	"MAverage.h"
#include	<cassert>
#include  <algorithm>


/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



MAverage::MAverage(std::vector <::PClip> clip_arr, int _mode, IScriptEnvironment *env)
  : GenericVideoFilter(clip_arr[0])
  , iMode(_mode)
{
  assert(!clip_arr.empty());
  assert(&env != 0);

  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  nbr_clips = (int)clip_arr.size();
  m_clip_arr.resize(nbr_clips);


  for (int clip_cnt = 0; clip_cnt < nbr_clips; ++clip_cnt)
  {
//    VectData &vect_data = _vect_arr[clip_cnt];
    const ::VideoInfo &vd_vi = clip_arr[clip_cnt]->GetVideoInfo();

    // Checks basic parameters
    if (vd_vi.num_frames != vi.num_frames)
    {
      env->ThrowError(
        "MAverage: all vector clips should have the same number of frames."
      );
    }

    // Checks if the clip contains vectors
    if (vd_vi.nchannels >= 0 && vd_vi.nchannels < 9)
    {
      env->ThrowError("MAverage: invalid vector stream.");
    }
#if !defined(MV_64BIT)
    const MVAnalysisData &	mad =
      *reinterpret_cast <MVAnalysisData *> (vd_vi.nchannels);
#else
    // hack!
    uintptr_t p = (((uintptr_t)(unsigned int)vi.nchannels ^ 0x80000000) << 32) | (uintptr_t)(unsigned int)vi.sample_type;
    const MVAnalysisData &	mad = *reinterpret_cast <MVAnalysisData *> (p);
#endif
    if (mad.GetMagicKey() != MVAnalysisData::MOTION_MAGIC_KEY)
    {
      env->ThrowError("MAverage: invalid vector stream.");
    }

    // copy pointer to class for the first MV clip and init analysis data 
    if (clip_cnt == 0)
    {
      mVectorsInfo = mad;
      nBlkX = mad.nBlkX;
      nBlkSizeX = mad.nBlkSizeX;
      nBlkY = mad.nBlkY;
      nBlkSizeY = mad.nBlkSizeY;
      nPel = mad.nPel;
    }

    // save ptrs to input MV clips
    m_clip_arr[clip_cnt] = clip_arr[clip_cnt];
  }

}



::PVideoFrame __stdcall	MAverage::GetFrame(int n, ::IScriptEnvironment *env_ptr)
{
  PVideoFrame src = child->GetFrame(n, env_ptr);
  PVideoFrame dst = has_at_least_v8 ? env_ptr->NewVideoFrameP(vi, &src) : env_ptr->NewVideoFrame(vi); // frame property support
  const int* pData = reinterpret_cast<const int*>(src->GetReadPtr());
  int* pDst = reinterpret_cast<int*>(dst->GetWritePtr());

  PVideoFrame src_frames[MAX_AREAMODE_STEPS];
  const int* pDataSrc[MAX_AREAMODE_STEPS];

  for (int i = 0; i < nbr_clips; i++)
  {
    src_frames[i] = m_clip_arr[i]->GetFrame(n, env_ptr);
    pDataSrc[i] = reinterpret_cast<const int*>(src_frames[i]->GetReadPtr());
  }

  // Copy and fix header
  int headerSize = *pData;
  memcpy(pDst, pData, headerSize);

  const MVAnalysisData& hdr_src =
    *reinterpret_cast <const MVAnalysisData*> (pData + 1);
  MVAnalysisData& hdr_dst =
    *reinterpret_cast <MVAnalysisData*> (pDst + 1);

  // Use the main header as default value
  // Copy delta and direction from the original frame (for multi-vector clips)
  // Fix the direction if required
  hdr_dst = mVectorsInfo;
  hdr_dst.nDeltaFrame = hdr_src.nDeltaFrame;
  hdr_dst.isBackward = hdr_src.isBackward;

  // Copy all planes
  pData = reinterpret_cast<const int*>(reinterpret_cast<const char*>(pData) + headerSize);
  pDst = reinterpret_cast<int*>(reinterpret_cast<char*>(pDst) + headerSize);
  memcpy(pDst, pData, *pData * sizeof(int));

  // Size and validity of all block data
  int* pPlanes = pDst;
  if (pPlanes[1] == 0) return dst; // Marked invalid
  int* pEnd = pPlanes + *pPlanes;
  pPlanes += 2;

  for (int i = 0; i < nbr_clips; i++)
  {
    int* pPlane = (int*)(reinterpret_cast<const char*>(pDataSrc[i]) + headerSize);
    if (pPlane[1] == 0) return dst; // Marked invalid 
    pSrcPlanes[i] = pPlane + 2;
  }

  // If scaling vectors only (blocksize remains same) then must check if new vectors go out of frame

  // Dimensions of frame covered by blocks (where frame is not exactly divisible by block size there is a small border that will not be motion compensated)
  int widthCovered = (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX) * mVectorsInfo.nBlkX + mVectorsInfo.nOverlapX;
  int heightCovered = (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY) * mVectorsInfo.nBlkY + mVectorsInfo.nOverlapY;

  // Go through blocks at each level
  int level = mVectorsInfo.nLvCount - 1; // Start at coarsest level
  while (level >= 0)
  {
    int blocksSize = *pPlanes;
    VECTOR* pBlocks = reinterpret_cast<VECTOR*>(pPlanes + 1);
    pPlanes += blocksSize;

    for (int i = 0; i < nbr_clips; i++)
    {
      pSrcBlocks[i] = reinterpret_cast<VECTOR*>(pSrcPlanes[i] + 1);
      pSrcPlanes[i] += blocksSize;
    }
 
    // Width and height of this level in blocks...
    int levelNumBlocksX = ((widthCovered >> level) - mVectorsInfo.nOverlapX) / (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX);
    int levelNumBlocksY = ((heightCovered >> level) - mVectorsInfo.nOverlapY) / (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY);

    // ... and in pixels
    int levelWidth = mVectorsInfo.nWidth;
    int levelHeight = mVectorsInfo.nHeight;
    for (int i = 1; i <= level; i++)
    {
      int xRatioUV = mVectorsInfo.xRatioUV;
      int yRatioUV = mVectorsInfo.yRatioUV;
      levelWidth = (mVectorsInfo.nHPadding >= xRatioUV) ? ((levelWidth / xRatioUV + 1) / 2) * xRatioUV : ((levelWidth / xRatioUV) / 2) * xRatioUV;
      levelHeight = (mVectorsInfo.nVPadding >= yRatioUV) ? ((levelHeight / yRatioUV + 1) / 2) * yRatioUV : ((levelHeight / yRatioUV) / 2) * yRatioUV;
    }
//    int extendedWidth = levelWidth + 2 * mVectorsInfo.nHPadding; // Including padding
//    int extendedHeight = levelHeight + 2 * mVectorsInfo.nVPadding;

    // Padding is effectively smaller on coarser levels
//    int paddingXScaled = mVectorsInfo.nHPadding >> level;
//    int paddingYScaled = mVectorsInfo.nVPadding >> level;

    // Loop through block positions (top-left of each block, coordinates relative to top-left of padding)
    int x = mVectorsInfo.nHPadding;
    int y = mVectorsInfo.nVPadding;
    int xEnd = x + levelNumBlocksX * (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX);
    int yEnd = y + levelNumBlocksY * (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY); 
    while (y < yEnd)
    {
      // Max/min vector length for this block
//      int yMin = -mVectorsInfo.nPel * (y - mVectorsInfo.nVPadding + paddingYScaled);
//      int yMax = mVectorsInfo.nPel * (extendedHeight - y - mVectorsInfo.nBlkSizeY - mVectorsInfo.nVPadding + paddingYScaled);

      while (x < xEnd)
      {
//        int xMin = -mVectorsInfo.nPel * (x - mVectorsInfo.nHPadding + paddingXScaled);
//        int xMax = mVectorsInfo.nPel * (extendedWidth - x - mVectorsInfo.nBlkSizeX - mVectorsInfo.nHPadding + paddingXScaled);

        for (int i = 0; i < nbr_clips; i++)
        {
          vAMResults[i] = *pSrcBlocks[i];
        }

        VECTOR vOut;
        vOut.x = 0;
        vOut.y = 0;
        vOut.sad = 0;

        switch (iMode)
        {
          case 0:
            GetModeVECTORxy<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 1:
            GetMeanVECTORxy<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 2:
            GetModeVECTORvad<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 3:
            GetModeVECTORvld<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 4:
            GetMedianVECTORg<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 5:
            Get_IQM_VECTORxy<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 6:
            GetModeVECTORxyda<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 7:
            GetModeVECTORxydadm<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;

          case 8:
            GetLowestDM<uint8_t>(&vAMResults[0], &vOut, nbr_clips);
            break;
        }

        *pBlocks = vOut;

        // need to update DM in future
        pBlocks++;

        for (int i = 0; i < nbr_clips; i++)
        {
          pSrcBlocks[i]++;
        }

        // Next block position
        x += mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX;
      }
      x = mVectorsInfo.nHPadding;
      y += mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY;
    }
    if (reinterpret_cast<int*>(pBlocks) != pPlanes) env_ptr->ThrowError("MAverage: Internal error"); // Debugging check
    level--;
  }
  if (pPlanes != pEnd) env_ptr->ThrowError("MAverage: Internal error"); // Debugging check

  return dst;

}


/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetModeVECTORxy(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  // process dual coords in scalar C ?
  const int iMaxMVlength = std::max(nBlkX * nBlkSizeX, nBlkY * nBlkSizeY) * 2 * nPel; // hope it is enough ? todo: make global constant ?
  int MaxSumDM = iNumMVs * iMaxMVlength;

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow_x = MaxSumDM;
  int sum_minrow_y = MaxSumDM;
  int i_idx_minrow_x = 0;
  int i_idx_minrow_y = 0;

  for (int dmt_row = 0; dmt_row < iNumMVs; dmt_row++)
  {
    int sum_row_x = 0;
    int sum_row_y = 0;

    for (int dmt_col = 0; dmt_col < iNumMVs; dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      sum_row_x += std::abs(toMedian[dmt_row].x - toMedian[dmt_col].x);
      sum_row_y += std::abs(toMedian[dmt_row].y - toMedian[dmt_col].y);
    }

    if (sum_row_x < sum_minrow_x)
    {
      sum_minrow_x = sum_row_x;
      i_idx_minrow_x = dmt_row;
    }

    if (sum_row_y < sum_minrow_y)
    {
      sum_minrow_y = sum_row_y;
      i_idx_minrow_y = dmt_row;
    }

  }

  vOut[0].x = toMedian[i_idx_minrow_x].x;
  vOut[0].y = toMedian[i_idx_minrow_y].y;

//  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
      vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//    else
//      vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}

template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetModeVECTORxyda(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  // process dual coords in scalar C ?
  const int iMaxMVlength = std::max(nBlkX * nBlkSizeX, nBlkY * nBlkSizeY) * 2 * nPel; // hope it is enough ? todo: make global constant ?
  int MaxSumDM = iNumMVs * (iMaxMVlength + 3);

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow_x = MaxSumDM;
  int sum_minrow_y = MaxSumDM;
  int i_idx_minrow_x = 0;
  int i_idx_minrow_y = 0;

  for (int dmt_row = 0; dmt_row < iNumMVs; dmt_row++)
  {
    float sum_row_x = 0;
    float sum_row_y = 0;

    for (int dmt_col = 0; dmt_col < iNumMVs; dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      float fd = fDiffAngleVect(toMedian[dmt_row].x, toMedian[dmt_row].y, toMedian[dmt_col].x, toMedian[dmt_col].y);

      sum_row_x += ((float)std::abs(toMedian[dmt_row].x - toMedian[dmt_col].x) + fd);
      sum_row_y += ((float)std::abs(toMedian[dmt_row].y - toMedian[dmt_col].y) + fd);
    }

    if (sum_row_x < sum_minrow_x)
    {
      sum_minrow_x = sum_row_x;
      i_idx_minrow_x = dmt_row;
    }

    if (sum_row_y < sum_minrow_y)
    {
      sum_minrow_y = sum_row_y;
      i_idx_minrow_y = dmt_row;
    }

  }

  vOut[0].x = toMedian[i_idx_minrow_x].x;
  vOut[0].y = toMedian[i_idx_minrow_y].y;

  //  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
  vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//    else
//      vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}

template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetModeVECTORxydadm(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  // process dual coords in scalar C ?
  const int iMaxMVlength = std::max(nBlkX * nBlkSizeX, nBlkY * nBlkSizeY) * 2 * nPel; // hope it is enough ? todo: make global constant ?
  int MaxSumDM = iNumMVs * (iMaxMVlength + 3);

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow_x = MaxSumDM;
  int sum_minrow_y = MaxSumDM;
  int i_idx_minrow_x = 0;
  int i_idx_minrow_y = 0;

  for (int dmt_row = 0; dmt_row < iNumMVs; dmt_row++)
  {
    float sum_row_x = 0;
    float sum_row_y = 0;

    for (int dmt_col = 0; dmt_col < iNumMVs; dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      float fd = fDiffAngleVect(toMedian[dmt_row].x, toMedian[dmt_row].y, toMedian[dmt_col].x, toMedian[dmt_col].y);

      sum_row_x += ((float)std::abs(toMedian[dmt_row].x - toMedian[dmt_col].x) + fd + (float)toMedian[dmt_row].sad);
      sum_row_y += ((float)std::abs(toMedian[dmt_row].y - toMedian[dmt_col].y) + fd + (float)toMedian[dmt_row].sad);
    }

    if (sum_row_x < sum_minrow_x)
    {
      sum_minrow_x = sum_row_x;
      i_idx_minrow_x = dmt_row;
    }

    if (sum_row_y < sum_minrow_y)
    {
      sum_minrow_y = sum_row_y;
      i_idx_minrow_y = dmt_row;
    }

  }

  vOut[0].x = toMedian[i_idx_minrow_x].x;
  vOut[0].y = toMedian[i_idx_minrow_y].y;

  //  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
  vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//    else
//      vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}


template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetMeanVECTORxy(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  int sum_dx = 0;
  int sum_dy = 0;

  for (int i = 0; i < iNumMVs; i++)
  {
    sum_dx += toMedian[i].x;
    sum_dy += toMedian[i].y;
  }

  sum_dx = (sum_dx + (iNumMVs >> 1)) / iNumMVs;
  sum_dy = (sum_dy + (iNumMVs >> 1)) / iNumMVs;

  vOut[0].x = sum_dx;
  vOut[0].y = sum_dy;

//  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
    vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//  else
//    vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}

template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetModeVECTORvad(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  // process dual coords in scalar C ?
  const int iMaxAngDiff = 3; // hope it is enough ? 
  int MaxSumDM = iNumMVs * iMaxAngDiff;

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  float sum_minrow = (float)MaxSumDM;
  int i_idx_minrow = 0;

  for (int dmt_row = 0; dmt_row < iNumMVs; dmt_row++)
  {
    float sum_row = 0;

    for (int dmt_col = 0; dmt_col < iNumMVs; dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      sum_row += fDiffAngleVect(toMedian[dmt_row].x, toMedian[dmt_row].y, toMedian[dmt_col].x, toMedian[dmt_col].y);
    }

    if (sum_row < sum_minrow)
    {
      sum_minrow = sum_row;
      i_idx_minrow = dmt_row;
    }

  }

  vOut[0].x = toMedian[i_idx_minrow].x;
  vOut[0].y = toMedian[i_idx_minrow].y;

//  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
    vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//  else
//    vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}

template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetModeVECTORvld(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  // process dual coords in scalar C ?
  const int iMaxMVlength = std::max(nBlkX * nBlkSizeX, nBlkY * nBlkSizeY) * 2 * nPel; // hope it is enough ? todo: make global constant ?
  int MaxSumDM = iNumMVs * iMaxMVlength;

  // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
  int sum_minrow = MaxSumDM * MaxSumDM; // squared vects difference ?
  int i_idx_minrow = 0;

  for (int dmt_row = 0; dmt_row < iNumMVs; dmt_row++)
  {
    int sum_row = 0;

    for (int dmt_col = 0; dmt_col < iNumMVs; dmt_col++)
    {
      if (dmt_row == dmt_col)
      { // with itself => DM=0
        continue;
      }

      sum_row += (toMedian[dmt_row].x - toMedian[dmt_col].x) * (toMedian[dmt_row].x - toMedian[dmt_col].x) + (toMedian[dmt_row].y - toMedian[dmt_col].y) * (toMedian[dmt_row].y - toMedian[dmt_col].y);
    }

    if (sum_row < sum_minrow)
    {
      sum_minrow = sum_row;
      i_idx_minrow = dmt_row;
    }

  }

  vOut[0].x = toMedian[i_idx_minrow].x;
  vOut[0].y = toMedian[i_idx_minrow].y;

//  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
    vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//  else
//    vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV
}

template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetMedianVECTORg(VECTOR* toMedian, VECTOR* vOut, int iNumMVs) // geometric median
{
  const int test_steps_dx[] = { -1, 0, 1, 0 };
  const int test_steps_dy[] = { 0, 1, 0, -1 };

  VECTOR_XY vGMedian;
  int iMinDist = 0;

  // need to estimate max radius of vectors area ?
  int iStep = 4 * nPel; // 16 max for pel=4, need to be lower at high levels ?

  // first estimation - center of gravity
  int iMeanX = 0;
  int iMeanY = 0;
  for (int i = 0; i < iNumMVs; i++)
  {
    iMeanX += toMedian[i].x;
    iMeanY += toMedian[i].y;
  }

  vGMedian.x = (iMeanX + (iNumMVs >> 1)) / iNumMVs;
  vGMedian.y = (iMeanY + (iNumMVs >> 1)) / iNumMVs;

  // init iMinDist with first estimate
  for (int i = 0; i < iNumMVs; i++)
  {
    iMinDist += (toMedian[i].x - vGMedian.x) * (toMedian[i].x - vGMedian.x) + (toMedian[i].y - vGMedian.y) * (toMedian[i].y - vGMedian.y);
  }

  if (iMinDist > 0)
  {
    while (iStep > 0)
    {
      bool bDone = false;
      for (int i = 0; i < 4; ++i)
      {
        VECTOR_XY vToCheck;
        vToCheck.x = vGMedian.x + iStep * test_steps_dx[i];
        vToCheck.y = vGMedian.y + iStep * test_steps_dy[i];

        int iCheckedSum = 0;
        for (int j = 0; j < iNumMVs; j++)
        {
          iCheckedSum += (toMedian[j].x - vToCheck.x) * (toMedian[j].x - vToCheck.x) + (toMedian[j].y - vToCheck.y) * (toMedian[j].y - vToCheck.y);
        }

        if (iCheckedSum < iMinDist)
        {
          iMinDist = iCheckedSum;
          vGMedian = vToCheck;

          bDone = true;
          break;
        }
      }

      if (!bDone)
        iStep /= 2;
    }
  }// if iMinDist > 0

  vOut[0].x = vGMedian.x;
  vOut[0].y = vGMedian.y;

//  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
    vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//  else
//    vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}

template<typename pixel_t>
MV_FORCEINLINE void MAverage::Get_IQM_VECTORxy(VECTOR* toMedian, VECTOR* vOut, int iNumMVs)
{
  int vX[MAX_AREAMODE_STEPS];
  int vY[MAX_AREAMODE_STEPS];

  // copy to temp vectors
  for (int i = 0; i < iNumMVs; i++)
  {
    vX[i] = toMedian[i].x;
    vY[i] = toMedian[i].y;
  }

  // make ordering sort
  std::sort(vX, vX + (iNumMVs - 0));
  std::sort(vY, vY + (iNumMVs - 0));

  if (iNumMVs < 4) // 3 possible ?
  {
    vOut[0].x = vX[1];
    vOut[0].y = vY[1];
  }
  else
  {
    int qStart = (iNumMVs + 1) / 4; // do we want bias here ?
    int qEnd = iNumMVs - ((iNumMVs + 1) / 4);

    int iXmean = 0;
    int iYmean = 0;
    for (int i = qStart; i < qEnd; i++)
    {
      iXmean += vX[i];
      iYmean += vY[i];
    }
    int iBias = (qEnd - qStart) / 2;

    iXmean = (iXmean + iBias) / (qEnd - qStart);
    iYmean = (iYmean + iBias) / (qEnd - qStart);

    vOut[0].x = iXmean;
    vOut[0].y = iYmean;
  }

//  if ((vOut[0].x == toMedian[0].x) && (vOut[0].y == toMedian[0].y))
    vOut[0].sad = toMedian[0].sad; // MV already checked in inpit of AreaMode
//  else
//    vOut[0].sad = GetDM<pixel_t>(workarea, vOut[0].x, vOut[0].y); // update DM for current block pos and selected bestMV

}

MV_FORCEINLINE float fDiffAngleVect(int x1, int y1, int x2, int y2)
{
  float fResult = 0.0f;
  // check if any of 2 input vectors is zero vector - return 0
  if ((x1 == 0) && (y1 == 0) || (x2 == 0) && (y2 == 0))
    return 0.0f;

  int iUpper = x1 * x2 + y1 * y2;

  if (iUpper > 0)
  {
    fResult = 1.0f - ((float)(iUpper * iUpper) / (float((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2))));
  }
  else
  {
    fResult = 1.0f + ((float)(iUpper * iUpper) / (float((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2))));
  }

  return fResult;

}


template<typename pixel_t>
MV_FORCEINLINE void MAverage::GetLowestDM(VECTOR* toProc, VECTOR* vOut, int iNumMVs)
{
  vOut[0].sad = toProc[0].sad; // pick first for init
  vOut[0].x = toProc[0].x;
  vOut[0].y = toProc[0].y;

  for (int i = 1; i < iNumMVs; i++)
  {
    if (toProc[i].sad < vOut[0].sad)
    {
      vOut[0].sad = toProc[i].sad;
      vOut[0].x = toProc[i].x;
      vOut[0].y = toProc[i].y;
    }
  }
}

/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
