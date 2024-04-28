/*****************************************************************************
        MTransform.cpp
*Tab=3***********************************************************************/



#if defined (_MSC_VER)
//#pragma warning (1 : 4130 4223 4705 4706)
//#pragma warning (4 : 4355 4786 4800)
#endif



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include	"MTransform.h"
#include	<cassert>
#include  <algorithm>


/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



MTransform::MTransform(PClip clip, int _mode, IScriptEnvironment *env)
  : GenericVideoFilter(clip)
  , iMode(_mode)
{
  assert(&env != 0);

  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  const ::VideoInfo &vd_vi = clip->GetVideoInfo();

  // Checks basic parameters

  // Checks if the clip contains vectors
  if (vd_vi.nchannels >= 0 && vd_vi.nchannels < 9)
  {
    env->ThrowError("MTransform: invalid vector stream.");
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
    env->ThrowError("MTransform: invalid vector stream.");
  }

  // copy pointer to class for the first MV clip and init analysis data 
    mVectorsInfo = mad;
    nBlkX = mad.nBlkX;
    nBlkSizeX = mad.nBlkSizeX;
    nBlkY = mad.nBlkY;
    nBlkSizeY = mad.nBlkSizeY;
    nPel = mad.nPel;

    if ((iMode == 2) || (iMode == 3)) // TurnLeft or TurnRight
    {
      mad_dst.nWidth = mad.nHeight;
      mad_dst.nHeight = mad.nWidth;
      mad_dst.nOverlapX = mad.nOverlapY;
      mad_dst.nOverlapY = mad.nOverlapX;
      mad_dst.nBlkX = mad.nBlkY;
      mad_dst.nBlkY = mad.nBlkX;
      mad_dst.nHPadding = mad.nVPadding;
      mad_dst.nVPadding = mad.nHPadding;
      mad_dst.nMagicKey = mad.nMagicKey;
      mad_dst.nVersion = mad.nVersion;
      mad_dst.nBlkSizeX = mad.nBlkSizeX;
      mad_dst.nBlkSizeY = mad.nBlkSizeY;
      mad_dst.nPel = mad.nPel;
      mad_dst.nLvCount = mad.nLvCount;
      mad_dst.nDeltaFrame = mad.nDeltaFrame;
      mad_dst.isBackward = mad.isBackward;
      mad_dst.nFlags = mad.nFlags;
      mad_dst.pixelType = mad.pixelType;
      mad_dst.xRatioUV = mad.xRatioUV;
      mad_dst.yRatioUV = mad.yRatioUV;
      mad_dst.chromaSADScale = mad.chromaSADScale;
      mad_dst.pixelsize = mad.pixelsize;
      mad_dst.bits_per_pixel = mad.bits_per_pixel;
      mad_dst.nTrad = mad.nTrad;

      uintptr_t p = reinterpret_cast <uintptr_t> (&mad_dst);
      vi.nchannels = 0x80000000L | (int)(p >> 32);
      vi.sample_type = (int)(p & 0xffffffffUL);
    }

  // save ptrs to input MV clips
  m_clip = clip;

}



::PVideoFrame __stdcall	MTransform::GetFrame(int n, ::IScriptEnvironment *env_ptr)
{
  PVideoFrame src = child->GetFrame(n, env_ptr);
  PVideoFrame dst = has_at_least_v8 ? env_ptr->NewVideoFrameP(vi, &src) : env_ptr->NewVideoFrame(vi); // frame property support
  const int* pData = reinterpret_cast<const int*>(src->GetReadPtr());
  int* pDst = reinterpret_cast<int*>(dst->GetWritePtr());

  PVideoFrame src_frame;
  const int* pDataSrc;

  src_frame = m_clip->GetFrame(n, env_ptr);
  pDataSrc = reinterpret_cast<const int*>(src_frame->GetReadPtr());

  // Copy and fix header
  int headerSize = *pData;
  memcpy(pDst, pData, headerSize);

  const MVAnalysisData& hdr_src =
    *reinterpret_cast <const MVAnalysisData*> (pData + 1);
  MVAnalysisData& hdr_dst =
    *reinterpret_cast <MVAnalysisData*> (pDst + 1);

  // Copy delta and direction from the original frame (for multi-vector clips)
  // Fix the direction if required
  hdr_dst.nDeltaFrame = hdr_src.nDeltaFrame;
  hdr_dst.isBackward = hdr_src.isBackward;

  // if 90-degree rotate - swap all H and V values
  if ((iMode == 2) || (iMode == 3)) // TurnLeft or TurnRight
  {
    hdr_dst.nWidth = hdr_src.nHeight;
    hdr_dst.nHeight = hdr_src.nWidth;
    hdr_dst.nOverlapX = hdr_src.nOverlapY;
    hdr_dst.nOverlapY = hdr_src.nOverlapX;
    hdr_dst.nBlkX = hdr_src.nBlkY;
    hdr_dst.nBlkY = hdr_src.nBlkX;
    hdr_dst.nHPadding = hdr_src.nVPadding;
    hdr_dst.nVPadding = hdr_src.nHPadding;
  }

  // Copy all planes
  pData = reinterpret_cast<const int*>(reinterpret_cast<const char*>(pData) + headerSize);
  pDst = reinterpret_cast<int*>(reinterpret_cast<char*>(pDst) + headerSize);
  memcpy(pDst, pData, *pData * sizeof(int));

  // Size and validity of all block data
  int* pPlanes = pDst;
  if (pPlanes[1] == 0) return dst; // Marked invalid
  int* pEnd = pPlanes + *pPlanes;
  pPlanes += 2;

  int* pPlane = (int*)(reinterpret_cast<const char*>(pDataSrc) + headerSize);
  if (pPlane[1] == 0) return dst; // Marked invalid 
  int* pSrcPlanes = pPlane + 2;

  switch (iMode)
  {
  case 0:
    FlipHorizontal(pSrcPlanes, pPlanes, pEnd, env_ptr);
    break;

  case 1:
    FlipVertical(pSrcPlanes, pPlanes, pEnd, env_ptr);
    break;

  case 2:
    TurnLeft(pSrcPlanes, pPlanes, pEnd, env_ptr);
    break;

  case 3:
    TurnRight(pSrcPlanes, pPlanes, pEnd, env_ptr);
    break;

  }

  return dst;

}


/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

void MTransform::FlipHorizontal(int* pSrcPlanes, int* pDstPlanes, int* pEnd, ::IScriptEnvironment* env_ptr)
{
  // Dimensions of frame covered by blocks (where frame is not exactly divisible by block size there is a small border that will not be motion compensated)
  int widthCovered = (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX) * mVectorsInfo.nBlkX + mVectorsInfo.nOverlapX;
  int heightCovered = (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY) * mVectorsInfo.nBlkY + mVectorsInfo.nOverlapY;

  // Go through blocks at each level
  int level = mVectorsInfo.nLvCount - 1; // Start at coarsest level
  while (level >= 0)
  {
    int blocksSize = *pDstPlanes;//*pPlanes;
    VECTOR* pBlocks = reinterpret_cast<VECTOR*>(/*pPlanes*/pDstPlanes + 1);
    /*pPlanes*/ pDstPlanes += blocksSize;

    VECTOR* pSrcBlocks = reinterpret_cast<VECTOR*>(pSrcPlanes + 1);
    pSrcPlanes += blocksSize;


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
    int extendedWidth = levelWidth + 2 * mVectorsInfo.nHPadding; // Including padding
    int extendedHeight = levelHeight + 2 * mVectorsInfo.nVPadding;

    // Padding is effectively smaller on coarser levels
    int paddingXScaled = mVectorsInfo.nHPadding >> level;
    int paddingYScaled = mVectorsInfo.nVPadding >> level;

    // Loop through block positions (top-left of each block, coordinates relative to top-left of padding)
    int x = mVectorsInfo.nHPadding;
    int y = mVectorsInfo.nVPadding;
    int xEnd = x + levelNumBlocksX * (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX);
    int yEnd = y + levelNumBlocksY * (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY);

    while (y < yEnd)
    {
      // Max/min vector length for this block
      int yMin = -mVectorsInfo.nPel * (y - mVectorsInfo.nVPadding + paddingYScaled);
      int yMax = mVectorsInfo.nPel * (extendedHeight - y - mVectorsInfo.nBlkSizeY - mVectorsInfo.nVPadding + paddingYScaled);

      // start from end of Src row
      VECTOR* pSrcRowBlocks = pSrcBlocks + (levelNumBlocksX - 1);

      while (x < xEnd)
      {
        int xMin = -mVectorsInfo.nPel * (x - mVectorsInfo.nHPadding + paddingXScaled);
        int xMax = mVectorsInfo.nPel * (extendedWidth - x - mVectorsInfo.nBlkSizeX - mVectorsInfo.nHPadding + paddingXScaled);

        VECTOR vProc = *pSrcRowBlocks;
        VECTOR vOut;

        vOut.x = 0 - vProc.x;
        vOut.y = vProc.y;
        vOut.sad = vProc.sad;

        *pBlocks = vOut;

        pBlocks++;
        pSrcRowBlocks--;

        // Next block position
        x += mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX;
      }
      x = mVectorsInfo.nHPadding;
      y += mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY;
      pSrcBlocks += levelNumBlocksX;
    }
    if (reinterpret_cast<int*>(pBlocks) != /*pPlanes*/ pDstPlanes) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check
    level--;
  }
  if (/*pPlanes*/pDstPlanes != pEnd) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check

}

void MTransform::FlipVertical(int* pSrcPlanes, int* pDstPlanes, int* pEnd, ::IScriptEnvironment* env_ptr)
{
  // Dimensions of frame covered by blocks (where frame is not exactly divisible by block size there is a small border that will not be motion compensated)
  int widthCovered = (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX) * mVectorsInfo.nBlkX + mVectorsInfo.nOverlapX;
  int heightCovered = (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY) * mVectorsInfo.nBlkY + mVectorsInfo.nOverlapY;

  // Go through blocks at each level
  int level = mVectorsInfo.nLvCount - 1; // Start at coarsest level
  while (level >= 0)
  {
    int blocksSize = *pDstPlanes;//*pPlanes;
    VECTOR* pBlocks = reinterpret_cast<VECTOR*>(/*pPlanes*/pDstPlanes + 1);
    /*pPlanes*/ pDstPlanes += blocksSize;

    VECTOR* pSrcBlocks = reinterpret_cast<VECTOR*>(pSrcPlanes + 1);
    pSrcPlanes += blocksSize;


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
    int extendedWidth = levelWidth + 2 * mVectorsInfo.nHPadding; // Including padding
    int extendedHeight = levelHeight + 2 * mVectorsInfo.nVPadding;

    // Padding is effectively smaller on coarser levels
    int paddingXScaled = mVectorsInfo.nHPadding >> level;
    int paddingYScaled = mVectorsInfo.nVPadding >> level;

    // Loop through block positions (top-left of each block, coordinates relative to top-left of padding)
    int x = mVectorsInfo.nHPadding;
    int y = mVectorsInfo.nVPadding;
    int xEnd = x + levelNumBlocksX * (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX);
    int yEnd = y + levelNumBlocksY * (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY);


    while (y < yEnd)
    {
      // Max/min vector length for this block
      int yMin = -mVectorsInfo.nPel * (y - mVectorsInfo.nVPadding + paddingYScaled);
      int yMax = mVectorsInfo.nPel * (extendedHeight - y - mVectorsInfo.nBlkSizeY - mVectorsInfo.nVPadding + paddingYScaled);

      // start from last row of Src rows
      VECTOR* pSrcRowBlocks = pSrcBlocks + (levelNumBlocksY - 1) * levelNumBlocksX;

      while (x < xEnd)
      {
        int xMin = -mVectorsInfo.nPel * (x - mVectorsInfo.nHPadding + paddingXScaled);
        int xMax = mVectorsInfo.nPel * (extendedWidth - x - mVectorsInfo.nBlkSizeX - mVectorsInfo.nHPadding + paddingXScaled);

        VECTOR vProc = *pSrcRowBlocks;
        VECTOR vOut;

        vOut.x = vProc.x;
        vOut.y = 0 - vProc.y;
        vOut.sad = vProc.sad;

        *pBlocks = vOut;

        pBlocks++;
        pSrcRowBlocks++;

        // Next block position
        x += mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX;
      }
      x = mVectorsInfo.nHPadding;
      y += mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY;
      pSrcBlocks -= levelNumBlocksX;
    }
    if (reinterpret_cast<int*>(pBlocks) != /*pPlanes*/ pDstPlanes) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check
    level--;
  }
  if (/*pPlanes*/pDstPlanes != pEnd) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check

}

void MTransform::TurnLeft(int* pSrcPlanes, int* pDstPlanes, int* pEnd, ::IScriptEnvironment* env_ptr)
{
  // Dimensions of frame covered by blocks (where frame is not exactly divisible by block size there is a small border that will not be motion compensated)
  int widthCovered = (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX) * mVectorsInfo.nBlkX + mVectorsInfo.nOverlapX;
  int heightCovered = (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY) * mVectorsInfo.nBlkY + mVectorsInfo.nOverlapY;

  // Go through blocks at each level
  int level = mVectorsInfo.nLvCount - 1; // Start at coarsest level
  while (level >= 0)
  {
    int blocksSize = *pDstPlanes;//*pPlanes;
    VECTOR* pBlocks = reinterpret_cast<VECTOR*>(/*pPlanes*/pDstPlanes + 1);
    /*pPlanes*/ pDstPlanes += blocksSize;

    VECTOR* pSrcBlocks = reinterpret_cast<VECTOR*>(pSrcPlanes + 1);
    pSrcPlanes += blocksSize;

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
    int extendedWidth = levelWidth + 2 * mVectorsInfo.nHPadding; // Including padding
    int extendedHeight = levelHeight + 2 * mVectorsInfo.nVPadding;

    // Padding is effectively smaller on coarser levels
    int paddingXScaled = mVectorsInfo.nHPadding >> level;
    int paddingYScaled = mVectorsInfo.nVPadding >> level;

    // Loop through block positions (top-left of each block, coordinates relative to top-left of padding)
    int x = mVectorsInfo.nHPadding;
    int y = mVectorsInfo.nVPadding;
    int xEnd = x + levelNumBlocksX * (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX);
    int yEnd = y + levelNumBlocksY * (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY);

    while (x < xEnd)
    {
      // Max/min vector length for this block
      int yMin = -mVectorsInfo.nPel * (y - mVectorsInfo.nVPadding + paddingYScaled);
      int yMax = mVectorsInfo.nPel * (extendedHeight - y - mVectorsInfo.nBlkSizeY - mVectorsInfo.nVPadding + paddingYScaled);

      // start from last column of Src rows
      VECTOR* pSrcRowBlocks = pSrcBlocks + (levelNumBlocksX - 1);

      while (y < yEnd)
      {
        int xMin = -mVectorsInfo.nPel * (x - mVectorsInfo.nHPadding + paddingXScaled);
        int xMax = mVectorsInfo.nPel * (extendedWidth - x - mVectorsInfo.nBlkSizeX - mVectorsInfo.nHPadding + paddingXScaled);

        VECTOR vProc = *pSrcRowBlocks;
        VECTOR vOut;
      
        vOut.x = vProc.y;
        vOut.y = 0 - vProc.x;
        vOut.sad = vProc.sad;

        *pBlocks = vOut;

        pBlocks++;

        pSrcRowBlocks += levelNumBlocksX;

        // Next block position
        y += mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY;
      }
      y = mVectorsInfo.nVPadding;
      x += mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX;
      pSrcBlocks--;
    }
    if (reinterpret_cast<int*>(pBlocks) != /*pPlanes*/ pDstPlanes) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check
    level--;
  }
  if (/*pPlanes*/pDstPlanes != pEnd) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check

}

void MTransform::TurnRight(int* pSrcPlanes, int* pDstPlanes, int* pEnd, ::IScriptEnvironment* env_ptr)
{
  // Dimensions of frame covered by blocks (where frame is not exactly divisible by block size there is a small border that will not be motion compensated)
  int widthCovered = (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX) * mVectorsInfo.nBlkX + mVectorsInfo.nOverlapX;
  int heightCovered = (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY) * mVectorsInfo.nBlkY + mVectorsInfo.nOverlapY;

  // Go through blocks at each level
  int level = mVectorsInfo.nLvCount - 1; // Start at coarsest level
  while (level >= 0)
  {
    int blocksSize = *pDstPlanes;//*pPlanes;
    VECTOR* pBlocks = reinterpret_cast<VECTOR*>(/*pPlanes*/pDstPlanes + 1);
    /*pPlanes*/ pDstPlanes += blocksSize;

    VECTOR* pSrcBlocks = reinterpret_cast<VECTOR*>(pSrcPlanes + 1);
    pSrcPlanes += blocksSize;

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
    int extendedWidth = levelWidth + 2 * mVectorsInfo.nHPadding; // Including padding
    int extendedHeight = levelHeight + 2 * mVectorsInfo.nVPadding;

    // Padding is effectively smaller on coarser levels
    int paddingXScaled = mVectorsInfo.nHPadding >> level;
    int paddingYScaled = mVectorsInfo.nVPadding >> level;

    // Loop through block positions (top-left of each block, coordinates relative to top-left of padding)
    int x = mVectorsInfo.nHPadding;
    int y = mVectorsInfo.nVPadding;
    int xEnd = x + levelNumBlocksX * (mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX);
    int yEnd = y + levelNumBlocksY * (mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY);

    while (x < xEnd)
    {
      // Max/min vector length for this block
      int yMin = -mVectorsInfo.nPel * (y - mVectorsInfo.nVPadding + paddingYScaled);
      int yMax = mVectorsInfo.nPel * (extendedHeight - y - mVectorsInfo.nBlkSizeY - mVectorsInfo.nVPadding + paddingYScaled);

      // start from first column last row of Src 
      VECTOR* pSrcRowBlocks = pSrcBlocks + (levelNumBlocksY - 1) * levelNumBlocksX;

      while (y < yEnd)
      {
        int xMin = -mVectorsInfo.nPel * (x - mVectorsInfo.nHPadding + paddingXScaled);
        int xMax = mVectorsInfo.nPel * (extendedWidth - x - mVectorsInfo.nBlkSizeX - mVectorsInfo.nHPadding + paddingXScaled);

        VECTOR vProc = *pSrcRowBlocks;
        VECTOR vOut;

        vOut.x = 0 - vProc.y;
        vOut.y = vProc.x;
        vOut.sad = vProc.sad;

        *pBlocks = vOut;

        pBlocks++;

        pSrcRowBlocks -= levelNumBlocksX;

        // Next block position
        y += mVectorsInfo.nBlkSizeY - mVectorsInfo.nOverlapY;
      }
      y = mVectorsInfo.nVPadding;
      x += mVectorsInfo.nBlkSizeX - mVectorsInfo.nOverlapX;
      pSrcBlocks++;
    }
    if (reinterpret_cast<int*>(pBlocks) != /*pPlanes*/ pDstPlanes) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check
    level--;
  }
  if (/*pPlanes*/pDstPlanes != pEnd) env_ptr->ThrowError("MTransform: Internal error"); // Debugging check

}


