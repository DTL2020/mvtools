
#define RS\
[\
    RootSignature\
    (\
       "CBV(b0, visibility=SHADER_VISIBILITY_ALL),\
        DescriptorTable(Sampler(s0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL),\
        DescriptorTable(SRV(t0, numDescriptors=5), visibility=SHADER_VISIBILITY_ALL),\
        DescriptorTable(UAV(u0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL)"\
    )\
]


RWTexture2D<int>OutputTexture : register(u0);
Texture2D<int>CurrentTexture_Y : register(t0);
Texture2D<int2>CurrentTexture_UV : register(t1);
Texture2D<int>ReferenceTexture_Y : register(t2);
Texture2D<int2>ReferenceTexture_UV : register(t3);
Texture2D<int2>ResolvedMVsTexture : register(t4);

SamplerState Sampler : register(s0);

cbuffer cb0
{
	int g_BlockSizeX : packoffset(c0.x);
	int g_BlockSizeY : packoffset(c0.y);
	int g_UseChroma : packoffset(c0.z);
	int g_precisionMVs : packoffset(c0.w);
	int g_chromaSADscale : packoffset(c1.x);
	float g_chromaSADscale_fine : packoffset(c1.y);
	int g_nPel : packoffset(c1.z);
  int g_KernelSize : packoffset(c1.w);
	float4 g_KernelShift_01_0 : packoffset(c2); // low 4 of kernel for 01 sub shift
  float4 g_KernelShift_01_1 : packoffset(c3); // high 4 of kernel for 01 sub shift

  float4 g_KernelShift_10_0 : packoffset(c4); // low 4 of kernel for 10 sub shift
  float4 g_KernelShift_10_1 : packoffset(c5); // high 4 of kernel for 10 sub shift

  float4 g_KernelShift_11_0 : packoffset(c6); // low 4 of kernel for 11 sub shift
  float4 g_KernelShift_11_1 : packoffset(c7); // high 4 of kernel for 11 sub shift

}

[numthreads(4, 4, 1)] // make sure to update value s_numShaderThreads in MAnalyse.cpp if this changes  
RS
void main(uint3 DTid : SV_DispatchThreadID)
{
//  int iKS = g_KernelSize;
	int iKS_d2 = g_KernelSize >> 1; // half
	int x;
	int y;

  // temp buf in registers, numshader threads are reduced to keep temp buf of threads group (4x4) in 16384 size (recommended) ?
	float2 fTempShiftedBlockH[(16) * (16 + 8)];// + iKS, use 16x16 max blocks ?  H shifted
  float2 fTempShiftedBlockHV[(16) * (16)];// use 16x16 max blocks ?  HV shifted

  float fKernelShift[8];

	int3 i3Coord;
		
	i3Coord.x = DTid.x;
	i3Coord.y = DTid.y;
	i3Coord.z = 0;

	int2 i2MV = ResolvedMVsTexture.Load(i3Coord);
  int2 i2MV_fp; // integer part of qpel shift

  i2MV_fp.r = i2MV.r >> 2; 
  i2MV_fp.g = i2MV.g >> 2;

	int iYsrc;
	int iYref;
	int2 iUVsrc;
	int2 iUVref;

  float fYref = 0.0f;
  float fSAD = 0.0f; // for pel > 1 sum
  float fChromaSAD = 0.0f; // for pel > 1 sum
  float2 fUVref;

	int iSAD = 0;
	int iChromaSAD = 0;
	int iChromaSADAdd = 0;

  int iBS_X_d2 = g_BlockSizeX >> 1; // chroma block size YV12 X
	int iBS_Y_d2 = g_BlockSizeY >> 1; // chroma block size YV12 Y

  switch (g_nPel)
  {
  case 1:

    i2MV.r = i2MV.r >> g_precisionMVs;// 2 = full frame search qpel/4, 1 = half frame search qpel/2
    i2MV.g = i2MV.g >> g_precisionMVs;

    for (y = 0; y < g_BlockSizeY; y++)
    {
      for (x = 0; x < g_BlockSizeX; x++)
      {
        i3Coord.x = DTid.x * g_BlockSizeX + x;
        i3Coord.y = DTid.y * g_BlockSizeY + y;
        iYsrc = CurrentTexture_Y.Load(i3Coord).r;

        i3Coord.x = DTid.x * g_BlockSizeX + x + i2MV.r;
        i3Coord.y = DTid.y * g_BlockSizeY + y + i2MV.g;

        iYref = ReferenceTexture_Y.Load(i3Coord).r;

        iSAD += abs(iYsrc - iYref);
      }
    }

    if (g_UseChroma != 0) // chroma SAD
    {
      for (y = 0; y < iBS_Y_d2; y++)
      {
        for (x = 0; x < iBS_X_d2; x++)
        {
          i3Coord.x = DTid.x * iBS_X_d2 + x;
          i3Coord.y = DTid.y * iBS_Y_d2 + y;
          iUVsrc = CurrentTexture_UV.Load(i3Coord);

          i3Coord.x = DTid.x * iBS_X_d2 + x + (i2MV.r >> 1);
          i3Coord.y = DTid.y * iBS_Y_d2 + y + (i2MV.g >> 1);

          iUVref = ReferenceTexture_UV.Load(i3Coord);

          iChromaSAD += (abs(iUVsrc.r - iUVref.r) + abs(iUVsrc.g - iUVref.g));
        }
      }

      if (g_chromaSADscale > 0)
      {
        iChromaSADAdd = iChromaSAD >> g_chromaSADscale;
      }
      else if (g_chromaSADscale < 0)
      {
        iChromaSADAdd = iChromaSAD << (-g_chromaSADscale);
      }
      else
        iChromaSADAdd = iChromaSAD;

      iChromaSADAdd = (float)iChromaSADAdd * g_chromaSADscale_fine;

      iSAD += iChromaSADAdd;
    }

    OutputTexture[DTid.xy] = iSAD;
    break; // nPel = 1

  case 2:

    // if any half shift required - fill kernel
    if ((((i2MV.r >> 1) & 1) == 1) || (((i2MV.g >> 1) & 1) == 1))
    {
      // fill kernel vector with half pel shift kernel
      fKernelShift[0] = g_KernelShift_10_0.x;
      fKernelShift[1] = g_KernelShift_10_0.y;
      fKernelShift[2] = g_KernelShift_10_0.z;
      fKernelShift[3] = g_KernelShift_10_0.w;
      fKernelShift[4] = g_KernelShift_10_1.x;
      fKernelShift[5] = g_KernelShift_10_1.y;
      fKernelShift[6] = g_KernelShift_10_1.z;
      fKernelShift[7] = g_KernelShift_10_1.w;
    }

    // check if half shifts required
    if (((i2MV.r >> 1) & 1) == 1) // h half sub != 0
    {
      // shift ref block to TempShiftedBlockH, Y plane
      // hor shift
      for (int j = 0; j < (g_BlockSizeY + g_KernelSize); j++)
      {
        for (int i = 0; i < g_BlockSizeX; i++)
        {
          float fOut = 0.0f;

          for (int k = 0; k < g_KernelSize; k++)
          {
            i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r - iKS_d2 + i + k;
            i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + j - iKS_d2;
            iYref = ReferenceTexture_Y.Load(i3Coord).r;
            fOut += (float)iYref * fKernelShift[k];
          }

          fTempShiftedBlockH[j * g_BlockSizeX + i].r = fOut; 
        }
      }

    }
    else
    {
      // copy ref to temp buf
      for (int y_sh = 0; y_sh < (g_BlockSizeY + g_KernelSize); y_sh++)
      {
        for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
        {
          i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh;
          i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh - iKS_d2;
          fTempShiftedBlockH[(y_sh * g_BlockSizeX) + x_sh].r = (float)ReferenceTexture_Y.Load(i3Coord).r;
        }
      }

    }

    if (((i2MV.g >> 1) & 1) == 1) // v half sub != 0
    {
      // V shift ref block from fTempShiftedBlockH to fTempShiftedBlockHV, Y plane
      for (int i = 0; i < g_BlockSizeX; i++)
      {
        for (int j = 0; j < g_BlockSizeY; j++)
        {
          float fOut = 0.0f;

          for (int k = 0; k < g_KernelSize; k++)
          {
            float fSample = fTempShiftedBlockH[(j + k) * g_BlockSizeX + i].r;
            fOut += fSample * fKernelShift[k];
          }

          fTempShiftedBlockHV[j * g_BlockSizeX + i].r = fOut;
        }
      }
 
    }
    else // no V shift required, copy H to HV  
    {
      for (int y_sh = 0; y_sh < g_BlockSizeY; y_sh++) 
      {
        for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
        {
          float fOut = fTempShiftedBlockH[(y_sh + iKS_d2) * g_BlockSizeX + x_sh].r;
          fTempShiftedBlockHV[y_sh * g_BlockSizeX + x_sh].r = fOut;
        }
      }

    }

    // calc SAD Y

    for (y = 0; y < g_BlockSizeY; y++)
    {
      for (x = 0; x < g_BlockSizeX; x++)
      {
        i3Coord.x = DTid.x * g_BlockSizeX + x;
        i3Coord.y = DTid.y * g_BlockSizeY + y;
        iYsrc = CurrentTexture_Y.Load(i3Coord).r;

        fYref = fTempShiftedBlockHV[y * g_BlockSizeX + x].r;

        fSAD += abs((float)iYsrc - iYref);
      }
    }

    iSAD = (int)fSAD;
    
    if (g_UseChroma != 0) // chroma SAD
    {

      // check if half shifts required
      if (((i2MV.r >> 1) & 1) == 1) // h half sub != 0
      {
        // shift ref block to TempShiftedBlockH, UV plane
        // hor shift
        for (int j = 0; j < (iBS_Y_d2 + g_KernelSize); j++)
        {
          for (int i = 0; i < iBS_X_d2; i++)
          {
            float fOutU = 0.0f;
            float fOutV = 0.0f;

            for (int k = 0; k < g_KernelSize; k++)
            {
              i3Coord.x = DTid.x * iBS_X_d2 + i2MV_fp.r - iKS_d2 + i + k;
              i3Coord.y = DTid.y * iBS_Y_d2 + i2MV_fp.g + j - iKS_d2;
              iUVref = ReferenceTexture_UV.Load(i3Coord);

              //            float fSample = (float)CurrBlock[j * nSrcPitch[0] + i + k];
              fOutU += (float)iUVref.r * fKernelShift[k];
              fOutV += (float)iUVref.g * fKernelShift[k];
            }

            fTempShiftedBlockH[j * iBS_X_d2 + i].r = fOutU;
            fTempShiftedBlockH[j * iBS_X_d2 + i].g = fOutV;
          }
        }

      }
      else
      {
        // copy ref to temp buf
        for (int y_sh = 0; y_sh < (iBS_Y_d2 + g_KernelSize); y_sh++)
        {
          for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
          {
            i3Coord.x = DTid.x * iBS_X_d2 + i2MV_fp.r + x_sh;
            i3Coord.y = DTid.y * iBS_Y_d2 + i2MV_fp.g + y_sh - iKS_d2;
            iUVref = ReferenceTexture_UV.Load(i3Coord);
            fTempShiftedBlockH[(y_sh * iBS_X_d2) + x_sh].r = (float)iUVref.r;
            fTempShiftedBlockH[(y_sh * iBS_Y_d2) + x_sh].g = (float)iUVref.g;
          }
        }


      }

      if (((i2MV.g >> 1) & 1) == 1) // v half sub != 0
      {

        // V shift ref block from TempShiftedBlockY to iTempShiftedBlockY2, Y plane
        for (int i = 0; i < iBS_X_d2; i++)
        {
          for (int j = 0; j < iBS_Y_d2; j++)
          {
            float fOutU = 0.0f;
            float fOutV = 0.0f;

            for (int k = 0; k < g_KernelSize; k++)
            {
              float fSampleU = fTempShiftedBlockH[(j + k) * iBS_X_d2 + i].r;
              float fSampleV = fTempShiftedBlockH[(j + k) * iBS_X_d2 + i].g;
              fOutU += fSampleU * fKernelShift[k];
              fOutV += fSampleV * fKernelShift[k];
            }

            fTempShiftedBlockHV[j * iBS_X_d2 + i].r = fOutU;
            fTempShiftedBlockHV[j * iBS_X_d2 + i].g = fOutV;
          }
        }

      }
      else // no V shift required, copy H to HV 
      {
        for (int y_sh = 0; y_sh < iBS_Y_d2; y_sh++)
        {
          for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
          {
            float fOutU = fTempShiftedBlockH[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].r;
            float fOutV = fTempShiftedBlockH[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].g;

            fTempShiftedBlockHV[y_sh * iBS_X_d2 + x_sh].r = fOutU;
            fTempShiftedBlockHV[y_sh * iBS_X_d2 + x_sh].g = fOutV;
          }
        }

      }

      for (y = 0; y < iBS_Y_d2; y++)
      {
        for (x = 0; x < iBS_X_d2; x++)
        {
          i3Coord.x = DTid.x * iBS_X_d2 + x;
          i3Coord.y = DTid.y * iBS_Y_d2 + y;
          iUVsrc = CurrentTexture_UV.Load(i3Coord);

          fUVref = fTempShiftedBlockHV[y * iBS_X_d2 + x]; 

          fChromaSAD += (abs((float)iUVsrc.r - iUVref.r) + abs((float)iUVsrc.g - iUVref.g));
        }
      }

      iChromaSAD = (int)fChromaSAD;
      
      if (g_chromaSADscale > 0)
      {
        iChromaSADAdd = iChromaSAD >> g_chromaSADscale;
      }
      else if (g_chromaSADscale < 0)
      {
        iChromaSADAdd = iChromaSAD << (-g_chromaSADscale);
      }
      else
        iChromaSADAdd = iChromaSAD;

      iChromaSADAdd = (float)iChromaSADAdd * g_chromaSADscale_fine;

      iSAD += iChromaSADAdd;
    }
    
    OutputTexture[DTid.xy] = iSAD;
    break; // nPel=2

  case 4: // nPel = 4

    // check if sub shifts required
    if ((i2MV.r & 3) != 0 ) // h sub != 0
    {
      //load required shift kernel
      switch (i2MV.r & 3)
      {
      case 1:
        fKernelShift[0] = g_KernelShift_01_0.x;
        fKernelShift[1] = g_KernelShift_01_0.y;
        fKernelShift[2] = g_KernelShift_01_0.z;
        fKernelShift[3] = g_KernelShift_01_0.w;
        fKernelShift[4] = g_KernelShift_01_1.x;
        fKernelShift[5] = g_KernelShift_01_1.y;
        fKernelShift[6] = g_KernelShift_01_1.z;
        fKernelShift[7] = g_KernelShift_01_1.w;
        break;
      case 2:
        fKernelShift[0] = g_KernelShift_10_0.x;
        fKernelShift[1] = g_KernelShift_10_0.y;
        fKernelShift[2] = g_KernelShift_10_0.z;
        fKernelShift[3] = g_KernelShift_10_0.w;
        fKernelShift[4] = g_KernelShift_10_1.x;
        fKernelShift[5] = g_KernelShift_10_1.y;
        fKernelShift[6] = g_KernelShift_10_1.z;
        fKernelShift[7] = g_KernelShift_10_1.w;
        break;
      case 3:
        fKernelShift[0] = g_KernelShift_11_0.x;
        fKernelShift[1] = g_KernelShift_11_0.y;
        fKernelShift[2] = g_KernelShift_11_0.z;
        fKernelShift[3] = g_KernelShift_11_0.w;
        fKernelShift[4] = g_KernelShift_11_1.x;
        fKernelShift[5] = g_KernelShift_11_1.y;
        fKernelShift[6] = g_KernelShift_11_1.z;
        fKernelShift[7] = g_KernelShift_11_1.w;
        break;
      }

      // shift ref block to TempShiftedBlockH, Y plane
      // hor shift
      for (int j = 0; j < (g_BlockSizeY + g_KernelSize); j++)
      {
        for (int i = 0; i < g_BlockSizeX; i++)
        {
          float fOut = 0.0f;

          for (int k = 0; k < g_KernelSize; k++)
          {
            i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r - iKS_d2 + i + k;
            i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + j - iKS_d2;
            iYref = ReferenceTexture_Y.Load(i3Coord).r;
            fOut += (float)iYref * fKernelShift[k];
          }

          fTempShiftedBlockH[j * g_BlockSizeX + i].r = fOut;
        }
      }

    }
    else
    {
      // copy ref to temp buf
      for (int y_sh = 0; y_sh < (g_BlockSizeY + g_KernelSize); y_sh++)
      {
        for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
        {
          i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh;
          i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh - iKS_d2;
          fTempShiftedBlockH[(y_sh * g_BlockSizeX) + x_sh].r = (float)ReferenceTexture_Y.Load(i3Coord).r;
        }
      }

    }

    if ((i2MV.g & 3) != 0) // v sub != 0
    {
      switch (i2MV.g & 3)
      {
      case 1:
        fKernelShift[0] = g_KernelShift_01_0.x;
        fKernelShift[1] = g_KernelShift_01_0.y;
        fKernelShift[2] = g_KernelShift_01_0.z;
        fKernelShift[3] = g_KernelShift_01_0.w;
        fKernelShift[4] = g_KernelShift_01_1.x;
        fKernelShift[5] = g_KernelShift_01_1.y;
        fKernelShift[6] = g_KernelShift_01_1.z;
        fKernelShift[7] = g_KernelShift_01_1.w;
        break;
      case 2:
        fKernelShift[0] = g_KernelShift_10_0.x;
        fKernelShift[1] = g_KernelShift_10_0.y;
        fKernelShift[2] = g_KernelShift_10_0.z;
        fKernelShift[3] = g_KernelShift_10_0.w;
        fKernelShift[4] = g_KernelShift_10_1.x;
        fKernelShift[5] = g_KernelShift_10_1.y;
        fKernelShift[6] = g_KernelShift_10_1.z;
        fKernelShift[7] = g_KernelShift_10_1.w;
        break;
      case 3:
        fKernelShift[0] = g_KernelShift_11_0.x;
        fKernelShift[1] = g_KernelShift_11_0.y;
        fKernelShift[2] = g_KernelShift_11_0.z;
        fKernelShift[3] = g_KernelShift_11_0.w;
        fKernelShift[4] = g_KernelShift_11_1.x;
        fKernelShift[5] = g_KernelShift_11_1.y;
        fKernelShift[6] = g_KernelShift_11_1.z;
        fKernelShift[7] = g_KernelShift_11_1.w;
        break;
      }

      // V shift ref block from TempShiftedBlockY to iTempShiftedBlockY2, Y plane
      for (int i = 0; i < g_BlockSizeX; i++)
      {
        for (int j = 0; j < g_BlockSizeY; j++)
        {
          float fOut = 0.0f;

          for (int k = 0; k < g_KernelSize; k++)
          {
            float fSample = fTempShiftedBlockH[(j + k) * g_BlockSizeX + i].r;
            fOut += fSample * fKernelShift[k];
          }

          fTempShiftedBlockHV[j * g_BlockSizeX + i].r = fOut;
        }
      }

    }
    else // no V shift required, copy H to HV  
    {
      for (int y_sh = 0; y_sh < g_BlockSizeY; y_sh++)
      {
        for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
        {
          float fOut = fTempShiftedBlockH[(y_sh + iKS_d2) * g_BlockSizeX + x_sh].r;
          fTempShiftedBlockHV[y_sh * g_BlockSizeX + x_sh].r = fOut;
        }
      }

    }

    // calc SAD Y
    for (y = 0; y < g_BlockSizeY; y++)
    {
      for (x = 0; x < g_BlockSizeX; x++)
      {
        i3Coord.x = DTid.x * g_BlockSizeX + x;
        i3Coord.y = DTid.y * g_BlockSizeY + y;
        iYsrc = CurrentTexture_Y.Load(i3Coord).r;

        fYref = fTempShiftedBlockHV[y * g_BlockSizeX + x].r;

        fSAD += abs((float)iYsrc - fYref);
      }
    }

    iSAD = fSAD;

    if (g_UseChroma != 0) // chroma SAD
    {
      // check if half shifts required
      if ((i2MV.r & 3) != 0) // h sub != 0
      {
        //load required shift kernel
        switch (i2MV.r & 3)
        {
        case 1:
          fKernelShift[0] = g_KernelShift_01_0.x;
          fKernelShift[1] = g_KernelShift_01_0.y;
          fKernelShift[2] = g_KernelShift_01_0.z;
          fKernelShift[3] = g_KernelShift_01_0.w;
          fKernelShift[4] = g_KernelShift_01_1.x;
          fKernelShift[5] = g_KernelShift_01_1.y;
          fKernelShift[6] = g_KernelShift_01_1.z;
          fKernelShift[7] = g_KernelShift_01_1.w;
          break;
        case 2:
          fKernelShift[0] = g_KernelShift_10_0.x;
          fKernelShift[1] = g_KernelShift_10_0.y;
          fKernelShift[2] = g_KernelShift_10_0.z;
          fKernelShift[3] = g_KernelShift_10_0.w;
          fKernelShift[4] = g_KernelShift_10_1.x;
          fKernelShift[5] = g_KernelShift_10_1.y;
          fKernelShift[6] = g_KernelShift_10_1.z;
          fKernelShift[7] = g_KernelShift_10_1.w;
          break;
        case 3:
          fKernelShift[0] = g_KernelShift_11_0.x;
          fKernelShift[1] = g_KernelShift_11_0.y;
          fKernelShift[2] = g_KernelShift_11_0.z;
          fKernelShift[3] = g_KernelShift_11_0.w;
          fKernelShift[4] = g_KernelShift_11_1.x;
          fKernelShift[5] = g_KernelShift_11_1.y;
          fKernelShift[6] = g_KernelShift_11_1.z;
          fKernelShift[7] = g_KernelShift_11_1.w;
          break;
        }

        // shift ref block to TempShiftedBlockH, UV plane
        // hor shift
        for (int j = 0; j < (iBS_Y_d2 + g_KernelSize); j++)
        {
          for (int i = 0; i < iBS_X_d2; i++)
          {
            float fOutU = 0.0f;
            float fOutV = 0.0f;

            for (int k = 0; k < g_KernelSize; k++)
            {
              i3Coord.x = DTid.x * iBS_X_d2 + i2MV_fp.r - iKS_d2 + i + k;
              i3Coord.y = DTid.y * iBS_Y_d2 + i2MV_fp.g + j - iKS_d2;
              iUVref = ReferenceTexture_UV.Load(i3Coord);

              //            float fSample = (float)CurrBlock[j * nSrcPitch[0] + i + k];
              fOutU += (float)iUVref.r * fKernelShift[k];
              fOutV += (float)iUVref.g * fKernelShift[k];
            }

            fTempShiftedBlockH[j * iBS_X_d2 + i].r = fOutU;
            fTempShiftedBlockH[j * iBS_X_d2 + i].g = fOutV;
          }
        }

      }
      else
      {
        // copy ref to temp buf
        for (int y_sh = 0; y_sh < (iBS_Y_d2 + g_KernelSize); y_sh++)
        {
          for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
          {
            i3Coord.x = DTid.x * iBS_X_d2 + i2MV_fp.r + x_sh;
            i3Coord.y = DTid.y * iBS_Y_d2 + i2MV_fp.g + y_sh - iKS_d2;
            iUVref = ReferenceTexture_UV.Load(i3Coord);
            fTempShiftedBlockH[(y_sh * iBS_X_d2) + x_sh].r = (float)iUVref.r;
            fTempShiftedBlockH[(y_sh * iBS_Y_d2) + x_sh].g = (float)iUVref.g;
          }
        }


      }

      if ((i2MV.g & 3) != 0) // v sub != 0
      {
        //load required shift kernel
        switch (i2MV.g & 3)
        {
        case 1:
          fKernelShift[0] = g_KernelShift_01_0.x;
          fKernelShift[1] = g_KernelShift_01_0.y;
          fKernelShift[2] = g_KernelShift_01_0.z;
          fKernelShift[3] = g_KernelShift_01_0.w;
          fKernelShift[4] = g_KernelShift_01_1.x;
          fKernelShift[5] = g_KernelShift_01_1.y;
          fKernelShift[6] = g_KernelShift_01_1.z;
          fKernelShift[7] = g_KernelShift_01_1.w;
          break;
        case 2:
          fKernelShift[0] = g_KernelShift_10_0.x;
          fKernelShift[1] = g_KernelShift_10_0.y;
          fKernelShift[2] = g_KernelShift_10_0.z;
          fKernelShift[3] = g_KernelShift_10_0.w;
          fKernelShift[4] = g_KernelShift_10_1.x;
          fKernelShift[5] = g_KernelShift_10_1.y;
          fKernelShift[6] = g_KernelShift_10_1.z;
          fKernelShift[7] = g_KernelShift_10_1.w;
          break;
        case 3:
          fKernelShift[0] = g_KernelShift_11_0.x;
          fKernelShift[1] = g_KernelShift_11_0.y;
          fKernelShift[2] = g_KernelShift_11_0.z;
          fKernelShift[3] = g_KernelShift_11_0.w;
          fKernelShift[4] = g_KernelShift_11_1.x;
          fKernelShift[5] = g_KernelShift_11_1.y;
          fKernelShift[6] = g_KernelShift_11_1.z;
          fKernelShift[7] = g_KernelShift_11_1.w;
          break;
        }

        // V shift ref block from fTempShiftedBlockH to iTempShiftedBlockHV, Y plane
        for (int i = 0; i < iBS_X_d2; i++)
        {
          for (int j = 0; j < iBS_Y_d2; j++)
          {
            float fOutU = 0.0f;
            float fOutV = 0.0f;

            for (int k = 0; k < g_KernelSize; k++)
            {
              float fSampleU = fTempShiftedBlockH[(j + k) * iBS_X_d2 + i].r;
              float fSampleV = fTempShiftedBlockH[(j + k) * iBS_X_d2 + i].g;
              fOutU += fSampleU * fKernelShift[k];
              fOutV += fSampleV * fKernelShift[k];
            }

            fTempShiftedBlockHV[j * iBS_X_d2 + i].r = fOutU;
            fTempShiftedBlockHV[j * iBS_X_d2 + i].g = fOutV;
          }
        }

      }
      else // no V shift required, copy H to HV 
      {
        for (int y_sh = 0; y_sh < iBS_Y_d2; y_sh++)
        {
          for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
          {
            float fOutU = fTempShiftedBlockH[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].r;
            float fOutV = fTempShiftedBlockH[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].g;

            fTempShiftedBlockHV[y_sh * iBS_X_d2 + x_sh].r = fOutU;
            fTempShiftedBlockHV[y_sh * iBS_X_d2 + x_sh].g = fOutV;
          }
        }

      }

      for (y = 0; y < iBS_Y_d2; y++)
      {
        for (x = 0; x < iBS_X_d2; x++)
        {
          i3Coord.x = DTid.x * iBS_X_d2 + x;
          i3Coord.y = DTid.y * iBS_Y_d2 + y;
          iUVsrc = CurrentTexture_UV.Load(i3Coord);

          fUVref =fTempShiftedBlockHV[y * iBS_X_d2 + x];

          fChromaSAD += (abs((float)iUVsrc.r - iUVref.r) + abs((float)iUVsrc.g - iUVref.g));
        }
      }

      iChromaSAD = (int)fChromaSAD;

      if (g_chromaSADscale > 0)
      {
        iChromaSADAdd = iChromaSAD >> g_chromaSADscale;
      }
      else if (g_chromaSADscale < 0)
      {
        iChromaSADAdd = iChromaSAD << (-g_chromaSADscale);
      }
      else
        iChromaSADAdd = iChromaSAD;

      iChromaSADAdd = (float)iChromaSADAdd * g_chromaSADscale_fine;

      iSAD += iChromaSADAdd;
    }

    OutputTexture[DTid.xy] = iSAD;

    break;
  }
}
