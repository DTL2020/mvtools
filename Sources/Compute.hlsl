
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
	// c1.w - pad
	float4 gKernel1d2_0_2 : packoffset(c2); // K0 = k(2+1/2), K1 = k(1+1/2)=k(-1-1/2), K2 = k(1/2)=k(-1/2)
	
}

[numthreads(8, 8, 1)]
RS
void main(uint3 DTid : SV_DispatchThreadID)
{
	int iKS_d2 = 2;
	int x;
	int y;

	int iTempShiftedBlockY[(16 + (2 * 2)) * (16 + (2 * 2))];// use 16x16 max blocks ? iKS_d2 * 2, H shifted
  int iTempShiftedBlockY2[(16 + (2 * 2)) * (16 + (2 * 2))];// use 16x16 max blocks ? iKS_d2 * 2, V shifted
	int2 iTempShiftedBlockUV[(8 + (2 * 2)) * (8 + (2 * 2))]; // YV12 8x8 max UV + iKS_d2 * 2, H shifted
  int2 iTempShiftedBlockUV2[(8 + (2 * 2)) * (8 + (2 * 2))]; // YV12 8x8 max UV + iKS_d2 * 2, V shifted 

	int3 i3Coord;
		
	i3Coord.x = DTid.x;
	i3Coord.y = DTid.y;
	i3Coord.z = 0;

	int2 i2MV = ResolvedMVsTexture.Load(i3Coord);

	int iYsrc;
	int iYref;
	int2 iUVsrc;
	int2 iUVref;

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

		int2 i2MV_fp;
		i2MV_fp.r = i2MV.r >> 1;
		i2MV_fp.g = i2MV.g >> 1;

		// check if half shifts required
		if (((i2MV.r >> 1) && 1) == 1) // h half sub != 0
		{
			// shift ref block to TempShiftedBlock, Y plane
			// hor shift

			for (int y_sh = -iKS_d2; y_sh < g_BlockSizeY + iKS_d2; y_sh++) // -2..+2 for v shift if required
			{
				for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
				{
					float fYsum = 0;
					int s_idx = -2;

					i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh + s_idx;
					i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh;
					iYref = ReferenceTexture_Y.Load(i3Coord).r;
					fYsum += (float)iYref * gKernel1d2_0_2.x;// K0 = k(2+1/2)

					s_idx++;

					i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh + s_idx;
					i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh;
					iYref = ReferenceTexture_Y.Load(i3Coord).r;
					fYsum += (float)iYref * gKernel1d2_0_2.y;// K1 = k(1+(1/2))

					s_idx++;

					i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh + s_idx;
					i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh;
					iYref = ReferenceTexture_Y.Load(i3Coord).r;
					fYsum += (float)iYref * gKernel1d2_0_2.z;// K2 = k(1/2)

					s_idx++;

					i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh + s_idx;
					i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh;
					iYref = ReferenceTexture_Y.Load(i3Coord).r;
					fYsum += (float)iYref * gKernel1d2_0_2.z;// K2 = k(1/2)=k(-1/2)

					s_idx++;

					i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh + s_idx;
					i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh;
					iYref = ReferenceTexture_Y.Load(i3Coord).r;
					fYsum += (float)iYref * gKernel1d2_0_2.y;// K1 = k(1+(1/2))=k(-1-(1/2))

					iTempShiftedBlockY[y_sh * g_BlockSizeX + (x_sh + iKS_d2)] = (int)fYsum;

				}
			}

			// chroma h shift if required
			if (g_UseChroma != 0)
			{
				for (int y_sh = -iKS_d2; y_sh < iBS_Y_d2 + iKS_d2; y_sh++) // -2..+2 for v shift if required
				{
					for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
					{
            float2 fUVsum;
            fUVsum.x = 0;
            fUVsum.y = 0;
						int s_idx = -2;

						i3Coord.x = DTid.x * iBS_X_d2 + (i2MV_fp.r >> 1) + x_sh + s_idx;
						i3Coord.y = DTid.y * iBS_Y_d2 + (i2MV_fp.g >> 1) + y_sh;
						iUVref = ReferenceTexture_UV.Load(i3Coord);
						fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.x;// K0 = k(2+1/2)
						fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.x;// K0 = k(2+1/2)

						s_idx++;

						i3Coord.x = DTid.x * iBS_X_d2 + (i2MV_fp.r >> 1) + x_sh + s_idx;
						i3Coord.y = DTid.y * iBS_Y_d2 + (i2MV_fp.g >> 1) + y_sh;
						iUVref = ReferenceTexture_UV.Load(i3Coord);
						fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.y;// K1 = k(1+(1/2))
						fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.y;// K1 = k(1+(1/2))

						s_idx++;

            i3Coord.x = DTid.x * iBS_X_d2 + (i2MV_fp.r >> 1) + x_sh + s_idx;
            i3Coord.y = DTid.y * iBS_Y_d2 + (i2MV_fp.g >> 1) + y_sh;
            iUVref = ReferenceTexture_UV.Load(i3Coord);
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.z;// K2 = k(1/2)
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.z;// K2 = k(1/2)

						s_idx++;

            i3Coord.x = DTid.x * iBS_X_d2 + (i2MV_fp.r >> 1) + x_sh + s_idx;
            i3Coord.y = DTid.y * iBS_Y_d2 + (i2MV_fp.g >> 1) + y_sh;
            iUVref = ReferenceTexture_UV.Load(i3Coord);
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.z;// K2 = k(1/2)=k(-1/2)
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.z;// K2 = k(1/2)=k(-1/2)

						s_idx++;

            i3Coord.x = DTid.x * iBS_X_d2 + (i2MV_fp.r >> 1) + x_sh + s_idx;
            i3Coord.y = DTid.y * iBS_Y_d2 + (i2MV_fp.g >> 1) + y_sh;
            iUVref = ReferenceTexture_UV.Load(i3Coord);
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.y;// K1 = k(1+(1/2))=k(-1-(1/2))
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.y;// K1 = k(1+(1/2))=k(-1-(1/2))

            iTempShiftedBlockUV[y_sh * iBS_X_d2 + (x_sh + iKS_d2)].x = (int)fUVsum.x;
            iTempShiftedBlockUV[y_sh * iBS_X_d2 + (x_sh + iKS_d2)].y = (int)fUVsum.y;

					}
				}

			}

		}
		else
		{
			// copy ref to temp buf
			for (int y_sh = -iKS_d2; y_sh < g_BlockSizeY + iKS_d2; y_sh++) // -iKS_d2..+iKS_d2 for v shift if required
			{
				for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
				{
					i3Coord.x = DTid.x * g_BlockSizeX + i2MV_fp.r + x_sh;
					i3Coord.y = DTid.y * g_BlockSizeY + i2MV_fp.g + y_sh;
					iTempShiftedBlockY[y_sh * g_BlockSizeX + (x_sh + iKS_d2)] = ReferenceTexture_Y.Load(i3Coord).r;
				}
			}

      if (g_UseChroma != 0)
      {
        for (int y_sh = -iKS_d2; y_sh < iBS_Y_d2 + iKS_d2; y_sh++) // -iKS_d2..+iKS_d2 for v shift if required
        {
          for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
          {
            i3Coord.x = DTid.x * iBS_X_d2 + (i2MV_fp.r >> 1) + x_sh;
            i3Coord.y = DTid.y * iBS_Y_d2 + (i2MV_fp.g >> 1) + y_sh;
            iUVref = ReferenceTexture_UV.Load(i3Coord);
            iTempShiftedBlockUV[y_sh * iBS_X_d2 + (x_sh + iKS_d2)] = ReferenceTexture_UV.Load(i3Coord);
          }
        }
      }
		}

		if (((i2MV.g >> 1) && 1) == 1) // v half sub != 0
		{
			// V shift ref block from TempShiftedBlockY to iTempShiftedBlockY2, Y plane

			for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++) // -2..+2 for v shift if required
			{
				for (int y_sh = 0; y_sh < g_BlockSizeY; y_sh++)
				{
					float fYsum = 0;
					int s_idx = -2;

          iYref = iTempShiftedBlockY[(y_sh + s_idx) * g_BlockSizeX + x_sh];
					fYsum += (float)iYref * gKernel1d2_0_2.x;// K0 = k(2+1/2)

					s_idx++;

          iYref = iTempShiftedBlockY[(y_sh + s_idx) * g_BlockSizeX + x_sh];
          fYsum += (float)iYref * gKernel1d2_0_2.y;// K1 = k(1+(1/2))

					s_idx++;

          iYref = iTempShiftedBlockY[(y_sh + s_idx) * g_BlockSizeX + x_sh];
          fYsum += (float)iYref * gKernel1d2_0_2.z;// K2 = k(1/2)

					s_idx++;

          iYref = iTempShiftedBlockY[(y_sh + s_idx) * g_BlockSizeX + x_sh];
          fYsum += (float)iYref * gKernel1d2_0_2.z;// K2 = k(1/2)=k(-1/2)

					s_idx++;

          iYref = iTempShiftedBlockY[(y_sh + s_idx) * g_BlockSizeX + x_sh];
          fYsum += (float)iYref * gKernel1d2_0_2.y;// K1 = k(1+(1/2))=k(-1-(1/2))

					iTempShiftedBlockY2[(y_sh + iKS_d2) * g_BlockSizeX + x_sh] = (int)fYsum;

				}
			}

      // chroma v shift if required
      if (g_UseChroma != 0)
      {
        for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++) 
        {
          for (int y_sh = 0; y_sh < iBS_Y_d2; y_sh++)
          {
            float2 fUVsum;
            fUVsum.x = 0;
            fUVsum.y = 0;
            int s_idx = -2;

            iUVref = iTempShiftedBlockUV[(y_sh + s_idx) * iBS_X_d2 + x_sh];
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.x;// K0 = k(2+1/2)
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.x;// K0 = k(2+1/2)

            s_idx++;

            iUVref = iTempShiftedBlockUV[(y_sh + s_idx) * iBS_X_d2 + x_sh];
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.y;// K1 = k(1+(1/2))
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.y;// K1 = k(1+(1/2))

            s_idx++;

            iUVref = iTempShiftedBlockUV[(y_sh + s_idx) * iBS_X_d2 + x_sh];
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.z;// K2 = k(1/2)
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.z;// K2 = k(1/2)

            s_idx++;

            iUVref = iTempShiftedBlockUV[(y_sh + s_idx) * iBS_X_d2 + x_sh];
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.z;// K2 = k(1/2)=k(-1/2)
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.z;// K2 = k(1/2)=k(-1/2)

            s_idx++;

            iUVref = iTempShiftedBlockUV[(y_sh + s_idx) * iBS_X_d2 + x_sh];
            fUVsum.x = (float)iUVref.x * gKernel1d2_0_2.y;// K1 = k(1+(1/2))=k(-1-(1/2))
            fUVsum.y = (float)iUVref.y * gKernel1d2_0_2.y;// K1 = k(1+(1/2))=k(-1-(1/2))

            iTempShiftedBlockUV2[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].x = (int)fUVsum.x;
            iTempShiftedBlockUV2[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].y = (int)fUVsum.y;

          }
        }
      } // if g_UseChroma

		}
    else // copy tempY to tempY2, UV to UV2
    {
      for (int y_sh = -iKS_d2; y_sh < g_BlockSizeY + iKS_d2; y_sh++) // -iKS_d2..+iKS_d2 for v shift if required
      {
        for (int x_sh = 0; x_sh < g_BlockSizeX; x_sh++)
        {
          iTempShiftedBlockY2[(y_sh + iKS_d2) * g_BlockSizeX + x_sh] = iTempShiftedBlockY[(y_sh + iKS_d2) * g_BlockSizeX + x_sh];
        }
      }

      if (g_UseChroma != 0)
      {
        for (int y_sh = -iKS_d2; y_sh < iBS_Y_d2 + iKS_d2; y_sh++) // -iKS_d2..+iKS_d2 for v shift if required
        {
          for (int x_sh = 0; x_sh < iBS_X_d2; x_sh++)
          {
            iTempShiftedBlockUV2[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].x = iTempShiftedBlockUV[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].x;
            iTempShiftedBlockUV2[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].y = iTempShiftedBlockUV[(y_sh + iKS_d2) * iBS_X_d2 + x_sh].y;
          }
        }
      }
    }

		// calc SAD finally

		for (y = 0; y < g_BlockSizeY; y++)
		{
			for (x = 0; x < g_BlockSizeX; x++)
			{
				i3Coord.x = DTid.x * g_BlockSizeX + x;
				i3Coord.y = DTid.y * g_BlockSizeY + y;
				iYsrc = CurrentTexture_Y.Load(i3Coord).r;

				iYref = iTempShiftedBlockY2[(y+iKS_d2)* g_BlockSizeX + x];

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

          iUVref = iTempShiftedBlockUV2[(y + iKS_d2) * iBS_X_d2 + x]; // make it float and convert here ?

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
		break; // nPel=2

	default: // nPel = 4

		break;
	}
}
