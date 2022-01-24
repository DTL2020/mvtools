
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
}

[numthreads(8, 8, 1)]
RS
void main(uint3 DTid : SV_DispatchThreadID)
{
	int3 i3Coord;
		
	i3Coord.x = DTid.x;
	i3Coord.y = DTid.y;
	i3Coord.z = 0;

	int2 i2MV = ResolvedMVsTexture.Load(i3Coord);
	i2MV.r = i2MV.r >> g_precisionMVs;// 2 = full frame search qpel/4, 1 = half frame search qpel/2
	i2MV.g = i2MV.g >> g_precisionMVs;

	int iYsrc;
	int iYref;
	int2 iUVsrc;
	int2 iUVref;

	int iSAD = 0;
	int iChromaSAD = 0;
	int iChromaSADAdd = 0;

	for (int y = 0; y < g_BlockSizeY; y++)
	{
		for (int x = 0; x < g_BlockSizeX; x++)
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
		int iBS_X_d2 = g_BlockSizeX >> 1;
		int iBS_Y_d2 = g_BlockSizeY >> 1;

		for (int y = 0; y < iBS_Y_d2; y++)
		{
			for (int x = 0; x < iBS_X_d2; x++)
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
}