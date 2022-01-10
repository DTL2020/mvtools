
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
	float4 g_MaxThreadIter : packoffset(c0);
	float4 g_Window : packoffset(c1);
}

//[numthreads(8, 8, 1)]
[numthreads(8, 8, 1)]
RS
void main(uint3 DTid : SV_DispatchThreadID)
{
	int3 i3Coord;
		
	i3Coord.x = DTid.x;
	i3Coord.y = DTid.y;
	i3Coord.z = 0;

	int iBlockSize = 8;

	int2 i2MV = ResolvedMVsTexture.Load(i3Coord);
	i2MV.r = i2MV.r >> 2; // full frame search qpel/4
	i2MV.g = i2MV.g >> 2;

	int iYsrc;
	int iYref;
	int2 iUVsrc;
	int2 iUVref;

	int iSAD = 0;

	for (int x = 0; x < iBlockSize; x++)
	{
		for (int y = 0; y < iBlockSize; y++)
		{
			i3Coord.x = DTid.x * iBlockSize + x;
			i3Coord.y = DTid.y * iBlockSize + y;
			iYsrc = CurrentTexture_Y.Load(i3Coord).r;

			i3Coord.x = DTid.x * iBlockSize + x + i2MV.r;
			i3Coord.y = DTid.y * iBlockSize + y + i2MV.g;

			iYref = ReferenceTexture_Y.Load(i3Coord).r;

			iSAD += abs(iYsrc - iYref);
		}
	}

	OutputTexture[DTid.xy] = iSAD; // i2MV.r;
}