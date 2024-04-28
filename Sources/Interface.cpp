// Set of tools based on Motion Compensation

// Author: Manao
// Copyright(c)2006 A.G.Balakhnin aka Fizick - some modifications, additions (see releases history in provided documentation)
// See legal notice in Copying.txt for more information

// Avisynth: www.avisynth.org

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

// Use MVClip / MVFilter helpers
#include "MVMask.h"
#include "MVShow.h"
#include "MVCompensate.h"
#include "MVSCDetection.h"

#include "MVDepan.h" // added by Fizick
#include "MVFlow.h"
#include "MVFlowInter.h"
#include "MVFlowFps.h"
#include "MVFlowBlur.h"
/*
#include "MVDegrain1.h"
#include "MVDegrain2.h"
*/
#include "MVDegrain3.h"
#include "MDegrainN.h"
#include "MVBlockFps.h"

// Analysing filter
#include "MVAnalyse.h"
#include "MVRecalculate.h"
#include "MVSuper.h"
#include "MAverage.h"


// Test & helpers filters
#include "Padding.h"
#include "MVFinest.h"
#include "MRestoreVect.h"
#include "MScaleVect.h"
#include "MStoreVect.h"
#include "MTransform.h"

#include <avisynth.h>
#include <stdint.h>



AVSValue __cdecl Create_Padding(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  return new Padding(
    args[0].AsClip(),
    args[1].AsInt(8),
    args[2].AsInt(8),
    args[3].AsBool(false), // planar
    env
  );
}

AVSValue __cdecl Create_MVShow(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  int sc = 1;
  int sil = 0;
  int tol = 20000;
  return new MVShow(
    args[0].AsClip(),       // super
    args[1].AsClip(),       // vec
    args[2].AsInt(sc),
    args[3].AsInt(sil),
    args[4].AsInt(tol),
    args[5].AsBool(false),  // showsad
    args[6].AsInt(-1),      // number
    args[7].AsInt(MV_DEFAULT_SCD1),
    args[8].AsInt(MV_DEFAULT_SCD2),
    args[9].AsBool(true),
    args[10].AsBool(false), // planar
    env
  );
}

AVSValue __cdecl Create_MVCompensate(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  const sad_t thsad = args[5].AsInt(10000);   // thSAD

  return new MVCompensate(
    args[0].AsClip(),       // source
    args[1].AsClip(),       // super
    args[2].AsClip(),       // vec
    args[3].AsBool(true),
    args[4].AsFloat(0),     // recursion
    thsad,                  // thSAD
    args[6].AsBool(false),  // fields
    args[7].AsFloat(100), //time in percent
    args[8].AsInt(MV_DEFAULT_SCD1),
    args[9].AsInt(MV_DEFAULT_SCD2),
    args[10].AsBool(true),
    args[11].AsBool(false), // planar
    args[12].AsBool(true),  // mt
    args[13].AsInt(0),		// tr
    args[14].AsBool(true),	// center
    args[15].IsClip() ? args[16].AsClip() : 0, // cclip
    args[16].AsInt(thsad),  // thSAD2  todo sad_t float
    args[17].AsBool(false), // show RNB
    env
  );
}

AVSValue __cdecl Create_MVSCDetection(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  return new MVSCDetection(
    args[0].AsClip(),
    args[1].AsClip(),
    args[2].AsFloatf(-9999.0f),
    args[3].AsInt(MV_DEFAULT_SCD1),
    args[4].AsInt(MV_DEFAULT_SCD2),
    args[5].AsBool(true),
    env
  );
}

AVSValue __cdecl Create_MVAnalyse(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  int blksize = args[1].AsInt(8);       // block size horizontal
  int blksizeV = args[2].AsInt(blksize); // block size vertical

  int lambda;
  int lsad;
  int pnew;
  int plevel;
  bool global;
  int overlap = args[18].AsInt(0);

  int AreaMode = args[41].AsInt(0); // AreaMode 2.7.46, number of steps around center block position
  int PredictorType = args[34].AsInt(0);   // optpredictortype 2.7.46

  bool truemotion = args[11].AsBool(true); // preset added in v0.9.13
  if (truemotion)
  {
    lambda = args[8].AsInt(1000 * blksize*blksizeV / 64);
    lsad = args[12].AsInt(1200);
    pnew = args[15].AsInt(50); // relative to 256 in v1.5.8
    plevel = args[13].AsInt(1);
    global = args[14].AsBool(true);
  }
  else // old versions 0.9.9.1 compatibility mode
  {
    lambda = args[8].AsInt(0);
    lsad = args[12].AsInt(400);
    pnew = args[15].AsInt(0);
    plevel = args[13].AsInt(0);
    global = args[14].AsBool(false);
  }

  return new MVAnalyse(
    args[0].AsClip(),       // super
    blksize,
    blksizeV,                // v.1.7
    args[3].AsInt(0),       // levels skip
    args[4].AsInt(4),       // search type
    args[5].AsInt(2),       // search parameter
    args[6].AsInt(0),       // search parameter at finest level
    args[7].AsBool(false),  // is backward
    lambda,                  // lambda
    args[9].AsBool(true),   // chroma = true since v1.0.1
    args[10].AsInt(1),       // delta frame
    lsad,                    // lsad - SAD limit for lambda using - added by Fizick (was internal fixed to 400 since v0.9.7)
    plevel,                  // plevel - exponent for lambda scaling on level size  - added by Fizick
    global,                  // use global motion predictor - added by Fizick
    pnew,                    // pnew - cost penalty for new candidate vector - added by Fizick
    args[16].AsInt(pnew),    // pzero - v1.10.3
    args[17].AsInt(0),       // pglobal
    overlap,                 // overlap horizontal
    args[19].AsInt(overlap), // overlap vertical - v1.7
    args[20].AsString(""),   // outfile - v1.2.6
    args[21].AsInt(0),       // dct
    args[22].AsInt(0),       // divide
    args[23].AsInt(0),       // sadx264
    args[24].AsInt(10000),   // badSAD
    args[25].AsInt(24),      // badrange
    args[26].AsBool(true),   // isse
    args[27].AsBool(true),   // meander blocks scan
    args[28].AsBool(false),  // temporal predictor
    args[29].AsBool(false),  // try many
    args[30].AsBool(false),  // multi
    args[31].AsBool(true),   // mt
    args[32].AsInt(0),   // scaleCSAD
    args[33].AsInt(0),   // optsearchoption 2.7.46
//    args[34].AsInt(0),   // optpredictortype 2.7.46
    PredictorType, 
    args[35].AsFloat(1.0),   // scaleCSADfine 2.7.46
    args[36].AsInt(0), // accnum 2.7.46
    args[37].AsInt(0), // UseSubShift 2.7.46
    args[38].AsClip(), // SuperCurrent 2.7.46
    args[39].AsInt(0), // SearchDirMode 2.7.46
    args[40].AsInt(1), // DMFlags 2.7.46
//    args[41].AsInt(0), // AreaMode 2.7.46, number of steps around center block position
    AreaMode,
    args[42].AsInt(0), // AreaMode MVs difference hint to SAD addition
    args[43].AsInt(0), // AreaMode step size in samples, 0 - auto (depend on block size, 1 for 8x8, 2 for 16x16 and so on)
    args[44].AsInt(0), // AreaMode offset for start search, 0 - no (start from center block)
    args[45].AsInt(AreaMode), // AreaMode pel, 0 - no AreaMode at sub-sample levels of search (0 for pel=2 and 0 and 1 for pel=4), > 0 = number of steps at sub-sample levels
    args[46].AsInt(PredictorType), // PredictorType at sub-sample levels
    args[47].AsInt(1), // AMflags - 1 for diagonals steps only, 2 - sides steps
    args[48].AsInt(0), // AMavg
    args[49].AsInt(0), // AMpt - predictor type for AM searches
    args[50].AsInt(3), // AMst - search type for AM searches
    args[51].AsInt(2), // AMsp - search param for AM searches
    args[52].AsInt(-1), // tmavg - trymany MVs averaging mode, -1 - minSAD(DM) mode, 0 and higher - equal to AMavg modes
    args[53].AsInt(-1), // mdp - MotionDistortion predictor, -1 - hierarchy predictor, 0 and higher - AMavg mode average of (some) predictors
    args[54].AsInt(1), // vs - number of vertical scan passes (in each level), if >1 - perform meander V-scan in attempt to increase quality of updated predictor
    env
  );
}
//tmavg]i[mdp]

AVSValue __cdecl Create_MVMask(AVSValue args, void*, IScriptEnvironment* env)
{
  double ml = args[2].AsFloat(100);
  if (ml <= 0)
    env->ThrowError("MVMask: ML must be > 0.0");
  float gamma = 1;
  int Ysc = 0;
  return new MVMask(
    args[0].AsClip(), // source
    args[1].AsClip(), // vec
    ml,
    args[3].AsFloat(gamma),
    args[4].AsInt(0), // kind - replaced by Fizick
    args[5].AsFloat(100), // time percent
    args[6].AsInt(Ysc),
    args[7].AsInt(MV_DEFAULT_SCD1),
    args[8].AsInt(MV_DEFAULT_SCD2),
    args[9].AsBool(true),
    args[10].AsBool(false),
    env
  );
}

AVSValue __cdecl Create_MVDepan(AVSValue args, void*, IScriptEnvironment* env) // Added by Fizick
{
  return new MVDepan(
    args[0].AsClip(),
    args[1].AsClip(),
    (args[2].Defined() ? args[2].AsClip() : 0), // mask clip
    args[3].AsBool(true),           // zoom
    args[4].AsBool(true),           // rot
    (float)args[5].AsFloat(1),      // pixaspect
    (float)args[6].AsFloat(15),     // error
    args[7].AsBool(false),          // info
    args[8].AsString(""),           // log
    (float)args[9].AsFloat(10),     // wrong difference
    (float)args[10].AsFloat(0.05f), // zeroMV weight
    args[11].AsInt(0),              // range
    args[12].AsInt(MV_DEFAULT_SCD1),
    args[13].AsInt(MV_DEFAULT_SCD2),
    args[14].AsBool(true),
    args[15].AsBool(false),         // planar
    env
  );
}


AVSValue __cdecl Create_MVFlow(AVSValue args, void*, IScriptEnvironment* env)
{
  double time = args[3].AsFloat(100.0);
  if (time < 0 || time>100)
  {
    env->ThrowError("MVFlow: Time must be from 0 to 100 percent.");
  }
  int time256 = int(time*256.0 / 100.0);
  return new MVFlow(
    args[0].AsClip(),      // source
    args[1].AsClip(),      // super
    args[2].AsClip(),      // vec
    time256,
    args[4].AsInt(0),      // fetch or shift
    args[5].AsBool(false), // fields
    args[6].AsInt(MV_DEFAULT_SCD1),
    args[7].AsInt(MV_DEFAULT_SCD2),
    args[8].AsBool(true),
    args[9].AsBool(false), // planar
    args[10].IsClip() ? args[10].AsClip() : 0,
    env
  );
}

AVSValue __cdecl Create_MVFlowInter(AVSValue args, void*, IScriptEnvironment* env)
{
  double time = args[4].AsFloat(50.0);
  if (time < 0 || time>100)
  {
    env->ThrowError("MVFlowInter: Time must be from 0 to 100 percent.");
  }
  int time256 = int(time*256.0 / 100.0);
  double ml = args[5].AsFloat(100);
  if (ml <= 0)
  {
    env->ThrowError("MVFlowInter: ML must be > 0.0");
  }
  return new MVFlowInter(
    args[0].AsClip(),       // source
    args[1].AsClip(),       // finest
    args[2].AsClip(),       // mvbw
    args[3].AsClip(),       // mvfw
    time256,
    ml,                     // ml
    args[6].AsBool(true),   // blend
    args[7].AsInt(MV_DEFAULT_SCD1),
    args[8].AsInt(MV_DEFAULT_SCD2),
    args[9].AsBool(true),   // isse
    args[10].AsBool(false), // planar
    env);
}

AVSValue __cdecl Create_MVFlowFps(AVSValue args, void*, IScriptEnvironment* env)
{
  double ml = args[7].AsFloat(100);
  if (ml <= 0)
  {
    env->ThrowError("MVFlowFps: ML must be > 0.0");
  }
  return new MVFlowFps(
    args[0].AsClip(),       // source
    args[1].AsClip(),       // finest
    args[2].AsClip(),       // mvbw
    args[3].AsClip(),       // mvfw
    args[4].AsInt(25),      // num
    args[5].AsInt(1),       // den
    args[6].AsInt(2),       // maskmode - v1.8.1
    ml,                     // ml
    args[8].AsBool(true),   // blend
    args[9].AsInt(MV_DEFAULT_SCD1),
    args[10].AsInt(MV_DEFAULT_SCD2),
    args[11].AsBool(true),  // isse
    args[12].AsBool(false), // planar
    args[13].AsInt(0), // optDebug
    env
  );
}

AVSValue __cdecl Create_MVFlowBlur(AVSValue args, void*, IScriptEnvironment* env)
{
  double time = args[4].AsFloat(50.0);
  if (time < 0 || time>200)
  {
    env->ThrowError("MVFlowBlur: Blur time must be from 0 to 200 percent.");
  }
  int blur256 = int(time*256.0 / 200.0);
  return new MVFlowBlur(
    args[0].AsClip(),      // source
    args[1].AsClip(),      // finest
    args[2].AsClip(),      // mvbw
    args[3].AsClip(),      // mvfw
    blur256,
    args[5].AsInt(1),      // prec
    args[6].AsInt(MV_DEFAULT_SCD1),
    args[7].AsInt(MV_DEFAULT_SCD2),
    args[8].AsBool(true),  // isse
    args[9].AsBool(false), // planar
    env
  );
}

AVSValue __cdecl Create_MVDegrainX(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  int level = (int)(intptr_t)user_data;

  int plane_param_index = 6; // base: MDegrain1
  int thsad_param_index = 4;
  int limit_param_index = 7;

  int param_index_shift = 0;
  switch (level) {
  case 6: param_index_shift = 10; break; // new PF 170105 MDegrain6
  case 5: param_index_shift = 8; break; // new PF 160926 MDegrain5
  case 4: param_index_shift = 6; break; // new PF 160926 MDegrain4
  case 3: param_index_shift = 4; break;
  case 2: param_index_shift = 2; break;
  }

  int plane = args[plane_param_index + param_index_shift].AsInt(4);
  int YUVplanes;

  switch (plane)
  {
  case 0:
    YUVplanes = 1;
    break;
  case 1:
    YUVplanes = 2;
    break;
  case 2:
    YUVplanes = 4;
    break;
  case 3:
    YUVplanes = 6;
    break;
  case 4:
  default:
    YUVplanes = 7;
  }

  int thSAD = args[thsad_param_index + param_index_shift].AsInt(400);  // thSAD

  // change limit. 2.7.25-: use 255 as default for all bit depth. v42: float
  const float limit = args[limit_param_index + param_index_shift].AsFloatf(255.f);
  return new MVDegrainX(
    args[0].AsClip(),       // source
    args[1].AsClip(),       // super
    args[2].AsClip(),       // mvbw
    args[3].AsClip(),       // mvfw
    level >= 2 ? args[4].AsClip() : nullptr,       // mvbw2
    level >= 2 ? args[5].AsClip() : nullptr,       // mvfw2
    level >= 3 ? args[6].AsClip() : nullptr,       // mvbw3
    level >= 3 ? args[7].AsClip() : nullptr,       // mvfw3
    level >= 4 ? args[8].AsClip() : nullptr,       // mvbw4
    level >= 4 ? args[9].AsClip() : nullptr,       // mvfw4
    level >= 5 ? args[10].AsClip() : nullptr,       // mvbw5
    level >= 5 ? args[11].AsClip() : nullptr,       // mvfw5
    level >= 6 ? args[12].AsClip() : nullptr,       // mvbw6
    level >= 6 ? args[13].AsClip() : nullptr,       // mvfw6
    thSAD,                  // thSAD
    args[5 + param_index_shift].AsInt(thSAD),   // thSAD
    YUVplanes,              // YUV planes
    limit,                  // limit
    args[8 + param_index_shift].AsFloatf(limit),  // limitC
    args[9 + param_index_shift].AsInt(MV_DEFAULT_SCD1), // thSCD1
    args[10 + param_index_shift].AsInt(MV_DEFAULT_SCD2), // thSCD2
    args[11 + param_index_shift].AsBool(true),  // isse
    args[12 + param_index_shift].AsBool(false), // planar
    args[13 + param_index_shift].AsBool(false), // lsb
    args[14 + param_index_shift].AsBool(true),  // mt
    args[15 + param_index_shift].AsBool(false),  // out16
    args[16 + param_index_shift].AsBool(false),  // out32
    level,
    env
  );
}

AVSValue __cdecl Create_MDegrainN(AVSValue args, void*, IScriptEnvironment* env)
{
  int plane = args[6].AsInt(4);
  int YUVplanes;

  switch (plane)
  {
  case 0:
    YUVplanes = 1;
    break;
  case 1:
    YUVplanes = 2;
    break;
  case 2:
    YUVplanes = 4;
    break;
  case 3:
    YUVplanes = 6;
    break;
  case 4:
  default:
    YUVplanes = 7;
    break;
  }

  const int tr = args[3].AsInt(1);       // tr
  const int thSAD = args[4].AsInt(400);     // thSAD
  const int thSADC = args[5].AsInt(thSAD);   // thSADC
  // bit-depth adaptive limit? not here
  const float limit = args[7].AsFloatf(255.f); // change limit. 2.7.25-: use 255 as default for all bit depth v42:float
  const int thSAD2 = args[14].AsInt(thSAD);  // thSAD2
  const int thSADC2 = args[15].AsInt(thSAD2); // thSADC2

  // Switch to MDegrain1/2/3/4/5/6 when possible (faster)
  if (thSAD2 == thSAD && thSADC == thSADC2)
  {
    if (tr <= MAX_DEGRAIN) // up to MDegrain5 160926, MDegrain6 170105
    {

      return new MVDegrainX(
        args[0].AsClip(),        // source
        args[1].AsClip(),        // super clip
        args[2].AsClip(),        // mvbw
        ::PClip(),                // mvfw
        tr >= 2 ? ::PClip() : nullptr,                // mvbw2
        tr >= 2 ? ::PClip() : nullptr,                // mvfw2
        tr >= 3 ? ::PClip() : nullptr,                // mvbw3
        tr >= 3 ? ::PClip() : nullptr,                // mvfw3
        tr >= 4 ? ::PClip() : nullptr,                // mvbw4
        tr >= 4 ? ::PClip() : nullptr,                // mvfw4
        tr >= 5 ? ::PClip() : nullptr,                // mvbw5
        tr >= 5 ? ::PClip() : nullptr,                // mvfw5
        tr >= 6 ? ::PClip() : nullptr,                // mvbw6
        tr >= 6 ? ::PClip() : nullptr,                // mvfw6
        thSAD,                     // thSAD
        thSADC,                    // thSADC
        YUVplanes,                 // YUV planes
        limit,                     // limit
        args[8].AsFloatf(limit),   // limitC
        args[9].AsInt(MV_DEFAULT_SCD1),  // thSCD1
        args[10].AsInt(MV_DEFAULT_SCD2), // thSCD2
        args[11].AsBool(true),   // isse
        args[12].AsBool(false),  // planar
        args[13].AsBool(false),  // lsb
        args[16].AsBool(true),   // mt
        args[17].AsBool(false),  // out16
        false, // args[18].AsBool(false),  // cannot pass: no out32 parameter for MDegrainN
        tr, // PF: level
        env
      );
    }
  }
  // Standard MDegrainN
  return new MDegrainN(
    args[0].AsClip(),        // source
    args[1].AsClip(),        // super clip
    args[2].AsClip(),        // mvmulti
    tr,                        // tr
    thSAD,                     // thSAD
    thSADC,                    // thSADC
    YUVplanes,                 // YUV planes
    limit,                     // limit
    args[8].AsFloatf(limit),   // limitC
    args[9].AsInt(MV_DEFAULT_SCD1),  // thSCD1
    args[10].AsInt(MV_DEFAULT_SCD2), // thSCD2
    args[11].AsBool(true),   // isse
    args[12].AsBool(false),  // planar
    args[13].AsBool(false),  // lsb
    thSAD2,                    // thSAD2
    thSADC2,                   // thSADC2
    args[16].AsBool(true),   // mt
    args[17].AsBool(false),   // out16
    args[18].AsInt(2),      // wpow, 2.7.46
    args[19].AsFloat(1.0f), // adjSADzeromv
    args[20].AsFloat(1.0f), // adjSADcohmv
    args[21].AsInt(-1),     // thCohMV
    args[22].AsFloat(1.0f), // MVLPFCutoff
    args[23].AsFloat(1.0f), // MVLPFSlope
    args[24].AsFloat(0.0f), // MVLPFGauss
    args[25].AsInt(0), // thMVLPFCorr
    args[26].AsFloat(1.0f), //adjSADLPFedmv
    args[27].AsInt(0), // UseSubShift 0 = disabled (use super clip pel planes), >0 = kernel size for speed/quality balance ? or shift kernel type ?
    args[28].AsInt(0), // IntOvlp = 0 - interpolate overlap, default disabled
    args[29].AsClip(), // mvmultirs - reverse search mvmulti clip
    args[30].AsInt(0), // thFWBWmvpos
    args[31].AsInt(10), // MPBthSub
    args[32].AsInt(20), // MPBthAdd 
    args[33].AsInt(0), // MPBNumIt
    args[34].AsFloat(0.7f), // MPB_SPCsub - single pass coeff for subtractively checked blocks
    args[35].AsFloat(1.5f), // MPB_SPCadd - single pass coeff for additively checked blocks, typically MPB_SPCadd ~ 1/MPB_SPCsub as start values for tuning
    args[36].AsBool(false), // MPB_PartBlend
    args[37].AsInt(0), // MPBthIVS - threshold isMVsStable
    args[38].AsBool(false), // showIVSmask
    args[39].AsClip(), // mvmultivs - for IVS mask search with separate MVclip, optional
    args[40].AsInt(1), // MPB_DMFlags - dissimilarity metric flags for MPB, default 1 = use SAD only
    args[41].AsInt(0), // MPBchroma - use or not chroma in MPB (separately from main clip)
    args[42].AsInt(1), // MPBtgtTR
    args[43].AsInt(4), // MPB_MVlth
    args[44].AsInt(0), // pmode, default=0 (blend mode), 1 is MEL block select mode
    args[45].AsInt(1), // dissimilarity metric flags for TTH, default 1 = use SAD only
    args[46].AsInt(0), // TTH_thUPD - TTH memory update threshold, if 0 - TTH processing disabled in both pmode 0 and 1.
    args[47].AsInt(1), // TTH_BAS - TTH block area size (in single side surrounding block count)
    args[48].AsBool(true), // TTH_chroma - use or not chroma part at dismetric ca
    args[49].AsClip(), // dnmask - Y8 (?) mask clip to control ref frames additional weight (to reduce degrain at masked blocks or samples ? (depend on mask clip frame size ?))
    args[50].AsFloat(0), // thSADA_a - a-param (multiplier) of auto-thSAD calculation
    args[51].AsFloat(0), // thSADA_b - b-param (additive) of auto-thSAD calculation
    args[52].AsInt(0), // MVDedF - MV Median-like filterting radius, 0 - default disabled
    args[53].AsInt(0), // MVMedF_em - MV Median-like filterting temporal edges processing mode: 0 - use all edge MVs, 1 - skip non-filtered MVs (invalidate SAD)
    args[54].AsInt(0), // MVMedF_cm - MV Median-like filterting temporal coordinates processing mode: 0 - use separated x,y filtering, 1 - use vector length dismetric
    args[55].AsInt(0), // MVF_fm - MV filtering blocks fail mode: 0 - pass blocks with too bad filtered MVs SADs to blending, 1 - invalidate blocks with too bad filtered MVs SADs (skip from blending)
    args[56].AsInt(0), // MGR - multi-generation MVs refining processing. Integer number of additional refining generations. 0 - disabled.
    args[57].AsInt(0), // MGR_sr - search radius
    args[58].AsInt(0), // MGR_st - search type, 0 - NStepSearch, 1 - Logariphmic/Diamond, 2 - Exhaustive, 3 - Hexagon, 4 - UMH ?
    args[59].AsInt(1), // MGR_pm - predictors bitmask (1 - input source, 2 - after MVF)
                       // fixme: out32
    args[60].AsInt(0), // LtComp - compesate for lighting changes 0 - default disabled, 1 - only DC comp mode
    env
  );
}

AVSValue __cdecl Create_MVRecalculate(AVSValue args, void*, IScriptEnvironment* env)
{
  int blksize = args[4].AsInt(8);       // block size horizontal
  int blksizeV = args[5].AsInt(blksize); // block size vertical

  int lambda;
  int pnew;

  int pglobal;
  int pzero;

  int overlap = args[12].AsInt(0);

  bool truemotion = args[10].AsBool(true); // preset added in v0.9.13
  if (truemotion)
  {
    lambda = args[8].AsInt(1000 * blksize*blksizeV / 64);
    pnew = args[11].AsInt(50); // relative to 256 in v1.5.8
    pzero = args[35].AsInt(50);
    pglobal = args[36].AsInt(50); 
  }
  else // old versions 0.9.9.1 compatibility mode
  {
    lambda = args[8].AsInt(0);
    pnew = args[11].AsInt(0);
    pzero = args[35].AsInt(0);
    pglobal = args[36].AsInt(0);
  }
  return new MVRecalculate(
    args[0].AsClip(),       // super
    args[1].AsClip(),       // vectors
    args[2].AsInt(200),     // thSAD
    args[3].AsInt(1),       // smooth
    blksize,
    blksizeV,                // v.1.7
    args[6].AsInt(4),       // search type
    args[7].AsInt(2),       // search parameter
    lambda,                  // lambda
    args[9].AsBool(true),   // chroma = true since v1.0.1
    pnew,                    // pnew - cost penalty for new candidate vector - added by Fizick
    overlap,                 // overlap horizontal
    args[13].AsInt(overlap), // overlap vertical - v1.7
    args[14].AsString(""),   // outfile - v1.2.6
    args[15].AsInt(0),       // dct
    args[16].AsInt(0),       // divide
    args[17].AsInt(0),       // sadx264
    args[18].AsBool(true),   // isse2
    args[19].AsBool(true),   // meander
    args[20].AsInt(0),       // tr
    args[21].AsBool(true),   // mt
    args[22].AsInt(0), // scaleCSAD
    args[23].AsInt(0), // optsearchoption 2.7.46
    args[24].AsInt(3), // optpredictortype 2.7.46. Value 3 = pre-2.7.46 no new predictors, Refine() only
    args[25].AsInt(1), // DMFlags, default 1 = SAD only
    args[26].AsInt(0), // AreaMode
    args[27].AsInt(0), // AMdiffSAD
    args[28].AsInt(0), // AMstep
    args[29].AsInt(0), // AMoffset
    args[30].AsClip(), // SuperCurrent
    args[31].AsFloat(10.0f), // AMthVSMang - threshold of angular Vectors Stability Metric, default 10.0f - disabled
    args[32].AsInt(1), // AMflags
    args[33].AsInt(0), // AMavg
    args[34].AsBool(true), // global (use global predictor or not
    pzero,
    pglobal,
    env
  );
}

AVSValue __cdecl Create_MVBlockFps(AVSValue args, void*, IScriptEnvironment* env)
{
  return new MVBlockFps(
    args[0].AsClip(),       // src
    args[1].AsClip(),       // super
    args[2].AsClip(),       // bw
    args[3].AsClip(),       // fw
    args[4].AsInt(25),      // num
    args[5].AsInt(1),       // den
    args[6].AsInt(3), // mode
    args[7].AsFloat(100.0), // ml
    args[8].AsBool(true),   // blend
    args[9].AsInt(MV_DEFAULT_SCD1),
    args[10].AsInt(MV_DEFAULT_SCD2),
    args[11].AsBool(true),  // isse2
    args[12].AsBool(false), // planar
    args[13].AsBool(true),  // mt
    env
  );
}

AVSValue __cdecl Create_MVSuper(AVSValue args, void*, IScriptEnvironment* env)
{
  return new MVSuper(
    args[0].AsClip(),      // source
    args[1].AsInt(8),      // hpad
    args[2].AsInt(8),      // vpad
    args[3].AsInt(2),      // pel
    args[4].AsInt(0),      // levels
    args[5].AsBool(true),  // chroma
    args[6].AsInt(2),      // sharp
    args[7].AsInt(2),      // rfilter
    (args[8].Defined() ? args[8].AsClip() : 0), // pelclip
    args[9].AsBool(true),   // isse2
    args[10].AsBool(false), // planar
    args[11].AsBool(true), // mt
    args[12].AsBool(true), // pelrefine
    env
  );
}

AVSValue __cdecl Create_MVFinest(AVSValue args, void*, IScriptEnvironment* env)
{
  return new MVFinest(
    args[0].AsClip(),    // super
    args[1].AsInt(true), // isse2
    env
  );
}

AVSValue __cdecl Create_MStoreVect(AVSValue args, void*, IScriptEnvironment* env_ptr)
{
  const int		nbr_clips = args[0].ArraySize();
  if (nbr_clips < 1)
  {
    env_ptr->ThrowError("MStoreVect: you must specify at least one vector clip.");
  }

  std::vector <::PClip>	vect_arr;
  for (int clip_cnt = 0; clip_cnt < nbr_clips; ++clip_cnt)
  {
    ::PClip			clip_ptr = args[0][clip_cnt].AsClip();
    vect_arr.push_back(clip_ptr);
  }

  return new MStoreVect(
    vect_arr,               // vectors
    args[1].AsString(""), // vccs
    env_ptr
  );
}

AVSValue __cdecl Create_MAverage(AVSValue args, void*, IScriptEnvironment* env_ptr)
{
  const int		nbr_clips = args[0].ArraySize();
  if (nbr_clips < 1)
  {
    env_ptr->ThrowError("MAverage: you must specify at least one vector clip.");
  }

  std::vector <::PClip>	vect_arr;
  for (int clip_cnt = 0; clip_cnt < nbr_clips; ++clip_cnt)
  {
    ::PClip			clip_ptr = args[0][clip_cnt].AsClip();
    vect_arr.push_back(clip_ptr);
  }

  return new MAverage(
    vect_arr,               // vectors
    args[1].AsInt(0), // mode
    env_ptr
  );
}

AVSValue __cdecl Create_MTransform(AVSValue args, void*, IScriptEnvironment* env_ptr)
{
  return new MTransform(
    args[0].AsClip(), // mvclip
    args[1].AsInt(0), // mode
    env_ptr
  );
}

AVSValue __cdecl Create_MRestoreVect(AVSValue args, void*, IScriptEnvironment* env_ptr)
{
  return new MRestoreVect(
    args[0].AsClip(), // wrapped vectors
    args[1].AsInt(0), // index
    env_ptr
  );
}

AVSValue __cdecl Create_MScaleVect(AVSValue args, void*, IScriptEnvironment* env)
{
  enum { CLIP, SCALE, SCALEV, MODE, FLIP, ADJUSTSUBPEL, BITS };
  int bits = args[BITS].AsInt(0);
  float scaleX = float(args[SCALE].AsFloat(bits == 0 ? 2.0f : 1.0f));
  float scaleY = float(args[SCALEV].AsFloat(scaleX));
  return new MScaleVect(args[CLIP].AsClip(),
    scaleX, scaleY,
    static_cast<MScaleVect::ScaleMode>(args[MODE].AsInt(0)),
    args[FLIP].AsBool(scaleX < 0 && scaleX == scaleY), // Default flip if new direction is exactly reversed
    args[ADJUSTSUBPEL].AsBool(false),
    bits,
    env);
}


#ifdef AVISYNTH_PLUGIN_25
extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env) {
#else
/* New 2.6 requirement!!! */
// Declare and initialise server pointers static storage.
const AVS_Linkage *AVS_linkage = 0;

/* New 2.6 requirement!!! */
// DLL entry point called from LoadPlugin() to setup a user plugin.
extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {

  /* New 2.6 requirment!!! */
  // Save the server pointers.
  AVS_linkage = vectors;
#endif
  env->AddFunction("MShow", "cc[scale]i[sil]i[tol]i[showsad]b[number]i[thSCD1]i[thSCD2]i[isse]b[planar]b", Create_MVShow, 0);
  env->AddFunction("MAnalyse", "c[blksize]i[blksizeV]i[levels]i[search]i[searchparam]i[pelsearch]i[isb]b[lambda]i[chroma]b[delta]i[truemotion]b[lsad]i[plevel]i[global]b[pnew]i[pzero]i[pglobal]i[overlap]i[overlapV]i[outfile]s[dct]i[divide]i[sadx264]i[badSAD]i[badrange]i[isse]b[meander]b[temporal]b[trymany]b[multi]b[mt]b[scaleCSAD]i[optsearchoption]i[optpredictortype]i[scaleCSADfine]f[accnum]i[UseSubShift]i[SuperCurrent]c[SearchDirMode]i[DMFlags]i[AreaMode]i[AMdiffSAD]i[AMstep]i[AMoffset]i[AMpel]i[PTpel]i[AMflags]i[AMavg]i[AMpt]i[AMst]i[AMsp]i[tmavg]i[mdp]i[vs]i", Create_MVAnalyse, 0);
  env->AddFunction("MMask", "cc[ml]f[gamma]f[kind]i[time]f[Ysc]i[thSCD1]i[thSCD2]i[isse]b[planar]b", Create_MVMask, 0);
  env->AddFunction("MCompensate", "ccc[scbehavior]b[recursion]f[thSAD]i[fields]b[time]f[thSCD1]i[thSCD2]i[isse]b[planar]b[mt]b[tr]i[center]b[cclip]c[thSAD2]i[showRNB]b", Create_MVCompensate, 0);
  env->AddFunction("MSCDetection", "cc[Ysc]i[thSCD1]i[thSCD2]i[isse]b", Create_MVSCDetection, 0);
  env->AddFunction("MDepan", "cc[mask]c[zoom]b[rot]b[pixaspect]f[error]f[info]b[log]s[wrong]f[zerow]f[range]i[thSCD1]i[thSCD2]i[isse]b[planar]b", Create_MVDepan, 0);
  env->AddFunction("MFlow", "ccc[time]f[mode]i[fields]b[thSCD1]i[thSCD2]i[isse]b[planar]b[tclip]c", Create_MVFlow, 0);
  env->AddFunction("MFlowInter", "cccc[time]f[ml]f[blend]b[thSCD1]i[thSCD2]i[isse]b[planar]b[tclip]c", Create_MVFlowInter, 0);
  env->AddFunction("MFlowFps", "cccc[num]i[den]i[mask]i[ml]f[blend]b[thSCD1]i[thSCD2]i[isse]b[planar]b[optDebug]i", Create_MVFlowFps, 0);
  env->AddFunction("MFlowBlur", "cccc[blur]f[prec]i[thSCD1]i[thSCD2]i[isse]b[planar]b", Create_MVFlowBlur, 0);
  env->AddFunction("MDegrain1", "cccc[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[mt]b[out16]b[out32]b", Create_MVDegrainX, (void *)1);
  env->AddFunction("MDegrain2", "cccccc[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[mt]b[out16]b[out32]b", Create_MVDegrainX, (void *)2);
  env->AddFunction("MDegrain3", "cccccccc[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[mt]b[out16]b[out32]b", Create_MVDegrainX, (void *)3);
  env->AddFunction("MDegrain4", "cccccccccc[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[mt]b[out16]b[out32]b", Create_MVDegrainX, (void *)4);
  env->AddFunction("MDegrain5", "cccccccccccc[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[mt]b[out16]b[out32]b", Create_MVDegrainX, (void *)5);
  env->AddFunction("MDegrain6", "cccccccccccccc[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[mt]b[out16]b[out32]b", Create_MVDegrainX, (void *)6);
  env->AddFunction("MDegrainN", "ccci[thSAD]i[thSADC]i[plane]i[limit]f[limitC]f[thSCD1]i[thSCD2]i[isse]b[planar]b[lsb]b[thsad2]i[thsadc2]i[mt]b[out16]b[wpow]i[adjSADzeromv]f[adjSADcohmv]f[thCohMV]i[MVLPFCutoff]f[MVLPFSlope]f[MVLPFGauss]f[thMVLPFCorr]i[adjSADLPFedmv]f[UseSubShift]i[IntOvlp]i[mvmultirs]c[thFWBWmvpos]i[MPBthSub]i[MPBthAdd]i[MPBNumIt]i[MPB_SPCsub]f[MPB_SPCadd]f[MPB_PartBlend]b[MPBthIVS]i[showIVSmask]b[mvmultivs]c[MPB_DMFlags]i[MPBchroma]i[MPBtgtTR]i[MPB_MVlth]i[pmode]i[TTH_DMFlags]i[TTH_thUPD]i[TTH_BAS]i[TTH_chroma]b[dnmask]c[thSADA_a]f[thSADA_b]f[MVMedF]i[MVMedF_em]i[MVMedF_cm]i[MVF_fm]i[MGR]i[MGR_sr]i[MGR_st]i[MGR_pm]i[LtComp]i", Create_MDegrainN, 0);
  env->AddFunction("MRecalculate", "cc[thsad]i[smooth]i[blksize]i[blksizeV]i[search]i[searchparam]i[lambda]i[chroma]b[truemotion]b[pnew]i[overlap]i[overlapV]i[outfile]s[dct]i[divide]i[sadx264]i[isse]b[meander]b[tr]i[mt]b[scaleCSAD]i[optsearchoption]i[optpredictortype]i[DMFlags]i[AreaMode]i[AMdiffSAD]i[AMstep]i[AMoffset]i[SuperCurrent]c[AMthVSMang]f[AMflags]i[AMavg]i[global]b[pzero]i[pglobal]i", Create_MVRecalculate, 0);
  env->AddFunction("MBlockFps", "cccc[num]i[den]i[mode]i[ml]f[blend]b[thSCD1]i[thSCD2]i[isse]b[planar]b[mt]b", Create_MVBlockFps, 0);
  env->AddFunction("MSuper", "c[hpad]i[vpad]i[pel]i[levels]i[chroma]b[sharp]i[rfilter]i[pelclip]c[isse]b[planar]b[mt]b[pelrefine]b", Create_MVSuper, 0);
  env->AddFunction("MStoreVect", "c+[vccs]s", Create_MStoreVect, 0);
  env->AddFunction("MRestoreVect", "c[index]i", Create_MRestoreVect, 0);
  env->AddFunction("MScaleVect", "c[scale]f[scaleV]f[mode]i[flip]b[adjustSubPel]b[bits]i", Create_MScaleVect, 0);
  //	env->AddFunction("MVFinest",     "c[isse]b", Create_MVFinest, 0);
  env->AddFunction("MAverage", "c+[mode]i", Create_MAverage, 0);
  env->AddFunction("MTransform", "c[mode]i", Create_MTransform, 0);
  return("MVTools : set of tools based on a motion estimation engine");
}


#ifdef _WIN32
static void	Interface_dll_load(::HINSTANCE hinst)
{
#if defined (_MSC_VER) && ! defined (NDEBUG) && defined (_DEBUG)
  {
    const int	mode = (1 * _CRTDBG_MODE_DEBUG)
      | (1 * _CRTDBG_MODE_WNDW);
    //		::_CrtSetReportMode (_CRT_WARN, mode);
    ::_CrtSetReportMode(_CRT_ERROR, mode);
    ::_CrtSetReportMode(_CRT_ASSERT, mode);

    const int	old_flags = ::_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    ::_CrtSetDbgFlag(old_flags
      | (1 * _CRTDBG_LEAK_CHECK_DF)
      | (0 * _CRTDBG_CHECK_ALWAYS_DF));
    ::_CrtSetBreakAlloc(-1);	// Specify here a memory bloc number
  }
#endif	// _MSC_VER, NDEBUG
}



static void	Interface_dll_unload(::HINSTANCE hinst)
{
#if defined (_MSC_VER) && ! defined (NDEBUG) && defined (_DEBUG)
  {
    const int	mode = (1 * _CRTDBG_MODE_DEBUG)
      | (0 * _CRTDBG_MODE_WNDW);
    //		::_CrtSetReportMode (_CRT_WARN, mode);
    ::_CrtSetReportMode(_CRT_ERROR, mode);
    ::_CrtSetReportMode(_CRT_ASSERT, mode);

    ::_CrtMemState	mem_state;
    ::_CrtMemCheckpoint(&mem_state);
    ::_CrtMemDumpStatistics(&mem_state);
  }
#endif	// _MSC_VER, NDEBUG
}


BOOL WINAPI DllMain(::HINSTANCE hinst, ::DWORD reason, ::LPVOID reserved_ptr)
{
  switch (reason)
  {
  case	DLL_PROCESS_ATTACH:
    Interface_dll_load(hinst);
    break;

  case	DLL_PROCESS_DETACH:
    Interface_dll_unload(hinst);
    break;
  }

  return (TRUE);
}
#endif // _WIN32

