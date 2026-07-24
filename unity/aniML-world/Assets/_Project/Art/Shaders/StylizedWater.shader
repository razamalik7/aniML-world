// Stylized water: depth-faded color (shallow teal -> deep blue), animated
// shore foam where the terrain rises to meet the surface, and a moving
// two-octave shimmer. Rivers set _FlowUV 1 and scroll along the ribbon's
// V axis so the water visibly runs downstream; lakes/ocean shimmer in
// world space. Needs the URP camera depth texture (enabled in AniML_URP).
Shader "AniML/StylizedWater"
{
    Properties
    {
        _ShallowColor("Shallow Color", Color) = (0.33, 0.66, 0.72, 0.55)
        _DeepColor("Deep Color", Color) = (0.09, 0.30, 0.44, 0.92)
        _DepthRange("Depth Fade Range (m)", Float) = 6
        _FoamColor("Foam Color", Color) = (0.95, 0.98, 1.0, 0.85)
        _FoamDepth("Foam Depth (m)", Float) = 0.6
        _WaveScale("Shimmer Scale", Float) = 0.09
        _WaveSpeed("Shimmer Speed", Float) = 0.35
        _WaveStrength("Shimmer Strength", Range(0, 0.3)) = 0.07
        _FlowUV("Use Mesh UV Flow (rivers)", Float) = 0
        _FlowSpeed("Flow Speed", Float) = 0.55
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent" "RenderPipeline" = "UniversalPipeline" }

        Pass
        {
            Name "ForwardUnlitWater"
            Tags { "LightMode" = "UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull Back

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS  : TEXCOORD1;
                float2 uv          : TEXCOORD0;
                float4 screenPos   : TEXCOORD2;
            };

            CBUFFER_START(UnityPerMaterial)
                float4 _ShallowColor;
                float4 _DeepColor;
                float4 _FoamColor;
                float _DepthRange;
                float _FoamDepth;
                float _WaveScale;
                float _WaveSpeed;
                float _WaveStrength;
                float _FlowUV;
                float _FlowSpeed;
            CBUFFER_END

            float Hash21(float2 p)
            {
                p = frac(p * float2(234.34, 435.345));
                p += dot(p, p + 34.23);
                return frac(p.x * p.y);
            }

            float ValueNoise(float2 p)
            {
                float2 i = floor(p);
                float2 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = Hash21(i);
                float b = Hash21(i + float2(1, 0));
                float c = Hash21(i + float2(0, 1));
                float d = Hash21(i + float2(1, 1));
                return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
            }

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                float3 wp = TransformObjectToWorld(IN.positionOS.xyz);
                // Gentle swell on still water; rivers keep their flat run.
                if (_FlowUV < 0.5)
                {
                    wp.y += (sin(wp.x * 0.11 + _Time.y * 1.15)
                             + cos(wp.z * 0.083 + _Time.y * 1.45)) * 0.07;
                }
                OUT.positionWS = wp;
                OUT.positionHCS = TransformWorldToHClip(wp);
                OUT.uv = IN.uv;
                OUT.screenPos = ComputeScreenPos(OUT.positionHCS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 suv = IN.screenPos.xy / IN.screenPos.w;

                // Water depth = scene eye-depth behind the surface minus the
                // surface's own eye-depth.
                float sceneEye = LinearEyeDepth(SampleSceneDepth(suv), _ZBufferParams);
                float surfEye = IN.screenPos.w;
                float depth = max(sceneEye - surfEye, 0.0);

                // Moving shimmer: rivers scroll along the ribbon, still water
                // drifts in world space.
                float2 nuv = _FlowUV > 0.5
                    ? float2(IN.uv.x * 3.0, IN.uv.y - _Time.y * _FlowSpeed)
                    : IN.positionWS.xz * _WaveScale + _Time.y * _WaveSpeed * float2(0.71, 0.43);
                float wave = (ValueNoise(nuv * 6.0) + ValueNoise(nuv * 16.3 + 71.7) * 0.5) / 1.5;

                half4 col = lerp(_ShallowColor, _DeepColor, saturate(depth / _DepthRange));
                col.rgb += (wave - 0.5) * _WaveStrength;

                // Same stylized lighting family as the terrain: sit between
                // ambient and the main light so water never blows out.
                Light mainLight = GetMainLight();
                float3 ambient = SampleSH(float3(0, 1, 0));
                col.rgb *= lerp(ambient, mainLight.color, 0.72);

                // Shore foam AFTER lighting so it stays bright — a noisy
                // band where the water is thinnest.
                float foamBand = saturate(1.0 - depth / (_FoamDepth * (0.55 + wave)));
                float foam = smoothstep(0.45, 0.8, foamBand);
                col.rgb = lerp(col.rgb, _FoamColor.rgb, foam * _FoamColor.a);
                col.a = max(col.a, foam * _FoamColor.a);

                return col;
            }
            ENDHLSL
        }
    }
}
