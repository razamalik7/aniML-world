// Terrain: 4-way texture splat (grass/dirt/rock/snow) blended by mesh
// weights, tinted by per-face biome color, smooth stylized lighting.
// Weights live in TEXCOORD1 (xyzw), tiling UVs are world-space XZ.
Shader "AniML/TerrainVertexColor"
{
    Properties
    {
        _GrassTex("Grass", 2D) = "white" {}
        _DirtTex("Dirt", 2D) = "white" {}
        _RockTex("Rock", 2D) = "white" {}
        _SnowTex("Snow", 2D) = "white" {}
        _TileScale("Texture Tiling (per meter)", Range(0.05, 2)) = 0.35
        _TintStrength("Biome Tint Strength", Range(0, 1)) = 0.55
        _BaseColor("Global Tint", Color) = (1, 1, 1, 1)
        _ToonSteps("Toon Steps (0 = smooth)", Range(0, 6)) = 0
        _ShadowStrength("Shadow Strength", Range(0, 1)) = 0.75
        _SeaLevel("Sea Level (world Y)", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _SHADOWS_SOFT

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float4 color      : COLOR;
                float4 weights    : TEXCOORD1;   // grass, dirt, rock, snow
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS    : TEXCOORD0;
                float3 positionWS  : TEXCOORD2;
                float4 color       : COLOR;
                float4 weights     : TEXCOORD1;
            };

            TEXTURE2D(_GrassTex); SAMPLER(sampler_GrassTex);
            TEXTURE2D(_DirtTex);  SAMPLER(sampler_DirtTex);
            TEXTURE2D(_RockTex);  SAMPLER(sampler_RockTex);
            TEXTURE2D(_SnowTex);  SAMPLER(sampler_SnowTex);

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float _TileScale;
                float _TintStrength;
                float _ToonSteps;
                float _ShadowStrength;
                float _SeaLevel;
            CBUFFER_END

            // Set by the runtime sky/weather systems; zero means inert, so
            // editor captures stay correct without them.
            float _AniML_Wetness;
            float _AniML_SnowLineY;
            float _AniML_BurnMaskOn;
            float _AniML_WorldSize;
            TEXTURE2D(_AniML_BurnMask); SAMPLER(sampler_AniML_BurnMask);

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                OUT.positionHCS = TransformWorldToHClip(OUT.positionWS);
                OUT.normalWS = TransformObjectToWorldNormal(IN.normalOS);
                OUT.color = IN.color;
                OUT.weights = IN.weights;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 uv = IN.positionWS.xz * _TileScale;
                float4 w = IN.weights;
                float3 n = normalize(IN.normalWS);

                // The living snowline: winter (and snow spells) pull it down
                // the slopes; gentle ground whitens above it.
                if (_AniML_SnowLineY > 0.5)
                {
                    float seasonSnow = saturate((IN.positionWS.y - _AniML_SnowLineY) / 12.0)
                                       * saturate(n.y * 1.6 - 0.35);
                    w.w = max(w.w, seasonSnow * 2.0);
                }

                float total = max(w.x + w.y + w.z + w.w, 1e-4);
                w /= total;

                float3 tex =
                    SAMPLE_TEXTURE2D(_GrassTex, sampler_GrassTex, uv).rgb * w.x +
                    SAMPLE_TEXTURE2D(_DirtTex,  sampler_DirtTex,  uv).rgb * w.y +
                    SAMPLE_TEXTURE2D(_RockTex,  sampler_RockTex,  uv).rgb * w.z +
                    SAMPLE_TEXTURE2D(_SnowTex,  sampler_SnowTex,  uv).rgb * w.w;

                // Steep faces resample rock projected along the wall
                // (triplanar-lite) so cliffs stop smearing top-down tiling.
                float steep = saturate((1.0 - n.y) * 2.2 - 0.9);
                if (steep > 0.01)
                {
                    float2 uvSide = (abs(n.x) > abs(n.z))
                        ? IN.positionWS.zy * _TileScale
                        : IN.positionWS.xy * _TileScale;
                    float3 rockSide = SAMPLE_TEXTURE2D(_RockTex, sampler_RockTex, uvSide).rgb;
                    tex = lerp(tex, rockSide, steep * saturate(w.z + w.y));
                }

                // Biome tint recolors the texture toward the palette without
                // flattening it (strength 0 = raw texture, 1 = full tint).
                float texLuma = dot(tex, float3(0.299, 0.587, 0.114));
                float3 tinted = lerp(tex, IN.color.rgb * (texLuma * 1.6 + 0.2), _TintStrength);
                float3 albedo = tinted * _BaseColor.rgb;

                // The shore stays damp: a darkened band hugging sea level,
                // fading out both up the beach and down the seabed.
                float wet = (1.0 - saturate((IN.positionWS.y - _SeaLevel - 0.12) / 0.95))
                          * saturate((IN.positionWS.y - _SeaLevel + 2.6) / 2.1);
                albedo *= lerp(1.0, 0.74, saturate(wet) * 0.9);

                // Rain-wet ground, driven by the weather system.
                albedo *= lerp(1.0, 0.78, saturate(_AniML_Wetness) * 0.85);

                // Live wildfire char, written by the wildfire system.
                if (_AniML_BurnMaskOn > 0.5)
                {
                    float2 buv = IN.positionWS.xz / max(_AniML_WorldSize, 1.0) + 0.5;
                    float burn = SAMPLE_TEXTURE2D(_AniML_BurnMask, sampler_AniML_BurnMask, buv).r;
                    albedo = lerp(albedo, float3(0.13, 0.11, 0.10), saturate(burn) * 0.85);
                }
                float4 shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
                Light mainLight = GetMainLight(shadowCoord);

                float ndl = saturate(dot(n, mainLight.direction));
                float lit = ndl;
                if (_ToonSteps > 0.5)
                {
                    lit = floor(ndl * _ToonSteps) / max(_ToonSteps - 1.0, 1.0);
                    lit = saturate(lit);
                }

                float shadow = lerp(1.0, mainLight.shadowAttenuation, _ShadowStrength);
                float3 ambient = SampleSH(n);
                float3 lighting = lerp(ambient, mainLight.color, lit * shadow);
                return half4(albedo * lighting, 1.0);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On
            ZTest LEqual
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            float3 _LightDirection;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                float3 positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(IN.normalOS);
                positionWS = ApplyShadowBias(positionWS, normalWS, _LightDirection);
                OUT.positionHCS = TransformWorldToHClip(positionWS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }
    }
}
