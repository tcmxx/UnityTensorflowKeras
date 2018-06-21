Shader "HeatMap/HeatMap5Color"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			// make fog work
			#pragma multi_compile_fog
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			float  _RedThreshold;
			float  _GreenThreshold;
			float  _BlueThreshold;
			float  _MinThreshold;

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				// sample the texture
				fixed4 col = tex2D(_MainTex, i.uv);

				float value = col.a;

			    const int NUM_COLORS = 5;
			    float3 color[NUM_COLORS] = { float3(0,0,1),float3(0,1,1), float3(0,1,0), float3(1,1,0), float3(1,0,0) };
			    // A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each.
			 
			    int idx1;        // |-- Our desired color will be between these two indexes in "color".
			    int idx2;        // |
			    float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.
			 
			    if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
			    else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
			    else
			    {
				    value = value * (NUM_COLORS-1);        // Will multiply value by 3.
				    idx1  = floor(value);                  // Our desired color will be after this index.
				    idx2  = idx1+1;                        // ... and before this index (inclusive).
				    fractBetween = value - float(idx1);    // Distance between the two indexes (0-1).
			    }
			 	
			 	float3 result = (color[idx2] - color[idx1])*fractBetween + color[idx1];
			
				col.xyz = result;

				// apply fog
				UNITY_APPLY_FOG(i.fogCoord, col);



				return col;
			}
			ENDCG
		}
	}
}
