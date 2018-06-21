using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshRenderer))]
public class HeatMap : MonoBehaviour
{
    protected MeshRenderer mRenderer;
    protected Material mat;

    protected Texture2D tex;

    public Material heatMapMaterial;
    public Vector2Int textureDimension;

    public float[,] ValueData { get; private set; }


    private void Awake()
    {
        mRenderer = GetComponent<MeshRenderer>();
        mat = new Material(heatMapMaterial);
        mRenderer.material = mat;
        CreateHeatMapTexture(textureDimension.x, textureDimension.y);
        mat.SetTexture("_MainTex", tex);
    }




    protected void CreateHeatMapTexture(int width, int height)
    {
        if (tex != null)
        {
            Destroy(tex);
        }
        tex = new Texture2D(width, height, TextureFormat.ARGB32, false);

        tex.SetPixels(new Color[width * height], 0);

        tex.Apply();

        ValueData = new float[height, width];
    }

    public void UpdateHeatMapTexture(float[,] data)
    {
        UpdateTextureAlpha(tex, data);

        data.CopyTo(ValueData, 0);
    }





    /// <summary>
    /// 
    /// </summary>
    /// <param name="sampleFunction">sample function to get the heat map value. value = function(x,y). x,y will be between 0 and 1</param>
    public void StartSampling(Func<float, float, float> sampleFunction, int samplePerLoop = 1, int strideLevel = 3, Action onSamplingDone = null)
    {
        StopAllCoroutines();
        //StartCoroutine(SamplingCoroutine(sampleFunction, samplePerLoop, strideLevel));

        StartCoroutine(SamplingBatchCoroutine((x, y) =>
        {
            var result = new List<float>();
            for (int i = 0; i < x.Count; ++i)
            {
                result.Add(sampleFunction(x[i], y[i]));
            }
            return result;
        },1,samplePerLoop,strideLevel, onSamplingDone));
    }

    public void StartSampling(Func<List<float>, List<float>, List<float>> sampleBatchFunction, int batchsize, int samplePerLoop = 1, int strideLevel = 3, Action onSamplingDone = null)
    {
        StopAllCoroutines();
        StartCoroutine(SamplingBatchCoroutine(sampleBatchFunction, batchsize, samplePerLoop, strideLevel, onSamplingDone));

    }


    protected class SampleData
    {

        public float x, y;
        public List<Vector2Int> outputCoords = new List<Vector2Int>();
    }

    protected void SampleBatchAndAssignValue(List<SampleData> datas, Func<List<float>, List<float>, List<float>> sampleBatchFunction)
    {
        List<float> x = new List<float>();
        List<float> y = new List<float>();

        foreach (var d in datas)
        {
            x.Add(d.x);
            y.Add(d.y);
        }

        var result = sampleBatchFunction(x, y);
        Debug.Assert(result.Count == datas.Count, "sampleBatchFunction output size different from input size");

        for (int i = 0; i < result.Count; ++i)
        {
            var data = datas[i];
            foreach (var coord in data.outputCoords)
            {
                ValueData[coord.x, coord.y] = result[i];
            }
        }
    }

    protected IEnumerator SamplingBatchCoroutine(Func<List<float>, List<float>, List<float>> sampleBatchFunction, int batchsize, int samplePerLoop = 1, int strideLevel = 3, Action onSamplingDone = null)
    {

        int sampleCount = 0;
        int batchCount = 0;

        List<SampleData> batchSamples = new List<SampleData>();

        int initialStride = Mathf.RoundToInt(Mathf.Pow(2, strideLevel));

        for (int y = 0; y < textureDimension.y; y += initialStride)
        {
            for (int x = 0; x < textureDimension.x; x += initialStride)
            {
                SampleData data = new SampleData();
                data.x = (x + 0.5f) / textureDimension.x;
                data.y = (y + 0.5f) / textureDimension.y;

                for (int i = 0; i < initialStride; ++i)
                {
                    for (int j = 0; j < initialStride; ++j)
                    {
                        if(y + i < textureDimension.y && x + j < textureDimension.x)
                            data.outputCoords.Add(new Vector2Int(y + i, x + j));
                    }
                }
                batchSamples.Add(data);
                batchCount++;
                if (batchCount >= batchsize)
                {
                    SampleBatchAndAssignValue(batchSamples, sampleBatchFunction);
                    batchSamples.Clear();
                    batchCount = 0;
                    sampleCount++;
                    if (sampleCount >= samplePerLoop)
                    {
                        UpdateTextureAlpha(tex, ValueData);
                        yield return new WaitForEndOfFrame();
                        sampleCount = 0;
                    }
                }

            }
        }


        int offset = initialStride / 2;
        int stride = initialStride;

        while (offset > 0)
        {
            for (int off = 0; off < 3; ++off)
            {
                int yoff = ((off == 0) ? 0 : offset);
                int xoff = ((off == 1) ? 0 : offset);

                for (int y = yoff; y < textureDimension.y; y += stride)
                {
                    for (int x = xoff; x < textureDimension.x; x += stride)
                    {
                        SampleData data = new SampleData();
                        data.x = (x + 0.5f) / textureDimension.x;
                        data.y = (y + 0.5f) / textureDimension.y;

                        for (int i = 0; i < offset; ++i)
                        {
                            for (int j = 0; j < offset; ++j)
                            {
                                if (y + i < textureDimension.y && x + j < textureDimension.x)
                                    data.outputCoords.Add(new Vector2Int(y + i, x + j));
                            }
                        }
                        batchSamples.Add(data);
                        batchCount++;
                        if (batchCount >= batchsize)
                        {
                            SampleBatchAndAssignValue(batchSamples, sampleBatchFunction);
                            batchSamples.Clear();
                            batchCount = 0;
                            sampleCount++;
                            if (sampleCount >= samplePerLoop)
                            {
                                UpdateTextureAlpha(tex, ValueData);
                                yield return new WaitForEndOfFrame();
                                sampleCount = 0;
                            }
                        }
                    }
                }
            }



            offset = offset / 2;
            stride = stride / 2;
        }
        if (onSamplingDone != null)
            onSamplingDone.Invoke();
        Debug.Log("Heatmap Sampling Done");
    }


    /*protected IEnumerator SamplingCoroutine(Func<float, float, float> sampleFunction, int samplePerLoop = 1, int strideLevel = 3)
    {

        int sampleCount = 0;

        int initialStride = Mathf.RoundToInt(Mathf.Pow(2, strideLevel));

        for (int y = 0; y < textureDimension.y; y += initialStride)
        {
            for (int x = 0; x < textureDimension.x; x += initialStride)
            {
                float value = sampleFunction((x + 0.5f) / textureDimension.x, (y + 0.5f) / textureDimension.y);
                for (int i = 0; i < initialStride; ++i)
                {
                    for (int j = 0; j < initialStride; ++j)
                    {
                        ValueData[y + i, x + j] = value;
                    }
                }

                sampleCount++;
                if (sampleCount >= samplePerLoop)
                {
                    UpdateTextureAlpha(tex, ValueData);
                    yield return new WaitForEndOfFrame();
                    sampleCount = 0;
                }
            }
        }


        int offset = initialStride / 2;
        int stride = initialStride;
        while (offset > 0)
        {
            for (int y = offset; y < textureDimension.y; y += stride)
            {
                for (int x = offset; x < textureDimension.x; x += stride)
                {
                    float value = sampleFunction((x + 0.5f) / textureDimension.x, (y + 0.5f) / textureDimension.y);
                    for (int i = 0; i < offset; ++i)
                    {
                        for (int j = 0; j < offset; ++j)
                        {
                            ValueData[y + i, x + j] = value;
                        }
                    }

                    sampleCount++;
                    if (sampleCount >= samplePerLoop)
                    {
                        UpdateTextureAlpha(tex, ValueData);
                        yield return new WaitForEndOfFrame();
                        sampleCount = 0;
                    }
                }
            }

            for (int y = 0; y < textureDimension.y; y += stride)
            {
                for (int x = offset; x < textureDimension.x; x += stride)
                {
                    float value = sampleFunction((x + 0.5f) / textureDimension.x, (y + 0.5f) / textureDimension.y);
                    for (int i = 0; i < offset; ++i)
                    {
                        for (int j = 0; j < offset; ++j)
                        {
                            ValueData[y + i, x + j] = value;
                        }
                    }

                    sampleCount++;
                    if (sampleCount >= samplePerLoop)
                    {
                        UpdateTextureAlpha(tex, ValueData);
                        yield return new WaitForEndOfFrame();
                        sampleCount = 0;
                    }
                }
            }

            for (int y = offset; y < textureDimension.y; y += stride)
            {
                for (int x = 0; x < textureDimension.x; x += stride)
                {
                    float value = sampleFunction((x + 0.5f) / textureDimension.x, (y + 0.5f) / textureDimension.y);
                    for (int i = 0; i < offset; ++i)
                    {
                        for (int j = 0; j < offset; ++j)
                        {
                            ValueData[y + i, x + j] = value;
                        }
                    }

                    sampleCount++;
                    if (sampleCount >= samplePerLoop)
                    {
                        UpdateTextureAlpha(tex, ValueData);
                        yield return new WaitForEndOfFrame();
                        sampleCount = 0;
                    }
                }
            }


            offset = offset / 2;
            stride = stride / 2;
        }

        Debug.Log("Heatmap Sampling Done");
    }*/






    private void OnDestroy()
    {
        if (tex != null)
        {
            Destroy(tex);
        }
    }





    public static void UpdateTextureAlpha(Texture2D tex, float[,] data)
    {
        int texHeight = tex.height;
        int texWidth = tex.width;

        Debug.Assert(texHeight == data.GetLength(0) && texWidth == data.GetLength(1), "data and texture dimension not the same.");

        Color[] colors = new Color[data.Length];
        for (int i = 0; i < data.GetLength(0); ++i)
        {
            for (int j = 0; j < data.GetLength(1); ++j)
            {
                colors[i * texWidth + j] = new Color(0, 0, 0, data[i, j]);
            }
        }
        tex.SetPixels(colors, 0);
        tex.Apply();
    }
}
