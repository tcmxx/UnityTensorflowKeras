using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DataPlane2D : MonoBehaviour {

    public Shader dataPointShader;
    public Texture2D dataPointTexture;
    public Mesh dataPointMesh;
    public float drawScale = 0.2f;
    public string shaderColorVarName = "_Color";
    public string shaderTextureVarName = "_MainTex";
    public LayerMask drawLayer;
    public Dictionary<int, List<Vector2>> dataset;

    public Color[] dataTypeColors;

    //public TestSeqNNTrain2D network;

    protected List<Material> mats;
    
	// Use this for initialization
	void Start () {
        InitializeDataPlane();

    }
	
	// Update is called once per frame
	void Update () {
        RenderAll();

    }


    public void RenderAll()
    {
        for(int i = 0; i < dataset.Keys.Count; ++i)
        {
            var points = dataset[i];
            foreach(var p in points)
            {
                Matrix4x4 mat = Matrix4x4.TRS(new Vector3(p.x, p.y, transform.position.z), Quaternion.identity, Vector3.one*drawScale);
                Graphics.DrawMesh(dataPointMesh, mat, mats[i], drawLayer);
            }
        }
    }

    public void AddDatapoint(Vector2 position, int type)
    {
        if(0 <= type && dataTypeColors.Length > type)
            dataset[type].Add(position);
        else
        {
            //float pred = network.EvalPosition(position);
            //dataset[Mathf.Clamp(Mathf.RoundToInt(pred),0,dataTypeColors.Length-1)].Add(position);
        }
    }
    public void RemovePointsOfType(int type)
    {
        dataset[type] = new List<Vector2>();
    }

    public void InitializeDataPlane()
    {
        dataset = new Dictionary<int, List<Vector2>>();
        mats = new List<Material>();

        for (int i = 0; i < dataTypeColors.Length;++i)
        {
            dataset[i] = new List<Vector2>();
            var newMat = new Material(dataPointShader);
            newMat.SetColor(shaderColorVarName, dataTypeColors[i]);
            newMat.SetTexture(shaderTextureVarName, dataPointTexture);
            mats.Add(newMat);
        }
        
    }

    public float[] GetDataLabels()
    {
        int count = 0;
        foreach (var v in dataset)
        {
            count += v.Value.Count;
        }
        var result = new float[count];

        int i = 0;
        foreach (var v in dataset)
        {
            foreach (var p in v.Value)
            {
                result[i] = v.Key;
                ++i;
            }
        }
        return result;
    }

    public float[] GetDataPositions()
    {
        int count = 0;
        foreach(var v in dataset)
        {
            count += v.Value.Count;
        }
        var result = new float[count*2];

        int i = 0;
        foreach (var v in dataset)
        {
            foreach(var p in v.Value)
            {
                result[i * 2] = p.x;
                result[i * 2 + 1] = p.y;
                ++i;
            }
        }
        return result;
    }




    public void Generate1DSeperationData(float slope)
    {
        for(int i = 0; i < 5000; ++i)
        {
            Vector2 pos = new Vector2(Random.Range(-1.0f, 1.0f), Random.Range(-1.0f, 1.0f)) * 5;
            if(pos.x*slope - pos.y >= 0)
            {
                AddDatapoint(pos, 0);
            }
            else
            {
                AddDatapoint(pos, 1);
            }
        }
    }


    public void GenerateCenterCircleSeperationData(float radius)
    {
        for (int i = 0; i < 5000; ++i)
        {
            Vector2 pos = new Vector2(Random.Range(-1.0f, 1.0f), Random.Range(-1.0f, 1.0f)) * 5;
            if (pos.magnitude >= radius)
            {
                AddDatapoint(pos, 0);
            }
            else
            {
                AddDatapoint(pos, 1);
            }
        }
    }
    public void Generate4GuassianData(float s)
    {
        for (int i = 0; i < 1000; ++i)
        {
            Vector2 pos = new Vector2(MathUtils.NextGaussianFloat()* s, MathUtils.NextGaussianFloat() * s);
            AddDatapoint(pos +  new Vector2(-2,-2), 0);
        }
        for (int i = 0; i < 1000; ++i)
        {
            Vector2 pos = new Vector2(MathUtils.NextGaussianFloat() * s, MathUtils.NextGaussianFloat() * s);
            AddDatapoint(pos + new Vector2(2, -2), 0);
        }
        for (int i = 0; i < 1000; ++i)
        {
            Vector2 pos = new Vector2(MathUtils.NextGaussianFloat() * s, MathUtils.NextGaussianFloat() * s);
            AddDatapoint(pos + new Vector2(2, 2), 0);
        }
        for (int i = 0; i < 1000; ++i)
        {
            Vector2 pos = new Vector2(MathUtils.NextGaussianFloat() * s, MathUtils.NextGaussianFloat() * s);
            AddDatapoint(pos + new Vector2(-2, 2), 0);
        }
    }
}
