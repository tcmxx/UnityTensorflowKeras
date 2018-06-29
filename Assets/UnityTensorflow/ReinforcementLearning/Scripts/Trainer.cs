using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Math;
using MLAgents;

public struct TakeActionOutput
{



    // public Dictionary<Agent, float[]> outputAction;
    // public Dictionary<Agent, float[]> allProbabilities; //used for RL
    //public Dictionary<Agent, float> value;//use for RL
     public float[] outputAction;
     public float[] allProbabilities; //used for RL
    public  float value;//use for RL
    //public Dictionary<Agent, float[]> memory;

    //public Dictionary<Agent, string> textAction;
}




public abstract class Trainer : MonoBehaviour
{

    public Academy academyRef;
    public bool isTraining;
    protected bool prevIsTraining;

    private void Start()
    {
        prevIsTraining = isTraining;
        academyRef.SetIsInference(!isTraining);
    }
    public virtual void Update()
    {
        if (prevIsTraining != isTraining)
        {
            prevIsTraining = isTraining;
            academyRef.SetIsInference(!isTraining);
        }
    }


    protected virtual void FixedUpdate()
    {
        if (IsReadyUpdate() && isTraining && GetStep() <= GetMaxStep())
        {
            UpdateModel();
        }
    }

    public abstract void SetBrain(Brain brain);
    public abstract void Initialize();

    public abstract int GetStep();
    public abstract int GetMaxStep();

    public abstract Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos);
    public abstract void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent,TakeActionOutput> actionOutput);
    public abstract void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo);
    public abstract bool IsReadyUpdate();
    public abstract void UpdateModel();
    public abstract void IncrementStep();





    public static float[,,] TextureToArray(Texture2D tex, bool blackAndWhite)
    {
        int width = tex.width;
        int height = tex.height;
        int pixels = 0;
        if (blackAndWhite)
            pixels = 1;
        else
            pixels = 3;
        float[,,] result = new float[ height, width, pixels];
        float[] resultTemp = new float[ height * width * pixels];
        int wp = width * pixels;

        Color32[] cc = tex.GetPixels32();
        for (int h = height-1; h >=0; h--)
        {
            for (int w = 0; w < width; w++)
            {
                Color32 currentPixel = cc[(height - h - 1) * width + w];
                if (!blackAndWhite)
                {
                    // For Color32, the r, g and b values are between
                    // 0 and 255.
                    /*result[height - h - 1, w, 0] =
                        currentPixel.r / 255.0f;
                    result[height - h - 1, w, 1] =
                        currentPixel.g / 255.0f;
                    result[height - h - 1, w, 2] =
                        currentPixel.b / 255.0f;*/
                    resultTemp[h * wp + w * pixels] = currentPixel.r / 255.0f;
                    resultTemp[h * wp + w * pixels + 1] = currentPixel.g / 255.0f;
                    resultTemp[h * wp + w * pixels + 2] = currentPixel.b / 255.0f;
                }
                else
                {
                    /*result[tex.height - h - 1, w, 0] =
                        (currentPixel.r + currentPixel.g + currentPixel.b)
                        / 3;*/
                    resultTemp[h * wp + w * pixels] =
                    (currentPixel.r + currentPixel.g + currentPixel.b)
                    / 3;
                }
            }
        }

        Buffer.BlockCopy(resultTemp, 0, result, 0, height * width * pixels * sizeof(float));
        return result;
    }
    public static List<float[,,,]> CreateVisualIInputBatch(Dictionary<Agent, AgentInfo> currentInfo, List<Agent> agentList, resolution[] cameraResolutions)
    {
        if (cameraResolutions == null || cameraResolutions.Length <= 0)
            return null;

        var observationMatrixList = new List<float[,,,]>();
        var texturesHolder = new List<Texture2D>();

        for (int observationIndex = 0; observationIndex < cameraResolutions.Length; observationIndex++)
        {
            texturesHolder.Clear();
            foreach (Agent agent in agentList)
            {
                texturesHolder.Add(currentInfo[agent].visualObservations[observationIndex]);
            }
            observationMatrixList.Add(texturesHolder.BatchVisualObservations(cameraResolutions[observationIndex].blackAndWhite));
        }

        return observationMatrixList;
    }


    public static float[,] CreateVectorIInputBatch(Dictionary<Agent, AgentInfo> currentInfo, List<Agent> agentList)
    {
        int obsSize = currentInfo[agentList[0]].stackedVectorObservation.Count;
        if(obsSize == 0)
            return null;
        var result = new float[agentList.Count,obsSize];

        int i = 0;
        foreach (Agent agent in agentList)
        {
            result.SetRow(i, currentInfo[agent].stackedVectorObservation.ToArray());
            i++;
        }

        return result;
    }


 
}
