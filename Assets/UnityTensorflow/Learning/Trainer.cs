using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Math;
using MLAgents;
using System.IO;
using KerasSharp.Backends;

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

    protected Academy academyRef;
    public LearningModelBase modelRef;
    public bool isTraining;
    protected bool prevIsTraining;

    public TrainerParams parameters;
    public bool continueFromCheckpoint = true;
    public string checkpointPath = @"Assets\testcheckpoint.bytes";

    [ReadOnly]
    [SerializeField]
    private int steps = 0;


    public Brain BrainToTrain { get; private set; }

    private void Start()
    {
        academyRef = FindObjectOfType<Academy>();
        Debug.Assert(academyRef != null, "No Academy in this scene!");
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
        if (isTraining)
            modelRef.SetLearningRate(parameters.learningRate);

        if (IsReadyUpdate() && isTraining && GetStep() <= GetMaxStep())
        {
            UpdateModel();
        }
    }

    public virtual void SetBrain(Brain brain)
    {
        this.BrainToTrain = brain;
    }

    public abstract void Initialize();

    public virtual int GetMaxStep()
    {
        return parameters.maxTotalSteps;
    }

    public virtual int GetStep()
    {
        return steps;
    }
    public virtual void IncrementStep()
    {
        steps++;
        if (steps % parameters.saveModelInterval == 0)
        {
            SaveModel();
        }
    }

    public virtual void ResetTrainer()
    {
        steps = 0;
    }

    public abstract Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos);
    public abstract void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent,TakeActionOutput> actionOutput);
    public abstract void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo);
    public abstract bool IsReadyUpdate();
    public abstract void UpdateModel();



    public void SaveModel()
    {
        var data = modelRef.SaveCheckpoint();
        var fullPath = Path.GetFullPath(checkpointPath);
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        File.WriteAllBytes(fullPath, data);
        Debug.Log("Saved model checkpoint to " + fullPath);
    }
    public void LoadModel()
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        if (!File.Exists(fullPath))
        {
            Debug.Log("Model checkpoint not exist at: " + fullPath);
            return;
        }
        var bytes = File.ReadAllBytes(fullPath);
        modelRef.RestoreCheckpoint(bytes);
        Debug.Log("Model loaded  from checkpoint " + fullPath);
    }


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
                    resultTemp[h * wp + w * pixels] = currentPixel.r / 255.0f;
                    resultTemp[h * wp + w * pixels + 1] = currentPixel.g / 255.0f;
                    resultTemp[h * wp + w * pixels + 2] = currentPixel.b / 255.0f;
                }
                else
                {
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
