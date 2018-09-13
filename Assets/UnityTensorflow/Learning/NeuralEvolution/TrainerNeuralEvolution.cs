using Accord.Math;
using ICM;
using KerasSharp.Backends;
using KerasSharp.Engine.Topology;
using MLAgents;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

public class TrainerNeuralEvolution : Trainer
{

    /// Reference to the brain that uses this CoreBrainInternal
    protected INeuralEvolutionModel modeNE;
    protected TrainerParamsNeuralEvolution parametersNE;

    protected OptimizationSample[] samples;
    protected OptimizationSample bestSample;
    protected IMAES optimizer;

    public string evolutionDataSaveFileName = @"NEData.bytes";

    [ReadOnly]
    [SerializeField]
    protected int currentEvaluationIndex = 0;
    [ReadOnly]
    [SerializeField]
    protected int currentGeneration = 0;

    [ReadOnly]
    [SerializeField]
    protected int paramDimension;
    List<Tensor> tensorsToOptimize;
    List<int> tensorSizes = new List<int>();

    [Serializable]
    protected class EvolutionData
    {
        public OptimizationSample[] samples;
        public int currentEvaluationIndex = 0;
        public int currentGeneration = 0;
        public OptimizationSample bestSample;
    }

    protected Dictionary<Agent, List<float>> agentsRewards = null;
    protected List<float> rewardsOfCurrentChild = null;

    public StatsLogger stats { get; protected set; }





    public override void Update()
    {
        if (prevIsTraining != isTraining)
        {
            prevIsTraining = isTraining;
            academyRef.SetIsInference(!isTraining);
            if (isTraining)
            {
                SetWeights(samples[currentEvaluationIndex]);
            }
            else
            {
                SetWeights(bestSample);
            }

            var agentList = agentsRewards.Keys;
            foreach (var agent in agentList)
            {
                agent.AgentReset();
                agentsRewards[agent].Clear();
            }
        }
    }



    /// Create the reference to the brain
    public override void Initialize()
    {
        modeNE = modelRef as INeuralEvolutionModel;
        Debug.Assert(modeNE != null, "Please assign a INeuralEvolutionModel to modelRef");
        parametersNE = parameters as TrainerParamsNeuralEvolution;
        Debug.Assert(parametersNE != null, "Please Specify TrainerNeuralEvolution Trainer Parameters");


        modelRef.Initialize(BrainToTrain.brainParameters, isTraining, parameters);

        agentsRewards = new Dictionary<Agent, List<float>>();
        rewardsOfCurrentChild = new List<float>();


        tensorsToOptimize = modeNE.GetWeightsForNeuralEvolution();
        paramDimension = 0;
        foreach (var t in tensorsToOptimize)
        {
            int size = t.shape.Aggregate((t1, t2) => t1 * t2).Value;
            tensorSizes.Add(size);
            paramDimension += size;
        }


        optimizer = parametersNE.optimizerType == ESOptimizer.ESOptimizerType.LMMAES ? (IMAES)new LMMAES() : (IMAES)new MAES();

        samples = new OptimizationSample[parametersNE.populationSize];
        for (int i = 0; i < parametersNE.populationSize; ++i)
        {
            samples[i] = new OptimizationSample(paramDimension);
        }

        //initialize the optimizer
        optimizer.init(paramDimension, parametersNE.populationSize, new double[paramDimension], parametersNE.initialStepSize, parametersNE.mode);

        if (continueFromCheckpoint)
        {
            if (!LoadNEDataFromFile())
            {
                optimizer.generateSamples(samples);
            }
        }
        else
            optimizer.generateSamples(samples);
        if(isTraining)
            SetWeights(samples[currentEvaluationIndex]);
        else if(bestSample != null)
            SetWeights(bestSample);

        stats = new StatsLogger();
    }


    public override Dictionary<Agent, TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();

        var agentList = new List<Agent>(agentInfos.Keys);

        float[,] vectorObsAll = CreateVectorInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);

        float[,] actions = null;
        actions = modeNE.EvaluateAction(vectorObsAll, visualObsAll);

        int i = 0;
        foreach (var agent in agentList)
        {
            var info = agentInfos[agent];
            //use result from neural network directly
            var tempAction = new TakeActionOutput();
            tempAction.outputAction = actions.GetRow(i);
            result[agent] = tempAction;
            i++;
        }

        return result;
    }






    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent, TakeActionOutput> actionOutput)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            if (!agentsRewards.ContainsKey(agent))
            {
                agentsRewards[agent] = new List<float>();
            }
            agentsRewards[agent].Add(newInfo[agent].reward);

        }
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            var agentNewInfo = newInfo[agent];
            if (agentNewInfo.done || agentNewInfo.maxStepReached || agentsRewards[agent].Count > parametersNE.timeHorizon)
            {
                //update process the episode data for PPO.
                float accumulatedRewards = agentsRewards[agent].Sum();
                rewardsOfCurrentChild.Add(accumulatedRewards);

                agentsRewards[agent].Clear();
            }
        }
    }

    public override bool IsReadyUpdate()
    {
        return rewardsOfCurrentChild.Count >= parametersNE.sampleCountForEachChild;
    }

    public override void UpdateModel()
    {

        float aveRewards = 0;
        for (int i = 0; i < rewardsOfCurrentChild.Count; ++i)
        {
            aveRewards += rewardsOfCurrentChild[i];
        }
        aveRewards = aveRewards / rewardsOfCurrentChild.Count;
        rewardsOfCurrentChild.Clear();
        stats.AddData("accumulatedRewards", aveRewards);


        samples[currentEvaluationIndex].objectiveFuncVal = aveRewards;

        
        currentEvaluationIndex++;

        //reset stuff
        if (currentEvaluationIndex < parametersNE.populationSize)
        {
            SetWeights(samples[currentEvaluationIndex]);
        }
        
        if (currentEvaluationIndex >= parametersNE.populationSize)
        {


            optimizer.update(samples);//update the optimizer

            currentGeneration++;
            currentEvaluationIndex = 0;
            optimizer.generateSamples(samples);//generate new samples

            if (bestSample == null)
            {
                bestSample = new OptimizationSample();
                bestSample.x = optimizer.getBest();
                bestSample.objectiveFuncVal = optimizer.getBestObjectiveFuncValue();
            }
            else if((parametersNE.mode == OptimizationModes.maximize && bestSample.objectiveFuncVal< optimizer.getBestObjectiveFuncValue()) ||
                (parametersNE.mode == OptimizationModes.minimize && bestSample.objectiveFuncVal > optimizer.getBestObjectiveFuncValue()))
            {
                bestSample.x = optimizer.getBest();
                bestSample.objectiveFuncVal = optimizer.getBestObjectiveFuncValue();
            }
            SetWeights(samples[currentEvaluationIndex]);//set weight for the first sample
        }

        //reset all agents
        var agentList = agentsRewards.Keys;
        foreach(var agent in agentList)
        {
            agent.AgentReset();
            agentsRewards[agent].Clear();
        }

        
    }
    public override void IncrementStep()
    {
        steps++;
        if (steps % parameters.saveModelInterval == 0)
        {
            SaveModel();
            SaveNEDataToFile();
        }
    }

    public override void ResetTrainer()
    {
        return;
    }







    /// <summary>
    /// set the weights using an optimizatoin sample
    /// </summary>
    /// <param name="sample"></param>
    protected void SetWeights(OptimizationSample sample)
    {
        float[] floatValues = Array.ConvertAll(sample.x, (t) => (float)t);
        int currentStartIndex = 0;
        for (int i = 0; i < tensorsToOptimize.Count; ++i)
        {
            Current.K.set_value(tensorsToOptimize[i], SubArray(floatValues, currentStartIndex, tensorSizes[i]));
            currentStartIndex += tensorSizes[i];
        }
    }

    /// <summary>
    /// Get current weights as an array of floats
    /// </summary>
    /// <returns></returns>
    protected float[] GetWeights()
    {
        float[] allWeights = new float[paramDimension];
        int currentStartIndex = 0;
        for (int i = 0; i < tensorsToOptimize.Count; ++i)
        {
            Array w = (Array)tensorsToOptimize[i].eval();
            Array.Copy(w, 0, allWeights, currentStartIndex, tensorSizes[i]);
            currentStartIndex += tensorSizes[i];
        }
        return allWeights;
    }

    protected static T[] SubArray<T>(T[] data, int index, int length)
    {
        T[] result = new T[length];
        Array.Copy(data, index, result, 0, length);
        return result;
    }

    /// <summary>
    /// Restore the training checkpoint from byte array. . Use <see cref="SaveNECheckpoint()"/> to restore from it.
    /// </summary>
    /// <param name="data"></param>
    public virtual void RestoreNECheckpoint(byte[] data)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var restoredData = (EvolutionData)binFormatter.Deserialize(mStream);

        samples = restoredData.samples;
        currentGeneration = restoredData.currentGeneration;
        currentEvaluationIndex = restoredData.currentEvaluationIndex;
        bestSample = restoredData.bestSample;
    }

    /// <summary>
    /// save the current training data to a byte array
    /// </summary>
    /// <returns>the byte array that represend the curren training data. Use <see cref="RestoreCheckpoint(byte[] data)"/> to restore from it.</returns>
    public virtual byte[] SaveNECheckpoint()
    {
        EvolutionData data = new EvolutionData();

        data.samples = samples;
        data.currentEvaluationIndex = currentEvaluationIndex;
        data.currentGeneration = currentGeneration;
        data.bestSample = bestSample;

        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, data);
        return mStream.ToArray();
    }

    /// <summary>
    /// save the checkpoint data to the path specified by checkpointPath field.
    /// </summary>
    public void SaveNEDataToFile()
    {
        var data = this.SaveNECheckpoint();
        string fullPath = Path.GetFullPath(Path.Combine(checkpointPath, evolutionDataSaveFileName));
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        File.WriteAllBytes(fullPath, data);
        Debug.Log("Saved Neural Evolution data to " + fullPath);
    }

    /// <summary>
    /// load the checkpoint data to the path specified by checkpointPath field .
    /// </summary>
    /// /// <returns>Whether loaded successfully</returns>
    public bool LoadNEDataFromFile()
    {
        string fullPath = Path.GetFullPath(Path.Combine(checkpointPath, evolutionDataSaveFileName));
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        if (!File.Exists(fullPath))
        {
            Debug.Log("Neural Evolution data checkpoint not exist at: " + fullPath);
            return false;
        }
        var bytes = File.ReadAllBytes(fullPath);
        this.RestoreNECheckpoint(bytes);
        Debug.Log("Neural Evolution data loaded  from " + fullPath);
        return true;
    }
}
