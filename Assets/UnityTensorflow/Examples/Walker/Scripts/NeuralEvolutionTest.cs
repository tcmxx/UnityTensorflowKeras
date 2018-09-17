using ICM;
using KerasSharp.Engine.Topology;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;
using KerasSharp.Backends;
using Accord;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

public class NeuralEvolutionTest : MonoBehaviour
{
    /*
    public TrainerPPO trainer;
    public RLModelPPOHierarchy model;
    public ESOptimizer.ESOptimizerType optimizerType;
    public int populationSize = 16;
    public OptimizationModes mode;
    public float initialStepSize = 1;

    public int evaluationSteps = 50000;
    public int evaluationLastRewardsNum = 100;

    protected IMAES optimizer;

    public string checkpointPath = @"Assets\testEvolutionData.bytes";
    public bool continueFromPrev = false;
    [ReadOnly]
    [SerializeField]
    protected int paramDimension;

    List<Tensor> tensorsToOptimize;
    List<int> tensorSizes = new List<int>();


    [ReadOnly]
    [SerializeField]
    protected int currentEvaluationIndex = 0;
    [ReadOnly]
    [SerializeField]
    protected int currentGeneration = 0;

    protected OptimizationSample[] samples;

    public OptimizationSample Best { get; private set; }

    //protected EvolutionData data;
    public bool isOptimizing = true;
    protected bool prevIsOptimizing = true;

    [Serializable]
    protected class EvolutionData
    {
        public OptimizationSample[] samples;
        public int currentEvaluationIndex = 0;
        public int currentGeneration = 0;
        public OptimizationSample best;
    }

    void Start()
    {
        tensorsToOptimize = model.networkHierarchy.GetLowLevelWeights();
        paramDimension = 0;
        foreach (var t in tensorsToOptimize)
        {
            int size = t.shape.Aggregate((t1, t2) => t1 * t2).Value;
            tensorSizes.Add(size);
            paramDimension += size;
        }


        optimizer = optimizerType == ESOptimizer.ESOptimizerType.LMMAES ? (IMAES)new LMMAES() : (IMAES)new MAES();

        samples = new OptimizationSample[populationSize];
        for (int i = 0; i < populationSize; ++i)
        {
            samples[i] = new OptimizationSample(paramDimension);
        }

        //initialize the optimizer
        optimizer.init(paramDimension, populationSize, new double[paramDimension], initialStepSize, mode);

        if (continueFromPrev)
            LoadFromFile();
        else
            optimizer.generateSamples(samples);

        SetWeights(samples[currentEvaluationIndex]);
    }

    private void Update()
    {
        if(prevIsOptimizing != isOptimizing )
        {
            prevIsOptimizing = isOptimizing;
            Current.K.try_initialize_variables(false);
            trainer.ResetTrainer();
            if (isOptimizing){
                SetWeights(samples[currentEvaluationIndex]);
            }
            else{
                SetWeights(Best);
            }
        }
    }
    private void FixedUpdate()
    {
        if (!isOptimizing)
            return;

        if (trainer.GetStep() >= evaluationSteps)
        {
            //set the objective function value to the samples
            var rewards = trainer.stats.GetStat("accumulatedRewards");
            
            float aveRewards = 0;
            for (int i = 0; i < evaluationLastRewardsNum; ++i)
            {
                aveRewards += rewards[rewards.Count - 1 - i].Item2;
            }
            aveRewards = aveRewards / evaluationLastRewardsNum;
            samples[currentEvaluationIndex].objectiveFuncVal = aveRewards;

            //reset stuff
            currentEvaluationIndex++;
            Current.K.try_initialize_variables(false);
            trainer.ResetTrainer();
            if (currentEvaluationIndex < populationSize)
            {
                SaveToFile();
                SetWeights(samples[currentEvaluationIndex]);
            }

            
        }

        if (currentEvaluationIndex >= populationSize)
        {
            optimizer.update(samples);//update the optimizer

            currentGeneration++;
            currentEvaluationIndex = 0;
            optimizer.generateSamples(samples);//generate new samples

            if (Best == null)
            {
                Best = new OptimizationSample();
                Best.x = optimizer.getBest();
                Best.objectiveFuncVal = optimizer.getBestObjectiveFuncValue();
            }
            else if ((mode == OptimizationModes.maximize && Best.objectiveFuncVal < optimizer.getBestObjectiveFuncValue()) ||
                (mode == OptimizationModes.minimize && Best.objectiveFuncVal > optimizer.getBestObjectiveFuncValue()))
            {
                Best.x = optimizer.getBest();
                Best.objectiveFuncVal = optimizer.getBestObjectiveFuncValue();
            }

            SaveToFile();

            SetWeights(samples[currentEvaluationIndex]);//set weight for the first sample

        }
    }

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



    public virtual void RestoreCheckpoint(byte[] data)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        object deserizlied = binFormatter.Deserialize(mStream);
        var restoredData = deserizlied as EvolutionData;
        if (restoredData == null)
        {
            Debug.LogError("loaded data error");
        }
        else{
            samples = restoredData.samples;
            currentGeneration = restoredData.currentGeneration;
            currentEvaluationIndex = restoredData.currentEvaluationIndex;
            Best = restoredData.best;
        }



    }

    /// <summary>
    /// save the models all parameters to a byte array
    /// </summary>
    /// <returns></returns>
    public virtual byte[] SaveCheckpoint()
    {
        EvolutionData data = new EvolutionData();

        data.samples = samples;
        data.currentEvaluationIndex = currentEvaluationIndex;
        data.currentGeneration = currentGeneration;
        data.best = Best;
        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, data);
        return mStream.ToArray();
    }


    public void SaveToFile()
    {
        var data = this.SaveCheckpoint();
        var fullPath = Path.GetFullPath(checkpointPath);
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        File.WriteAllBytes(fullPath, data);
        Debug.Log("Saved evolution childrens to " + fullPath);
    }
    public void LoadFromFile()
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        if (!File.Exists(fullPath))
        {
            Debug.Log("evolution childrens checkpoint not exist at: " + fullPath);
            return;
        }
        var bytes = File.ReadAllBytes(fullPath);
        this.RestoreCheckpoint(bytes);
        Debug.Log("evolution childrens loaded  from " + fullPath);
    }*/
}
