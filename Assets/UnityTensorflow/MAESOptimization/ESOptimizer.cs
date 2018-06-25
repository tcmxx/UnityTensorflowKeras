using ICM;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ESOptimizer : MonoBehaviour
{

    public int iterationPerUpdate = 10;
    public int populationSize = 16;
    public ESOptimizerType optimizerType;
    public int intialStepSize = 1;
    public OptimizationModes mode;
    public int maxIteration = 100;
    public double targetValue = 2;

    public int evaluationBatchSize = 1;

    protected IESOptimizable optimizable = null;

    [ReadOnly]
    [SerializeField]
    protected int iteration;
    protected OptimizationSample[] samples;
    protected IMAES optimizer;
    protected Action<double[]> onReady = null;


    public double BestScore { get; private set; }
    public double[] BestParams { get; private set; }
    public bool IsOptimizing { get; private set; } = false;

    public enum ESOptimizerType
    {
        MAES,
        LMMAES
    }


    private void Update()
    {
        if (IsOptimizing)
        {
            for (int it = 0; it < iterationPerUpdate; ++it)
            {
                optimizer.generateSamples(samples);
                for(int s = 0; s <= samples.Length/evaluationBatchSize; ++s)
                {
                    List<double[]> paramList = new List<double[]>();
                    for(int b = 0; b < evaluationBatchSize; ++b)
                    {
                        int ind = s * evaluationBatchSize + b;
                        if (ind < samples.Length)
                        {
                            paramList.Add(samples[ind].x);
                        }
                    }

                    var values = optimizable.Evaluate(paramList);

                    for (int b = 0; b < evaluationBatchSize; ++b)
                    {
                        int ind = s * evaluationBatchSize + b;
                        if (ind < samples.Length)
                        {
                            samples[ind].objectiveFuncVal = values[b];
                        }
                    }

                }
                /*foreach (OptimizationSample s in samples)
                {
                    float value = optimizable.Evaluate(new List<double[]>() { s.x })[0];
                    s.objectiveFuncVal = value;
                }*/
                optimizer.update(samples);
                BestScore = optimizer.getBestObjectiveFuncValue();

                BestParams = optimizer.getBest();

                iteration++;

                if ((iteration >= maxIteration && maxIteration > 0) ||
                    (BestScore <= targetValue && mode == OptimizationModes.minimize) ||
                    (BestScore >= targetValue && mode == OptimizationModes.maximize))
                {
                    //optimizatoin is done
                    if(onReady != null)
                        onReady.Invoke(BestParams);
                    IsOptimizing = false;
                }
            }
        }
    }


    public void StartOptimizeAsync(IESOptimizable optimizeTarget, Action<double[]> onReady = null)
    {
        optimizable = optimizeTarget;

        optimizer = optimizerType == ESOptimizerType.LMMAES ? (IMAES)new LMMAES() : (IMAES)new MAES();

        samples = new OptimizationSample[populationSize];
        for (int i = 0; i < populationSize; ++i)
        {
            samples[i] = new OptimizationSample(optimizable.GetParamDimension());
        }
        iteration = 0;

        optimizer.init(optimizable.GetParamDimension(), populationSize, new double[optimizable.GetParamDimension()], intialStepSize, mode);

        IsOptimizing = true;

        this.onReady = onReady;
    }

    public double[] Optimize(IESOptimizable optimizeTarget, Action<double[]> onReady = null )
    {

        var tempOptimizer = (optimizerType == ESOptimizerType.LMMAES ? (IMAES)new LMMAES() : (IMAES)new MAES());

        var tempSamples = new OptimizationSample[populationSize];
        for (int i = 0; i < populationSize; ++i)
        {
            tempSamples[i] = new OptimizationSample(optimizeTarget.GetParamDimension());
        }
 
        tempOptimizer.init(optimizeTarget.GetParamDimension(), populationSize, new double[optimizeTarget.GetParamDimension()], intialStepSize, mode);

        double[] bestParams = null;
        for (int it = 0; it < maxIteration; ++it)
        {
            tempOptimizer.generateSamples(tempSamples);
            for (int s = 0; s <= tempSamples.Length / evaluationBatchSize; ++s)
            {
                List<double[]> paramList = new List<double[]>();
                for (int b = 0; b < evaluationBatchSize; ++b)
                {
                    int ind = s * evaluationBatchSize + b;
                    if (ind < tempSamples.Length)
                    {
                        paramList.Add(tempSamples[ind].x);
                    }
                }

                var values = optimizeTarget.Evaluate(paramList);

                for (int b = 0; b < evaluationBatchSize; ++b)
                {
                    int ind = s * evaluationBatchSize + b;
                    if (ind < tempSamples.Length)
                    {
                        tempSamples[ind].objectiveFuncVal = values[b];
                    }
                }

            }

            tempOptimizer.update(tempSamples);
            double bestScore = tempOptimizer.getBestObjectiveFuncValue();

            bestParams = tempOptimizer.getBest();
            
            if ((bestScore <= targetValue && mode == OptimizationModes.minimize) ||
                (bestScore >= targetValue && mode == OptimizationModes.maximize))
            {
                //optimizatoin is done
                if (onReady != null)
                    onReady.Invoke (bestParams);
                break;
            }
        }

        return bestParams;
        
    }

    
    public void StopOptimize(Action<double[]> onReady = null)
    {
        IsOptimizing = false;
        if (onReady != null)
        {
            onReady.Invoke(BestParams);
        }
    }
    
}
