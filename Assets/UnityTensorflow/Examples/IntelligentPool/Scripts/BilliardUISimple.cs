using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BilliardUISimple : MonoBehaviour {
    public Text predictedScoreTextRef;
    public Text populationSizeTextRef;
    public Text maxItrTextRef;

    public Slider populationSizeSliderRef;
    public Slider maxItrSliderRef;

    public Toggle rewardShapingToggleRef;

    public BilliardSimple agentRef;
    public ESOptimizer optimizerRef;
    public BilliardGameSystem gameSystemRef;
    public HeatMap heatmapRef;

    private void Start()
    {
        populationSizeSliderRef.value = optimizerRef.populationSize;
        maxItrSliderRef.value = optimizerRef.maxIteration;
        populationSizeTextRef.text = "Population size: " + optimizerRef.populationSize.ToString();
        maxItrTextRef.text = "Max Iter: " + optimizerRef.maxIteration;

        rewardShapingToggleRef.isOn = gameSystemRef.defaultArena.rewardShaping;
    }

    private void Update()
    {
        populationSizeSliderRef.value = optimizerRef.populationSize;
        maxItrSliderRef.value = optimizerRef.maxIteration;
        rewardShapingToggleRef.isOn = gameSystemRef.defaultArena.rewardShaping;

        predictedScoreTextRef.text = "Predicted score: " + gameSystemRef.predictedShotScore;
    }

    public void OnPopulationSliderChanged(float value)
    {
        optimizerRef.populationSize = Mathf.RoundToInt(value);
        populationSizeTextRef.text = "Population size: " + optimizerRef.populationSize.ToString();
    }

    public void OnIterationSliderChanged(float value)
    {
        optimizerRef.maxIteration = Mathf.RoundToInt(value);
        maxItrTextRef.text = "Max Iter: " + optimizerRef.maxIteration;

    }

    public void OnOptimizationButtonClicked()
    {
        optimizerRef.StartOptimize(agentRef);
    }

    public void OnEndOptimizationButtonClicked()
    {
        optimizerRef.StopOptimize(true);
    }

    public void OnRewardShapingToggled(bool value)
    {
        gameSystemRef.defaultArena.rewardShaping = value;
    }

    public void GenerateHeatMap()
    {
        //heatmapRef.StartSampling(SamplingFunc,5,1);
        heatmapRef.StartSampling(SamplingFuncBatch, 8, 2, 2);
    }

    
    public float SamplingFunc(float x, float y)
    {
        return Mathf.Clamp01(((gameSystemRef.evaluateShot(agentRef.SamplePointToForceVectorXY(x,y), Color.gray)) + 0.4f)/2.4f);
    }

    public List<float> SamplingFuncBatch(List<float> x, List<float> y)
    {
        List<Vector3> forces = new List<Vector3>();
        for(int i = 0; i < x.Count; ++i)
        {
            forces.Add(agentRef.SamplePointToForceVectorXY(x[i], y[i]));
        }
        var values = gameSystemRef.evaluateShots(forces, Color.gray);
        for(int i = 0; i < values.Count; ++i)
        {
            values[i] = Mathf.Clamp01((values[i] + 0.4f) / 2.4f);
        }
        return values;
    }
}
