using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BilliardUIMLAgent : MonoBehaviour {
    public Text predictedScoreTextRef;
    public Text populationSizeTextRef;
    public Text maxItrTextRef;

    public Slider populationSizeSliderRef;
    public Slider maxItrSliderRef;

    public Toggle rewardShapingToggleRef;

    public BilliardAgent agentRef;
    public BilliardGameSystem gameSystemRef;
    public HeatMap heatmapRef;

    private void Start()
    {
        populationSizeSliderRef.value = agentRef.populationSize;
        maxItrSliderRef.value = agentRef.maxIteration;
        populationSizeTextRef.text = "Population size: " + agentRef.populationSize.ToString();
        maxItrTextRef.text = "Max Iter: " + agentRef.maxIteration;

        rewardShapingToggleRef.isOn = gameSystemRef.rewardShaping;
    }

    private void Update()
    {
        populationSizeSliderRef.value = agentRef.populationSize;
        maxItrSliderRef.value = agentRef.maxIteration;
        rewardShapingToggleRef.isOn = gameSystemRef.rewardShaping;

        predictedScoreTextRef.text = "Predicted score: " + gameSystemRef.predictedShotScore;
    }

    public void OnPopulationSliderChanged(float value)
    {
        agentRef.populationSize = Mathf.RoundToInt(value);
        populationSizeTextRef.text = "Population size: " + agentRef.populationSize.ToString();
    }

    public void OnIterationSliderChanged(float value)
    {
        agentRef.maxIteration = Mathf.RoundToInt(value);
        maxItrTextRef.text = "Max Iter: " + agentRef.maxIteration;

    }

    public void OnOptimizationButtonClicked()
    {
        agentRef.RequestDecision();
    }

    public void OnEndOptimizationButtonClicked()
    {
        agentRef.ForceEndOptimization();
    }

    public void OnRewardShapingToggled(bool value)
    {
        gameSystemRef.rewardShaping = value;
    }

    public void GenerateHeatMap()
    {
        heatmapRef.StartSampling(SamplingFunc,5,1);
    }

    
    public float SamplingFunc(float x, float y)
    {
        return Mathf.Clamp01(((gameSystemRef.evaluateShot(agentRef.SamplePointToForceVectorXY(x,y), Color.gray)) + 0.4f)/2.4f);
    }

    
}
