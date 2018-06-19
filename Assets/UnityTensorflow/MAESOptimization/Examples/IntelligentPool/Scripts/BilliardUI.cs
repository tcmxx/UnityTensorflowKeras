using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BilliardUI : MonoBehaviour {
    public Text predictedScoreTextRef;
    public Text populationSizeTextRef;
    public Text maxItrTextRef;
    public Text optButtonTextRef;

    public Slider populationSizeSliderRef;
    public Slider maxItrSliderRef;

    public Toggle rewardShapingToggleRef;

    public BilliardAgent agentRef;
    public BilliardGameSystem gameSystemRef;


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
}
