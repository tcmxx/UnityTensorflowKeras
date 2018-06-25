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

        rewardShapingToggleRef.isOn = gameSystemRef.defaultArena.rewardShaping;
    }

    private void Update()
    {
        populationSizeSliderRef.value = agentRef.populationSize;
        maxItrSliderRef.value = agentRef.maxIteration;
        rewardShapingToggleRef.isOn = gameSystemRef.defaultArena.rewardShaping;

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
        Physics.autoSimulation = false;
    }

    public void OnEndOptimizationButtonClicked()
    {
        agentRef.ForceEndOptimization();
        Physics.autoSimulation = true;
    }

    public void OnRewardShapingToggled(bool value)
    {
        gameSystemRef.defaultArena.rewardShaping = value;
    }

    public void GenerateHeatMap()
    {
        //heatmapRef.StartSampling(SamplingFunc,5,1);
        Physics.autoSimulation = false;
        heatmapRef.StartSampling(SamplingFuncBatch, 8, 2, 2, () => { Physics.autoSimulation = true; });
    }


    public List<float> SamplingFuncBatch(List<float> x, List<float> y)
    {
        List<Vector3> forces = new List<Vector3>();
        for (int i = 0; i < x.Count; ++i)
        {
            forces.Add(agentRef.SamplePointToForceVectorXY(x[i], y[i]));
        }
        var values = gameSystemRef.EvaluateShotBatch(forces, Color.gray);
        for (int i = 0; i < values.Count; ++i)
        {
            values[i] = Mathf.Clamp01((values[i] + 0.4f) / 2.4f);
        }
        return values;
    }


}
