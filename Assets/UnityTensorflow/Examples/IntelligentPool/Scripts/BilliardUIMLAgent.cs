using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BilliardUIMLAgent : MonoBehaviour {
    public Text predictedScoreTextRef;
    public Text predictedActionTextRef;
    public Text populationSizeTextRef;
    public Text maxItrTextRef;

    public Slider populationSizeSliderRef;
    public Slider maxItrSliderRef;

    public Toggle rewardShapingToggleRef;
    public Toggle autoRequestToggleRef;
    public Dropdown playModeRef;

    protected BilliardAgent agentRef;
    protected BilliardGameSystem gameSystemRef;
    protected HeatMap heatmapRef;
    protected TrainerMimic trainerRef;
    protected DecisionMAES agentDecisionRef;
    private void Awake()
    {
        agentRef = FindObjectOfType<BilliardAgent>();
        gameSystemRef = FindObjectOfType<BilliardGameSystem>();
        heatmapRef = FindObjectOfType<HeatMap>();
        trainerRef = FindObjectOfType<TrainerMimic>();
        agentDecisionRef = agentRef.GetComponent<DecisionMAES>();
    }

    private void Start()
    {
        populationSizeSliderRef.value = agentRef.populationSize;
        maxItrSliderRef.value = agentRef.maxIteration;
        populationSizeTextRef.text = "Population size: " + agentRef.populationSize.ToString();
        maxItrTextRef.text = "Max Iter: " + agentRef.maxIteration;

        rewardShapingToggleRef.isOn = gameSystemRef.defaultArena.rewardShaping;
        autoRequestToggleRef.isOn = agentRef.autoRequestDecision;

        OnPlayModeChanged(playModeRef.value);
    }

    private void Update()
    {
        populationSizeSliderRef.value = agentRef.populationSize;
        maxItrSliderRef.value = agentRef.maxIteration;
        rewardShapingToggleRef.isOn = gameSystemRef.defaultArena.rewardShaping;

        predictedScoreTextRef.text = "Predicted score: " + gameSystemRef.bestScore;
        if (gameSystemRef.bestActions != null && gameSystemRef.bestActions.Count > 0)
            predictedActionTextRef.text = "Best action: " + gameSystemRef.bestActions[0].x + ", " + gameSystemRef.bestActions[0].z;
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
        gameSystemRef.bestScore = Mathf.NegativeInfinity;
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

    public void OnAutoRequestToggled(bool value)
    {
        agentRef.autoRequestDecision = value;
    }

    public void OnResetClicked()
    {
        gameSystemRef.Reset();
    }

    public void OnPlayModeChanged(int mode)
    {
        if (agentDecisionRef == null)
            return;
        if(mode == 0)
        {
            agentDecisionRef.useHeuristic = false;
            agentDecisionRef.useDecision = true;
           // trainerRef.isTraining = false;
        }else if(mode == 1)
        {
            //trainerRef.isTraining = true;
            agentDecisionRef.useDecision = false;
        }else if(mode == 2)
        {
            agentDecisionRef.useHeuristic = true;
            agentDecisionRef.useDecision = true;
            //trainerRef.isTraining = true;
        }
    }

    public void GenerateHeatMap()
    {
        gameSystemRef.bestScore = Mathf.NegativeInfinity;
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
