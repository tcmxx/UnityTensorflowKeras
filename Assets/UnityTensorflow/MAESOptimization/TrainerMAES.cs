using ICM;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents;

public class TrainerMAES : Trainer
{

    /// Reference to the brain that uses this CoreBrainInternal
    protected Brain brain;
    public ESOptimizerType optimizer;

    public OptimizationModes optimizationMode;
    public int iterationPerFrame = 20;
    public int evaluationBatchSize = 8;
    public bool debugVisualization = true;

    private Dictionary<AgentES, OptimizationData> currentOptimizingAgents;


    public enum ESOptimizerType
    {
        MAES,
        LMMAES
    }

    protected class OptimizationData
    {
        public OptimizationData(int populationSize, IMAES optimizerToUse, int dim)
        {
            samples = new OptimizationSample[populationSize];
            for (int i = 0; i < populationSize; ++i)
            {
                samples[i] = new OptimizationSample(dim);
            }
            interation = 0;
            optimizer = optimizerToUse;
        }

        public int interation;
        public OptimizationSample[] samples;
        public IMAES optimizer;
    }



    private void FixedUpdate()
    {
        ContinueOptimization();
    }
    /// Create the reference to the brain
    public override void Initialize()
    {
        currentOptimizingAgents = new Dictionary<AgentES, OptimizationData>();
    }




    protected void AddOptimization(List<AgentES> agents)
    {
        foreach (var agent in agents)
        {
            currentOptimizingAgents[agent] = new OptimizationData(agent.populationSize, optimizer == ESOptimizerType.LMMAES ? (IMAES)new LMMAES() : (IMAES)new MAES(), agent.GetParamDimension());
            currentOptimizingAgents[agent].optimizer.init(brain.brainParameters.vectorActionSize,
                agent.populationSize, new double[brain.brainParameters.vectorActionSize], agent.initialStepSize, optimizationMode);
            agent.OnEndOptimizationRequested += OnEndOptimizationRequested;
        }
    }

    protected void ContinueOptimization()
    {
        for (int it = 0; it < iterationPerFrame; ++it)
        {
            List<AgentES> agentList = currentOptimizingAgents.Keys.ToList();
            foreach (var agent in agentList)
            {
                var optData = currentOptimizingAgents[agent];
                optData.optimizer.generateSamples(optData.samples);


                agent.SetVisualizationMode(debugVisualization ? AgentES.VisualizationMode.Sampling : AgentES.VisualizationMode.None);

                for (int s = 0; s <= optData.samples.Length / evaluationBatchSize; ++s)
                {
                    List<double[]> paramList = new List<double[]>();
                    for (int b = 0; b < evaluationBatchSize; ++b)
                    {
                        int ind = s * evaluationBatchSize + b;
                        if (ind < optData.samples.Length)
                        {
                            paramList.Add(optData.samples[ind].x);
                        }
                    }

                    var values = agent.Evaluate(paramList);

                    for (int b = 0; b < evaluationBatchSize; ++b)
                    {
                        int ind = s * evaluationBatchSize + b;
                        if (ind < optData.samples.Length)
                        {
                            optData.samples[ind].objectiveFuncVal = values[b];
                        }
                    }

                }
                /*foreach (OptimizationSample s in optData.samples)
                {
                    float value = agent.Evaluate(new List<double[]> { s.x })[0];
                    s.objectiveFuncVal = value;
                }*/



                optData.optimizer.update(optData.samples);
                double bestScore = optData.optimizer.getBestObjectiveFuncValue();
                //Debug.Log("Best shot score " + optData.optimizer.getBestObjectiveFuncValue());
                agent.SetVisualizationMode(debugVisualization ? AgentES.VisualizationMode.Best : AgentES.VisualizationMode.None);
                agent.Evaluate(new List<double[]> { optData.optimizer.getBest() });

                optData.interation++;
                if ((optData.interation >= agent.maxIteration && agent.maxIteration > 0) ||
                    (bestScore <= agent.targetValue && optimizationMode == OptimizationModes.minimize) ||
                    (bestScore >= agent.targetValue && optimizationMode == OptimizationModes.maximize))
                {
                    //optimizatoin is done
                    agent.OnReady(optData.optimizer.getBest());
                    currentOptimizingAgents.Remove(agent);
                }
            }
        }
    }

    protected void OnEndOptimizationRequested(AgentES agent)
    {
        if (currentOptimizingAgents.ContainsKey(agent))
        {
            var optData = currentOptimizingAgents[agent];
            agent.OnReady(optData.optimizer.getBest());
            currentOptimizingAgents.Remove(agent);
            agent.OnEndOptimizationRequested -= OnEndOptimizationRequested;
        }
    }


    public override int GetStep()
    {
        return 0;
    }

    public override int GetMaxStep()
    {
        return int.MaxValue;
    }

    public override Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var agentList = agentInfos.Keys;
        List<AgentES> agentsToOptimize = new List<AgentES>();
        foreach (Agent a in agentList)
        {
            if (!(a is AgentES))
            {
                Debug.LogError("Agents using CoreBrainMAES must inherit from AgentES");
            }
            if (!currentOptimizingAgents.ContainsKey((AgentES)a))
            {
                agentsToOptimize.Add((AgentES)a);
            }
            else
            {
                //Debug.LogError("new decision requested while last decision is not made yet");
            }
        }

        if (agentsToOptimize.Count > 0)
            AddOptimization(agentsToOptimize);
        

        return new Dictionary<Agent, TakeActionOutput>();
    }

    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent, TakeActionOutput> actionOutput)
    {
        return;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        return;
    }

    public override bool IsReadyUpdate()
    {
        return false;
    }

    public override void UpdateModel()
    {
        return;
    }

    public override void IncrementStep()
    {
        return;
    }

    public override void SetBrain(Brain brain)
    {
        this.brain = brain; ;
    }
}
