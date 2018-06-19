using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ICM;
using System;

#if UNITY_EDITOR
using UnityEditor;
#endif

using System.Linq;

/// CoreBrain which decides actions using internally embedded TensorFlow model.
public class CoreBrainMAES : ScriptableObject, CoreBrain
{
    /// Reference to the brain that uses this CoreBrainInternal
    public Brain brain;
    public ESOptimizer optimizer;
   
    public OptimizationModes optimizationMode;
    public int iterationPerFrame = 20;

    public bool debugVisualization = false;

    private Dictionary<AgentES, OptimizationData> currentOptimizingAgents;


    public enum ESOptimizer
    {
        MAES,
        LMMAES
    }

    protected class OptimizationData
    {
        public OptimizationData(int populationSize, IMAES optimizerToUse, int dim)
        {
            samples = new OptimizationSample[populationSize];
            for(int i = 0; i < populationSize; ++i)
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


    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    
    public void InitializeCoreBrain(MLAgents.Batcher brainBatcher)
    {
        currentOptimizingAgents = new Dictionary<AgentES, OptimizationData>();
    }



    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
    {
        var agentList = agentInfo.Keys;
        List<AgentES> agentsToOptimize = new List<AgentES>();
        foreach(Agent a in agentList)
        {
            if(!(a is AgentES))
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

        if(agentsToOptimize.Count > 0)
            AddOptimization(agentsToOptimize);
        ContinueOptimization();
        
    }



    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public void OnInspector()
    {
#if UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

        var serializedBrain = new SerializedObject(this);



        var debugVis = serializedBrain.FindProperty("debugVisualization");
        var optMode = serializedBrain.FindProperty("optimizationMode");
        var itPerFrame = serializedBrain.FindProperty("iterationPerFrame");
        var opt = serializedBrain.FindProperty("optimizer");

        serializedBrain.Update();
        EditorGUILayout.PropertyField(debugVis, true);
        EditorGUILayout.PropertyField(optMode, true);
        EditorGUILayout.PropertyField(itPerFrame, true);
        EditorGUILayout.PropertyField(opt, true);
        serializedBrain.ApplyModifiedProperties();
#endif
    }





    protected void AddOptimization(List<AgentES> agents)
    {
        foreach(var agent in agents)
        {
            currentOptimizingAgents[agent] = new OptimizationData(agent.populationSize, optimizer== ESOptimizer.LMMAES?(IMAES)new LMMAES(): (IMAES)new MAES(), brain.brainParameters.vectorActionSize);
            currentOptimizingAgents[agent].optimizer.init(brain.brainParameters.vectorActionSize, 
                agent.populationSize, new double[brain.brainParameters.vectorActionSize],agent.initialStepSize, optimizationMode);
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
                foreach (OptimizationSample s in optData.samples)
                {
                    float value = agent.EvaluateAction(s.x);
                    s.objectiveFuncVal = value;
                }
                optData.optimizer.update(optData.samples);
                double bestScore = optData.optimizer.getBestObjectiveFuncValue();
                //Debug.Log("Best shot score " + optData.optimizer.getBestObjectiveFuncValue());
                agent.SetVisualizationMode(debugVisualization ? AgentES.VisualizationMode.Best : AgentES.VisualizationMode.None);
                agent.EvaluateAction(optData.optimizer.getBest());

                optData.interation++;
                if ((optData.interation >= agent.maxIteration && agent.maxIteration > 0) ||
                    (bestScore < agent.valueThresholdToStop && optimizationMode == OptimizationModes.minimize) ||
                    (bestScore > agent.valueThresholdToStop && optimizationMode == OptimizationModes.maximize))
                {
                    //optimizatoin is done
                    agent.OnActionReady(optData.optimizer.getBest());
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
            agent.OnActionReady(optData.optimizer.getBest());
            currentOptimizingAgents.Remove(agent);
        }
    }
    
}
