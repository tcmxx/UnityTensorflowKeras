using ICM;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents;
using System;

public class TrainerMAES : MonoBehaviour, ITrainer
{

    /// Reference to the brain that uses this CoreBrainInternal
    protected Brain brain;
    
    public bool debugVisualization = true;

    

    public void Initialize()
    {
    }



    public int GetStep()
    {
        return 0;
    }

    public int GetMaxStep()
    {
        return int.MaxValue;
    }

    public Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();
        foreach (var a in agentInfos)
        {
            AgentES agent = a.Key as AgentES;
            if (agent != null)
            {
                if (agent.synchronizedDecision)
                {

                    result[agent] = new TakeActionOutput() { outputAction = Array.ConvertAll(agent.Optimize(), t => (float)t) };
                }
                else
                {
                    agent.OptimizeAsync();
                }
            }
        }
        return new Dictionary<Agent, TakeActionOutput>();
    }

    public void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent, TakeActionOutput> actionOutput)
    {
        return;
    }

    public void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        return;
    }

    public bool IsReadyUpdate()
    {
        return false;
    }

    public void UpdateModel()
    {
        return;
    }

    public void IncrementStep()
    {
        return;
    }

    public void SetBrain(Brain brain)
    {
        this.brain = brain; ;
    }

    public void ResetTrainer()
    {
        return;
    }

    public bool IsTraining()
    {
        return false;
    }
}
