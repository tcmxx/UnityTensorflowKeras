using ICM;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents;
using System;

public class TrainerMAES : TrainerBase
{
    public bool debugVisualization = true;

    

    public override void Initialize()
    {
    }

    public override int GetStep()
    {
        return 0;
    }

    public override int GetMaxStep()
    {
        return int.MaxValue;
    }

    public override Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfoInternal> agentInfos)
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

    public override void AddExperience(Dictionary<Agent, AgentInfoInternal> currentInfo, Dictionary<Agent, AgentInfoInternal> newInfo, Dictionary<Agent, TakeActionOutput> actionOutput)
    {
        return;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfoInternal> currentInfo, Dictionary<Agent, AgentInfoInternal> newInfo)
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



    public override void ResetTrainer()
    {
        return;
    }

    public override bool IsTraining()
    {
        return false;
    }

    public override float[] PostprocessingAction(float[] rawAction)
    {
        return rawAction;
    }
    
}
