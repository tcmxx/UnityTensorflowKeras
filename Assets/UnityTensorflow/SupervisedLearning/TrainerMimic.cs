using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrainerMimic : Trainer {

    public AgentDependentDecision decisionToMimic;



    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, TakeActionOutput actionOutput)
    {
        throw new System.NotImplementedException();
    }

    public override int GetMaxStep()
    {
        throw new System.NotImplementedException();
    }

    public override int GetStep()
    {
        throw new System.NotImplementedException();
    }

    public override void IncrementStep()
    {
        throw new System.NotImplementedException();
    }

    public override void Initialize()
    {
        throw new System.NotImplementedException();
    }

    public override bool IsReadyUpdate()
    {
        throw new System.NotImplementedException();
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        throw new System.NotImplementedException();
    }

    public override void SetBrain(Brain brain)
    {
        throw new System.NotImplementedException();
    }

    public override TakeActionOutput TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        throw new System.NotImplementedException();
    }

    public override void UpdateModel()
    {
        throw new System.NotImplementedException();
    }
    
}
