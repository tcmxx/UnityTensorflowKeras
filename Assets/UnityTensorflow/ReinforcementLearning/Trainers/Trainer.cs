using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public struct TakeActionOutput
{
    public float[] outputAction;
    public float[] allProbabilities;
    public float[] value;
    public float[] entropy;

    public float[] learningRate;

    public float[] memory;

    public string[] textAction;
}

public abstract class Trainer:MonoBehaviour {

    public abstract int GetStep();
    public abstract int GetMaxStep();

    public abstract TakeActionOutput TakeAction(Dictionary<Agent, AgentInfo> agentInfos);
    public abstract void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, TakeActionOutput actionOutput);
    public abstract void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo);
    public abstract bool IsReadyUpdate();
    public abstract void UpdateModel();
    public abstract void WriteSummary();
    public abstract void IncrementStep();
    public abstract void UpdateLastReward();
}
