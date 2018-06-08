using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public struct TakeActionOutput
{



    public Dictionary<Agent,float[]> outputAction;
    public Dictionary<Agent, float[]> allProbabilities;
    public Dictionary<Agent, float> value;
    //public Dictionary<Agent, float> entropy;
    
    //public Dictionary<Agent, float[]> memory;

    //public Dictionary<Agent, string> textAction;
}

public abstract class Trainer:MonoBehaviour {

    public Academy academyRef;
    public bool isTraining;
    protected bool prevIsTraining;

    private void Start()
    {
        prevIsTraining = isTraining;
        academyRef.SetIsInference(!isTraining);
    }
    public virtual void Update()
    {
        if(prevIsTraining != isTraining)
        {
            prevIsTraining = isTraining;
            academyRef.SetIsInference(!isTraining);
        }
    }

    public abstract void SetBrain(Brain brain);
    public abstract int GetStep();
    public abstract int GetMaxStep();

    public abstract TakeActionOutput TakeAction(Dictionary<Agent, AgentInfo> agentInfos);
    public abstract void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, TakeActionOutput actionOutput);
    public abstract void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo);
    public abstract bool IsReadyUpdate();
    public abstract void UpdateModel();
    public abstract void IncrementStep();
}
