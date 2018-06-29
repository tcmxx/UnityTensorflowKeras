using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using System.Linq;
using MLAgents;

/// CoreBrain which decides actions using internally embedded TensorFlow model.
public class CoreBrainInternalTrainable : ScriptableObject, CoreBrain
{
    /// Reference to the brain that uses this CoreBrainInternal
    public Brain brain;
    public Trainer trainer;

    private Dictionary<Agent, AgentInfo> currentInfo;
    private Dictionary<Agent, TakeActionOutput> prevActionOutput;






    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
        if (trainer)
            trainer.SetBrain(b);
    }


    public void InitializeCoreBrain(MLAgents.Batcher brainBatcher)
    {
        trainer.Initialize();
    }



    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    public void DecideAction(Dictionary<Agent, AgentInfo> newAgentInfos)
    {
        int currentBatchSize = newAgentInfos.Count();
        List<Agent> newAgentList = newAgentInfos.Keys.ToList();
        List<Agent> recordableAgentList = newAgentList.Where((a) => currentInfo != null && currentInfo.ContainsKey(a)).ToList();

        if (currentBatchSize == 0)
        {
            return;
        }


        //get the datas only for the agents in the agentInfo input
        var prevInfo = GetValueForAgents(currentInfo, recordableAgentList);    
        var prevActionActions = GetValueForAgents(prevActionOutput, recordableAgentList);
        var newInfo = GetValueForAgents(newAgentInfos, recordableAgentList);

        if (recordableAgentList.Count > 0 && trainer.isTraining && trainer.GetStep() <= trainer.GetMaxStep())
        {
            trainer.AddExperience(prevInfo, newInfo, prevActionActions);
            trainer.ProcessExperience(prevInfo, newInfo);
        }



        if (trainer.isTraining && trainer.GetStep() <= trainer.GetMaxStep())
        {
            trainer.IncrementStep();
        }

        //update the info
        UpdateInfos(ref currentInfo, newAgentInfos);

        var actionOutputs = trainer.TakeAction(GetValueForAgents(currentInfo, newAgentList));
        UpdateActionOutputs(ref prevActionOutput, actionOutputs);

        //TODO Update the agent's other info if there is
        foreach (Agent agent in newAgentList)
        {
            if (actionOutputs.ContainsKey(agent) && actionOutputs[agent].outputAction != null)
                agent.UpdateVectorAction(actionOutputs[agent].outputAction);
        }

    }

    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public void OnInspector()
    {
#if UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

        var serializedBrain = new SerializedObject(this);



        var trainerProperty = serializedBrain.FindProperty("trainer");
        serializedBrain.Update();
        EditorGUILayout.PropertyField(trainerProperty, true);
        serializedBrain.ApplyModifiedProperties();
#endif
    }



    protected static Dictionary<Agent, T> GetValueForAgents<T>(Dictionary<Agent, T> allInfos, List<Agent> agents)
    {


        Dictionary<Agent, T> result = new Dictionary<Agent, T>();
        foreach (var agent in agents)
        {
            result[agent] = allInfos[agent];
        }
        return result;
    }

    protected static void UpdateInfos(ref Dictionary<Agent, AgentInfo> allInfos, Dictionary<Agent, AgentInfo> newInfos)
    {
        if (allInfos == null)
            allInfos = new Dictionary<Agent, AgentInfo>();

        foreach (var agent in newInfos.Keys)
        {

            //TODO remove this once Unity fixed their texture not released bug
            if (allInfos.ContainsKey(agent))
            {
                foreach (var v in allInfos[agent].visualObservations)
                {
                    Destroy(v);
                }
            }

            allInfos[agent] = CopyAgentInfo(newInfos[agent]);
        }
    }

    protected static void UpdateActionOutputs(ref Dictionary<Agent, TakeActionOutput> actionOutputs, Dictionary<Agent, TakeActionOutput> newActionOutputs)
    {
        if (actionOutputs == null)
            actionOutputs = new Dictionary<Agent, TakeActionOutput>();

        foreach (var agent in newActionOutputs.Keys)
        {
            actionOutputs[agent] = newActionOutputs[agent];
        }
    }

    public static AgentInfo CopyAgentInfo(AgentInfo agentInfo)
    {
        var result = new AgentInfo()
        {
            vectorObservation = new List<float>(agentInfo.vectorObservation),
            stackedVectorObservation = new List<float>(agentInfo.stackedVectorObservation),
            visualObservations = new List<Texture2D>(agentInfo.visualObservations),
            textObservation = (string)agentInfo.textObservation?.Clone(),
            storedVectorActions = (float[])agentInfo.storedVectorActions.Clone(),
            storedTextActions = (string)agentInfo.storedTextActions?.Clone(),
            memories = new List<float>(agentInfo.memories),
            reward = agentInfo.reward,
            done = agentInfo.done,
            maxStepReached = agentInfo.maxStepReached,
            id = agentInfo.id
        };

        return result;
    }
}
