using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using System.Linq;
using MLAgents;
using System.IO;
using System;





//agent info for internal training use
public struct AgentInfoInternal
{
    public List<float> stackedVectorObservation;

    /// <summary>
    /// Agent camera observation converted into float array.
    /// </summary>
    public List<float[,,]> visualObservations;

    /// <summary>
    /// Most recent text observation.
    /// </summary>
    public string textObservation;

    /// <summary>
    /// For discrete control, specifies the actions that the agent cannot take. Is true if
    /// the action is masked.
    /// </summary>
    public bool[] actionMasks;

    /// <summary>
    /// Used by the Trainer to store information about the agent. This data
    /// structure is not consumed or modified by the agent directly, they are
    /// just the owners of their trainier's memory. Currently, however, the
    /// size of the memory is in the Brain properties.
    /// </summary>
    public List<float> memories;

    /// <summary>
    /// Current agent reward.
    /// </summary>
    public float reward;

    /// <summary>
    /// Whether the agent is done or not.
    /// </summary>
    public bool done;

    /// <summary>
    /// Whether the agent has reached its max step count for this episode.
    /// </summary>
    public bool maxStepReached;

}

[CreateAssetMenu(fileName = "NewInternalLearningBrain", menuName = "ML-Agents/Internal Learning Brain")]
/// which decides actions using internally embedded TensorFlow model.
public class InternalLearningBrain : Brain
{
    public TrainerBase trainerBase;
    private Dictionary<Agent, AgentInfoInternal> currentInfo;

    private Dictionary<Agent, TakeActionOutput> prevActionOutput;

    protected List<float> secondUpdate = new List<float>();
    protected int maxReccord = 20;
    protected int numOfRecord = -1;
    protected string savePath = "SecPerUpdate.csv";
    /// <inheritdoc/>
    protected override void Initialize()
    {
        if (trainerBase)
        {

        }
        else
        {

            Debug.LogError("NO TrainerBased is assigned to this internal learning brain. Make sure there is a trainer that uses this brain in the scene.");
        }
        trainerBase.Initialize();
        numOfRecord = -1;
        secondUpdate = new List<float>();
    }


    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    protected override void DecideAction()
    {
        var newAgentInfoRaw = agentInfos;

        int currentBatchSize = newAgentInfoRaw.Count();
        List<Agent> newAgentList = newAgentInfoRaw.Keys.ToList();
        List<Agent> recordableAgentList = newAgentList.Where((a) => currentInfo != null && currentInfo.ContainsKey(a) && prevActionOutput.ContainsKey(a)).ToList();

        //clone the raw agent info into the agent info we need
        Dictionary<Agent, AgentInfoInternal> newAgentInfo = new Dictionary<Agent, AgentInfoInternal>();
        foreach (var a in newAgentList)
        {
            newAgentInfo[a] = CopyAgentInfo(newAgentInfoRaw[a], a.brain);
        }


        if (currentBatchSize == 0)
        {
            return;
        }


        //get the datas only for the agents in the agentInfo input
        var prevInfo = GetValueForAgents(currentInfo, recordableAgentList);
        var prevActionActions = GetValueForAgents(prevActionOutput, recordableAgentList);
        var newInfo = GetValueForAgents(newAgentInfo, recordableAgentList);

        if (recordableAgentList.Count > 0 && trainerBase.IsTraining() && trainerBase.GetStep() <= trainerBase.GetMaxStep())
        {
            trainerBase.AddExperience(prevInfo, newInfo, prevActionActions);
            trainerBase.ProcessExperience(prevInfo, newInfo);
        }



        if (trainerBase.IsTraining() && trainerBase.GetStep() <= trainerBase.GetMaxStep())
        {
            trainerBase.IncrementStep();
        }

        //update the info
        UpdateInfos(ref currentInfo, newAgentInfo);

        var actionOutputs = trainerBase.TakeAction(GetValueForAgents(currentInfo, newAgentList));
        UpdateActionOutputs(ref prevActionOutput, actionOutputs);

        //TODO Update the agent's other info if there is
        foreach (Agent agent in newAgentList)
        {
            if (actionOutputs.ContainsKey(agent) && actionOutputs[agent].outputAction != null)
            {
                agent.UpdateVectorAction(trainerBase.PostprocessingAction(actionOutputs[agent].outputAction));
            }
        }



        if (trainerBase.IsReadyUpdate() && trainerBase.IsTraining() && trainerBase.GetStep() <= trainerBase.GetMaxStep())
        {
            float t = Time.realtimeSinceStartup;
            trainerBase.UpdateModel();
            if (numOfRecord >= 0)
            {
                float dt = Time.realtimeSinceStartup - t;
                secondUpdate.Add(dt);
                Debug.Log(dt + ":"+numOfRecord);
            }
            numOfRecord++;
            if (numOfRecord == maxReccord)
            {
                SaveToFile(savePath);
            }
        }

        //clear the prev record if the agent is done
        foreach (Agent agent in newAgentList)
        {
            if (newAgentInfo[agent].done || newAgentInfo[agent].maxStepReached)
            {
                currentInfo.Remove(agent);
            }
        }

        agentInfos.Clear();

    }

    protected void SaveToFile(string path)
    {
        File.WriteAllText(Path.Combine(Directory.GetCurrentDirectory(), path), string.Join(Environment.NewLine, secondUpdate));
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

    protected static void UpdateInfos(ref Dictionary<Agent, AgentInfoInternal> allInfos, Dictionary<Agent, AgentInfoInternal> newInfos)
    {
        if (allInfos == null)
            allInfos = new Dictionary<Agent, AgentInfoInternal>();

        foreach (var agent in newInfos.Keys)
        {
            allInfos[agent] = newInfos[agent];
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

    public static AgentInfoInternal CopyAgentInfo(AgentInfo agentInfo, Brain brain)
    {
        var result = new AgentInfoInternal()
        {
            actionMasks = (bool[])agentInfo.actionMasks?.Clone(),
            stackedVectorObservation = new List<float>(agentInfo.stackedVectorObservation),
            visualObservations = agentInfo.visualObservations.Select((x, i) => x.TextureToArray(brain.brainParameters.cameraResolutions[i].blackAndWhite)).ToList(),
            textObservation = (string)agentInfo.textObservation?.Clone(),
            memories = new List<float>(agentInfo.memories),
            reward = agentInfo.reward,
            done = agentInfo.done,
            maxStepReached = agentInfo.maxStepReached,
        };

        return result;
    }
}
