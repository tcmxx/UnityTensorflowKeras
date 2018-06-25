using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using System.Linq;

/// CoreBrain which decides actions using internally embedded TensorFlow model.
public class CoreBrainInternalTrainable : ScriptableObject, CoreBrain
{
    /// Reference to the brain that uses this CoreBrainInternal
    public Brain brain;
    public Trainer trainer;
    
    private Dictionary<Agent, AgentInfo> currentInfo;
    private TakeActionOutput prevActionOutput;



    


    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
        if(trainer)
            trainer.SetBrain(b);
    }

    
    public void InitializeCoreBrain(MLAgents.Batcher brainBatcher)
    {
        trainer.Initialize();
    }



    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
    {
        int currentBatchSize = agentInfo.Count();
        List<Agent> agentList = agentInfo.Keys.ToList();
        if (currentBatchSize == 0)
        {
            return;
        }


        if (trainer.GetStep() > 0 && trainer.isTraining && trainer.GetStep() <= trainer.GetStep())
        {
            trainer.AddExperience(currentInfo, agentInfo, prevActionOutput);
            trainer.ProcessExperience(currentInfo, agentInfo);
        }

        if (trainer.IsReadyUpdate() && trainer.isTraining && trainer.GetStep() <= trainer.GetStep())
        {
            trainer.UpdateModel();
        }
        
        if (trainer.isTraining && trainer.GetStep() <= trainer.GetStep())
        {
            trainer.IncrementStep();
        }

        //copy the info
        if(currentInfo != null)
        {
            foreach(var a in currentInfo)
            {
                foreach(var v in a.Value.visualObservations)
                {
                    Destroy(v);
                }
            }
        }
        currentInfo = new Dictionary<Agent, AgentInfo>();
        foreach (Agent agent in agentList)
        {
            currentInfo[agent] = CopyAngentInfo(agentInfo[agent]);

        }

        prevActionOutput = trainer.TakeAction(currentInfo);

        //TODO Update the agent's other info if there is
        foreach (Agent agent in agentList)
        {
            if(prevActionOutput.outputAction != null && prevActionOutput.outputAction.ContainsKey(agent))
                agent.UpdateVectorAction(prevActionOutput.outputAction[agent]);
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


    public static AgentInfo CopyAngentInfo(AgentInfo agentInfo)
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
