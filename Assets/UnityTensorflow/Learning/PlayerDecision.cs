using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

public class PlayerDecision : AgentDependentDecision
{
    public KeyCode toggleDecisionUsageKey = KeyCode.U;

    [System.Serializable]
    private struct DiscretePlayerAction
    {
        public KeyCode key;
        public int value;
    }

    [System.Serializable]
    private struct KeyContinuousPlayerAction
    {
        public KeyCode key;
        public int index;
        public float value;
    }

    [System.Serializable]
    private struct AxisContinuousPlayerAction
    {
        public string axis;
        public int index;
        public float scale;
    }

    [SerializeField]
    [FormerlySerializedAs("continuousPlayerActions")]
    [Tooltip("The list of keys and the value they correspond to for continuous control.")]
    /// Contains the mapping from input to continuous actions
    private KeyContinuousPlayerAction[] keyContinuousPlayerActions;

    [SerializeField]
    [Tooltip("The list of axis actions.")]
    /// Contains the mapping from input to continuous actions
    private AxisContinuousPlayerAction[] axisContinuousPlayerActions;

    [SerializeField]
    [Tooltip("The list of keys and the value they correspond to for discrete control.")]
    /// Contains the mapping from input to discrete actions
    private DiscretePlayerAction[] discretePlayerActions;
    [SerializeField]
    private int defaultAction = 0;


    private void Update()
    {
        if (Input.GetKeyDown(toggleDecisionUsageKey))
        {
            useDecision = !useDecision;
        }
    }

    public override float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null)
    {
        if (agent.brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {

            var action = new float[agent.brain.brainParameters.vectorActionSize];
            foreach (KeyContinuousPlayerAction cha in keyContinuousPlayerActions)
            {
                if (Input.GetKey(cha.key))
                {
                    action[cha.index] = cha.value;
                }
            }


            foreach (AxisContinuousPlayerAction axisAction in axisContinuousPlayerActions)
            {
                var axisValue = Input.GetAxis(axisAction.axis);
                axisValue *= axisAction.scale;
                if (Mathf.Abs(axisValue) > 0.0001)
                {
                    action[axisAction.index] = axisValue;
                }
            }
            return action;

        }
        else
        {

            var action = new float[1] { defaultAction };
            foreach (DiscretePlayerAction dha in discretePlayerActions)
            {
                if (Input.GetKey(dha.key))
                {
                    action[0] = (float)dha.value;
                    break;
                }
            }
            return action;
        }

    }
}
