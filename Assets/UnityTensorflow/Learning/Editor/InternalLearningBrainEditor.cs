using UnityEngine;
using UnityEditor;

namespace MLAgents
{
    /// <summary>
    /// CustomEditor for the InternalLearningBrain class. Defines the default Inspector view for a
    /// InternalLearningBrain.
    /// Shows the BrainParameters of the Brain and expose a tool to deep copy BrainParameters
    /// between brains.
    /// the Internal Learning Brain.
    /// </summary>
    [CustomEditor(typeof(InternalLearningBrain))]
    public class InternalLearningBrainEditor : BrainEditor
    {
        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Internal Learning Brain", EditorStyles.boldLabel);
            base.OnInspectorGUI();
        }
    }
}
