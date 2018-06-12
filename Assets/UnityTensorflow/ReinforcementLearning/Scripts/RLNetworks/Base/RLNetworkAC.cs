using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// actor critic network abstract class
/// </summary>
public abstract class RLNetworkAC : ScriptableObject {



    public abstract void BuildNetwork(Tensor inVectorstate, List<Tensor> inVisualState, Tensor inMemery, Tensor inPrevAction, int outActionSize,SpaceType actionSpace,
        out Tensor outAction, out Tensor outValue);


    public abstract List<Tensor> GetWeights();

    public virtual void OnInspector()
    {
#if UNITY_EDITOR
        Editor.CreateEditor(this).OnInspectorGUI();
#endif
    }
}
