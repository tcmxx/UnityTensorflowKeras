using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using UnityEditor.SceneManagement;

[CustomPropertyDrawer(typeof(UnityNetwork.SimpleDenseLayerDef))]
public class SimpleDenseLayerDefPropertyDrawer : PropertyDrawer
{
    // Draw the property inside the given rect
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        // Using BeginProperty / EndProperty on the parent property means that
        // prefab override logic works on the entire property.
        EditorGUI.BeginProperty(position, label, property);
        var layer = EditorUtils.GetActualObjectForSerializedProperty<UnityNetwork.SimpleDenseLayerDef>(fieldInfo, property) ;
        if (layer != null && layer.initialScale == 0)
        {
            layer.Reinitialize();
        }
        // Draw label
        position = EditorGUI.PrefixLabel(position, GUIUtility.GetControlID(FocusType.Passive), label);

        // Don't make child fields be indented
        var indent = EditorGUI.indentLevel;
        EditorGUI.indentLevel = 0;

        float standardSpacing = EditorGUI.GetPropertyHeight(property, label, false);

        // Calculate rects
        var sizeRect = new Rect(position.x, position.y, position.width, standardSpacing);
        var useBiasRect  = new Rect(position.x, position.y + EditorGUIUtility.standardVerticalSpacing + standardSpacing, position.width - position.width, standardSpacing);
        var initScaleRect = new Rect(position.x, useBiasRect.y + EditorGUIUtility.standardVerticalSpacing + standardSpacing, position.width, standardSpacing);
        var activationRect = new Rect(position.x, initScaleRect.y + EditorGUIUtility.standardVerticalSpacing + standardSpacing, position.width, standardSpacing);
        //EditorGUIUtility.labelWidth = 50;
        // Draw fields - passs GUIContent.none to each so they are drawn without labels
        EditorGUI.PropertyField(sizeRect, property.FindPropertyRelative("size"), new GUIContent("Size"));
        EditorGUI.PropertyField(useBiasRect, property.FindPropertyRelative("useBias"), new GUIContent("UseBias"));

        EditorGUI.PropertyField(initScaleRect, property.FindPropertyRelative("initialScale"), new GUIContent("InitialScale"));
        
        EditorGUI.PropertyField(activationRect, property.FindPropertyRelative("activationFunction"), new GUIContent("Activation"));
        // Set indent back to what it was
        EditorGUI.indentLevel = indent;
        EditorGUI.EndProperty();

    }


    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        float standardSpacing = EditorGUI.GetPropertyHeight(property, label, false);
        return (standardSpacing + EditorGUIUtility.standardVerticalSpacing)*4;
    }
    
}