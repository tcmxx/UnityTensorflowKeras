using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

[CustomPropertyDrawer(typeof(OptimizerCreator))]
public class OptmizerCreatorPropertyDrawer : PropertyDrawer
{
    protected OptimizerCreator.OptimizerType prevType;
    bool showParams;
    // Draw the property inside the given rect
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        //SerializedObject childObj = new SerializedObject(property.objectReferenceValue);
        // SerializedProperty ite = childObj.GetIterator();

        EditorGUI.BeginChangeCheck();

        OptimizerCreator opt = fieldInfo.GetValue(property.serializedObject.targetObject) as OptimizerCreator;

        var typeProp = property.FindPropertyRelative("optimizerType");
        float typePropHeight = EditorGUI.GetPropertyHeight(typeProp, label, true);

        float prevHeight = 0;
        Rect rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, typePropHeight);
        var newType = opt.optimizerType;
        if (newType != prevType)
        {
            prevType = newType;
            opt.parameterList.Clear();
        }
        EditorGUI.PropertyField(rect, typeProp, label);
        prevHeight = EditorGUI.GetPropertyHeight(typeProp, label, true) + EditorGUIUtility.standardVerticalSpacing;

        var indent = EditorGUI.indentLevel;
        EditorGUI.indentLevel += 1;

        rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, typePropHeight);
        showParams = EditorGUI.Foldout(rect, showParams, "Optimizer Initial Parameters");

        prevHeight += typePropHeight + EditorGUIUtility.standardVerticalSpacing;

        Type type = OptimizerCreator.TypeFromEnum(opt.optimizerType);
        var ctors = type.GetConstructors();
        var ctor = ctors[0];    //assume there is only one constructor
        var paramInfos = ctor.GetParameters();
        int i = 0;
        foreach (var param in ctor.GetParameters())
        {
            rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, typePropHeight);

            if (opt.parameterList.Count > i && (param.ParameterType == typeof(float)
                || param.ParameterType == typeof(double)
                || param.ParameterType == typeof(int)))
            {
                if (showParams)
                    opt.parameterList[i] = EditorGUI.FloatField(rect, param.Name, opt.parameterList[i]);
            }
            else if (opt.parameterList.Count > i && param.ParameterType == typeof(bool))
            {
                if (showParams)
                    opt.parameterList[i] = EditorGUI.Toggle(rect, param.Name, opt.parameterList[i] > 0) ? 1 : 0;
            }
            else if ((param.ParameterType == typeof(float)
                || param.ParameterType == typeof(double)
                || param.ParameterType == typeof(int)) && param.HasDefaultValue)
            {
                opt.parameterList.Add((float)(double)param.RawDefaultValue);
            }
            else if (param.ParameterType == typeof(bool) && param.HasDefaultValue)
            {
                opt.parameterList.Add((bool)param.RawDefaultValue ? 1 : 0);
            }
            else if (opt.parameterList.Count <= i)
            {
                opt.parameterList.Add(0);
                if (showParams)
                    EditorGUI.LabelField(rect, param.Name + ": NA");
            }
            prevHeight += typePropHeight + EditorGUIUtility.standardVerticalSpacing;
            i++;
        }
        //childObj.ApplyModifiedProperties();
        EditorGUI.indentLevel = indent;

        if (EditorGUI.EndChangeCheck()) EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
    }



    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {

        OptimizerCreator opt = fieldInfo.GetValue(property.serializedObject.targetObject) as OptimizerCreator;

        //EditorGUI
        var typeProp = property.FindPropertyRelative("optimizerType");

        float typePropHeight = EditorGUI.GetPropertyHeight(typeProp, label, true);
        float prevHeight = EditorGUI.GetPropertyHeight(typeProp, label, true) + EditorGUIUtility.standardVerticalSpacing;
        prevHeight += typePropHeight + EditorGUIUtility.standardVerticalSpacing;


        Type type = OptimizerCreator.TypeFromEnum(opt.optimizerType);
        var ctors = type.GetConstructors();
        var ctor = ctors[0];    //assume there is only one constructor
        var paramInfos = ctor.GetParameters();

        if (showParams)
            for (int i = 0; i < paramInfos.Length; ++i)
            {
                prevHeight += typePropHeight + EditorGUIUtility.standardVerticalSpacing;
            }

        return prevHeight;
    }
}
