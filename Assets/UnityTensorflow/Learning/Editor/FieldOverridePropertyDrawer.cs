using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
using System.Linq;

[CustomPropertyDrawer(typeof(TrainerParamOverride.FieldOverride))]
public class FieldOverrideDrawer : PropertyDrawer
{
    // Draw the property inside the given rect
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {


        //EditorGUI
        EditorGUI.BeginChangeCheck();

        //if (property.objectReferenceValue == null)
        //{
        //EditorGUI.PropertyField(position, property, true);
        //return;
        //}


        var nameProp = property.FindPropertyRelative("name");
        float namePropHeight = EditorGUI.GetPropertyHeight(nameProp, label, true);
        float prevHeight = 0;
        Rect rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, namePropHeight);
        EditorGUI.PropertyField(rect, nameProp, new GUIContent("parameter name"));
        prevHeight += rect.height + EditorGUIUtility.standardVerticalSpacing;

        var methodProp = property.FindPropertyRelative("method");
        float methodPropHeight = EditorGUI.GetPropertyHeight(methodProp, new GUIContent("method"), true);
        rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, methodPropHeight);
        EditorGUI.PropertyField(rect, methodProp, new GUIContent("method"));
        prevHeight += rect.height + EditorGUIUtility.standardVerticalSpacing;


        var obj = fieldInfo.GetValue(property.serializedObject.targetObject);
        TrainerParamOverride.FieldOverride ov = obj as TrainerParamOverride.FieldOverride;
        if (obj.GetType().IsArray || obj.GetType() == typeof(List<TrainerParamOverride.FieldOverride>))
        {
            var index = Convert.ToInt32(new string(property.propertyPath.Where(c => char.IsDigit(c)).ToArray()));
            ov = ((TrainerParamOverride.FieldOverride[])obj)[index];
        }


        if (ov.method == TrainerParamOverride.Method.AnimationCurve)
        {
            var curveProp = property.FindPropertyRelative("curve");
            float curvePropHeight = EditorGUI.GetPropertyHeight(curveProp, new GUIContent("curve"), true);
            
            rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, curvePropHeight);
            EditorGUI.PropertyField(rect, curveProp, new GUIContent("curve"));
            prevHeight += rect.height + EditorGUIUtility.standardVerticalSpacing;
        }
        else if(ov.method == TrainerParamOverride.Method.PolynomialDecay)
        {
            var endProp = property.FindPropertyRelative("endValue");
            float endPropHeight = EditorGUI.GetPropertyHeight(endProp, new GUIContent("endValue"), true);

            rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, endPropHeight);
            EditorGUI.PropertyField(rect, endProp, new GUIContent("endValue"));
            prevHeight += rect.height + EditorGUIUtility.standardVerticalSpacing;

            var powerProp = property.FindPropertyRelative("power");
            float powerPropHeight = EditorGUI.GetPropertyHeight(powerProp, new GUIContent("power"), true);

            rect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, powerPropHeight);
            EditorGUI.PropertyField(rect, powerProp, new GUIContent("power"));
            prevHeight += rect.height + EditorGUIUtility.standardVerticalSpacing;
        }
        
        EditorGUI.EndChangeCheck();
    }



    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        var obj = fieldInfo.GetValue(property.serializedObject.targetObject);
        TrainerParamOverride.FieldOverride ov = obj as TrainerParamOverride.FieldOverride;
        if (obj.GetType().IsArray || obj.GetType() == typeof(List<TrainerParamOverride.FieldOverride>))
        {
            var index = Convert.ToInt32(new string(property.propertyPath.Where(c => char.IsDigit(c)).ToArray()));
            ov = ((TrainerParamOverride.FieldOverride[])obj)[index];
        }

        float prevHeight = 0;

        var nameProp = property.FindPropertyRelative("name");
        prevHeight += EditorGUI.GetPropertyHeight(nameProp, label, true) + EditorGUIUtility.standardVerticalSpacing*2;

        var methodProp = property.FindPropertyRelative("method");
        prevHeight += EditorGUI.GetPropertyHeight(methodProp, label, true) +EditorGUIUtility.standardVerticalSpacing * 2;

        if (ov.method == TrainerParamOverride.Method.AnimationCurve)
        {
            var curveProp = property.FindPropertyRelative("curve");
            prevHeight += EditorGUI.GetPropertyHeight(curveProp, label, true) + EditorGUIUtility.standardVerticalSpacing * 2;
        }
        else if (ov.method == TrainerParamOverride.Method.PolynomialDecay)
        {
            var prop = property.FindPropertyRelative("endValue");
            prevHeight += EditorGUI.GetPropertyHeight(prop, label, true) + EditorGUIUtility.standardVerticalSpacing * 2;

            prop = property.FindPropertyRelative("power");
            prevHeight += EditorGUI.GetPropertyHeight(prop, label, true) + EditorGUIUtility.standardVerticalSpacing * 2;
        }


        return prevHeight;
    }
}
