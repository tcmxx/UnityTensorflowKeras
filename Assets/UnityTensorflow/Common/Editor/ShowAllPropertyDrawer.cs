using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;

[CustomPropertyDrawer(typeof(ShowAllPropertyAttr),true)]
public class ShowAllPropertyDrawer : PropertyDrawer
{
    // Draw the property inside the given rect
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {


        //EditorGUI

        if (property.objectReferenceValue == null)
        {
            EditorGUI.PropertyField(position, property, true);
            return;
        }

        ShowAllPropertyAttr showAll = attribute as ShowAllPropertyAttr;
        var notShowNames = showAll.notShowNamesList;

        EditorGUI.LabelField(position, "", GUI.skin.horizontalSlider);
        float standardSpacing = EditorGUI.GetPropertyHeight(property, label, false);
        float prevHeight = standardSpacing + EditorGUIUtility.standardVerticalSpacing;

        Rect anotherRect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, standardSpacing);
        EditorGUI.PropertyField(anotherRect, property, true);
        prevHeight = prevHeight + anotherRect.height + EditorGUIUtility.standardVerticalSpacing;



        var indent = EditorGUI.indentLevel;
        EditorGUI.indentLevel += 1;

        SerializedObject childObj = new SerializedObject(property.objectReferenceValue);
        SerializedProperty ite = childObj.GetIterator();
        ite.NextVisible(true);
        while (ite.NextVisible(false))
        {
            if (notShowNames.Contains(ite.name))
                continue;
            Rect newRect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, EditorGUI.GetPropertyHeight(ite, label, true));
            prevHeight += newRect.height + EditorGUIUtility.standardVerticalSpacing;
            EditorGUI.PropertyField(newRect, ite,true);
        }

        EditorGUI.indentLevel = indent;
        anotherRect = new Rect(position.x, position.y + prevHeight + EditorGUIUtility.standardVerticalSpacing, position.width, standardSpacing);
        EditorGUI.LabelField(anotherRect, "", GUI.skin.horizontalSlider);

        prevHeight+= anotherRect.height + EditorGUIUtility.standardVerticalSpacing;
        childObj.ApplyModifiedProperties();

        
    }



    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        if (property.objectReferenceValue == null)
            return base.GetPropertyHeight(property, label);

        ShowAllPropertyAttr showAll = attribute as ShowAllPropertyAttr;
        var notShowNames = showAll.notShowNamesList;

        SerializedObject childObj = new SerializedObject(property.objectReferenceValue);
        SerializedProperty ite = childObj.GetIterator();

        float standardSpacing = EditorGUI.GetPropertyHeight(property, label, false);
        float prevHeight = standardSpacing + EditorGUIUtility.standardVerticalSpacing;

        prevHeight = prevHeight + standardSpacing + EditorGUIUtility.standardVerticalSpacing;
        
        
        ite.NextVisible(true);
        while (ite.NextVisible(false))
        {
            if (notShowNames.Contains(ite.name))
                continue;
            prevHeight += EditorGUI.GetPropertyHeight(ite, label, true) + EditorGUIUtility.standardVerticalSpacing;
        }
        prevHeight += standardSpacing + EditorGUIUtility.standardVerticalSpacing;

        return prevHeight;
    }
}
