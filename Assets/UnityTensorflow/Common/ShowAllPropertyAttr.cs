// This is not an editor script. The property attribute class should be placed in a regular script file.
using System.Collections.Generic;
using UnityEngine;

public class ShowAllPropertyAttr : PropertyAttribute
{

    public List<string> notShowNamesList = new List<string>();
    public ShowAllPropertyAttr()
    {
    }

    public ShowAllPropertyAttr(params string[] notShowNames)
    {

        notShowNamesList.AddRange(notShowNames);
    }
    
}