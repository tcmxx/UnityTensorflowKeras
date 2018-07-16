using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;


public static class EditorUtils
{
    public static void SaveTextureToPNGFile(Texture2D tex, string defaultName = "image")
    {
        string path = EditorUtility.SaveFilePanelInProject("Save png", defaultName, "png",
                "Please enter a file name to save the texture to");
        if (path.Length != 0)
        {
            byte[] pngData = tex.EncodeToPNG();
            if (pngData != null)
            {
                File.WriteAllBytes(path, pngData);

                // As we are saving to the asset folder, tell Unity to scan for modified or new assets
                AssetDatabase.Refresh();
            }
        }
    }

    [MenuItem("CONTEXT/Renderer/SaveMainTextureToFile")]
    public static void EditorSaveTextureToPNGFile(MenuCommand menuCommand)
    {
        var rend = menuCommand.context as Renderer;
        if (rend.sharedMaterial == null)
        {
            Debug.LogWarning("No material in renderer");
            return;
        }
        var tex = rend.sharedMaterial.mainTexture as Texture2D;
        if (tex == null)
        {
            Debug.LogWarning("No Texture2d in material");
            return;
        }
        string path = EditorUtility.SaveFilePanelInProject("Save png", "image", "png",
                "Please enter a file name to save the texture to");
        if (path.Length != 0)
        {
            var tempText = Images.GetReadableTextureFromUnreadable(tex);
            byte[] pngData = tempText.EncodeToPNG();
            GameObject.DestroyImmediate(tempText);
            if (pngData != null)
            {
                File.WriteAllBytes(path, pngData);

                // As we are saving to the asset folder, tell Unity to scan for modified or new assets
                AssetDatabase.Refresh();
            }
        }
    }
}

