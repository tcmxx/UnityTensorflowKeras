using UnityEditor;
using UnityEngine;

namespace NWH
{
    public class SettingsWindow : EditorWindow
    {
        [MenuItem("Window/Grapher Settings")]
        public static void Init()
        {
            SettingsWindow window = (SettingsWindow)EditorWindow.GetWindow(typeof(SettingsWindow));
            window.Show();
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(0, 0, position.width, position.height));

            GUILayout.Space(3);
            GUILayout.Label("Graph", EditorStyles.boldLabel);
            GUILayout.Space(3);

            // Time window
            GraphSettings.HorizontalResolution = FloatField("Horizontal resolution (time)", GraphSettings.HorizontalResolution, 0f, 30);

            // Shared Y Range
            GraphSettings.SharedVerticalResolution = (int)FloatField("Share vertical resolution", GraphSettings.SharedVerticalResolution, 0, 1);

            // Line style selection
            GraphSettings.GraphLineStyle = (int)FloatField("Line style", GraphSettings.GraphLineStyle, 0, 1);

            GUILayout.Space(3);
            GUILayout.Label("Logging", EditorStyles.boldLabel);
            GUILayout.Space(3);

            // Overwrite existing files
            GraphSettings.OverwriteFiles = (int)FloatField("Overwrite existing files", GraphSettings.OverwriteFiles, 0, 1);

            GUILayout.Space(3);
            // save to files when stopped
            GraphSettings.SaveWhenStopped = (int)FloatField("Save to files when stopped", GraphSettings.SaveWhenStopped, 0, 1);

            GUILayout.Space(3);
            GUILayout.Label("Defaults", EditorStyles.boldLabel);
            GUILayout.Space(3);


            // Default Y Range
            GraphSettings.DefaultVerticalResolution = FloatField("Vertical resolution", GraphSettings.DefaultVerticalResolution, 1, Mathf.Infinity);

            // Default log to file
            GraphSettings.DefaultLogToFile = (int)FloatField("Log To File", GraphSettings.DefaultLogToFile, 0, 1);

            // Default log to console
            GraphSettings.DefaultLogToConsole = (int)FloatField("Log To Console", GraphSettings.DefaultLogToConsole, 0, 1);

            GUILayout.Space(10);

            GUILayout.EndArea();
        }

        public float FloatField(string label, float value, float min, float max)
        {
            float result;
            GUILayout.BeginHorizontal();
            GUILayout.Space(5);
            GUILayout.Label(label, GUILayout.Width(160));
            result = float.Parse(GUILayout.TextField(value.ToString(), 10, GUILayout.Width(100)));
            if (GUILayout.Button("-", GUILayout.Width(20)))
            {
                result -= 1;
            }
            else if (GUILayout.Button("+", GUILayout.Width(20)))
            {
                result += 1;
            }
            GUILayout.EndHorizontal();
            return Mathf.Clamp(result, min, max);
        }
    }
}
