using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;


using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System;

using NWH;

[ExecuteInEditMode]
public partial class Grapher : EditorWindow
{
    private static int frameCounter = 0;
    private static bool wasPlayingOrPaused = false;
    private static string consoleString = "";
    public static float sharedVerticalResolution = 10;
    public static bool beingManuallyAdjusted = false;

    public static string grapherPath = "";
    public static bool validGrapherPath = true;

    public static string SessionName { get; set; } = "";

    private enum MouseState
    {
        Up, Down
    }

    private Vector3 mousePosition;
    private MouseState mouseState;
    private bool mouseInside;

    private static List<Channel> channels = new List<Channel>();

    public static Grapher grapherWindow;

    [MenuItem("Window/Grapher %g")]
    public static void Init()
    {
        grapherWindow = (Grapher)GetWindow(typeof(Grapher));
        grapherWindow.Show();
    }

    void OnEnable()
    {
        SetupStyles();

        // Check for first run
        if (!EditorPrefs.HasKey("GrapherFirstRun"))
        {
            Debug.Log("Read IMPORTANT_README!.txt in the Grapher folder. As the name says, it is important :). This message is shown only once.");
            EditorPrefs.SetBool("GrapherFirstRun", false);
        }

        // Add additional update functions
        EditorApplication.update += Update;
        EditorApplication.update += TimeKeeper.Update;
    }

    void OnDisable()
    {
        // Remove additional update functions
        EditorApplication.update -= Update;
        EditorApplication.update -= TimeKeeper.Update;
    }

    void Update()
    {
        frameCounter++;

        // Detect when application stops playing
        if (EditorApplication.isPlaying || EditorApplication.isPaused)
        {
            wasPlayingOrPaused = true;
        }
        else
        {
            if (wasPlayingOrPaused) OnStopped();
            wasPlayingOrPaused = false;
            SetupStyles();
        }

        // Find max auto scale value and check for manual adjustment
        float max = 0;
        foreach (Channel ch in channels)
        {
            if (ch.rangeSlider != 0)
                ch.AutoScale = false;

            if (ch.AutoScale && ch.autoScaleResolution > max && ch.Show)
                max = ch.autoScaleResolution;
        }

        // Adjust scale values
        foreach (Channel ch in channels)
        {
            if (ch.AutoScale)
            {
                if (GraphSettings.SharedVerticalResolution == 0)
                {
                    ch.verticalResolution = ch.autoScaleResolution;
                }
                else
                {
                    ch.verticalResolution = max;
                }
            }
        }
    }

    void OnGUI()
    {
        // Style for toggle button
        toggleButtonStyle = new GUIStyle(GUI.skin.button);
        toggleButtonStyle.margin = new RectOffset(4, 4, 2, 2);
        toggleButtonStyle.padding = new RectOffset(2, 2, 2, 2);

        // Get mouse state and position
        mousePosition = Event.current.mousePosition;

        // Determine LMB click state
        if (Event.current.type == EventType.MouseUp || Input.GetMouseButtonUp(0))
            mouseState = MouseState.Up;
        else if (Event.current.type == EventType.MouseDown || Input.GetMouseButtonDown(0))
            mouseState = MouseState.Down;

        // Check if mouse inside graph
        if (graphRect.Contains(mousePosition) && mousePosition.x < graphRect.width)
        {
            mouseInside = true;
        }
        else
        {
            mouseInside = false;
        }

        // Draw GUI
        Handles.BeginGUI();

        DrawStatic();
        // Avoid double calculations during layout and repaint
        if (Event.current.type == EventType.Repaint)
            DrawGraph();
        DrawRules();
        DrawBottomControls();

        Handles.EndGUI();

        // Force GUI repaint every frame
        Repaint();
    }


    /// <summary>
    /// Convert world position to graph position.
    /// </summary>
    private Vector2 WToG(Vector2 pos)
    {
        Vector2 r = new Vector3();
        r.x = GraphSettings.graphMargins.x + graphRect.width - pos.x;
        r.y = GraphSettings.graphMargins.y + graphRect.height - pos.y;
        RectClamp(graphRect, r);
        return r;
    }

    /// <summary>
    /// Convert graph position to world position.
    /// </summary>
    private Vector2 GToW(Vector2 pos)
    {
        Vector2 r = new Vector3();
        r.x = -(pos.x - GraphSettings.graphMargins.x - graphRect.width);
        r.y = -(pos.y - GraphSettings.graphMargins.y - graphRect.height);
        return r;
    }

    /// <summary>
    /// Clamp a point to inside of a rect.
    /// </summary>
    private Vector2 RectClamp(Rect rect, Vector2 point)
    {
        point.x = Mathf.Clamp(point.x, rect.x, rect.x + rect.width);
        point.y = Mathf.Clamp(point.y, rect.y, rect.y + rect.height);
        return new Vector2(point.x, point.y);
    }

    /// <summary>
    /// Reset Grapher.
    /// </summary>
    public static void Reset()
    {
        try
        {
            replayFiles.Clear();
            channels.Clear();
        }
        catch { }
    }

    /// <summary>
    /// Add new channel to graph.
    /// </summary>
    private static Channel AddChannel()
    {
        Channel ch;
        channels.Add(ch = new Channel(channels.Count));
        ch.Init();

        return ch;
    }

    /// <summary>
    /// Main Log function.
    /// </summary>
    public static void Log(object obj, string name, Color color, float x, string datetimeString)
    {
        // Check for vectors
        Type type = obj.GetType();

        if (type == typeof(Vector2))
        {
            Vector2 v = (Vector2)obj;
            Log(v.x, name + " X", color, x);
            Log(v.y, name + " Y", color, x);
            return;
        }
        else if (type == typeof(Vector3))
        {
            Vector3 v = (Vector3)obj;
            Log(v.x, name + " X", color, x, datetimeString);
            Log(v.y, name + " Y", color, x, datetimeString);
            Log(v.z, name + " Z", color, x, datetimeString);
            return;
        }
        else if (type == typeof(Vector4))
        {
            Vector3 v = (Vector3)obj;
            Log(v.x, name + " X", color, x, datetimeString);
            Log(v.y, name + " Y", color, x, datetimeString);
            Log(v.z, name + " Z", color, x, datetimeString);
            return;
        }
        else if (typeof(IEnumerable).IsAssignableFrom(type))
        {
            IEnumerable enumerable = (IEnumerable)obj;
            int n = 0;
            foreach (object item in enumerable)
            {
                Log(item, name + "[" + n + "]", color, x, datetimeString);
                n++;
            }
            return;
        }

        float d = ToFloat(obj);

        Channel ch = null;

        float maxX = 0, minX = 0;
        foreach (var c in channels)
        {
            maxX = Mathf.Max(c.MaxX, maxX);
            minX = Mathf.Min(c.MinX, minX);
        }
        if ((ch = channels.Find(i => i.name == name)) == null)
        {
            ch = AddChannel();
            ch.name = name;
            ch.color = color;
            SetChannel(color, name);
            ch.XScale = GraphSettings.HorizontalResolution > 0 ? GraphSettings.HorizontalResolution : (maxX - minX);

            // Self get
            ch.verticalResolution = ch.verticalResolution;
            ch.LogToFile = ch.LogToFile;
        }

        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            if (obj != null)
            {
                if (ch.lastFrame == frameCounter && EditorApplication.isPlaying)
                {
                    //Debug.LogWarning("Grapher received 2 values in the same frame. You might be logging to the same channel name twice in a frame. Only the first value has been accepted.");
                }
                else
                {
                    ch.newestObj = obj;
                    ch.Enqueue(d, datetimeString, x);
                }
                ch.lastFrame = frameCounter;
            }
        }
    }

    public static void Log(object obj, string name, float x)
    {
        Color chColor = GetChannelColor(name);
        SetChannel(chColor, name);
        Log(obj, name, chColor, x, TimeKeeper.DateTimeString);
    }

    public static void Log(object obj, string name, Color color, float x)
    {
        Log(obj, name, color, x, TimeKeeper.DateTimeString);
    }


    /// <summary>
    /// Called when Editor Application is stopped.
    /// </summary>
    private static void OnStopped()
    {
        if (GraphSettings.SaveWhenStopped != 0)
            SaveToFiles(SessionName);
    }


    public static void SaveToFiles(string sessionName = "", string directoryName = "")
    {
        // Generate session filename
        string sessionFilename = "";

        // Make names unique or keep the same name
        if (string.IsNullOrEmpty(sessionName) && GraphSettings.OverwriteFiles == 0)
        {
            IncrementRecordingSessionID();
            sessionFilename = "S" + GetRecordingSessionID() + "_" + GetFilenameTimestamp() + ".ses";
        }
        else if (string.IsNullOrEmpty(sessionName))
        {
            sessionFilename = "S" + GetRecordingSessionID() + ".ses";
        }
        else if (GraphSettings.OverwriteFiles == 0)
        {
            sessionFilename = FileHandler.CleanFilename(sessionName) + "_" + GetFilenameTimestamp() + ".ses";
        }
        else
        {
            sessionFilename = FileHandler.CleanFilename(sessionName) + ".ses";
        }

        // Log to file
        string sessionList = "";

        for (int i = 0; i < channels.Count; i++)
        {
            Channel ch = channels[i];

            if (ch.LogToFile)
            {
                // Generate filename
                string filename = "";
                if (string.IsNullOrEmpty(sessionName) && GraphSettings.OverwriteFiles == 1)
                {
                    filename = FileHandler.CleanFilename(ch.name) + ".csv";
                }
                else if (string.IsNullOrEmpty(sessionName))
                {
                    filename = "S" + GetRecordingSessionID() + "_" + GetFilenameTimestamp() + "_" + FileHandler.CleanFilename(ch.name) + ".csv";
                }
                else if (GraphSettings.OverwriteFiles == 0)
                {
                    filename = FileHandler.CleanFilename(sessionName) + "_" + GetFilenameTimestamp() + "_" + FileHandler.CleanFilename(ch.name) + ".csv";
                }
                else
                {
                    filename = FileHandler.CleanFilename(sessionName) + "_" + FileHandler.CleanFilename(ch.name) + ".csv";
                }

                // Write header
                string header = "";
                header += ch.name + "," + ch.verticalResolution + "," + ch.color.r + "," + ch.color.g + "," + ch.color.b + Environment.NewLine;
                FileHandler.WriteStringToCSV(header, filename, directoryName);

                // Append samples
                FileHandler.AppendSamplesToCSV(ch.rawSampleList, filename, directoryName);

                // Append to session
                sessionList += filename;
                if (i != channels.Count - 1) sessionList += Environment.NewLine;
            }
        }

        // Add channel filename to session filename list
        if (sessionList != "") FileHandler.WriteStringToCSV(sessionList, sessionFilename, directoryName);
    }



    // ***********************
    // *      HELPERS        *
    // ***********************

    private static float ToFloat(object d)
    {
        Type type = d.GetType();
        float x = 0f;

        if (type == typeof(float))
        {
            x = (float)d;
        }
        else if (type == typeof(int))
        {
            x = (float)(int)d;
        }
        else if (type.IsEnum)
        {
            x = (float)(int)d;
        }
        else
        {
            try
            {
                x = (float)d;
            }
            catch
            {
                Debug.LogWarning("Grapher: Variable type you are trying to graph is not recognized.");
                return x;
            }
        }

        return x;
    }

    private static bool IsNullOrValue<T>(T value)
    {
        return object.Equals(value, default(T));
    }

    private static string GetFilenameTimestamp()
    {
        return "_" + DateTime.Now.ToString("ddMMyyyy") + DateTime.Now.ToString("HHmmss");
    }

    private static int GetRecordingSessionID()
    {
        string key = "GrapherSessionID";
        if (EditorPrefs.HasKey(key))
        {
            return EditorPrefs.GetInt(key);
        }
        else
        {
            EditorPrefs.SetInt(key, 0);
            return 0;
        }
    }

    private static int IncrementRecordingSessionID()
    {
        string key = "GrapherSessionID";
        if (EditorPrefs.HasKey(key))
        {
            int id = EditorPrefs.GetInt(key) + 1;
            EditorPrefs.SetInt(key, id);
            return id;
        }
        else
        {
            EditorPrefs.SetInt(key, 0);
            return 0;
        }
    }
}
#else
public partial class Grapher{
    /// <summary>
    /// Main Log function.
    /// </summary>
    public static void Log(object obj, string name, Color color, float x, string datetimeString)
    {
    }

    public static void Log(object obj, string name, float x)
    {
    }

    public static void Log(object obj, string name, Color color, float x)
    {
    }

}

#endif