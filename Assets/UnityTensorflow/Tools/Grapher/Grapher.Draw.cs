using UnityEngine;

#if UNITY_EDITOR

using UnityEditor;
using System.Collections.Generic;
using System;
using System.IO;
using NWH;

public partial class Grapher : EditorWindow
{
    private static Rect graphRect;
    private Rect borderRect;
    private Rect toolbarRect;

    private GUIStyle toggleButtonStyle;

    private static Texture2D showTexture;
    private static Texture2D autoScaleTexture;
    private static Texture2D logTexture;
    private static Texture2D consoleTexture;

    private void SetupStyles()
    {
        // Get textures
        showTexture = GraphSettings.GetTextureFrom64(GraphSettings.showIcon64);
        autoScaleTexture = GraphSettings.GetTextureFrom64(GraphSettings.scaleIcon64);
        logTexture = GraphSettings.GetTextureFrom64(GraphSettings.logIcon64);
        consoleTexture = GraphSettings.GetTextureFrom64(GraphSettings.consoleIcon64);

        // Button Background
        GUIStyleState bg = new GUIStyleState();
        bg.background = GenerateMonotoneTexture(new Vector2(32, 32), GraphSettings.buttonBackgroundColor);

        // Button Hover
        GUIStyleState hover = new GUIStyleState();
        hover.background = GenerateMonotoneTexture(new Vector2(32, 32), GraphSettings.buttonHoverColor);

        // Button Active
        GUIStyleState active = new GUIStyleState();
        active.background = GenerateMonotoneTexture(new Vector2(32, 32), GraphSettings.buttonActiveColor);
    }

    /// <summary>
    /// Draw horizontal and vertical rule lines
    /// </summary>
    private void DrawRules()
    {
        // Draw rules
        if (graphRect.Contains(mousePosition))
        {
            Handles.color = GraphSettings.ruleLineColor;

            // Horizontal
            if (GraphSettings.showHorizontalRule)
            {
                Vector2 horizontalRuleStart = RectClamp(graphRect, new Vector2(0, mousePosition.y));
                Vector2 horizontalRuleEnd = RectClamp(graphRect, new Vector2(position.width, mousePosition.y));
                Handles.DrawLine(new Vector3(horizontalRuleStart.x, horizontalRuleStart.y), new Vector3(horizontalRuleEnd.x, horizontalRuleEnd.y));
            }

            //Vertical
            if (GraphSettings.showVerticalRule)
            {
                Vector2 verticalRuleStart = RectClamp(graphRect, new Vector2(mousePosition.x, 0));
                Vector2 verticalRuleEnd = RectClamp(graphRect, new Vector2(mousePosition.x, position.height));
                Handles.DrawLine(new Vector3(verticalRuleStart.x, verticalRuleStart.y), new Vector3(verticalRuleEnd.x, verticalRuleEnd.y));
            }
        }
    }

    /// <summary>
    /// Draw graph lines / points and tags.
    /// </summary>
    private void DrawGraph()
    {
        // Draw graph
        float maxXAll = 0, minXAll = 0;
        foreach (var c in channels)
        {
            maxXAll = Mathf.Max(c.MaxX, maxXAll);
            minXAll = Mathf.Min(c.MinX, minXAll);
        }
        float xRangeOverall = maxXAll - minXAll;

        foreach (Channel ch in channels)
        {
            if (ch.Show)
            {
                // Update time scale
                ch.XScale = GraphSettings.HorizontalResolution > 0 ? GraphSettings.HorizontalResolution : xRangeOverall;

                Vector3 graphSpaceMousePos = mousePosition;
                ch.pointAtMousePosition = Vector3.zero;
                Sample sampleAtMousePosition = null;

                if (ch.sampleNo > 0)
                {
                    Sample[] samples = ch.GetSamples();

                    List<Vector3> points = new List<Vector3>();

                    /*float newestSampleTime = Mathf.Max(0f, samples[samples.Length - 1].time);
                    float oldestSampleTime = Mathf.Max(0f, samples[0].time);
                    float timeSpan = newestSampleTime - oldestSampleTime;
                    timeSpan = Mathf.Clamp(timeSpan, 0f, ch.TimeScale); */

                    // Determine scale
                    float xScale = graphRect.width / ch.XScale;
                    float yScale = (ch.YMax / (ch.verticalResolution / 2f)) * ((graphRect.height / 2f) / ch.YMax) * GraphSettings.autoScalePercent;

                    // Signal offset
                    float xOffset = 0f;
                    float yOffset = graphRect.height / 2f;

                    float graphXEnd = GraphSettings.graphMargins.x + graphRect.width;
                    float graphYEnd = GraphSettings.graphMargins.y + graphRect.height;
                    
                    //minTime = 0;
                    int pointCount = 0;

                    for (int i = 0; i < ch.sampleNo; i++)
                    {
                        float value = samples[i].y;
                        float st = samples[i].x;
                        
                        float t = maxXAll - samples[i].x;                

                        // Convert to graph space (faster WToG)
                        float x = graphXEnd - ((t * xScale) + xOffset);
                        float y = graphYEnd - ((value * yScale) + yOffset);

                        // Clamp without function calls
                        x = x < graphRect.x ? graphRect.x : x > (graphRect.x + graphRect.width) ? (graphRect.x + graphRect.width) : x;
                        y = y < graphRect.y ? graphRect.y : y > (graphRect.y + graphRect.height) ? (graphRect.y + graphRect.height) : y;

                        Vector2 point = new Vector2(x, y);
                        points.Add(point);
                        pointCount++;

                        // Check for mouse position
                        if (pointCount > 1 && points[pointCount - 1].x > graphSpaceMousePos.x && points[pointCount - 2].x < graphSpaceMousePos.x)
                        {
                            ch.pointAtMousePosition = new Vector2(x, y);
                            sampleAtMousePosition = samples[i];
                        }
                    }

                    if (pointCount > 0)
                    {
                        // Right-side indicator
                        Handles.color = ch.color;
                        Handles.DrawLine(WToG(new Vector2(0, GToW(points[pointCount - 1]).y)), WToG(new Vector2(-50, GToW(points[pointCount - 1]).y)));

                        // Right side label
                        if(ch.newestSample != null)
                            DrawHorizontalLabel(WToG(new Vector2(4, GToW(points[pointCount - 1]).y + 8)), FloatToCompact(ch.newestSample.y), ch.color);

                        // Draw polyline (fastest)
                        if (GraphSettings.GraphLineStyle == 0)
                        {
                            Handles.DrawAAPolyLine(points.ToArray());
                        }
                        // Draw dots
                        else if (GraphSettings.GraphLineStyle == 1)
                        {
                            if (points.Count > 0)
                            {
                                for (int i = 1; i < points.Count - 1; i++)
                                {
                                    Handles.DrawSolidDisc(points[i], Vector3.forward, 1f);
                                }
                            }
                        }

                        // Intersection marker and labels at mouse position
                        if (mouseInside && sampleAtMousePosition != null)
                        {
                            ch.tagY = sampleAtMousePosition.y;
                            ch.tagX = sampleAtMousePosition.x;

                            // Draw tag at the mouse position with graph value at that point
                            Handles.DrawSolidDisc(ch.pointAtMousePosition, Vector3.forward, 3f);
                            DrawHorizontalTag(ch.pointAtMousePosition, " " + ch.name + " = " + FloatToCompact(ch.tagY), ch.color);

                            // Draw time indicator below graph
                            int textWidth = 80;
                            int outOfBoundsOfset = 0;
                            if (mousePosition.x < textWidth / 2) outOfBoundsOfset += (textWidth / 2) - (int)mousePosition.x;
                            Vector2 timeIndicatorPosition = new Vector2(mousePosition.x - textWidth / 2 + outOfBoundsOfset, graphRect.height + 10);
                            string valueAtPointer = ch.tagX.ToString("0.00");
                            DrawHorizontalLabel(timeIndicatorPosition, valueAtPointer + ", "+sampleAtMousePosition.time, Color.black);
                        }
                    }
                }
            }
            
        }

        // Draw time marker when mouse outside of graph
        if (!mouseInside)
        {
            string label = "(" + minXAll.ToString()+", " + maxXAll.ToString() + ")";
            DrawHorizontalLabel(new Vector2(graphRect.width - 25, graphRect.height + 10), label, Color.black);
        }
    }

    /// <summary>
    /// Draw backgrounds and channel side panels
    /// </summary>
    private void DrawStatic()
    {
        Vector2 labelOffset = new Vector2(-4f, -7f);

        // Draw window background
        Rect bgRect = new Rect(0, 0, position.width, position.height);
        Handles.color = GraphSettings.windowBackgroundColor;
        Handles.DrawSolidRectangleWithOutline(bgRect, GraphSettings.windowBackgroundColor, GraphSettings.windowBackgroundColor);

        // Draw graph border
        Handles.color = GraphSettings.borderBackgroundColor;
        borderRect = new Rect(
            GraphSettings.graphMargins.x, GraphSettings.graphMargins.y,
            position.width - GraphSettings.graphMargins.z + 49f, position.height - GraphSettings.graphMargins.w + 20f
            );
        Handles.DrawSolidRectangleWithOutline(borderRect, GraphSettings.borderBackgroundColor, GraphSettings.borderBackgroundColor);

        // Draw graph background
        Handles.color = GraphSettings.graphBackgroundColor;
        graphRect = new Rect(
            GraphSettings.graphMargins.x, GraphSettings.graphMargins.y,
            position.width - GraphSettings.graphMargins.z, position.height - GraphSettings.graphMargins.w
            );
        Handles.DrawSolidRectangleWithOutline(graphRect, GraphSettings.graphBackgroundColor, Color.white);

        // Draw bottom toolbar
        Handles.color = GraphSettings.panelBackgroundColor;
        toolbarRect = new Rect(
            new Vector2(0, borderRect.height),
            new Vector2(borderRect.width, position.height - borderRect.height)
            );
        Handles.DrawSolidRectangleWithOutline(toolbarRect, GraphSettings.panelBackgroundColor, GraphSettings.panelBackgroundColor);

        // Draw grid
        Handles.color = GraphSettings.gridLineColor;
        Handles.DrawLine(
            new Vector3(GraphSettings.graphMargins.x, GraphSettings.graphMargins.y + graphRect.height / 2f),
            new Vector3(GraphSettings.graphMargins.x + graphRect.width, GraphSettings.graphMargins.y + graphRect.height / 2f));

        // Draw scales
        Handles.color = Color.black;
        Handles.Label(WToG(new Vector2(-5f, graphRect.height / 2f)) + labelOffset, "0"); // Left zero

        for (int i = 0; i < channels.Count; i++)
        {
           DrawChannelSidebar(i);
        }
    }

    /// <summary>
    /// Draw buttons on the bottom of the Grapher window.
    /// </summary>
    private void DrawBottomControls()
    {
        GUILayout.BeginArea(toolbarRect);
        GUILayout.BeginHorizontal();

        GUI.enabled = false;

        if (!EditorApplication.isPlaying && (channels.Count > 0)) GUI.enabled = true;

        if (GUILayout.Button("Save" + Grapher.SessionName))
        {
            string path = FileHandler.BrowserSaveFiles(Grapher.SessionName);
            string dirPath = Path.GetDirectoryName(path);
            string fileName = Path.GetFileNameWithoutExtension(path);
            Grapher.SaveToFiles(fileName, dirPath);
        }
        GUI.enabled = false;
        
        if (!EditorApplication.isPlaying) GUI.enabled = true;
        // SHOW IN EXPLORER BUTTON
        if (GUILayout.Button("Show in Explorer"))
        {
            OpenInFileBrowser.Open(FileHandler.defaultWritePath);
        }

        // SETTINGS BUTTON
        GUI.enabled = true;
        if (GUILayout.Button("Settings"))
        {
            SettingsWindow.Init();
        }

        // RESET BUTTON
        if (GUILayout.Button("Reset"))
        {
            Reset();
        }


        if (!EditorApplication.isPlaying) GUI.enabled = true;
        // OPEN BUTTON
        if (GUILayout.Button("Load"))
        {
            //Reset();
            OpenFiles();
            ReplayInit();
        }

        GUI.enabled = false;
        

        GUILayout.EndHorizontal();
        GUILayout.EndArea();
    }

    private void DrawChannelSidebar(int chId)
    {
        Channel ch = channels[chId];

        // Determine panel position
        float x0 = GraphSettings.graphMargins.x + graphRect.width + 50f;
        Color color = channels[chId].color;

        float segmentHeight = 50f;
        float verticalOffset = segmentHeight * chId;

        // Draw panel
        Handles.color = GraphSettings.panelBackgroundColor;
        Rect panelRect = new Rect(GraphSettings.graphMargins.x + graphRect.width + 50f, verticalOffset, position.width - x0, segmentHeight);
        Handles.DrawSolidRectangleWithOutline(panelRect, GraphSettings.panelBackgroundColor, Color.grey);

        // Draw color header
        float headerHeight = 22;
        Handles.color = GraphSettings.panelHeaderColor;
        Rect statRect = new Rect(panelRect.x, panelRect.y, panelRect.width, headerHeight);
        Handles.DrawSolidRectangleWithOutline(statRect, GraphSettings.panelHeaderColor, Color.grey);

        // Draw marker
        Handles.color = color;
        Rect markerRect = new Rect(panelRect.x + 4, panelRect.y + 4, headerHeight - 8, headerHeight - 8);
        Handles.DrawSolidRectangleWithOutline(markerRect, color, Color.black);

        // Draw name
        GUIStyle titleStyle = new GUIStyle();
        titleStyle.fontStyle = FontStyle.Bold;
        // With type
        //Handles.Label(new Vector2(panelRect.x + headerHeight + 3, panelRect.y + 5f), name + " (" + ch.TypeString + ")", titleStyle);
        Handles.Label(new Vector2(panelRect.x + headerHeight + 3, panelRect.y + 5f), ch.name, titleStyle);

        // Draw buttons
        float buttonsWidth = GraphSettings.chButtonSize * 4f;
        GUILayout.BeginArea(new Rect(panelRect.x + panelRect.width - buttonsWidth * 1.27f, panelRect.y + 2f, panelRect.width - buttonsWidth, GraphSettings.chButtonSize + 5f));
        GUILayout.BeginHorizontal();
        ch.Show = DrawToggleButton(ch.Show, ch.name + "Show", showTexture);
        ch.AutoScale = DrawToggleButton(ch.AutoScale, ch.name + "AutoScale", autoScaleTexture);


        
        ch.LogToFile = DrawToggleButton(ch.LogToFile, ch.name + "LogToFile", logTexture);


        GUILayout.EndHorizontal();
        GUILayout.EndArea();

        GUILayout.BeginArea(new Rect(panelRect.x + 5f, panelRect.y + 26, panelRect.width, panelRect.height - 20));

        // Vertical Resolution
        GUILayout.BeginHorizontal();
        GUILayout.Label("Vert. Res.:", GUILayout.Width(70));

        ch.rangeSlider = GUILayout.HorizontalSlider(ch.rangeSlider, -GraphSettings.sliderSensitivity, GraphSettings.sliderSensitivity, GUILayout.Width(95));

        ch.beingManuallyAdjusted = false;

        try
        {
            float rangeInput = float.Parse(GUILayout.TextField(ch.verticalResolution.ToString("0.00000"), 10, GUILayout.Width(70)));
            if (Mathf.Abs(rangeInput) > ch.verticalResolution + 0.00001f)
            {
                ch.AutoScale = false;
                ch.verticalResolution = rangeInput;
                ch.beingManuallyAdjusted = true;
            }
        }
        catch { Debug.LogWarning("Input is not a number."); }

        // Check for mouse up
        if (mouseState == MouseState.Up)
            ch.rangeSlider = 0;

        ch.verticalResolution += ch.verticalResolution * ch.rangeSlider * TimeKeeper.systemDeltaTime;
        ch.verticalResolution = Mathf.Max(0.00001f, ch.verticalResolution);
        

        GUILayout.EndHorizontal();

        GUILayout.EndArea();
    }

    /// <summary>
    /// Generates persistent button with toggle functionality.
    /// </summary>
    private bool DrawToggleButton(bool toggle, string key, Texture2D tex)
    {
        Color def = GUI.color;
        if (toggle)
            GUI.color = GraphSettings.buttonActiveColor;
        
        // Draw button with supplied style
        if (GUILayout.Button(tex, toggleButtonStyle, GUILayout.Width(GraphSettings.chButtonSize), GUILayout.Height(GraphSettings.chButtonSize)))
        {
            toggle = !toggle;
        }
        GUI.color = def;

        return toggle;
    }

    private void DrawHorizontalTag(Vector2 p, string text, Color color)
    {
        // Prevent jitter by drawing tags only on repaint (avoid layout)
        if (Event.current.type == EventType.Repaint)
        {
            Handles.color = color;
            float charWidth = 6f;
            Rect tagRect = new Rect(p.x + 3, p.y - 7, charWidth * text.Length + 5, 15);
            Handles.DrawSolidRectangleWithOutline(tagRect, Color.white, color);
            Handles.Label(new Vector2(p.x + 4, p.y - 7), text);
        }
    }

    private void DrawHorizontalLabel(Vector2 p, string text, Color color)
    {
        if (Event.current.type == EventType.Repaint)
        {
            Handles.color = color;
            Handles.Label(new Vector2(p.x + 4, p.y - 7), text);
        }
    }

    /// <summary>
    /// Returns single color texture.
    /// </summary>
    private static Texture2D GenerateMonotoneTexture(Vector2 size, Color32 color)
    {
        Texture2D tex = new Texture2D(32, 32);
        Color[] px = tex.GetPixels();
        for (int i = 0; i < px.Length; i++)
            px[i] = GraphSettings.buttonHoverColor;
        tex.SetPixels(px);
        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Converts large values to more readable format.
    /// </summary>
    private static string FloatToCompact(float x)
    {
        string first = " ";
        if(x < 0)
        {
            x = Mathf.Abs(x);
            first = "-";
        }
        string appendix = " ";
        float xAbs = Mathf.Abs(x);

        if (xAbs >= 1000000f)
        {
            x = x / 1000000f;
            appendix = "M";
        }
        else if (xAbs >= 1000000000f)
        {
            x = x / 1000000000f;
            appendix = "G";
        }
        else if (xAbs >= 1000f)
        {
            x = x / 1000f;
            appendix = "k";
        }else if(xAbs < 0.001)
        {
            int e = (int)(-Mathf.Log10(xAbs))+1;
            float multi = Mathf.Pow(10, e);
            x = x * multi;
            appendix = "e" + (-e);
        }

        return first + x.ToString("0.0000") + appendix;
    }

    private static string OutOfScreenFormat(string s, bool outside)
    {
        if (!outside)
            return s;
        else
            return "-.-- ";
    }

    /// <summary>
    /// Saves channel color to EditorPrefs.
    /// </summary>
    private static void SetChannel(Color32 color, string name)
    {
        EditorPrefs.SetInt("GrapherCH" + name + "R", color.r);
        EditorPrefs.SetInt("GrapherCH" + name + "G", color.g);
        EditorPrefs.SetInt("GrapherCH" + name + "B", color.b);
    }

    /// <summary>
    /// Tries to get channel color from EditorPrefs by chanel name. If chanel hasn't been previously
    /// used it generates a random color.
    /// </summary>
    private static Color32 GetChannelColor(string name)
    {
        Color32 res = new Color32();

        // Check for existing key
        if (EditorPrefs.HasKey("GrapherCH" + name + "R"))
        {
            res.r = (byte)EditorPrefs.GetInt("GrapherCH" + name + "R");
            res.g = (byte)EditorPrefs.GetInt("GrapherCH" + name + "G");
            res.b = (byte)EditorPrefs.GetInt("GrapherCH" + name + "B");
        }
        // Key does not exist
        else
        {
            int sum = 0;

            // Get random color, avoid too dark for visibility
            while(sum < 300)
            {
                sum = 0;
                res.r = (byte)UnityEngine.Random.Range(40, 255);
                res.g = (byte)UnityEngine.Random.Range(40, 255);
                res.b = (byte)UnityEngine.Random.Range(40, 255);
                sum = res.r + res.g + res.b;
            }

            SetChannel(new Color32(res.r, res.g, res.b, 255), name);
        }
        res.a = 255;

        return res;
    }
}

#endif