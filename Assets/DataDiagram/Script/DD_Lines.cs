using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DD_Lines : DD_DrawGraphic {

    //[SerializeField]
    //数据折线的粗细
    private float m_Thickness = 5;

    //[SerializeField]
    private bool m_IsShow = true;
    private bool m_CurIsShow = true;//用来判断是否触发UpdateGeometry();

    private List<Vector2> PointList = new List<Vector2>();
    //private const int MaxPointNum = 65535;
    private int CurStartPointSN = 0;

    private DD_DataDiagram m_DataDiagram = null;
    private DD_CoordinateAxis m_Coordinate = null;

    [NonSerialized]
    public string lineName = "";

    public float Thickness {
        get { return m_Thickness; }
        set { m_Thickness = value; }
    }

    public bool IsShow {
        get { return m_IsShow; }
        set {
            if(value != m_IsShow) {
                ///触发OnPopulateMesh的更新
                UpdateGeometry();
            }

            m_IsShow = value;
        }
    }

    protected override void Awake() {

        m_DataDiagram = GetComponentInParent<DD_DataDiagram>();
        if (null == m_DataDiagram) {
            Debug.Log(this + "null == m_DataDiagram");
        }

        m_Coordinate = GetComponentInParent<DD_CoordinateAxis>();
        if(null == m_Coordinate) {
            Debug.Log(this + "null == m_Coordinate");
        }

        GameObject parent = gameObject.transform.parent.gameObject;
        if(null == parent) {
            Debug.Log(this + "null == parent");
        }

        RectTransform parentrt = parent.GetComponent<RectTransform>();
        RectTransform localrt = gameObject.GetComponent<RectTransform>();
        if ((null == localrt) || (null == parentrt)) {
            Debug.Log(this + "null == localrt || parentrt");
        }

        //设置锚点为左下角
        localrt.anchorMin = Vector2.zero;
        localrt.anchorMax = new Vector2(1, 1);
        //设置轴心为左下角
        localrt.pivot = Vector2.zero;
        //设置轴心的坐标为坐标系区域的左下角
        localrt.anchoredPosition = Vector2.zero;
        //设置平铺的margin为0
        localrt.sizeDelta = Vector2.zero;

        if(null != m_Coordinate) {
            m_Coordinate.CoordinateRectChangeEvent += OnCoordinateRectChange;
            m_Coordinate.CoordinateScaleChangeEvent += OnCoordinateScaleChange;
            m_Coordinate.CoordinateeZeroPointChangeEvent += OnCoordinateZeroPointChange;
        }
        //m_ViewRect.Set(0, 0, m_Rect.width, m_Rect.height);
    }

    private void Update() {

        if (m_CurIsShow == m_IsShow)
            return;

        m_CurIsShow = m_IsShow;

        ///触发OnPopulateMesh的更新
        UpdateGeometry();
    }

    private float ScaleX(float x) {

        if (null == m_Coordinate) {
            Debug.Log(this + "null == m_Coordinate");
            return x;
        }

        return (x / m_Coordinate.coordinateAxisViewRangeInPixel.width);
    }

    private float ScaleY(float y) {

        if (null == m_Coordinate) {
            Debug.Log(this + "null == m_Coordinate");
            return y;
        }

        return (y / m_Coordinate.coordinateAxisViewRangeInPixel.height);
    }

    private int GetStartPointSN(List<Vector2> points, float startX) {

        int ret = 0;
        float x = 0;
        foreach (Vector2 p in points) {
            if(x > startX) {
                return points.IndexOf(p);
            }
            x += p.x;//ScaleX(p.x);
            ret++;
        }

        return ret;
    }

    private void OnCoordinateRectChange(object sender, DD_CoordinateRectChangeEventArgs e) {

        UpdateGeometry();
    }

    private void OnCoordinateScaleChange(object sender, DD_CoordinateScaleChangeEventArgs e) {

        UpdateGeometry();
    }

    private void OnCoordinateZeroPointChange(object sender, DD_CoordinateZeroPointChangeEventArgs e) {

        CurStartPointSN = GetStartPointSN(PointList, m_Coordinate.coordinateAxisViewRangeInPixel.x);
        UpdateGeometry();
    }

    protected override void OnPopulateMesh(VertexHelper vh) {

        vh.Clear();

        if (false == m_IsShow) {
            return;
        }

        float x = 0;
        List<Vector2> points = new List<Vector2>();
        for (int i = CurStartPointSN; i < PointList.Count; i++) {
            points.Add(new Vector2(ScaleX(x), ScaleY(PointList[i].y - m_Coordinate.coordinateAxisViewRangeInPixel.y)));
            x += PointList[i].x;
            if (x >= m_Coordinate.coordinateAxisViewRangeInPixel.width * rectTransform.rect.width)
                break;
        }

        DrawHorizontalLine(vh, points, color, m_Thickness, new Rect(0, 0, rectTransform.rect.width, rectTransform.rect.height));
    }

    public void AddPoint(Vector2 point) {

        PointList.Insert(0, new Vector2(point.x, point.y));

        while (PointList.Count > m_DataDiagram.m_MaxPointNum) {
            PointList.RemoveAt(PointList.Count - 1);
            print(PointList.Count);
        }

        UpdateGeometry();
    }

    public bool Clear() {

        if (null == m_Coordinate) {
            Debug.LogWarning(this + "null == m_Coordinate");
        }

        try {
            m_Coordinate.CoordinateRectChangeEvent -= OnCoordinateRectChange;
            m_Coordinate.CoordinateScaleChangeEvent -= OnCoordinateScaleChange;
            m_Coordinate.CoordinateeZeroPointChangeEvent -= OnCoordinateZeroPointChange;

            PointList.Clear();
        } catch (NullReferenceException e) {
            print(this + " : " + e);
            return false;
        }

        return true;
    }
}
