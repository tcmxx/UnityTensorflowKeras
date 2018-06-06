using System;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UI;

public class DD_CoordinateRectChangeEventArgs : EventArgs {

    public Rect viewRectInPixel;

    public DD_CoordinateRectChangeEventArgs(Rect newRect) : base() {

        viewRectInPixel = newRect;
    }
}

public class DD_CoordinateScaleChangeEventArgs : EventArgs {

    public float scaleX;
    public float scaleY;

    public DD_CoordinateScaleChangeEventArgs(float scaleX, float scaleY) : base() {

        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }
}

/// <summary>
/// 改变当前观察区域的坐标零点事件
/// </summary>
public class DD_CoordinateZeroPointChangeEventArgs : EventArgs {

    public Vector2 zeroPoint;

    public DD_CoordinateZeroPointChangeEventArgs(Vector2 zeroPoint) : base() {

        this.zeroPoint = zeroPoint;
    }
}

public class DD_CoordinateAxis : DD_DrawGraphic {

#region const value
    private static readonly string MARK_TEXT_BASE_NAME = "MarkText";
    private static readonly string LINES_BASE_NAME = "Line";
    private static readonly string COORDINATE_RECT = "CoordinateRect";
    private const float INCH_PER_CENTIMETER = 0.3937008f;
    private readonly float[] MarkIntervalTab = { 1, 2, 5 };//c#中数组不支持const
#endregion

#region property
    /// <summary>
    /// 数据表格入口类
    /// </summary>
    //[SerializeField]
    private DD_DataDiagram m_DataDiagram = null;

    /// <summary>
    /// 实际折线绘制区域,以像素为单位
    /// </summary>
    private RectTransform m_CoordinateRectT = null;

    /// <summary>
    /// 折线的预设，提前load，提升性能
    /// </summary>
    private GameObject m_LinesPreb = null;

    /// <summary>
    /// 坐标轴字体的预设，提前load，提升性能
    /// </summary>
    private GameObject m_MarkTextPreb = null;

    /// <summary>
    /// 所有存在于该坐标系中的折线列表
    /// </summary>
    private List<GameObject> m_LineList = new List<GameObject>();

    /// <summary>
    /// 坐标轴显示区域范围，以像素为单位
    /// </summary>
    //private Rect m_CoordinatePixelRect = new Rect();

    /// <summary>
    /// 坐标轴缩放的速度
    /// </summary>
    private Vector2 m_ZoomSpeed = new Vector2(1, 1);

    /// <summary>
    /// 坐标轴移动的速度
    /// </summary>
    private Vector2 m_MoveSpeed = new Vector2(1, 1);

    /// <summary>
    /// 坐标轴最大可伸缩范围，以坐标轴为单位
    /// Y轴的通过长宽比例计算获得
    /// </summary>
    private float m_CoordinateAxisMaxWidth = 10000;
    private float m_CoordinateAxisMinWidth = 0.01f;

    /// <summary>
    /// 矩形框式坐标轴线条粗细
    /// </summary>
    private float m_RectThickness = 2;

    /// <summary>
    /// 矩形框式坐标轴背景颜色
    /// </summary>
    private Color m_BackgroundColor = new Color(0, 0, 0, 0.5f);

    /// <summary>
    /// 矩形框式坐标标记线颜色
    /// </summary>
    private Color m_MarkColor = new Color(0.8f, 0.8f, 0.8f, 1);

    ///// <summary>
    ///// 矩形框式坐标轴刻度的间距，以公分（CM）为单位
    ///// </summary>
    //[SerializeField]
    //private float m_CentimeterPerMark = 1f;

    ///// <summary>
    ///// 标准x轴上每单位长度对应的物理长度（未缩放状态下）
    ///// x轴长度单位为“秒”，物理长度单位为“公分”
    ///// </summary>
    //[SerializeField]
    //private float m_CentimeterPerCoordUnitX = 0.2f;

    ///// <summary>
    ///// 标准y轴上每单位长度对应的物理长度（未缩放状态下）
    ///// y轴长度单位为“米”，物理长度单位为“公分”
    ///// </summary>
    //[SerializeField]
    //private float m_CentimeterPerCoordUnitY = 0.1f;

    /// <summary>
    /// 存放所有的刻度值文字对象的列表
    /// 每次缩放时只对其进行调整，不再进行创建和销毁
    /// </summary>
    private List<GameObject> m_MarkHorizontalTexts = new List<GameObject>();

    /// <summary>
    /// 矩形框式坐标轴左侧坐标值字符的宽度
    /// </summary>
    //private float m_MinMarkTextWidth = 30;

    /// <summary>
    /// 坐标轴字体的高度，以像素为单位
    /// 坐标轴字体的宽度等于坐标系与控件左边留白区域宽度
    /// </summary>
    private float m_MinMarkTextHeight = 20;

    /// <summary>
    /// 矩形框式坐标轴刻度的间距，以屏幕像素单位
    /// </summary>
    private float m_PixelPerMark {
        get { return INCH_PER_CENTIMETER * m_DataDiagram.m_CentimeterPerMark * Screen.dpi; }
    }

    /// <summary>
    /// point表示坐标轴刻度值的零点位置
    /// size表示坐标轴初始设置时的刻度值，所有输入点都以该值为基准转化为像素值
    /// </summary>
    private Rect m_CoordinateAxisRange {
        get {
            try {
                Vector2 sizePixel = m_CoordinateRectT.rect.size;
                return new Rect(0, 0,
                    sizePixel.x / (m_DataDiagram.m_CentimeterPerCoordUnitX * INCH_PER_CENTIMETER * Screen.dpi),
                    sizePixel.y / (m_DataDiagram.m_CentimeterPerCoordUnitY * INCH_PER_CENTIMETER * Screen.dpi));
            } catch(NullReferenceException e) {
                Debug.Log(this + " : " + e);
            }
            return new Rect(Vector2.zero, GetComponent<RectTransform>().rect.size);
        }
    }

    /// <summary>
    /// point表示当前观察区域坐标轴刻度值的零点位置，用于实现坐标轴移动
    /// size表示相对于m_CoordinateAxisRange.size的缩放系数
    /// </summary>
    private Rect m_CoordinateAxisViewRange = new Rect(1, 1, 1, 1);

    private float m_CoordinateAxisViewSizeX {
        get {
            try {
                return m_CoordinateAxisRange.width * m_CoordinateAxisViewRange.width;
            } catch(NullReferenceException e) {
                Debug.Log(this + " : " + e);
            }
            return m_CoordinateAxisRange.width;
        }
    }

    private float m_CoordinateAxisViewSizeY {
        get {
            try {
                return m_CoordinateAxisRange.height * m_CoordinateAxisViewRange.height;
            } catch (NullReferenceException e) {
                Debug.Log(this + " : " + e);
            }
            return m_CoordinateAxisRange.width;
        }
    }

    /// <summary>
    /// point表示当前观察区域坐标轴刻度值的零点位置，用于实现坐标轴移动,以像素为单位
    /// size表示相对于m_CoordinateAxisRange.size的缩放系数
    /// </summary>
    public Rect coordinateAxisViewRangeInPixel {
        get {
            try {
                return new Rect(
                    CoordinateToPixel(m_CoordinateAxisViewRange.position - m_CoordinateAxisRange.position),
                    m_CoordinateAxisViewRange.size);
            } catch (NullReferenceException e) {
                Debug.Log(this + " : " + e);
            }

            return new Rect(CoordinateToPixel(m_CoordinateAxisRange.position),
                m_CoordinateAxisViewRange.size);
        }
    }

    public RectTransform coordinateRectT {
        //get { return m_CoordinatePixelRect; }
        get {
            try {
                return m_CoordinateRectT;
            } catch {
                return GetComponent<RectTransform>();
            }
        }
    }

    public int lineNum {
        get { return m_LineList.Count; }
    }

#endregion

#region delegate
    // 创建一个委托，返回类型为void，两个参数
    public delegate void CoordinateRectChangeHandler(object sender, DD_CoordinateRectChangeEventArgs e);
    public delegate void CoordinateScaleChangeHandler(object sender, DD_CoordinateScaleChangeEventArgs e);
    public delegate void CoordinateZeroPointChangeHandler(object sender, DD_CoordinateZeroPointChangeEventArgs e);
    // 将创建的委托和特定事件关联,在这里特定的事件为KeyDown
    public event CoordinateRectChangeHandler CoordinateRectChangeEvent;
    public event CoordinateScaleChangeHandler CoordinateScaleChangeEvent;
    public event CoordinateZeroPointChangeHandler CoordinateeZeroPointChangeEvent;
#endregion

    protected override void Awake() {
        
        if (null == (m_DataDiagram = GetComponentInParent<DD_DataDiagram>())) {
            Debug.Log(this + "Awake Error : null == m_DataDiagram");
            return;
        }

        m_LinesPreb = (GameObject)Resources.Load("Prefabs/Lines");
        if (null == m_LinesPreb) {
            Debug.Log("Error : null == m_LinesPreb");
        }

        m_MarkTextPreb = (GameObject)Resources.Load("Prefabs/MarkText");
        if(null == m_MarkTextPreb) {
            Debug.Log("Error : null == m_MarkTextPreb");
        }

        try {
            m_CoordinateRectT = FindInChild(COORDINATE_RECT).GetComponent<RectTransform>();
            if (null == m_CoordinateRectT) {
                Debug.Log("Error : null == m_CoordinateRectT");
                return;
            }
        } catch(NullReferenceException e) {
            Debug.Log(this + "," + e);
        }

        ///检查当前是否已经存在刻度值文本UI控件
        FindExistMarkText(m_MarkHorizontalTexts);

        GameObject parent = gameObject.transform.parent.gameObject;
        Rect parentRect = parent.GetComponent<RectTransform>().rect;

        ///计算坐标轴观察区域的大小，以刻度为单位，初始默认与初始坐标区域范围相同
        //m_CoordinateAxisViewRange = new Rect(m_CoordinateAxisRange);
        m_CoordinateAxisViewRange.position = m_CoordinateAxisRange.position;
        m_CoordinateAxisViewRange.size = new Vector2(1, 1);

        ///添加事件响应
        m_DataDiagram.RectChangeEvent += OnRectChange;
        m_DataDiagram.ZoomEvent += OnZoom;
        m_DataDiagram.MoveEvent += OnMove;
    }

    // Update is called once per frame
    void Update() {
        
    }

    private GameObject FindInChild(string name) {

        foreach (Transform t in transform) {
            if (name == t.gameObject.name) {
                return t.gameObject;
            }
        }

        return null;
    }

    private void ChangeRect(Rect newRect) {

        if (null != CoordinateRectChangeEvent)
            CoordinateRectChangeEvent(this,
                new DD_CoordinateRectChangeEventArgs(new Rect(
                CoordinateToPixel(m_CoordinateAxisRange.position - m_CoordinateAxisViewRange.position),
                newRect.size)));
    }

    private void ChangeScale(float ZoomX, float ZoomY) {

        Vector2 rangeSize = m_CoordinateAxisRange.size;
        Vector2 viewSize = new Vector2(m_CoordinateAxisViewRange.width * rangeSize.x,
            m_CoordinateAxisViewRange.height * rangeSize.y);

        float YtoXScale = (rangeSize.y / rangeSize.x);
        float zoomXVal = ZoomX * m_ZoomSpeed.x;
        float zoomYVal = (ZoomY * m_ZoomSpeed.y) * YtoXScale;

        viewSize.x += zoomXVal;
        viewSize.y += zoomYVal;

        if (viewSize.x > m_CoordinateAxisMaxWidth)
            viewSize.x = m_CoordinateAxisMaxWidth;

        if (viewSize.x < m_CoordinateAxisMinWidth)
            viewSize.x = m_CoordinateAxisMinWidth;

        if (viewSize.y > m_CoordinateAxisMaxWidth * YtoXScale)
            viewSize.y = m_CoordinateAxisMaxWidth * YtoXScale;

        if (viewSize.y < m_CoordinateAxisMinWidth * YtoXScale)
            viewSize.y = m_CoordinateAxisMinWidth * YtoXScale;

        m_CoordinateAxisViewRange.width = viewSize.x / rangeSize.x;
        m_CoordinateAxisViewRange.height = viewSize.y / rangeSize.y;
    }

    private void OnRectChange(object sender, DD_RectChangeEventArgs e) {

        ChangeRect(m_CoordinateRectT.rect);

        ///触发OnPopulateMesh的更新
        UpdateGeometry();
    }

    private void OnZoom(object sender, DD_ZoomEventArgs e) {

        if (null != CoordinateScaleChangeEvent)
            CoordinateScaleChangeEvent(this, new DD_CoordinateScaleChangeEventArgs(
                    m_CoordinateAxisViewRange.width, m_CoordinateAxisViewRange.height));

        ChangeScale(e.ZoomX, e.ZoomY);

        ///触发OnPopulateMesh的更新
        UpdateGeometry();
    }

    private void OnMove(object sender, DD_MoveEventArgs e) {

        if ((1 > Mathf.Abs(e.MoveX)) && (1 > Mathf.Abs(e.MoveY)))
            return;

        Vector2 coordDis = new Vector2(
            (e.MoveX / m_CoordinateRectT.rect.width) * m_CoordinateAxisViewSizeX,
            (e.MoveY / m_CoordinateRectT.rect.height) * m_CoordinateAxisViewSizeY);

        Vector2 dis = new Vector2(-coordDis.x * m_MoveSpeed.x, -coordDis.y * m_MoveSpeed.y);

        m_CoordinateAxisViewRange.position += dis;
        if (0 > m_CoordinateAxisViewRange.x)
            m_CoordinateAxisViewRange.x = 0;

        if (null != CoordinateeZeroPointChangeEvent)
            CoordinateeZeroPointChangeEvent(this, 
                new DD_CoordinateZeroPointChangeEventArgs(CoordinateToPixel(dis)));
        
        ///触发OnPopulateMesh的更新
        UpdateGeometry();
    }

    //private void OnMoveEnd(object sender, EventArgs e) {
    //    ///暂时没用
    //}

    private Vector2 CoordinateToPixel(Vector2 coordPoint) {

        return new Vector2((coordPoint.x / m_CoordinateAxisRange.width) * m_CoordinateRectT.rect.width,
            (coordPoint.y / m_CoordinateAxisRange.height) * m_CoordinateRectT.rect.height);
    }

#region draw rect coordinateAxis
    private int CalcMarkNum(float pixelPerMark, float totalPixel) {

        return Mathf.CeilToInt(totalPixel / (pixelPerMark > 0 ? pixelPerMark : 1));
    }

    private float CalcMarkLevel(float[] makeTab, int markNum, float viewMarkRange) {

        float dis = viewMarkRange / (markNum > 0 ? markNum : 1);
        float markScale = 1;
        float mark = makeTab[0];

        while ((dis < (mark * markScale)) || (dis >= (mark * markScale * 10))) {

            if(dis < (mark * markScale)) {
                markScale /= 10;
            } else if(dis >= (mark * markScale * 10)) {
                markScale *= 10;
            } else {
                break;
            }
        }

        dis /= markScale;
        for (int i = 1; i < makeTab.Length; i++) {
            if (Mathf.Abs(mark - dis) > Mathf.Abs(makeTab[i] - dis))
                mark = makeTab[i];
        }

        return (mark * markScale);
    }

    private float CeilingFormat(float markLevel, float Val) {

        /// + (markLevel / 100)防止除法时精度丢失，但可能引入不精确性
        //return Mathf.CeilToInt((Val + (markLevel / 100)) / markLevel) * markLevel;
        return Mathf.CeilToInt(Val / markLevel) * markLevel;
    }

    private float[] CalcMarkVals(float markLevel, float startViewMarkVal, float endViewMarkVal) {

        float[] markVals;
        List<float> tempList = new List<float>();
        float tempMarkVal = CeilingFormat(markLevel, startViewMarkVal);

        while(tempMarkVal < endViewMarkVal) {
            tempList.Add(tempMarkVal);
            tempMarkVal += markLevel;
        }

        markVals = new float[tempList.Count];
        tempList.CopyTo(markVals);

        return markVals;
    }

    private float MarkValToPixel(float markVal, float startViewMarkVal, 
        float endViewMarkVal, float stratCoordPixelVal, float endCoordPixelVal) {

        ///判断小于等于是为了避免差值为零，造成零除以零的问题
        if ((endViewMarkVal <= startViewMarkVal) || (markVal <= startViewMarkVal))
            return stratCoordPixelVal;

        return stratCoordPixelVal + 
            ((endCoordPixelVal - stratCoordPixelVal) * ((markVal - startViewMarkVal) / (endViewMarkVal - startViewMarkVal)));
    }

    private float[] MarkValsToPixel(float[] markVals, float startViewMarkVal,
        float endViewMarkVal, float stratCoordPixelVal, float endCoordPixelVal) {

        float[] pixelYs = new float[markVals.Length];

        for (int i = 0; i < pixelYs.Length; i++) {
            pixelYs[i] = MarkValToPixel(markVals[i], startViewMarkVal, 
                endViewMarkVal, stratCoordPixelVal, endCoordPixelVal);
        }

        return pixelYs;
    }

    private void SetMarkText(GameObject markText, Rect rect, string str, bool isEnable) {

        if (null == markText) {
            Debug.Log("SetMarkText Error : null == markText");
            return;
        }

        RectTransform rectTransform = markText.GetComponent<RectTransform>();
        if (null == rectTransform) {
            Debug.Log("SetMarkText Error : null == rectTransform");
            return;
        }

        Text text = markText.GetComponent<Text>();
        if (null == text) {
            Debug.Log("SetMarkText Error : null == Text");
            return;
        }

        //设置锚点为左下角
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
        //设置轴心为左下角
        rectTransform.pivot = new Vector2(0, 0);
        //设置轴心相对锚点的位置
        rectTransform.anchoredPosition = rect.position;
        //设置控件大小
        rectTransform.sizeDelta = rect.size;

        text.text = str;
        text.enabled = isEnable;
    }

    private void ResetMarkText(GameObject markText) {

        SetMarkText(markText, new Rect(new Vector2(0, m_CoordinateRectT.rect.y), 
            new Vector2(m_CoordinateRectT.rect.x, m_MinMarkTextHeight)), null, false);
    }

    private void ResetAllMarkText(List<GameObject> markTexts) {

        if (null == markTexts) {
            Debug.Log("DisableAllMarkText Error : null == markTexts");
            return;
        }

        foreach (GameObject g in markTexts) {
            ResetMarkText(g);
        }
    }

    private void DrawOneHorizontalMarkText(GameObject markText, 
        float markValY, float pixelY, Rect coordinateRect) {

        SetMarkText(markText, new Rect(new Vector2(0, pixelY - (m_MinMarkTextHeight / 2)),
            new Vector2(coordinateRect.x - 2, m_MinMarkTextHeight)),
            markValY.ToString(), true);
    }

    /// <summary>
    /// 实例化一个UI控件时调用了graphic rebuild操作，而OnPopulateMesh（）
    /// 函数是在graphic rebuild操作中被调用的，所以若在OnPopulateMesh（）
    /// 中创建一个新的UI控件时系统会提示错误：graphic rebuild操作被循环调用了
    /// 所以这里需要使用协程操作（IEnumerator）
    /// 在进入协程操作后，必须立即执行yield return new WaitForSeconds(0);
    /// 使当前协程暂时退出，让graphic rebuild操作先执行
    /// </summary>
    /// <param name="marksVals"></param>
    /// <param name="marksPixel"></param>
    /// <param name="coordinateRect"></param>
    /// <returns></returns>
    private IEnumerator DrawHorizontalTextMark(float[] marksVals, float[] marksPixel, Rect coordinateRect) {

        yield return new WaitForSeconds(0);

        while (marksPixel.Length > m_MarkHorizontalTexts.Count) {
            GameObject markText = Instantiate(m_MarkTextPreb, transform);
            markText.name = string.Format("{0}{1}", MARK_TEXT_BASE_NAME, m_MarkHorizontalTexts.Count);
            m_MarkHorizontalTexts.Add(markText);
        }

        ResetAllMarkText(m_MarkHorizontalTexts);

        for (int i = 0; i < marksPixel.Length; i++) {
            DrawOneHorizontalMarkText(m_MarkHorizontalTexts[i], marksVals[i], marksPixel[i], coordinateRect);
        }

        yield return 0;
    }

    private void DrawOneHorizontalMark(VertexHelper vh, float pixelY, Rect coordinateRect) {

        Vector2 startPoint = new Vector2(coordinateRect.x, pixelY);
        Vector2 endPoint = new Vector2(coordinateRect.x + coordinateRect.width, pixelY);

        DrawHorizontalSegmet(vh, startPoint, endPoint, m_MarkColor, m_RectThickness / 2);
    }

    private void DrawHorizontalMark(VertexHelper vh, Rect coordinateRect) {

        int markNum = CalcMarkNum(m_PixelPerMark, coordinateRect.height);

        float markLevel = CalcMarkLevel(MarkIntervalTab, markNum, m_CoordinateAxisViewSizeY);

        float[] marksVals = CalcMarkVals(markLevel, m_CoordinateAxisViewRange.y,
            m_CoordinateAxisViewRange.y + m_CoordinateAxisViewSizeY);

        float[] marksPixel = MarkValsToPixel(marksVals, m_CoordinateAxisViewRange.y,
            m_CoordinateAxisViewRange.y + m_CoordinateAxisViewSizeY,
            coordinateRect.y, coordinateRect.y + coordinateRect.height);

        for (int i = 0; i< marksPixel.Length; i++) {
            DrawOneHorizontalMark(vh, marksPixel[i], coordinateRect);
        }

        StartCoroutine(DrawHorizontalTextMark(marksVals, marksPixel, coordinateRect));
    }

    private void DrawRect(VertexHelper vh, Rect rect) {

        DrawRectang(vh, rect.position, new Vector2(rect.x, rect.y + rect.height),
            new Vector2(rect.x + rect.width, rect.y + rect.height), 
            new Vector2(rect.x + rect.width, rect.y), m_BackgroundColor);

    }

    private void DrawRectCoordinate(VertexHelper vh) {

        Rect marksRect = new Rect(m_CoordinateRectT.offsetMin, m_CoordinateRectT.rect.size);

        DrawRect(vh, new Rect(marksRect));

        DrawHorizontalMark(vh, marksRect);
    }

    /// <summary>
    /// 每次运行前先查询当前坐标下是否已经实例化了刻度值文本UI控件
    /// 如果已经存在，则先加入队列以待使用
    /// transform是一个迭代类型，可以迭代出其所有Child节点
    /// </summary>
    /// <param name="markTexts"></param>
    private void FindExistMarkText(List<GameObject> markTexts) {

        //Transform tempTrans = null;
        foreach (Transform t in transform) {
            if (Regex.IsMatch(t.gameObject.name, MARK_TEXT_BASE_NAME)) {
                t.gameObject.name = string.Format("{0}{1}", MARK_TEXT_BASE_NAME, m_MarkHorizontalTexts.Count);
                markTexts.Add(t.gameObject);
            }
            
        }
    }
#endregion

    protected override void OnPopulateMesh(VertexHelper vh) {

        vh.Clear();
        
        //DrawAxis(vh);
        DrawRectCoordinate(vh);
    }

    public void InputPoint(GameObject line, Vector2 point) {

        line.GetComponent<DD_Lines>().AddPoint(CoordinateToPixel(point));
    }

    public GameObject AddLine(string name) {
        
        if(null == m_LinesPreb)
            m_LinesPreb = (GameObject)Resources.Load("Prefabs/Lines");

        try {
            m_LineList.Add(Instantiate(m_LinesPreb, m_CoordinateRectT));
        } catch (NullReferenceException e) {
            Debug.Log(this + "," + e);
            return null;
        }

        m_LineList[m_LineList.Count - 1].GetComponent<DD_Lines>().lineName = name;
        m_LineList[m_LineList.Count - 1].GetComponent<DD_Lines>().color = Color.yellow;
        m_LineList[m_LineList.Count - 1].name = String.Format("{0}{1}", LINES_BASE_NAME, 
            m_LineList[m_LineList.Count - 1].GetComponent<DD_Lines>().lineName);

        return m_LineList[m_LineList.Count - 1];
    }

    public bool RemoveLine(GameObject line) {

        if (null == line)
            return true;

        if (false == m_LineList.Remove(line))
            return false;

        try {
            line.GetComponent<DD_Lines>().Clear();
        } catch (NullReferenceException) {

        }

        Destroy(line);

        return true;
    }

}
