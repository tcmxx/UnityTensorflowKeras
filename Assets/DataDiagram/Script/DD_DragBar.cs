using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class DD_DragBar : MonoBehaviour, IDragHandler {

    DD_ZoomButton m_ZoomButton = null;
    GameObject m_DataDiagram = null;
    GameObject m_Parent = null;
    RectTransform m_DataDiagramRT = null;

    public bool canDrag {
        get { return gameObject.activeSelf; }
        set {
            LayoutElement le = GetComponent<LayoutElement>();
            if(null == le) {
                Debug.LogWarning(this + " : can not find LayoutElement");
                return;
            } else {
                if (true == value) {
                    gameObject.SetActive(true);
                    le.ignoreLayout = false;
                } else {
                    gameObject.SetActive(false);
                    le.ignoreLayout = true;
                }
            }
        }
    }

    // Use this for initialization
    void Start() {

        GetZoomButton();

        DD_DataDiagram dd = GetComponentInParent<DD_DataDiagram>();
        if(null == dd) {
            Debug.LogWarning(this + " : can not find any gameobject with a DataDiagram object");
            return;
        } else {
            m_DataDiagram = dd.gameObject;
        }

        m_DataDiagramRT = m_DataDiagram.GetComponent<RectTransform>();

        if (null == m_DataDiagram.transform.parent) {
            m_Parent = null;
        } else {
            m_Parent = m_DataDiagram.transform.parent.gameObject;
        }
        if(null == m_Parent) {
            Debug.LogWarning(this + " : can not DataDiagram's parent");
            return;
        }

        //默认情况如果DataDiagram插件不在UI的最顶层，则不允许拖拽
        if (null == m_Parent.GetComponent<Canvas>()) {
            canDrag = false;
        } else {
            canDrag = true;
        }
    }

    private void GetZoomButton() {

        if (null == m_ZoomButton) {
            GameObject g = GameObject.Find("ZoomButton");
            if (null == g) {
                Debug.LogWarning(this + " : can not find gameobject ZoomButton");
                return;
            } else {
                if (null == g.GetComponentInParent<DD_DataDiagram>()) {
                    Debug.LogWarning(this + " : the gameobject ZoomButton is not under the DataDiagram");
                    return;
                } else {
                    m_ZoomButton = g.GetComponent<DD_ZoomButton>();
                    if (null == m_ZoomButton) {
                        Debug.LogWarning(this + " : can not find object DD_ZoomButton");
                        return;
                    } else {
                        m_ZoomButton.ZoomButtonClickEvent += OnCtrlButtonClick;
                    }
                }
            }
        } else {
            m_ZoomButton.ZoomButtonClickEvent += OnCtrlButtonClick;
        }
    }

    public void OnDrag(PointerEventData eventData) {

        if (null == m_DataDiagramRT)
            return;

        m_DataDiagramRT.anchoredPosition += eventData.delta;
    }

    void OnCtrlButtonClick(object sender, ZoomButtonClickEventArgs e) {

        if (null == m_DataDiagram.transform.parent) {
            Debug.LogWarning(this + " OnCtrlButtonClick : can not DataDiagram's parent");
            return;
        }

        if (m_Parent != m_DataDiagram.transform.parent.gameObject) {
            m_Parent = m_DataDiagram.transform.parent.gameObject;
            if (null != m_Parent.GetComponent<Canvas>()) {
                canDrag = true;
            } else {
                canDrag = false;
            }
        }
    }
}
