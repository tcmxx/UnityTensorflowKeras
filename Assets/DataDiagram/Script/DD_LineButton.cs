using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.EventSystems;
using UnityEngine;
using UnityEngine.UI;

public class DD_LineButton : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler {

    private GameObject m_Line;
    public GameObject m_Label;

    public GameObject line {
        get { return m_Line; }
        set {
            DD_Lines lines = value.GetComponent<DD_Lines>();
            if (null == lines) {
                Debug.LogWarning(this.ToString() + "LineButton error : set line null == value.GetComponent<Lines>()");
                return;
            } else {
                m_Line = value;
                SetLineButton(lines);
            }
        }
    }
	// Use this for initialization
	void Start () {

        //if ((null == m_Label) || (null == m_Label.GetComponent<Text>())) {
        //    m_Label = null;
        //}

        //try {
        //    m_Label.GetComponent<Text>().text = m_Line.GetComponent<DD_Lines>().lineName;
        //    m_Label.GetComponent<Text>().color = m_Line.GetComponent<DD_Lines>().color;
        //} catch {
        //    m_Label.GetComponent<Text>().color = Color.white;
        //}
        
    }
	
	// Update is called once per frame
	void Update () {
		
	}

    private void SetLabel(DD_Lines lines) {

        if ((null == m_Label) || (null == m_Label.GetComponent<Text>())) {
            m_Label = null;
        }

        try {
            m_Label.GetComponent<Text>().text = lines.GetComponent<DD_Lines>().lineName;
            m_Label.GetComponent<Text>().color = lines.GetComponent<DD_Lines>().color;
        } catch {
            m_Label.GetComponent<Text>().color = Color.white;
        }
    }

    public void SetLineButton(DD_Lines lines) {

        name = string.Format("Button{0}", lines.gameObject.name);
        GetComponent<Image>().color = lines.color;

        SetLabel(lines);
    }

    public void OnPointerEnter(PointerEventData eventData) {

        ///子物体（叠加在该物体上的物体）的消息也会监听到
        ///这里不希望响应子物体的鼠标进入消息，所以只响应本物体的该消息
        if (eventData.pointerCurrentRaycast.gameObject != gameObject)
            return;

        if (null == m_Label)
            return;

        DD_DataDiagram dd = GetComponentInParent<DD_DataDiagram>();
        if (null == dd) {
            return;
        }

        m_Label.transform.SetParent(dd.transform);
        m_Label.transform.position = transform.position + new Vector3(0, - GetComponent<RectTransform>().rect.height / 2, 0);
        m_Label.SetActive(true);
    }

    public void OnPointerExit(PointerEventData eventData) {

        ///离开区域事件不用判断
        ///而且eventData.pointerCurrentRaycast.gameObject的值为鼠标离开后所触碰的物体
        //if (eventData.pointerCurrentRaycast.gameObject != gameObject)
        //    return;

        if (null == m_Label)
            return;

        m_Label.transform.SetParent(transform);
        m_Label.SetActive(false);
    }

    public void OnButtonClick() {

        if (true == Input.GetKey(KeyCode.LeftControl)) {
            return;
        }

        if (null == m_Line) {
            Debug.LogWarning(this.ToString() + "error OnButtonClick : null == m_Line");
            return;
        }

        DD_Lines lines = m_Line.GetComponent<DD_Lines>();
        if(null == lines) {
            Debug.LogWarning(this.ToString() + "error OnButtonClick : null == DD_Lines");
            return;
        } else {
            lines.IsShow = !lines.IsShow;
        }
    }

    public void OnButtonClickWithCtrl() {

        if(true == Input.GetKey(KeyCode.LeftControl)) {
            try {
                transform.GetComponentInParent<DD_DataDiagram>().DestroyLine(m_Line);
            } catch (NullReferenceException) {
                Debug.LogWarning("OnButtonClickWithCtrl throw a NullReferenceException");
            }
        }
    }

    public void DestroyLineButton() {

        if(null != m_Label)
            Destroy(m_Label);
    }
}
