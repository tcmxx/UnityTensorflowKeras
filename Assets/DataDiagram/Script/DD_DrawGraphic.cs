using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DD_DrawGraphic : MaskableGraphic {

    private int IsPointHorizontalInRect(Vector2 p, Rect rect) {

        if (p.x < rect.x)
            return -1;

        if (p.x > (rect.x + rect.width))
            return 1;

        return 0;
    }

    private int IsPointVerticalityInRect(Vector2 p, Rect rect) {

        if (p.y < rect.y)
            return -1;

        if (p.y > (rect.y + rect.height))
            return 1;

        return 0;
    }

    private Vector2? CalcHorizontalCutPoint(Vector2 p1, Vector2 p2, float y) {

        ///避免零除以零
        if(p2.y == p1.y)
            return new Vector2?(new Vector2(p2.x, p2.y));

        float x = ((y - p1.y) / (p2.y - p1.y)) * (p2.x - p1.x);

        return new Vector2 ? (new Vector2(p1.x + x, y));
    }

    private int AddHorizontalCutPoints(List<Vector2> points, int sn, float y) {

        Vector2? left = null;
        Vector2? right = null;

        int ret = 0;

        if (sn > 0) {
            if(null != (left = CalcHorizontalCutPoint(points[sn], points[sn-1], y))) {
                points.Insert(sn, left.Value);
                sn++;
                ret++;
            }
        }

        if (sn < (points.Count - 1)) {
            if(null != (right = CalcHorizontalCutPoint(points[sn], points[sn + 1], y))) {
                points.Insert(sn + 1, right.Value);
                ret++;
            }
        }

        return ret;
    }

    private void HorizontalCut(List<Vector2> points, Rect range) {

        int flag = 0;

        for (int i = 0, j = 0; i < points.Count; i += j, j = 0) {

            flag = IsPointVerticalityInRect(points[i], range);

            if (flag > 0) {
                j = AddHorizontalCutPoints(points, i, range.y + range.height);
            } else if(flag < 0) {
                j = AddHorizontalCutPoints(points, i, range.y);
            }

            j += 1;
        }
    }

    protected bool IsPointInRect(Vector2 p, Rect rect) {

        if (0 != IsPointHorizontalInRect(p, rect))
            return false;

        if (0 != IsPointVerticalityInRect(p, rect))
            return false;

        return true;
    }

    protected void DrawRectang(VertexHelper vh, Vector2 point1st,
    Vector2 point2nd, Vector2 point3rd, Vector2 point4th, Color color) {

        UIVertex[] verts = new UIVertex[4];

        verts[0].position = point1st;
        verts[0].color = color;
        verts[0].uv0 = Vector2.zero;

        verts[1].position = point2nd;
        verts[1].color = color;
        verts[1].uv0 = Vector2.zero;

        verts[2].position = point3rd;
        verts[2].color = color;
        verts[2].uv0 = Vector2.zero;

        verts[3].position = point4th;
        verts[3].color = color;
        verts[3].uv0 = Vector2.zero;

        vh.AddUIVertexQuad(verts);
    }

    protected void DrawPoint(VertexHelper vh, Vector2 point, 
        Color color, float thickness, float scaleX = 1, float scaleY = 1) {

        Vector2 point1st = new Vector2((point.x - (thickness / 2)) * scaleX, (point.y - (thickness / 2)) * scaleY);
        Vector2 point2nd = new Vector2((point.x - (thickness / 2)) * scaleX, (point.y + (thickness / 2)) * scaleY);
        Vector2 point3rd = new Vector2((point.x + (thickness / 2)) * scaleX, (point.y + (thickness / 2)) * scaleY);
        Vector2 point4th = new Vector2((point.x + (thickness / 2)) * scaleX, (point.y - (thickness / 2)) * scaleY);

        DrawRectang(vh, point1st, point2nd, point3rd, point4th, color);
    }

    protected void DrawHorizontalSegmet(VertexHelper vh, Vector2 startPoint, 
        Vector2 endPoint, Color color, float thickness, float scaleX = 1, float scaleY = 1) {

        Vector2 point1st = new Vector2(startPoint.x * scaleX, (startPoint.y * scaleY) - (thickness / 2));
        Vector2 point2nd = new Vector2(startPoint.x * scaleX, (startPoint.y * scaleY) + (thickness / 2));
        Vector2 point3rd = new Vector2(endPoint.x * scaleX, (endPoint.y * scaleY) + (thickness / 2));
        Vector2 point4th = new Vector2(endPoint.x * scaleX, (endPoint.y * scaleY) - (thickness / 2));

        DrawRectang(vh, point1st, point2nd, point3rd, point4th, color);
    }

    protected void DrawVerticalitySegmet(VertexHelper vh, Vector2 startPoint, 
        Vector2 endPoint, Color color, float thickness, float scaleX = 1, float scaleY = 1) {

        Vector2 point1st = new Vector2((startPoint.x * scaleX) - (thickness / 2), startPoint.y * scaleY);
        Vector2 point2nd = new Vector2((endPoint.x * scaleX) - (thickness / 2), endPoint.y * scaleY);
        Vector2 point3rd = new Vector2((endPoint.x * scaleX) + (thickness / 2), endPoint.y * scaleY);
        Vector2 point4th = new Vector2((startPoint.x * scaleX) + (thickness / 2), startPoint.y * scaleY);

        DrawRectang(vh, point1st, point2nd, point3rd, point4th, color);
    }

    protected void DrawHorizontalLine(VertexHelper vh, List<Vector2> points, Color color, float thickness) {
                
        if (points.Count < 2)
            return;

        for (int i = 0; i < points.Count - 1; i++) {
            DrawHorizontalSegmet(vh, points[i], points[i + 1], color, thickness);
        }
    }

    protected void DrawHorizontalLine(VertexHelper vh, List<Vector2> points, Color color, float thickness, Rect range) {

        if (points.Count < 2)
            return;

        HorizontalCut(points, range);

        for (int i = 0; i < points.Count - 1; ) {

            if(false == IsPointInRect(points[i], range)) {
                points.RemoveAt(i);
                continue;
            }

            if (false == IsPointInRect(points[i + 1], range)) {
                points.RemoveAt(i + 1);
                i++;
                continue;
            }

            DrawHorizontalSegmet(vh, points[i], points[i + 1], color, thickness);
            i++;
        }
    }

    protected void DrawTriangle(VertexHelper vh, Vector2 points, 
        Color color, float thickness, float rotate, float scaleX = 1, float scaleY = 1) {
        ///暂时只画正三角形
        float edge = (thickness / 3) * 2;
    
        Vector2 point1st = new Vector2((points.x + Mathf.Sin(Mathf.Deg2Rad * rotate) * edge) * scaleX,
            (points.y + Mathf.Cos(Mathf.Deg2Rad * rotate) * edge) * scaleY);
        Vector2 point2nd = new Vector2((points.x + Mathf.Sin(Mathf.Deg2Rad * (rotate + 120)) * edge) * scaleX,
            (points.y + Mathf.Cos(Mathf.Deg2Rad * (rotate + 120)) * edge) * scaleY);
        Vector2 point3rd = new Vector2((points.x + Mathf.Sin(Mathf.Deg2Rad * (rotate + 240)) * edge) * scaleX,
            (points.y + Mathf.Cos(Mathf.Deg2Rad * (rotate + 240)) * edge) * scaleY);

        DrawRectang(vh, point3rd, point1st, point2nd, point3rd, color);
    }

    protected void DrawRectFrame(VertexHelper vh, Vector2 point1st, Vector2 point2nd, 
        Vector2 point3rd, Vector2 point4th, Color color, float thickness) {

        DrawVerticalitySegmet(vh, point1st, point2nd, color, thickness);
        DrawHorizontalSegmet(vh, point2nd, point3rd, color, thickness);
        DrawVerticalitySegmet(vh, point3rd, point4th, color, thickness);
        DrawHorizontalSegmet(vh, point4th, point1st, color, thickness);
    }

    ///暂时只画正三角形
    public static float GetTriangleCentreDis(float thickness) {

        return (thickness / 3);
    }
}
