using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DataProvider : MonoBehaviour
{

    // Ignore this line.
    public float t = 0;

    void Update()
    {
        // Some amazing demo calculations...
        t += Time.deltaTime;
        float cos1 = Mathf.Cos(t * 2f);
        float cos2 = Mathf.Cos(t * 2.5f);
        float tan = Mathf.Tan((t % 2) / 10f);

        // ********** Overloads **********

        // User defined color.
        Grapher.Log(cos1, "Cos1", Color.yellow);

        // Feeling lazy version.
        Grapher.Log(cos2, "Cos2", Color.red);

        // Don't like the provided time for some reason? Use your own.
        Grapher.Log(tan, "Tan", Color.green);

        // Alternative with defined color.
        Grapher.Log(cos1 + cos2, "Cos1 + Cos2", Color.cyan);

        // Different type examples

        // ********** List **********
        List<int> list = new List<int>();
        list.Add(1);
        list.Add(2);
        //Grapher.Log(list, "List", Color.white);


        // ********** List **********
        LinkedList<int> linkedList = new LinkedList<int>();
        linkedList.AddLast(1);
        linkedList.AddLast(2);
        //Grapher.Log(linkedList, "LinkedList", Color.white);


        // ********** Array **********
        //Grapher.Log(new int[3] { 1, 2, 3 }, "Array", Color.white);


        // ********** Queue **********
        Queue<int> queue = new Queue<int>();
        queue.Enqueue(1);
        queue.Enqueue(2);
        //Grapher.Log(queue, "Queue", Color.white);


        // ********** ArrayList **********
        ArrayList arrList = new ArrayList();
        arrList.Add(1);
        arrList.Add(5f);
        //Grapher.Log(arrList, "ArrayList", Color.white);

        // ********** Enum **********
        TestEnum tEnum = (int)t % 2 == 0 ? TestEnum.bird : TestEnum.alien;
        //Grapher.Log(tEnum, "Enum", Color.white);
    }

    public enum TestEnum
    {
        bird, horse, alien
    }
}
