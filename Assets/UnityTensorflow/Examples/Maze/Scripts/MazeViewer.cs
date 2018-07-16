using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeViewer : MonoBehaviour {
    public float blockDimension = 1.0f;
    public GameObject blockPrefab;

    public SpriteRenderer[,] blocks;

   

    public void UpdateGraphics(MazeAgent agent)
    {
        for (int i = 0; i < agent.mazeDimension.x; ++i)
        {
            for (int j = 0; j < agent.mazeDimension.y; ++j)
            {
                blocks[i, j].color = ChooseColor((int)agent.map[i, j], agent);
            }
        }
    }


    private Color ChooseColor(int blockType, MazeAgent agent)
    {
        if (blockType == agent.WallInt)
        {
            return Color.red;
        }
        else if (blockType == agent.GoalInt)
        {
            return Color.green;
        }
        else if (blockType == agent.PlayerInt)
        {
            return Color.yellow;
        }
        else
        {
            return Color.black;
        }
    }

    public void InitializeGraphic(MazeAgent agent)
    {
        blocks = new SpriteRenderer[agent.mazeDimension.x, agent.mazeDimension.y];

        Vector3 offset = transform.position - new Vector3((agent.mazeDimension.x - 1) * blockDimension / 2, (agent.mazeDimension.y - 1) * blockDimension / 2 );

        for (int i = 0; i < agent.mazeDimension.x; ++i)
        {
            for (int j = 0; j < agent.mazeDimension.y; ++j)
            {
                GameObject obj = GameObject.Instantiate(blockPrefab, new Vector3(i * blockDimension, j * blockDimension) + offset, Quaternion.identity, this.transform);
                blocks[i, j] = obj.GetComponent<SpriteRenderer>();
            }
        }
    }
}
