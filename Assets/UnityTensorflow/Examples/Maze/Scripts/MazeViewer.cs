using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeViewer : MonoBehaviour {
    public MazeAgent mazeEnvironment;

    public bool enableUpdate = true;
    public float blockDimension = 1.0f;
    public GameObject blockPrefab;

    public SpriteRenderer[,] blocks;


    // Use this for initialization
    void Start()
    {
        InitializeGraphic();

    }

    // Update is called once per frame
    void Update()
    {
        if (enableUpdate)
            UpdateGraphics(mazeEnvironment.map);

    }


    public void UpdateGraphics(float[,] map)
    {
        for (int i = 0; i < mazeEnvironment.mazeDimension.x; ++i)
        {
            for (int j = 0; j < mazeEnvironment.mazeDimension.y; ++j)
            {
                blocks[i, j].color = ChooseColor((int)map[i, j]);
            }
        }
    }


    private Color ChooseColor(int blockType)
    {
        if (blockType == mazeEnvironment.WallInt)
        {
            return Color.red;
        }
        else if (blockType == mazeEnvironment.GoalInt)
        {
            return Color.green;
        }
        else if (blockType == mazeEnvironment.PlayerInt)
        {
            return Color.yellow;
        }
        else
        {
            return Color.black;
        }
    }

    private void InitializeGraphic()
    {
        blocks = new SpriteRenderer[mazeEnvironment.mazeDimension.x, mazeEnvironment.mazeDimension.y];

        Vector3 offset = transform.position - new Vector3((mazeEnvironment.mazeDimension.x - 1) * blockDimension / 2, (mazeEnvironment.mazeDimension.y - 1) * blockDimension / 2 );

        for (int i = 0; i < mazeEnvironment.mazeDimension.x; ++i)
        {
            for (int j = 0; j < mazeEnvironment.mazeDimension.y; ++j)
            {
                GameObject obj = GameObject.Instantiate(blockPrefab, new Vector3(i * blockDimension, j * blockDimension) + offset, Quaternion.identity, this.transform);
                blocks[i, j] = obj.GetComponent<SpriteRenderer>();
            }
        }
    }
}
