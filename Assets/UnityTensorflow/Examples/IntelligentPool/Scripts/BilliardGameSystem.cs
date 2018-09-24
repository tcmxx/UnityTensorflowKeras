using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AaltoGames;

public class BilliardGameSystem : MonoBehaviour {
    [HideInInspector]
    public float bestScore = 0;
    [HideInInspector]
    public List<Vector3> bestActions = null;

    public BilliardArena defaultArena;

    protected float prevBounceThreshold;
    // Use this for initialization
    void Start()
    {
        /*for (int i = 1; i < 4 ; ++i)
        {
            var newA = Instantiate(defaultArena, defaultArena.transform.position + Vector3.right * 5 * i, defaultArena.transform.rotation);
        }*/
        prevBounceThreshold = Physics.bounceThreshold;
        Physics.bounceThreshold = 0.01f;
        
    }



    private void Update()
    {
        //evaluateShots(new List<Vector3>() { Vector3.forward, Vector3.forward, new Vector3(1,0,1) }, Color.green);
    }

    public void Reset(bool randomize = true)
    {
        defaultArena.Reset(randomize);
        //Physics.autoSimulation = true;
    }
    /// <summary>
    /// return the positions of all balls. The first ball is always the white ball.
    /// The Y coordinate of a ball will only be 1 and 0. 0 means this ball is not active right now.
    /// </summary>
    /// <returns></returns>
    public List<Vector3> GetBallsStatus()
    {
        return defaultArena.GetBallsStatus();
    }

    public bool GameComplete()
    {
        return defaultArena.GameComplete();
    }

    public void Shoot(Vector3 force)
    {
        defaultArena.Shoot(force);
    }

    public void ShootSequence(List<Vector3> forces)
    {
        defaultArena.ShootSequence(forces);
    }

    public bool AllShotsComplete()
    {
        return defaultArena.AllShotsComplete();
    }

    public float EvaluateShot(Vector3 force, Color drawColor, int maxSteps=1000)
    {
        var result = EvaluateShotBatch(new List<Vector3>() { force }, drawColor, maxSteps);
        return result[0];
    }

    public List<float> EvaluateShotBatch(List<Vector3> forces, Color drawColor, int maxSteps = 1000)
    {

        var forcesSequenced = new List<List<Vector3>>();
        foreach (var f in forces)
        {
            forcesSequenced.Add(new List<Vector3>() { f });
        }

        return EvaluateShotSequenceBatch(forcesSequenced, drawColor, maxSteps);
    }

    public float EvaluateShotSequence(List<Vector3> forces, Color drawColor, int maxSteps = 1000)
    {
        var result = EvaluateShotSequenceBatch(new List<List<Vector3>>() { forces }, drawColor, maxSteps);
        return result[0];
    }



    public List<float> EvaluateShotSequenceBatch(List<List<Vector3>> forces, Color drawColor, int maxSteps = 1000)
    {
        //Disable autosimulation, the manual simulation will otherwise have no effect
        bool oldAutoSimulation = Physics.autoSimulation;
        Physics.autoSimulation = false;

        int numOfForce = forces.Count;
        if (numOfForce <= 0)
            return new List<float>();

        List<BilliardArena> allArenas = new List<BilliardArena>();
        allArenas.Add(defaultArena);
        for (int i = 1; i < numOfForce; ++i)
        {
            var newA = Instantiate(defaultArena, defaultArena.transform.position + Vector3.right * 5 * i, defaultArena.transform.rotation);
            newA.GetComponent<BilliardArena>().InitializeArena();
            newA.name = " Arena" + forces[i] + "At:" + (defaultArena.transform.position + Vector3.right * 5 * i);
            allArenas.Add(newA);
        }

        //Physics.Simulate(Time.fixedDeltaTime);

        //save state, we don't want this preview simulation to have any effect after it's done
        /*foreach (var a in allArenas)
        {
            a.SaveState();
        }*/

        //initialize shot
        for (int i = 0; i < allArenas.Count; ++i)
        {
            allArenas[i].StartEvaluateShotSequence(forces[i], drawColor);
        }


        for (int step = 0; step < maxSteps; step++)
        {
            for (int i = 0; i < allArenas.Count; ++i)
            {
                allArenas[i].BeforeEvaluationUpdate();
                allArenas[i].ShotNextIfReady();
            }
            Physics.Simulate(Time.fixedDeltaTime);
            for (int i = 0; i < allArenas.Count; ++i)
            {
                allArenas[i].AfterEvaluationUpdate();
            }
            //check whether movement stopped, exit early if yes
            bool done = true;
            for (int i = 0; i < allArenas.Count; ++i)
            {
                if (!allArenas[i].AllShotsComplete())
                    done = false;
            }
            if (done)
                break;
        }

        float maxScore = Mathf.NegativeInfinity;
        List<float> resultScores = new List<float>();

        int count = 0;
        foreach (var a in allArenas)
        {
            var s = a.EndEvaluation();
            resultScores.Add(s);

            if(s > bestScore)
            {
                bestScore = s;
                bestActions = forces[count];
                bestScore = s;
            }

            count++;
        }

        foreach (var a in allArenas)
        {
            if (a != defaultArena)
                DestroyImmediate(a.gameObject);
        }

        Physics.autoSimulation = oldAutoSimulation;
        
        
        return resultScores;
    }

    private void OnDestroy()
    {
        Physics.bounceThreshold = prevBounceThreshold;
    }
}
