using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AaltoGames;

public class BilliardGameSystem : MonoBehaviour {
    [HideInInspector]
    public float predictedShotScore = 0;

    public BilliardArena defaultArena;

    // Use this for initialization
    void Start()
    {
        /*for (int i = 1; i < 4 ; ++i)
        {
            var newA = Instantiate(defaultArena, defaultArena.transform.position + Vector3.right * 5 * i, defaultArena.transform.rotation);
        }*/

       
    }

    private void Update()
    {
        //evaluateShots(new List<Vector3>() { Vector3.forward, Vector3.forward, new Vector3(1,0,1) }, Color.green);
    }
    public void shoot(Vector3 force)
    {
        //Rigidbody r = whiteBall.GetComponent<Rigidbody>();
        //r.velocity = force;
        //Uncomment the following to also set the angular velocity of the ball such that it agrees with velocity.
        //This should prevent the ball from first sliding and then decelerating rapidly. However, it doesn't seem to work in practice,
        //and better results are obtained simply by having everything with zero friction and basically no rotation (which however looks ok only without textures...)
        //Vector3 axis = Vector3.Cross(Vector3.up, force.normalized).normalized; //this is the axis of rotation
        //float ballRadius = whiteBall.transform.localScale.x * 0.5f;
        //r.angularVelocity = 10.0f*axis * force.magnitude / (ballRadius * 2.0f * Mathf.PI);

        defaultArena.Shoot(force);
    }

    public bool shotComplete()
    {
        return defaultArena.ShotComplete();
    }
    public void stopAll()
    {
        Rigidbody[] bodies = GameObject.FindObjectsOfType<Rigidbody>();
        foreach (Rigidbody b in bodies)
        {
            b.velocity = Vector3.zero;
        }
    }

    public float evaluateShot(Vector3 force, Color drawColor, int maxSteps=1000)
    {


        var result = evaluateShots(new List<Vector3>() { force }, drawColor, maxSteps);
        return result[0];
    }

    public List<float> evaluateShots(List<Vector3> forces, Color drawColor, int maxSteps = 1000)
    {
        //Disable autosimulation, the manual simulation will otherwise have no effect
        bool oldAutoSimulation = Physics.autoSimulation;
        Physics.autoSimulation = false;

        int numOfForce = forces.Count;
        if (numOfForce <= 0)
            return new List<float>();

        List<BilliardArena>  allArenas = new List<BilliardArena>();
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
        foreach (var a in allArenas)
        {
            a.SaveState();
        }

        //initialize shot
        for (int i = 0; i < allArenas.Count;++i)
        {
            allArenas[i].StartEvaluateShot(forces[i], drawColor);
        }


        for (int step = 0; step < maxSteps; step++)
        {
            for (int i = 0; i < allArenas.Count; ++i)
            {
                allArenas[i].BeforeEvaluationUpdate();
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
                if (!allArenas[i].ShotComplete())
                    done = false;
            }
            if (done)
                break;
        }

        float maxScore = Mathf.NegativeInfinity;
        List<float> resultScores = new List<float>();
        foreach(var a in allArenas)
        {
            var s = a.EndEvaluationShoot();
            resultScores.Add(s);
            maxScore = Mathf.Max(maxScore, s);
           
        }

        foreach(var a in allArenas)
        {
            if (a != defaultArena)
                DestroyImmediate(a.gameObject);
        }

        Physics.autoSimulation = oldAutoSimulation;
        predictedShotScore = maxScore;
        return resultScores;
    }
}
