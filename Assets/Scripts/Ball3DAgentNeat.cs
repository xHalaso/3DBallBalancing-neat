using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Rendering;

public class Ball3DAgentNeat : Agent
{
    [Header("Ball")]
    public GameObject ball;

    [Header("SENSORS")]
    public VectorSensor vectorSensor;

    Rigidbody m_BallRb;
    EnvironmentParameters m_ResetParams;

    public bool useVectorObservations;

    public override void Initialize()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        ResetBall();
    }

    // Let's collect the sensor data such as the rotation around x, z axis, the position of a ball on agent and ball velocity
    public override void CollectObservations(VectorSensor sensor)
    {
        // Total of 8 observations   
        if (true) // if (useVectorObservations)
        {
            var rotationX = gameObject.transform.rotation.x;
            var rotationZ = gameObject.transform.rotation.z;
            var displacement = ball.transform.position - gameObject.transform.position;
            var ballVelocity = m_BallRb.velocity;

            // Normalizations
           /* rotationX = (rotationX - (-1)) / (1 - (-1)) - 1;
            rotationZ = (rotationZ - (-1)) / (1 - (-1)) - 1;
*/
/*            displacement.x = (displacement.x - (-3.0f)) / (3.0f - (-3.0f)) - 1;
            displacement.z = (displacement.z - (-3.0f)) / (3.0f - (-3.0f)) - 1;
            displacement.y = (displacement.x - (-2.0f)) / (4.0f - (-2.0f)) - 1;

            ballVelocity.x = (ballVelocity.x - (-3.0f)) / (3.0f - (-3.0f)) - 1;
            ballVelocity.y = (ballVelocity.y - (-4.0f)) / (0.3f - (-4.0f)) - 1;
            ballVelocity.z = (ballVelocity.z - (-3.0f)) / (3.0f - (-3.0f)) - 1;*/

 /*           Debug.Log("Velocities: X " + ballVelocity.x + " Y " + ballVelocity.y + " Z " + ballVelocity.z);
            Debug.Log("Displacement: " + displacement.ToString());
            Debug.Log("Rotations: " + rotationX + " " + rotationZ);*/

            sensor.AddObservation(rotationZ); // This is a scalar
            sensor.AddObservation(rotationX); // This is a scalar
            sensor.AddObservation(displacement); // This is a Vector3
            sensor.AddObservation(ballVelocity  );  // This is a Vector3
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffer)
    {
        MoveAgent(actionBuffer);
    }

    public void MoveAgent(ActionBuffers actionBuffer)
    {
        var continousActions = actionBuffer.ContinuousActions;
        var discreteActions = actionBuffer.DiscreteActions;

        var actionZ = 2f * Mathf.Clamp(continousActions[0], -1.0f, 1.0f);
        var actionX = 2f * Mathf.Clamp(continousActions[1], -1.0f, 1.0f);
        // Debug.Log("ACTIONS: [X: " + actionX.ToString() + " Z: "+ actionZ.ToString() + "]");
        HandleRotations(actionZ, actionX);

        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            Debug.Log("Pre Reward: " + GetCumulativeReward());
            //AddReward(-1.0f);
            SetReward(-1.0f);
            Debug.Log("Post Reward: " + GetCumulativeReward());
            EndEpisode();
        }
        else
        {
            //SetReward(0.1f);
            AddReward(0.1f);
        }
    }

    private void HandleRotations(float actionZ, float actionX)
    {
        var rotationZ = gameObject.transform.rotation.z;
        var rotationX = gameObject.transform.rotation.x;

        if ((rotationZ < 0.25f && actionZ > 0.0f) || (rotationZ > -0.25f && actionZ < 0f))
        {
            gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
        }

        if ((rotationX < 0.25f && actionX > 0.0f) || (rotationX > -0.25f && actionX < 0f))
        {
            gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(Random.Range(-1f, 1f), 4.0f, Random.Range(-1f, 1f)) + gameObject.transform.position;
        ResetAgent();
        ResetBall();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = -Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }

    private void ResetAgent()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10.0f, 10.0f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10.0f, 10.0f));
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10.0f, 10.0f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10.0f, 10.0f));
    }

    private void ResetBall()
    {
        m_BallRb.mass = m_ResetParams.GetWithDefault("mass", 1.0f);
        var scale = m_ResetParams.GetWithDefault("scale", 1.0f);
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }
}
