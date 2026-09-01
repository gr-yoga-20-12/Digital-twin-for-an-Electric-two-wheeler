using UnityEngine;

public class ScooterAnimator : MonoBehaviour
{
    public Transform frontWheelPivot;
    public Transform rearWheelPivot;
    public float wheelRadius = 0.15f;

    void Update()
    {
        float speedMs = TCPReceiver.latestData.speed / 3.6f;
        float rotationSpeed = (speedMs / (2f * Mathf.PI * wheelRadius)) * 360f;
        float delta = rotationSpeed * Time.deltaTime;

        if (frontWheelPivot != null)
            frontWheelPivot.Rotate(delta, 0, 0, Space.Self);
        if (rearWheelPivot != null)
            rearWheelPivot.Rotate(delta, 0, 0, Space.Self);
    }
}
