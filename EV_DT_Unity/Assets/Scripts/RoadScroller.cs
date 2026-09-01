using UnityEngine;

public class RoadScroller : MonoBehaviour
{
    public float resetDistance = 90f;
    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
    }

    void Update()
    {
        float speedMs = TCPReceiver.latestData.speed / 3.6f;
        if (speedMs < 0.1f) return;

        // Move scooter forward
        transform.Translate(Vector3.forward * speedMs * Time.deltaTime, Space.World);

        // Loop back to start when it reaches end of road
        if (transform.position.z - startPosition.z > resetDistance)
            transform.position = startPosition;
    }
}