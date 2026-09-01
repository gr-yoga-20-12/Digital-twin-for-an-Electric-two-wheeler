using UnityEngine;

public class RoadFollower : MonoBehaviour
{
    [Header("Waypoint Settings")]
    public Transform waypointManager;
    public float waypointReachDistance = 3f;

    [Header("Movement Settings")]
    public float currentSpeed  = 0f;
    public float rotationSpeed = 5f;

    [Header("Ground Snapping")]
    public float     groundOffset = 0.0f;
    public LayerMask groundLayer;

    [Header("Wheel References")]
    public Transform frontWheel;
    public Transform rearWheel;
    public float     wheelRadius = 0.15f;

    private Transform[] waypoints;
    private int   currentWaypointIndex = 0;
    private float distanceTravelled    = 0f;

    void Start()
    {
        waypoints = new Transform[waypointManager.childCount];
        for (int i = 0; i < waypointManager.childCount; i++)
        {
            waypoints[i] = waypointManager.GetChild(i);
        }
        Debug.Log("RoadFollower: loaded " +
                  waypoints.Length + " waypoints");
    }

    void Update()
    {
        if (waypoints == null || waypoints.Length == 0)
            return;

        // Swap the commented line below with the hardcoded
        // one once Python is connected
        currentSpeed = TCPReceiver.latestData.speed / 3.6f;
        //currentSpeed = 5f;

        if (currentSpeed < 0.1f)
            return;

        MoveAlongRoad();
        SnapToGround();
    }

    void MoveAlongRoad()
    {
        Transform targetWaypoint = waypoints[currentWaypointIndex];

        Vector3 direction = (targetWaypoint.position -
                             transform.position).normalized;
        direction.y = 0;

        transform.position += direction *
                               currentSpeed * Time.deltaTime;

        if (direction != Vector3.zero)
        {
            Quaternion targetRotation =
                Quaternion.LookRotation(direction);
            transform.rotation = Quaternion.Slerp(
                transform.rotation,
                targetRotation,
                rotationSpeed * Time.deltaTime);
        }

        float distToWaypoint = Vector3.Distance(
            new Vector3(transform.position.x, 0,
                        transform.position.z),
            new Vector3(targetWaypoint.position.x, 0,
                        targetWaypoint.position.z));

        if (distToWaypoint < waypointReachDistance)
        {
            currentWaypointIndex =
                (currentWaypointIndex + 1) % waypoints.Length;
            Debug.Log("Reached waypoint " +
                      currentWaypointIndex);
        }

        distanceTravelled += currentSpeed * Time.deltaTime;
    }

    void SnapToGround()
    {
        if (frontWheel == null || rearWheel == null)
        {
            SimpleSnapToGround();
            return;
        }

        Vector3 frontRayOrigin = frontWheel.position +
                                  Vector3.up * 2f;
        Ray frontRay = new Ray(frontRayOrigin, Vector3.down);
        RaycastHit frontHit;

        Vector3 rearRayOrigin = rearWheel.position +
                                 Vector3.up * 2f;
        Ray rearRay = new Ray(rearRayOrigin, Vector3.down);
        RaycastHit rearHit;

        bool frontHitGround = Physics.Raycast(
            frontRay, out frontHit, 10f, groundLayer);
        bool rearHitGround  = Physics.Raycast(
            rearRay,  out rearHit,  10f, groundLayer);

        if (frontHitGround && rearHitGround)
        {
            float avgGroundHeight = (frontHit.point.y +
                                     rearHit.point.y) / 2f;

            float pivotToWheelBottom =
                transform.position.y -
                (frontWheel.position.y - wheelRadius);

            transform.position = new Vector3(
                transform.position.x,
                avgGroundHeight + pivotToWheelBottom,
                transform.position.z);

            float heightDiff = frontHit.point.y -
                               rearHit.point.y;
            float distance   = Vector3.Distance(
                new Vector3(frontWheel.position.x, 0,
                            frontWheel.position.z),
                new Vector3(rearWheel.position.x, 0,
                            rearWheel.position.z));

            if (distance > 0.01f)
            {
                float angle = Mathf.Atan2(
                    heightDiff, distance) * Mathf.Rad2Deg;
                transform.rotation = Quaternion.Euler(
                    angle,
                    transform.rotation.eulerAngles.y,
                    0);
            }
        }
        else
        {
            SimpleSnapToGround();
        }
    }

    void SimpleSnapToGround()
    {
        Vector3 rayOrigin = transform.position +
                             Vector3.up * 10f;
        Ray ray = new Ray(rayOrigin, Vector3.down);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 20f, groundLayer))
        {
            transform.position = new Vector3(
                transform.position.x,
                hit.point.y + groundOffset,
                transform.position.z);
        }
    }

    public int GetCurrentWaypointIndex()
    {
        return currentWaypointIndex;
    }

    public float GetDistanceTravelled()
    {
        return distanceTravelled;
    }
}