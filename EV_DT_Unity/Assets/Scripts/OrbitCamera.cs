using UnityEngine;

public class OrbitCamera : MonoBehaviour
{
    public Transform target;

    [Header("Orbit Settings")]
    public float distance = 5f;
    public float orbitSpeed = 3f;

    [Header("Zoom Settings")]
    public float zoomSpeed = 15f;
    public float minDistance = 0.5f;
    public float maxDistance = 50f;

    [Header("Pan Settings")]
    public float panSpeed = 0.02f;

    private float currentX = 0f;
    private float currentY = 20f;
    private Vector3 targetOffset = Vector3.zero;

    void Update()
    {
        // Left click drag — Orbit
        if (Input.GetMouseButton(0))
        {
            currentX += Input.GetAxis("Mouse X") * orbitSpeed;
            currentY -= Input.GetAxis("Mouse Y") * orbitSpeed;
            currentY = Mathf.Clamp(currentY, -10f, 89f);
        }

        // Middle click drag — Pan
        if (Input.GetMouseButton(2))
        {
            targetOffset -= transform.right * Input.GetAxis("Mouse X") * panSpeed * distance;
            targetOffset -= transform.up * Input.GetAxis("Mouse Y") * panSpeed * distance;
        }

        // Scroll wheel — Zoom
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (scroll != 0f)
        {
            distance -= scroll * zoomSpeed;
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
        }

        // R key — Reset view
        if (Input.GetKeyDown(KeyCode.R))
        {
            currentX = 0f;
            currentY = 20f;
            distance = 5f;
            targetOffset = Vector3.zero;
        }
    }

    void LateUpdate()
    {
        if (target == null) return;

        Quaternion rotation = Quaternion.Euler(currentY, currentX, 0);
        Vector3 focusPoint = target.position + targetOffset;
        Vector3 pos = focusPoint - (rotation * Vector3.forward * distance);
        pos.y = Mathf.Max(pos.y, 0.1f);

        transform.position = pos;
        transform.LookAt(focusPoint + Vector3.up * 0.5f);
    }
}