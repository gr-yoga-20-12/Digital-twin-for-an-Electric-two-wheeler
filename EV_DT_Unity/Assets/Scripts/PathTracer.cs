using UnityEngine;
using System.Collections.Generic;

public class PathTracer : MonoBehaviour
{
    [Header("Path Settings")]
    public float recordInterval    = 0.5f;
    public int   maxPathPoints     = 500;
    public Color pathColor         = Color.cyan;
    public float pathWidth         = 0.3f;

    [Header("References")]
    public Transform scooter;

    private LineRenderer    lineRenderer;
    private List<Vector3>   pathPoints;
    private float           timer;

    void Start()
    {
        pathPoints   = new List<Vector3>();
        lineRenderer = gameObject.AddComponent<LineRenderer>();

        lineRenderer.material          = new Material(
            Shader.Find("Universal Render Pipeline/Unlit"));
        lineRenderer.material.color    = pathColor;
        lineRenderer.startColor        = pathColor;
        lineRenderer.endColor          = new Color(
            pathColor.r, pathColor.g, pathColor.b, 0.3f);
        lineRenderer.startWidth        = pathWidth;
        lineRenderer.endWidth          = pathWidth * 0.5f;
        lineRenderer.positionCount     = 0;
        lineRenderer.useWorldSpace     = true;
    }

    void Update()
    {
        if (scooter == null) return;

        // Only record when scooter is moving
        if (TCPReceiver.latestData.speed < 0.5f) return;

        timer += Time.deltaTime;
        if (timer < recordInterval) return;
        timer = 0f;

        // Record position slightly above ground to stay visible
        Vector3 pos = scooter.position + Vector3.up * 0.3f;
        pathPoints.Add(pos);

        // Keep path within max points limit
        if (pathPoints.Count > maxPathPoints)
            pathPoints.RemoveAt(0);

        // Update LineRenderer
        lineRenderer.positionCount = pathPoints.Count;
        lineRenderer.SetPositions(pathPoints.ToArray());
    }

    public void ClearPath()
    {
        pathPoints.Clear();
        lineRenderer.positionCount = 0;
    }

    public List<Vector3> GetPathPoints()
    {
        return pathPoints;
    }
}