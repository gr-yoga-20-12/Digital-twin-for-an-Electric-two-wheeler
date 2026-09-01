using UnityEngine;

public class WaypointVisualizer : MonoBehaviour
{
    public Color waypointColor = Color.yellow;
    public float waypointSize  = 2f;

    void OnDrawGizmos()
    {
        Gizmos.color = waypointColor;

        for (int i = 0; i < transform.childCount; i++)
        {
            Transform wp = transform.GetChild(i);

            Gizmos.DrawSphere(wp.position, waypointSize);

            if (i < transform.childCount - 1)
            {
                Transform nextWp = transform.GetChild(i + 1);
                Gizmos.DrawLine(wp.position, nextWp.position);
            }
            else
            {
                Gizmos.DrawLine(wp.position,
                    transform.GetChild(0).position);
            }
        }
    }
}