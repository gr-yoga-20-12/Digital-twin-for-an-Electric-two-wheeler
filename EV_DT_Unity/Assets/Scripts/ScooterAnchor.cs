/*using UnityEngine;
using CesiumForUnity;
using Unity.Mathematics;

public class ScooterAnchor : MonoBehaviour
{
    [Header("GPS Coordinates")]
    public float latitude         = 13.555731f;
    public float longitude        = 80.026276f;
    public float heightAboveGround = 0.5f;

    private CesiumGlobeAnchor anchor;
    private CesiumGeoreference georef;

    void Start()
    {
        anchor = gameObject.GetComponent<CesiumGlobeAnchor>();
        if (anchor == null)
            anchor = gameObject.AddComponent<CesiumGlobeAnchor>();

        anchor.adjustOrientationForGlobeWhenMoving = true;
        anchor.detectTransformChanges = true;

        georef = FindObjectOfType<CesiumGeoreference>();

        StartCoroutine(WaitAndSnapToGround());
    }

    System.Collections.IEnumerator WaitAndSnapToGround()
    {
        yield return new WaitForSeconds(3f);
        SnapToGround();
    }

    public void SnapToGround()
    {
        Vector3 rayOrigin = transform.position + Vector3.up * 500f;
        Ray ray = new Ray(rayOrigin, Vector3.down);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 1000f))
        {
            transform.position = hit.point + Vector3.up * heightAboveGround;
            Debug.Log("Scooter snapped to ground at height: " + hit.point.y);
        }
        else
        {
            Debug.LogWarning("Ground raycast missed. " +
                             "Try increasing height offset or " +
                             "wait for tiles to load.");
        }
    }
}
*/


using UnityEngine;
using CesiumForUnity;
using Unity.Mathematics;

public class ScooterAnchor : MonoBehaviour
{
    [Header("GPS Coordinates - IIIT Sri City")]
    public double latitude          = 13.555850;
    public double longitude         = 80.026925;
    public double height            = 5.0;
    public float  heightAboveGround = 0.38f;

    private CesiumGlobeAnchor anchor;

    void Start()
    {
        anchor = GetComponent<CesiumGlobeAnchor>();
        if (anchor == null)
            anchor = gameObject.AddComponent<CesiumGlobeAnchor>();

        anchor.longitudeLatitudeHeight = new double3(
            longitude, latitude, height);

        anchor.adjustOrientationForGlobeWhenMoving = true;
        anchor.detectTransformChanges = true;

        StartCoroutine(SnapAfterDelay());
    }

    System.Collections.IEnumerator SnapAfterDelay()
    {
        yield return new WaitForSeconds(2f);

        Vector3 rayOrigin = transform.position + Vector3.up * 200f;
        Ray ray = new Ray(rayOrigin, Vector3.down);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 500f))
        {
            transform.position = hit.point + 
                                 Vector3.up * heightAboveGround;
            Debug.Log("Scooter grounded at: " + hit.point.y);
        }
        else
        {
            Debug.LogWarning("Ground not found — " +
                             "check GroundCollider exists");
        }
    }
}