using UnityEngine;
using CesiumForUnity;
using Unity.Mathematics;

public class CampusAnchor : MonoBehaviour
{
    void Start()
    {
        CesiumGlobeAnchor anchor = GetComponent<CesiumGlobeAnchor>();
        if (anchor == null)
            anchor = gameObject.AddComponent<CesiumGlobeAnchor>();

        // Center of IIIT Sri City campus
        anchor.longitudeLatitudeHeight = new double3(
            80.029167,  // longitude
            13.559383,  // latitude
            0.0       // height at ground level
        );

        anchor.adjustOrientationForGlobeWhenMoving = true;
    }
}