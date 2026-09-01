using UnityEngine;

public class CameraManager : MonoBehaviour
{
    public Camera dashboardCamera;  // Camera 1: fixed side view
    public Camera followCamera;     // Camera 2: follow behind
    public Camera orbitCamera;      // Camera 3: 360 orbit

    void Start()
    {
        SetCamera(1); // start with dashboard view
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1)) SetCamera(1);
        if (Input.GetKeyDown(KeyCode.Alpha2)) SetCamera(2);
        if (Input.GetKeyDown(KeyCode.Alpha3)) SetCamera(3);
    }

    void SetCamera(int index)
    {
        dashboardCamera.gameObject.SetActive(index == 1);
        followCamera.gameObject.SetActive(index == 2);
        orbitCamera.gameObject.SetActive(index == 3);

        Debug.Log("Camera switched to: " + index);
    }
}