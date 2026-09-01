using UnityEngine;
using UnityEngine.UI;

public class UIUpdater : MonoBehaviour
{
    public Text speedText;
    public Text rpmText;
    public Text tempText;
    public Text socText;
    public Text rangeText;
    public GameObject overheatWarning;
    public GameObject lowBatteryWarning;

    public float overheatThreshold = 80f;
    public float lowBatteryThreshold = 20f;

    void Update()
    {
        DTData d = TCPReceiver.latestData;

        if (speedText != null)   speedText.text  = "Speed:  " + d.speed.ToString("F1") + " km/h";
        if (rpmText != null)     rpmText.text    = "RPM:    " + d.rpm.ToString("F0");
        if (tempText != null)    tempText.text   = "Temp:   " + d.motor_temp.ToString("F1") + " C";
        if (socText != null)     socText.text    = "Battery:" + d.soc.ToString("F1") + " %";
        if (rangeText != null)   rangeText.text  = "Range:  " + d.remaining_range.ToString("F1") + " km";

        if (overheatWarning != null)
            overheatWarning.SetActive(d.motor_temp > overheatThreshold);
        if (lowBatteryWarning != null)
            lowBatteryWarning.SetActive(d.soc < lowBatteryThreshold);
    }
}