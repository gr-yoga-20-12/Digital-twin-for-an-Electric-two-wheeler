using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;

public class PathMapUI : MonoBehaviour
{
    [Header("References")]
    public PathTracer   pathTracer;
    public RawImage     mapBackground;
    public Transform    scooter;

    [Header("Map Settings")]
    public int   textureSize   = 250;
    public float worldScale    = 0.5f;
    public Color bgColor       = new Color(0.05f, 0.08f, 0.05f, 1f);
    public Color pathColor     = Color.cyan;
    public Color dotColor      = Color.yellow;
    public int   dotSize       = 4;

    private Texture2D   mapTexture;
    private Vector3     mapCenter;
    private bool        centerSet = false;

    void Start()
    {
        mapTexture            = new Texture2D(
            textureSize, textureSize);
        mapTexture.filterMode = FilterMode.Bilinear;
        mapBackground.texture = mapTexture;
        ClearMap();
    }

    void Update()
    {
        // Press S key to save path image instantly
        if (Input.GetKeyDown(KeyCode.S))
            SavePathImage();

        // Press C key to clear path
        if (Input.GetKeyDown(KeyCode.C))
            pathTracer.ClearPath();

        if (Input.GetKeyDown(KeyCode.H))
            SaveHighResPathImage();

        if (pathTracer == null || scooter == null) return;

        // Set map center to scooter starting position
        if (!centerSet)
        {
            mapCenter  = scooter.position;
            centerSet  = true;
        }

        RedrawMap();
    }

    void RedrawMap()
    {
        ClearMap();

        List<Vector3> points = pathTracer.GetPathPoints();

        // Draw path lines
        for (int i = 1; i < points.Count; i++)
        {
            Vector2 p1 = WorldToMap(points[i - 1]);
            Vector2 p2 = WorldToMap(points[i]);
            DrawLine(p1, p2, pathColor);
        }

        // Draw current scooter position as yellow dot
        Vector2 scooterPos = WorldToMap(scooter.position);
        DrawDot(scooterPos, dotColor, dotSize);

        mapTexture.Apply();
    }

    Vector2 WorldToMap(Vector3 worldPos)
    {
        float x = (worldPos.x - mapCenter.x) 
                   * worldScale + textureSize / 2f;
        float y = (worldPos.z - mapCenter.z) 
                   * worldScale + textureSize / 2f;
        return new Vector2(x, y);
    }

    void DrawLine(Vector2 p1, Vector2 p2, Color color)
    {
        int x0 = Mathf.RoundToInt(p1.x);
        int y0 = Mathf.RoundToInt(p1.y);
        int x1 = Mathf.RoundToInt(p2.x);
        int y1 = Mathf.RoundToInt(p2.y);

        int dx  = Mathf.Abs(x1 - x0);
        int dy  = Mathf.Abs(y1 - y0);
        int sx  = x0 < x1 ? 1 : -1;
        int sy  = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            SetPixelSafe(x0, y0, color);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 <  dx) { err += dx; y0 += sy; }
        }
    }

    void DrawDot(Vector2 center, Color color, int size)
    {
        int cx = Mathf.RoundToInt(center.x);
        int cy = Mathf.RoundToInt(center.y);
        for (int x = -size; x <= size; x++)
            for (int y = -size; y <= size; y++)
                if (x * x + y * y <= size * size)
                    SetPixelSafe(cx + x, cy + y, color);
    }

    void SetPixelSafe(int x, int y, Color color)
    {
        if (x >= 0 && x < textureSize &&
            y >= 0 && y < textureSize)
            mapTexture.SetPixel(x, y, color);
    }

    void ClearMap()
    {
        Color[] pixels = new Color[textureSize * textureSize];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = bgColor;
        mapTexture.SetPixels(pixels);
    }
    public void SavePathImage()
    {
        // Force a final redraw before saving
        RedrawMap();
        mapTexture.Apply();

        // Generate filename with timestamp
        string timestamp = System.DateTime.Now.ToString(
            "yyyy-MM-dd_HH-mm-ss");
        string filename  = "PathTrace_" + timestamp + ".png";

        // Save to Desktop for easy access
        string desktopPath = System.Environment.GetFolderPath(
            System.Environment.SpecialFolder.Desktop);
        string fullPath = Path.Combine(desktopPath, filename);

        // Convert texture to PNG bytes and write to file
        byte[] pngBytes = mapTexture.EncodeToPNG();
        File.WriteAllBytes(fullPath, pngBytes);

        Debug.Log("Path saved to: " + fullPath);
    }

    public void SaveHighResPathImage()
    {
        // Create a larger texture for presentation quality
        int hiResSize        = 1024;
        Texture2D hiResTex   = new Texture2D(hiResSize, hiResSize);
        hiResTex.filterMode  = FilterMode.Bilinear;

        // Fill background
        Color[] pixels = new Color[hiResSize * hiResSize];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = bgColor;
        hiResTex.SetPixels(pixels);

        // Redraw all path points at high resolution
        float hiResScale = worldScale * (hiResSize / textureSize);
        List<Vector3> points = pathTracer.GetPathPoints();

        for (int i = 1; i < points.Count; i++)
        {
            Vector2 p1 = WorldToMapCustom(
                points[i - 1], hiResSize, hiResScale);
            Vector2 p2 = WorldToMapCustom(
                points[i], hiResSize, hiResScale);
            DrawLineOnTexture(hiResTex, p1, p2,
                pathColor, hiResSize);
        }

        // Draw current scooter dot larger for hi-res
        Vector2 scooterPosHi = WorldToMapCustom(
            scooter.position, hiResSize, hiResScale);
        DrawDotOnTexture(hiResTex, scooterPosHi,
            dotColor, 12, hiResSize);

        hiResTex.Apply();

        // Save to Desktop
        string timestamp  = System.DateTime.Now.ToString(
            "yyyy-MM-dd_HH-mm-ss");
        string filename   = "PathTrace_HiRes_" + 
                            timestamp + ".png";
        string desktopPath = System.Environment.GetFolderPath(
            System.Environment.SpecialFolder.Desktop);
        string fullPath   = Path.Combine(desktopPath, filename);

        byte[] pngBytes   = hiResTex.EncodeToPNG();
        File.WriteAllBytes(fullPath, pngBytes);

        Destroy(hiResTex);
        Debug.Log("Hi-res path saved to: " + fullPath);
    }

    Vector2 WorldToMapCustom(Vector3 worldPos,
        int size, float scale)
    {
        float x = (worldPos.x - mapCenter.x) * scale + size / 2f;
        float y = (worldPos.z - mapCenter.z) * scale + size / 2f;
        return new Vector2(x, y);
    }

    void DrawLineOnTexture(Texture2D tex, Vector2 p1,
        Vector2 p2, Color color, int size)
    {
        int x0  = Mathf.RoundToInt(p1.x);
        int y0  = Mathf.RoundToInt(p1.y);
        int x1  = Mathf.RoundToInt(p2.x);
        int y1  = Mathf.RoundToInt(p2.y);
        int dx  = Mathf.Abs(x1 - x0);
        int dy  = Mathf.Abs(y1 - y0);
        int sx  = x0 < x1 ? 1 : -1;
        int sy  = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            if (x0 >= 0 && x0 < size && y0 >= 0 && y0 < size)
                tex.SetPixel(x0, y0, color);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 <  dx) { err += dx; y0 += sy; }
        }
    }

    void DrawDotOnTexture(Texture2D tex, Vector2 center,
        Color color, int size, int texSize)
    {
        int cx = Mathf.RoundToInt(center.x);
        int cy = Mathf.RoundToInt(center.y);
        for (int x = -size; x <= size; x++)
            for (int y = -size; y <= size; y++)
                if (x * x + y * y <= size * size)
                {
                    int px = cx + x;
                    int py = cy + y;
                    if (px >= 0 && px < texSize &&
                        py >= 0 && py < texSize)
                        tex.SetPixel(px, py, color);
                }
    }
}