// CameraStreamer.cs
// Copy this into Unity's Assets/Scripts when needed.

using UnityEngine;
using System.Net.Sockets;
using System.IO;

public class CameraStreamer : MonoBehaviour
{
    public Camera carCamera;
    public string serverIP = "127.0.0.1";
    public int serverPort = 5005;
    public int captureWidth = 320;
    public int captureHeight = 240;
    public float sendInterval = 0.1f;

    private TcpClient client;
    private NetworkStream stream;
    private RenderTexture renderTexture;
    private Texture2D texture;
    private float timer = 0f;

    void Start()
    {
        renderTexture = new RenderTexture(captureWidth, captureHeight, 24);
        texture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        carCamera.targetTexture = renderTexture;

        client = new TcpClient(serverIP, serverPort);
        stream = client.GetStream();
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= sendInterval)
        {
            timer = 0f;
            SendFrame();
        }
    }

    void SendFrame()
    {
        RenderTexture.active = renderTexture;
        carCamera.Render();
        texture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        texture.Apply();
        byte[] imageBytes = texture.EncodeToJPG();

        byte[] sizeBytes = System.BitConverter.GetBytes(imageBytes.Length);
        stream.Write(sizeBytes, 0, sizeBytes.Length);
        stream.Write(imageBytes, 0, imageBytes.Length);
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}
