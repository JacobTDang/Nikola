using System.Collections;
using UnityEngine;
using NativeWebSocket;

public class CameraStreamer : MonoBehaviour
{
    public string url = "ws://localhost:8765";
    public RenderTexture rt;
    [Range(1, 100)] public int jpegQuality = 60;

    WebSocket ws;

    async void Start()
    {
        ws = new WebSocket(url);
        ws.OnOpen += () => Debug.Log("WS connected");
        ws.OnError += e => Debug.LogError("WS error: " + e);
        ws.OnClose += e => Debug.Log("WS closed: " + e);
        await ws.Connect();
    }

    void LateUpdate()           // after camera rendered
    {
        if (ws == null || ws.State != WebSocketState.Open) return;
        StartCoroutine(SendFrame());
    }

    IEnumerator SendFrame()
    {
        var prev = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();
        byte[] jpg = tex.EncodeToJPG(jpegQuality);
        Destroy(tex);

        RenderTexture.active = prev;
        ws.Send(jpg);
        yield return null;
    }

    async void OnApplicationQuit()
    { if (ws != null) await ws.Close(); }
}
