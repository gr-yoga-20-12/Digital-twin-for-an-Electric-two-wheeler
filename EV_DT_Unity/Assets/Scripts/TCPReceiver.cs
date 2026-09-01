using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;

public class DTData {
    public float speed;
    public float rpm;
    public float motor_temp;
    public float soc;
    public float remaining_range;
    public float torque;
    public float throttle;
}

public class TCPReceiver : MonoBehaviour {
    public static DTData latestData = new DTData();
    private TcpListener listener;
    private Thread receiveThread;
    public int port = 5005;

    void Start() {
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData() {
        listener = new TcpListener(IPAddress.Any, port);
        listener.Start();
        while (true) {
            TcpClient client = listener.AcceptTcpClient();
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[4096];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string json = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            latestData = JsonConvert.DeserializeObject<DTData>(json);
            client.Close();
        }
    }

    void OnDestroy() {
        if (listener != null) listener.Stop();
        if (receiveThread != null) receiveThread.Abort();
    }
}