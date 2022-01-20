import java.io.*;
import java.net.*;
import java.net.http.HttpClient;
import java.net.http.HttpResponse;
import java.util.Scanner;

public class Oak {

    public static class Detection {
        String label;
        int x1, y1, x2, y2;
        double x, y, z;
    }

    private BufferedReader m_read;

    public Oak(String hostname, int port) {
        Socket so;
        try {
            so = new Socket(hostname, port);
            m_read = new BufferedReader(new InputStreamReader(so.getInputStream()));
        } catch (IOException e) {
            e.printStackTrace();
        }
        while(true) {
            try {
                if (m_read.ready()) {
                    String str = m_read.readLine();
                    String[] objs = str.split(";");
                    Detection[] dec = new Detection[objs.length];
                    int i = 0;
                    for(String obj: objs) {
                        String[] arr = obj.split(",");
                        Detection d = new Detection();
                        d.label = arr[0];
                        d.x1 = Integer.parseInt(arr[1]);
                        d.y1 = Integer.parseInt(arr[2]);
                        d.x2 = Integer.parseInt(arr[3]);
                        d.y2 = Integer.parseInt(arr[4]);
                        d.x = Double.parseDouble(arr[5]);
                        d.y = Double.parseDouble(arr[6]);
                        d.z = Double.parseDouble(arr[7]);
                        dec[i++] = d;
                    }
                    evaluateDetections(dec);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void evaluateDetections(Detection[] detections) {

    }

    public static void main(String[] args) throws IOException, InterruptedException {
        Oak o = new Oak("localhost", 12802);
    }
}
