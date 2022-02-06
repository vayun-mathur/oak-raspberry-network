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
    private PrintWriter m_out;

    public Oak(String hostname, int port) {
        Socket so;
        try {
            so = new Socket(hostname, port);
            m_read = new BufferedReader(new InputStreamReader(so.getInputStream()));
            m_out = new PrintWriter(so.getOutputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }
        while(true) {
            try {
                m_out.println("Hello world " + System.currentTimeMillis());
                m_out.flush();
                System.out.println(System.currentTimeMillis());
                Thread.sleep(20);
                if (m_read.ready()) {
                    String str = m_read.readLine();
                    String[] objs = str.split(";");
                    Detection[] dec = new Detection[objs.length];
                    int i = 0;
                    for(String obj: objs) {
                        if(obj.trim().equals("")) continue;
                        System.out.println(obj);
                        String[] arr = obj.split(",");
                        Detection d = new Detection();
                        d.label = arr[0];
                        d.x1 = Integer.parseInt(arr[1].trim());
                        d.y1 = Integer.parseInt(arr[2].trim());
                        d.x2 = Integer.parseInt(arr[3].trim());
                        d.y2 = Integer.parseInt(arr[4].trim());
                        d.x = Double.parseDouble(arr[5].trim());
                        d.y = Double.parseDouble(arr[6].trim());
                        d.z = Double.parseDouble(arr[7].trim());
                        dec[i++] = d;
                    }
                    evaluateDetections(dec);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public void evaluateDetections(Detection[] detections) {

    }

    public static void main(String[] args) throws IOException, InterruptedException {
        Oak o = new Oak("localhost", 12801);
    }
}
