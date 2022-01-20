import java.io.*;
import java.net.*;
import java.net.http.HttpClient;
import java.net.http.HttpResponse;
import java.util.Scanner;

public class python {
    public static void main(String[] args) throws IOException, InterruptedException {
        Socket so = new Socket("localhost", 12801);
        BufferedReader br = new BufferedReader(new InputStreamReader(so.getInputStream()));

        while(true) {
            if (br.ready()) {
                String str = br.readLine();
                System.out.println(str);
            }
            Thread.sleep(10);
        }
    }
}
