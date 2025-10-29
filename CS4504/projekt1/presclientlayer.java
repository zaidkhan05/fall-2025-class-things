import java.io.*;
import java.net.*;
//client
//presentation layer
public class presclientlayer {

    public static void main(String[] args) {
        String host = "localhost";
        int port = 7896; // Change to 7897 to test concurrent server
        try (
            Socket socket = new Socket(host, port);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
        ) {
            System.out.print("Enter a number for countdown: ");
            int n = Integer.parseInt(console.readLine());
            out.println(n);
            String response;
            while ((response = in.readLine()) != null) {
                System.out.println("Server: " + response);
                if (response.equals("Countdown complete!")) break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
