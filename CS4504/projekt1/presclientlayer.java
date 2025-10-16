// ---- /CountdownClient.java ----
import java.io.*;
import java.net.*;

/**
 * CountdownClient
 * ---------------
 * PRESENTATION LAYER
 * Accepts user input (integer n), sends it to the server via SERVICE LAYER,
 * and displays countdown results received from the server.
 */
public class presclientlayer {

    public static void main(String[] args) {
        String host = "localhost"; // Server hostname
        int port = 7896; // Change to 7897 to test concurrent server

        try (
            // SERVICE LAYER: Create socket and I/O streams
            Socket socket = new Socket(host, port);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
        ) {
            // Get user input for countdown
            System.out.print("Enter a number for countdown: ");
            int n = Integer.parseInt(console.readLine());

            // Send n to server
            out.println(n);

            // Display countdown results from server
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
