// ---- /CountdownHandler.java ----
import java.io.*;
import java.net.*;

/**
 * CountdownHandler
 * ----------------
 * SHARED LOGIC COMPONENT
 * Executes the countdown operation after receiving input from a client.
 * Operates over the SERVICE LAYER (socket I/O).
 */
public class servicelayer {

    public static void handleClient(Socket socket) {
        try (
            // SERVICE LAYER: I/O streams for communication
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        ) {
            // Read integer n from client
            String input = in.readLine();
            int n = Integer.parseInt(input);
            System.out.println("Received n = " + n + " from client.");

            // Perform countdown logic
            for (int i = n; i >= 1; i--) {
                out.println(i); // Send countdown value
                Thread.sleep(500); // Short delay to visualize sequence
            }

            out.println("Countdown complete!"); // Indicate end of countdown

        } catch (IOException | NumberFormatException | InterruptedException e) {
            System.err.println("Error handling client: " + e.getMessage());
        }
    }
}
