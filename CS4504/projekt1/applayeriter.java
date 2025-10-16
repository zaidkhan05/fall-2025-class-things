// ---- /CountdownServerIterative.java ----
import java.io.*;
import java.net.*;

/**
 * CountdownServerIterative
 * ------------------------
 * APPLICATION LAYER (Iterative)
 * Handles one client connection at a time.
 * Communicates via the SERVICE LAYER (socket I/O streams).
 * Receives an integer n and sends back n countdown messages.
 */
public class applayeriter {

    public static void main(String[] args) {
        int port = 7896; // Port for iterative server

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("[Iterative Server] Running on port " + port);

            // The server runs indefinitely, handling one client per loop
            while (true) {
                System.out.println("Waiting for a client...");
                Socket clientSocket = serverSocket.accept(); // SERVICE LAYER connection
                System.out.println("Connected to client: " + clientSocket.getInetAddress());

                // Delegate countdown logic
                servicelayer.handleClient(clientSocket);

                clientSocket.close(); // End session
                System.out.println("Client session closed.\n");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
