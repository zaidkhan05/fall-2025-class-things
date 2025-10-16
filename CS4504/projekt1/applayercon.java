// ---- /CountdownServerConcurrent.java ----
import java.io.*;
import java.net.*;

/**
 * CountdownServerConcurrent
 * -------------------------
 * APPLICATION LAYER (Concurrent)
 * Handles multiple clients simultaneously using threads.
 * Each thread uses its own SERVICE LAYER connection (socket streams).
 */
public class applayercon {

    public static void main(String[] args) {
        int port = 7897; // Different port for concurrent server

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("[Concurrent Server] Running on port " + port);

            while (true) {
                Socket clientSocket = serverSocket.accept(); // SERVICE LAYER
                System.out.println("Client connected: " + clientSocket.getInetAddress());

                // Create a thread to handle each client session concurrently
                new Thread(() -> {
                    servicelayer.handleClient(clientSocket);
                    try {
                        clientSocket.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    System.out.println("Client session ended.\n");
                }).start();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
