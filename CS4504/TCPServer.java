// ---- TCPServer.java ----
import java.io.*;
import java.net.*;

public class TCPServer {
    public static void main(String[] args) {
        int port = 6789;

        try (ServerSocket welcomeSocket = new ServerSocket(port)) {
            System.out.println("Server started. Waiting for connection...");

            Socket connectionSocket = welcomeSocket.accept();
            System.out.println("Client connected: " + connectionSocket.getInetAddress());

            BufferedReader inFromClient = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));

            String clientSentence;
            while ((clientSentence = inFromClient.readLine()) != null) {
                System.out.println("Server received: " + clientSentence);
            }

            System.out.println("Client disconnected.");
        } catch (IOException e) {
            System.err.println("Server error: " + e.getMessage());
        }
    }
}
