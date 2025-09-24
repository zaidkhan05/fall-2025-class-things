// ---- TCPClient.java ----
import java.io.*;
import java.net.*;

public class TCPClient {
    public static void main(String[] args) {
        String serverName = "localhost"; // or IP address
        int port = 6789;

        try (Socket clientSocket = new Socket(serverName, port);
             DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
             BufferedReader inFromUser = new BufferedReader(new InputStreamReader(System.in))) {

            System.out.println("Connected to server at " + serverName + ":" + port);
            System.out.println("Type 'exit' to quit.");

            String sentence;
            while (true) {
                System.out.print("Client> ");
                sentence = inFromUser.readLine();

                if (sentence == null || sentence.equalsIgnoreCase("exit")) {
                    System.out.println("Closing connection...");
                    break;
                }

                outToServer.writeBytes(sentence + '\n');
            }
        } catch (IOException e) {
            System.err.println("Client error: " + e.getMessage());
        }
    }
}
