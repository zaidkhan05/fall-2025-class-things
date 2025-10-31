import java.io.*;
import java.net.*;
//concurrent server
//application layer
public class applayercon {

    public static void main(String[] args) {
        int port = 7897;
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("[Concurrent Server] Running on port " + port);
            while (true) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("Client connected: " + clientSocket.getInetAddress());
                new Thread(() -> {
                    servicelayer.dataHandler(clientSocket);
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
