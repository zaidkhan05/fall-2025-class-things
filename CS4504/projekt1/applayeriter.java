import java.io.*;
import java.net.*;
//iterative server
//application layer
public class applayeriter {

    public static void main(String[] args) {
        int port = 7896;
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("[Iterative Server] Running on port " + port);
            while (true) {
                System.out.println("Waiting for a client...");
                Socket clientSocket = serverSocket.accept();
                System.out.println("Connected to client: " + clientSocket.getInetAddress());
                servicelayer.dataHandler(clientSocket);
                clientSocket.close();
                System.out.println("Client session closed.\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
