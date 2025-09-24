// Modified TCPServer.java
import java.net.*;
import java.io.*;

public class TCPServer {
    public static void main(String args[]) {
        try {
            int serverPort = 7896; // The server port
            ServerSocket listenSocket = new ServerSocket(serverPort);
            System.out.println("Server is listening on port " + serverPort + "...");
            while (true) {
                Socket clientSocket = listenSocket.accept();
                System.out.println("Connection established with " + clientSocket.getRemoteSocketAddress());
                Connection c = new Connection(clientSocket);
            }
        } catch (IOException e) {
            System.out.println("Listen socket:" + e.getMessage());
        }
    }
}

class Connection extends Thread {
    DataInputStream in;
    DataOutputStream out;
    Socket clientSocket;

    public Connection(Socket aClientSocket) {
        try {
            clientSocket = aClientSocket;
            in = new DataInputStream(clientSocket.getInputStream());
            out = new DataOutputStream(clientSocket.getOutputStream());
            this.start();
        } catch (IOException e) {
            System.out.println("Connection:" + e.getMessage());
        }
    }

    public void run() {
        try {
            // MODIFICATION: Loop to continuously read messages from the client
            while (true) {
                String data = in.readUTF(); // Read a line of data from the stream
                System.out.println("Received: " + data); // Print the received data
            }
        } catch (EOFException e) {
            System.out.println("Client disconnected.");
        } catch (IOException e) {
            System.out.println("Readline error:" + e.getMessage());
        } finally {
            try {
                clientSocket.close();
            } catch (IOException e) {
                /* close failed */
            }
        }
    }
}