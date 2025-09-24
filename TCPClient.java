// Modified TCPClient.java
import java.net.*;
import java.io.*;
import java.util.Scanner; // Import Scanner for user input

public class TCPClient {
    public static void main(String args[]) {
        // Arguments supply hostname
        if (args.length == 0) {
            System.out.println("Usage: java TCPClient <hostname>");
            return;
        }

        Socket s = null;
        try {
            int serverPort = 7896;
            s = new Socket(args[0], serverPort); // Use args[0] for hostname
            DataInputStream in = new DataInputStream(s.getInputStream());
            DataOutputStream out = new DataOutputStream(s.getOutputStream());
            
            // MODIFICATION: Add a scanner to read user input
            Scanner userInput = new Scanner(System.in);
            System.out.println("Enter messages to send to the server (type 'exit' to quit):");

            // MODIFICATION: Loop to read user input and send to server
            while (true) {
                String line = userInput.nextLine();
                out.writeUTF(line); // Send the user's line to the server
                if (line.equalsIgnoreCase("exit")) {
                    System.out.println("Exiting client.");
                    break; // Exit the loop
                }
            }
            userInput.close();

        } catch (UnknownHostException e) {
            System.out.println("Socket:" + e.getMessage());
        } catch (EOFException e) {
            System.out.println("EOF:" + e.getMessage());
        } catch (IOException e) {
            System.out.println("Readline:" + e.getMessage());
        } finally {
            if (s != null)
                try {
                    s.close();
                } catch (IOException e) {
                    System.out.println("Close:" + e.getMessage());
                }
        }
    }
}