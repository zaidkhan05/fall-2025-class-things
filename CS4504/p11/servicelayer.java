import java.io.*;
import java.net.*;
//data/service layer
public class servicelayer {
    public static void dataHandler(Socket socket) {
        try (
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        ){
            String input = in.readLine();
            int n = Integer.parseInt(input);
            int countdown = n;
            System.out.println("Received n = " + n + " from client.");
            for (int i = n; i >= 1; i--) {
                out.println(i);
                Thread.sleep(500);
            }
            System.out.println("Countdown from " + countdown + " complete!");
            out.println("Countdown from " + countdown + " complete!");
        }catch (IOException | NumberFormatException | InterruptedException e){
            System.err.println("Error handling client: " + e.getMessage());
        }
    }
}
