import java.net.*;
import java.io.*;
import java.util.Scanner;
public class TCPClient {
	public static void main (String args[]) {
		// arguments supply message and hostname
		Socket s = null;
		Scanner sc = new Scanner(System.in);
		while(true){		
			try{
				int serverPort = 7896;
				s = new Socket("localhost", serverPort);
				DataInputStream in = new DataInputStream( s.getInputStream());
				DataOutputStream out =new DataOutputStream( s.getOutputStream());
				System.out.print("Enter message: ");
				String message = sc.nextLine();
				out.writeUTF(message);
				String data = in.readUTF();
				System.out.println("Received: "+ data);
			}
			catch (UnknownHostException e){
				System.out.println("Socket:"+e.getMessage());
			}
			catch (EOFException e){
				System.out.println("EOF:"+e.getMessage());
			}
			catch (IOException e){
				System.out.println("readline:"+e.getMessage());
			}
			finally {
				if(s!=null) try {
					s.close();
				}
				catch (IOException e){
					System.out.println("close:"+e.getMessage());
				}
			}
    	}
	}
}
