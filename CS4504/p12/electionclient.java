import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

public class electionclient {
    public static void main(String[] args) {
        try {
            // Connect to the RMI registry on the server
            Registry registry = LocateRegistry.getRegistry("localhost", 1099);

            // Lookup the remote object
            remoteinterface stub = (remoteinterface) registry.lookup("ElectionService");

            Scanner scanner = new Scanner(System.in);
            System.out.println("Welcome to the Distributed Election System!");
            while (true) {
                System.out.println("\nOptions:");
                System.out.println("1. Cast Vote");
                System.out.println("2. View Results");
                System.out.println("3. Exit");
                System.out.print("Choose: ");
                int choice = scanner.nextInt();
                scanner.nextLine();

                switch (choice) {
                    case 1:
                        System.out.print("Enter candidate name (Zaid/Fred): ");
                        String name = scanner.nextLine();
                        stub.castVote(name);
                        System.out.println("Vote submitted successfully!");
                        break;

                    case 2:
                        int[] results = stub.getResult();
                        System.out.println("\nCurrent Results:");
                        System.out.println("Zaid: " + results[0]);
                        System.out.println("Fred: " + results[1]);
                        break;

                    case 3:
                        System.out.println("Exiting...");
                        scanner.close();
                        return;

                    default:
                        System.out.println("Invalid choice!");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
