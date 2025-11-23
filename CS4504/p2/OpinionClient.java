import java.rmi.*;
import java.util.Scanner;
public class OpinionClient {
    public static void main(String[] args) {
        String host = "localhost";
        int port = 1900;
        try{
            OpinionInterface opinionStub = (OpinionInterface)Naming.lookup("rmi://" + host + ":" + port + "/OpinionService");
            System.out.println("Welcome to the Opinion Voting System!");
            Scanner scanner = new Scanner(System.in);
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
                        System.out.println("Should we make darker tint legal?: ");
                        System.out.println("1. Yes, 2. No, 3. Don't Care");
                        int voteChoice = scanner.nextInt();
                        scanner.nextLine();
                        opinionStub.castVote(voteChoice);
                        System.out.println("Vote submitted successfully!");
                        break;

                    case 2:
                        int[] results = opinionStub.getResult();
                        System.out.println("\nCurrent Results:");
                        System.out.print(results[0] + " yes, ");
                        System.out.print(results[1] + " no, ");
                        System.out.println(results[2] + " don't care");
                        break;

                    case 3:
                        System.out.println("Exiting...");
                        scanner.close();
                        return;

                    default:
                        System.out.println("Invalid choice!");
                }
            }
        }
        catch (Exception e) {
            System.out.println("Opinion Client failed: " + e);
        }
    }
}
