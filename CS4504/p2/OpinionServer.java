import java.rmi.*;
import java.rmi.server.*;
import java.rmi.registry.*;


class Elect extends UnicastRemoteObject implements OpinionInterface {
    int yesVotes = 0;
    int noVotes = 0;
    int dontCareVotes = 0;

    Elect() throws RemoteException {
        super();
    }

    @Override
    public void castVote(int choice) throws RemoteException {
        switch (choice) {
            case 1:
                yesVotes++;
                System.out.println("Vote cast for Yes.");
                break;
            case 2:
                noVotes++;
                System.out.println("Vote cast for No.");
                break;
            case 3:
                dontCareVotes++;
                System.out.println("Vote cast for Don't Care.");
                break;
            default:
                System.out.println("Invalid vote choice: " + choice);
                break;
        }
    }

    @Override
    public int[] getResult() throws RemoteException {
        // "10 yes, 2 no, 5 don't care"
        return new int[]{yesVotes, noVotes, dontCareVotes};

    }
}


public class OpinionServer {
    public static void main(String[] args) {
        try{
            Elect server = new Elect();
            LocateRegistry.createRegistry(1900);
            Naming.rebind("rmi://localhost:1900/OpinionService", server);
            System.out.println("Opinion Server is ready.");
        }
        catch (Exception e) {
            System.out.println("Opinion Server failed: " + e);
        }
    }
}
