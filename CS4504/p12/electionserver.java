import java.rmi.*;
import java.rmi.server.*;
import java.rmi.registry.*;


class Elect extends UnicastRemoteObject implements ElectionInterface {
    int candidate1Votes = 0;
    int candidate2Votes = 0;

    Elect() throws RemoteException {
        super();
    }

    @Override
    public void castVote(String candidateName) throws RemoteException {
        // Implementation here
        if (candidateName.equalsIgnoreCase("Zaid")) {
            candidate1Votes++;
            System.out.println("Vote number " + candidate1Votes + " cast for Zaid.");
        }
        else if (candidateName.equalsIgnoreCase("Fred")) {
            candidate2Votes++;
            System.out.println("Vote number " + candidate2Votes + " cast for Fred.");
        }
        else {
            System.out.println("Invalid candidate: " + candidateName);
        }
    }

    @Override
    public int[] getResult() throws RemoteException {
        return new int[]{candidate1Votes, candidate2Votes};

    }
}


public class ElectionServer {
    public static void main(String[] args) {
        try{
            Elect server = new Elect();
            LocateRegistry.createRegistry(1900);
            Naming.rebind("rmi://localhost:1900/ElectionService", server);
            System.out.println("Election Server is ready.");
        }
        catch (Exception e) {
            System.out.println("Election Server failed: " + e);
        }
    }
}
