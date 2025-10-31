// ---- /src/ElectionServer.java ----
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class electionserver extends UnicastRemoteObject implements remoteinterface {

    private int candidate1Votes;
    private int candidate2Votes;

    protected electionserver() throws RemoteException {
        super();
        candidate1Votes = 0;
        candidate2Votes = 0;
    }

    @Override
    public synchronized void castVote(String candidateName) throws RemoteException {
        if (candidateName.equalsIgnoreCase("Zaid")) {
            candidate1Votes++;
            System.out.println("Vote cast for Zaid");
        } else if (candidateName.equalsIgnoreCase("Fred")) {
            candidate2Votes++;
            System.out.println("Vote cast for Fred");
        } else {
            System.out.println("Invalid candidate: " + candidateName);
        }
    }

    @Override
    public synchronized int[] getResult() throws RemoteException {
        return new int[]{candidate1Votes, candidate2Votes};
    }

    public static void main(String[] args) {
        try {
            // Create and export the remote object
            electionserver server = new electionserver();

            // Create RMI registry on port 1099 if not already running
            Registry registry = LocateRegistry.createRegistry(1099);

            // Bind the remote object to a name
            registry.rebind("ElectionService", server);

            System.out.println("Election server is ready!");
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }
}
