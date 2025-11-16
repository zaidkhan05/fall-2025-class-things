
// ---- /src/ElectionServer.java ----
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class ElectionServer extends UnicastRemoteObject implements RemoteInterface {

    private int candidate1Votes;
    private int candidate2Votes;

    protected ElectionServer() throws RemoteException {
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
            ElectionServer server = new ElectionServer();
            // NEW (connects to the already running registry)
            Registry registry;
            try {
                registry = LocateRegistry.getRegistry(1099);
                registry.list(); // test connection
            } catch (RemoteException e) {
                System.out.println("No registry found on port 1099, creating a new one...");
                registry = LocateRegistry.createRegistry(1099);
            }
            registry.rebind("ElectionService", server);
            System.out.println("Election server is ready!");
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }
}
