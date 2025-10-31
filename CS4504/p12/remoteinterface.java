import java.rmi.Remote;
import java.rmi.RemoteException;

public interface remoteinterface extends Remote {
    // Cast a vote for the given candidate name
    void castVote(String candidateName) throws RemoteException;

    // Return the current vote counts as [candidate1Votes, candidate2Votes]
    int[] getResult() throws RemoteException;
}
