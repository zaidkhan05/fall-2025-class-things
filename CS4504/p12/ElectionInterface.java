import java.rmi.*;

public interface ElectionInterface extends Remote {
    void castVote(String candidateName) throws RemoteException;

    int[] getResult() throws RemoteException;
}
