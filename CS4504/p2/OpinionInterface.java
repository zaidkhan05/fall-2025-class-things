import java.rmi.*;

public interface OpinionInterface extends Remote {
    void castVote(int choice) throws RemoteException;

    int[] getResult() throws RemoteException;
}
