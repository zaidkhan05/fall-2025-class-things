# CS4506 — Project 1 Phase 2

**Title:** Java RMI-Based Election Application

---

## Files included (copy each into the shown path)

### ---- /src/election/ElectionInterface.java ----
```java
package election;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface ElectionInterface extends Remote {
    void castVote(String candidateName) throws RemoteException;
    int[] getResult() throws RemoteException;
}
```

### ---- /src/election/ElectionServer.java ----
```java
package election;

import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.rmi.RemoteException;
import java.util.concurrent.atomic.AtomicInteger;

public class ElectionServer implements ElectionInterface {
    // Two candidates: index 0 -> Alice, index 1 -> Bob
    private final AtomicInteger aliceVotes = new AtomicInteger(0);
    private final AtomicInteger bobVotes = new AtomicInteger(0);

    public ElectionServer() {
    }

    @Override
    public void castVote(String candidateName) throws RemoteException {
        if (candidateName == null) return;
        String n = candidateName.trim().toLowerCase();
        switch (n) {
            case "alice":
                aliceVotes.incrementAndGet();
                System.out.println("[SERVER] Received vote for Alice. Total: " + aliceVotes.get());
                break;
            case "bob":
                bobVotes.incrementAndGet();
                System.out.println("[SERVER] Received vote for Bob. Total: " + bobVotes.get());
                break;
            default:
                System.out.println("[SERVER] Ignored vote for unknown candidate: " + candidateName);
                break;
        }
    }

    @Override
    public int[] getResult() throws RemoteException {
        return new int[]{aliceVotes.get(), bobVotes.get()};
    }

    public static void main(String[] args) {
        try {
            // Create and export the server object
            ElectionServer server = new ElectionServer();
            ElectionInterface stub = (ElectionInterface) UnicastRemoteObject.exportObject(server, 0);

            // Start (or locate) RMI registry on default port 1099
            try {
                LocateRegistry.createRegistry(1099);
                System.out.println("[SERVER] RMI registry created on port 1099");
            } catch (RemoteException re) {
                // registry already exists
                System.out.println("[SERVER] RMI registry already running");
            }

            // Bind the remote object's stub in the registry
            Registry registry = LocateRegistry.getRegistry();
            registry.rebind("Election", stub);
            System.out.println("[SERVER] Election server ready and bound as 'Election'");

        } catch (Exception e) {
            System.err.println("[SERVER] Server exception: " + e.toString());
            e.printStackTrace();
        }
    }
}
```

### ---- /src/election/ElectionClient.java ----
```java
package election;

import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

public class ElectionClient {
    private ElectionClient() {}

    public static void main(String[] args) {
        String host = (args.length >= 1) ? args[0] : "localhost";
        try (Scanner scanner = new Scanner(System.in)) {
            Registry registry = LocateRegistry.getRegistry(host);
            ElectionInterface stub = (ElectionInterface) registry.lookup("Election");

            System.out.println("Connected to Election server at: " + host);

            boolean running = true;
            while (running) {
                System.out.println("\nChoose: 1) Cast vote  2) View results  3) Exit");
                System.out.print("> ");
                String choice = scanner.nextLine().trim();
                switch (choice) {
                    case "1":
                        System.out.print("Enter candidate name (Alice or Bob): ");
                        String name = scanner.nextLine().trim();
                        try {
                            stub.castVote(name);
                            System.out.println("Vote submitted for: " + name);
                        } catch (Exception e) {
                            System.err.println("Error casting vote: " + e.getMessage());
                        }
                        break;
                    case "2":
                        try {
                            int[] res = stub.getResult();
                            System.out.println("Current Results:\nAlice: " + res[0] + "\nBob: " + res[1]);
                        } catch (Exception e) {
                            System.err.println("Error fetching results: " + e.getMessage());
                        }
                        break;
                    case "3":
                        running = false;
                        break;
                    default:
                        System.out.println("Invalid choice. Try 1, 2, or 3.");
                }
            }

            System.out.println("Client exiting.");

        } catch (Exception e) {
            System.err.println("Client exception: " + e.toString());
            e.printStackTrace();
        }
    }
}
```

---

## Implementation Report (required)

### Goal
Build a Java RMI distributed election app with a centralized server tracking votes for two candidates and clients that can cast votes and view results in real time.

### Design decisions
- **RMI**: Use Java RMI with a single remote interface `ElectionInterface` exposing `castVote` and `getResult`.
- **Thread-safety**: Server vote counters use `AtomicInteger` for safe concurrent updates from multiple clients.
- **Registry binding**: Server binds under the name `Election` on default port 1099. The server attempts to create a local registry if none exists.
- **Simple CLI client**: The client provides a text menu for casting votes and viewing results.

### How to compile
From the project root (where `src/` exists):

```bash
# compile
javac -d out src/election/*.java

# run the server
java -cp out election.ElectionServer

# in another terminal, run a client connecting to localhost
java -cp out election.ElectionClient localhost
```

**Notes**:
- If you get `java.rmi.ConnectException: Connection refused`, ensure the server is running and the registry is accessible on port 1099.
- If using a remote host, replace `localhost` with the server IP or hostname when starting the client.

### How to run (step-by-step)
1. Open terminal A: `java -cp out election.ElectionServer`
   - You should see: `[SERVER] RMI registry created on port 1099` (or `already running`) and `[SERVER] Election server ready and bound as 'Election'`.
2. Open terminal B (and C...): `java -cp out election.ElectionClient` (optionally pass hostname/IP as first arg).
3. In the client menu choose `1` to cast a vote or `2` to view results.

### Sample console output
**Server terminal**
```
[SERVER] RMI registry created on port 1099
[SERVER] Election server ready and bound as 'Election'
[SERVER] Received vote for Alice. Total: 1
[SERVER] Received vote for Bob. Total: 1
[SERVER] Received vote for Alice. Total: 2
```

**Client terminal (after casting votes)**
```
Connected to Election server at: localhost

Choose: 1) Cast vote  2) View results  3) Exit
> 1
Enter candidate name (Alice or Bob): Alice
Vote submitted for: Alice

> 2
Current Results:
Alice: 2
Bob: 1
```

### What to include in your submission
- Source files (`ElectionInterface.java`, `ElectionServer.java`, `ElectionClient.java`) in `src/election/`.
- A short report (this document's Implementation Report section) with the run steps and sample output screenshots (capture server+client terminals).

### Common troubleshooting
- `java.rmi.AccessException: remote object unexported` — make sure you keep the server process running.
- Firewall issues — ensure TCP port 1099 is open between client and server when testing across machines.
- Class version mismatch — compile and run with the same JDK.

---

## Extras (optional improvements you can add)
- Add authentication tokens or a simple password to prevent unauthorized voting.
- Persist votes to disk (file or simple DB) so results survive server restarts.
- Extend to N candidates by using a `Map<String, AtomicInteger>` rather than fixed fields.
- Add a GUI client (Swing/JavaFX) for better UX.

---

## Academic integrity
Include in your report a short statement that this code was written by you and any outside resources used (textbooks, online docs). Follow your course's collaboration rules.

---

**End of document**

