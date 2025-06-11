import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        // Initialize the RAG chatbot
        RAGChatbot chatbot = new RAGChatbot();

        // Upload a document
        String directoryPath = "src/main/resources/books";
        File directory = new File(directoryPath);
        for (File file : directory.listFiles()) {
            String result = chatbot.uploadDocument(String.valueOf(file));
            System.out.println(result);
        }

        // Send a message about the document
        String query = "Who is Blue?";
        String response = chatbot.sendMessage(query);
        System.out.println("\nQuestion: " + query);
        System.out.println("Answer: " + response);
    }
}