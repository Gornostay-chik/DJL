
public class Main {
    public static void main(String[] args) throws Exception {
        // Initialize the RAG chatbot
        RAGChatbot chatbot = new RAGChatbot();

        // Upload a document
        String result = chatbot.uploadDocument("src/main/resources/the_adventure_of_the_blue_carbuncle.txt");
        System.out.println(result);

        // Send a message about the document
        String query = "Who is Holmes?";
        String response = chatbot.sendMessage(query);
        System.out.println("\nQuestion: " + query);
        System.out.println("Answer: " + response);
    }
}