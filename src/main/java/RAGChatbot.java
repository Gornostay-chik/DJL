import ai.djl.translate.TranslateException;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

public class RAGChatbot {
    private final DocumentProcessor documentProcessor;
    private final ChatEngine chatEngine;

    public RAGChatbot() throws Exception {
        this.documentProcessor = new DocumentProcessor();
        this.chatEngine = new ChatEngine();
    }

    public String uploadDocument(String filePath) {
        try {
            documentProcessor.processDocument(filePath);
            return "Document successfully processed.";
        } catch (IllegalArgumentException e) {
            return "Error: " + e.getMessage();
        }
    }

    public String sendMessage(String message) {
        // Retrieve relevant document chunks based on the user's query
        List<TextSegment> relevantSegments = documentProcessor.retrieveRelevantContext(message, 3);

        // Initialize a StringBuilder for the context
        StringBuilder context = new StringBuilder();

        // Loop through each relevant document segment
        for (TextSegment segment : relevantSegments) {
            // Extract the content of the segment
            String content = segment.text();
            // Append the content to the context string
            context.append(content).append("\n\n");
        }

        // Send the user's message along with the context to the chat engine
        try {
            return chatEngine.sendMessage(message, context.toString());
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    public String resetDocuments() {
        documentProcessor.resetStore();
        return "Document knowledge has been reset.";
    }

    public String resetConversation() {
        chatEngine.resetConversation();
        return "Conversation history has been reset.";
    }

    public String resetAll() {
        resetConversation();
        resetDocuments();
        return "Both conversation history and document knowledge have been reset.";
    }


}