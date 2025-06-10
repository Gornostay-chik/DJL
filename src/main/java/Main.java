import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        // Включаем оффлайн-режим DJL
        System.setProperty("djl.offline", "true");

        // Определяем путь к текстовому документу
        String filePath = "src/main/resources/the_adventure_of_the_blue_carbuncle.txt";

        // Process a document and enable EmbeddingModel
        DocumentProcessor processor = new DocumentProcessor();
        processor.processDocument(filePath);

        // Define a query
        String query = "Who is Holmes?";

        // Take 3 relevant segments for context
        List<TextSegment> relevantSegments = processor.retrieveRelevantContext(query, 3);

        StringBuilder sb = new StringBuilder();
        relevantSegments.forEach(rs -> sb.append(rs.text()).append("\n"));
        String context = sb.toString();

        ChatEngine chatEngine = new ChatEngine();

        String response = chatEngine.sendMessage(query, context);

        System.out.println("Question: " + query);

        System.out.println("Answer: " + response);

        // Закрываем ресурсы моделей
     //   dialogModel.close();
        processor.closeModel();
        processor.resetStore();

    }
}
