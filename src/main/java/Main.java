import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;

import java.util.ArrayList;
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
        String query = "What is the main mystery in the story?";

        // Take 3 relevant segments for context
        List<TextSegment> relevantSegments = processor.retrieveRelevantContext(query, 3);

        String context = relevantSegments.toString();

        // Создаем подсказку (prompt) с учетом контекста и вопроса
        String promptTemplate = "Answer the following question based on the provided context.\n\n"
                + "Context:\n%s\n\n"
                + "Question: %s";

        String prompt = String.format(promptTemplate, context, query);

        System.out.println("Сформированная подсказка:");
        System.out.println(prompt);
        System.out.println("\n---\n");

        // Создаем объект диалоговой модели и генерируем ответ
        String tfModelDir = "/home/acer/IdeaProjects/DJI/src/main/resources/GPT2";
        LocalONNXDialogModel dialogModel = new LocalONNXDialogModel(tfModelDir);

        String answer = dialogModel.chat(prompt);
        System.out.println("Сгенерированный ответ:");
        System.out.println(answer);

        // Закрываем ресурсы моделей
        dialogModel.close();

    }
}
