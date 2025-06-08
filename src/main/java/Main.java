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

        // Загружаем документ с использованием TextDocumentParser
        Document document = FileSystemDocumentLoader.loadDocument(filePath, new TextDocumentParser());
        DocumentByParagraphSplitter splitter = new DocumentByParagraphSplitter(200, 10);
        List<TextSegment> segments = splitter.split(document);

        // Используем LocalONNXEmbeddingModel вместо LocalEmbeddingModel
        LocalONNXEmbeddingModel embeddingModel = new LocalONNXEmbeddingModel(
                "/home/acer/IdeaProjects/DJI/src/main/resources/allmpnetbasev2"
        );

        // Создаем in-memory хранилище эмбеддингов
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Генерируем эмбеддинги для всех сегментов и сохраняем их в хранилище
        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : segments) {
            Embedding emb = embeddingModel.embed(segment.text());
            embeddings.add(emb);
        }
        embeddingStore.addAll(embeddings, segments);

        // Пример запроса
        String query = "Who is Holmes?";
        Embedding queryEmbedding = embeddingModel.embed(query);

        // Создаем запрос на поиск (топ-3 сегмента)
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(3)
                .build();

        // Выполняем поиск по схожести
        EmbeddingSearchResult<TextSegment> matches = embeddingStore.search(searchRequest);

        // Комбинируем найденные сегменты в единый контекст
        StringBuilder contextBuilder = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : matches.matches()) {
            TextSegment segment = match.embedded();
            contextBuilder.append(segment.text()).append("\n\n");
        }

        String context = contextBuilder.toString();

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
        embeddingModel.close();
    }
}
