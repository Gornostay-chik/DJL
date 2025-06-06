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

        // Генерируем эмбеддинги для всех сегментов
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
        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);
        for (EmbeddingMatch<TextSegment> match : searchResult.matches()) {
            TextSegment matchedSegment = match.embedded();
            System.out.println("Relevance Score: " + match.score());
            String snippet = matchedSegment.text().substring(0, Math.min(300, matchedSegment.text().length()));
            System.out.println(snippet + "...\n");
        }

        // Закрываем модель
        embeddingModel.close();
    }
}
