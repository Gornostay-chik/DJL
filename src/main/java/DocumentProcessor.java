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
import java.util.List;
import java.util.ArrayList;

public class DocumentProcessor {
    private final int chunkSize = 500;
    private final int chunkOverlap = 50;
    private final LocalONNXEmbeddingModel embeddingModel;
    private InMemoryEmbeddingStore<TextSegment> vectorStore;

    public DocumentProcessor() throws Exception {
        this.embeddingModel = new LocalONNXEmbeddingModel(
                "/home/acer/IdeaProjects/DJI/src/main/resources/allmpnetbasev2"
        );
        this.vectorStore = null;
    }

    public Document loadDocument(String filePath) {
        if (filePath.endsWith(".txt")) {
            return FileSystemDocumentLoader.loadDocument(filePath, new TextDocumentParser());
        } else {
            throw new IllegalArgumentException("Unsupported file format");
        }
    }

    public void processDocument(String filePath) {
        // Load the document
        Document doc = loadDocument(filePath);

        // Split the document into chunks
        DocumentByParagraphSplitter splitter = new DocumentByParagraphSplitter(chunkSize, chunkOverlap);
        List<TextSegment> segments = splitter.split(doc);

        // Generate embeddings for all chunks
        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : segments) {
            Embedding emb = embeddingModel.embed(segment.text());
            embeddings.add(emb);
        }

        // Create or update the vector store
        if (vectorStore == null) {
            vectorStore = new InMemoryEmbeddingStore<>();
        }

        // Add all embeddings and chunks to the store
        vectorStore.addAll(embeddings, segments);
    }

    public List<TextSegment> retrieveRelevantContext(String query, int k) {
        if (vectorStore == null) {
            return List.of();
        }

        // Generate embedding for the query
        Embedding queryEmbedding = embeddingModel.embed(query);

        // Create a search request
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(k)
                .build();

        // Perform the search
        EmbeddingSearchResult<TextSegment> searchResult = vectorStore.search(searchRequest);

        // Extract the segments from the matches
        List<TextSegment> relevantSegments = new ArrayList<>();
        for (EmbeddingMatch<TextSegment> match : searchResult.matches()) {
            relevantSegments.add(match.embedded());
        }

        return relevantSegments;
    }

    // TODO: Implement the method to clear the vector store

    public void resetStore() {
        this.vectorStore = null;
    }

    public void closeModel() throws Exception {
        this.embeddingModel.close();
    }

}