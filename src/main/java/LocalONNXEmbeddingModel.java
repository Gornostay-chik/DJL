import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class LocalONNXEmbeddingModel {

    private final ZooModel<String, float[]> model;
    private final Predictor<String, float[]> predictor;

    /**
     * Конструктор принимает путь к директории, где лежит ONNX-модель и файлы токенизатора.
     * Файл модели должен называться "allMiniLML6v2.onnx".
     *
     * @param modelDir абсолютный путь к директории с моделью и токенизатором.
     * @throws Exception если происходит ошибка загрузки модели.
     */
    public LocalONNXEmbeddingModel(String modelDir) throws Exception {
        String localModelUrl = "file://" + modelDir;
        Translator<String, float[]> translator = new ONNXTranslator(modelDir);
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optEngine("OnnxRuntime")
                .optModelUrls(localModelUrl)
                .optModelName("model.onnx")  // Проверяем, что именно этот файл присутствует
                .optTranslator(translator)
                .build();
        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    public Embedding embed(String text) {
        float[] vector;
        try {
            vector = predictor.predict(text);
        } catch (TranslateException e) {
            throw new RuntimeException("Ошибка при предсказании эмбеддинга", e);
        }
        return new Embedding(vector);
    }

    public List<Embedding> embedAll(List<TextSegment> segments) {
        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : segments) {
            embeddings.add(embed(segment.text()));
        }
        return embeddings;
    }

    public void close() throws Exception {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }

    /**
     * Переводчик для ONNX-модели, который формирует входы:
     * input_ids, attention_mask и token_type_ids,
     * а затем выполняет pooling по токенам для получения фиксированного вектора.
     */
    private static class ONNXTranslator implements Translator<String, float[]> {
        private final HuggingFaceTokenizer tokenizer;

        public ONNXTranslator(String tokenizerDir) throws Exception {
            // Инициализируем токенизатор из указанной директории
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir));
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            NDManager manager = ctx.getNDManager();
            // Токенизируем входной текст
            Encoding encoding = tokenizer.encode(input);

            long[] tokenIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();
            long[] tokenTypeIds = encoding.getTypeIds();

            // Если tokenTypeIds отсутствуют, создаем массив нулей той же длины
            if (tokenTypeIds == null || tokenTypeIds.length == 0) {
                tokenTypeIds = new long[tokenIds.length];
            }

            // Создаем NDArrays без дополнительного развертывания измерений
            NDArray idsArray = manager.create(tokenIds);
            NDArray maskArray = manager.create(attentionMask);
            NDArray tokenTypeArray = manager.create(tokenTypeIds);

            // Порядок входов: [input_ids, attention_mask, token_type_ids]
            return new NDList(idsArray, maskArray, tokenTypeArray);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            // Если выход имеет форму [batch, seq_length, hidden_dim], усредняем по оси токенов
            if (output.getShape().dimension() == 3) {
                output = output.mean(new int[]{1}); // усредняем по оси seq_length
                output = output.squeeze(0);           // удаляем размер батча, если он равен 1
            } else if (output.getShape().dimension() == 2) {
                // Если выхода [seq_length, hidden_dim], усредняем по оси 0
                output = output.mean(new int[]{0});
            }
            return output.toFloatArray();
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}
