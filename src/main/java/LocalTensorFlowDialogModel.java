import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;

import java.nio.file.Path;
import java.nio.file.Paths;

public class LocalTensorFlowDialogModel {

    private ZooModel<String, String> model;
    private Predictor<String, String> predictor;

    /**
     * Переводчик для диалоговой модели.
     * Он токенизирует входной текст, формирует входные NDArrays и после инференса декодирует выход в строку.
     */
    private static class TensorFlowDialogTranslator implements Translator<String, String> {
        private final HuggingFaceTokenizer tokenizer;

        public TensorFlowDialogTranslator(String tokenizerDir) throws Exception {
            // Предполагается, что в tokenizerDir находится полностью сохранённый tokenizer.json
            Path tokenizerDirectory = Paths.get(tokenizerDir);
            this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizerDirectory);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            NDManager manager = ctx.getNDManager();
            Encoding encoding = tokenizer.encode(input);
            long[] tokenIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();
            long[] tokenTypeIds = encoding.getTypeIds();
            if (tokenTypeIds == null || tokenTypeIds.length == 0) {
                tokenTypeIds = new long[tokenIds.length];
            }
            NDArray idsArray = manager.create(tokenIds);
            NDArray maskArray = manager.create(attentionMask);
            NDArray tokenTypeArray = manager.create(tokenTypeIds);
            return new NDList(idsArray, maskArray, tokenTypeArray);
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            long[] outputTokenIds = output.toLongArray();
            return tokenizer.decode(outputTokenIds, true);
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }

    /**
     * Конструктор принимает путь к директории, где лежат:
     *  – Директория SavedModel TensorFlow (например, с файлом saved_model.pb)
     *  – Файлы токенизатора (tokenizer.json, vocab.json, merges.txt, и т.д.)
     *
     * @param modelDir Абсолютный путь к директории с моделью и токенизатором.
     * @throws Exception в случае ошибки загрузки.
     */
    public LocalTensorFlowDialogModel(String modelDir) throws Exception {
        // Здесь modelDir указывает на директорию, которая содержит директорию SavedModel. Например, если SavedModel лежит в папке saved_tf_model,
        // то укажите modelDir как "/path/to/model/directory/saved_tf_model"
        String localModelUrl = "file://" + modelDir;
        Translator<String, String> translator = new TensorFlowDialogTranslator(modelDir);
        Criteria<String, String> criteria = Criteria.builder()
                .setTypes(String.class, String.class)
                .optEngine("TensorFlow")
                .optModelUrls(localModelUrl)
                .optModelName("")  // При использовании SavedModel можно не задавать конкретное имя модели.
                .optTranslator(translator)
                .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    /**
     * Генерирует ответ по заданному тексту (prompt).
     *
     * @param prompt Входной текст (контекст диалога).
     * @return Сгенерированный ответ.
     */
    public String generateResponse(String prompt) {
        try {
            return predictor.predict(prompt);
        } catch (TranslateException e) {
            throw new RuntimeException("Ошибка при генерации ответа", e);
        }
    }

    /**
     * Закрывает ресурсы модели и предиктора.
     *
     * @throws Exception при возникновении ошибок закрытия.
     */
    public void close() throws Exception {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }
}
