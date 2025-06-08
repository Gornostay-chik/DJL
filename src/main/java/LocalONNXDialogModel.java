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

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class LocalONNXDialogModel {

    private ZooModel<String, String> model;
    private Predictor<String, String> predictor;

    /**
     * Переводчик для диалоговой ONNX-модели.
     * Он принимает строковый вход (диалоговый запрос или контекст)
     * и возвращает строку – ответ модели.
     *
     * Процесс включает:
     * 1. Токенизацию входного текста с помощью HuggingFaceTokenizer.
     * 2. Формирование необходимых NDArrays для входа модели.
     * 3. Постобработку выходного NDArray через декодирование токенов в текст.
     */
    private static class ONNXDialogTranslator implements Translator<String, String> {

        private final HuggingFaceTokenizer tokenizer;

        public ONNXDialogTranslator(String tokenizerDir) throws Exception {
            // Загружаем токенизатор из указанной директории
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir));
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            NDManager manager = ctx.getNDManager();
            // Токенизируем входной диалоговый запрос
            Encoding encoding = tokenizer.encode(input);

            long[] tokenIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();

            // Создаем NDArrays и добавляем измерение батча, чтобы получить форму [1, seq_length]
            NDArray idsArray = manager.create(tokenIds).expandDims(0);
            NDArray maskArray = manager.create(attentionMask).expandDims(0);

            // Возвращаем только два входа, т.к. модель ожидает: [input_ids, attention_mask]
            return new NDList(idsArray, maskArray);
        }



        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            // Предполагаем, что модель возвращает logits с размерностью [batch, seq_length, vocab_size]
            NDArray output = list.get(0);

            // Если выход имеет 3 измерения, применяем argmax по оси vocab_size
            if (output.getShape().dimension() == 3) {
                // Выбираем токен с максимальной вероятностью для каждой позиции
                output = output.argMax(2);
                // Удаляем измерение батча, если оно равно 1 (например, переходим от [1, seq_length] к [seq_length])
                output = output.squeeze(0);
            } else if (output.getShape().dimension() == 2) {
                // Если выход имеет форму [batch, seq_length] — удаляем размер батча
                output = output.squeeze(0);
            }

            // Приводим полученные индексы к типу long, если это необходимо
            NDArray tokenIds = output.toType(ai.djl.ndarray.types.DataType.INT64, false);

            // Преобразуем в массив токенов и декодируем их через токенизатор
            long[] outputTokenIds = tokenIds.toLongArray();
            return tokenizer.decode(outputTokenIds, true);
        }





        @Override
        public Batchifier getBatchifier() {
            // Для диалоговых моделей часто предпочтительнее обрабатывать по одному запросу
            return null;
        }
    }

    /**
     * Конструктор, принимающий абсолютный путь к директории,
     * в которой расположены файлы ONNX-модели и токенизатора.
     *
     * @param modelDir абсолютный путь к директории с моделью и токенизатором.
     * @throws Exception в случае проблем при загрузке модели.
     */
    public LocalONNXDialogModel(String modelDir) throws Exception {
        String localModelUrl = "file://" + modelDir;
        Translator<String, String> translator = new ONNXDialogTranslator(modelDir);
        // Имя файла модели может отличаться.
        // Замените "dialog_model.onnx" на имя файла вашей модели.
        Criteria<String, String> criteria = Criteria.builder()
                .setTypes(String.class, String.class)
                .optEngine("OnnxRuntime")
                .optModelUrls(localModelUrl)
                .optModelName("dialog_model.onnx")
                .optTranslator(translator)
                .build();
        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    /**
     * Метод для получения ответа от диалоговой модели.
     *
     * @param prompt входной диалоговый запрос или контекст.
     * @return сгенерированный ответ модели.
     */
    public String chat(String prompt) {
        try {
            return predictor.predict(prompt);
        } catch (TranslateException e) {
            throw new RuntimeException("Ошибка при генерации ответа диалога", e);
        }
    }

    /**
     * Освобождает ресурсы, используемые моделью и предиктором.
     *
     * @throws Exception в случае ошибок при закрытии модели.
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
