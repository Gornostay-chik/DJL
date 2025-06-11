import ai.djl.Model;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;

import java.nio.file.Paths;
import java.util.*;

public class LocalONNXSmolLM2Model {

    // Для модели SmolLM2‑135M‑Instruct допустимые индексы лежат в диапазоне [0, 49151].
    // Поэтому вместо GPT‑2 значения (50256) используем 0 для начала и конца генерации.
    private static final long BOS_ID = 0;
    private static final long EOS_ID = 0;

    // Поля: модель, предиктор и наш Translator
    private final ZooModel<List<Long>, List<Long>> model;
    private final Predictor<List<Long>, List<Long>> predictor;
    private final SmolLM2ONNXTranslator translator;

    /**
     * Конструктор.
     *
     * @param modelDir Директория, где находятся ONNX‑модель (например, "model.onnx") и файлы токенизатора.
     */
    public LocalONNXSmolLM2Model(String modelDir) throws Exception {
        String localModelUrl = "file://" + modelDir;
        translator = new SmolLM2ONNXTranslator(modelDir);
        Criteria<List<Long>, List<Long>> criteria = Criteria.builder()
                .setTypes((Class<List<Long>>) (Class<?>) List.class, (Class<List<Long>>) (Class<?>) List.class)
                .optEngine("OnnxRuntime")
                .optModelUrls(localModelUrl)
                .optModelName("model.onnx") // Убедитесь, что имя файла модели совпадает
                .optTranslator(translator)
                .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    /**
     * Автогрессивная генерация текста. Принимает входной prompt и добавляет до maxNewTokens токенов.
     */
    public String generate(String prompt, int maxNewTokens) throws TranslateException {
        NDManager localManager = NDManager.newBaseManager();
        SimpleTranslatorContext simpleCtx = new SimpleTranslatorContext(localManager);

        // Токенизация prompt. Если он не начинается с нужного маркера, добавляем BOS.
        List<Long> promptTokens = new ArrayList<>();
        if (!prompt.startsWith("<s>")) {
            promptTokens.add(BOS_ID);
        }
        Encoding encoding = translator.tokenizer.encode(prompt);
        for (long id : encoding.getIds()) {
            promptTokens.add(id);
        }
        // Начинаем генерацию с токенов prompt-а.
        List<Long> generatedTokens = new ArrayList<>(promptTokens);

        // Итеративная генерация: добавляем токены до maxNewTokens или до встречи EOS.
        for (int i = 0; i < maxNewTokens; i++) {
            List<Long> outputTokens = predictor.predict(generatedTokens);
            long nextToken = outputTokens.get(0);
            if (nextToken == EOS_ID) {
                break;
            }
            generatedTokens.add(nextToken);
        }

        long[] resultArray = generatedTokens.stream().mapToLong(Long::longValue).toArray();
        String result = translator.tokenizer.decode(resultArray, true);
        localManager.close();
        return result;
    }

    /**
     * Метод chat – обёртка для generate с фиксированным числом генерируемых токенов (например, 50).
     */
    public String chat(String prompt) throws TranslateException {
        return generate(prompt, 50);
    }

    /**
     * Метод chat для переписки с историей сообщений.
     * Собирает prompt из истории, генерирует ответ и очищает его от повторов.
     */
    public ChatResponse chat(List<ChatMessage> history) throws TranslateException {
        StringBuilder promptBuilder = new StringBuilder();
        for (ChatMessage msg : history) {
            if (msg instanceof SystemMessage) {
                continue;
            } else if (msg instanceof UserMessage) {
                promptBuilder.append("User: ")
                        .append(((UserMessage) msg).contents())
                        .append("\n");
            } else if (msg instanceof AiMessage) {
                promptBuilder.append("Assistant: ")
                        .append(((AiMessage) msg).text())
                        .append("\n");
            }
        }
        promptBuilder.append("Assistant:");

        String rawOutput = generate(promptBuilder.toString(), 50);
        String answer = rawOutput.substring(rawOutput.indexOf("Assistant:") + "Assistant:".length()).trim();

        return ChatResponse.builder()
                .aiMessage(new AiMessage(answer))
                .build();
    }

    /**
     * Освобождает ресурсы модели и предиктора.
     */
    public void close() throws Exception {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }

    /**
     * Простейшая реализация TranslatorContext.
     */
    private static class SimpleTranslatorContext implements TranslatorContext {

        private final NDManager manager;

        public SimpleTranslatorContext(NDManager manager) {
            this.manager = manager;
        }

        @Override
        public NDManager getNDManager() {
            return manager;
        }

        @Override
        public Model getModel() {
            return null;
        }

        @Override
        public NDManager getPredictorManager() {
            return manager;
        }

        @Override
        public Block getBlock() {
            return null;
        }

        @Override
        public Metrics getMetrics() {
            return null;
        }

        @Override
        public Object getAttachment(String key) {
            return null;
        }

        @Override
        public void setAttachment(String key, Object value) {
            // Пустая реализация
        }

        @Override
        public void close() {
            // Закрытие ресурсов, если потребуется
        }
    }

    /**
     * Реализация Translator для SmolLM2‑135M‑Instruct в ONNX‑формате.
     * Вход и выход – списки токенов (List<Long>).
     */
    public static class SmolLM2ONNXTranslator implements Translator<List<Long>, List<Long>> {

        public final HuggingFaceTokenizer tokenizer;

        /**
         * Конструктор.
         *
         * @param tokenizerDir Путь к директории с файлами токенизатора.
         * @throws Exception При ошибках загрузки токенизатора.
         */
        public SmolLM2ONNXTranslator(String tokenizerDir) throws Exception {
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir));
        }

        /**
         * Формирует NDList входов модели.
         * Ожидаются входы: input_ids, attention_mask, position_ids и для каждого слоя past_key_values.
         */
        @Override
        public NDList processInput(TranslatorContext ctx, List<Long> inputTokens) {
            NDManager manager = ctx.getNDManager();

            // 1. input_ids: [1, seq_len]
            long[] tokenIds = inputTokens.stream().mapToLong(Long::longValue).toArray();
            NDArray inputIds = manager.create(tokenIds).expandDims(0);

            // 2. attention_mask: [1, seq_len] (единицы для всех токенов)
            long[] attentionMaskArr = new long[tokenIds.length];
            Arrays.fill(attentionMaskArr, 1);
            NDArray attentionMask = manager.create(attentionMaskArr).expandDims(0);

            // 3. position_ids: создаём массив [0, 1, 2, …, seq_len - 1]
            long[] positions = new long[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) {
                positions[i] = i;
            }
            NDArray positionIds = manager.create(positions).expandDims(0);

            // Формируем NDList входов
            NDList ndList = new NDList();
            ndList.add(inputIds);       // 0-й вход
            ndList.add(attentionMask);    // 1-й вход
            ndList.add(positionIds);      // 2-й вход

            // 4. Добавляем пустые тензоры для past_key_values для каждого из NUM_LAYERS.
            // Обратите внимание: модель SmolLM2 ожидает число attention-голов равное 3.
            final int NUM_LAYERS = 30;     // При необходимости скорректируйте
            final int NUM_HEADS = 3;       // Число attention‑голов = 3
            final int HEAD_DIM = 64;       // Убедитесь, что HEAD_DIM соответствует конфигурации модели

            // Для начального шага длина кэша равна 0, поэтому форма: [1, NUM_HEADS, 0, HEAD_DIM]
            long[] pastShape = {1, NUM_HEADS, 0, HEAD_DIM};

            for (int i = 0; i < NUM_LAYERS; i++) {
                NDArray pastKey = manager.create(new float[0], new Shape(pastShape));
                NDArray pastValue = manager.create(new float[0], new Shape(pastShape));
                ndList.add(pastKey);
                ndList.add(pastValue);
            }
            return ndList;
        }

        /**
         * Извлекает логиты, выбирает последний логит и сэмплирует следующий токен
         * с использованием nucleus (top‑p) сэмплирования, температуры 0.2 и top‑p 0.9.
         */
        @Override
        public List<Long> processOutput(TranslatorContext ctx, NDList list) {
            // logits имеет форму [1, seq_length, vocab_size]
            NDArray logits = list.get(0);
            long seqLength = logits.getShape().get(1);
            NDArray lastLogits = logits.get(0).get((int) (seqLength - 1));
            int nextToken = sampleFromLogits(lastLogits, 0.2f, 0.9f);
            List<Long> result = new ArrayList<>();
            result.add((long) nextToken);
            return result;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        /**
         * Реализация nucleus (top‑p) сэмплирования.
         *
         * @param logits      1D NDArray логитов для текущего шага.
         * @param temperature Температура для масштабирования логитов.
         * @param topP        Порог nucleus фильтрации (например, 0.9 для накопления 90% вероятности).
         * @return Выбранный индекс токена.
         */
        private int sampleFromLogits(NDArray logits, float temperature, float topP) {
            NDArray scaled = logits.div(temperature);
            NDArray probs = scaled.softmax(0);
            float[] probArray = probs.toFloatArray();
            int vocabSize = probArray.length;

            // Создаем массив индексов для сортировки по убыванию вероятности
            Integer[] indices = new Integer[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                indices[i] = i;
            }
            Arrays.sort(indices, (a, b) -> Float.compare(probArray[b], probArray[a]));

            // Собираем минимальное множество токенов, суммарная вероятность которых >= topP
            float cumulative = 0f;
            boolean[] keep = new boolean[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                int idx = indices[i];
                cumulative += probArray[idx];
                keep[idx] = true;
                if (cumulative >= topP) {
                    break;
                }
            }

            // Фильтруем распределение вероятностей
            float filteredSum = 0f;
            float[] filteredProb = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                if (keep[i]) {
                    filteredProb[i] = probArray[i];
                    filteredSum += filteredProb[i];
                } else {
                    filteredProb[i] = 0f;
                }
            }
            if (filteredSum == 0f) {
                filteredProb = probArray;
                filteredSum = 0f;
                for (float p : filteredProb) {
                    filteredSum += p;
                }
            }
            float rnd = new Random().nextFloat() * filteredSum;
            float cumulativeProbability = 0f;
            for (int i = 0; i < vocabSize; i++) {
                cumulativeProbability += filteredProb[i];
                if (cumulativeProbability >= rnd) {
                    return i;
                }
            }
            return vocabSize - 1;
        }
    }
}
