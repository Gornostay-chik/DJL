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

public class LocalONNXLlamaModel {

    // Пример специальных токенов: BOS и EOS (настройте по необходимости)
    private static final long BOS_ID = 1;
    private static final long EOS_ID = 2;
    // Поля модели и предиктора
    private final ZooModel<List<Long>, List<Long>> model;
    private final Predictor<List<Long>, List<Long>> predictor;
    private final ONNXLlamaTranslator translator;

    /**
     * Конструктор. modelDir — директория, содержащая ONNX-модель и файлы токенизатора.
     */
    public LocalONNXLlamaModel(String modelDir) throws Exception {
        String localModelUrl = "file://" + modelDir;
        translator = new ONNXLlamaTranslator(modelDir);
        Criteria<List<Long>, List<Long>> criteria = Criteria.builder()
                .setTypes((Class<List<Long>>) (Class<?>) List.class, (Class<List<Long>>) (Class<?>) List.class)
                .optEngine("OnnxRuntime")
                .optModelUrls(localModelUrl)
                .optModelName("model_quantized.onnx")
                // При необходимости укажите опцию для расширения ONNX:
                // .optOption("customOpLibrary", "/path/to/onnxruntime_extensions.so")
                .optTranslator(translator)
                .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    /**
     * Генерирует автогрессивно текст, начиная с начального запроса (prompt)
     * с ограничением на количество генерируемых токенов (maxNewTokens).
     */
    public String generate(String prompt, int maxNewTokens) throws TranslateException {
        NDManager localManager = NDManager.newBaseManager();
        SimpleTranslatorContext simpleCtx = new SimpleTranslatorContext(localManager);

        // Токенизация prompt-а. Если строка не начинается с символа начала (<s>), добавляем BOS.
        List<Long> promptTokens = new ArrayList<>();
        if (!prompt.startsWith("<s>")) {
            promptTokens.add(BOS_ID);
        }
        Encoding encoding = translator.tokenizer.encode(prompt);
        for (long id : encoding.getIds()) {
            promptTokens.add(id);
        }
        List<Long> generatedTokens = new ArrayList<>(promptTokens);

        // Итеративная автогрессивная генерация токенов.
        for (int i = 0; i < maxNewTokens; i++) {
            List<Long> outputTokens = predictor.predict(generatedTokens);
            long nextToken = outputTokens.get(0); // processOutput возвращает один токен для следующего шага
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

    public ChatResponse chat(List<ChatMessage> history) throws TranslateException {
        // 1) Собираем текст промпта из истории
        StringBuilder promptBuilder = new StringBuilder();
        for (ChatMessage msg : history) {
            if (msg instanceof SystemMessage) {
                // системные месседжи можно не включать в генерируемый текст,
                // а держать отдельно, если чат-подсистема их подтягивает сама.
                // promptBuilder.append("[System] ")
                //              .append(msg.text())
                //              .append("\n");
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
        // Сигнализируем модели, что дальше пойдёт ответ ассистента
        promptBuilder.append("Assistant:");

        // 2) Генерируем
        String rawOutput = generate(promptBuilder.toString(), /*maxNewTokens*/100);

        // 3) “Чистим” – вырезаем обратно промт (если модель его повторила),
        //    оставляем только сгенерированный ассистентом текст
        String answer = rawOutput
                .substring(rawOutput.indexOf("Assistant:") + "Assistant:".length())
                .trim();

        // 4) Упаковываем в ChatResponse (пример, API билдера может отличаться)
        return ChatResponse.builder()
                .aiMessage(new AiMessage(answer))
                // если нужен full history, то можно передать и его:
                //.conversationHistory(updatedHistory)
                .build();
    }

    /**
     * Обёртка для generate с фиксированным числом генерируемых токенов (например, 50).
     */
    public String chat(String prompt) throws TranslateException {
        return generate(prompt, 50);
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
     * Простейшая реализация TranslatorContext, удовлетворяющая контракту интерфейса.
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

        // Возвращаем тот же NDManager в качестве менеджера предиктора
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
            // Если необходимо, можно закрыть ресурсы
        }
    }

    /**
     * Реализация Translator для модели ONNX Llama.
     * Вход и выход — списки токенов (List<Long>).
     */
    public static class ONNXLlamaTranslator implements Translator<List<Long>, List<Long>> {

        public final HuggingFaceTokenizer tokenizer;
        // Параметры модели (настроить согласно конфигурации модели)
        private final int numLayers = 16;
        private final long numHeads = 8;
        private final long headDim = 64;

        /**
         * Конструктор. tokenizerDir — путь к файлам токенизатора.
         */
        public ONNXLlamaTranslator(String tokenizerDir) throws Exception {
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir));
        }

        /**
         * Формирует NDList входов модели из списка токенов.
         */
        @Override
        public NDList processInput(TranslatorContext ctx, List<Long> inputTokens) {
            NDManager manager = ctx.getNDManager();
            long[] tokenIds = inputTokens.stream().mapToLong(Long::longValue).toArray();
            NDArray inputIds = manager.create(tokenIds).expandDims(0);

            // attention_mask: единицы для каждого токена
            long[] attentionMaskArr = new long[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) {
                attentionMaskArr[i] = 1;
            }
            NDArray attentionMask = manager.create(attentionMaskArr).expandDims(0);

            // position_ids: последовательность от 0 до (seq_length - 1)
            long[] positionIdsArr = new long[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) {
                positionIdsArr[i] = i;
            }
            NDArray positionIds = manager.create(positionIdsArr).expandDims(0);

            NDList ndList = new NDList();
            ndList.add(inputIds);       // input_ids
            ndList.add(attentionMask);    // attention_mask
            ndList.add(positionIds);      // position_ids

            // Добавляем пустые past_key_values для каждого слоя (если модель их ожидает).
            Shape emptyShape = new Shape(1, numHeads, 0, headDim);
            for (int i = 0; i < numLayers; i++) {
                NDArray pastKey = manager.create(new float[0]).reshape(emptyShape);
                NDArray pastValue = manager.create(new float[0]).reshape(emptyShape);
                ndList.add(pastKey);
                ndList.add(pastValue);
            }
            return ndList;
        }

        /**
         * Извлекает логиты, берет последний логит и выполняет сэмплирование следующего токена,
         * используя температуру и top‑p (nucleus) фильтрацию.
         */
        @Override
        public List<Long> processOutput(TranslatorContext ctx, NDList list) {
            // logits имеет форму [1, seq_length, vocab_size]
            NDArray logits = list.get(0);
            long seqLength = logits.getShape().get(1);
            NDArray lastLogits = logits.get(0).get((int) (seqLength - 1));
            // Задаем температуру 0.5 и topP 0.9
            int nextToken = sampleFromLogits(lastLogits, 0.3f, 0.9f);
            List<Long> result = new ArrayList<>();
            result.add((long) nextToken);
            return result;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        /**
         * Реализует сэмплирование с nucleus (top‑p) фильтрацией.
         *
         * @param logits      логиты для текущего токена (1D-массив)
         * @param temperature коэффициент температуры для масштабирования логитов
         * @param topP        порог nucleus фильтрации (например, 0.9 для 90% вероятности)
         * @return индекс выбранного токена
         */
        private int sampleFromLogits(NDArray logits, float temperature, float topP) {
            // Масштабирование логитов: делим на температуру
            NDArray scaled = logits.div(temperature);
            NDArray probs = scaled.softmax(0);
            float[] probArray = probs.toFloatArray();
            int vocabSize = probArray.length;

            // Создаем массив индексов
            Integer[] indices = new Integer[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                indices[i] = i;
            }
            // Сортируем индексы по убыванию вероятностей
            Arrays.sort(indices, new Comparator<Integer>() {
                @Override
                public int compare(Integer a, Integer b) {
                    return Float.compare(probArray[b], probArray[a]);
                }
            });

            // Определяем nucleus: минимальный набор токенов, суммарная вероятность которых >= topP
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

            // Формируем новое распределение вероятностей: оставляем только те токены, которые попали в nucleus
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
            // Если по неизвестной причине все вероятности обнулились, используем исходное распределение
            if (filteredSum == 0) {
                filteredProb = probArray;
                filteredSum = 0f;
                for (float p : filteredProb) {
                    filteredSum += p;
                }
            }

            // Сэмплируем токен на основе отфильтрованного распределения
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
