import ai.djl.Model;
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
import ai.djl.metric.Metrics;
import ai.djl.nn.Block;

import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.SystemMessage;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;
import java.util.Comparator;

public class LocalGPT2ONNXModel {

    // Поля: модель, предиктор и наш Translator
    private ZooModel<List<Long>, List<Long>> model;
    private Predictor<List<Long>, List<Long>> predictor;
    private GPT2ONNXTranslator translator;

    // Для GPT‑2 обычно используется один и тот же токен для завершения генерации.
    // В оригинальной GPT‑2 eos_token_id = 50256.
    private static final long BOS_ID = 50256;
    private static final long EOS_ID = 50256;

    /**
     * Конструктор.
     * @param modelDir Директория, где находятся ONNX‑модель и файлы токенизатора.
     *                 Например, в ней должен быть файл модели (например, "model_quantized.onnx")
     *                 и файлы токенизатора (например, tokenizer.json).
     */
    public LocalGPT2ONNXModel(String modelDir) throws Exception {
        String localModelUrl = "file://" + modelDir;
        translator = new GPT2ONNXTranslator(modelDir);
        Criteria<List<Long>, List<Long>> criteria = Criteria.builder()
                .setTypes((Class<List<Long>>)(Class<?>)List.class, (Class<List<Long>>)(Class<?>)List.class)
                .optEngine("OnnxRuntime")
                .optModelUrls(localModelUrl)
                // Имя файла модели – при необходимости замените на актуальное (например, "model_quantized.onnx")
                .optModelName("model.onnx")
                .optTranslator(translator)
                .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    /**
     * Автогрессивная генерация текста. Метод принимает prompt и добавляет до maxNewTokens новых токенов.
     */
    public String generate(String prompt, int maxNewTokens) throws TranslateException {
        NDManager localManager = NDManager.newBaseManager();
        SimpleTranslatorContext simpleCtx = new SimpleTranslatorContext(localManager);

        // Токенизируем prompt. Если он не содержит специального маркера,
        // добавляем BOS (в GPT‑2 часто используется тот же токен, что и EOS – 50256).
        List<Long> promptTokens = new ArrayList<>();
        if (!prompt.startsWith("<s>")) {  // Измените условие, если требуется иной формат
            promptTokens.add(BOS_ID);
        }
        Encoding encoding = translator.tokenizer.encode(prompt);
        for (long id : encoding.getIds()) {
            promptTokens.add(id);
        }
        // Начинаем генерацию с токенов prompt-а.
        List<Long> generatedTokens = new ArrayList<>(promptTokens);

        // Итеративно генерируем новые токены.
        for (int i = 0; i < maxNewTokens; i++) {
            List<Long> outputTokens = predictor.predict(generatedTokens);
            long nextToken = outputTokens.get(0); // processOutput возвращает один токен для шага
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
            }
            else if (msg instanceof UserMessage) {
                promptBuilder.append("User: ")
                        .append(((UserMessage) msg).contents())
                        .append("\n");
            }
            else if (msg instanceof AiMessage) {
                promptBuilder.append("Assistant: ")
                        .append(((AiMessage) msg).text())
                        .append("\n");
            }
        }
        // Сигнализируем модели, что дальше пойдёт ответ ассистента
        promptBuilder.append("Assistant:");

        // 2) Генерируем
        String rawOutput = generate(promptBuilder.toString(), /*maxNewTokens*/50);

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

        private NDManager manager;

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

        // Возвращаем тот же менеджер в качестве predictor manager
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
            // Закрываем ресурсы, если требуется
        }
    }

    /**
     * Реализация Translator для GPT‑2 в ONNX‑формате.
     * Вход и выход – списки токенов (List<Long>).
     */
    public static class GPT2ONNXTranslator implements Translator<List<Long>, List<Long>> {

        public final HuggingFaceTokenizer tokenizer;

        /**
         * Конструктор.
         * @param tokenizerDir Путь к файлам токенизатора.
         * @throws Exception при ошибках загрузки токенизатора.
         */
        public GPT2ONNXTranslator(String tokenizerDir) throws Exception {
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir));
        }

        /**
         * Формирует NDList входов модели.
         * В ONNX‑версии ожидаются только два входа: input_ids и attention_mask.
         */
        @Override
        public NDList processInput(TranslatorContext ctx, List<Long> inputTokens) {
            NDManager manager = ctx.getNDManager();
            long[] tokenIds = inputTokens.stream().mapToLong(Long::longValue).toArray();
            NDArray inputIds = manager.create(tokenIds).expandDims(0);

            // Формируем attention_mask: массив единиц того же размера, что и input_ids.
            long[] attentionMaskArr = new long[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) {
                attentionMaskArr[i] = 1;
            }
            NDArray attentionMask = manager.create(attentionMaskArr).expandDims(0);

            NDList ndList = new NDList();
            ndList.add(inputIds);       // Ключ: "input_ids"
            ndList.add(attentionMask);    // Ключ: "attention_mask"
            return ndList;
        }

        /**
         * Извлекает логиты, принимает последний логит и сэмплирует следующий токен,
         * используя nucleus (top‑p) сэмплирование с температурой 0.5 и top‑p 0.9.
         */
        @Override
        public List<Long> processOutput(TranslatorContext ctx, NDList list) {
            // logits имеет форму [1, seq_length, vocab_size]
            NDArray logits = list.get(0);
            long seqLength = logits.getShape().get(1);
            NDArray lastLogits = logits.get(0).get((int)(seqLength - 1));
            int nextToken = sampleFromLogits(lastLogits, 0.5f, 0.9f);
            List<Long> result = new ArrayList<>();
            result.add((long) nextToken);
            return result;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        /**
         * Реализует nucleus (top‑p) сэмплирование.
         *
         * @param logits      1D NDArray логитов для текущего шага.
         * @param temperature Коэффициент температуры (чем ниже – тем детерминированнее).
         * @param topP        Порог nucleus фильтрации (например, 0.9 для накопления 90% вероятности).
         * @return Индекс выбранного токена.
         */
        private int sampleFromLogits(NDArray logits, float temperature, float topP) {
            NDArray scaled = logits.div(temperature);
            NDArray probs = scaled.softmax(0);
            float[] probArray = probs.toFloatArray();
            int vocabSize = probArray.length;

            // Создаем массив индексов для сортировки по убыванию вероятности.
            Integer[] indices = new Integer[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                indices[i] = i;
            }
            Arrays.sort(indices, new Comparator<Integer>() {
                @Override
                public int compare(Integer a, Integer b) {
                    return Float.compare(probArray[b], probArray[a]);
                }
            });

            // Определяем nucleus: собираем минимальное множество токенов, суммарная вероятность которых >= topP.
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

            // Формируем фильтрованное распределение вероятностей.
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
                // Если что-то пошло не так, используем исходное распределение.
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
