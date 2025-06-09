import dev.langchain4j.data.segment.TextSegment;
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
        String query = "Who is Holmes?";

        // Take 3 relevant segments for context
        List<TextSegment> relevantSegments = processor.retrieveRelevantContext(query, 3);

        String context = relevantSegments.toString();

        // Создаем подсказку (prompt) с учетом контекста и вопроса
// Пример изменённого шаблона prompt-а:
// Пример изменённого шаблона prompt-а:
        String promptTemplate = "Answer the following question based solely on the provided context.\n\n"
                + "Context:\n%s\n\n"
                + "Question: %s\n\n"
                + "Final Answer:";

        String prompt = String.format(promptTemplate, context, query);

        System.out.println("Сформированная подсказка:");
        System.out.println(prompt);
        System.out.println("\n---\n");

// Создаем объект модели (у вас LocalGPT2ONNXModel)
        // Создаем объект диалоговой модели и генерируем ответ
        //    String tfModelDir = "/home/acer/IdeaProjects/DJI/src/main/resources/Llama-32-1B";
        //    LocalONNXLlamaModel dialogModel = new LocalONNXLlamaModel(tfModelDir);

        String tfModelDir = "/home/acer/IdeaProjects/DJI/src/main/resources/GPT2";
        LocalGPT2ONNXModel dialogModel = new LocalGPT2ONNXModel(tfModelDir);

// Генерируем полный текст
        String fullOutput = dialogModel.chat(prompt);

// Постобработка: удаляем все, что до маркера "Final Answer:"
        String answer;
        int markerIndex = fullOutput.indexOf("Final Answer:");
        if (markerIndex != -1) {
            // берем часть после маркера
            answer = fullOutput.substring(markerIndex + "Final Answer:".length()).trim();
        } else {
            // Если маркер не найден, можно использовать полный вывод или применить другую логику
            answer = fullOutput.trim();
        }

        System.out.println("Сгенерированный ответ:");
        System.out.println(answer);








        // Закрываем ресурсы моделей
        dialogModel.close();
        processor.closeModel();
        processor.resetStore();

    }
}
