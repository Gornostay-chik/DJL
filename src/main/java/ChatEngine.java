import ai.djl.translate.TranslateException;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;

import java.util.ArrayList;
import java.util.List;


public class ChatEngine {
    // private final LocalONNXGPT2Model chatModel;
    //private LocalONNXLlamaModel chatModel;
    private final LocalONNXSmolLM2Model chatModel;
    private final String systemMessage;
    private final List<ChatMessage> conversationHistory;
    private final String promptTemplate;
    private final String tfModelDirGPT2 = "/home/acer/IdeaProjects/DJI/src/main/resources/GPT2";
    private final String tfModelDirLLama = "/home/acer/IdeaProjects/DJI/src/main/resources/Llama-32-1B";
    private final String tfModelDirSmoILM = "/home/acer/IdeaProjects/DJI/src/main/resources/SmolLM2-135M-Instruct";

    public ChatEngine() throws Exception {
        // this.chatModel = new LocalONNXGPT2Model(tfModelDirGPT2);
        // this.chatModel = new LocalONNXLlamaModel(tfModelDirLLama);
        this.chatModel = new LocalONNXSmolLM2Model(tfModelDirSmoILM);
        this.systemMessage =
               "You are a helpful assistant that ONLY answers questions based on the "
                        + "provided context. If no relevant context is provided, politely inform "
                        + "the user that you don't have the necessary information to answer their "
                        + "question accurately.";



        // Initialize conversation history with system message
        this.conversationHistory = new ArrayList<>();
        this.conversationHistory.add(new SystemMessage(systemMessage));

        // Define the prompt template that instructs the model to answer questions
        // based only on the provided context, with placeholders for context and question

        this.promptTemplate = "Based on the above context, provide a concise and direct answer in one sentence.\n" +
                "Context:\n" +
                "%s\n" +
                "\n" +
                "Question: %s\n" +
                "Answer:\n";


    }

    public List<ChatMessage> getConversationHistory() {
        return conversationHistory;
    }

    // method to format the prompt template with context and question parameters
    public String formatMessage(String userMessage, String context) {
        String formattedUserMessage = String.format(this.promptTemplate, context, userMessage);
        return formattedUserMessage;

    }

    public String sendMessage(String userMessage, String context) throws TranslateException {

        // Format the message using the prompt template
        String promt = formatMessage(context, userMessage);
        // Add the current message to the conversation history
        this.conversationHistory.add(new UserMessage(promt));
        // Get the response from the model
        ChatResponse aiMessage = chatModel.chat(this.conversationHistory);
        String responseContent = aiMessage.aiMessage().text();
        // Add the response to conversation history
        this.conversationHistory.add(new AiMessage(responseContent));
        // Return the response content
        return responseContent;
    }

    public void resetConversation() {
        // Reset the conversation history
        this.conversationHistory.clear();
        this.conversationHistory.add(new SystemMessage(systemMessage));
    }

}