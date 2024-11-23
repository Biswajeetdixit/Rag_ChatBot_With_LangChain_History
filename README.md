# Rag_ChatBot_With_LangChain_History
# 🚀 Conversational RAG With PDF Uploads and Chat History 🤖  

Welcome to the **Conversational RAG** project! This application allows users to upload PDF files 📄, extract their content, and chat interactively with the AI assistant 🤖. It's built with cutting-edge tools like **Streamlit**, **LangChain**, **Groq API**, and **HuggingFace** embeddings.

---

## 🌟 **Features**  

Here’s what this project offers:  
- **📄 PDF Uploading and Processing**: Upload multiple PDF files to extract their content seamlessly.  
- **🧩 Context-Aware Retrieval**: The system uses **LangChain** to handle historical chat context effectively.  
- **🤖 AI-Powered Conversations**: Ask questions about the uploaded documents, and get concise, accurate answers.  
- **🔄 Persistent Chat History**: Maintains a clean and interactive chat experience.  
- **🛠️ Modular Design**: Flexible and extensible architecture for future updates.

- ## 🧠 Chat History and Memory in the Project

### What is Chat History? 🗨️  
Chat history refers to the record of interactions between the user and the assistant. In this project, it ensures the assistant can maintain the **context** of the conversation, making responses more relevant and natural.  

For example:


---

## 🎬 **Demo & Screenshots**  

🎥 **Video Demo**: [Link to Demo](#)  
📸 **Screenshots**:  
1. **Home Page**  
   ![Home Page Screenshot](#)  
2. **PDF Upload Interface**  
   ![PDF Upload Interface Screenshot](#)  
3. **Chat Session with RAG Assistant**  
   ![Chat Screenshot](#)  

Here, the assistant uses chat history to understand "What about tomorrow?" without needing extra details from the user.

---

### How Memory Works in This Project 🧠  
Memory allows the assistant to recall previous interactions during the current session. It is **stateful**, meaning it tracks the conversation context throughout the session.  

#### Key Features:
- **Context-Aware Responses**:  
  The assistant remembers prior messages to provide meaningful and relevant answers.  
- **Session-Based Memory**:  
  Chat history is stored temporarily using `st.session_state` during each session.  
- **History in Retrieval-Augmented Generation (RAG)**:  
  - **Why?** To improve question-answering by using the chat's context and relevant document knowledge.  
  - **How?** The `RunnableWithMessageHistory` in LangChain enriches the user's query with past messages before retrieving or generating responses.

---

### Benefits of Chat History and Memory in the Project 🌟  
1. **Improved Contextuality**:  
   Enables the assistant to respond based on the flow of the conversation.
2. **Personalization**:  
   Makes interactions feel more natural and personalized for the user.
3. **Enhanced Efficiency**:  
   Avoids repetitive questions by "remembering" what was already discussed.

---

### Technical Implementation  
- **Chat History**:  
  Managed using LangChain's `ChatMessageHistory`, which organizes the conversation as a sequence of **user** and **assistant** messages.  
- **Memory Integration**:  
  Implemented using `RunnableWithMessageHistory`, which connects the RAG pipeline with the conversation history for contextualized question answering.  

#### Example Code:
```python
def get_session_history(session: str) -> BaseChatMessageHistory:
    # Initialize chat history if not available
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Attach chat history to the RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer" 
)
```
## **Why It Matters**🔑
Incorporating chat history and memory allows the assistant to handle complex, multi-turn conversations, enhancing the user experience and making the interactions more human-like.
---

## 📂 **Project Structure**  

- **`app.py`**: Main Streamlit application script.  
- **`requirements.txt`**: Contains all the dependencies needed to run the project.  
- **`temp.pdf`**: Temporary storage for uploaded PDFs.  
- **`.env`**: Stores API keys securely.  

---

## 🔧 **Setup Instructions**  

Follow these steps to get started:  
1. **Clone the Repository** 🛠️:  
   ```bash
   git clone https://github.com/your-repo/conversational-rag.git
   cd conversational-rag



## 🎯 **Future Improvements**
Here’s what’s planned for future releases:

- **📊` Improved Analytics`**: Add data visualization for insights into document contents.

  
- **🌐 `Multi-Language Support`**: Enable conversations in multiple languages.


- **🔐 ` Enhanced Security `**: Add authentication for user sessions.


- **⚡` Performance Optimization`**: Enhance processing speed for large documents.

## 🌟 About the Developer

This project is proudly built and maintained by **Biswajeet Dixit** 🧑‍💻.  
As a passionate learner and aspiring Generative AI engineer, I handled every aspect of this project, from ideation to deployment. Feel free to connect with me on LinkedIn or GitHub for any feedback or collaboration opportunities!

🔗 [LinkedIn Profile](https://www.linkedin.com/in/your-profile-link/)  
🔗 [GitHub Profile](https://github.com/your-profile-link/)  


