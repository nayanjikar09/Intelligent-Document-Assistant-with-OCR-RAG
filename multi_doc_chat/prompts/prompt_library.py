from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------------

# 1. Contextualize Question - Rewrites user question with chat history
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Context QA - Answer using retrieved context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 3. OCR Context QA - For answering from OCR-extracted text
ocr_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant analyzing text extracted from images/documents using OCR. Use the provided context "
        "to answer questions. The text may contain OCR errors or incomplete words. If uncertain, say 'I don't know.'\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 4. Document Summary Prompt - For summarizing uploaded documents
document_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Summarize the following document content in 2-3 sentences. Focus on the main topic and key points.\n\n{context}"
    )),
    ("human", "Summarize this document."),
])

# -------------------------------------------------------------------------
# Prompt Registry
# -------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
    "ocr_qa": ocr_qa_prompt,
    "document_summary": document_summary_prompt,
}


# -------------------------------------------------------------------------
# Prompt Types (for type safety)
# -------------------------------------------------------------------------

class PromptType:
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"
    OCR_QA = "ocr_qa"
    DOCUMENT_SUMMARY = "document_summary"