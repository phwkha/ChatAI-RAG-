import os
import time

# Lấy địa chỉ của máy AI từ file docker-compose
# Nếu không thấy thì mặc định là localhost
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

print(f"🔌 Đang kết nối tới máy AI tại: {OLLAMA_URL}")

# Import các công cụ
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def main():
    print("⏳ Đang chờ máy AI khởi động (5 giây)...")
    time.sleep(5) 

    # 1. ĐỌC FILE
    print("📂 Đang đọc tài liệu data.txt...")
    try:
        loader = TextLoader("data.txt", encoding='utf-8')
        docs = loader.load()
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # 2. CẮT NHỎ & MÃ HÓA
    print("✂️  Đang xử lý dữ liệu...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. TẠO BỘ NHỚ VECTOR
    # Dùng model 'nomic-embed-text' để hiểu văn bản
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 4. KHỞI TẠO DEEPSEEK
    # Dùng model 'deepseek-r1:1.5b' cho nhẹ máy
    print("🤖 Đang kết nối với DeepSeek...")
    llm = ChatOllama(
        model="deepseek-r1:1.5b",
        base_url=OLLAMA_URL,
        temperature=0.3
    )

    # 5. TẠO HỘI THOẠI
    system_prompt = (
        "Bạn là một trợ lý AI nghiêm túc và trung thực. "
        "Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên thông tin được cung cấp trong phần ngữ cảnh (context) bên dưới.\n"
        "QUY TẮC:\n"
        "1. TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài (như lịch sử, địa lý, code...) nếu không có trong văn bản.\n"
        "2. Nếu thông tin không tồn tại trong ngữ cảnh, hãy trả lời chính xác câu này: 'Xin lỗi, dữ liệu của tôi không có thông tin này.'\n\n"
        "Ngữ cảnh:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    print("\n✅ HỆ THỐNG ĐÃ SẴN SÀNG! (Gõ 'exit' để thoát)")
    
    # 6. VÒNG LẶP CHAT
    while True:
        try:
            query = input("\n🗣️  Bạn hỏi: ")
            if query.lower() in ['exit', 'thoat']: break
            if not query: continue
            print("Thinking...", end="", flush=True)
            result = rag_chain.invoke({"input": query})
            print(f"\n💡 Trả lời: {result['answer']}")
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()