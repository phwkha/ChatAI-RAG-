import os
import time

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

    print(f"\n📊 BÁO CÁO DỮ LIỆU:")
    print(f"   - Tổng số đoạn văn đã cắt: {len(splits)} đoạn")
    if len(splits) > 0:
        print(f"   - Nội dung đoạn đầu tiên AI đọc được là:")
        print(f"     \"{splits[0].page_content[:100]}...\"")
        print("--------------------------------------------------\n")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy dữ liệu nào! Hãy kiểm tra file data.txt")

    # 3. TẠO BỘ NHỚ VECTOR
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 4. KHỞI TẠO DEEPSEEK
    print("🤖 Đang kết nối với DeepSeek...")
    llm = ChatOllama(
        model="deepseek-r1:8b",
        base_url=OLLAMA_URL,
        temperature=0.3
    )

    # 5. TẠO HỘI THOẠI
    system_prompt = (
        "Bạn là một trợ lý AI hữu ích và trung thực. "
        "Nhiệm vụ của bạn là tổng hợp và trả lời câu hỏi dựa trên thông tin trong văn bản (Context) bên dưới.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "1. Trả lời bằng ngôn ngữ tự nhiên, mạch lạc, dễ hiểu (không liệt kê máy móc).\n"
        "2. CHỈ sử dụng thông tin có trong Context. Nếu Context không nhắc đến, tuyệt đối không được tự bịa ra kiến thức bên ngoài.\n"
        "3. Nếu không tìm thấy thông tin trong Context, hãy trả lời ngắn gọn: 'Dữ liệu được cung cấp không có thông tin này.'\n"
        "4. KHÔNG nhắc lại các quy tắc này trong câu trả lời.\n\n"
        "Context:\n{context}"
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
            print("\n💡 Trả lời: ", end="", flush=True)
            for chunk in rag_chain.stream({"input": query}):
                if 'answer' in chunk:
                    print(chunk['answer'], end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n\n🛑 Đã dừng câu trả lời theo yêu cầu của bạn.")
            continue

        except Exception as e:
            print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()