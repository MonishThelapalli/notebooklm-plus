import os
import sys
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Union
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

from dotenv import load_dotenv
from git import Repo
import tiktoken

# Core LangChain imports
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# Google Gemini model and embedding imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# Document loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    YoutubeLoader, 
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Centralized configuration for the RAG system."""
    
    # Chunking parameters
    CHUNK_SIZES = {
        "repo": 1500,
        "pdf": 1000,
        "youtube": 1500,
        "website": 1000,
        "docx": 1000,
        "csv": 500,
        "markdown": 1200,
        "text": 1000
    }
    
    CHUNK_OVERLAPS = {
        "repo": 200,
        "pdf": 150,
        "youtube": 200,
        "website": 150,
        "docx": 150,
        "csv": 50,
        "markdown": 150,
        "text": 100
    }
    
    # Retrieval parameters
    RETRIEVAL_K = 7
    MAX_CONTEXT_LENGTH = 8000
    
    # Model parameters
    MODEL_NAME = "gemini-2.0-flash-exp"
    EMBEDDING_MODEL = "models/text-embedding-004"
    TEMPERATURE = 0.1
    
    # File extensions for repo loading
    CODE_EXTENSIONS = [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", 
        ".hpp", ".cs", ".rb", ".go", ".rs", ".php", ".swift", ".kt", ".scala",
        ".r", ".m", ".sql", ".sh", ".bash", ".ps1", ".bat"
    ]
    
    DOC_EXTENSIONS = [
        ".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".conf", ".json", 
        ".yml", ".yaml", ".xml", ".html", ".htm", ".css", ".toml", ".env",
        ".properties", ".asciidoc", ".adoc", ".tex"
    ]
    
    # Cache settings
    CACHE_DIR = Path("./rag_cache")
    DB_DIR = Path("./vector_stores")
    REPO_DIR = Path("./repositories")
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.CACHE_DIR, cls.DB_DIR, cls.REPO_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# --- Utility Functions ---
class Utils:
    """Utility functions for the RAG system."""
    
    @staticmethod
    def sanitize_name(name: str, max_length: int = 50) -> str:
        """Sanitize a string to be used as a directory/file name."""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[^\w\s-]', '', name)
        sanitized = re.sub(r'[-\s]+', '-', sanitized)
        return sanitized[:max_length].strip('-').lower()
    
    @staticmethod
    def get_hash(text: str) -> str:
        """Generate a hash for the given text."""
        return hashlib.md5(text.encode()).hexdigest()[:10]
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate the number of tokens in text."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if a string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get the file extension from a path."""
        return Path(file_path).suffix.lower()

# --- Enhanced Document Loaders ---
class DocumentLoaderFactory:
    """Factory class for creating appropriate document loaders."""
    
    @staticmethod
    def get_loader(source_type: str, source_path: str, **kwargs):
        """Get the appropriate loader based on source type."""
        loaders = {
            "repo": DocumentLoaderFactory._load_repo,
            "pdf": DocumentLoaderFactory._load_pdf,
            "youtube": DocumentLoaderFactory._load_youtube,
            "website": DocumentLoaderFactory._load_website,
            "docx": DocumentLoaderFactory._load_docx,
            "csv": DocumentLoaderFactory._load_csv,
            "markdown": DocumentLoaderFactory._load_markdown,
            "text": DocumentLoaderFactory._load_text,
            "directory": DocumentLoaderFactory._load_directory
        }
        
        loader_func = loaders.get(source_type)
        if not loader_func:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return loader_func(source_path, **kwargs)
    
    @staticmethod
    def _load_repo(repo_url: str, **kwargs) -> List[Document]:
        """Load documents from a Git repository."""
        repo_name = Utils.sanitize_name(repo_url.split('/')[-1].replace('.git', ''))
        repo_path = Config.REPO_DIR / repo_name
        
        # Clone or update repository
        if not repo_path.exists():
            logger.info(f"Cloning repository: {repo_url}")
            Repo.clone_from(repo_url, to_path=repo_path)
        else:
            logger.info(f"Updating existing repository: {repo_path}")
            try:
                repo = Repo(repo_path)
                repo.remotes.origin.pull()
            except Exception as e:
                logger.warning(f"Could not update repo: {e}")
        
        # Load all relevant files
        all_extensions = Config.CODE_EXTENSIONS + Config.DOC_EXTENSIONS
        glob_pattern = "**/*"
        
        loader = GenericLoader.from_filesystem(
            str(repo_path),
            glob=glob_pattern,
            suffixes=all_extensions,
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        
        return loader.load()
    
    @staticmethod
    def _load_pdf(pdf_path: str, **kwargs) -> List[Document]:
        """Load documents from a PDF file."""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        loader = PyMuPDFLoader(pdf_path)
        return loader.load()
    
    @staticmethod
    def _load_youtube(video_url: str, **kwargs) -> List[Document]:
        """
        Load transcript from a YouTube video. If a transcript is not available,
        it generates one using Whisper, uploads it as a sidecar caption, and then loads it.
        """
        if "youtube.com" not in video_url and "youtu.be" not in video_url:
            raise ValueError("Invalid YouTube URL")

        try:
            # First, attempt to load existing captions
            logger.info(f"Attempting to load existing captions for {video_url}")
            loader = YoutubeLoader.from_youtube_url(
                video_url,
                add_video_info=True,
                language=["en", "en-US"],
                translation="en"
            )
            documents = loader.load()
            if documents and documents[0].page_content:
                logger.info("Successfully loaded existing captions.")
                return documents
            raise ValueError("Empty transcript loaded.")
        except Exception as e:
            logger.warning(f"Could not retrieve existing transcript for {video_url}: {e}. Attempting to generate one.")
            
            video_id_match = re.search(r"(?<=v=)[\w-]+|(?<=be/)[\w-]+", video_url)
            if not video_id_match:
                raise ValueError("Could not extract video ID from URL.")
            video_id = video_id_match.group(0)

            # Generate new captions
            srt_filename = f"{video_id}.srt"
            
            try:
                if not Path(srt_filename).exists():
                    print("🔊 Generating captions with Whisper... (This may take a while for long videos)")
                    model = whisper.load_model("base") # "base" is fast, "medium" is more accurate
                    result = model.transcribe(video_url, verbose=False)
                    
                    # Write to SRT file
                    with open(srt_filename, "w", encoding="utf-8") as srt_file:
                        for i, segment in enumerate(result['segments']):
                            start = segment['start']
                            end = segment['end']
                            text = segment['text'].strip()
                            
                            srt_file.write(f"{i + 1}\n")
                            srt_file.write(f"{Utils.format_timestamp(start)} --> {Utils.format_timestamp(end)}\n")
                            srt_file.write(f"{text}\n\n")
                    print(f"✅ Captions generated and saved to {srt_filename}")
                else:
                    print(f"ℹ️ Using existing generated caption file: {srt_filename}")

                # Upload the generated captions
                uploader = YouTubeUploader()
                uploader.upload_caption(video_id, srt_filename)

                # Now, try loading the transcript again
                return DocumentLoaderFactory._load_youtube_from_generated(video_url)

            except Exception as gen_e:
                logger.error(f"Failed to generate and upload captions: {gen_e}")
                raise RuntimeError(f"Could not process YouTube video after failing to generate captions: {gen_e}")

    @staticmethod
    def _load_youtube_from_generated(video_url: str) -> List[Document]:
        """A helper to load a YouTube transcript after ensuring captions exist."""
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        return loader.load()
    @staticmethod
    def _load_website(url: str, **kwargs) -> List[Document]:
        """Load content from a website."""
        if not Utils.validate_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        loader = WebBaseLoader(url)
        return loader.load()
    
    @staticmethod
    def _load_docx(docx_path: str, **kwargs) -> List[Document]:
        """Load documents from a Word file."""
        if not Path(docx_path).exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")
        
        loader = UnstructuredWordDocumentLoader(docx_path)
        return loader.load()
    
    @staticmethod
    def _load_csv(csv_path: str, **kwargs) -> List[Document]:
        """Load documents from a CSV file."""
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        loader = CSVLoader(csv_path)
        return loader.load()
    
    @staticmethod
    def _load_markdown(md_path: str, **kwargs) -> List[Document]:
        """Load documents from a Markdown file."""
        if not Path(md_path).exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        
        loader = UnstructuredMarkdownLoader(md_path)
        return loader.load()
    
    @staticmethod
    def _load_text(text_path: str, **kwargs) -> List[Document]:
        """Load documents from a text file."""
        if not Path(text_path).exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        
        loader = TextLoader(text_path, encoding=kwargs.get("encoding", "utf-8"))
        return loader.load()
    
    @staticmethod
    def _load_directory(dir_path: str, **kwargs) -> List[Document]:
        """Load all supported documents from a directory."""
        if not Path(dir_path).exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        documents = []
        supported_loaders = {
            ".pdf": DocumentLoaderFactory._load_pdf,
            ".docx": DocumentLoaderFactory._load_docx,
            ".csv": DocumentLoaderFactory._load_csv,
            ".md": DocumentLoaderFactory._load_markdown,
            ".txt": DocumentLoaderFactory._load_text
        }
        
        for file_path in Path(dir_path).rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in supported_loaders:
                    try:
                        docs = supported_loaders[ext](str(file_path))
                        documents.extend(docs)
                        logger.info(f"Loaded: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")
        
        return documents

# --- Enhanced Text Splitter ---
class SmartTextSplitter:
    """Intelligent text splitting with context preservation."""
    
    @staticmethod
    def split_documents(
        documents: List[Document], 
        source_type: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """Split documents intelligently based on content type."""
        
        if not documents:
            return []
        
        # Use config defaults if not specified
        chunk_size = chunk_size or Config.CHUNK_SIZES.get(source_type, 1000)
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAPS.get(source_type, 100)
        
        # Choose appropriate splitter
        if source_type == "repo":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True
            )
        
        # Split with metadata preservation
        split_docs = splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["total_chunks"] = len(split_docs)
            doc.metadata["source_type"] = source_type
        
        return split_docs

# --- Vector Store Manager ---
class VectorStoreManager:
    """Manages vector store creation, loading, and updates."""
    
    def __init__(self):
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        Config.setup_directories()
    
    def get_store_path(self, source_type: str, source_id: str) -> Path:
        """Generate a unique path for the vector store."""
        sanitized_id = Utils.sanitize_name(source_id)
        store_hash = Utils.get_hash(f"{source_type}_{source_id}")
        return Config.DB_DIR / f"{source_type}_{sanitized_id}_{store_hash}"
    
    def load_or_create_store(
        self, 
        source_type: str, 
        source_id: str,
        force_recreate: bool = False
    ):
        """Load existing vector store or create a new one."""
        store_path = self.get_store_path(source_type, source_id)
        
        # Check for existing store
        if store_path.exists() and not force_recreate:
            logger.info(f"Loading existing vector store from: {store_path}")
            try:
                vectorstore = Chroma(
                    persist_directory=str(store_path),
                    embedding_function=self.embedding_function
                )
                
                # Verify store is valid
                test_query = vectorstore.similarity_search("test", k=1)
                logger.info(f"Vector store loaded successfully ({len(test_query)} test results)")
                return vectorstore
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                logger.info("Creating new vector store...")
        
        # Create new store
        return self._create_new_store(source_type, source_id, store_path)
    
    def _create_new_store(
        self, 
        source_type: str, 
        source_id: str, 
        store_path: Path
    ):
        """Create a new vector store from source."""
        logger.info(f"Creating new vector store for: {source_id}")
        
        # Load documents
        try:
            documents = DocumentLoaderFactory.get_loader(source_type, source_id)
            logger.info(f"Loaded {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
        
        if not documents:
            raise ValueError("No documents found in the source")
        
        # Split documents
        chunks = SmartTextSplitter.split_documents(documents, source_type)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        if store_path.exists():
            shutil.rmtree(store_path)
        
        vectorstore = Chroma.from_documents(
            chunks,
            self.embedding_function,
            persist_directory=str(store_path)
        )
        
        logger.info(f"Vector store created and saved to: {store_path}")
        return vectorstore

# --- Enhanced Graph State ---
class GraphState(TypedDict):
    """Enhanced state for the LangGraph pipeline."""
    question: str
    context: str
    chat_history: List[BaseMessage]
    answer: str
    metadata: Dict
    confidence: float

# --- RAG Pipeline ---
class RAGPipeline:
    """Enhanced RAG pipeline with advanced features."""
    
    def __init__(self, retriever, source_type: str):
        self.retriever = retriever
        self.source_type = source_type
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            convert_system_message_to_human=True
        )
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        
        def rewrite_query(state: GraphState) -> GraphState:
            """Rewrite query based on chat history."""
            if not state.get("chat_history"):
                return state
            
            logger.info("Rewriting query based on chat history")
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Given a chat history and a follow-up question, 
                rephrase the follow-up question to be a standalone question that includes 
                all necessary context. Be specific and clear."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])
            
            chain = prompt | self.llm
            response = chain.invoke(state)
            state["question"] = response.content
            return state
        
        def retrieve_documents(state: GraphState) -> GraphState:
            """Retrieve relevant documents."""
            logger.info(f"Retrieving documents for: {state['question']}")
            
            # Retrieve documents
            docs = self.retriever.invoke(state["question"])
            
            # Build context with metadata
            context_parts = []
            total_tokens = 0
            
            for i, doc in enumerate(docs):
                doc_text = f"[Doc {i+1}]\n{doc.page_content}\n"
                doc_tokens = Utils.estimate_tokens(doc_text)
                
                if total_tokens + doc_tokens > Config.MAX_CONTEXT_LENGTH:
                    break
                
                context_parts.append(doc_text)
                total_tokens += doc_tokens
            
            state["context"] = "\n---\n".join(context_parts)
            state["metadata"] = {
                "num_docs": len(context_parts),
                "total_tokens": total_tokens
            }
            
            logger.info(f"Retrieved {len(context_parts)} documents ({total_tokens} tokens)")
            return state
        
        def generate_answer(state: GraphState) -> GraphState:
            """Generate answer using LLM."""
            logger.info("Generating answer")
            
            system_prompts = {
                "repo": """You are an expert programmer and code analyst. 
                Answer questions based on the provided code context. 
                Be specific, provide code examples when relevant, and explain technical concepts clearly.""",
                
                "pdf": """You are a document analyst. 
                Answer questions based on the provided PDF content. 
                Be accurate and cite specific sections when possible.""",
                
                "youtube": """You are analyzing video transcript content. 
                Answer questions based on the provided transcript. 
                Reference specific parts of the video when relevant.""",
                
                "website": """You are analyzing web content. 
                Answer questions based on the provided website information. 
                Be accurate and comprehensive.""",
                
                "default": """You are an AI assistant. 
                Answer questions based on the provided context. 
                Be accurate, helpful, and concise."""
            }
            
            system_content = system_prompts.get(self.source_type, system_prompts["default"])
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_content),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", """Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. 
If the context doesn't contain enough information, say so."""),
            ])
            
            chain = prompt | self.llm
            response = chain.invoke(state)
            state["answer"] = response.content
            
            # Calculate confidence (simplified)
            if state["metadata"]["num_docs"] > 0:
                state["confidence"] = min(0.95, 0.5 + (state["metadata"]["num_docs"] * 0.1))
            else:
                state["confidence"] = 0.1
            
            return state
        
        def should_rewrite(state: GraphState) -> str:
            """Determine if query should be rewritten."""
            return "rewrite_query" if state.get("chat_history") else "retrieve_documents"
        
        # Build workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("retrieve_documents", retrieve_documents)
        workflow.add_node("generate_answer", generate_answer)
        
        # Add edges
        workflow.set_conditional_entry_point(should_rewrite)
        workflow.add_edge("rewrite_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def run(self, question: str, chat_history: List[BaseMessage] = None) -> Dict:
        """Run the RAG pipeline."""
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "metadata": {},
            "confidence": 0.0
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            return {
                "answer": final_state.get("answer", "I couldn't find an answer."),
                "confidence": final_state.get("confidence", 0.0),
                "metadata": final_state.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

# --- Interactive CLI ---
class InteractiveCLI:
    """Enhanced command-line interface for the RAG system."""
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.pipeline = None
        self.chat_history = []
        self.current_source = None
    
    def run(self):
        """Run the interactive CLI."""
        self.setup_environment()
        self.print_welcome()
        
        while True:
            try:
                choice = self.get_source_choice()
                if choice == "exit":
                    break
                
                source_type, source_id = self.get_source_details(choice)
                if not source_id:
                    continue
                
                # Setup RAG system
                self.setup_rag_system(source_type, source_id)
                
                # Run Q&A loop
                self.run_qa_loop()
                
                # Ask if user wants to load another source
                if not self.ask_continue():
                    break
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\nError: {e}")
                if not self.ask_continue():
                    break
    
    def setup_environment(self):
        """Setup environment and check requirements."""
        load_dotenv()
        
        if not os.environ.get("GOOGLE_API_KEY"):
            print("❌ Error: GOOGLE_API_KEY not found!")
            print("Please set it in your .env file or environment variables.")
            sys.exit(1)
        
        Config.setup_directories()
        print("✅ Environment configured successfully\n")
    
    def print_welcome(self):
        """Print welcome message."""
        print("=" * 60)
        print("🚀 Advanced Multi-Source RAG System")
        print("=" * 60)
        print("\nSupported sources:")
        print("• GitHub repositories")
        print("• PDF documents")
        print("• YouTube videos")
        print("• Websites")
        print("• Word documents (.docx)")
        print("• CSV files")
        print("• Markdown files")
        print("• Text files")
        print("• Local directories\n")
    
    def get_source_choice(self) -> str:
        """Get source type choice from user."""
        print("\n" + "=" * 40)
        print("Select source type:")
        print("1. GitHub Repository")
        print("2. PDF File")
        print("3. YouTube Video")
        print("4. Website")
        print("5. Word Document (.docx)")
        print("6. CSV File")
        print("7. Markdown File")
        print("8. Text File")
        print("9. Local Directory")
        print("0. Exit")
        print("=" * 40)
        
        choice = input("\nEnter choice (0-9): ").strip()
        
        mapping = {
            "1": "repo",
            "2": "pdf",
            "3": "youtube",
            "4": "website",
            "5": "docx",
            "6": "csv",
            "7": "markdown",
            "8": "text",
            "9": "directory",
            "0": "exit"
        }
        
        return mapping.get(choice, "invalid")
    
    def get_source_details(self, source_type: str) -> tuple:
        """Get source details from user."""
        if source_type == "invalid":
            print("❌ Invalid choice!")
            return None, None
        
        prompts = {
            "repo": "Enter GitHub repository URL: ",
            "pdf": "Enter PDF file path: ",
            "youtube": "Enter YouTube video URL: ",
            "website": "Enter website URL: ",
            "docx": "Enter Word document path: ",
            "csv": "Enter CSV file path: ",
            "markdown": "Enter Markdown file path: ",
            "text": "Enter text file path: ",
            "directory": "Enter directory path: "
        }
        
        prompt = prompts.get(source_type)
        if not prompt:
            return None, None
        
        source_id = input(f"\n{prompt}").strip()
        
        # Validate input
        if not source_id:
            print("❌ No input provided!")
            return None, None
        
        # Additional validation based on type
        if source_type in ["pdf", "docx", "csv", "markdown", "text", "directory"]:
            if not Path(source_id).exists():
                print(f"❌ Path does not exist: {source_id}")
                return None, None
        elif source_type in ["repo", "website"]:
            if not Utils.validate_url(source_id):
                print(f"❌ Invalid URL: {source_id}")
                return None, None
        elif source_type == "youtube":
            if "youtube.com" not in source_id and "youtu.be" not in source_id:
                print(f"❌ Invalid YouTube URL: {source_id}")
                return None, None
        
        return source_type, source_id
    
    def setup_rag_system(self, source_type: str, source_id: str):
        """Setup the RAG system for the given source."""
        print(f"\n📊 Processing: {source_id}")
        print("This may take a few moments...\n")
        
        try:
            # Check if user wants to force recreate
            force_recreate = False
            store_path = self.vector_manager.get_store_path(source_type, source_id)
            if store_path.exists():
                response = input("Vector store exists. Recreate it? (y/N): ").strip().lower()
                force_recreate = response == 'y'
            
            # Load or create vector store
            vectorstore = self.vector_manager.load_or_create_store(
                source_type, 
                source_id,
                force_recreate=force_recreate
            )
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": Config.RETRIEVAL_K}
            )
            
            # Create pipeline
            self.pipeline = RAGPipeline(retriever, source_type)
            self.current_source = source_id
            self.chat_history = []  # Reset chat history for new source
            
            print(f"\n✅ RAG system ready for: {source_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            print(f"❌ Error: {e}")
            raise
    
    def run_qa_loop(self):
        """Run the question-answering loop."""
        print("\n" + "=" * 60)
        print("💬 Ask questions about the content")
        print("Commands: 'exit', 'clear' (clear history), 'info' (source info)")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n🤔 You: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() == 'exit':
                    break
                elif question.lower() == 'clear':
                    self.chat_history = []
                    print("✅ Chat history cleared")
                    continue
                elif question.lower() == 'info':
                    print(f"📄 Current source: {self.current_source}")
                    print(f"💬 History length: {len(self.chat_history)} messages")
                    continue
                
                # Get answer
                print("\n🤖 AI: ", end="", flush=True)
                result = self.pipeline.run(question, self.chat_history)
                
                # Display answer
                print(result["answer"])
                
                # Show confidence if low
                if result["confidence"] < 0.5:
                    print(f"\n⚠️  Confidence: {result['confidence']:.1%}")
                
                # Update chat history
                self.chat_history.extend([
                    HumanMessage(content=question),
                    AIMessage(content=result["answer"])
                ])
                
                # Keep history manageable
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
                
            except KeyboardInterrupt:
                print("\n\nExiting Q&A...")
                break
            except Exception as e:
                logger.error(f"Q&A error: {e}")
                print(f"\n❌ Error: {e}")
    
    def ask_continue(self) -> bool:
        """Ask user if they want to continue."""
        response = input("\n\n🔄 Do you want to load another source? (y/N): ").strip().lower()
        return response == 'y'


# --- Main Execution ---
def main():
    """Main entry point for the RAG system."""
    # Set USER_AGENT to avoid warning
    os.environ['USER_AGENT'] = 'RAG-System/1.0'
    
    try:
        cli = InteractiveCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        print("\n👋 Thank you for using the RAG system!")


if __name__ == "__main__":
    main()
