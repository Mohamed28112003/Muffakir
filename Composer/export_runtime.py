"""Standalone Python source appended to exported configuration; no template engine."""

RUNTIME = r'''

def load_config():
    """Resolve project paths and environment variables only when explicitly called."""
    import copy
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    config = copy.deepcopy(CONFIG)
    missing = []
    for item in ENVIRONMENT:
        value = os.environ.get(item["name"]) or item.get("default", "")
        if not value and item["required"]:
            missing.append(item["name"])
        if not item.get("path"):
            continue
        target = config
        for key in item["path"][:-1]:
            target = target[int(key)] if isinstance(target, list) else target.setdefault(key, {})
        key = item["path"][-1]
        target[int(key) if isinstance(target, list) else key] = value or None
    if missing:
        raise ValueError("Set these environment variables before running: " + ", ".join(missing))
    state = Path(os.environ.get("MUFFAKIR_INDEX_DIR", "./index"))
    state = (ROOT / state).resolve()
    if config.get("retrieval_source") != "web_search_only":
        corpus = Path(os.environ.get("MUFFAKIR_DOCUMENTS_DIR", "./knowledge-base"))
        config["data_dir"] = str((ROOT / corpus).resolve())
        config["db_path"] = str(state / "vectors")
        vector = config.setdefault("vector_db_config", {})
        provider = config["vector_db_provider"]
        config["collection_name"] = os.environ.get("MUFFAKIR_COLLECTION", "muffakir_" + EXPORT_METADATA["export_id"])
        vector["collection_name"] = config["collection_name"]
        if provider == "chroma":
            vector["path"] = config["db_path"]
        elif provider == "faiss":
            vector["folder_path"] = config["db_path"]
            vector["index_name"] = "index"
        elif provider == "pinecone":
            vector["index_name"] = config["collection_name"]
    config["skip_document_ingestion"] = True
    return config, state


def index_signature(config):
    import hashlib
    # Exclude secrets; changing query-time settings does not rebuild the corpus.
    keys = ("chunking_method", "chunk_size", "chunk_overlap", "embedding_provider",
            "embedding_model", "vector_db_provider", "collection_name", "data_dir",
            "document_parser", "use_ocr", "language")
    values = {key: config.get(key) for key in keys}
    vector = config.get("vector_db_config", {})
    values["connection"] = {key: vector.get(key) for key in ("location", "url", "index_name")}
    values["uri"] = vector.get("connection_args", {}).get("uri")
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def create_components(config):
    from Embedding.EmbeddingProvider import EmbeddingProvider
    from VectorDB import create_vector_db

    embedding = EmbeddingProvider(
        model_name=config["embedding_model"],
        provider=config.get("embedding_provider", "sentence_transformers"),
        api_key=config.get("api_key"),
        batch_size=config.get("embedding_batch_size", 16),
        device=config.get("device", "auto"),
    )
    options = dict(config.get("vector_db_config") or {})
    options.setdefault("model_name", config["embedding_model"])
    database = create_vector_db(config["vector_db_provider"], embedding_provider=embedding, **options)
    return embedding, database


def assert_empty_remote(config):
    """Check dedicated remote targets before writing documents."""
    provider = config["vector_db_provider"]
    vector = config.get("vector_db_config", {})
    collection = config["collection_name"]
    if provider == "qdrant":
        from qdrant_client import QdrantClient
        client = QdrantClient(url=vector["location"], api_key=vector.get("api_key"))
        try:
            if client.collection_exists(collection):
                raise ValueError("Collection already exists. Set MUFFAKIR_COLLECTION to a fresh name.")
        finally:
            client.close()
    elif provider == "pinecone":
        from pinecone import Pinecone
        index = Pinecone(api_key=vector["api_key"]).Index(collection)
        if index.describe_index_stats().total_vector_count:
            raise ValueError("Pinecone index is not empty. Supply a dedicated empty index.")
    elif provider == "milvus":
        from pymilvus import MilvusClient
        client = MilvusClient(**vector["connection_args"])
        try:
            if client.has_collection(collection):
                raise ValueError("Collection already exists. Set MUFFAKIR_COLLECTION to a fresh name.")
        finally:
            client.close()


def index_documents():
    config, state = load_config()
    if config.get("retrieval_source") == "web_search_only":
        raise ValueError("Web Search Only needs no index. Use --question instead.")
    if not Path(config["data_dir"]).is_dir():
        raise ValueError("Set MUFFAKIR_DOCUMENTS_DIR to an existing document directory.")
    # Exclusive creation also blocks repeated or concurrent indexing after a partial failure.
    try:
        state.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("Index location already exists. Use a fresh MUFFAKIR_INDEX_DIR and collection to rebuild.") from None
    assert_empty_remote(config)
    from TextProcessor.MuffakirChunking import MuffakirChunking
    from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing

    parser = None
    if config.get("document_parser"):
        from DocumentParser import create_document_parser
        parser = create_document_parser(config["document_parser"], **(config.get("document_parser_config") or {}))
    elif config.get("use_ocr") and config.get("azure_endpoint"):
        from DocumentParser import create_document_parser
        parser = create_document_parser("azure", endpoint=config["azure_endpoint"], api_key=config.get("azure_api_key"))
    chunking = MuffakirChunking(
        chunker=config["chunking_method"],
        chunker_config={"size": config["chunk_size"], "overlap": config["chunk_overlap"]},
        language=config.get("language", "auto"),
    )
    processor = ChunkingAndProcessing(directory_path=config["data_dir"], muffakir_chunking=chunking, document_parser=parser)
    documents = processor.process_all(chunking_method=config["chunking_method"], use_ocr=bool(config.get("use_ocr")))
    if not documents:
        raise ValueError("No document chunks were produced. Check the corpus and parser settings.")
    _, database = create_components(config)
    database.add_documents(documents)
    # Portable chunk cache supports hybrid/BM25 retrieval across backend restarts.
    chunks = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    (state / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, default=str), encoding="utf-8")
    (state / "ready.json").write_text(json.dumps({"signature": index_signature(config)}), encoding="utf-8")
    print(f"Indexed {len(documents)} chunks. Now run with --question.")


class CorpusIndex:
    """Delegate vector operations and supply the complete corpus for hybrid retrieval."""
    def __init__(self, database, documents):
        self.database = database
        self.documents = documents

    def __getattr__(self, name):
        return getattr(self.database, name)

    def get_all_documents(self):
        return self.documents

    def load_all_documents(self):
        return self.documents


def build_pipeline():
    """Load one selected pipeline without re-running Composer or ingesting documents."""
    config, state = load_config()
    if config.get("retrieval_source") == "web_search_only":
        from Muffakir import MuffakirSearch
        return MuffakirSearch(config=config)
    ready = state / "ready.json"
    if not ready.is_file() or not (state / "chunks.json").is_file():
        raise ValueError("No completed export index found. Run python rag_app.py --index first.")
    if json.loads(ready.read_text(encoding="utf-8")).get("signature") != index_signature(config):
        raise ValueError("Index settings changed. Index into a fresh location and collection first.")
    if config["vector_db_provider"] in {"chroma", "faiss"} and not Path(config["db_path"]).is_dir():
        raise ValueError("The vector index is missing. Restore the complete index directory or build in a fresh location.")
    from langchain_core.documents import Document
    documents = [Document(**item) for item in json.loads((state / "chunks.json").read_text(encoding="utf-8"))]
    embedding, database = create_components(config)
    database = CorpusIndex(database, documents)
    if config.get("pipeline_mode") == "retrieval_only":
        from Muffakir import MuffakirRetrieval
        return MuffakirRetrieval(config, embedding_provider=embedding, db_manager=database)
    from Muffakir import MuffakirRAG
    return MuffakirRAG(config, embedding_provider=embedding, db_manager=database)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG pipeline exported from a Composer trial.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--index", action="store_true", help="Build a fresh index from your documents")
    action.add_argument("--question", help="Ask a question using the selected pipeline")
    args = parser.parse_args()
    try:
        if args.index:
            index_documents()
            return
        pipeline = build_pipeline()
        if CONFIG.get("pipeline_mode") == "retrieval_only":
            result = [{"text": doc.page_content, "metadata": doc.metadata}
                      for doc in pipeline.get_similar_documents(args.question)]
        else:
            result = pipeline.ask(args.question)
        print(json.dumps(result, ensure_ascii=False, indent=2,
                         default=lambda value: {"text": value.page_content, "metadata": value.metadata}
                         if hasattr(value, "page_content") else str(value)))
    except Exception as exc:
        parser.exit(1, f"Pipeline failed ({type(exc).__name__}). Check credentials, provider availability, and the index setup.\n"
                    + (str(exc) + "\n" if isinstance(exc, (ValueError, FileNotFoundError)) else ""))


if __name__ == "__main__":
    main()
'''
