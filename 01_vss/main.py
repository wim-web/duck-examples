import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import duckdb

# You can download models from the Hugging Face Hub 🤗 as follows:
tokenizer = AutoTokenizer.from_pretrained(
    "pfnet/plamo-embedding-1b", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "pfnet/plamo-embedding-1b", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

documents = [
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "PLaMo-Embedding-1Bとは何ですか？",
    "最近は随分と暖かくなりましたね。",
    "いぬも歩けば棒にあたります。"
]

with torch.inference_mode():
    # For other texts/sentences, please use the `encode_document` method.
    # Also, for applications other than information retrieval, please use the `encode_document` method.
    document_embeddings = model.encode_document(documents, tokenizer)

# DuckDBの初期化とテーブル作成
db = duckdb.connect()
db.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding DOUBLE[2048]
    )
""")

# 文書とベクトルの保存
for i, (doc, embedding) in enumerate(zip(documents, document_embeddings)):
    # PyTorchテンソルを1次元配列に変換してからリスト化
    embedding_list = embedding.squeeze().cpu().tolist()
    db.execute("""
        INSERT INTO documents (id, content, embedding)
        VALUES (?, ?, ?)
    """, [i, doc, embedding_list])


query = "PLaMo-Embedding-1Bとは何ですか？"

with torch.inference_mode():
    # For embedding query texts in information retrieval, please use the `encode_query` method.
    # You also need to pass the `tokenizer`.
    query_embedding = model.encode_query(query, tokenizer)

# クエリベクトルを使用した類似度検索
query_vector = query_embedding.squeeze().cpu().tolist()
results = db.execute("""
    SELECT
        content,
        array_cosine_distance(embedding::DOUBLE[2048], ?::DOUBLE[2048]) as distance
    FROM documents
    ORDER BY distance ASC
    LIMIT 5
""", [query_vector]).fetchall()

print("\nDuckDBによる検索結果:")
for content, distance in results:
    print(f"距離: {distance:.4f} - {content}")
