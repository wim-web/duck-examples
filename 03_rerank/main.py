import torch
from transformers import AutoModel, AutoTokenizer
import duckdb
from lindera_py import Segmenter, Tokenizer, load_dictionary
from sentence_transformers import CrossEncoder

documents = [
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "PLaMo-Embedding-1Bとは何ですか？",
    "最近は随分と暖かくなりましたね。",
    "いぬも歩けば棒にあたります。"
]

db = duckdb.connect()
db.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding DOUBLE[2048]
)
""")
db.execute("""
CREATE TABLE IF NOT EXISTS docs (
    id      INTEGER,
    text    TEXT,
    tok     TEXT
);
""")


def vectorize(text: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(
        "pfnet/plamo-embedding-1b", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "pfnet/plamo-embedding-1b", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with torch.inference_mode():
        document_embeddings = model.encode_document(text, tokenizer)

    for i, (doc, embedding) in enumerate(zip(documents, document_embeddings)):
        # PyTorchテンソルを1次元配列に変換してからリスト化
        embedding_list = embedding.squeeze().cpu().tolist()
        db.execute("""
            INSERT INTO documents (id, content, embedding)
            VALUES (?, ?, ?)
        """, [i, doc, embedding_list])


def tokenize(text: str) -> str:
    dictionary = load_dictionary("ipadic")             # または "unidic-lite" など
    segmenter = Segmenter("normal", dictionary)
    tokenizer = Tokenizer(segmenter)
    """
    Lindera で形態素解析し、表層形をスペース区切りで返す
    """
    tokens = tokenizer.tokenize(text)
    surfaces = [token.text for token in tokens]
    return " ".join(surfaces)


def full_text_search_insert(text: list[str]):
    for doc_id, text in enumerate(text):
        tok = tokenize(text)
        db.execute("INSERT INTO docs VALUES (?, ?, ?)", (doc_id, text, tok))
    db.execute("""
PRAGMA create_fts_index(
    'docs',    -- テーブル名
    'id',      -- ドキュメント識別子
    'tok',     -- トークン列
    stemmer       = 'none',
    stopwords     = 'none',
    ignore        = '',
    lower         = false,
    strip_accents = false
);
""")


def full_text_search(query: str):
    tok_query = tokenize(query)
    sql = """
WITH scored AS (
    SELECT
        id,
        text,
        fts_main_docs.match_bm25(
            id,
            ?,
            fields := 'tok'
        ) AS score
    FROM docs
)
SELECT
    id,
    text,
    score
FROM scored
WHERE score IS NOT NULL
ORDER BY score DESC;
"""
    results = db.execute(sql, [tok_query]).fetchall()
    return results


def vector_similarity_search(query: str):
    with torch.inference_mode():
        tokenizer = AutoTokenizer.from_pretrained(
            "pfnet/plamo-embedding-1b", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "pfnet/plamo-embedding-1b", trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

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
    return results


def search(query: str):
    fts_results = full_text_search(query)
    print(fts_results)

    vss_results = vector_similarity_search(query)
    print(vss_results)

    rerank(query, fts_results, vss_results)


def rerank(query, fts_results, vss_results):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r_model = CrossEncoder(
        "hotchpotch/japanese-bge-reranker-v2-m3-v1", max_length=512, device=device
    )

    # FTSとVSSの結果を結合
    pairs = []
    for fts_result in fts_results:
        pairs.append([query, fts_result[1]])  # FTSの結果のテキストを使用
    for vss_result in vss_results:
        pairs.append([query, vss_result[0]])  # VSSの結果のテキストを使用

    # 重複を削除
    pairs = list(dict.fromkeys(map(tuple, pairs)))

    # rerankerモデルでスコアリング
    with torch.inference_mode():
        scores = r_model.predict(pairs)

    # スコアとテキストを組み合わせる
    reranked_results = []
    for i, score in enumerate(scores):
        reranked_results.append((score, pairs[i][1]))  # スコアとテキストを保持

    # スコアでソート
    reranked_results.sort(key=lambda x: x[0], reverse=True)

    # 結果の表示
    print("Reranked Results:")
    for score, text in reranked_results:
        print(f"Score: {score:.4f}, Text: {text}")


def main():
    vectorize(documents)
    full_text_search_insert(documents)

    query = "PLaMo-Embedding-1Bについて教えて下さい"
    search(query)


if __name__ == "__main__":
    main()
