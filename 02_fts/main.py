import duckdb
from lindera_py import Segmenter, Tokenizer, load_dictionary

# 1. Lindera の辞書を読み込んでセグメンターとトークナイザーを準備
dictionary = load_dictionary("ipadic")             # または "unidic-lite" など
segmenter = Segmenter("normal", dictionary)
tokenizer = Tokenizer(segmenter)


def tokenize(text: str) -> str:
    """
    Lindera で形態素解析し、表層形をスペース区切りで返す
    """
    tokens = tokenizer.tokenize(text)
    surfaces = [token.text for token in tokens]
    return " ".join(surfaces)


# 2. サンプルデータ
docs = [
    (1, "今日はいい天気ですね。明日も晴れるといいな。"),
    (2, "昨日は雨で寒かったけど、今日は暖かい。"),
    (3, "おはようございます。気持ちのいい朝です。")
]

# 3. DuckDB に接続してテーブルを作成
conn = duckdb.connect()
conn.execute("""
CREATE TABLE docs (
    id      INTEGER,
    text    TEXT,
    tok     TEXT
);
""")

# 4. トークン化してデータ投入
for doc_id, text in docs:
    tok = tokenize(text)
    conn.execute("INSERT INTO docs VALUES (?, ?, ?)", (doc_id, text, tok))

# 5. FTS インデックス作成
conn.execute("""
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

# 6. 検索例
query = "いい 天気"
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
results = conn.execute(sql, (tok_query,)).fetchall()

for row in results:
    print(f"id={row[0]}: {row[1]} (score={row[2]})")
