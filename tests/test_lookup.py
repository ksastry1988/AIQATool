import pytest

from qatool import qa
from qatool.chunking import Chunk
from qatool.lookup import extract_query_terms, merge_results
from qatool.vectorstore import VectorStore


class FakeModelClient:
    def __init__(self):
        self.context = None

    def generate(self, question, context):
        self.context = context
        return "answer"


def _store(tmp_path):
    store = VectorStore.open(tmp_path / ".qatool" / "index")
    store.upsert(
        [
            Chunk("def replace_file(): ...", {"file": "qatool/vectorstore.py", "start_line": 5,
                                              "end_line": 6, "symbol": "replace_file"}),
            Chunk("class VectorStore: ...", {"file": "qatool/vectorstore.py", "start_line": 1,
                                             "end_line": 4, "symbol": "VectorStore"}),
            Chunk("def main(): ...", {"file": "qatool/cli.py", "start_line": 1,
                                      "end_line": 2, "symbol": "main"}),
            Chunk("# docs", {"file": "README.md", "start_line": 1, "end_line": 1, "symbol": None}),
        ],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
    )
    return store


@pytest.mark.parametrize(
    ("question", "symbols", "filenames"),
    [
        ("where is replace_file called?", {"replace_file"}, set()),
        ("How does VectorStore persist data?", {"VectorStore"}, set()),
        ("what does `main` do", {"main"}, set()),
        ("who calls embed_texts()?", {"embed_texts"}, set()),
        ("explain VectorStore.find_exact", {"VectorStore", "find_exact"}, set()),
        ("explain VectorStore.query", {"VectorStore", "query"}, set()),
        ("what does store.query return?", {"query"}, set()),
        ("how is os.path.join used", {"join"}, set()),
        ("see e.g. the docs, i.e. README", set(), set()),
        ("what is in store.json?", set(), {"store.json"}),
        ("what is in setup.CFG", set(), {"setup.CFG"}),
        ("copy .qatoolignore.example", set(), {".qatoolignore.example"}),
        ("open qatool/store.query", set(), {"qatool/store.query"}),
        ("when is HTTPError raised", {"HTTPError"}, set()),
        ("what is in qatool/cli.py?", set(), {"qatool/cli.py"}),
        ("summarize `./README.md`.", set(), {"README.md"}),
        ("what goes in .qatoolignore", set(), {".qatoolignore"}),
        ("Where does the Index get built? It is main.", set(), set()),
    ],
)
def test_extract_query_terms(question, symbols, filenames):
    terms = extract_query_terms(question)
    assert terms.symbols == symbols
    assert terms.filenames == filenames


def test_find_exact_matches_symbols_before_filenames(tmp_path):
    results = _store(tmp_path).find_exact({"main"}, {"vectorstore.py"}, vector=[1.0, 0.0])

    assert [(r["metadata"]["symbol"], r["match"]) for r in results] == [
        ("main", "symbol"),
        ("VectorStore", "filename"),
        ("replace_file", "filename"),
    ]


def test_dotted_reference_finds_member_symbol_chunk(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        [Chunk("def query(self, vector): ...", {"file": "qatool/vectorstore.py", "start_line": 8,
                                                "end_line": 9, "symbol": "query"})],
        [[0.0, 1.0]],
    )
    terms = extract_query_terms("explain VectorStore.query")

    results = store.find_exact(terms.symbols, terms.filenames)

    assert terms.filenames == set()
    assert ("query", "symbol") in [(r["metadata"]["symbol"], r["match"]) for r in results]


def test_find_exact_filename_requires_whole_path_segments(tmp_path):
    store = _store(tmp_path)

    assert {r["metadata"]["file"] for r in store.find_exact(set(), {"qatool/cli.py"})} == {"qatool/cli.py"}
    assert store.find_exact(set(), {"li.py"}) == []
    assert store.find_exact(set(), set()) == []


def _whole_file_store(tmp_path):
    """Mirror the current index: one chunk per file, no symbol metadata."""
    store = VectorStore.open(tmp_path / ".qatool" / "index")
    store.upsert(
        [
            Chunk("class VectorStore:\n    def replace_file(self): ...\n",
                  {"file": "qatool/vectorstore.py", "start_line": 1, "end_line": 2, "symbol": None}),
            Chunk("store.replace_file(path)  # caller only\n",
                  {"file": "qatool/indexing.py", "start_line": 1, "end_line": 1, "symbol": None}),
            Chunk("func (s *Store) Replace_file() {}\n",
                  {"file": "store.go", "start_line": 1, "end_line": 1, "symbol": None}),
            Chunk("# docs", {"file": "README.md", "start_line": 1, "end_line": 1, "symbol": None}),
        ],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
    )
    return store


def test_find_exact_matches_definitions_in_whole_file_chunks(tmp_path):
    store = _whole_file_store(tmp_path)

    results = store.find_exact({"replace_file", "Replace_file"}, set())

    assert [(r["metadata"]["file"], r["match"]) for r in results] == [
        ("qatool/vectorstore.py", "definition"),
        ("store.go", "definition"),
    ]


def test_find_exact_ranks_symbol_then_definition_then_filename(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        [Chunk("def main():\n    pass\n", {"file": "tools/run.py", "start_line": 1, "end_line": 2})],
        [[0.0, 1.0]],
    )

    results = store.find_exact({"main"}, {"README.md"})

    assert [(r["metadata"]["file"], r["match"]) for r in results] == [
        ("qatool/cli.py", "symbol"),
        ("tools/run.py", "definition"),
        ("README.md", "filename"),
    ]


def test_ask_finds_symbol_definition_in_whole_file_index(tmp_path, monkeypatch):
    _whole_file_store(tmp_path)
    monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])
    client = FakeModelClient()

    qa.ask_question(tmp_path, "explain `VectorStore`", top_k=1, model_client=client)

    assert "Source: qatool/vectorstore.py:L1-L2 | Match: exact definition" in client.context
    assert "README.md" not in client.context


def test_merge_results_dedupes_and_ranks_exact_first():
    exact = [{"id": "a", "match": "symbol"}]
    semantic = [{"id": "b", "match": "vector"}, {"id": "a", "match": "vector"}, {"id": "c"}]

    merged = merge_results(exact, semantic, top_k=3)

    assert [(r["id"], r.get("match")) for r in merged] == [
        ("a", "symbol"), ("b", "vector"), ("c", None),
    ]
    assert len(merge_results(exact, semantic, top_k=1)) == 1


def test_merge_results_dedupes_by_metadata_without_ids():
    chunk = {"text": "x", "metadata": {"file": "a.py", "start_line": 1, "end_line": 2}}
    assert merge_results([chunk], [dict(chunk)], top_k=5) == [chunk]


def test_ask_includes_exact_symbol_match_despite_low_similarity(tmp_path, monkeypatch):
    _store(tmp_path)
    monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])
    client = FakeModelClient()

    answer = qa.ask_question(tmp_path, "where is replace_file defined?", top_k=2,
                             model_client=client)

    assert "Symbol: replace_file | Match: exact symbol" in client.context
    assert "README.md" in client.context  # best vector hit still fills the remaining slot
    assert "qatool/cli.py" not in client.context
    assert answer.splitlines()[-2:] == [
        "- [qatool/vectorstore.py:L5-L6]",
        "- [README.md:L1-L1]",
    ]


def test_ask_includes_exact_filename_match(tmp_path, monkeypatch):
    _store(tmp_path)
    monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])
    client = FakeModelClient()

    qa.ask_question(tmp_path, "what does cli.py do?", top_k=1, model_client=client)

    assert "Source: qatool/cli.py:L1-L2" in client.context
    assert "Match: exact filename" in client.context
