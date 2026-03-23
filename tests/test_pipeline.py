from src.pipeline import build_pipeline, query_pipeline


def test_build_pipeline_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    pipe = build_pipeline(str(empty), "pro_life")
    assert pipe["chunk_count"] == 0
    assert pipe["chunks"] == []
    assert pipe["group"] == "pro_life"
    assert "index" in pipe and "model" in pipe


def test_build_pipeline_with_text_file(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    doc = folder / "sample.txt"
    long_text = "Word " * 80
    doc.write_text(long_text, encoding="utf-8")
    pipe = build_pipeline(str(folder), "pro_choice")
    assert pipe["chunk_count"] > 0
    assert pipe["doc_count"] >= 1


def test_query_pipeline_returns_list(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("context about empathy and listening. " * 30, encoding="utf-8")
    pipe = build_pipeline(str(folder), "pro_life")
    out = query_pipeline(pipe, "empathy listening", top_k=3)
    assert isinstance(out, list)


def test_pipeline_dict_has_required_keys(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    pipe = build_pipeline(str(folder), "pro_life")
    for key in ("group", "chunks", "index", "model", "doc_count", "chunk_count"):
        assert key in pipe
