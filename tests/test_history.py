from bgremover.history import HistoryStore


def test_history_persists_and_newest_first(tmp_path):
    store=HistoryStore(tmp_path/"history.json")
    store.add({"source_name":"first","total":1})
    store.add({"source_name":"second","total":2})
    reloaded=HistoryStore(tmp_path/"history.json").load()
    assert [row["source_name"] for row in reloaded] == ["second","first"]
