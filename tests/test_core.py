"""Run: python -m pytest tests -q   (needs data built first: python -m src.build)"""
import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from src.analytics import load_portfolio
from src.config import PROCESSED
from src.llm import SQLRejected, ask, build_db, safe_sql, verify_numbers


@pytest.fixture(scope="module")
def df():
    return load_portfolio()


@pytest.fixture(scope="module")
def con(df):
    return build_db(df, json.load(open(PROCESSED / "model_metrics.json")))


def test_join_is_lossless(df):
    assert len(df) == 14685 and df.account_id.is_unique


def test_gender_standardised(df):
    assert set(df.gender.unique()) == {"Male", "Female"}


def test_no_target_leakage_columns_in_model():
    from src.model import NUM, CAT
    assert not [c for c in NUM + CAT if c.startswith("coll_") or c in ("target_default", "gender", "marital_status")]


def test_scores_are_probabilities(df):
    assert df.pd_lgbm.between(0, 1).all() and df.anomaly_flag.sum() > 0


@pytest.mark.parametrize("sql", ["DROP TABLE portfolio", "SELECT 1; SELECT 2", "PRAGMA query_only=OFF",
                                 "WITH x AS (SELECT 1) DELETE FROM portfolio"])
def test_sql_guard(con, sql):
    with pytest.raises(SQLRejected):
        safe_sql(con, sql)


def test_grounding_check():
    r = [pd.DataFrame({"rate": [0.2465], "n": [1432]})]
    assert verify_numbers("Mortgage defaults at 24.7% across 1,432 accounts", r) == []
    assert verify_numbers("Mortgage defaults at 31.2%", r) == ["31.2%"]


def test_rules_fallback_answers_and_declines(con):
    a = ask("Which loan type has the highest default rate?", con, api_key=None)
    assert a.status == "answered" and "Mortgage" in a.text and a.evidence
    assert ask("Will defaults rise next year?", con).status == "declined"


def _fake_client(final_json):
    calls = {"n": 0}

    class M:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                blk = NS(type="tool_use", id="t1", name="run_sql",
                         input={"sql": "SELECT loan_type, COUNT(*) n, AVG(target_default) r FROM portfolio GROUP BY 1 ORDER BY r DESC"})
                return NS(stop_reason="tool_use", content=[blk])
            return NS(stop_reason="end_turn", content=[NS(type="text", text=final_json)])
    return NS(messages=M())


def test_llm_tool_loop_grounded(con):
    a = ask("Which loan type has the highest default rate?", con,
            client=_fake_client('{"status":"answered","answer":"Mortgage has the highest default rate at 24.7% (1,432 accounts)."}'))
    assert a.status == "answered" and a.engine == "llm" and a.evidence and not a.unverified


def test_llm_hallucinated_number_is_flagged(con):
    a = ask("Which loan type has the highest default rate?", con,
            client=_fake_client('{"status":"answered","answer":"Mortgage has the highest default rate at 41.9% (1,432 accounts)."}'))
    assert a.status == "declined" and a.unverified == ["41.9%"]


def test_llm_answer_without_query_is_refused(con):
    class M:
        def create(self, **kw):
            return NS(stop_reason="end_turn", content=[NS(type="text", text='{"status":"answered","answer":"It is 20%."}')])
    a = ask("what is the default rate", con, client=NS(messages=M()))
    assert a.status == "declined"
@pytest.mark.parametrize("sql", [
    "SELECT name FROM sqlite_master",
    "SELECT load_extension('malicious')",
    "SELECT randomblob(1000000000)",
    "SELECT * FROM pragma_database_list",
])
def test_sql_authorizer_rejects_unsafe_queries(con, sql):
    with pytest.raises(SQLRejected):
        safe_sql(con, sql)


def test_sql_authorizer_allows_aggregation(con):
    assert len(safe_sql(con, "SELECT loan_type, COUNT(*) n FROM portfolio GROUP BY loan_type")) > 0


def test_invalid_llm_json_is_declined(con):
    class M:
        def create(self, **kw):
            return NS(stop_reason="end_turn", content=[NS(type="text", text="not JSON")])
    assert ask("default rate?", con, client=NS(messages=M())).status == "declined"
