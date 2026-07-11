"""TRACE benchmark loaders for LLM continual learning.

The official TRACE benchmark (Wang et al., 2023, https://arxiv.org/abs/2310.06762)
has 8 heterogeneous tasks: C-STANCE / FOMC / MeetingBank / Py150 / ScienceQA /
NumGLUE-cm / NumGLUE-ds / 20Minuten. The original data is distributed via
Google Drive (not HF). This loader uses HF mirrors / substitutes that
preserve task domain and answer modality:

  task         official source           HF source we use
  ----         ---------------           ----------------
  C-STANCE     custom (zh stance)        yfhe/C-STANCE-A
  FOMC         FOMC-Communication        gtfintechlab/fomc_communication
  MeetingBank  Meeting summarization     huuuyeah/meetingbank
  Py150        Python code completion    angie-chen55/python-github-code
  ScienceQA    Multimodal QA             derek-thomas/ScienceQA  (text-only filter)
  NumGLUE-cm   Numerical (arithmetic)    tasksource/num-glue config=type_4
  NumGLUE-ds   Numerical (passage QA)    tasksource/num-glue config=type_5
  20Minuten    German news simpl.        giuliadc/20minuten

Eval framing. The training pipeline uses teacher-forced exact-match on
the supervised tokens. For classification tasks (C-STANCE, FOMC,
ScienceQA, NumGLUE-cm/ds) the answer is a short label/number so
exact-match is meaningful. For the three generation tasks (MeetingBank,
Py150, 20Minuten) we restrict the supervised target to a SHORT prefix
of the gold output (~12 words) so loss is well-defined and exact-match
has a nonzero (but small) ceiling. Loss remains the primary metric for
these three. This matches the TRACE convention of reporting per-task
heterogeneous metrics while keeping our pipeline single-headed.

Each loader returns a HuggingFace ``Dataset`` with ``prompt`` and
``answer`` fields, identical contract to ``tasks_cls.py``.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from datasets import Dataset, load_dataset

from data import Task


# ---------------------------------------------------------------------------
# Label vocabularies + truncation helpers.
# ---------------------------------------------------------------------------
# C-STANCE labels are Chinese 支持/反对/中立; map to English for cleaner
# tokenisation under the Qwen2.5 BPE.
CSTANCE_LABELS = {"支持": "favor", "反对": "against", "中立": "neutral"}

# FOMC labels in the gtfintechlab mirror are integer-encoded:
#   0 -> dovish, 1 -> hawkish, 2 -> neutral
FOMC_LABELS = ["dovish", "hawkish", "neutral"]

# Maximum words clipped before tokenisation. Generation tasks need a
# bigger prompt window because the context (transcript / code prefix) is
# the signal; the answer prefix supervision is short either way.
MAX_PROMPT_WORDS = 120
MAX_PROMPT_WORDS_LONG = 200
MAX_ANSWER_WORDS_GEN = 12


def _clip(text: str, max_words: int = MAX_PROMPT_WORDS) -> str:
    if text is None:
        return ""
    words = str(text).split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).replace("\n", " ").strip()


def _take(ds: Dataset, n: int, seed: int) -> Dataset:
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def _to_prompt_answer(
    ds: Dataset, fmt_prompt: Callable, fmt_answer: Callable
) -> Dataset:
    keep = ds.column_names

    def _map(ex):
        return {"prompt": fmt_prompt(ex), "answer": fmt_answer(ex)}

    return ds.map(_map, remove_columns=keep)


def _split_train_eval(ds: Dataset, n_eval: int, seed: int) -> Tuple[Dataset, Dataset]:
    """For sources that only ship a train split, hold out a slice for eval."""
    ds = ds.shuffle(seed=seed)
    n_eval = min(n_eval, max(1, len(ds) // 5))
    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Per-task loaders. Each returns (train_ds, eval_ds) of (prompt, answer).
# ---------------------------------------------------------------------------

def _load_cstance(seed: int, n_train: int, n_eval: int):
    raw = load_dataset("yfhe/C-STANCE-A", split="train")
    raw = raw.filter(lambda ex: ex.get("Stance") in CSTANCE_LABELS)
    train_raw, eval_raw = _split_train_eval(raw, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    p = lambda ex: (
        "What is the stance of the text toward the target (favor, against, "
        "or neutral)? "
        f"Target: {_clip(ex['Target'], 20)}. "
        f"Text: {_clip(ex['Text'])}\nAnswer: "
    )
    a = lambda ex: CSTANCE_LABELS[ex["Stance"]]
    return _to_prompt_answer(train_raw, p, a), _to_prompt_answer(eval_raw, p, a)


def _load_fomc(seed: int, n_train: int, n_eval: int):
    raw = load_dataset("gtfintechlab/fomc_communication", split="train")
    train_raw, eval_raw = _split_train_eval(raw, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    p = lambda ex: (
        "Classify this FOMC sentence as dovish, hawkish, or neutral. "
        f"Sentence: {_clip(ex['sentence'])}\nAnswer: "
    )
    a = lambda ex: FOMC_LABELS[int(ex["label"])]
    return _to_prompt_answer(train_raw, p, a), _to_prompt_answer(eval_raw, p, a)


def _load_meetingbank(seed: int, n_train: int, n_eval: int):
    raw = load_dataset("huuuyeah/meetingbank", split="train")
    raw = raw.filter(lambda ex: ex.get("summary") and ex.get("transcript"))
    train_raw, eval_raw = _split_train_eval(raw, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    p = lambda ex: (
        "Summarise the following meeting transcript in a short sentence. "
        f"Transcript: {_clip(ex['transcript'], MAX_PROMPT_WORDS_LONG)}\n"
        "Summary: "
    )
    a = lambda ex: _clip(ex["summary"], MAX_ANSWER_WORDS_GEN)
    return _to_prompt_answer(train_raw, p, a), _to_prompt_answer(eval_raw, p, a)


def _load_py150(seed: int, n_train: int, n_eval: int):
    raw = load_dataset(
        "angie-chen55/python-github-code", split="train", streaming=True
    )
    keep = []
    # Pre-filter from the stream: take files long enough to split.
    target = (n_train + n_eval) * 3
    for ex in raw:
        code = ex.get("code") or ""
        if len(code) < 400:
            continue
        keep.append({"code": code})
        if len(keep) >= target:
            break
    materialised = Dataset.from_list(keep)
    train_raw, eval_raw = _split_train_eval(materialised, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    def _split_code(code: str):
        # Take ~200 chars as prompt, next ~80 chars as answer.
        head = code[:240].rstrip()
        tail = code[240:340].lstrip()
        return head, tail

    def _prompt(ex):
        head, _ = _split_code(ex["code"])
        return (
            "Continue this Python code snippet with the next few lines. "
            f"Code so far:\n{head}\nContinuation: "
        )

    def _answer(ex):
        _, tail = _split_code(ex["code"])
        return _clip(tail, MAX_ANSWER_WORDS_GEN)

    return _to_prompt_answer(train_raw, _prompt, _answer), \
           _to_prompt_answer(eval_raw, _prompt, _answer)


def _load_scienceqa(seed: int, n_train: int, n_eval: int):
    raw = load_dataset("derek-thomas/ScienceQA", split="train")
    # TRACE uses the text-only ScienceQA subset (no image dependence).
    raw = raw.filter(lambda ex: ex.get("image") is None and len(ex.get("choices", [])) >= 2)
    train_raw, eval_raw = _split_train_eval(raw, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    letters = ["A", "B", "C", "D", "E"]

    def _prompt(ex):
        choices = ex["choices"]
        labelled = ". ".join(f"{letters[i]}) {c}" for i, c in enumerate(choices[:5]))
        hint = ex.get("hint") or ""
        hint_block = f" Context: {_clip(hint, 40)}." if hint else ""
        return (
            "Answer this multiple-choice question with a single letter "
            f"(A-E).{hint_block} Question: {_clip(ex['question'], 60)}. "
            f"Choices: {labelled}\nAnswer: "
        )

    def _answer(ex):
        idx = int(ex["answer"])
        return letters[min(idx, 4)]

    return _to_prompt_answer(train_raw, _prompt, _answer), \
           _to_prompt_answer(eval_raw, _prompt, _answer)


def _load_numglue(seed: int, n_train: int, n_eval: int, type_cfg: str):
    raw = load_dataset("tasksource/num-glue", type_cfg, split="train")
    train_raw, eval_raw = _split_train_eval(raw, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    def _prompt(ex):
        q = _clip(ex["question"], 80)
        passage = ex.get("passage") or "None"
        if passage and passage != "None":
            return (
                "Read the passage and answer the numerical question with a single number. "
                f"Passage: {_clip(passage, 100)}. "
                f"Question: {q}\nAnswer: "
            )
        return (
            "Answer this numerical question with a single number. "
            f"Question: {q}\nAnswer: "
        )

    def _answer(ex):
        ans = str(ex.get("answer", "")).strip()
        # Keep numbers short; drop trailing text after the first whitespace.
        return ans.split()[0] if ans else "0"

    return _to_prompt_answer(train_raw, _prompt, _answer), \
           _to_prompt_answer(eval_raw, _prompt, _answer)


def _load_numglue_cm(seed: int, n_train: int, n_eval: int):
    return _load_numglue(seed, n_train, n_eval, type_cfg="type_4")


def _load_numglue_ds(seed: int, n_train: int, n_eval: int):
    return _load_numglue(seed, n_train, n_eval, type_cfg="type_5")


def _load_20minuten(seed: int, n_train: int, n_eval: int):
    raw = load_dataset("giuliadc/20minuten", split="train")
    raw = raw.filter(lambda ex: ex.get("reference-summary") and ex.get("text"))
    train_raw, eval_raw = _split_train_eval(raw, n_eval, seed)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))

    p = lambda ex: (
        "Fasse den folgenden deutschen Artikel in einem kurzen Satz zusammen. "
        f"Artikel: {_clip(ex['text'], MAX_PROMPT_WORDS_LONG)}\n"
        "Zusammenfassung: "
    )
    a = lambda ex: _clip(ex["reference-summary"], MAX_ANSWER_WORDS_GEN)
    return _to_prompt_answer(train_raw, p, a), _to_prompt_answer(eval_raw, p, a)


# ---------------------------------------------------------------------------
# Task suites.
# ---------------------------------------------------------------------------

# Canonical TRACE task order (per the original paper's curriculum).
TRACE_SPECS: List[Tuple[str, Callable]] = [
    ("cstance", _load_cstance),
    ("fomc", _load_fomc),
    ("meetingbank", _load_meetingbank),
    ("py150", _load_py150),
    ("scienceqa", _load_scienceqa),
    ("numglue_cm", _load_numglue_cm),
    ("numglue_ds", _load_numglue_ds),
    ("twenty_minuten", _load_20minuten),
]


def make_trace_tasks(
    num_per_task: int = 500,
    num_eval_per_task: int = 80,
    seed: int = 0,
    n_tasks: int = 8,
    task_order: int = 1,
) -> List[Task]:
    """Build the TRACE-8 CL stream.

    task_order matches the convention in tasks_cls.py: 1 = canonical order,
    2 / 3 = deterministic shuffles seeded by 1000 + task_order.
    """
    specs = list(TRACE_SPECS)
    if task_order == 1:
        pass
    elif task_order in (2, 3):
        import random as _random
        _random.Random(1000 + task_order).shuffle(specs)
    else:
        raise ValueError(f"task_order must be 1, 2, or 3; got {task_order}")

    specs = specs[:n_tasks]

    tasks: List[Task] = []
    for i, (name, loader) in enumerate(specs):
        train_ds, eval_ds = loader(seed + i, num_per_task, num_eval_per_task)
        tasks.append(Task(id=i, name=name, train_ds=train_ds, eval_ds=eval_ds))
    return tasks
