if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
from typing import Iterator

TRAIN_PHASE = "SELECTING_HAND"
SHOP_TRAIN_PHASES = {"SHOP", "SMODS_BOOSTER_OPENED"}
SCHEMA_VERSION = "record_v1"


REQUIRED_FIELDS = {
    "timestamp",
    "episode_id",
    "step_id",
    "instance_id",
    "base_url",
    "phase",
    "done",
    "hand_size",
    "legal_action_ids",
    "expert_action_id",
    "macro_action",
    "reward",
    "reward_info",
    "features",
}


DISTILL_REQUIRED_FIELDS = {"schema", "phase", "state_features", "teacher_topk", "metadata"}


def validate_record(record: dict) -> None:
    schema = str(record.get("schema") or "")
    if schema == "distill_v1":
        missing = DISTILL_REQUIRED_FIELDS - set(record.keys())
        if missing:
            raise ValueError(f"distill record missing required fields: {sorted(missing)}")
        return

    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise ValueError(f"record missing required fields: {sorted(missing)}")

    phase = str(record.get("phase"))
    hand_size = int(record.get("hand_size") or 0)
    legal_action_ids = record.get("legal_action_ids")
    expert_action_id = record.get("expert_action_id")
    macro_action = record.get("macro_action")
    shop_legal_action_ids = record.get("shop_legal_action_ids")
    shop_expert_action_id = record.get("shop_expert_action_id")
    shop_features = record.get("shop_features")

    if not isinstance(legal_action_ids, list):
        raise ValueError("legal_action_ids must be list")

    if phase == TRAIN_PHASE:
        if hand_size <= 0:
            raise ValueError("SELECTING_HAND record requires hand_size > 0")
        if expert_action_id is None:
            raise ValueError("SELECTING_HAND record requires expert_action_id")
        if shop_expert_action_id is not None:
            raise ValueError("SELECTING_HAND record shop_expert_action_id must be null")
    elif phase in SHOP_TRAIN_PHASES:
        if macro_action is None:
            raise ValueError("SHOP phase must include macro_action")
        if shop_legal_action_ids is not None and not isinstance(shop_legal_action_ids, list):
            raise ValueError("shop_legal_action_ids must be list when present")
        if shop_expert_action_id is not None and not isinstance(shop_expert_action_id, int):
            raise ValueError("shop_expert_action_id must be int or null")
        if shop_features is not None and not isinstance(shop_features, dict):
            raise ValueError("shop_features must be dict or null")
    else:
        if expert_action_id is not None and not isinstance(expert_action_id, int):
            raise ValueError("non-hand phase expert_action_id must be int or null")
        if macro_action is None:
            raise ValueError("non-hand phase must include macro_action")


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = None

    def __enter__(self):
        self._fp = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write_record(self, record: dict) -> None:
        validate_record(record)
        assert self._fp is not None
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")



def write_jsonl(path: str | Path, records: list[dict]) -> None:
    with JsonlWriter(path) as writer:
        for record in records:
            writer.write_record(record)



def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            yield record



def iter_train_samples(path: str | Path) -> Iterator[dict]:
    for record in read_jsonl(path):
        if str(record.get("phase")) != TRAIN_PHASE:
            continue
        if record.get("expert_action_id") is None:
            continue
        yield record


def iter_shop_samples(path: str | Path) -> Iterator[dict]:
    for record in read_jsonl(path):
        phase = str(record.get("phase"))
        if phase not in SHOP_TRAIN_PHASES:
            continue
        if record.get("shop_expert_action_id") is None:
            continue
        yield record



def summarize_dataset(path: str | Path) -> dict:
    total = 0
    hand = 0
    shop = 0
    detected_schema = SCHEMA_VERSION
    for r in read_jsonl(path):
        total += 1
        rec_schema = str(r.get("schema") or "")
        if rec_schema == "distill_v1":
            detected_schema = "distill_v1"
            phase = str(r.get("phase") or "")
            if phase == "HAND":
                hand += 1
            elif phase == "SHOP":
                shop += 1
        else:
            if str(r.get("phase")) == TRAIN_PHASE:
                hand += 1
            if str(r.get("phase")) in SHOP_TRAIN_PHASES:
                shop += 1
    return {
        "schema": detected_schema,
        "total_records": total,
        "hand_records": hand,
        "shop_records": shop,
    }

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Dataset utilities for trainer jsonl.")
    parser.add_argument("--path", required=True, help="jsonl dataset path")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    parser.add_argument("--validate", action="store_true", help="Validate all records")
    args = parser.parse_args()

    if args.summary:
        print(json.dumps(summarize_dataset(args.path), ensure_ascii=False, indent=2))

    if args.validate:
        count = 0
        for rec in read_jsonl(args.path):
            validate_record(rec)
            count += 1
        print(f"validated_records={count}")

    if not args.summary and not args.validate:
        print("No action selected. Use --summary and/or --validate.")

