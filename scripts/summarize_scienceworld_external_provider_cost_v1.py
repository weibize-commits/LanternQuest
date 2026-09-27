from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def estimate_profile_cost(
    profile: dict[str, Any], pricing: dict[str, Any], usd_to_cny: float
) -> dict[str, Any]:
    summaries = profile["summaries"]
    input_tokens = sum(int(item["input_tokens"]) for item in summaries.values())
    output_tokens = sum(int(item["output_tokens"]) for item in summaries.values())
    input_mtok = input_tokens / 1_000_000
    output_mtok = output_tokens / 1_000_000
    low = (
        input_mtok * float(pricing["cached_input_low"])
        + output_mtok * float(pricing["output_low"])
    )
    high = (
        input_mtok * float(pricing["uncached_input_high"])
        + output_mtok * float(pricing["output_high"])
    )
    currency = str(pricing["currency"])
    conversion = usd_to_cny if currency == "USD" else 1.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "currency": currency,
        "estimated_low_native": low,
        "estimated_high_native": high,
        "estimated_low_cny": low * conversion,
        "estimated_high_cny": high * conversion,
        "range_basis": pricing["range_basis"],
        "pricing_source": pricing["source"],
    }


def build_summary(audit: dict[str, Any], price_book: dict[str, Any]) -> dict[str, Any]:
    usd_to_cny = float(price_book["usd_to_cny"])
    estimates: dict[str, Any] = {}
    for profile_id, profile in audit["profiles"].items():
        pricing = price_book["profiles"].get(profile_id)
        if pricing is None:
            raise ValueError(f"pricing is missing for profile {profile_id}")
        estimates[profile_id] = estimate_profile_cost(
            profile, pricing, usd_to_cny
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": audit["experiment_id"],
        "pricing_retrieved_at": price_book["retrieved_at"],
        "usd_to_cny": usd_to_cny,
        "profiles": estimates,
        "total_estimated_low_cny": sum(
            item["estimated_low_cny"] for item in estimates.values()
        ),
        "total_estimated_high_cny": sum(
            item["estimated_high_cny"] for item in estimates.values()
        ),
        "limitations": price_book["limitations"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "profile_id",
        "input_tokens",
        "output_tokens",
        "currency",
        "estimated_low_native",
        "estimated_high_native",
        "estimated_low_cny",
        "estimated_high_cny",
        "range_basis",
        "pricing_source",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for profile_id, item in payload["profiles"].items():
            writer.writerow({"profile_id": profile_id, **item})
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/external_provider_v1_audit.json"
        ),
    )
    parser.add_argument(
        "--pricing",
        type=Path,
        default=Path("configs/scienceworld_external_provider_pricing_20260924.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/external_provider_v1_cost_summary.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/external_provider_v1_cost_summary.csv"
        ),
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    price_book = json.loads(args.pricing.read_text(encoding="utf-8-sig"))
    payload = build_summary(audit, price_book)
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
