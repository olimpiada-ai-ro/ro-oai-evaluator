import hashlib
import io
import zipfile

import numpy as np
import pytest

from app.evaluator.providers.remote_url import RemoteURLProvider
from app.evaluator.services.problem_proposal_validation import (
    ProblemProposalAssetValidationError,
    ProblemProposalAssetValidator,
)


def _asset(asset_id: str, role: str, name: str, content: bytes):
    return {
        "asset_id": asset_id,
        "role": role,
        "url": f"https://assets.example/{asset_id}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "content_type": "application/octet-stream",
        "file_name": name,
    }


def _inventory():
    contents = {
        "dataset": b"id,feature\n1,0.5\n",
        "truth": b"id,target\n1,1\n",
        "evaluator": (
            b"def compute_scores(predictions_df, ground_truth_df):\n"
            b"    return 100.0, 1.0, 100.0, 1.0\n"
        ),
        "sample": b"id,prediction\n1,0\n",
    }
    assets = [
        _asset("dataset", "PUBLIC_DATASET", "dataset.csv", contents["dataset"]),
        _asset("truth", "GROUND_TRUTH", "ground_truth.csv", contents["truth"]),
        _asset("evaluator", "EVALUATOR_SCRIPT", "evaluator.py", contents["evaluator"]),
        _asset("sample", "SAMPLE_SUBMISSION", "sample.csv", contents["sample"]),
    ]
    return contents, assets


def _numpy_payload(extension: str, values) -> bytes:
    buffer = io.BytesIO()
    array = np.asarray(values)
    if extension == "npy":
        np.save(buffer, array)
    else:
        np.savez(buffer, predictions=array)
    return buffer.getvalue()


def _message(assets):
    by_role = {asset["role"]: asset for asset in assets}
    return {
        "proposal_assets": assets,
        "datasource_provider": "remote_url",
        "dataset_path": by_role["GROUND_TRUTH"]["url"],
        "predictions_path": by_role["SAMPLE_SUBMISSION"]["url"],
        "prediction_format": "csv",
        "evaluation_script_path": by_role["EVALUATOR_SCRIPT"]["url"],
    }


@pytest.mark.asyncio
async def test_proposal_assets_are_downloaded_hashed_and_inspected(monkeypatch):
    contents, assets = _inventory()

    async def fetch_dataset(_self, url):
        return contents[url.rsplit("/", 1)[-1]]

    async def close(_self):
        return None

    monkeypatch.setattr(RemoteURLProvider, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(RemoteURLProvider, "close", close)

    checks = await ProblemProposalAssetValidator().validate_message(_message(assets))

    assert len(checks) == 5
    assert all(check["passed"] for check in checks)


@pytest.mark.asyncio
async def test_sample_zip_with_root_jsonl_is_accepted(monkeypatch):
    contents, assets = _inventory()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "predictions.jsonl",
            b'{"id": 1, "prediction": 0}\n',
        )
    contents["sample"] = archive_buffer.getvalue()
    assets[3] = _asset(
        "sample",
        "SAMPLE_SUBMISSION",
        "sample.zip",
        contents["sample"],
    )

    async def fetch_dataset(_self, url):
        return contents[url.rsplit("/", 1)[-1]]

    async def close(_self):
        return None

    monkeypatch.setattr(RemoteURLProvider, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(RemoteURLProvider, "close", close)

    message = _message(assets)
    message["prediction_format"] = "zip"
    checks = await ProblemProposalAssetValidator().validate_message(message)

    assert len(checks) == 5
    assert all(check["passed"] for check in checks)


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", ["npy", "npz"])
async def test_numpy_sample_and_reference_submissions_are_accepted(
    monkeypatch, extension
):
    contents, assets = _inventory()
    contents["sample"] = _numpy_payload(extension, [0])
    contents["reference"] = _numpy_payload(extension, [1])
    assets[3] = _asset(
        "sample",
        "SAMPLE_SUBMISSION",
        f"sample.{extension}",
        contents["sample"],
    )
    assets.append(
        _asset(
            "reference",
            "REFERENCE_SUBMISSION",
            f"reference.{extension}",
            contents["reference"],
        )
    )

    async def fetch_dataset(_self, url):
        return contents[url.rsplit("/", 1)[-1]]

    async def close(_self):
        return None

    monkeypatch.setattr(RemoteURLProvider, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(RemoteURLProvider, "close", close)

    message = _message(assets)
    message["prediction_format"] = extension
    checks = await ProblemProposalAssetValidator().validate_message(message)

    assert len(checks) == 6
    assert all(check["passed"] for check in checks)


@pytest.mark.asyncio
async def test_invalid_npz_sample_is_rejected_during_preflight(monkeypatch):
    contents, assets = _inventory()
    archive_buffer = io.BytesIO()
    np.savez(
        archive_buffer,
        first=np.array([0]),
        second=np.array([1]),
    )
    contents["sample"] = archive_buffer.getvalue()
    assets[3] = _asset(
        "sample",
        "SAMPLE_SUBMISSION",
        "sample.npz",
        contents["sample"],
    )

    async def fetch_dataset(_self, url):
        return contents[url.rsplit("/", 1)[-1]]

    async def close(_self):
        return None

    monkeypatch.setattr(RemoteURLProvider, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(RemoteURLProvider, "close", close)

    message = _message(assets)
    message["prediction_format"] = "npz"
    with pytest.raises(
        ProblemProposalAssetValidationError,
        match="not valid NPZ: .*exactly one array",
    ):
        await ProblemProposalAssetValidator().validate_message(message)


@pytest.mark.asyncio
async def test_proposal_asset_hash_mismatch_fails_closed(monkeypatch):
    contents, assets = _inventory()
    assets[0]["sha256"] = "0" * 64

    async def fetch_dataset(_self, url):
        return contents[url.rsplit("/", 1)[-1]]

    async def close(_self):
        return None

    monkeypatch.setattr(RemoteURLProvider, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(RemoteURLProvider, "close", close)

    with pytest.raises(
        ProblemProposalAssetValidationError, match="SHA-256"
    ) as exc_info:
        await ProblemProposalAssetValidator().validate_message(_message(assets))

    assert exc_info.value.checks[-1]["passed"] is False


@pytest.mark.asyncio
async def test_proposal_archives_reject_pickle_payloads(monkeypatch):
    contents, assets = _inventory()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("unsafe.pkl", b"not-even-a-real-pickle")
    contents["dataset"] = archive_buffer.getvalue()
    assets[0] = _asset(
        "dataset",
        "PUBLIC_DATASET",
        "dataset.zip",
        contents["dataset"],
    )

    async def fetch_dataset(_self, url):
        return contents[url.rsplit("/", 1)[-1]]

    async def close(_self):
        return None

    monkeypatch.setattr(RemoteURLProvider, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(RemoteURLProvider, "close", close)

    with pytest.raises(ProblemProposalAssetValidationError, match="unsafe serialized"):
        await ProblemProposalAssetValidator().validate_message(_message(assets))


@pytest.mark.asyncio
async def test_proposal_evaluation_paths_must_match_preflighted_assets():
    _contents, assets = _inventory()
    message = _message(assets)
    message["evaluation_script_path"] = "https://attacker.example/evaluator.py"

    with pytest.raises(
        ProblemProposalAssetValidationError, match="immutable proposal asset"
    ):
        await ProblemProposalAssetValidator().validate_message(message)


@pytest.mark.asyncio
async def test_ground_truth_cannot_be_duplicated_in_a_publishable_asset():
    _contents, assets = _inventory()
    assets[0]["sha256"] = assets[1]["sha256"]

    with pytest.raises(
        ProblemProposalAssetValidationError, match="Ground truth content"
    ):
        await ProblemProposalAssetValidator().validate_message(_message(assets))
