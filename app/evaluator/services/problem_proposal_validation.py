"""Security preflight for participant-authored problem proposal assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluator.parsers.prediction_parser import (
    PredictionParser,
    PredictionParsingError,
)
from app.evaluator.parsers.zip_extractor import ZipExtractor
from app.evaluator.providers.remote_url import RemoteURLProvider


class ProblemProposalAssetValidationError(Exception):
    """Raised when a quarantined proposal asset fails security validation."""

    def __init__(self, message: str, checks: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.checks = checks or []


class ProblemProposalAsset(BaseModel):
    asset_id: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=50)
    url: str = Field(min_length=1, max_length=8_192)
    sha256: str
    size_bytes: int = Field(gt=0)
    content_type: str = Field(min_length=1, max_length=200)
    file_name: str = Field(min_length=1, max_length=255)

    model_config = ConfigDict(extra="forbid")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("sha256 must be a hexadecimal SHA-256 digest")
        return normalized

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        if "\x00" in value or "/" in value or "\\" in value:
            raise ValueError("file_name must not contain a path")
        return value


class ProblemProposalAssetValidator:
    """Download, hash, and structurally inspect quarantined proposal files."""

    MAX_ASSETS = 20
    MAX_TOTAL_BYTES = 750 * 1024 * 1024
    REQUIRED_ROLES = {
        "PUBLIC_DATASET",
        "GROUND_TRUTH",
        "EVALUATOR_SCRIPT",
        "SAMPLE_SUBMISSION",
    }
    ROLE_ALIASES = {
        "DATASET": "PUBLIC_DATASET",
        "PUBLIC_DATASET": "PUBLIC_DATASET",
        "GROUND_TRUTH": "GROUND_TRUTH",
        "EVALUATOR": "EVALUATOR_SCRIPT",
        "EVALUATOR_SCRIPT": "EVALUATOR_SCRIPT",
        "SAMPLE": "SAMPLE_SUBMISSION",
        "SAMPLE_SUBMISSION": "SAMPLE_SUBMISSION",
        "REFERENCE": "REFERENCE_SUBMISSION",
        "REFERENCE_SUBMISSION": "REFERENCE_SUBMISSION",
        "EXTRA": "SUPPORTING_FILE",
        "SUPPORTING": "SUPPORTING_FILE",
        "SUPPORTING_FILE": "SUPPORTING_FILE",
    }
    MAX_ROLE_BYTES = {
        "PUBLIC_DATASET": 500 * 1024 * 1024,
        "GROUND_TRUTH": 100 * 1024 * 1024,
        "EVALUATOR_SCRIPT": 512 * 1024,
        "SAMPLE_SUBMISSION": 25 * 1024 * 1024,
        "REFERENCE_SUBMISSION": 100 * 1024 * 1024,
        "SUPPORTING_FILE": 25 * 1024 * 1024,
    }
    ALLOWED_EXTENSIONS = {
        "PUBLIC_DATASET": {".zip", ".csv", ".txt", ".json"},
        "GROUND_TRUTH": {".zip", ".csv", ".txt", ".json"},
        "EVALUATOR_SCRIPT": {".py"},
        "SAMPLE_SUBMISSION": {".zip", ".csv", ".txt", ".json", ".npy", ".npz"},
        "REFERENCE_SUBMISSION": {".zip", ".csv", ".txt", ".json", ".npy", ".npz"},
        "SUPPORTING_FILE": {
            ".zip",
            ".pdf",
            ".md",
            ".txt",
            ".csv",
            ".json",
            ".png",
            ".jpg",
            ".jpeg",
        },
    }
    UNSAFE_ARCHIVE_EXTENSIONS = {".pkl", ".pickle", ".joblib"}

    async def validate_message(
        self, message_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        raw_assets = message_data.get("proposal_assets")
        if not isinstance(raw_assets, list):
            raise ProblemProposalAssetValidationError(
                "Problem proposal validation requires a proposal_assets inventory"
            )
        if not raw_assets or len(raw_assets) > self.MAX_ASSETS:
            raise ProblemProposalAssetValidationError(
                f"Problem proposals must contain between 1 and {self.MAX_ASSETS} assets"
            )

        try:
            assets = [
                ProblemProposalAsset.model_validate(asset) for asset in raw_assets
            ]
        except Exception as exc:
            raise ProblemProposalAssetValidationError(
                "Invalid problem proposal asset inventory"
            ) from exc

        checks: list[dict[str, Any]] = []
        normalized_roles: list[str] = []
        role_counts: dict[str, int] = {}
        asset_ids: set[str] = set()
        total_bytes = 0

        for asset in assets:
            if asset.asset_id in asset_ids:
                raise ProblemProposalAssetValidationError(
                    f"Duplicate proposal asset id: {asset.asset_id}", checks
                )
            asset_ids.add(asset.asset_id)

            role = self.ROLE_ALIASES.get(asset.role.upper())
            if role is None:
                raise ProblemProposalAssetValidationError(
                    f"Unsupported proposal asset role: {asset.role}", checks
                )
            normalized_roles.append(role)
            role_counts[role] = role_counts.get(role, 0) + 1
            if role != "SUPPORTING_FILE" and role_counts[role] > 1:
                raise ProblemProposalAssetValidationError(
                    f"Only one active {role} asset is allowed", checks
                )
            total_bytes += asset.size_bytes
            if asset.size_bytes > self.MAX_ROLE_BYTES[role]:
                raise ProblemProposalAssetValidationError(
                    f"{role} exceeds its validation size limit", checks
                )

            extension = Path(asset.file_name).suffix.lower()
            if extension not in self.ALLOWED_EXTENSIONS[role]:
                raise ProblemProposalAssetValidationError(
                    f"{asset.file_name} has an unsupported extension for {role}", checks
                )

        if total_bytes > self.MAX_TOTAL_BYTES:
            raise ProblemProposalAssetValidationError(
                "Problem proposal assets exceed the total validation size limit", checks
            )

        missing_roles = self.REQUIRED_ROLES.difference(normalized_roles)
        if missing_roles:
            raise ProblemProposalAssetValidationError(
                "Missing required proposal assets: " + ", ".join(sorted(missing_roles)),
                checks,
            )

        self._validate_evaluation_binding(
            message_data=message_data,
            assets=assets,
            normalized_roles=normalized_roles,
        )
        checks.append(
            {
                "code": "EVALUATION_BINDING",
                "passed": True,
                "message": (
                    "Evaluation paths are bound to the immutable proposal "
                    "asset inventory"
                ),
            }
        )

        provider = RemoteURLProvider()
        try:
            for asset, role in zip(assets, normalized_roles, strict=True):
                content = await provider.fetch_dataset(asset.url)
                try:
                    self._validate_content(asset, role, content)
                except ProblemProposalAssetValidationError as exc:
                    exc.checks = [
                        *checks,
                        {
                            "code": f"ASSET_{role}_{asset.asset_id}",
                            "passed": False,
                            "message": str(exc),
                        },
                    ]
                    raise
                checks.append(
                    {
                        "code": f"ASSET_{role}_{asset.asset_id}",
                        "passed": True,
                        "message": (
                            f"{asset.file_name}: size, SHA-256, type, and structure passed"
                        ),
                    }
                )
        except ProblemProposalAssetValidationError:
            raise
        except Exception as exc:
            raise ProblemProposalAssetValidationError(
                f"Proposal asset preflight failed: {exc}", checks
            ) from exc
        finally:
            await provider.close()

        return checks

    def _validate_evaluation_binding(
        self,
        *,
        message_data: dict[str, Any],
        assets: list[ProblemProposalAsset],
        normalized_roles: list[str],
    ) -> None:
        """Ensure preflighted assets are exactly the assets later evaluated."""

        by_role = {
            role: asset
            for asset, role in zip(assets, normalized_roles, strict=True)
            if role != "SUPPORTING_FILE"
        }

        if message_data.get("datasource_provider") != "remote_url":
            raise ProblemProposalAssetValidationError(
                "Problem proposals must use the isolated remote_url provider"
            )
        if message_data.get("predictions") not in (None, ""):
            raise ProblemProposalAssetValidationError(
                "Problem proposals must not contain inline prediction data"
            )

        expected_paths = {
            "dataset_path": by_role["GROUND_TRUTH"].url,
            "predictions_path": by_role["SAMPLE_SUBMISSION"].url,
            "evaluation_script_path": by_role["EVALUATOR_SCRIPT"].url,
        }
        for field, expected_value in expected_paths.items():
            if message_data.get(field) != expected_value:
                raise ProblemProposalAssetValidationError(
                    f"{field} must reference the matching immutable proposal asset"
                )

        sample_extension = Path(by_role["SAMPLE_SUBMISSION"].file_name).suffix.lower()
        expected_format = {
            ".csv": "csv",
            ".txt": "txt",
            ".json": "json",
            ".zip": "zip",
            ".npy": "npy",
            ".npz": "npz",
        }.get(sample_extension)
        if message_data.get("prediction_format") != expected_format:
            raise ProblemProposalAssetValidationError(
                "prediction_format does not match the sample submission asset"
            )

        urls = [asset.url for asset in assets]
        if len(urls) != len(set(urls)):
            raise ProblemProposalAssetValidationError(
                "Each proposal asset must use a distinct immutable URL"
            )

        # Publishing an asset byte-for-byte identical to the private ground
        # truth would disclose the answers as soon as the proposal is approved.
        ground_truth_digest = by_role["GROUND_TRUTH"].sha256
        duplicated_ground_truth = [
            role
            for asset, role in zip(assets, normalized_roles, strict=True)
            if role != "GROUND_TRUTH" and asset.sha256 == ground_truth_digest
        ]
        if duplicated_ground_truth:
            raise ProblemProposalAssetValidationError(
                "Ground truth content is duplicated by another proposal asset: "
                + ", ".join(sorted(set(duplicated_ground_truth)))
            )

    def _validate_content(
        self,
        asset: ProblemProposalAsset,
        role: str,
        content: bytes,
    ) -> None:
        if len(content) != asset.size_bytes:
            raise ProblemProposalAssetValidationError(
                f"{asset.file_name} does not match its declared size"
            )
        if hashlib.sha256(content).hexdigest() != asset.sha256:
            raise ProblemProposalAssetValidationError(
                f"{asset.file_name} does not match its declared SHA-256 digest"
            )

        extension = Path(asset.file_name).suffix.lower()
        if extension == ".zip":
            self._validate_zip(asset.file_name, content)
        elif extension == ".py":
            try:
                compile(content.decode("utf-8"), asset.file_name, "exec")
            except (UnicodeDecodeError, SyntaxError) as exc:
                raise ProblemProposalAssetValidationError(
                    f"{asset.file_name} is not valid UTF-8 Python: {exc}"
                ) from exc
        elif extension == ".json" and len(content) <= 25 * 1024 * 1024:
            try:
                json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProblemProposalAssetValidationError(
                    f"{asset.file_name} is not valid JSON: {exc}"
                ) from exc
        elif extension in {".npy", ".npz"}:
            try:
                PredictionParser().parse(content, extension.removeprefix("."))
            except PredictionParsingError as exc:
                raise ProblemProposalAssetValidationError(
                    f"{asset.file_name} is not valid "
                    f"{extension.removeprefix('.').upper()}: {exc}"
                ) from exc
        elif extension == ".pdf" and not content.startswith(b"%PDF-"):
            raise ProblemProposalAssetValidationError(
                f"{asset.file_name} is not a valid PDF"
            )
        elif extension == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ProblemProposalAssetValidationError(
                f"{asset.file_name} is not a valid PNG"
            )
        elif extension in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
            raise ProblemProposalAssetValidationError(
                f"{asset.file_name} is not a valid JPEG"
            )

    def _validate_zip(self, file_name: str, content: bytes) -> None:
        extractor = ZipExtractor()
        extraction_path = None
        try:
            extraction_path = extractor.extract(content)
            for root, _, files in os.walk(extraction_path):
                for name in files:
                    if Path(name).suffix.lower() in self.UNSAFE_ARCHIVE_EXTENSIONS:
                        raise ProblemProposalAssetValidationError(
                            f"{file_name} contains unsafe serialized Python data"
                        )
        except ProblemProposalAssetValidationError:
            raise
        except Exception as exc:
            raise ProblemProposalAssetValidationError(
                f"{file_name} failed secure ZIP inspection: {exc}"
            ) from exc
        finally:
            if extraction_path:
                extractor.cleanup(extraction_path)


problem_proposal_asset_validator = ProblemProposalAssetValidator()
