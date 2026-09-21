from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import soundfile as sf
from conftest import write_tone
from fastapi.testclient import TestClient

from story_voice_pipeline.config import PipelineConfig
from story_voice_pipeline.control_service.app import create_app
from story_voice_pipeline.control_service.config import ControlSettings

SAMPLE_LIBRARY = Path(__file__).parents[2] / "story_library"


def test_prepare_qwen_reference_download(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app = create_app(
        ControlSettings(
            data_root=data_root,
            story_library_path=SAMPLE_LIBRARY / "library.yaml",
            database_url=f"sqlite:///{tmp_path / 'control.db'}",
            tts_engine="fake",
            pipeline=PipelineConfig(output_root=data_root),
        )
    )
    source = write_tone(tmp_path / "source.wav", duration=1.0, sample_rate=16_000).read_bytes()

    with TestClient(app) as client:
        assert client.get("/references/prepare").status_code == 200
        no_consent = client.post(
            "/references/prepare",
            data={"transcript": "A matching transcript.", "language": "en"},
            files={"audio": ("source.wav", source, "audio/wav")},
        )
        assert no_consent.status_code == 422
        assert "permission" in no_consent.text

        response = client.post(
            "/references/prepare",
            data={
                "transcript": "A matching transcript.",
                "language": "en",
                "consent_confirmed": "true",
            },
            files={"audio": ("source.wav", source, "audio/wav")},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert "qwen-reference.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        assert set(bundle.namelist()) == {"reference.wav", "reference.txt"}
        assert bundle.read("reference.txt").decode() == "A matching transcript.\n"
        samples, sample_rate = sf.read(io.BytesIO(bundle.read("reference.wav")), always_2d=True)
        assert sample_rate == 24_000
        assert samples.shape[1] == 1

    assert not list((data_root / "reference_exports").glob("reference_*"))


def test_control_service_full_fake_lifecycle(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    shutil.copytree(SAMPLE_LIBRARY, library_root)
    data_root = tmp_path / "data"
    settings = ControlSettings(
        data_root=data_root,
        story_library_path=library_root / "library.yaml",
        database_url=f"sqlite:///{tmp_path / 'control.db'}",
        tts_engine="fake",
        pipeline=PipelineConfig(output_root=data_root),
    )
    app = create_app(settings)
    reference = write_tone(tmp_path / "reference.wav", duration=1.0).read_bytes()

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/voices/new").status_code == 200
        assert client.get("/jobs/new").status_code == 200
        languages = client.get("/api/v1/languages").json()
        assert [item["code"] for item in languages] == [
            "en",
            "zh",
            "es",
            "fr",
            "de",
            "ja",
            "ko",
            "pt",
        ]
        assert {item["code"] for item in languages if item["available"]} == {
            "en",
            "zh",
            "ja",
            "ko",
            "de",
            "pt",
            "es",
        }

        no_consent = client.post(
            "/api/v1/voices",
            data={
                "name": "No consent",
                "english_transcript": "Exact reference transcript.",
            },
            files={"english_audio": ("reference.wav", reference, "audio/wav")},
        )
        assert no_consent.status_code == 422

        created_voice = client.post(
            "/api/v1/voices",
            data={
                "name": "Test caregiver",
                "english_transcript": "Exact reference transcript.",
                "consent_confirmed": "true",
                "notes": "Integration fixture",
            },
            files={"english_audio": ("reference.wav", reference, "audio/wav")},
        )
        assert created_voice.status_code == 201, created_voice.text
        voice = created_voice.json()
        assert voice["consent_confirmed"] is True
        assert voice["has_mandarin_reference"] is False
        assert (
            client.get(f"/api/v1/voices/{voice['id']}").json()["english_reference_transcript"]
            == "Exact reference transcript."
        )
        assert client.get("/api/v1/devices/pi_1/latest-manifest").status_code == 404

        created_job = client.post(
            "/api/v1/jobs",
            json={
                "voice_id": voice["id"],
                "story_ids": ["forest"],
                "languages": ["en"],
            },
        )
        assert created_job.status_code == 202, created_job.text
        job_id = created_job.json()["id"]
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "READY_FOR_REVIEW", job
        assert job["completed_chunks"] == job["total_chunks"]
        assert job["total_chunks"] > 0
        assert job["progress"] == 1.0
        pack_id = job["pack_id"]

        pack = client.get(f"/api/v1/story-packs/{pack_id}").json()
        assert pack["status"] == "READY_FOR_REVIEW"
        assert len(pack["audio"]) == 2
        assert {audio["audio_type"] for audio in pack["audio"]} == {"story", "word"}
        assert client.get(f"/packs/{pack_id}/review").status_code == 200

        audio = pack["audio"][0]
        review_file = client.get(f"/review-files/{pack_id}/{audio['path']}")
        assert review_file.status_code == 200
        assert hashlib.sha256(review_file.content).hexdigest() == audio["sha256"]
        assert client.get(f"/api/v1/packs/{pack_id}/files/{audio['path']}").status_code == 404

        approval = client.post(f"/api/v1/story-packs/{pack_id}/approve")
        assert approval.status_code == 200
        assert approval.json()["status"] == "APPROVED"
        assert approval.json()["approved_at"] is not None
        assert client.post(f"/api/v1/story-packs/{pack_id}/approve").status_code == 409

        latest = client.get("/api/v1/devices/pi_1/latest-manifest")
        assert latest.status_code == 200
        assert latest.json()["status"] == "approved"
        assert latest.json()["pack_id"] == pack_id
        published_file = client.get(f"/api/v1/packs/{pack_id}/files/{audio['path']}")
        assert published_file.status_code == 200
        assert hashlib.sha256(published_file.content).hexdigest() == audio["sha256"]


def test_job_rejects_language_without_complete_library_translation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app = create_app(
        ControlSettings(
            data_root=data_root,
            story_library_path=SAMPLE_LIBRARY / "library.yaml",
            database_url=f"sqlite:///{tmp_path / 'control.db'}",
            tts_engine="fake",
            pipeline=PipelineConfig(output_root=data_root),
        )
    )
    reference = write_tone(tmp_path / "reference.wav").read_bytes()
    with TestClient(app) as client:
        voice = client.post(
            "/api/v1/voices",
            data={
                "name": "Language fixture",
                "english_transcript": "Exact transcript.",
                "consent_confirmed": "true",
            },
            files={"english_audio": ("reference.wav", reference, "audio/wav")},
        ).json()
        response = client.post(
            "/api/v1/jobs",
            json={"voice_id": voice["id"], "story_ids": ["forest"], "languages": ["fr"]},
        )

    assert response.status_code == 422
    assert "no complete translation for: fr" in response.json()["detail"]


def test_pack_rejection_prevents_publication(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    shutil.copytree(SAMPLE_LIBRARY, library_root)
    data_root = tmp_path / "data"
    app = create_app(
        ControlSettings(
            data_root=data_root,
            story_library_path=library_root / "library.yaml",
            database_url=f"sqlite:///{tmp_path / 'control.db'}",
            tts_engine="fake",
            pipeline=PipelineConfig(output_root=data_root),
        )
    )
    reference = write_tone(tmp_path / "reference.wav").read_bytes()
    with TestClient(app) as client:
        voice = client.post(
            "/api/v1/voices",
            data={
                "name": "Reviewer",
                "english_transcript": "Exact transcript.",
                "consent_confirmed": "true",
            },
            files={"english_audio": ("reference.wav", reference, "audio/wav")},
        ).json()
        response = client.post(
            "/api/v1/jobs",
            json={"voice_id": voice["id"], "story_ids": ["moon"]},
        )
        job = client.get(f"/api/v1/jobs/{response.json()['id']}").json()
        pack_id = job["pack_id"]
        rejected = client.post(f"/api/v1/story-packs/{pack_id}/reject")
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "REJECTED"
        assert client.get("/api/v1/devices/pi_1/latest-manifest").status_code == 404


def test_failed_job_can_retry_after_reference_is_restored(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    shutil.copytree(SAMPLE_LIBRARY, library_root)
    data_root = tmp_path / "data"
    app = create_app(
        ControlSettings(
            data_root=data_root,
            story_library_path=library_root / "library.yaml",
            database_url=f"sqlite:///{tmp_path / 'control.db'}",
            tts_engine="fake",
            pipeline=PipelineConfig(output_root=data_root),
        )
    )
    reference_path = write_tone(tmp_path / "reference.wav")
    reference = reference_path.read_bytes()
    with TestClient(app) as client:
        voice = client.post(
            "/api/v1/voices",
            data={
                "name": "Retry fixture",
                "english_transcript": "Exact transcript.",
                "consent_confirmed": "true",
            },
            files={"english_audio": ("reference.wav", reference, "audio/wav")},
        ).json()
        normalized = data_root / "references" / voice["id"] / "v1" / "en" / "reference.wav"
        normalized.unlink()
        response = client.post(
            "/api/v1/jobs",
            json={
                "voice_id": voice["id"],
                "story_ids": ["bear"],
                "languages": ["en"],
            },
        )
        job_id = response.json()["id"]
        failed = client.get(f"/api/v1/jobs/{job_id}").json()
        assert failed["status"] == "FAILED"
        assert "does not exist" in failed["error_message"]

        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(reference)
        retry = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert retry.status_code == 202
        completed = client.get(f"/api/v1/jobs/{job_id}").json()
        assert completed["status"] == "READY_FOR_REVIEW"
        assert completed["pack_id"] is not None
