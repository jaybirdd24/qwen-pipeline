from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import write_tone
from fastapi.testclient import TestClient
from sqlalchemy import select

from story_voice_pipeline.config import PipelineConfig
from story_voice_pipeline.control_service.app import create_app
from story_voice_pipeline.control_service.config import ControlSettings
from story_voice_pipeline.control_service.database import Database
from story_voice_pipeline.control_service.models import Voice, VoiceReference
from story_voice_pipeline.control_service.passages import RECORDING_PASSAGES
from story_voice_pipeline.manifest import load_manifest

LIBRARY = Path(__file__).parents[2] / "story_library/library.yaml"


@pytest.fixture
def service(tmp_path):
    settings = ControlSettings(
        data_root=tmp_path / "data",
        story_library_path=LIBRARY,
        database_url=f"sqlite:///{tmp_path / 'control.db'}",
        tts_engine="fake",
        pipeline=PipelineConfig(output_root=tmp_path / "data"),
    )
    app = create_app(settings)
    audio = write_tone(tmp_path / "recording.wav").read_bytes()
    with TestClient(app) as client:
        yield client, app, audio


def upload_voice(client, audio, codes, endpoint="/api/v1/voices", **extra):
    return client.post(
        endpoint,
        data={"name": "Participant 04", "consent_confirmed": "true", "languages": codes, **extra},
        files=[("audios", ("phone.wav", audio, "audio/wav")) for _ in codes],
        follow_redirects=False,
    )


@pytest.mark.parametrize("codes", [["en", "es"], ["es", "zh"], ["en", "es", "zh"]])
def test_generic_voice_generates_with_matching_references(service, codes):
    client, app, audio = service
    response = upload_voice(client, audio, codes)
    assert response.status_code == 201, response.text
    voice = response.json()
    assert set(voice["languages"]) == set(codes)
    root = app.state.settings.data_root
    for code in codes:
        stored = root / "references" / voice["id"] / "v1" / code
        assert (stored / "reference.txt").read_text().strip() == RECORDING_PASSAGES[code]
        assert (
            voice["references"][code]["audio_sha256"]
            == hashlib.sha256((stored / "reference.wav").read_bytes()).hexdigest()
        )
    assert client.get("/").status_code == 200
    page = client.get(f"/jobs/new?voice_id={voice['id']}")
    assert page.status_code == 200
    assert f'value="{voice["id"]}" selected' in page.text
    assert 'name="language"' not in page.text
    response = client.post("/api/v1/jobs", json={"voice_id": voice["id"], "story_ids": ["forest"]})
    assert response.status_code == 202, response.text
    job = client.get(f"/api/v1/jobs/{response.json()['id']}").json()
    assert job["status"] == "READY_FOR_REVIEW", job
    assert set(job["languages"]) == set(codes)
    manifest = load_manifest(root / "packs" / job["pack_id"] / "manifest.json")
    for story in manifest["stories"]:
        for group in (story["audio"], story["word_audio"]):
            for code in codes:
                assert group[code]["reference_language"] == code


def test_missing_reference_and_empty_selection_rejected(service):
    client, _, audio = service
    voice = upload_voice(client, audio, ["en"]).json()
    for fields, message in [
        ({"languages": ["es"]}, "Missing language reference: es"),
        ({"languages": ["zh"]}, "Missing language reference: zh"),
        ({"languages": []}, "Invalid language selection"),
        ({"story_ids": []}, "Invalid story selection"),
    ]:
        response = client.post("/api/v1/jobs", json={"voice_id": voice["id"], **fields})
        assert response.status_code == 422
        assert response.json()["detail"] == message


def test_fixed_passages_form_and_validation(service):
    client, app, audio = service
    page = client.get("/voices/new").text
    assert 'name="english_transcript"' not in page
    assert 'name="audios"' in page
    checked = client.post(
        "/api/v1/references/validate",
        data={"language": "es"},
        files={"audio": ("phone.wav", audio, "audio/wav")},
    )
    assert checked.status_code == 200
    assert checked.json()["duration_seconds"] > 0
    assert not list((app.state.settings.data_root / "uploads").iterdir())
    created = upload_voice(
        client,
        audio,
        ["en", "es"],
        "/voices/new",
        transcript="Client cannot override the passage",
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/jobs/new?voice_id=")
    voice = client.get("/api/v1/voices").json()[0]
    detail = client.get(f"/api/v1/voices/{voice['id']}").json()
    assert detail["references"]["en"]["transcript"] == RECORDING_PASSAGES["en"]
    # Browser submissions derive languages from the participant, even if tampered with.
    job = client.post(
        "/jobs/new",
        data={"voice_id": voice["id"], "story": "moon", "language": "de"},
        follow_redirects=False,
    )
    assert job.status_code == 303
    result = client.get("/api/v1" + job.headers["location"]).json()
    assert result["languages"] == ["en", "es"]


def test_invalid_uploads_leave_no_partial_voice(service):
    client, app, audio = service
    for codes in (["en", "en"], ["xx", "es"]):
        assert upload_voice(client, audio, codes).status_code == 422
    assert upload_voice(client, audio, ["en"], "/voices/new").status_code == 422
    response = client.post(
        "/api/v1/voices",
        data={"name": "Broken", "consent_confirmed": "true", "languages": ["en", "es"]},
        files=[
            ("audios", ("ok.wav", audio, "audio/wav")),
            ("audios", ("empty.wav", b"", "audio/wav")),
        ],
    )
    assert response.status_code == 422
    assert client.get("/api/v1/voices").json() == []
    assert not list((app.state.settings.data_root / "references").glob("voice_*"))


def test_existing_voice_migration_is_idempotent(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'legacy.db'}")
    db.create_schema()
    audio = write_tone(tmp_path / "en.wav")
    with db.sessions() as session:
        session.add(
            Voice(
                id="legacy",
                name="Existing participant",
                english_reference_path="en.wav",
                english_reference_transcript="Original",
                english_reference_duration=1.0,
                consent_confirmed=True,
            )
        )
        session.commit()
    VoiceReference.__table__.drop(db.engine)
    db.create_schema(tmp_path)
    db.create_schema(tmp_path)
    with db.sessions() as session:
        references = session.scalars(select(VoiceReference)).all()
        assert len(references) == 1
        assert references[0].transcript == "Original"
        assert references[0].audio_sha256 == hashlib.sha256(audio.read_bytes()).hexdigest()
    db.dispose()
