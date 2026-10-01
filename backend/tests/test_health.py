"""Tests for the health endpoint."""

from __future__ import annotations


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

    def test_health_reports_model_loaded(self, client):
        response = client.get("/api/health")
        body = response.json()
        assert body["model_loaded"] is True
        assert body["model_version"]

    def test_health_reports_api_version(self, client):
        response = client.get("/api/health")
        body = response.json()
        assert body["api_version"]

    def test_root_redirects_to_docs(self, client):
        response = client.get("/api/")
        assert response.status_code == 200
        assert response.json()["docs"] == "/docs"

    def test_unknown_route_returns_404(self, client):
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
