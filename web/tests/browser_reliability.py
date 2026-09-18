"""Mocked UI regressions. Start Vite on5188; no request reaches a real API."""
import json
import os
import re
from playwright.sync_api import sync_playwright

BASE = os.environ.get("ZEVO_TEST_URL", "http://127.0.0.1:5188")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        state = {"runs_error": True, "preflight": "blocked", "launches": [], "checks": [], "model": False}
        errors = []

        def api(route):
            request = route.request
            path = request.url.split('/api/', 1)[1].split('?', 1)[0]
            data = []
            status = 200
            if path == "runs" and request.method == "POST":
                state["launches"].append(request.post_data_json)
                status, data = 422, {"detail": "Mock launch stopped: no backend write"}
            elif path == "preflight":
                state["checks"].append(request.post_data_json)
                severity = "blocker" if state["preflight"] == "blocked" else "risk"
                data = {"status": state["preflight"], "summary": "Mock check", "items": [{"code": severity, "severity": severity, "message": "Verify selected compute", "hint": "Check settings"}]}
            elif request.method not in ("GET", "HEAD"):
                raise AssertionError(f"Unexpected mutation {request.method} {path}")
            elif path == "runs" and state["runs_error"]:
                status, data = 503, {"detail": "Mock outage"}
            elif path == "settings":
                data = {"entries": [], "ssh_connections": []}
            elif path == "hardware/ssh":
                data = [{"id": "host-a", "name": "Test compute", "host": "example.invalid", "category": "instance", "status": "verified"}]
            elif path == "cost/total":
                data = {"total_usd": 0, "agent_usd": 0, "gpu_usd": 0}
            elif path == "models" and state["model"]:
                data = [{"version_tag": "M-old-model", "run_id": "older-than-500", "base_model": "example/model", "training_method": "sft", "dataset_source": "example/data", "model_path": "/model", "model_path_abs": "/model", "task_objective": "Old successful model", "task_name": "old-task", "metric": "accuracy", "metric_direction": "max", "eval": {}, "registered_at": "2026-01-01T00:00:00Z", "iteration": 1, "champion_test_score": .8, "baseline_test_score": .6, "improvement": .2}]
            route.fulfill(status=status, content_type="application/json", body=json.dumps(data), headers={"X-Total-Count": "0"})

        context.route("**/api/**", api)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(BASE + "/runs")
        page.get_by_role("alert").filter(has_text="Could not refresh runs").wait_for()
        assert page.get_by_text("No runs on record yet.").count() == 0
        state["runs_error"] = False
        page.goto(BASE + "/runs?q=absent")
        page.get_by_text("No runs match your search.").wait_for()
        page.get_by_role("button", name="Clear search").click()
        page.get_by_text("No runs on record yet.").wait_for()
        state["model"] = True
        page.goto(BASE + "/models")
        page.get_by_text("M-old-model", exact=True).first.wait_for()
        # A registry model must remain visible even though its producing run is not in /runs.
        page.keyboard.press("Control+k")
        page.get_by_placeholder("Jump to a run, agent, page, or launch a run…").fill("Launch a new run")
        page.keyboard.press("Enter")
        dialog = page.get_by_role("dialog", name="Start a new run")
        dialog.wait_for()
        page.get_by_placeholder("what to call this run, e.g. bar-exam-take1").fill("mock-run")
        page.get_by_placeholder("name the task, e.g. bar-exam-reasoning").fill("mock-task")
        page.get_by_placeholder("e.g. Improve a small open model's accuracy on US bar-exam style MCQs.").fill("Improve accuracy on a test task")
        page.get_by_role("button", name="GPU backend", exact=True).click()
        page.get_by_role("option", name="Test compute").click()
        # Closing a nested select/palette must not close the parent launch dialog.
        page.get_by_role("button", name="GPU backend", exact=True).click()
        page.keyboard.press("Escape")
        assert dialog.is_visible()
        page.keyboard.press("Control+k")
        page.get_by_role("dialog", name="Jump to", exact=True).wait_for()
        page.keyboard.press("Escape")
        assert dialog.is_visible()
        dialog.get_by_role("button", name="close", exact=True).focus()
        page.keyboard.press("Shift+Tab")
        assert dialog.evaluate("el => el.contains(document.activeElement)")
        page.get_by_role("button", name="start run", exact=True).click()
        page.get_by_text("Resolve these launch checks").wait_for()
        assert not state["launches"]
        assert state["checks"][-1]["ssh_host_id"] == "host-a"
        assert state["checks"][-1]["mode"] == "auto"
        state["preflight"] = "risky"
        page.get_by_role("button", name="start run", exact=True).click()
        page.get_by_text("Review before launching").wait_for()
        assert not state["launches"]
        page.get_by_role("button", name="start run", exact=True).click()
        page.get_by_text("API 422: Mock launch stopped: no backend write").wait_for()
        launch = state["launches"][0]
        assert (launch["iteration_budget"], launch["max_cost_usd"], launch["max_runtime_hours"], launch["num_gpus"]) == (3, 10, 1, 1)
        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="How to run: Auto", exact=True).click()
        bounds = page.locator("[data-mode-info].fixed").bounding_box()
        assert bounds and bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 390
        page.get_by_role("button", name="How to run: Auto", exact=True).click()
        dialog.get_by_role("button", name=re.compile("^Standard")).click()
        assert dialog.get_by_text("Other files", exact=True).count() == 0
        page.keyboard.press("Escape")
        page.goto(BASE + "/runs")
        assert page.locator("main").bounding_box()["width"] >= 380
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        assert not errors, errors
        browser.close()
        print("PASS: outage/search states, old models, bounded payload, selected-target preflight, blocker/risk gates, nested dialog keyboard, mobile help and navigation. All APIs mocked.")


if __name__ == "__main__":
    main()
