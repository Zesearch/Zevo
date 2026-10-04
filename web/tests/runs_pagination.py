"""Runs pagination regressions against Vite, with all API requests mocked."""
import json
import os
import re
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright, expect


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        state = {"total": 151}
        requests = []
        errors = []
        page.on("pageerror", lambda error: (errors.append(str(error)), print(error)))

        def api(route):
            url = urlparse(route.request.url)
            query = parse_qs(url.query)
            data = []
            total = state["total"]
            if url.path == "/api/runs":
                offset = int(query.get("offset", [0])[0])
                limit = int(query.get("limit", [100])[0])
                requests.append((query.get("group", [""])[0], offset, limit))
                statuses = ["success", "running", "planning", "degraded", "failed", "cancelled"]
                data = [dict(id=f"run-{i}", run_name=f"Run {i}", task_name=f"Task {i // 3}",
                             status=statuses[min(i // 25, 5)], is_terminal=True, history=[],
                             iterations_completed=0, max_iterations=1,
                             started_at="2026-01-01T00:00:00Z")
                        for i in range(state["total"])]
                if query.get("task"):
                    data = [row for row in data if row['task_name'] in query['task']]
                if query.get("status"):
                    data = [row for row in data if row['status'] in query['status']]
                total = len(data)
                data = data[offset:offset + limit]
            elif url.path == "/api/cost/total":
                data = {"total_usd": 0, "agent_usd": 0, "gpu_usd": 0}
            elif url.path == "/api/runs/statistics":
                data = {"total": state["total"], "active": 0, "succeeded": state["total"],
                        "failed": 0, "runtime_seconds": {"all": 0}, "improvements": []}
            elif url.path == "/api/runs/task-counts":
                data = {f"Task {i // 3}": 3 for i in range(state["total"])}
            elif url.path == "/api/runs/status-counts":
                data = {status: 25 for status in ["success", "running", "planning", "degraded", "failed", "cancelled"]}
            route.fulfill(content_type="application/json", body=json.dumps(data),
                          headers={"X-Total-Count": str(total)})

        page.route("**/api/**", api)
        base = os.environ.get("ZEVO_TEST_URL", "http://127.0.0.1:5188")
        pager = page.get_by_text(re.compile(r"^page \d+ / \d+$"))
        page.set_viewport_size({"width": 1000, "height": 650})
        for group in ("Status", "Task"):
            page.goto(base + "/runs")
            page.get_by_role("button", name=f"Group By {group}", exact=True).click()
            page.wait_for_timeout(1000)
            _, count = map(int, re.findall(r"\d+", pager.inner_text()))
            limit = requests[-1][2]
            for n in range(2, 6):
                # Clicking the footer scrolls it into view. Group titles make
                # the list taller, but must not change the request's page size.
                page.get_by_title("Next page", exact=True).click()
                # Browser/layout resize notifications can arrive while the
                # user is scrolled to the footer, without extra viewport space.
                page.locator("main").evaluate("el => { el.scrollTop = el.scrollHeight; }")
                page.evaluate("window.dispatchEvent(new Event('resize'))")
                page.wait_for_timeout(800)
                expect(pager).to_have_text(f"page {n} / {count}")
                assert requests[-1][2] == limit, requests[-5:]
            page.get_by_title("Previous page", exact=True).click()
            page.wait_for_timeout(800)
            expect(pager).to_have_text(f"page 4 / {count}")
            assert requests[-1][2] == limit

        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(base + "/runs")
        page.get_by_role("button", name="Group By Status", exact=True).click()
        page.get_by_role("button", name="25", exact=True).click()
        pager = page.get_by_text(re.compile(r"^page \d+ / \d+$"))
        for n in range(1, 7):
            expect(pager).to_have_text(f"page {n} / 7")
            page.get_by_title("Next page", exact=True).click()
        expect(pager).to_have_text("page 7 / 7")
        expect(page.get_by_role("link", name="Run 150", exact=True)).to_be_visible()

        # A polling refresh removes the last page while we are viewing it.
        state["total"] = 150
        expect(pager).to_have_text("page 6 / 6", timeout=10000)
        expect(page.get_by_role("link", name="Run 149", exact=True)).to_be_visible()
        expect(page.get_by_text("No runs on record yet.")).to_have_count(0)
        expect(page.get_by_title("Next page", exact=True)).to_be_disabled()

        # Auto sizing must recover when a taller window reduces page count.
        page.get_by_role("button", name="auto", exact=True).click()
        for _ in range(100):
            if page.get_by_title("Next page", exact=True).is_disabled():
                break
            page.get_by_title("Next page", exact=True).click()
            page.wait_for_timeout(40)
        page.set_viewport_size({"width": 1440, "height": 1500})
        expect(page.get_by_role("link", name=re.compile(r"^Run \d+$")).first).to_be_visible()
        page.wait_for_timeout(700)
        current, count = map(int, re.findall(r"\d+", pager.inner_text()))
        assert current <= count, pager.inner_text()
        expect(page.get_by_text("No runs on record yet.")).to_have_count(0)
        # Pick tasks that are absent from the first page; combine selections,
        # clear them, and ensure switching groups does not leave hidden filters.
        page.goto(base + "/runs")
        page.get_by_role("button", name="Group By Task", exact=True).click()
        page.locator("summary").filter(has_text="All tasks").click()
        page.get_by_role("textbox", name="Find a task").fill("Task 4")
        page.get_by_role("checkbox", name="Task 40", exact=True).check()
        expect(page.get_by_role("link", name="Run 120", exact=True)).to_be_visible()
        page.get_by_role("checkbox", name="Task 41", exact=True).check()
        expect(page.get_by_text("6 matches", exact=True)).to_be_visible()
        page.get_by_role("button", name="All tasks", exact=True).click()
        expect(page.locator("summary")).to_have_text("All tasks")
        page.get_by_role("checkbox", name="Task 40", exact=True).check()
        page.get_by_role("button", name="Group By Status", exact=True).click()
        page.locator("summary").filter(has_text="All statuses").click()
        expect(page.get_by_role("checkbox", name="Halted", exact=True)).to_have_count(0)
        page.get_by_role("checkbox", name="Running", exact=True).check()
        expect(page.get_by_text("25 matches", exact=True)).to_be_visible()
        page.get_by_role("checkbox", name="Success", exact=True).check()
        expect(page.get_by_text("50 matches", exact=True)).to_be_visible()
        page.get_by_role("checkbox", name="Running", exact=True).uncheck()
        page.get_by_role("checkbox", name="Success", exact=True).uncheck()
        page.get_by_role("button", name="All statuses", exact=True).click()
        expect(page.get_by_role("link", name="Run 0", exact=True)).to_be_visible()
        assert not errors, errors
        browser.close()
        print("Runs: grouped paging, polling shrink, auto resize and multi-select filters passed")


if __name__ == "__main__":
    main()
