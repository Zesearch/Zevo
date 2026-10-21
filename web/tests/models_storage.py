"""Browser regression for saved locations, real upload actions, and run links (mock API)."""
import json
import os
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, expect


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        state = {'token': False, 'uploaded': False, 'polls': 0, 'uploading': False}
        requests, errors = [], []
        page.on('pageerror', lambda e: (errors.append(str(e)), print(e.stack)))
        tag = 'M-12345678'
        def api(route):
            path = urlparse(route.request.url).path
            data = {}
            if path == '/api/auth/refresh':
                data = {'access_token': 'test'}
            elif path == '/api/auth/me':
                tenant = dict(tenant_id='t', tenant_name='Test', slug='test', role='admin')
                data = dict(user=dict(id='u', email='u@example.com', name='Test', completed_onboarding=True,
                                     needs_workspace_setup=False, is_superadmin=False),
                            memberships=[tenant], active_tenant=tenant)
            elif path == '/api/models':
                data = [dict(version_tag=tag, run_id='12345678-run', iteration=1, base_model='base/model',
                    training_method='grpo', dataset_source='data', model_path='ssh://alice@gpu:22/lustre/models/m',
                    model_path_abs='', task_objective='Code', metric='accuracy', metric_direction='max', eval={},
                    registered_at='2026-10-21T00:00:00Z', task_name='Code', champion_test_score=None,
                    baseline_test_score=None, improvement=None)]
            elif path.endswith('/storage'):
                if state['uploading']:
                    state['polls'] += 1
                    if state['polls'] >= 3: state['uploaded'] = True
                data = dict(location=dict(kind='remote', machine='alice@gpu:22', path='/lustre/models/m', url=''),
                            hf_url='https://huggingface.co/alice/model/tree/abc' if state['uploaded'] else '',
                            can_upload=True, upload={'status': 'succeeded' if state['uploaded'] else 'uploading'} if state['uploading'] else {})
            elif path == '/api/hardware/vpn/profiles':
                data = {'available': False, 'profiles': []}
            elif path == '/api/credits/balance':
                data = {'unlimited': True, 'available': None}
            elif path == '/api/settings':
                data = {'entries': [{'name': 'HF_TOKEN', 'present': state['token']}]}
            elif path.endswith('/upload-hf'):
                requests.append(route.request.post_data_json)
                state['uploading'] = True
                data = {'status': 'uploading'}
            elif path == '/api/runs/statistics':
                data = {'total': 0, 'active': 0, 'succeeded': 0, 'failed': 0, 'runtime_seconds': {'all': 0}, 'improvements': []}
            elif path == '/api/cost/total':
                data = {'total_usd': 0, 'agent_usd': 0, 'gpu_usd': 0}
            elif path.endswith('/card'):
                route.fulfill(body='# Model card', content_type='text/markdown')
                return
            else:
                data = []
            route.fulfill(body=json.dumps(data), content_type='application/json')
        page.route('**/api/**', api)
        base = os.environ.get('ZEVO_TEST_URL', 'http://127.0.0.1:5192')
        page.goto(base + '/models')
        expect(page.get_by_role('link', name=tag, exact=True)).to_have_attribute('href', '/runs/12345678-run')
        page.get_by_role('button', name='where it is', exact=True).click()
        expect(page.get_by_text('alice@gpu:22', exact=True)).to_be_visible()
        expect(page.get_by_text('/lustre/models/m', exact=True)).to_be_visible()
        expect(page.get_by_text('Already on this machine.', exact=False)).to_have_count(0)
        page.get_by_role('button', name='Upload to Hugging Face', exact=True).click()
        expect(page.get_by_text('Set a Hugging Face write token in', exact=False)).to_be_visible()
        expect(page.get_by_role('link', name='Settings', exact=True).last).to_have_attribute('href', '/settings')
        assert not requests
        state['token'] = True
        page.reload()
        page.get_by_role('button', name='Upload to Hugging Face', exact=True).click()
        page.get_by_role('textbox', name='Hugging Face repository').fill('alice/model')
        page.get_by_role('button', name='Upload', exact=True).click()
        expect(page.get_by_text('Upload in progress. You can leave this page.')).to_be_visible()
        expect(page.get_by_role('link', name='View on Hugging Face')).to_be_visible(timeout=15000)
        expect(page.get_by_role('button', name='Upload to Hugging Face', exact=True)).to_have_count(0)
        assert requests == [{'weights': 'hf', 'hf_repo_id': 'alice/model', 'hf_private': True}]
        page.get_by_role('button', name='where it is', exact=True).click()
        expect(page.get_by_text('/lustre/models/m', exact=True)).to_be_visible()
        expect(page.get_by_role('link', name='Hugging Face: https://huggingface.co/alice/model/tree/abc')).to_be_visible()
        assert not errors, errors
        browser.close()
        print('Models location, token guidance, upload polling, HF link, and run-link checks passed.')

if __name__ == '__main__':
    main()
