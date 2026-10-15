"""Regression checks for source freshness and publication/delivery gates."""
import copy
import importlib.util
import json
import tempfile
import io
import os
import subprocess
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('daily_pipeline', ROOT / 'scripts/daily_pipeline.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def listing(day='Thursday, 01 October 2026', shown=1, total=1, kind='New', version=1):
    html_link = f'<a href="https://arxiv.org/html/2609.40360v{version}">html</a>' if version is not None else ''
    return f'''<h3>Showing new listings for {day}</h3><dl>
<h3>{kind} submissions (showing {shown} of {total} entries)</h3>
<dt><a href ="/abs/2609.40360" title="Abstract">arXiv:2609.40360</a> (replaced) {html_link}</dt><dd>
<div class="list-title mathjax"><span>Title:</span> Cross month video</div>
<div class="list-authors"><a>Alice</a></div>
<div class="list-subjects">Computer Vision (cs.CV)</div>
<p class="mathjax">A video generation model with reinforcement learning on its generated videos.</p>
</dd><h3>Replacements</h3><dt><a href="/abs/2609.00001">old</a></dt></dl>'''


def row():
    return {'id': '2609.40360', 'topic': '视频强化学习', 'short': 'Video：提升生成视频质量',
            'affiliations': [], 'coverage': '任务内通用', 'task': '视频生成',
            'problem': '论文问题。' * 10, 'method': '论文方法。' * 10,
            'experiment': '论文实验。' * 10, 'judgment': '论文判断。' * 10,
            'evidence': [{'claim': '视频生成', 'quote': 'A video generation model'}],
            'paper': {'id': '2609.40360', 'title': 'Video title', 'authors': ['Alice'], 'summary': 'A video generation model'}}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def state(self, status='prepared'):
        value = {'day': '2026-10-14', 'status': status, 'url': 'https://example.com/daily/2026-10-14/', 'content_id': 'hash', 'ids': ['2610.10000']}
        p.atomic(self.data / '2026-10-14/state.json', value)
        return value

    def test_cross_month_new_and_replacement_separation(self):
        result = p.parse_announcements(listing(), '2026-10-01', ['cs.CV'])
        self.assertEqual([r['id'] for r in result], ['2609.40360'])
        self.assertIn('reinforcement learning', result[0]['summary'])
        self.assertEqual(result[0]['listing_kind'], 'new')
    def test_replacement_only_listing_is_valid_daily_source(self):
        result = p.parse_announcements(listing(kind='Replacement', version=3), '2026-10-01', ['cs.CV'])
        self.assertEqual([r['id'] for r in result], ['2609.40360'])
        self.assertEqual(result[0]['listing_kind'], 'replacement')
        self.assertEqual(result[0]['version'], 3)
    def test_real_arxiv_replacement_only_html_fixture(self):
        source = (ROOT / 'scripts/tests/fixtures/arxiv_replacements_only.html').read_text()
        result = p.parse_announcements(source, '2026-10-15', ['cs.CL', 'cs.AI', 'cs.LG'])
        self.assertEqual([r['id'] for r in result], ['2311.04902'])
        self.assertEqual(result[0]['version'], 3)
    def test_replacement_without_html_link_uses_latest_version(self):
        result = p.parse_announcements(listing(kind='Replacement', version=None), '2026-10-01', ['cs.CV'])
        self.assertIsNone(result[0]['version'])
    def test_new_section_has_priority_over_replacement(self):
        replacement = listing(kind='Replacement').split('<dl>', 1)[1].split('</dl>', 1)[0]
        source = listing().replace('</dl>', replacement + '</dl>')
        result = p.parse_announcements(source, '2026-10-01', ['cs.CV'])
        self.assertTrue(all(r['listing_kind'] == 'new' for r in result))
    def test_same_day_source_cache_requires_matching_audit_hash(self):
        source_dir = self.data / 'source'
        source_dir.mkdir()
        source = listing()
        path = source_dir / 'cs-new.html'
        path.write_text(source)
        digest = p.hashlib.sha256(source.encode()).hexdigest()
        (source_dir / 'sources.jsonl').write_text(json.dumps({'url': p.LISTING_URL, 'file': path.name, 'sha256': digest}) + '\n')
        self.assertEqual(p.cached_listing(source_dir, '2026-10-01'), source)
        path.write_text(source + 'tampered')
        self.assertIsNone(p.cached_listing(source_dir, '2026-10-01'))
    def test_prepare_reuses_audited_same_day_source_without_fetch(self):
        source_dir = self.data / '2026-10-01' / 'source'
        source_dir.mkdir(parents=True)
        source = listing()
        (source_dir / 'cs-new.html').write_text(source)
        digest = p.hashlib.sha256(source.encode()).hexdigest()
        (source_dir / 'sources.jsonl').write_text(json.dumps({'url': p.LISTING_URL, 'file': 'cs-new.html', 'sha256': digest}) + '\n')
        settings = {'categories': ['cs.CV']}
        with patch.object(p, 'DATA', self.data), patch.object(p, 'config', return_value=settings), \
             patch.object(p, 'DeepSeek', return_value=object()), patch.object(p, 'classify', return_value=[]), \
             patch.object(p.ArxivClient, 'get', side_effect=AssertionError('same-day source was fetched again')):
            result = p.prepare('2026-10-01')
        self.assertEqual(result['status'], 'no_matches')
        self.assertEqual(result['listing_kind'], 'new')
    def test_replacement_fingerprint_stays_stable_across_dates(self):
        paper = {'id': '2609.40360', 'title': 'x', 'authors': ['a'], 'summary': 's', 'categories': ['cs.CV'],
                 'listing_kind': 'replacement', 'announcement_date': '2026-10-01'}
        other = dict(paper, announcement_date='2026-10-02')
        self.assertEqual(p.paper_metadata_fingerprint(paper), p.paper_metadata_fingerprint(other))
    def test_replacement_version_ledger_allows_newer_version(self):
        paper = {'id': '2609.40360', 'version': 4, 'listing_kind': 'replacement', 'title': 'x',
                 'authors': ['a'], 'summary': 's', 'categories': ['cs.CV'], 'announcement_date': '2026-10-02'}
        seen = {'2609.40360': {'version': 3, 'announcement_date': '2026-10-01',
                               'metadata_hash': p.paper_metadata_fingerprint(dict(paper, version=3))}}
        self.assertFalse(p.already_processed(paper, seen))
        same_day = dict(paper, version=3, announcement_date='2026-10-01')
        self.assertTrue(p.already_processed(same_day, seen))
        self.assertFalse(p.already_processed(paper, {'2609.40360': 'old timestamp'}))
    def test_stale_page_is_pending(self):
        with self.assertRaises(p.Pending):
            p.parse_announcements(listing(), '2026-10-14', ['cs.CV'])
    def test_valid_announcement_without_configured_category_is_no_matches(self):
        html = listing().replace('Computer Vision (cs.CV)', 'Machine Learning (cs.LG)')
        with patch.object(p, 'DATA', self.data), patch.object(p, 'config', return_value={'categories': ['cs.CV']}), \
             patch.object(p.ArxivClient, 'get', return_value=html.encode()):
            result = p.prepare('2026-10-01')
        self.assertEqual(result['status'], 'no_matches')
        self.assertEqual(result['source_count'], 0)
    def test_partial_listing_is_rejected(self):
        with self.assertRaises(RuntimeError):
            p.parse_announcements(listing(shown=1, total=2001), '2026-10-01', ['cs.CV'])
    def test_weekend_and_holiday_use_beijing_day(self):
        settings = {'noAnnouncementDatesET': ['2026-10-12']}
        self.assertFalse(p.expected_day(date(2026, 10, 13), settings))
        self.assertTrue(p.expected_day(date(2026, 10, 12), settings))
        self.assertFalse(p.expected_day(date(2026, 10, 17), settings))
    def test_missing_id_and_fabricated_quote_rejected(self):
        settings = p.config()
        papers = p.parse_announcements(listing(), '2026-10-01', ['cs.CV'])
        with self.assertRaises(ValueError): p.decisions_valid({'decisions': []}, papers, settings)
        bad = {'decisions': [{'id': '2609.40360', 'accepted': True, 'reason': 'yes', 'topic': '视频强化学习', 'evidence': 'invented quote from a different paper', 'reject_check': 'ok'}]}
        with self.assertRaises(ValueError): p.decisions_valid(bad, papers, settings)
    def test_message_is_bounded_with_bare_url(self):
        rows = [row() for _ in range(30)]
        url = 'https://example.com/daily/2026-10-14/'
        message = p.render_message('2026-10-14', rows, url)
        self.assertEqual(message.count('— '), 8)
        self.assertEqual(message.splitlines()[-1], url)
        self.assertLess(len(message), 1200)
    def test_message_identifies_replacement_updates(self):
        item = row()
        item['paper']['listing_kind'] = 'replacement'
        message = p.render_message('2026-10-14', [item], 'https://example.com/daily/')
        self.assertIn('替换版本', message.splitlines()[0])
    def test_page_fingerprint_stable(self):
        rows = [row()]
        a, ident = p.render_page('2026-10-14', rows)
        b, other = p.render_page('2026-10-14', rows)
        self.assertEqual(ident, other)
        self.assertEqual(a, b)
        self.assertIn('daily-content-' + ident, a)
        self.assertIn('daily-filter', a)
    def test_latest_update_render_uses_unversioned_arxiv_links(self):
        item = row()
        item['paper']['listing_kind'] = 'replacement'
        item['paper']['version'] = None
        page, _ = p.render_page('2026-10-14', [item])
        self.assertIn('https://arxiv.org/abs/2609.40360)', page)
        self.assertIn('阅读版本**：当前版本', page)
    def test_http_200_stale_page_does_not_send(self):
        self.state()
        with patch.object(p, 'DATA', self.data), patch.object(p, 'public_content_matches', return_value=False), patch.object(p, 'run_command') as call:
            with self.assertRaises(p.Pending): p.deliver('2026-10-14')
            call.assert_not_called()
    def test_failed_receipt_does_not_write_ledger(self):
        self.state()
        with patch.object(p, 'DATA', self.data), patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'run_command'), patch.object(p, 'receipt_status', return_value=False):
            with self.assertRaises(RuntimeError): p.deliver('2026-10-14')
            self.assertEqual(p.read(self.data / '2026-10-14/state.json')['status'], 'prepared')
    def test_success_receipt_recovers_without_resend(self):
        self.state()
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), patch.object(p, 'config', return_value={}), patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'receipt_status', return_value=True), patch.object(p, 'run_command') as call:
            result = p.deliver('2026-10-14')
            self.assertEqual(result['status'], 'delivered')
            call.assert_not_called()
            self.assertIn('2610.10000', p.read(self.data / 'refs/scans/daily-ledger.json')['seen'])
    def test_before_release_no_source_request(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 14, 7, 59, tzinfo=p.SHANGHAI)), patch.object(p, 'prepare') as prep:
            self.assertEqual(p.tick('2026-10-14')['status'], 'waiting_announcement')
            prep.assert_not_called()
    def test_before_delivery_no_send(self):
        self.state('publishing')
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 14, 9, 29, tzinfo=p.SHANGHAI)), patch.object(p, 'deliver') as send:
            p.tick('2026-10-14')
            send.assert_not_called()

    def test_at_delivery_deadline_triggers_delivery(self):
        self.state('published')
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 14, 9, 30, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'deliver', return_value={'status': 'delivered'}) as send:
            p.tick('2026-10-14')
            send.assert_called_once_with('2026-10-14')

    def test_arxiv_rate_limit_persists_retry_after(self):
        directory = self.data / 'source'
        client = p.ArxivClient(directory)
        error = p.urllib.error.HTTPError('https://arxiv.org/list/cs/new', 429, 'limited', {'Retry-After': '90'}, None)
        with patch.object(p.urllib.request, 'urlopen', side_effect=error), patch.object(p.time, 'sleep'):
            with self.assertRaises(p.Pending): client.raw('https://arxiv.org/list/cs/new')
        retry = p.read(directory / 'retry.json')
        self.assertEqual(retry['http_status'], 429)
        self.assertGreaterEqual((datetime.fromisoformat(retry['retry_at']) - p.now()).total_seconds(), 50)

    def test_successful_arxiv_response_clears_retry_backoff(self):
        directory = self.data / 'source'; directory.mkdir()
        p.atomic(directory / 'retry.json', {'failures': 3, 'retry_at': '2026-10-15T00:00:00+08:00'})
        response = type('Response', (), {'geturl': lambda self: 'https://arxiv.org/list/cs/new',
                                         'read': lambda self: b'ok', '__enter__': lambda self: self,
                                         '__exit__': lambda self, *args: None})()
        with patch.object(p.urllib.request, 'urlopen', return_value=response):
            self.assertEqual(p.ArxivClient(directory).raw('https://arxiv.org/list/cs/new'), b'ok')
        self.assertFalse((directory / 'retry.json').exists())

    def test_cron_runs_each_minute_during_delivery_window(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import paper_daily
        line = paper_daily.cron_line(p.config())
        self.assertIn('* 8-12 * * 1-5', line)

    def test_pdf_line_break_normalization_preserves_numbers(self):
        p.evidence_valid({'evidence': [{'claim': 'data', 'quote': 'Training on DeepMath-103K gives 35.5 percent.'}]},
                         'Training on DeepMath-\n 103K gives 35.5 percent.')
        p.evidence_valid({'evidence': [{'claim': 'method', 'quote': 'The representation is pre-trained.'}]},
                         'The represen-\n tation is pre-trained.')
        p.evidence_valid({'evidence': [{'claim': 'abbreviation', 'quote': 'This non-AR model is efficient.'}]},
                         'This non-\n AR model is efficient.')
        with self.assertRaises(ValueError):
            p.evidence_valid({'evidence': [{'claim': 'percentage', 'quote': 'Up to 3 percent on HealthBench.'}]},
                             'Up to 3% on HealthBench.')
        with self.assertRaises(ValueError):
            p.evidence_valid({'evidence': [{'claim': 'data', 'quote': 'Training on DeepMath-103K gives 55.5 percent.'}]},
                             'Training on DeepMath-\n 103K gives 35.5 percent.')
    def test_paper_text_uses_raw_extraction_and_reuses_cached_pdf(self):
        directory = self.data / 'papers'; directory.mkdir()
        pdf = directory / '2609.40360v2.pdf'; pdf.write_bytes(b'%PDF test')
        paper = {'id': '2609.40360', 'version': 2}
        def extract(args, **kwargs):
            self.assertEqual(args[1], '-raw')
            Path(args[-1]).write_text('paper body ' * 200)
            return ''
        with patch.object(p, 'run_command', side_effect=extract) as command:
            first = p.paper_text(paper, object(), directory)
            second = p.paper_text(paper, object(), directory)
        self.assertEqual(first, second)
        self.assertEqual(command.call_count, 1)

    def test_truncated_provider_response_never_cached(self):
        ds = object.__new__(p.DeepSeek)
        ds.settings = {}; ds.audit = self.data / 'ds'; ds.audit.mkdir(); ds.key = 'test-only'
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"approved":true,"reason":"valid"}'}}]}
        def fake_response(*args, **kwargs):
            return io.StringIO(json.dumps(response))
        with patch.object(p.urllib.request, 'urlopen', side_effect=fake_response) as http, patch.object(p.time, 'sleep'):
            with self.assertRaises(RuntimeError): ds.ask('review', 'JSON', {}, p.require_review)
            self.assertEqual(http.call_count, 3)
        self.assertFalse(any('parsed' in p.read(f, {}) for f in ds.audit.glob('*.json')))

    def test_publisher_never_resets_unowned_directory(self):
        self.state()
        (self.data / 'publisher').mkdir()
        with patch.object(p, 'DATA', self.data), patch.object(p, 'public_content_matches', return_value=False), patch.object(p, 'run_command') as command:
            with self.assertRaises(RuntimeError): p.publish('2026-10-14')
            self.assertEqual(command.call_count, 1)
            self.assertEqual(command.call_args.args[0][:2], ['git', 'fetch'])

    def test_generated_page_requires_closed_cards_and_matching_labels(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        import validate_daily
        page, _ = p.render_page('2026-10-14', [row()])
        file = self.data / 'report.md'
        file.write_text(page)
        self.assertEqual(validate_daily.validate(file), [])
        file.write_text(page.replace('</article>', ''))
        self.assertTrue(validate_daily.validate(file))
        file.write_text(page.replace('任务类型=`视频生成`', '任务类型=`驾驶规划`'))
        self.assertTrue(validate_daily.validate(file))

    def test_publication_pushes_only_daily_paths_and_preserves_user_edits(self):
        repo = self.data / 'repo'; repo.mkdir()
        remote = self.data / 'remote.git'
        data = repo / 'refs/scans/daily-runs'
        environment = {'GIT_AUTHOR_NAME': 'Daily Test', 'GIT_AUTHOR_EMAIL': 'daily@example.invalid',
                       'GIT_COMMITTER_NAME': 'Daily Test', 'GIT_COMMITTER_EMAIL': 'daily@example.invalid'}
        def git(args, cwd=repo):
            return p.run_command(['git', *args], cwd)
        original = p.run_command
        def commands(args, cwd=repo, env=None):
            if args[1:3] == ['-m', 'mkdocs']:
                return ''  # The production MkDocs build is verified separately.
            return original(args, cwd, env)
        with patch.dict(os.environ, environment):
            git(['init', '--bare', '--initial-branch=main', str(remote)], self.data)
            git(['init', '-b', 'main'])
            p.atomic(repo / 'docs/daily/index.md', '# 日报\n\n## 日期归档\n')
            p.atomic(repo / 'mkdocs.yml', 'site_name: Test\nnav:\n  - 日报:\n    - 日报首页: daily/index.md\n')
            p.atomic(repo / 'docs/unrelated.md', 'committed\n')
            git(['add', 'docs', 'mkdocs.yml']); git(['commit', '-m', 'Initialize daily publication fixture'])
            git(['remote', 'add', 'origin', str(remote)]); git(['push', 'origin', 'main'])
            p.atomic(repo / 'docs/unrelated.md', 'user unfinished edit\n')
            directory = data / '2026-10-14'
            artifacts = p.write_artifacts('2026-10-14', [row()], directory, p.config())
            p.atomic(directory / 'state.json', {'day': '2026-10-14', 'status': 'prepared', **artifacts})
            with patch.object(p, 'ROOT', repo), patch.object(p, 'DATA', data), \
                 patch.object(p, 'config', return_value={}), patch.object(p, 'public_content_matches', return_value=False), \
                 patch.object(p, 'run_command', side_effect=commands):
                result = p.publish('2026-10-14')
            self.assertEqual(result['status'], 'publishing')
            self.assertEqual((repo / 'docs/unrelated.md').read_text(), 'user unfinished edit\n')
            changed = git(['diff-tree', '--no-commit-id', '--name-only', '-r', result['publication_commit']], data / 'publisher').splitlines()
            self.assertEqual(sorted(changed), ['docs/daily/2026-10-14.md', 'docs/daily/index.md', 'mkdocs.yml'])
            self.assertEqual(git(['rev-parse', 'main'], remote), result['publication_commit'])


if __name__ == '__main__':
    unittest.main()
