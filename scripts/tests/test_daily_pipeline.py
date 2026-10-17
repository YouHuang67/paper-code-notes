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
            'affiliations': [], 'affiliation_types': [], 'level': '垂类',
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
        item = row()
        item['id'] = item['paper']['id'] = '2610.10000'
        item['affiliation_status'] = 'unlisted'
        directory = self.data / '2026-10-14'
        artifacts = p.write_artifacts('2026-10-14', [item], directory, p.config())
        analyses = directory / 'analyses'
        analyses.mkdir(parents=True, exist_ok=True)
        p.atomic(analyses / '2610.10000.json', {'status': 'approved', 'result': item})
        papers = directory / 'papers'
        papers.mkdir(parents=True, exist_ok=True)
        (papers / '2610.10000v1-raw.txt').write_text('A video generation model. ' * 10)
        value = {'day': '2026-10-14', 'status': status, **artifacts}
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

    def test_repeated_announcement_date_mismatch_pauses_for_operator(self):
        stale = listing(day='Monday, 12 October 2026').encode()
        times = [datetime(2026, 10, 14, hour, minute, tzinfo=p.SHANGHAI)
                 for hour, minute in ((9, 30), (9, 46), (10, 2))]
        for current in times:
            with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), \
                 patch.object(p, 'config', return_value={'categories': ['cs.CV']}), \
                 patch.object(p, 'now', return_value=current), patch.object(p.ArxivClient, 'get', return_value=stale):
                state = p.prepare('2026-10-14')
        self.assertEqual(state['status'], 'pending')
        self.assertEqual(state['source_day'], '2026-10-12')
        self.assertEqual(state['source_mismatch_attempts'], p.ANNOUNCEMENT_MISMATCH_LIMIT)
        self.assertTrue(state['requires_operator'])
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), \
             patch.object(p, 'config', return_value={'categories': ['cs.CV']}), \
             patch.object(p.ArxivClient, 'get', side_effect=AssertionError('paused day must not poll')):
            paused = p.prepare('2026-10-14')
        self.assertEqual(paused, state)
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
    def test_affiliation_types_distinguish_universities_companies_and_research_labs(self):
        self.assertEqual(p.affiliation_type_rule('Department of Computer Science, Stanford University'), '高校')
        self.assertEqual(p.affiliation_type_rule('Workbuddy Team, Tencent'), '企业')
        self.assertIsNone(p.affiliation_type_rule('Risk AI Research Lab'))
        self.assertIsNone(p.affiliation_type_rule('Toyota Research Institute'))
        self.assertEqual(p.affiliation_type_rule('Chinese Academy of Sciences'), '科研机构')
        self.assertEqual(p.affiliation_type_rule('Georgia Institute of Technology'), '高校')
        self.assertEqual(p.affiliation_type_rule('Independent Researcher'), '个人')
        self.assertTrue(p.affiliation_is_address('Atlanta, GA, USA'))
        self.assertTrue(p.affiliation_is_address('Nagoya, Aichi, Japan'))
        self.assertTrue(p.affiliation_is_address('Shenzhen, China'))
        self.assertFalse(p.affiliation_is_address('Nagoya University, Nagoya, Aichi, Japan'))
        self.assertIsNone(p.affiliation_type_rule('Atlanta, GA, USA'))

    def test_conflicting_affiliation_markers_remain_unclassified(self):
        self.assertIsNone(p.affiliation_type_rule('University of X, Tencent Research'))

    def test_affiliation_schema_requires_one_aligned_type_per_unit(self):
        item = row()
        item.update({'affiliations': ['Stanford University'], 'affiliation_status': 'listed',
                     'affiliation_types': [{'affiliation': 'Stanford University', 'type': '企业'}]})
        text = 'Stanford University. A video generation model. ' * 20
        with self.assertRaises(ValueError):
            p.analysis_valid(item, text)
        item['affiliation_types'][0]['type'] = '高校'
        p.analysis_valid(item, text)

    def test_affiliation_must_be_in_the_title_author_block(self):
        item = row()
        item.update({'affiliations': ['Stanford University'], 'affiliation_status': 'listed',
                     'affiliation_types': [{'affiliation': 'Stanford University', 'type': '高校'}]})
        source = 'Extended Abstract\fPaper Title Alice Stanford University Abstract body text. A video generation model.'
        p.analysis_valid(item, source)
        with self.assertRaisesRegex(ValueError, 'first page'):
            p.analysis_valid(item, 'Paper Title Alice Abstract body.\fBody cites Stanford University.')

    def test_render_page_excludes_a_standalone_postal_address(self):
        item = row()
        item.update({'affiliations': ['Stanford University', 'Atlanta, GA, USA'],
                     'affiliation_types': [
                         {'affiliation': 'Stanford University', 'type': '高校'},
                         {'affiliation': 'Atlanta, GA, USA', 'type': '其他'}]})
        page, _ = p.render_page('2026-10-14', [item])
        self.assertIn('**作者单位**：Stanford University', page)
        self.assertIn('**单位分类**：1=高校', page)
        self.assertIn('data-org="高校"', page)
        self.assertNotIn('Atlanta, GA, USA', page)

    def test_postal_address_is_rejected_as_an_institution(self):
        item = row()
        item.update({'affiliations': ['Atlanta, GA, USA'], 'affiliation_status': 'listed',
                     'affiliation_types': [{'affiliation': 'Atlanta, GA, USA', 'type': '其他'}]})
        with self.assertRaisesRegex(ValueError, 'postal address'):
            p.analysis_valid(item, 'Atlanta, GA, USA. A video generation model.')
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
        self.assertNotIn('\n## ', a, 'daily paper cards should not be grouped by task/topic headings')
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
            state = p.read(self.data / '2026-10-14/state.json')
            self.assertEqual(state['status'], 'prepared')
            self.assertEqual(state['delivery_attempts'], 1)
            self.assertEqual(state['last_error'], p.WECHAT_RECEIPT_FAILURE)
            ledger = p.read(self.data / 'refs/scans/daily-ledger.json', {'seen': {}})
            self.assertNotIn('2610.10000', ledger.get('seen', {}))

    def test_failed_receipt_waits_for_a_new_bridge_context(self):
        self.state()
        state_path = self.data / '2026-10-14/state.json'
        value = p.read(state_path)
        value.update({'last_error': p.WECHAT_RECEIPT_FAILURE, 'delivery_context_mtime_ns': 10**20})
        p.atomic(state_path, value)
        context = self.data / 'context_tokens.json'
        context.write_text('{}')
        with patch.object(p, 'DATA', self.data), patch.object(p, 'BRIDGE_CONTEXT_CACHE', context), \
             patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'receipt_status', return_value=False), \
             patch.object(p, 'run_command') as send:
            result = p.deliver('2026-10-14')
        self.assertEqual(result['delivery_deferred_reason'], 'WeChat context token has not refreshed since the last failed delivery')
        send.assert_not_called()

    def test_delivery_retry_backoff_suppresses_early_attempt(self):
        self.state()
        state_path = self.data / '2026-10-14/state.json'
        value = p.read(state_path)
        value.update({'last_error': 'WeChat interface unavailable', 'delivery_retry_at': '2026-10-16T09:35:00+08:00'})
        p.atomic(state_path, value)
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 16, 9, 31, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'receipt_status', return_value=False), \
             patch.object(p, 'run_command') as send:
            result = p.deliver('2026-10-14')
        self.assertIn('delivery retry deferred until', result['delivery_deferred_reason'])
        send.assert_not_called()

    def test_refreshed_bridge_context_allows_pending_delivery(self):
        self.state('publishing')
        state_path = self.data / '2026-10-14/state.json'
        value = p.read(state_path)
        value.update({'last_error': p.WECHAT_RECEIPT_FAILURE, 'delivery_context_mtime_ns': 1})
        p.atomic(state_path, value)
        context = self.data / 'context_tokens.json'
        context.write_text('{}')
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), patch.object(p, 'config', return_value={}), patch.object(p, 'BRIDGE_CONTEXT_CACHE', context), \
             patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'receipt_status', side_effect=[False, True]), \
             patch.object(p, 'run_command') as send, patch.object(p, 'now', return_value=datetime(2026, 10, 16, 9, 31, tzinfo=p.SHANGHAI)):
            result = p.deliver('2026-10-14')
        self.assertEqual(result['status'], 'delivered')
        send.assert_called_once()
    def test_success_receipt_recovers_without_resend(self):
        self.state()
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), patch.object(p, 'config', return_value={}), patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'receipt_status', return_value=True), patch.object(p, 'run_command') as call:
            result = p.deliver('2026-10-14')
            self.assertEqual(result['status'], 'delivered')
            call.assert_not_called()
            self.assertIn('2610.10000', p.read(self.data / 'refs/scans/daily-ledger.json')['seen'])

    def test_success_receipt_recovers_even_if_context_has_not_refreshed(self):
        self.state('publishing')
        state_path = self.data / '2026-10-14/state.json'
        value = p.read(state_path)
        value.update({'last_error': p.WECHAT_RECEIPT_FAILURE, 'delivery_context_mtime_ns': 10**20})
        p.atomic(state_path, value)
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), patch.object(p, 'config', return_value={}), \
             patch.object(p, 'BRIDGE_CONTEXT_CACHE', self.data / 'missing-context.json'), \
             patch.object(p, 'public_content_matches', return_value=True), patch.object(p, 'receipt_status', return_value=True), \
             patch.object(p, 'run_command') as send:
            result = p.deliver_oldest_pending_before('2026-10-15')
        self.assertEqual(result, {'day': '2026-10-14', 'status': 'delivered'})
        send.assert_not_called()
    def test_before_release_no_source_request(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 14, 7, 59, tzinfo=p.SHANGHAI)), patch.object(p, 'prepare') as prep:
            self.assertEqual(p.tick('2026-10-14')['status'], 'waiting_announcement')
            prep.assert_not_called()
    def test_before_delivery_no_send(self):
        self.state('publishing')
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 14, 9, 29, tzinfo=p.SHANGHAI)), patch.object(p, 'deliver') as send:
            p.tick('2026-10-14')
            send.assert_not_called()

    def test_pending_previous_delivery_blocks_current_day(self):
        self.state('publishing')
        p.atomic(self.data / '2026-10-16/state.json', {
            'day': '2026-10-16', 'status': 'published',
            'url': 'https://example.com/daily/2026-10-16/', 'content_id': 'today-hash', 'ids': ['2610.20000'],
        })
        with patch.object(p, 'DATA', self.data), patch.object(p, 'now', return_value=datetime(2026, 10, 16, 9, 30, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'config', return_value={'noAnnouncementDatesET': [], 'checkTime': '09:30'}), \
             patch.object(p, 'deliver_oldest_pending_before', return_value={'day': '2026-10-15', 'status': 'deferred', 'reason': 'context'}):
            result = p.tick('2026-10-16')
        self.assertEqual(result['status'], 'waiting_previous_delivery')
        self.assertEqual(result['blocked_day'], '2026-10-15')

    def test_pending_previous_delivery_preserves_diagnostics_while_preparing_today(self):
        with patch.object(p, 'DATA', self.data), \
             patch.object(p, 'now', return_value=datetime(2026, 10, 16, 9, 30, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'config', return_value={'noAnnouncementDatesET': [], 'checkTime': '09:30'}), \
             patch.object(p, 'expected_day', return_value=True), \
             patch.object(p, 'deliver_oldest_pending_before', return_value={'day': '2026-10-15', 'status': 'failed', 'reason': 'send failed'}), \
             patch.object(p, 'prepare', return_value={'day': '2026-10-16', 'status': 'pending'}) as prepare:
            result = p.tick('2026-10-16')
        self.assertEqual(result['backlog'], {'day': '2026-10-15', 'status': 'failed', 'reason': 'send failed'})
        self.assertEqual(result['status'], 'pending')
        prepare.assert_called_once_with('2026-10-16')

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
        self.assertIn('* 8-23 * * 1-5', line)

    def test_rejected_review_is_adjudicated_against_full_original(self):
        result = row()
        result['affiliation_status'] = 'unlisted'
        text = 'A video generation model. Full original including limitations.'
        ds = type('DS', (), {'settings': p.config()})()
        stages = []
        def ask(stage, system, data, validate):
            stages.append(stage)
            if stage == 'extract': return {'notes': 'notes', 'evidence': result['evidence']}
            if stage == 'interpret': return result
            if stage == 'review': return {'approved': False, 'reason': 'contradictory rejection'}
            self.assertEqual(data['paper_text'], text)
            return {'issues': [], 'reason': 'original supports the draft'}
        ds.ask = ask
        actual = p.analyze(result['paper'], {'topic': result['topic']}, text, ds)
        self.assertEqual(actual['id'], result['id'])
        self.assertEqual(stages, ['extract', 'interpret', 'review', 'adjudicate'])

    def test_adjudication_requires_literal_error_and_source_quotes(self):
        draft = {'method': 'The compared model uses 100 training steps.'}
        source = 'The compared model uses 200 training steps.'
        result = {'issues': [{'draft_quote': draft['method'], 'source_quote': source,
                              'reason': 'wrong step count', 'fix': 'write 200'}], 'reason': 'one factual error'}
        p.require_adjudication(result, draft, source)
        result['issues'][0]['draft_quote'] = 'An invented draft assertion never written.'
        with self.assertRaises(ValueError): p.require_adjudication(result, draft, source)
        with self.assertRaises(ValueError): p.require_adjudication({'approved': False, 'reason': 'no error'}, draft, source)

    def test_checkpoint_reuses_approved_result_without_model_request(self):
        result = row(); result['affiliation_status'] = 'unlisted'
        ds = type('DS', (), {'settings': p.config(), 'directory': self.data})()
        with patch.object(p, 'analyze', return_value=result) as model:
            p.analyze_checkpoint(result['paper'], {'topic': result['topic']}, 'A video generation model', ds)
            p.analyze_checkpoint(result['paper'], {'topic': result['topic']}, 'A video generation model', ds)
        model.assert_called_once()

    def test_checkpoint_failure_backs_off_and_changes_recovery_input(self):
        result = row(); result['affiliation_status'] = 'unlisted'
        ds = type('DS', (), {'settings': p.config(), 'directory': self.data})()
        with patch.object(p, 'now', return_value=datetime(2026, 10, 16, 10, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'analyze', side_effect=RuntimeError('review rejected')) as model:
            with self.assertRaises(RuntimeError): p.analyze_checkpoint(result['paper'], {'topic': result['topic']}, 'A video generation model', ds)
            with self.assertRaises(p.Pending): p.analyze_checkpoint(result['paper'], {'topic': result['topic']}, 'A video generation model', ds)
            self.assertEqual(model.call_count, 1)
        with patch.object(p, 'now', return_value=datetime(2026, 10, 16, 11, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'analyze', return_value=result) as model:
            p.analyze_checkpoint(result['paper'], {'topic': result['topic']}, 'A video generation model', ds)
            self.assertEqual(model.call_args.args[-1], 1)

    def test_preparation_collects_all_paper_failures_and_withholds_artifacts(self):
        papers = [dict(row()['paper'], id=str(i)) for i in range(3)]
        decisions = [{'id': x['id'], 'topic': row()['topic']} for x in papers]
        directory = self.data / '2026-10-16'
        settings = p.config()
        p.atomic(directory / 'candidate-snapshot.json', {'day': '2026-10-16', 'categories': settings['categories'],
                 'listing_parser_version': p.LISTING_PARSER_VERSION, 'papers': papers})
        def analyze(paper, *args):
            if paper['id'] != '0': raise RuntimeError('needs repair ' + paper['id'])
            result = copy.deepcopy(row())
            result['id'] = result['paper']['id'] = '0'
            result['affiliation_status'] = 'unlisted'
            return result
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), patch.object(p, 'config', return_value=settings), \
             patch.object(p, 'DeepSeek', return_value=object()), patch.object(p, 'classify', return_value=decisions), \
             patch.object(p, 'paper_text', return_value='paper text'), patch.object(p, 'analyze_checkpoint', side_effect=analyze), \
             patch.object(p, 'write_artifacts') as write:
            state = p.prepare('2026-10-16')
        self.assertEqual(state['approved'], 1)
        self.assertEqual(set(state['failed_papers']), {'1', '2'})
        self.assertEqual(state['status'], 'failed')
        write.assert_not_called()

    def test_old_failed_preparation_is_recovered(self):
        p.atomic(self.data / '2026-10-15/state.json', {'day': '2026-10-15', 'status': 'failed'})
        with patch.object(p, 'DATA', self.data), patch.object(p, 'prepare', return_value={'day': '2026-10-15', 'status': 'prepared'}), \
             patch.object(p, 'publish', return_value={'day': '2026-10-15', 'status': 'publishing'}), \
             patch.object(p, 'deliver', return_value={'day': '2026-10-15', 'status': 'delivered'}):
            self.assertEqual(p.deliver_oldest_pending_before('2026-10-16'), {'day': '2026-10-15', 'status': 'delivered'})

    def test_backoff_of_oldest_preparation_does_not_starve_other_runs(self):
        p.atomic(self.data / '2026-10-13/state.json', {'day': '2026-10-13', 'status': 'pending',
                                                  'retry_at': '2026-10-16T10:15:00+08:00'})
        p.atomic(self.data / '2026-10-14/state.json', {'day': '2026-10-14', 'status': 'failed'})
        with patch.object(p, 'DATA', self.data), \
             patch.object(p, 'now', return_value=datetime(2026, 10, 16, 10, tzinfo=p.SHANGHAI)):
            self.assertEqual(p.pending_delivery_before('2026-10-16')['day'], '2026-10-14')
        p.atomic(self.data / '2026-10-15/state.json', {'day': '2026-10-15', 'status': 'prepared'})
        with patch.object(p, 'DATA', self.data):
            self.assertEqual(p.pending_delivery_before('2026-10-16')['day'], '2026-10-15')

    def test_manual_source_gap_does_not_make_current_daily_health_unhealthy(self):
        p.atomic(self.data / '2026-10-14/state.json', {
            'day': '2026-10-14', 'status': 'pending', 'requires_operator': True,
            'reason': 'historical source snapshot unavailable'})
        with patch.object(p, 'DATA', self.data), \
             patch.object(p, 'now', return_value=datetime(2026, 10, 17, 9, tzinfo=p.SHANGHAI)):
            result = p.health('2026-10-17')
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['unresolved'], [])
        self.assertEqual(result['manual_review'][0]['day'], '2026-10-14')

    def test_dry_run_does_not_prepare_or_create_state(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'prepare') as prepare:
            result = p.tick('2026-10-16', True)
            p.main(['prepare', '--day', '2026-10-16', '--dry-run'])
        self.assertTrue(result['dry_run'])
        prepare.assert_not_called()
        self.assertFalse((self.data / '2026-10-16').exists())

    def test_health_reports_overdue_failed_run(self):
        p.atomic(self.data / '2026-10-15/state.json', {'day': '2026-10-15', 'status': 'failed'})
        with patch.object(p, 'DATA', self.data):
            result = p.health('2026-10-16')
        self.assertEqual(result['status'], 'unhealthy')
        self.assertEqual(result['unresolved'][0]['day'], '2026-10-15')

    def test_non_context_receipt_failure_obeys_backoff(self):
        with patch.object(p, 'now', return_value=datetime(2026, 10, 16, 10, tzinfo=p.SHANGHAI)):
            reason = p.delivery_deferred_reason({'last_error': p.WECHAT_RECEIPT_FAILURE,
                        'delivery_requires_context': False, 'delivery_retry_at': '2026-10-16T10:05:00+08:00'})
        self.assertIn('retry deferred', reason)

    def test_delay_notice_success_is_not_resent(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'receipt_status', return_value=True), \
             patch.object(p, 'now', return_value=datetime(2026, 10, 16, 10, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'run_command') as send:
            self.assertEqual(p.notify_delay('2026-10-16')['status'], 'sent')
        send.assert_not_called()

    def test_delay_notice_failure_is_backed_off(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'receipt_status', return_value=False), \
             patch.object(p, 'receipt_needs_context', return_value=False), \
             patch.object(p, 'now', return_value=datetime(2026, 10, 16, 10, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'run_command') as send:
            self.assertEqual(p.notify_delay('2026-10-16')['status'], 'failed')
            p.notify_delay('2026-10-16')
        self.assertEqual(send.call_count, 1)

    def test_watchdog_runs_without_acquiring_scheduler_lock(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'notify_delay') as notice, \
             patch.object(p, 'health', return_value={'status': 'healthy'}):
            p.main(['watch', '--day', '2026-10-16'])
        notice.assert_called_once_with('2026-10-16')
        self.assertFalse((self.data / 'scheduler.lock').exists())

    def test_watchdog_session_prefix_does_not_suppress_main_job(self):
        shim = self.data / 'tmux'
        calls = self.data / 'calls.jsonl'
        shim.write_text('#!/usr/bin/python3\nimport sys,json,os\n'
                        'if sys.argv[1] == "has-session":\n'
                        '    sys.exit(0 if sys.argv[-1] == "exec-paper-daily" else 1)\n'
                        'with open(os.environ["DAILY_TEST_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
        shim.chmod(0o755)
        subprocess.run(['bash', str(ROOT / 'scripts/daily_job.sh')], check=True,
                       env={**os.environ, 'PATH': str(self.data) + ':' + os.environ['PATH'],
                            'DAILY_TEST_CALLS': str(calls)}, capture_output=True)
        sessions = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertTrue(any(args[:4] == ['new-session', '-d', '-s', 'exec-paper-daily'] for args in sessions))

    def test_operator_retry_preserves_success_and_changes_failed_generation(self):
        directory = self.data / '2026-10-16'
        p.atomic(directory / 'state.json', {'day': '2026-10-16', 'status': 'failed', 'requires_operator': True,
                                            'source_mismatch_attempts': 3, 'source_day': '2026-10-12'})
        p.atomic(directory / 'analyses/good.json', {'status': 'approved', 'result': 'preserved'})
        p.atomic(directory / 'analyses/bad.json', {'status': 'failed', 'attempts': 3, 'retry_at': 'later'})
        with patch.object(p, 'DATA', self.data):
            state = p.retry_failed('2026-10-16')
        self.assertNotIn('requires_operator', state)
        self.assertNotIn('source_mismatch_attempts', state)
        self.assertNotIn('source_day', state)
        bad = p.read(directory / 'analyses/bad.json')
        self.assertEqual((bad['attempts'], bad['generation']), (0, 1))
        self.assertNotIn('retry_at', bad)
        self.assertEqual(p.read(directory / 'analyses/good.json')['result'], 'preserved')

    def test_unfinished_old_preparation_allows_today_delivery(self):
        self.state('published')
        with patch.object(p, 'DATA', self.data), \
             patch.object(p, 'now', return_value=datetime(2026, 10, 14, 10, tzinfo=p.SHANGHAI)), \
             patch.object(p, 'deliver_oldest_pending_before', return_value={'day': '2026-10-13', 'status': 'deferred', 'blocks_delivery': False}), \
             patch.object(p, 'deliver', return_value={'day': '2026-10-14', 'status': 'delivered'}) as send:
            result = p.tick('2026-10-14')
        self.assertEqual(result['status'], 'delivered')
        send.assert_called_once_with('2026-10-14')

    def test_failed_preparation_exits_nonzero(self):
        with patch.object(p, 'DATA', self.data), patch.object(p, 'ROOT', self.data), \
             patch.object(p, 'prepare', return_value={'day': '2026-10-16', 'status': 'failed'}):
            with self.assertRaises(SystemExit) as error:
                p.main(['prepare', '--day', '2026-10-16'])
        self.assertEqual(error.exception.code, 1)

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
    def test_paper_text_does_not_cache_non_pdf_response(self):
        directory = self.data / 'papers'
        with self.assertRaises(RuntimeError):
            p.paper_text({'id': '2609.40360', 'version': 2},
                         type('Source', (), {'get': lambda self, url, filename: b'not a pdf'})(), directory)
        self.assertFalse((directory / '2609.40360v2.pdf').exists())

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

    def test_generated_page_uses_institution_types_and_hierarchy_only(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        import validate_daily
        item = row()
        item.update({'affiliations': ['Stanford University', 'Workbuddy Team, Tencent'],
                     'affiliation_types': [
                         {'affiliation': 'Stanford University', 'type': '高校'},
                         {'affiliation': 'Workbuddy Team, Tencent', 'type': '企业'}]})
        page, _ = p.render_page('2026-10-14', [item])
        file = self.data / 'report.md'
        file.write_text(page)
        self.assertEqual(validate_daily.validate(file), [])
        file.write_text(page.replace('</article>', ''))
        self.assertTrue(validate_daily.validate(file))
        file.write_text(page.replace('data-org="高校+企业"', 'data-org="企业"'))
        self.assertTrue(validate_daily.validate(file))
        file.write_text(page.replace('data-filter-group="level"', 'data-filter-group="task"'))
        self.assertTrue(validate_daily.validate(file))
        file.write_text(page.replace('data-filter-group="level"', 'data-filter-group="coverage"'))
        self.assertTrue(validate_daily.validate(file))
        # Matching labels and aggregate fields must still fail for a clearly mistyped school.
        wrong = page.replace('高校+企业', '企业').replace('1=高校', '1=企业')
        file.write_text(wrong)
        self.assertTrue(any('Stanford University should be 高校' in e for e in validate_daily.validate(file)))
        file.write_text(page)
        self.assertEqual(validate_daily.validate(file, [item]), [])
        wrong_topic = page.replace('data-topic="视频强化学习"', 'data-topic="拟合与蒸馏"')
        wrong_topic = wrong_topic.replace('主题=`视频强化学习`', '主题=`拟合与蒸馏`')
        file.write_text(wrong_topic)
        self.assertTrue(any('topic differs from the approved report data' in e
                            for e in validate_daily.validate(file, [item])))

    def test_rendered_abstract_restores_pdf_verified_arxiv_link_fragments(self):
        item = row()
        item['paper']['summary'] = (
            'Generative gameplay requires immediate responses to player this http URL should have persistent consequences. '
            'The same head set is reused across this http URL evaluate on ECC and GoEmotions. '
            'Project page: this https URL . Code is available at this https URL.'
        )
        page, _ = p.render_page('2026-10-14', [item])
        self.assertIn('responses to player actions. Actions should have persistent consequences.', page)
        self.assertIn('reused across tasks. We evaluate SIAB on the ECC and GoEmotions.', page)
        self.assertNotIn('this https URL', page)
        self.assertEqual(p.clean_arxiv_summary('Unknown this http URL fragment.'),
                         'Unknown this http URL fragment.')

    def test_analysis_rejects_negative_contrast_style(self):
        item = row()
        item.update({'short': 'Video: useful', 'level': '垂类'})
        source = 'A video generation model with useful evidence. ' * 8
        item['problem'] = '该方法并非只依赖文本，而是结合视频证据进行推理。'
        item['evidence'] = [{'claim': 'method', 'quote': 'A video generation model'}]
        with self.assertRaisesRegex(ValueError, 'affirmatively'):
            p.analysis_valid(item, source)
        item['problem'] = '该结论并非在所有实验设置中成立。' + '论文分别报告不同数据集与评估条件下的结果，适用范围由对应证据确定。'
        item['affiliation_status'] = 'unlisted'
        p.analysis_valid(item, source)

    def test_source_validation_rejects_affiliations_not_present_in_pdf(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        import validate_daily
        item = row()
        item.update({'affiliations': ['Applied AI Institute'],
                     'affiliation_types': [{'affiliation': 'Applied AI Institute', 'type': '科研机构'}]})
        page, _ = p.render_page('2026-10-16', [item])
        report = self.data / 'report.md'
        report.write_text(page)
        papers = self.data / 'papers'; papers.mkdir()
        (papers / '2609.40360v1-raw.txt').write_text(
            'Department of Computer Science, Stanford University. ' * 30)
        errors = validate_daily.validate(report, [item], papers)
        self.assertTrue(any('author unit is absent from the source PDF' in error for error in errors))

    def test_approved_checkpoint_must_match_report_and_page(self):
        item = row()
        item.update({'affiliation_status': 'unlisted'})
        directory = self.data / '2026-10-14'
        artifacts = p.write_artifacts('2026-10-14', [item], directory, p.config())
        state = {'day': '2026-10-14', 'status': 'prepared', **artifacts}
        analysis_dir = directory / 'analyses'; analysis_dir.mkdir()
        p.atomic(analysis_dir / f'{item["id"]}.json', {'status': 'approved', 'result': item})
        papers = directory / 'papers'; papers.mkdir()
        (papers / f'{item["id"]}v1-raw.txt').write_text('A video generation model. ' * 10)
        p.validate_artifacts('2026-10-14', directory, state)
        report = p.read(directory / 'report.json')
        report[0]['topic'] = 'wrong paper topic'
        p.atomic(directory / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'approved checkpoint'):
            p.validate_artifacts('2026-10-14', directory, state)

    def test_repair_rebuilds_delivered_rows_from_checkpoints_by_id(self):
        day = '2026-10-16'
        directory = self.data / day
        first = copy.deepcopy(row())
        first['id'] = first['paper']['id'] = '2610.10001'
        first['paper']['title'] = 'First paper title'
        first['paper']['announcement_date'] = day
        first.update({'task': 'first task', 'coverage': '跨任务通用',
                      'affiliation_status': 'unlisted'})
        first.pop('level')
        second = copy.deepcopy(row())
        second['id'] = second['paper']['id'] = '2610.10002'
        second['paper']['title'] = 'Second paper title'
        second['paper']['announcement_date'] = day
        second.update({'task': 'second task', 'coverage': '细分任务',
                       'affiliation_status': 'unlisted'})
        second.pop('level')
        # The prior aggregate was assembled in completion order. The state and
        # checkpoint IDs are the authoritative order and identity.
        p.atomic(directory / 'state.json', {
            'day': day, 'status': 'delivered', 'ids': [second['id'], first['id']],
            'content_id': 'old-content', 'delivered_at': '2026-10-16T09:30:00+08:00',
            'delivery_attempts': 1,
        })
        p.atomic(directory / 'report.json', [first, second])
        analyses = directory / 'analyses'
        papers = directory / 'papers'
        for item in (first, second):
            p.atomic(analyses / f"{item['id']}.json", {
                'status': 'approved', 'result': item,
            })
            p.atomic(papers / f"{item['id']}v1-raw.txt", 'A video generation model. ' * 10)

        with patch.object(p, 'DATA', self.data):
            repaired = p.repair(day)

        report = p.read(directory / 'report.json')
        self.assertEqual([item['id'] for item in report], [second['id'], first['id']])
        self.assertEqual([item['paper']['title'] for item in report],
                         ['Second paper title', 'First paper title'])
        self.assertEqual([item['level'] for item in report], ['垂类', '通用'])
        self.assertTrue(all('task' not in item and 'coverage' not in item for item in report))
        self.assertEqual(repaired['status'], 'delivered')
        self.assertEqual(repaired['delivered_at'], '2026-10-16T09:30:00+08:00')
        self.assertEqual(repaired['delivery_attempts'], 1)
        p.validate_artifacts(day, directory, repaired)

    def test_repair_recovers_delivered_report_without_checkpoints_from_pdf_text(self):
        day = '2026-10-15'
        item = copy.deepcopy(row())
        item['id'] = item['paper']['id'] = '2610.10003'
        item['paper']['announcement_date'] = day
        item['affiliation_status'] = 'unlisted'
        directory = self.data / day
        papers = directory / 'papers'
        papers.mkdir(parents=True)
        (papers / '2610.10003v1.pdf').write_bytes(b'%PDF-test')
        (papers / '2610.10003v1-raw.txt').write_text('A video generation model. ' * 10)
        p.atomic(directory / 'report.json', [item])
        p.atomic(directory / 'state.json', {
            'day': day, 'status': 'delivered', 'ids': [item['id']],
            'delivered_at': '2026-10-15T09:30:00+08:00', 'delivery_attempts': 1,
        })

        with patch.object(p, 'DATA', self.data):
            repaired = p.repair(day)

        self.assertEqual(repaired['status'], 'delivered')
        self.assertEqual(repaired['artifact_verification'], 'delivered_report_and_pdf_text')
        self.assertEqual(repaired['delivery_attempts'], 1)
        self.assertEqual(repaired['delivered_at'], '2026-10-15T09:30:00+08:00')
        self.assertFalse((directory / 'analyses').exists())
        p.validate_artifacts(day, directory, repaired)

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
            item = row(); item['affiliation_status'] = 'unlisted'
            artifacts = p.write_artifacts('2026-10-14', [item], directory, p.config())
            p.atomic(directory / 'state.json', {'day': '2026-10-14', 'status': 'prepared', **artifacts})
            p.atomic(directory / 'analyses/2609.40360.json', {'status': 'approved', 'result': item})
            (directory / 'papers').mkdir()
            (directory / 'papers/2609.40360v1-raw.txt').write_text('A video generation model.')
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
