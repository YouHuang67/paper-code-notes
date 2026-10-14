"""Regression checks for source freshness and publication/delivery gates."""
import copy
import importlib.util
import json
import tempfile
import io
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('daily_pipeline', ROOT / 'scripts/daily_pipeline.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def listing(day='Thursday, 01 October 2026', shown=1, total=1):
    return f'''<h3>Showing new listings for {day}</h3><dl>
<h3>New submissions (showing {shown} of {total} entries)</h3>
<dt><a href="/abs/2609.40360">id</a></dt><dd>
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
    def test_page_fingerprint_stable(self):
        rows = [row()]
        a, ident = p.render_page('2026-10-14', rows)
        b, other = p.render_page('2026-10-14', rows)
        self.assertEqual(ident, other)
        self.assertEqual(a, b)
        self.assertIn('daily-content-' + ident, a)
        self.assertIn('daily-filter', a)
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

    def test_pdf_line_break_normalization_preserves_numbers(self):
        p.evidence_valid({'evidence': [{'claim': 'data', 'quote': 'Training on DeepMath-103K gives 35.5 percent.'}]},
                         'Training on DeepMath- 103K gives 35.5 percent.')
        with self.assertRaises(ValueError):
            p.evidence_valid({'evidence': [{'claim': 'data', 'quote': 'Training on DeepMath-103K gives 55.5 percent.'}]},
                             'Training on DeepMath- 103K gives 35.5 percent.')

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


if __name__ == '__main__':
    unittest.main()
