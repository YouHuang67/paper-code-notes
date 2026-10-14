#!/usr/bin/env python3
"""Verify real classification and full-text interpretation without push or send.

Requires the official research snapshot acquired under its robots policy.
Audit outputs stay in the ignored refs/scans/daily-runs/smoke directory.
"""
import daily_pipeline as pipeline
import validate_daily


def main():
    snapshot = pipeline.ROOT / 'refs/research/daily-delivery-pipeline/crawl/results/pages/arxiv.org__list__cs__new__8f1c8d7501b9b49e.html'
    settings = pipeline.config()
    candidates = pipeline.parse_announcements(snapshot.read_text(), '2026-10-12', settings['categories'])
    ids = {'2610.13205', '2610.13232', '2610.13708'}
    papers = [p for p in candidates if p['id'] in ids]
    if len(papers) != 3:
        raise RuntimeError('expected official positive and negative samples missing')
    directory = pipeline.DATA / 'smoke'
    directory.mkdir(parents=True, exist_ok=True)
    pipeline.atomic(directory / 'candidates.json', papers)
    ds = pipeline.DeepSeek(directory, settings)
    decisions = pipeline.classify(papers, settings, ds)
    if [r['id'] for r in decisions] != ['2610.13205']:
        raise RuntimeError('known OPD positive / driving and KV-cache negatives failed')
    paper = next(p for p in papers if p['id'] == decisions[0]['id'])
    text = pipeline.paper_text(paper, pipeline.ArxivClient(directory / 'source'), directory / 'papers')
    result = pipeline.analyze(paper, decisions[0], text, ds)
    artifacts = pipeline.write_artifacts('2026-10-12', [result], directory, settings)
    errors = validate_daily.validate(directory / 'report.md')
    if errors:
        raise RuntimeError('; '.join(errors))
    pipeline.atomic(directory / 'result.json', {'status': 'smoke_passed', 'source_characters': len(text), **artifacts})
    print('smoke_passed:', paper['id'], len(text), 'source characters')


if __name__ == '__main__':
    main()
