"""Run the actual-renderer Chromium checks; requires local Node + Playwright.
Usage: python web/admin/frontend/src/components/test_xss_browser.py [--api-replay FILE]
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import xss_fixtures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-replay', type=Path)
    args=parser.parse_args()
    if args.api_replay and not args.api_replay.is_file(): parser.error('API replay file does not exist')
    sys.argv=[sys.argv[0]]+([str(args.api_replay)] if args.api_replay else [])
    output=xss_fixtures.ROOT/'output/xss-tests/frontend-fixtures.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(xss_fixtures.fixtures()), encoding='utf-8')
    bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node=shutil.which('node') or str(bundled/'bin/node.exe')
    env=dict(os.environ)
    if (bundled/'node_modules/playwright').is_dir(): env.setdefault('NODE_PATH', str(bundled/'node_modules'))
    subprocess.run([node,str(Path(__file__).with_name('test_xss_dom.cjs')),str(output)],env=env,check=True)
    print('Persisted API replay included:',bool(args.api_replay))


if __name__=='__main__': main()
