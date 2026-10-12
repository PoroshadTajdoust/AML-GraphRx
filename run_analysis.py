"""Run exactly one manuscript-linked analysis against an existing project."""
from pathlib import Path
import argparse, json, os, subprocess, sys

def main():
    root=Path(__file__).resolve().parent
    catalog=json.loads((root/'config/analyses.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis',nargs='?',choices=[r['id'] for r in catalog])
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--project-root',type=Path)
    parser.add_argument('--beataml-dir',type=Path)
    args=parser.parse_args()
    if args.list:
        for row in catalog: print(row['id']+' — '+row['manuscript_location'])
        return 0
    if not args.analysis or args.project_root is None:
        parser.error('Choose an analysis and provide --project-root.')
    project=args.project_root.expanduser().resolve()
    if not project.is_dir() or not (project/'results/audit').is_dir():
        parser.error('Expected an existing AML_GraphRx project with results/audit; see docs/INPUTS.md.')
    env=os.environ.copy();env['AML_GRAPHRX_ROOT']=str(project)
    if args.beataml_dir is not None:
        cache=args.beataml_dir.expanduser().resolve()
        if not cache.is_dir():parser.error('--beataml-dir must be an existing directory.')
        env['AML_GRAPHRX_BEATAML_DIR']=str(cache)
    row=next(x for x in catalog if x['id']==args.analysis)
    return subprocess.call([sys.executable,str(root/row['file'])],env=env)

if __name__=='__main__':raise SystemExit(main())
