#!/usr/bin/env python3
"""Package a validated release with portable roots and per-file SHA-256 manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def included(path):
    return (path.is_file() and not path.is_symlink()
            and not any(p == '__pycache__' or p.endswith('.egg-info') or p in ('.venv', '.git')
                        for p in path.relative_to(ROOT).parts)
            and path.suffix not in ('.pyc', '.aux', '.out', '.tmp')
            and path.name not in ('MANIFEST.sha256', 'main_paper.log'))


def archive(source, root_name, target):
    paths = sorted(p for p in source.rglob('*') if included(p))
    manifest = ''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  '
                       + p.relative_to(source).as_posix() + '\n' for p in paths)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, root_name + '/' + p.relative_to(source).as_posix())
        z.writestr(root_name + '/MANIFEST.sha256', manifest)
    # Check every stored member against its content hash, including decompression.
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP integrity failure: ' + str(target))
        for line in manifest.splitlines():
            digest, rel = line.split('  ', 1)
            if hashlib.sha256(z.read(root_name + '/' + rel)).hexdigest() != digest:
                raise RuntimeError('Archive hash mismatch: ' + rel)
    return dict(filename=target.name, files=len(paths)+1, bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--destination', type=Path, required=True)
    args = ap.parse_args()
    destination = args.destination.resolve()
    if destination.is_relative_to(ROOT):
        raise SystemExit('Destination must be outside the project root')
    integrity = json.loads((ROOT/'outputs/analysis/metadata/integrity_report.json').read_text())
    phases = ['development_optimizer','development_baselines','development_scaling','final_study','vqe_convergence']
    if integrity['status'] != 'PASS' or not all((ROOT/'outputs'/phase/'RUN_COMPLETE').exists() for phase in phases):
        raise SystemExit('Verified completed run required')
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for source, root_name, name in [
        (ROOT, 'D-NCQAS', 'D_NCQAS_Revision3_Project.zip'),
        (ROOT/'outputs', 'D-NCQAS_outputs', 'D_NCQAS_Revision3_Outputs.zip'),
        (ROOT/'supporting_assets', 'Supporting_Assets', 'D_NCQAS_Revision3_Supporting_Assets.zip')]:
        results.append(archive(source, root_name, destination/name))
    (destination/'archive_checks.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
