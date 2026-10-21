"""Strict source checks with one explicit, non-numerical release-version migration."""
import hashlib

OLD_INIT = b'"""D-NCQAS: corrected, controlled simulation research implementation."""\n__version__ = "2.0.0"\n'
NEW_INIT = OLD_INIT.replace(b'"2.0.0"', b'"3.0.0"')
VERSION_PATH = 'src/d_ncqas/__init__.py'


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'src').rglob('*.py'))}


def verify_sources(root, expected):
    """Reject every change except the exact two-line version-only migration."""
    actual = source_hashes(root)
    if set(actual) != set(expected):
        raise RuntimeError('Scientific source file set changed')
    migrations = []
    for rel, digest in expected.items():
        if actual[rel] == digest:
            continue
        if (rel == VERSION_PATH
                and digest == hashlib.sha256(OLD_INIT).hexdigest()
                and (root / rel).read_bytes() == NEW_INIT):
            migrations.append(dict(path=rel, experiment_sha256=digest,
                                   release_sha256=actual[rel],
                                   change='Only __version__: 2.0.0 -> 3.0.0'))
        else:
            raise RuntimeError('Scientific source mismatch: ' + rel)
    return migrations
