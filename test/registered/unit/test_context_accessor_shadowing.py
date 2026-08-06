"""No local may shadow a ``runtime_context`` accessor it also calls.

A mechanical sweep that rewrites ``self.server_args.mamba_cache_chunk_size``
into ``mamba_cache_chunk_size()`` turns

    mamba_cache_chunk_size = self.server_args.mamba_cache_chunk_size

into ``mamba_cache_chunk_size = mamba_cache_chunk_size()``, which is a
self-referential local: the name is local for the whole function, so the call
raises ``UnboundLocalError`` the first time that line runs. Five of these
shipped in one sweep and only one had unit coverage — a mamba model on the
radix-cache-v2 path found it at request time.

This scans for the shape directly: a function-scope assignment whose target
name is an imported accessor.
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))
_CONTEXT_MODULE = "sglang.srt.runtime_context"


def _imported_accessors(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _CONTEXT_MODULE:
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _bound_names(target):
    """Every name a binding target introduces, unpacking included.

    ``a, (b, c) = ...`` and ``for x, y in ...`` bind through Tuple/List/Starred
    nodes, so a check that only accepts a bare ``ast.Name`` misses them.
    """
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound_names(element)


def _shadowing_assignments(tree: ast.AST, accessors: set[str]):
    """Function-local bindings whose name shadows an accessor the module
    imported -- every statement form that binds a local, not just ``=``.

    Python decides a name is local from *any* binding in the function, so a
    loop variable, a ``with ... as``, a walrus, a comprehension target, or an
    ``except ... as`` all shadow the accessor for the whole function body,
    exactly like an assignment does.

    A function-scope *import* of the accessor is not in here: it binds the name
    to the same callable, so calls after it behave identically (and the module
    is full of deliberate local imports).

    Each function is scanned for its OWN bindings only: a nested function's
    local is that scope's binding, not its parent's, and the outer
    ``ast.walk(tree)`` visits the nested def itself — descending here would
    misattribute the offense to the enclosing function and report it twice.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        own_scope = []
        pending = list(node.body)
        while pending:
            stmt = pending.pop()
            if isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                continue
            own_scope.append(stmt)
            pending.extend(ast.iter_child_nodes(stmt))
        for inner in own_scope:
            targets = []
            if isinstance(inner, ast.Assign):
                targets = inner.targets
            elif isinstance(inner, (ast.AnnAssign, ast.AugAssign)):
                targets = [inner.target]
            elif isinstance(inner, (ast.For, ast.AsyncFor, ast.comprehension)):
                targets = [inner.target]
            elif isinstance(inner, ast.NamedExpr):
                targets = [inner.target]
            elif isinstance(inner, (ast.With, ast.AsyncWith)):
                targets = [i.optional_vars for i in inner.items if i.optional_vars]
            elif isinstance(inner, ast.ExceptHandler) and inner.name:
                targets = [ast.Name(id=inner.name, ctx=ast.Store())]
            for target in targets:
                for name in _bound_names(target):
                    if name in accessors:
                        yield node.name, name, getattr(inner, "lineno", node.lineno)


class TestNoAccessorShadowing(CustomTestCase):
    def test_no_local_shadows_a_context_accessor(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith("srt/runtime_context.py"):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            accessors = _imported_accessors(tree)
            if not accessors:
                continue
            for func, name, lineno in _shadowing_assignments(tree, accessors):
                offenders.append(f"{rel}:{lineno}: {func}() binds {name!r}")
        self.assertFalse(
            offenders,
            "locals shadow a runtime_context accessor imported in the same "
            "module; the name is local for the whole function, so any call to "
            "the accessor there raises UnboundLocalError:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
