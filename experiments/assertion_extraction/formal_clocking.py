"""Validated clock declarations shared by reduction and checker coverage.

An optional audited case-level context overrides automatically collected event
signals. In particular, an asynchronous reset event must not automatically
turn a reset input into a periodic formal clock. Compatible with Python 3.6.
"""
import re


def clock_commands(clocks, context=None):
    if context is None:
        declarations = [{'signal': name, 'edges': 'rising'} for name in clocks]
    else:
        if not isinstance(context, dict) or context.get('version') != 1:
            raise ValueError('Unsupported formal_clocking context')
        declarations = context.get('declarations')
        if not isinstance(declarations, list) or not declarations:
            raise ValueError('formal_clocking requires explicit clock declarations')
    seen = set()
    commands = []
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError('Invalid formal clock declaration')
        name, edges = declaration.get('signal'), declaration.get('edges')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_$][\w$]*', name):
            raise ValueError('Invalid formal clock identifier: {!r}'.format(name))
        if name in seen or edges not in ('rising', 'both'):
            raise ValueError('Duplicate clock or unsupported edge policy: ' + name)
        seen.add(name)
        commands.append('clock ' + ('-both_edges ' if edges == 'both' else '') + name)
    return '\n'.join(commands) if commands else 'clock -none'


def context_from_case(path):
    import json
    from pathlib import Path
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    context = doc.get('formal_clocking') if isinstance(doc, dict) else None
    if context is not None:
        clock_commands([], context)  # Validate before inserting commands into Tcl.
    return context
