"""Extract explicit file-access attempts, not prose mentioning a protected path.

This is transcript evidence, not an OS access log. The sandbox remains the
access-control boundary; uncertain references must not be reported as reads.
"""
import ast
import json
import re
import shlex


READ_CALLS = {"open", "read_text", "read_bytes", "read_csv", "read_json", "read_jsonl",
              "read_parquet", "load_dataset", "load_from_disk", "load", "loadtxt", "genfromtxt"}


def python_reads(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name not in READ_CALLS:
            continue
        # open(path, 'w') is not evidence of reading the old file.
        if name == "open":
            mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and mode.value[:1] in {'w', 'a', 'x'} and '+' not in mode.value:
                continue
        values.extend(child.value for child in ast.walk(node)
                      if isinstance(child, ast.Constant) and isinstance(child.value, str))
    return values


def shell_reads(command: str) -> list[str]:
    values = []
    # Analyze embedded Python structurally: comments and print strings are inert.
    for match in re.finditer(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\1(?:\n|$)", command, re.S):
        values.extend(python_reads(match.group(2)))
    command = re.sub(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\1(?:\n|$)", "", command, flags=re.S)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = '#'
        tokens = list(lexer)
    except ValueError:
        return values
    segments, current = [], []
    for token in tokens:
        if token in {';', '&&', '||', '|', '&'}:
            segments.append(current); current = []
        else:
            current.append(token)
    segments.append(current)
    for tokens in segments:
        if not tokens:
            continue
        program = tokens[0].rsplit('/', 1)[-1]
        if program in {'python', 'python3'} and '-c' in tokens:
            at = tokens.index('-c') + 1
            if at < len(tokens):
                values.extend(python_reads(tokens[at]))
        elif program in {'cat', 'head', 'tail', 'less', 'more', 'sed', 'awk', 'grep', 'rg',
                         'cut', 'wc', 'sort', 'jq', 'cp', 'scp', 'rsync'}:
            # Programs/regexes are text, not input filenames. Account for the
            # common explicit-script/pattern options before collecting files.
            has_program = program in {'sed', 'awk', 'grep', 'rg', 'jq'}
            positional_program = has_program
            skip_value = False
            read_value = False
            for token in tokens[1:]:
                if skip_value:
                    if read_value:
                        values.append(token)
                    skip_value = read_value = False
                    continue
                if token in {'-e', '--regexp', '--expression'} and has_program:
                    positional_program = False
                    skip_value = True
                    continue
                if token == '-f' and has_program:
                    positional_program = False
                    skip_value = read_value = True
                    continue
                if token in {'--files', '--files-with-matches'} and program == 'rg':
                    if token == '--files':
                        positional_program = False
                    continue
                if token in {'-F', '--field-separator'} and program == 'awk':
                    skip_value = True
                    continue
                if token.startswith('-'):
                    continue
                if positional_program:
                    positional_program = False
                    continue
                values.append(token)
        # Redirection explicitly opens a file regardless of the command name.
        for index, token in enumerate(tokens[:-1]):
            if token == '<':
                values.append(tokens[index + 1])
    return values


def access_arguments(event: dict) -> list[str]:
    payload = event.get('payload') or {}
    raw = payload.get('input') or {key: payload[key] for key in ('command', 'cmd') if key in payload}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return shell_reads(raw)
        raw = parsed if isinstance(parsed, dict) else {}
    if not isinstance(raw, dict):
        return []
    name = str(payload.get('name') or payload.get('tool') or payload.get('tool_name') or '').lower()
    if name in {'read', 'readfile'} or any(key in name for key in ('read_file', 'view_file', 'read_text')):
        return [str(raw.get(k) or '') for k in ('path', 'file_path', 'filename')]
    values = []
    for key in ('command', 'cmd'):
        if isinstance(raw.get(key), str):
            values.extend(shell_reads(raw[key]))
    if isinstance(raw.get('code'), str):
        values.extend(python_reads(raw['code']))
    return values
