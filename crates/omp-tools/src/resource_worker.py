import io, json, tarfile, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, parse_qs
import time

LIMIT = 24000

def materialize(resource):
    path = Path(resource['path'])
    selector = resource.get('selector') or {}
    member = unquote(selector.get('subpath') or '') or None
    query = unquote(selector.get('query') or '') or None
    suffix = path.name.lower()
    def bounded(stream):
        data = stream.read(LIMIT + 1)
        return {'content': data[:LIMIT].decode('utf-8', 'replace'), 'truncated': len(data) > LIMIT}
    if suffix.endswith(('.sqlite', '.sqlite3', '.db')):
        import sqlite3
        db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        db.execute('PRAGMA query_only=ON')
        deadline = time.monotonic() + 5
        db.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1000000)
        db.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action in (
            sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH) else sqlite3.SQLITE_OK)
        schema = None
        if query:
            cursor = db.execute(query)
        elif member:
            table = member.split(':', 1)[0]
            escaped = table.replace('"', '""')
            schema = [dict(zip(('cid', 'name', 'type', 'notnull', 'default', 'pk'), row))
                      for row in db.execute('PRAGMA table_info("' + escaped + '")')]
            key = member.split(':', 1)[1] if ':' in member else None
            if key is not None:
                keys = [column['name'] for column in schema if column['pk']]
                if len(keys) != 1: raise ValueError('Key selector requires one primary-key column')
                cursor = db.execute('SELECT * FROM "' + escaped + '" WHERE "' +
                                    keys[0].replace('"', '""') + '" = ? LIMIT 101', (key,))
            else:
                cursor = db.execute('SELECT * FROM "' + escaped + '" LIMIT 101')
        else:
            cursor = db.execute("SELECT name, type, sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY name")
        columns = [column[0] for column in cursor.description or []]
        rows, size, truncated = [], 0, False
        for row in cursor:
            record = {key: (value.hex() if isinstance(value, bytes) else value) for key, value in zip(columns, row)}
            encoded = json.dumps(record)
            if len(rows) >= 100 or size + len(encoded) > LIMIT:
                truncated = True
                break
            rows.append(record)
            size += len(encoded)
        db.close()
        return {'schema':schema, 'columns':columns, 'rows':rows, 'truncated':truncated}
    if suffix.endswith('.asar'):
        with path.open('rb') as stream:
            prefix = stream.read(16)
            if len(prefix) != 16: raise ValueError('Truncated ASAR header')
            header_size = int.from_bytes(prefix[4:8], 'little')
            json_size = int.from_bytes(prefix[12:16], 'little')
            if json_size > 1000000: raise ValueError('ASAR header too large')
            tree = json.loads(stream.read(json_size))
            if not member: return tree
            node = tree
            for part in member.replace('\\', '/').split('/'):
                if part in ('', '.', '..'): raise ValueError('Invalid archive member')
                node = node['files'][part]
            if node.get('unpacked') or 'link' in node: raise ValueError('External ASAR member is not permitted')
            size = int(node['size'])
            stream.seek(8 + header_size + int(node['offset']))
            return bounded(io.BytesIO(stream.read(min(size, LIMIT + 1))))
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if suffix.endswith(('.docx', '.pptx', '.xlsx', '.epub')) and not member:
                names = [name for name in archive.namelist() if name.endswith(('.xml', '.xhtml', '.html')) and (
                    name == 'word/document.xml' or name.startswith(('ppt/slides/slide', 'xl/sharedStrings', 'xl/worksheets/')) or suffix.endswith('.epub'))]
                parts, size = [], 0
                for name in sorted(names):
                    info = archive.getinfo(name)
                    if info.file_size > 2000000: raise ValueError('Document XML member exceeds budget')
                    tree = ET.fromstring(archive.read(name))
                    for text in tree.itertext():
                        if size + len(text) > LIMIT: return {'content':'\n'.join(parts), 'truncated':True}
                        parts.append(text)
                        size += len(text)
                return {'content':'\n'.join(parts), 'truncated':False}
            if member:
                with archive.open(member) as stream: return bounded(stream)
            entries = [{'name':item.filename, 'bytes':item.file_size} for item in archive.infolist()[:1000]]
            return {'entries':entries, 'truncated':len(archive.infolist()) > 1000}
    with tarfile.open(path, 'r:*') as archive:
        if member:
            item = archive.getmember(member)
            if not item.isfile(): raise ValueError('Only regular archive members can be read')
            with archive.extractfile(item) as stream: return bounded(stream)
        entries = []
        for item in archive:
            if len(entries) == 1000: return {'entries':entries, 'truncated':True}
            entries.append({'name':item.name, 'bytes':item.size})
        return {'entries':entries, 'truncated':False}

result = materialize(RESOURCE)
encoded = json.dumps(result, ensure_ascii=False)
while len(encoded.encode('utf-8')) > 90000:
    if 'content' in result: result['content'] = result['content'][:len(result['content']) // 2]
    elif result.get('rows'): result['rows'].pop()
    elif result.get('entries'): result['entries'].pop()
    else: raise ValueError('Resource metadata exceeds output budget')
    result['truncated'] = True
    encoded = json.dumps(result, ensure_ascii=False)
print(encoded)
