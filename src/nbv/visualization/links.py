"""Markdown link checks against a supplied tracked-file inventory only."""
import posixpath
import re
from urllib.parse import unquote,urlsplit


def broken_links(texts,tracked,exceptions=()):
    tracked=set(tracked);skip={(e['file'],e['link']) for e in exceptions};bad=[]
    for name,text in sorted(texts.items()):
        text=re.sub(r'(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$','',text)
        for match in re.finditer(r'!?\[[^\]]*\]\(([^\n]+?)\)',text):
            link=match.group(1).strip()
            link=link[1:link.index('>')] if link.startswith('<') else link.split(' "')[0]
            if not link or link.startswith('#') or urlsplit(link).scheme or (name,link) in skip:continue
            path=unquote(link.split('#')[0].split('?')[0])
            target=posixpath.normpath(posixpath.join(posixpath.dirname(name),path))
            if target not in tracked and not any(p.startswith(target.rstrip('/')+'/') for p in tracked):bad.append((name,link))
    return bad
