"""清理元数据并导出最终匿名PDF。"""
from __future__ import annotations
import shutil, zipfile, tempfile
from pathlib import Path
import win32com.client as win32
import fitz

BASE = Path(r'D:\数学建模\数学建模B2-RL')
PAPER = BASE / '论文'
SRC_DOCX = PAPER / 'B题_方案B_强化学习_论文稿_v4.docx'
FINAL_DOCX = PAPER / 'B题_方案B_强化学习_论文稿_匿名终稿.docx'
TMP_PDF = PAPER / '_final_tmp.pdf'
FINAL_PDF = PAPER / 'B题_方案B_强化学习_论文稿_匿名终稿.pdf'

def clean_docx(src: Path, dst: Path):
    with zipfile.ZipFile(src, 'r') as zin:
        names = set(zin.namelist())
        remove_names = set()
        for candidate in ['word/comments.xml','word/commentsExtended.xml','word/commentsIds.xml',
                          'word/commentsExtensible.xml','word/people.xml']:
            if candidate in names:
                remove_names.add(candidate)
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in remove_names:
                    continue
                data = zin.read(item.filename)
                name = item.filename
                if name == 'docProps/core.xml':
                    s = data.decode('utf-8')
                    for tag in ['dc:title','dc:creator','cp:lastModifiedBy','cp:keywords',
                                'dc:subject','dc:description','cp:category']:
                        import re
                        s = re.sub(r'<%s>.*?</%s>' % (tag, tag), '<%s></%s>' % (tag, tag), s, flags=re.S)
                    s = re.sub(r'<dcterms:created[^>]*>.*?</dcterms:created>', '', s, flags=re.S)
                    s = re.sub(r'<dcterms:modified[^>]*>.*?</dcterms:modified>', '', s, flags=re.S)
                    data = s.encode('utf-8')
                elif name == 'docProps/app.xml':
                    s = data.decode('utf-8')
                    import re
                    for tag in ['Application','Company','Manager','Template','HyperlinkBase']:
                        if '<%s/>' % tag in s:
                            s = s.replace('<%s/>' % tag, '<%s></%s>' % (tag, tag))
                        else:
                            s = re.sub(r'<%s>.*?</%s>' % (tag, tag), '<%s></%s>' % (tag, tag), s, flags=re.S)
                    data = s.encode('utf-8')
                elif name == 'word/settings.xml':
                    s = data.decode('utf-8')
                    import re
                    s = re.sub(r'<w:rsids>.*?</w:rsids>', '', s, flags=re.S)
                    data = s.encode('utf-8')
                elif name == 'word/document.xml':
                    s = data.decode('utf-8')
                    import re
                    for tag in ['w:commentRangeStart','w:commentRangeEnd','w:commentReference',
                                'w:ins','w:del','w:moveFrom','w:moveTo','w:rPrChange','w:pPrChange',
                                'w:tblPrChange','w:trPrChange','w:tcPrChange','w:sectPrChange']:
                        s = re.sub(r'<%s[^>]*>.*?</%s>' % (tag, tag), '', s, flags=re.S)
                        s = re.sub(r'<%s[^>]*/>' % tag, '', s, flags=re.S)
                    data = s.encode('utf-8')
                zout.writestr(item, data)
    print('cleaned docx ->', dst)

def export_pdf(docx: Path, pdf: Path):
    app = win32.DispatchEx('KWPS.Application')
    app.Visible = False; app.DisplayAlerts = False
    doc = None
    try:
        doc = app.Documents.Open(str(docx), ReadOnly=False)
        try: doc.Fields.Update()
        except Exception: pass
        doc.SaveAs2(str(pdf), 17)
        doc.Close(False); doc = None
    finally:
        try:
            if doc is not None: doc.Close(False)
        except Exception: pass
        try: app.Quit()
        except Exception: pass
    print('exported pdf ->', pdf)

def clean_pdf(src: Path, dst: Path):
    d = fitz.open(src)
    meta = {'title':'','author':'','subject':'','keywords':'','creator':'','producer':'',
            'creationDate':'','modDate':'','trapped':''}
    d.set_metadata(meta)
    try:
        d.set_xml_metadata('')
    except Exception:
        try: d.del_xml_metadata()
        except Exception: pass
    d.save(dst, garbage=4, deflate=True, clean=True)
    d.close()
    print('cleaned pdf ->', dst)

if __name__ == '__main__':
    clean_docx(SRC_DOCX, FINAL_DOCX)
    if TMP_PDF.exists(): TMP_PDF.unlink()
    export_pdf(FINAL_DOCX, TMP_PDF)
    if FINAL_PDF.exists(): FINAL_PDF.unlink()
    clean_pdf(TMP_PDF, FINAL_PDF)
    # 验证
    import json
    d = fitz.open(FINAL_PDF)
    print('final pdf pages', d.page_count)
    print('metadata', json.dumps(d.metadata, ensure_ascii=False))
    xmp = ''
    try: xmp = d.get_xml_metadata()
    except Exception: pass
    print('xmp length', len(xmp or ''))
    d.close()
    # 复制到项目根
    shutil.copyfile(FINAL_DOCX, BASE / FINAL_DOCX.name)
    shutil.copyfile(FINAL_PDF, BASE / FINAL_PDF.name)
    print('copied to root')