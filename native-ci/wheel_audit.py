"""Full-wheel SHA256, RECORD, PE inventory and import names (pefile)."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import zipfile
import pefile

rows=[]
for path in Path(sys.argv[1]).glob('*.whl'):
    row=dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pe=[], record_checked=0)
    with zipfile.ZipFile(path) as z:
        for member, digest, size in csv.reader(io.StringIO(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))).decode())):
            if not digest:
                continue
            algorithm, wanted=digest.split('=',1)
            data=z.read(member)
            actual=base64.urlsafe_b64encode(hashlib.new(algorithm,data).digest()).rstrip(b'=').decode()
            assert actual==wanted, member
            assert len(data)==int(size), member
            row['record_checked']+=1
        for n in z.namelist():
            if n.endswith(('.pyd','.dll','.exe')):
                p=pefile.PE(data=z.read(n))
                imports=[d.dll.decode() for d in getattr(p,'DIRECTORY_ENTRY_IMPORT',[])]
                row['pe'].append(dict(path=n,machine=hex(p.FILE_HEADER.Machine),imports=imports))
        row['metadata']=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        row['wheel']=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
    rows.append(row)
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2),encoding='utf-8')
assert rows
for row in rows:
    print(row['file'],row['sha256'],'RECORD',row['record_checked'])
    for p in row['pe']:
        print(p)
        assert p['machine']=='0xaa64',p
