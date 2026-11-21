import hashlib, io, zipfile
from pathlib import Path
from urllib.request import urlopen
import torch

BOOKS={'Emma':('austen-emma','71b0923d75fcf2f1faa897b22da8de8e2476d7478a2b22388eb58a3420e00215'),
       'Moby-Dick':('melville-moby_dick','dc2466ee3c2e6e2ea1f86233d142e4b5578fdc1d6f0ca657bee7572202a980c7')}
URLS=['https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip',
      'https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/corpora/gutenberg.zip']
def texts(tokenizer,paths,offset,maxlen):
    if paths:
        inputs=[(Path(p).name,Path(p).read_bytes(),'utf-8',str(Path(p).name)) for p in paths]
    else:
        issues=[]
        for url in URLS:
            try:
                with urlopen(url,timeout=45) as f,zipfile.ZipFile(io.BytesIO(f.read())) as z:
                    inputs=[]
                    for name,(filename,digest) in BOOKS.items():
                        b=z.read('gutenberg/'+filename+'.txt')
                        if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Unexpected corpus checksum')
                        inputs.append((name,b,'latin-1',url))
                break
            except (ValueError,OSError,zipfile.BadZipFile) as e: issues.append(str(e))
        else:raise RuntimeError('Corpus download : '+str(issues))
    results=[];seen=set()
    for name,raw,enc,url in inputs:
        digest=hashlib.sha256(raw).hexdigest()
        if digest in seen:raise ValueError('Duplicate input texts.')
        seen.add(digest)
        ids=tokenizer(raw.decode(enc),add_special_tokens=False,verbose=False)['input_ids']
        if len(ids)<offset+maxlen:raise ValueError(f'{name} : {len(ids)} available tokens; required count: {offset+maxlen}.')
        results.append((dict(title=name,source=url,sha256=digest,offset=offset,available_tokens=len(ids)),
                        torch.tensor(ids[offset:offset+maxlen],dtype=torch.long)))
    return results
