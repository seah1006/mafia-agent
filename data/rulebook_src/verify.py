import sys; sys.path.insert(0,r"..\..");import config
from langchain_community.document_loaders import PyPDFLoader
d=PyPDFLoader(str(config.PDF_PATH)).load()
print(len(d))
for i,x in enumerate(d):
    t=[l for l in x.page_content.splitlines() if l.strip()]
    print(i,len(x.page_content),t[0].strip()==config.PAGE_META[i]["title"],t[0])
