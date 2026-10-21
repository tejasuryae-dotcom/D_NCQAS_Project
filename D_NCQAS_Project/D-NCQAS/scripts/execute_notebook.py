#!/usr/bin/env python3
"""Portable sequential cell verifier for restricted environments without kernel sockets.

Executes the notebook's actual Python cells in one namespace, captures text/display
outputs, and saves an executed notebook. This is not a live Colab/browser test.
"""
import argparse,os,sys,time,traceback
from pathlib import Path
import nbformat
from IPython.utils.capture import capture_output
from IPython.display import display

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--notebook',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cwd',type=Path,required=True);args=ap.parse_args()
    book=nbformat.read(args.notebook.resolve(),as_version=4);out=args.output.resolve();os.chdir(args.cwd.resolve())
    namespace={'__name__':'__main__','display':display};count=0
    for cell in book.cells:
        if cell.cell_type!='code':continue
        count+=1;cell.execution_count=count
        with capture_output() as cap:
            exec(compile(cell.source,f'notebook-cell-{count}','exec'),namespace)
        outputs=[]
        if cap.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
        if cap.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
        for item in cap.outputs:outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        cell.outputs=outputs;print('PASS cell',count,flush=True)
    book.metadata['execution_validation']={'mode':'sequential Python cells; in-process','live_colab':False,'note':'Kernel socket creation unavailable in validation environment'}
    nbformat.write(book,out)
if __name__=='__main__':main()
