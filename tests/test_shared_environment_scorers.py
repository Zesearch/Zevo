import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]/'scripts/evaluators/olmo_shared_environment'
MANIFEST=json.loads((ROOT/'manifest.json').read_text())

def test_revised_scorers_are_frozen_and_do_not_switch_interpreters():
    for entry in MANIFEST['scorers']:
        p=ROOT/(entry['new_sha256']+'.py')
        source=p.read_text()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['new_sha256']
        assert 'os.execv(RUNTIME' not in source
        tree=ast.parse(source)
        assert not any(isinstance(n,ast.Assign) and any(getattr(t,'id','')=='RUNTIME' for t in n.targets) for n in tree.body)

def test_validation_runs_in_current_interpreter_without_prefix_override(tmp_path):
    entry=next(e for e in MANIFEST['scorers'] if e['family']=='olmo_validation_v2')
    predictions=tmp_path/'pred.csv';gold=tmp_path/'gold.csv';out=tmp_path/'metrics.json'
    predictions.write_text('id,prediction,complete_prediction\n1,A,Answer: B\n2,C,No answer\n')
    gold.write_text('id,scoring_kind,allowed_labels,answer\n1,mc,ABCD,B\n2,mc,ABCD,C\n')
    result=subprocess.run([sys.executable,str(ROOT/(entry['new_sha256']+'.py')),str(predictions),str(gold),str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(out.read_text())['score']==0.5
