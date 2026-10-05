import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from tools.object_vlm.evaluate import digest, selected_checkpoint, summarize, input_length_audit
from tools.object_vlm.model import INTERFACE


def checkpoint_fixture(root):
    history=[{'epoch':i,'coverage':1.,'avg_l2_1_2_3s':float(i)} for i in range(1,7)]
    # A low-error but incomplete output must not win over complete predictions.
    history[5].update(coverage=.5,avg_l2_1_2_3s=.001)
    (root/'complete.json').write_text(json.dumps({'epochs':6,'steps':1398}))
    (root/'history.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
    checkpoint=root/'epoch_1';(checkpoint/'adapter').mkdir(parents=True)
    (checkpoint/'interface.json').write_text(json.dumps({'interface':INTERFACE,'representation':'text','max_length':7168}))
    for name in ['adapter_config.json','adapter_model.safetensors']:(checkpoint/'adapter'/name).write_bytes(b'fixture')
    best={'epoch':1,'path':str(checkpoint),'metrics':history[0]}
    (root/'best_checkpoint.json').write_text(json.dumps(best))
    return best


def test_validation_selection_prioritizes_coverage_and_rejects_stale_best(tmp_path):
    best=checkpoint_fixture(tmp_path)
    assert selected_checkpoint(tmp_path)['epoch']==1
    best['epoch']=6
    (tmp_path/'best_checkpoint.json').write_text(json.dumps(best))
    with pytest.raises(ValueError,match='validation selection'):selected_checkpoint(tmp_path)


def test_incomplete_training_cannot_enter_test(tmp_path):
    checkpoint_fixture(tmp_path)
    (tmp_path/'complete.json').write_text(json.dumps({'epochs':1,'steps':3}))
    with pytest.raises(ValueError,match='incomplete'):selected_checkpoint(tmp_path)


def test_summary_preserves_parse_failures_and_missing_error(tmp_path):
    (tmp_path/'frozen_selection.json').write_text('{}')
    for label,parsed,error in [('T1',480,1.),('S1',0,None)]:
        folder=tmp_path/label;folder.mkdir()
        (folder/'complete.json').write_text(json.dumps({'samples':480,'selection_sha256':digest(tmp_path/'frozen_selection.json')}))
        report={'epoch':5,'samples':480,'parsed':parsed,'coverage':parsed/480,'parse_failures':480-parsed,'avg_l2_1_2_3s':error,'mean_input_tokens':1200.}
        (folder/'metrics.json').write_text(json.dumps(report))
    with patch('builtins.print'):summarize(Namespace(output_dir=tmp_path))
    result=json.loads((tmp_path/'comparison.json').read_text())
    assert result['relative_avg_l2_reduction'] is None
    assert result['experiments']['S1']['parse_failures']==480
    assert '0/480' in (tmp_path/'report.md').read_text()


def test_test_input_over_training_guard_uses_shared_supported_context():
    class Tokenizer:
        def __call__(self,text,**kwargs):return {'input_ids':[1,2]+[3]*6449}
        def convert_tokens_to_ids(self,text):return 2
    processor=Namespace(tokenizer=Tokenizer(),apply_chat_template=lambda *a,**k:'prompt')
    config={'vision_config':{'image_size':384,'patch_size':14},'vision_feature_select_strategy':'full','image_token_index':1,'text_config':{'max_position_embeddings':32768}}
    rows=[{'observation':{'objects':{'geometry':[[]]}}}]
    with patch('tools.object_vlm.evaluate.prompt',return_value='observation'):
        audit=input_length_audit(rows,processor,config,7168)
        assert audit['experiments']['T1']['max_input_tokens']==7179
        assert audit['evaluation_max_length']==8192
        assert not audit['uses_test_targets']
        config['text_config']['max_position_embeddings']=8192
        with pytest.raises(ValueError,match='generation budget'):input_length_audit(rows,processor,config,7168)
