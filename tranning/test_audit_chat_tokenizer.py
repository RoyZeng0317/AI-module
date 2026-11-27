from audit_chat_tokenizer import audit_rows
from bpe_tokenizer import UNK


class Tokens:
    def encode(self, text):
        return [UNK if char == '?' else 5 for char in text]


def test_supervised_answer_truncation_and_unknown_coverage():
    rows = [dict(prompt='abcde', reply='answer'), dict(prompt='a', reply='?'),
            dict(prompt='ab', reply='four')]
    report = audit_rows(rows, Tokens(), block_size=5)
    assert report['empty_answer_targets'] == 1
    assert report['truncated_rows'] == 2
    assert report['unknown_tokens'] == 1
    assert report['unknown_characters'] == ['?']
    assert report['issues'][0]['retained_answer_targets'] == 0
    assert report['issues'][1]['retained_answer_targets'] == 2  # answer and EOS
    assert report['issues'][2]['retained_answer_targets'] == 3
