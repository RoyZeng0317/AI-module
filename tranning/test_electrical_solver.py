from fractions import Fraction
import pytest
from electrical_solver import electrical_reply, solve_two_mesh

@pytest.mark.parametrize('message,answer', [
    ('100 歐姆、20V，依歐姆定律求電流', '0.2 A'),
    ('20 V, 0.1 kΩ, calculate current', '0.2 A'),
    ('電流 200 mA，電阻 100 Ω，求電壓', '20 V'),
    ('電壓 2e1 V，電流 .2 A，求電阻', '100 Ω'),
    ('電壓 -20 V，電阻 100 Ω，求電流', '-0.2 A')])
def test_ohm(message, answer):
    reason, reply = electrical_reply(message)
    assert answer in reply
    assert '模型' in reason or 'model' in reason

@pytest.mark.parametrize('message', ['今天心情很好', '我買了100個蘋果', '翻譯「20 V、100 Ω 求電流」', '請寫電流的故事', '請介紹電容'])
def test_no_route(message):
    assert electrical_reply(message) is None

@pytest.mark.parametrize('message', ['20 V求電流', '0 Ω、20 V求電流', '-1 Ω、20 V求電流', '20 V、0 A求電阻', '20 V、100 Ω、200 Ω求電流', '20 V、100 Ω並聯求電流'])
def test_guard(message):
    assert '=' not in electrical_reply(message)[1]

def test_mesh():
    result = solve_two_mesh(10, 4, 2, 2, 4)
    assert (result['vx'], result['i1'], result['i2'], result['center_current']) == (Fraction('5.6'), Fraction('2.2'), Fraction('.8'), Fraction('1.4'))
    assert result['nodal_residual'] == 0
    assert not any(result['mesh_residuals'] + result['mapping_residuals'])
    for resistance in (0, -1):
        with pytest.raises(ValueError):
            solve_two_mesh(10, 4, resistance, 2, 4)

@pytest.mark.parametrize('values', [(1,2,3,4,5), (-10,4,2,2,4), ('1.2','-.7','.2',3,4), (0,0,1,1,1)])
def test_mesh_exact_invariants(values):
    result = solve_two_mesh(*values)
    assert result['nodal_residual'] == 0
    assert not any(result['mesh_residuals'] + result['mapping_residuals'])

def test_teaching_and_capability():
    assert '5.6 V' in electrical_reply('教我節點電壓和網目電流轉換方法')[1]
    assert '兩端節點' in electrical_reply('教我我的電路的節點方法')[1]
    assert '不代表聊天模型' in electrical_reply('你會解基本電學嗎？')[1]

def test_route_integration():
    from tools import route_reply
    assert '0.2 A' in route_reply('100 歐姆、20V，依歐姆定律求電流')[1]

@pytest.mark.parametrize('message', ['電阻20 V、電流2 A求電壓', '1e999999 V、1 Ω求電流'])
def test_invalid_quantity_guard(message):
    assert '=' not in electrical_reply(message)[1]

def test_unit_not_partial():
    assert '=' not in electrical_reply('20 VA、100 Ω求電流')[1]

def test_extra_unlabelled_number():
    assert '未標示' in electrical_reply('我今年30歲，20 V、100 Ω求電流')[1]


@pytest.mark.parametrize('message', ['電流 2 MA，電阻 100 Ω，求電壓', '20 V、100 Ω，求交流電路電流', '20 V,100 ohm, calculate AC current', '20 V、100 Ω、電容求電流'])
def test_unsupported_units_and_ac(message):
    assert '=' not in electrical_reply(message)[1]

@pytest.mark.parametrize('message', ['20 V、100 Ω，頻率60Hz求電流', '20 V,100 ohm, calculate RLC current', '2 KOhm,20 V,calculate current', '2 MΩ、20 V求電流'])
def test_more_scope_guards(message):
    assert '=' not in electrical_reply(message)[1]
