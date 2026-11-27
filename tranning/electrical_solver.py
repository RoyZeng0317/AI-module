"""Narrow local circuit calculations; no model, network, or weight updates."""
import re
from fractions import Fraction

SOURCE = 'https://ocw.mit.edu/courses/6-071j-introduction-to-electronics-signals-and-measurement-spring-2006/9d19116cbabeada1c98004a4367a0ee0_nodal_mesh_methd.pdf'
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
QUANTITY = re.compile(rf'({NUMBER})\s*(kΩ|kΩ|k[Oo]hm(?:s)?|千歐姆|千歐|歐姆|欧姆|Ω|Ω|ohms?|mA|毫安培|毫安|安培|amperes?|A|伏特|volts?|V)(?![a-zA-Z])', re.I)


def solve_two_mesh(VL, VR, RL, RR, RC):
    """Fixed topology: RL L-X, RR X-R, RC X-ground; source nodes L/R."""
    vl, vr, rl, rr, rc = map(lambda x: Fraction(str(x)), (VL, VR, RL, RR, RC))
    if min(rl, rr, rc) <= 0:
        raise ValueError('Passive resistances must be positive.')
    vx = (vl / rl + vr / rr) / (1 / rl + 1 / rr + 1 / rc)
    a, d, determinant = rl + rc, rr + rc, (rl + rc) * (rr + rc) - rc * rc
    i1, i2 = (vl * d - rc * vr) / determinant, (rc * vl - a * vr) / determinant
    return dict(vx=vx, i1=i1, i2=i2, center_current=vx / rc,
                nodal_residual=(vx-vl)/rl+(vx-vr)/rr+vx/rc,
                mesh_residuals=(a*i1-rc*i2-vl, -rc*i1+d*i2+vr),
                mapping_residuals=(i1-(vl-vx)/rl, i2-(vx-vr)/rr, vx/rc-i1+i2))


def _lesson(zh):
    if zh:
        return ('【本機固定電路教學；不是模型生成】\n網目法通常求「網目電流」，節點法求節點電壓。以下只示範已知連線，不解析任意電路圖。\n'
                '參考地為 0 V；電源設定 L=10 V、R=4 V。RL=2 Ω 接 L–X，RR=2 Ω 接 X–R，RC=4 Ω 接 X–地。\n'
                '1. 節點 KCL：(Vx−10)/2 + (Vx−4)/2 + Vx/4 = 0，因此 Vx=5.6 V。\n'
                '2. 順時針 I1 沿 L→X→地，I2 沿 X→R→地→X。網目 KVL：6I1−4I2=10；−4I1+6I2=−4。獨立解得 I1=2.2 A、I2=0.8 A。\n'
                '3. 轉換驗算：I1=(10−5.6)/2=2.2 A；I2=(5.6−4)/2=0.8 A；RC 向下電流=Vx/4=I1−I2=1.4 A。KCL 與兩條 KVL 殘差皆為 0。\n'
                '自訂電路請一次提供接地點、每個元件的兩端節點、阻值、電源大小與正負方向。')
    return ('[Local fixed-circuit lesson; not model output]\nMesh analysis finds mesh currents. Nodal analysis finds node voltages.\n'
            'Ground=0 V; L=10 V; R=4 V. RL=2 ohm joins L-X; RR=2 ohm joins X-R; RC=4 ohm joins X-ground.\n'
            'KCL: (Vx-10)/2+(Vx-4)/2+Vx/4=0. Vx=5.6 V.\n'
            'Clockwise I1 goes L-X-ground; I2 goes X-R-ground-X. KVL: 6I1-4I2=10; -4I1+6I2=-4. I1=2.2 A; I2=0.8 A.\n'
            'Check: I1=(10-5.6)/2; I2=(5.6-4)/2; Vx/4=I1-I2=1.4 A. All KCL and KVL errors are zero.\n'
            'For your circuit, give ground, both end nodes of each part, all values, and source directions.')


def electrical_reply(message):
    """Route only direct capability, explicit lesson, or unit-labelled Ohm requests."""
    text = message.strip()
    zh = bool(re.search(r'[\u4e00-\u9fff]', text))
    if re.search(r'[「」“”"`]|翻譯|故事|笑話|translate|story|quote', text, re.I):
        return None
    reason = '本機電學公式／固定教學；未使用模型生成' if zh else 'Local circuit formula or fixed lesson; no model generation'
    if re.fullmatch(r'(?:那)?你(?:會|能|可以|懂|了解)(?:解|計算|做)?(?:基本)?電學(?:題目|題)?嗎[?？!！]*', text) or re.fullmatch(r'(?:can you (?:solve|do)|do you understand) (?:basic )?(?:electrical circuits?|electricity)\??', text, re.I):
        return reason, ('目前可用本機公式計算單一電阻的歐姆定律，並展示固定電路的節點電壓與網目電流轉換。這不代表聊天模型已學會電學；任意電路仍需完整連線與數值。' if zh else 'I can use local formulas for one resistor and show a fixed nodal and mesh example. This does not mean the chat model has learned electricity. Other circuits need full connections and values.')
    if re.search(r'節點|节点|網目|网目|nodal|mesh', text, re.I) and re.search(r'教|說明|轉換|转换|方法|怎麼|如何|teach|explain|convert|method', text, re.I):
        if re.search(r'我的|這個|此電路|my circuit|this circuit', text, re.I):
            return reason, ('請一次提供接地點、每個元件的兩端節點、所有阻值、電源大小及正負方向，才能建立節點與網目方程式。' if zh else 'Please give ground, both end nodes of each part, all values, and source directions.')
        return reason, _lesson(zh) + '\n來源 / Source: ' + SOURCE
    target = next((key for key, pattern in [('I', r'(?:求|算|計算|多少|多大).*電流|電流.*(?:多少|多大|為何)|(?:find|calculate|what is).*current'), ('V', r'(?:求|算|計算).*電壓|電壓.*多少|(?:find|calculate|what is).*voltage'), ('R', r'(?:求|算|計算).*電阻|電阻.*多少|(?:find|calculate|what is).*resistance')] if re.search(pattern, text, re.I)), None)
    if target is None:
        return None
    if re.search(r'交流|電容|电容|電感|电感|相位|阻抗|頻率|频率|\b(?:AC|RLC|capacitor|inductor|phase|impedance|frequency)\b', text, re.I):
        return reason, ('此簡易計算限單一被動電阻的直流歐姆定律；交流、電容或電感問題請提供完整電路與頻率等條件。' if zh else 'This tool is for one DC resistor. For AC, capacitors, or inductors, give the full circuit and frequency.')
    values = {}
    for match in QUANTITY.finditer(text):
        num, unit = match.groups()
        if (unit.lower() == 'ma' and unit != 'mA') or unit.startswith('K'):
            return reason, ('此單位大小寫不在支援範圍；請使用 V、A、mA、Ω 或 kΩ，並確認倍率。' if zh else 'This unit case is not supported. Use V, A, mA, ohm, or kOhm and check the scale.')
        if len(num) > 40 or ('e' in num.lower() and abs(int(num.lower().split('e')[1])) > 30):
            return reason, ('數值超出此簡易計算器範圍，請使用一般尺度的有限數值。' if zh else 'This value is outside this small calculator range. Please use a smaller finite value.')
        unit = unit.lower()
        kind = 'R' if re.search(r'ω|Ω|ohm|歐|欧', unit) else 'I' if re.search(r'a|安', unit) else 'V'
        label = re.search(r'(電阻|電壓|電流|resistance|voltage|current)\s*(?:為|是|=|:|：)?\s*$', text[:match.start()], re.I)
        if label and {'電阻':'R','電壓':'V','電流':'I','resistance':'R','voltage':'V','current':'I'}[label[1].lower()] != kind:
            return reason, ('物理量名稱與單位不符，請核對電壓 V、電流 A、電阻 Ω。' if zh else 'The name and unit do not match. Check V for voltage, A for current, and ohm for resistance.')
        scale = 1000 if unit.startswith('k') or unit.startswith('千') else Fraction(1,1000) if unit.startswith('m') or unit.startswith('毫') else 1
        values.setdefault(kind, []).append(Fraction(num) * scale)
    if not values and not re.search(r'歐姆定律|欧姆定律|ohm', text, re.I):
        return None
    if re.search(NUMBER, QUANTITY.sub('', text)):
        return reason, ('存在未標示物理量的數字，請只提供同一電阻的已知量及單位。' if zh else 'Some numbers have no clear units. Give only the known values and units for one resistor.')
    if re.search(r'並聯|串聯|并联|串联|parallel|series|多個|兩個|两个', text, re.I) or any(len(v) != 1 for v in values.values()) or target in values:
        return reason, ('數值或電路關係不唯一。請提供單一電阻兩端的電壓、流過它的電流或阻值，並指出要求的未知量。' if zh else 'The values or connections are unclear. Give values for one resistor and name the unknown value.')
    required = {'I': ('V','R'), 'V': ('I','R'), 'R': ('V','I')}[target]
    if any(k not in values for k in required):
        return reason, ('資訊不足。請提供同一電阻的兩個已知量及單位（電壓 V、電流 A、電阻 Ω）。' if zh else 'Not enough data. Give two known values and units for the same resistor: V, A, or ohm.')
    v = {k: x[0] for k,x in values.items()}
    if v.get('R', 1) <= 0 or target == 'R' and v['I'] == 0:
        return reason, ('無法使用此被動電阻計算：電阻必須大於 0；求電阻時電流不可為 0。' if zh else 'Resistance must be above zero. To find resistance, current must not be zero.')
    a,b = required
    result = v[a]*v[b] if target == 'V' else v[a]/v[b]
    if target == 'R' and result <= 0:
        return reason, ('結果不符合正值被動電阻；請核對電壓與電流方向。' if zh else 'This is not a positive passive resistance. Check voltage and current directions.')
    fmt = lambda n: f'{float(n):.10g}'
    formula = {'I':'I = V/R', 'V':'V = I×R', 'R':'R = V/I'}[target]
    unit = {'I':'A','V':'V','R':'Ω'}[target]
    operation = '×' if target == 'V' else '/'
    extra = f'（{fmt(result*1000)} mA）' if target == 'I' else ''
    label = '本機歐姆定律計算；不是模型生成' if zh else 'Local Ohm law calculation; not model output'
    return reason, f'【{label}】\n{formula} = {fmt(v[a])}{operation}{fmt(v[b])} = {fmt(result)} {unit}{extra}'

