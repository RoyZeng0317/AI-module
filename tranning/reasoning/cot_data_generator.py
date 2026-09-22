import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifiers import RuleBasedVerifier

def generate_circuit_cot_data(output_path="data/reasoning_pairs.jsonl"):
    # for simple circuit/logic asking and answer convert to including <htink> hidden of thinking SFT dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    raw_samples = [
        {
            "instruction": "請分析",
            "raw_answer": "",
            "cot_steps": [
                "",
                "",
                ""
            ]
        },
        {
            "instruction": "請分析",
            "raw_answer": "",
            "cot_steps": [
                "",
                "",
                ""
            ]
        },
        {
            "instruction": "請分析",
            "raw_answer": "",
            "cot_steps": [
                "",
                "",
                ""
            ]
        },
    ]

    kept = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in raw_samples:
            think_content = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(sample["cot_steps"])])
            full_response = f"<think>\n{think_content}\n</think>\n\n{sample['raw_answer']}"

            # 用 verifiers 檢查格式，沒通過(例如 raw_answer 還是空的)就不寫入
            if RuleBasedVerifier.verify_think_format(full_response) < 1.0:
                print(f"skip: <think> 格式驗證失敗，略過 prompt={sample['instruction']!r}")
                continue

            json_record = {
                "prompt": sample["instruction"],
                "response": full_response,
                "think": think_content,
                "target": sample["raw_answer"],
                "format_score": RuleBasedVerifier.verify_think_format(full_response),
                "circuit_score": RuleBasedVerifier.verify_circuit_rule(full_response)
            }
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            kept += 1

        print(f"success: CoT tranning generate dataset successful, kept {kept}/{len(raw_samples)}, save to: {output_path}")

if __name__ == "__main__":
    generate_circuit_cot_data()