import re

class RuleBasedVerifier:
    # for GRPO / RL tranning of Reward Verifier, check the model output of thinking format and logic correct or not
    @staticmethod
    def verify_think_format(completion: str) -> float:
        # check the has correct <think>...</think> label struction
        pattern = r"^<think>[\s\S]*?</think>[\s\S]+"
        if re.match(pattern, completion.strip()):
            return 1.0
        return 0.0

    @staticmethod
    def verify_code_execution(code_snippet: str) -> float:
        # verify Python code is can successful compaing (Sandbox simluation)
        try:
            compiled = compile(code_snippet, "<string>", "exec")
            return 0.5  # success compilng
        except Exception:
            return -1.0

    @staticmethod
    def verify_circuit_rule(response: str) -> float:
        # verify in the reply must has circuit throy keyword and format
        score = 0.0
        keywords = ["V_UT", "V_LT", "遲滯", "回授", "Hysteresis"]
        for kw in keywords:
            if kw in response:
                score +=0.2
        return min(score, 1.0)

if __name__ == "__main__":
    sample_text = "<think>\n1. anaylizing 正回授 circuit...\n</think>\n\n遲滯 Voltage V_hys = V_UT - V_LT"
    print("Foramt Score:", RuleBasedVerifier.verify_think_format(sample_text))
    print("Circuit Rule Score:", RuleBasedVerifier.verify_circuit_rule(sample_text))