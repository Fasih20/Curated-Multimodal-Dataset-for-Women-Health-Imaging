from _load_stage_module import load_module
mod = load_module("11_compositional_panel_vqa/07_benchmark_medical_vlms.py", "stage11_medvlm")
def test_candidate_parser_handles_exact_and_reasoned_answers():
    assert mod.parse_candidate("different", ["same", "different"]) == "different"
    assert mod.parse_candidate("Reasoning... Final answer: B", ["A", "B", "none"]) == "B"
    assert mod.parse_candidate("I cannot determine", ["0", "1", "2"]) is None
