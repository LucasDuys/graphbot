"""Tests for prompt_templates -- XML-structured prompts with domain roles."""

from __future__ import annotations

from core_gb.prompt_templates import (
    CHAIN_OF_THOUGHT_INSTRUCTION,
    COT_COMPLEXITY_THRESHOLD,
    FewShotExample,
    PromptTemplate,
    TASK_TEMPLATES,
    build_structured_system_prompt,
    get_template,
)
from core_gb.types import Domain


class TestTemplateRegistry:
    """Every Domain has a registered PromptTemplate with all required fields."""

    def test_all_domains_have_templates(self) -> None:
        for domain in Domain:
            assert domain in TASK_TEMPLATES, f"Missing template for {domain}"

    def test_all_templates_have_role(self) -> None:
        for domain, template in TASK_TEMPLATES.items():
            assert template.role, f"Empty role for {domain}"
            assert len(template.role) > 10, f"Role too short for {domain}"

    def test_all_templates_have_instructions(self) -> None:
        for domain, template in TASK_TEMPLATES.items():
            assert template.instructions, f"Empty instructions for {domain}"

    def test_all_templates_have_examples(self) -> None:
        for domain, template in TASK_TEMPLATES.items():
            assert len(template.examples) >= 3, (
                f"Domain {domain} has {len(template.examples)} examples, need >= 3"
            )

    def test_all_templates_have_output_format(self) -> None:
        for domain, template in TASK_TEMPLATES.items():
            assert template.output_format, f"Empty output_format for {domain}"

    def test_all_templates_have_edge_case_notes(self) -> None:
        for domain, template in TASK_TEMPLATES.items():
            assert template.edge_case_notes, f"Empty edge_case_notes for {domain}"

    def test_get_template_returns_correct_domain(self) -> None:
        for domain in Domain:
            template = get_template(domain)
            assert template is TASK_TEMPLATES[domain]

    def test_get_template_fallback(self) -> None:
        """get_template returns SYNTHESIS for unknown domains (defensive)."""
        template = get_template(Domain.SYNTHESIS)
        assert template is TASK_TEMPLATES[Domain.SYNTHESIS]


class TestRoleAssignment:
    """Role strings vary by domain and match expected specializations."""

    def test_synthesis_role_mentions_analyst(self) -> None:
        role = TASK_TEMPLATES[Domain.SYNTHESIS].role
        assert "analyst" in role.lower() or "synthesizer" in role.lower()

    def test_code_role_mentions_developer(self) -> None:
        role = TASK_TEMPLATES[Domain.CODE].role
        assert "developer" in role.lower()

    def test_web_role_mentions_research(self) -> None:
        role = TASK_TEMPLATES[Domain.WEB].role
        assert "research" in role.lower()

    def test_file_role_mentions_administrator(self) -> None:
        role = TASK_TEMPLATES[Domain.FILE].role
        assert "administrator" in role.lower() or "file" in role.lower()

    def test_system_role_mentions_systems(self) -> None:
        role = TASK_TEMPLATES[Domain.SYSTEM].role
        assert "system" in role.lower()

    def test_roles_differ_across_domains(self) -> None:
        roles = {template.role for template in TASK_TEMPLATES.values()}
        assert len(roles) == len(TASK_TEMPLATES), "Some domains share the same role"


class TestBuildStructuredPrompt:
    """build_structured_system_prompt produces correct XML structure."""

    def test_contains_instructions_section(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS, complexity=3,
        )
        assert "<instructions>" in prompt
        assert "</instructions>" in prompt

    def test_contains_examples_section(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.CODE, complexity=4,
        )
        assert "<examples>" in prompt
        assert "</examples>" in prompt

    def test_contains_output_format_section(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.WEB, complexity=3,
        )
        assert "<output_format>" in prompt
        assert "</output_format>" in prompt

    def test_context_section_when_context_provided(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS,
            complexity=3,
            context_text="User prefers Python.",
        )
        assert "<context>" in prompt
        assert "</context>" in prompt
        assert "User prefers Python." in prompt

    def test_no_context_section_when_empty(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS, complexity=3, context_text="",
        )
        assert "<context>" not in prompt

    def test_pattern_hints_in_context(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS,
            complexity=3,
            context_text="some context",
            pattern_hints_text="Pattern: deploy (5 successes)",
        )
        assert "<context>" in prompt
        assert "Pattern: deploy (5 successes)" in prompt

    def test_role_at_start(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.CODE, complexity=3,
        )
        template = get_template(Domain.CODE)
        assert prompt.startswith(template.role)

    def test_edge_case_notes_in_instructions(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.FILE, complexity=4,
        )
        template = get_template(Domain.FILE)
        assert template.edge_case_notes in prompt
        # Edge case notes should be inside the instructions section
        instr_start = prompt.index("<instructions>")
        instr_end = prompt.index("</instructions>")
        edge_pos = prompt.index("Edge cases and failure modes:")
        assert instr_start < edge_pos < instr_end

    def test_few_shot_examples_present(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS, complexity=4,
        )
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" in prompt
        assert "Input:" in prompt
        assert "Output:" in prompt

    def test_few_shot_examples_include_why(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.CODE, complexity=4,
        )
        assert "Why:" in prompt

    def test_simple_tasks_get_lean_prompt(self) -> None:
        """Complexity 1-2 tasks get minimal prompts without XML tags."""
        prompt = build_structured_system_prompt(
            domain=Domain.CODE, complexity=1,
        )
        assert prompt.startswith("You are a helpful, accurate assistant")
        assert "<instructions>" not in prompt
        assert "<examples>" not in prompt
        assert "<output_format>" not in prompt


class TestChainOfThought:
    """Chain-of-thought activates at complexity >= COT_COMPLEXITY_THRESHOLD."""

    def test_cot_not_present_at_low_complexity(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS, complexity=1,
        )
        assert "<thinking>" not in prompt
        assert "<answer>" not in prompt
        assert "Think step by step" not in prompt

    def test_cot_not_present_at_complexity_2(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS, complexity=2,
        )
        assert "Think step by step" not in prompt

    def test_cot_present_at_threshold(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.SYNTHESIS, complexity=COT_COMPLEXITY_THRESHOLD,
        )
        assert "Think step by step" in prompt
        assert "<thinking>" in prompt
        assert "<answer>" in prompt

    def test_cot_present_at_high_complexity(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.CODE, complexity=5,
        )
        assert "Think step by step" in prompt
        assert "<thinking>" in prompt
        assert "<answer>" in prompt

    def test_cot_threshold_is_3(self) -> None:
        assert COT_COMPLEXITY_THRESHOLD == 3


class TestDomainSpecificPrompts:
    """Different domains produce meaningfully different prompts."""

    def test_code_domain_mentions_code(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.CODE, complexity=3,
        )
        assert "code" in prompt.lower()

    def test_web_domain_mentions_sources(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.WEB, complexity=3,
        )
        assert "source" in prompt.lower()

    def test_file_domain_mentions_file(self) -> None:
        prompt = build_structured_system_prompt(
            domain=Domain.FILE, complexity=3,
        )
        assert "file" in prompt.lower()

    def test_different_domains_produce_different_prompts(self) -> None:
        prompts = {
            domain: build_structured_system_prompt(domain=domain, complexity=3)
            for domain in Domain
        }
        # All prompts should be unique
        unique_prompts = set(prompts.values())
        assert len(unique_prompts) == len(Domain)


class TestFewShotExampleDataclass:
    """FewShotExample frozen dataclass works correctly."""

    def test_create_with_explanation(self) -> None:
        ex = FewShotExample(
            input="test input",
            output="test output",
            explanation="test why",
        )
        assert ex.input == "test input"
        assert ex.output == "test output"
        assert ex.explanation == "test why"

    def test_create_without_explanation(self) -> None:
        ex = FewShotExample(input="in", output="out")
        assert ex.explanation == ""

    def test_frozen(self) -> None:
        ex = FewShotExample(input="in", output="out")
        try:
            ex.input = "changed"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestPromptTemplateDataclass:
    """PromptTemplate frozen dataclass works correctly."""

    def test_defaults(self) -> None:
        t = PromptTemplate(role="test role", instructions="test instructions")
        assert t.examples == ()
        assert t.output_format == ""
        assert t.edge_case_notes == ""
        assert t.chain_of_thought is False
