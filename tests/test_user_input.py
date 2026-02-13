import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from colophon.user_input import (
    AgentSDKUserInputHandler,
    CODEX_ASK_USER_QUESTION_TOOL_NAME,
    GUIDANCE_STAGE_COORDINATION,
    GUIDANCE_STAGE_OUTLINE,
    GUIDANCE_STAGE_PLANNING,
    GUIDANCE_STAGE_RECOMMENDATIONS,
    OpenAICodexUserInputHandler,
    UserGuidanceBundle,
    PlanningGuidance,
    apply_guidance_to_pipeline_flags,
    apply_coordination_guidance,
    apply_outline_guidance,
    apply_recommendation_guidance,
    build_coordination_breakdown_questionnaire,
    build_recommendation_questionnaire,
    build_outline_questionnaire,
    collect_user_guidance_bundle,
    build_codex_ask_user_question_tool,
    build_planning_questionnaire,
    collect_user_answers,
    normalize_guidance_stages,
    infer_coordination_guidance,
    infer_outline_guidance,
    infer_recommendation_guidance,
    infer_planning_guidance,
)


class UserInputTests(unittest.TestCase):
    def test_build_planning_questionnaire_has_core_questions(self) -> None:
        payload = build_planning_questionnaire()
        questions = payload.get("questions", [])
        self.assertEqual(len(questions), 4)
        rendered = " ".join(str(item.get("question", "")).lower() for item in questions if isinstance(item, dict))
        self.assertIn("planning document style", rendered)
        self.assertIn("recommendation", rendered)
        self.assertIn("expand the outline", rendered)

    def test_collect_user_answers_accepts_numeric_choices(self) -> None:
        prompts = {
            "questions": [
                {
                    "question": "Should I expand the outline?",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                }
            ]
        }
        responses = iter(["1"])

        answers = collect_user_answers(
            input_data=prompts,
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )

        self.assertEqual(answers["Should I expand the outline?"], "Yes")

    def test_infer_planning_guidance_parses_preferences(self) -> None:
        answers = {
            "What planning document style do you want?": "Risk-first plan",
            "Should I incorporate recommendation proposals into the plan?": "Yes, include recommendations",
            "Should I expand the outline before drafting?": "No, keep current outline",
            "Any additional guidance for planning and execution?": "Focus on tradeoffs and milestones.",
        }

        guidance = infer_planning_guidance(answers)

        self.assertEqual(guidance.planning_document_focus, "Risk-first plan")
        self.assertTrue(guidance.incorporate_recommendations)
        self.assertFalse(guidance.expand_outline)
        self.assertIn("tradeoffs", guidance.additional_notes.lower())

    def test_apply_guidance_to_pipeline_flags_overrides(self) -> None:
        guidance = PlanningGuidance(incorporate_recommendations=True, expand_outline=False)
        recommendations, outline_expander = apply_guidance_to_pipeline_flags(
            guidance=guidance,
            enable_paper_recommendations=False,
            enable_outline_expander=True,
        )

        self.assertTrue(recommendations)
        self.assertFalse(outline_expander)

    def test_collect_user_answers_limits_to_top_ten_by_importance(self) -> None:
        questions = [
            {"question": f"Question {index}", "importance": index}
            for index in range(1, 13)
        ]
        responses = iter([f"answer-{idx}" for idx in range(1, 11)])

        answers = collect_user_answers(
            input_data={"questions": questions},
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )

        self.assertEqual(len(answers), 10)
        self.assertNotIn("Question 1", answers)
        self.assertNotIn("Question 2", answers)
        self.assertIn("Question 12", answers)

    def test_normalize_guidance_stages(self) -> None:
        stages = normalize_guidance_stages("plan,recommendations,outline_expansion,coordination-breakdown")
        self.assertEqual(
            stages,
            [GUIDANCE_STAGE_PLANNING, GUIDANCE_STAGE_RECOMMENDATIONS, GUIDANCE_STAGE_OUTLINE, GUIDANCE_STAGE_COORDINATION],
        )

    def test_build_recommendation_questionnaire(self) -> None:
        payload = build_recommendation_questionnaire(current_enabled=True, current_top_k=6, current_per_seed_limit=4, current_min_score=0.3)
        questions = payload.get("questions", [])
        self.assertGreaterEqual(len(questions), 3)

    def test_build_outline_questionnaire(self) -> None:
        payload = build_outline_questionnaire(current_enabled=True, current_max_subsections=5)
        questions = payload.get("questions", [])
        rendered = " ".join(str(item.get("question", "")).lower() for item in questions if isinstance(item, dict))
        self.assertIn("outline expansion", rendered)

    def test_build_coordination_breakdown_questionnaire_uses_gaps(self) -> None:
        payload = build_coordination_breakdown_questionnaire(
            gap_requests=[
                {"component": "bibliography", "request": "Add sources"},
                {"component": "outline", "request": "Align sections"},
                {"component": "bibliography", "request": "Increase evidence"},
            ],
            coordination_messages=[],
            current_max_iterations=4,
        )
        questions = payload.get("questions", [])
        self.assertTrue(questions)
        first = questions[0]
        self.assertIn("Detected coordination pressure", first.get("question", ""))

    def test_infer_and_apply_recommendation_guidance(self) -> None:
        guidance = infer_recommendation_guidance(
            {
                "Recommendation workflow is currently disabled. Enable it for this run?": "Yes, enable recommendations",
                "Choose recommendation strategy (current top_k=8, per_seed=5, min_score=0.20).": "Aggressive (top_k=12, per_seed=8, min_score=0.10)",
            }
        )
        enabled, top_k, per_seed, min_score = apply_recommendation_guidance(
            guidance=guidance,
            enable_recommendations=False,
            top_k=8,
            per_seed_limit=5,
            min_score=0.2,
        )
        self.assertTrue(enabled)
        self.assertEqual(top_k, 12)
        self.assertEqual(per_seed, 8)
        self.assertAlmostEqual(min_score, 0.1)

    def test_infer_and_apply_outline_guidance(self) -> None:
        guidance = infer_outline_guidance(
            {
                "Outline expansion is currently disabled. Enable expansion this run?": "Yes, expand outline",
                "Select outline expansion depth (current max_subsections=3).": "Deep (max_subsections=6)",
                "Which expansion signals should be emphasized?": "Transitions across sections, Counterarguments",
            }
        )
        enabled, max_subsections = apply_outline_guidance(
            guidance=guidance,
            enable_outline_expander=False,
            max_subsections=3,
        )
        self.assertTrue(enabled)
        self.assertEqual(max_subsections, 6)
        self.assertTrue(guidance.include_transitions)
        self.assertTrue(guidance.include_counterarguments)

    def test_infer_and_apply_coordination_guidance(self) -> None:
        guidance = infer_coordination_guidance(
            {
                "Choose coordination strategy for this run.": "Strict outline alignment",
                "Coordination currently uses max_iterations=4. Increase revision iterations?": "Yes, increase iterations",
                "Which coordination breakdown components should be prioritized?": "bibliography, outline",
            }
        )
        iterations = apply_coordination_guidance(guidance=guidance, coordination_max_iterations=4)
        self.assertGreaterEqual(iterations, 6)
        self.assertTrue(guidance.strict_outline_alignment)

    def test_collect_user_guidance_bundle(self) -> None:
        responses = iter(
            [
                "1",  # planning style
                "1",  # planning recommendations
                "1",  # planning outline
                "Planning note",
                "1",  # rec enable
                "2",  # rec strategy
                "1,3",  # rec focus
                "Rec note",
                "1",  # outline enable
                "2",  # outline depth
                "1,2",  # outline signals
                "Outline note",
                "2",  # coordination strategy
                "1",  # coordination increase iterations
                "1,2",  # coordination priorities
                "Coord note",
            ]
        )
        bundle = collect_user_guidance_bundle(
            stages=[GUIDANCE_STAGE_PLANNING, GUIDANCE_STAGE_RECOMMENDATIONS, GUIDANCE_STAGE_OUTLINE, GUIDANCE_STAGE_COORDINATION],
            context_by_stage={},
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )
        self.assertIsInstance(bundle, UserGuidanceBundle)
        self.assertIn(GUIDANCE_STAGE_PLANNING, bundle.answers_by_stage)
        self.assertIn(GUIDANCE_STAGE_RECOMMENDATIONS, bundle.answers_by_stage)

    def test_build_codex_ask_user_question_tool_schema(self) -> None:
        tool = build_codex_ask_user_question_tool()
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], CODEX_ASK_USER_QUESTION_TOOL_NAME)
        parameters = tool.get("parameters", {})
        props = parameters.get("properties", {})
        self.assertIn("questions", props)
        self.assertIn("question", props)

    def test_openai_codex_handler_handles_question_tool_calls(self) -> None:
        responses = iter(["1"])
        handler = OpenAICodexUserInputHandler(
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )
        payload = {
            "question": "Should I expand the outline?",
            "options": [{"label": "Yes"}, {"label": "No"}],
        }

        result = handler.handle_function_call(CODEX_ASK_USER_QUESTION_TOOL_NAME, payload)

        self.assertEqual(result.get("status"), "ok")
        answers = result.get("answers", {})
        self.assertEqual(answers.get("Should I expand the outline?"), "Yes")

    def test_openai_codex_handler_limits_questions_to_top_ten(self) -> None:
        questions = [{"question": f"Q{i}", "importance": i} for i in range(1, 13)]
        responses = iter([f"a{i}" for i in range(1, 11)])
        handler = OpenAICodexUserInputHandler(
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )

        result = handler.handle_function_call(CODEX_ASK_USER_QUESTION_TOOL_NAME, {"questions": questions})

        answers = result.get("answers", {})
        self.assertEqual(len(answers), 10)
        self.assertNotIn("Q1", answers)
        self.assertNotIn("Q2", answers)
        meta = result.get("meta", {})
        self.assertEqual(meta.get("questions_requested"), 12)
        self.assertEqual(meta.get("questions_asked"), 10)

    def test_openai_codex_handler_rejects_concurrent_question_calls(self) -> None:
        def _slow_input(_: str) -> str:
            time.sleep(0.1)
            return "1"

        handler = OpenAICodexUserInputHandler(
            input_fn=_slow_input,
            output_fn=lambda _: None,
        )
        payload = {
            "question": "Should I expand the outline?",
            "options": [{"label": "Yes"}, {"label": "No"}],
        }
        barrier = threading.Barrier(2)

        def _invoke() -> dict:
            barrier.wait()
            return handler.handle_function_call(CODEX_ASK_USER_QUESTION_TOOL_NAME, payload)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(_invoke)
            second = executor.submit(_invoke)
            results = [first.result(), second.result()]

        statuses = sorted(str(result.get("status", "")) for result in results if isinstance(result, dict))
        self.assertEqual(statuses.count("ok"), 1)
        self.assertEqual(statuses.count("busy"), 1)


class UserInputHandlerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_handler_injects_answers_for_ask_user_question(self) -> None:
        responses = iter(["1", "1", "2", "Keep the existing intro framing."])
        handler = AgentSDKUserInputHandler(
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )

        result = await handler.can_use_tool("AskUserQuestion", build_planning_questionnaire())

        if isinstance(result, dict):
            self.assertEqual(result.get("behavior"), "allow")
            updated = result.get("updated_input", {})
            self.assertIn("answers", updated)
            self.assertEqual(len(updated["answers"]), 4)
        self.assertIsNotNone(handler.last_guidance)
        assert handler.last_guidance is not None
        self.assertTrue(handler.last_guidance.incorporate_recommendations)
        self.assertFalse(handler.last_guidance.expand_outline)

    async def test_handler_allows_non_question_tools(self) -> None:
        handler = AgentSDKUserInputHandler(
            input_fn=lambda _: "",
            output_fn=lambda _: None,
        )
        result = await handler.can_use_tool("Read", {"file_path": "notes.md"})

        if isinstance(result, dict):
            self.assertEqual(result.get("behavior"), "allow")
            self.assertEqual(result.get("updated_input", {}).get("file_path"), "notes.md")

    async def test_handler_limits_ask_user_question_to_ten_inputs(self) -> None:
        questions = [{"question": f"Q{index}", "importance": index} for index in range(1, 13)]
        responses = iter([f"a{index}" for index in range(1, 11)])
        handler = AgentSDKUserInputHandler(
            input_fn=lambda _: next(responses),
            output_fn=lambda _: None,
        )

        result = await handler.can_use_tool("AskUserQuestion", {"questions": questions})

        if isinstance(result, dict):
            updated = result.get("updated_input", {})
            answers = updated.get("answers", {})
            self.assertEqual(len(answers), 10)
            self.assertEqual(len(updated.get("questions", [])), 10)
            self.assertNotIn("Q1", answers)
            self.assertNotIn("Q2", answers)

    async def test_handler_rejects_concurrent_ask_user_question_calls(self) -> None:
        def _slow_input(_: str) -> str:
            time.sleep(0.1)
            return "1"

        handler = AgentSDKUserInputHandler(
            input_fn=_slow_input,
            output_fn=lambda _: None,
        )
        payload = {
            "question": "Should I expand the outline?",
            "options": [{"label": "Yes"}, {"label": "No"}],
        }
        barrier = threading.Barrier(2)

        def _invoke() -> object:
            barrier.wait()
            return asyncio.run(handler.can_use_tool("AskUserQuestion", payload))

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(_invoke)
            second = executor.submit(_invoke)
            results = [first.result(), second.result()]

        behaviors = [
            result.get("behavior")
            for result in results
            if isinstance(result, dict)
        ]
        self.assertEqual(behaviors.count("allow"), 1)
        self.assertEqual(behaviors.count("deny"), 1)


if __name__ == "__main__":
    unittest.main()
