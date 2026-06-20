import argparse
import json
import traceback

from Config import (
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
    ANTHROPIC_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_API_KEY,
    discussion_template1,
)
from MultiAgentDiscussion import SingleAgent, ErrorSolver, parse_json


DEFAULT_QUESTION = "synthetic build error: missing fuzz target and linker failure"
DEFAULT_PROJECT = "diagnose-project"


def short_repr(value, limit=1200):
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "... [TRUNCATED]"
    return text


def print_banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_configs():
    return [
        {
            "name": "OPENAI",
            "base_url": OPENAI_BASE_URL,
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
        },
        {
            "name": "ANTHROPIC",
            "base_url": ANTHROPIC_BASE_URL,
            "model": ANTHROPIC_MODEL,
            "api_key": ANTHROPIC_API_KEY,
        },
        {
            "name": "DEEPSEEK",
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
        },
    ]


def diagnose_single_model(cfg, question, project_name):
    print_banner(f"Model Diagnostic: {cfg['name']}")
    print(f"base_url={cfg['base_url']}")
    print(f"model={cfg['model']}")
    print(f"project={project_name}")
    print(f"question={question}")

    agent = SingleAgent(
        base_url=cfg["base_url"],
        model_name=cfg["model"],
        api_key=cfg["api_key"],
    )

    agent_executor = None
    raw_output = None
    invoke_exception = None
    generate_exception = None
    generate_result = None

    try:
        print("[1] Creating agent executor...")
        agent_executor = agent.create_agent(discussion_template1)
        print(f"[1] Agent executor type: {type(agent_executor)}")
    except Exception as exc:
        print(f"[1] Agent creation failed: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return {
            "name": cfg["name"],
            "ok": False,
            "stage": "create_agent",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        print("[2] Calling AgentExecutor.invoke(...) directly...")
        raw_output = agent_executor.invoke({"input": question, "project": project_name})
        print(f"[2] Raw output type: {type(raw_output)}")
        print(f"[2] Raw output repr: {short_repr(raw_output)}")
        if isinstance(raw_output, dict):
            print(f"[2] Raw output keys: {list(raw_output.keys())}")
            for key, value in raw_output.items():
                print(f"[2] key={key} | type={type(value)} | repr={short_repr(value, 400)}")
    except Exception as exc:
        invoke_exception = exc
        print(f"[2] invoke failed: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())

    if raw_output:
        print("[3] Parsing raw output using current MultiAgentDiscussion.parse_json logic...")
        try:
            parsed = parse_json(raw_output)
            print(f"[3] parse_json result type: {type(parsed)}")
            print(f"[3] parse_json result repr: {short_repr(parsed)}")
        except Exception as exc:
            print(f"[3] parse_json raised: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
    else:
        print("[3] Skipped parse_json because raw output is falsy or unavailable.")

    print("[4] Calling SingleAgent.generate_answer(...) using real system path...")
    try:
        generate_result = agent.generate_answer(
            template=discussion_template1,
            question=question,
            project_name=project_name,
        )
        print(f"[4] generate_answer type: {type(generate_result)}")
        print(f"[4] generate_answer repr: {short_repr(generate_result)}")
    except Exception as exc:
        generate_exception = exc
        print(f"[4] generate_answer failed: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())

    ok = generate_exception is None
    return {
        "name": cfg["name"],
        "ok": ok,
        "stage": "generate_answer" if not ok else "success",
        "raw_output_type": str(type(raw_output)) if raw_output is not None else None,
        "raw_output_truthy": bool(raw_output) if raw_output is not None else False,
        "invoke_error": None if invoke_exception is None else f"{type(invoke_exception).__name__}: {invoke_exception}",
        "generate_error": None if generate_exception is None else f"{type(generate_exception).__name__}: {generate_exception}",
        "generate_result": generate_result,
    }


def diagnose_error_solver(question, project_name):
    print_banner("ErrorSolver End-to-End Diagnostic")
    solver = ErrorSolver(project_name=project_name)
    try:
        result = solver.discussion(question)
        print(f"discussion result type={type(result)}")
        print(f"discussion result repr={short_repr(result)}")
        print(f"logger_entries={len(solver.logger)}")
        if solver.logger:
            print(f"logger_last_entry_repr={short_repr(solver.logger[-1], 800)}")
        return {
            "ok": not (isinstance(result, str) and result.startswith("Error during discussion:")),
            "result": result,
            "logger_entries": len(solver.logger),
        }
    except Exception as exc:
        print(f"discussion raised: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return {
            "ok": False,
            "result": f"{type(exc).__name__}: {exc}",
            "logger_entries": len(solver.logger),
        }


def main():
    parser = argparse.ArgumentParser(description="Diagnose API behavior at the MultiAgentDiscussion.py layer.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Synthetic error text to feed into discussion_template1.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project name passed to the discussion agent.")
    parser.add_argument("--skip-errorsolver", action="store_true", help="Only test individual models, skip ErrorSolver end-to-end run.")
    args = parser.parse_args()

    summaries = []
    for cfg in build_configs():
        summaries.append(diagnose_single_model(cfg, args.question, args.project))

    solver_summary = None
    if not args.skip_errorsolver:
        solver_summary = diagnose_error_solver(args.question, args.project)

    print_banner("Summary")
    print(json.dumps({"models": summaries, "error_solver": solver_summary}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
