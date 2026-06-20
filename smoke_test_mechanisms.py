import argparse
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path


def ensure_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = ensure_module(parent_name)
        setattr(parent, child_name, module)
    return module


def install_import_stubs():
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def bind(self, *args, **kwargs):
            return self

        def invoke(self, *args, **kwargs):
            return {}

        def partial(self, *args, **kwargs):
            return self

        def __or__(self, other):
            return self

        def __ror__(self, other):
            return self

    class DummyTool:
        def __init__(self, name=None, func=None, description=None, **kwargs):
            self.name = name
            self.func = func
            self.description = description
            self.return_direct = kwargs.get("return_direct", False)

    class DummyPromptTemplate(Dummy):
        @classmethod
        def from_template(cls, template=None):
            return cls()

    class DummyDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class DummyAgentExecutor(Dummy):
        pass

    ensure_module("langchain_openai").ChatOpenAI = Dummy
    ensure_module("langchain_openai").OpenAIEmbeddings = Dummy
    ensure_module("langchain.agents").Tool = DummyTool
    ensure_module("langchain.agents").AgentType = Dummy
    ensure_module("langchain.agents").AgentExecutor = DummyAgentExecutor
    ensure_module("langchain.agents").create_react_agent = lambda *args, **kwargs: Dummy()
    ensure_module("langchain.agents").initialize_agent = lambda *args, **kwargs: Dummy()
    ensure_module("langchain.agents.output_parsers.openai_tools").OpenAIToolsAgentOutputParser = Dummy
    ensure_module("langchain_core.agents").AgentAction = Dummy
    ensure_module("langchain_core.agents").AgentFinish = Dummy
    ensure_module("langchain.prompts").PromptTemplate = DummyPromptTemplate
    ensure_module("langchain.schema").StrOutputParser = Dummy
    ensure_module("langchain_core.tools.render").render_text_description = lambda tools: ""
    ensure_module("langchain_core.utils.function_calling").convert_to_openai_tool = lambda tool: tool
    ensure_module("langchain.tools").tool = lambda func=None, *args, **kwargs: func if func else (lambda inner: inner)
    ensure_module("langchain_chroma").Chroma = Dummy
    ensure_module("langchain_community.utilities").GoogleSerperAPIWrapper = Dummy
    ensure_module("langchain_core.runnables").RunnableLambda = Dummy
    ensure_module("langchain.schema.runnable").RunnablePassthrough = Dummy
    ensure_module("langchain.docstore.document").Document = DummyDocument
    ensure_module("langchain_text_splitters").RecursiveCharacterTextSplitter = Dummy
    ensure_module("fake_useragent").UserAgent = Dummy
    ensure_module("ipdb").set_trace = lambda *args, **kwargs: None
    ensure_module("googleapiclient.discovery").build = lambda *args, **kwargs: None
    ensure_module("GoogleSearch").search_agent = lambda *args, **kwargs: "stub search result"


def run_patch_smoke_test(workspace: Path) -> dict:
    from Tools import HostShell

    patch_dir = Path(tempfile.mkdtemp(prefix="autocompiler_patch_smoke_", dir=str(workspace)))
    target_file = patch_dir / "patch_target.txt"
    log_file = patch_dir / "hostshell.log"
    target_file.write_text("before\n", encoding="utf-8")

    shell = HostShell(build_log_path=str(log_file))
    command = (
        "python3 -c 'from pathlib import Path; "
        f"Path(r\"{target_file}\").write_text(\"after\\n\", encoding=\"utf-8\")'"
    )
    output = shell.execute_command(command)
    content = target_file.read_text(encoding="utf-8")

    return {
        "ok": "after" in content,
        "temp_dir": str(patch_dir),
        "target_file": str(target_file),
        "log_file": str(log_file),
        "command": command,
        "output": output,
        "content": content,
    }


def build_stubbed_generate_answer():
    def stubbed_generate_answer(self, template, question, project_name, chat_history=None):
        suffix = self.model_name.lower()
        return {
            "reasoning": f"stub reasoning from {suffix}",
            "solution": f"apply fix suggested by {suffix}",
            "confidence_level": 1.0,
        }

    return stubbed_generate_answer


def run_discussion_smoke_test(real_apis: bool) -> dict:
    from MultiAgentDiscussion import ErrorSolver, SingleAgent

    original_generate_answer = SingleAgent.generate_answer
    if not real_apis:
        SingleAgent.generate_answer = build_stubbed_generate_answer()

    stats = {"discussion_triggered": "NO"}
    solver = ErrorSolver(project_name="smoke-project")

    def monitored_discussion(mistake):
        stats["discussion_triggered"] = "YES"
        return solver.discussion(mistake)

    try:
        result = monitored_discussion("synthetic build error: missing fuzz target")
        return {
            "ok": stats["discussion_triggered"] == "YES" and isinstance(result, str) and not result.startswith("Error during discussion:"),
            "discussion_triggered": stats["discussion_triggered"],
            "result": result,
            "mode": "real" if real_apis else "stub",
            "logger_entries": len(solver.logger),
            "log_file": str(Path("test") / "MultiAgentDiscussion.log"),
        }
    finally:
        SingleAgent.generate_answer = original_generate_answer


def main():
    parser = argparse.ArgumentParser(description="Safe smoke tests for patch application and multi-agent discussion.")
    parser.add_argument("--real-discussion", action="store_true", help="Use configured LLM APIs for discussion instead of stubbed responses.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary patch test directory for inspection.")
    args = parser.parse_args()

    workspace = Path("/tmp/opencode")
    workspace.mkdir(parents=True, exist_ok=True)

    install_import_stubs()

    patch_result = run_patch_smoke_test(workspace)
    discussion_result = run_discussion_smoke_test(real_apis=args.real_discussion)

    print("=== Patch Smoke Test ===")
    print(f"ok={patch_result['ok']}")
    print(f"command={patch_result['command']}")
    print(f"output={patch_result['output'].strip()}")
    print(f"target_file={patch_result['target_file']}")
    print(f"log_file={patch_result['log_file']}")
    print(f"content={patch_result['content'].strip()}")

    print("\n=== Discussion Smoke Test ===")
    print(f"ok={discussion_result['ok']}")
    print(f"mode={discussion_result['mode']}")
    print(f"discussion_triggered={discussion_result['discussion_triggered']}")
    print(f"result={discussion_result['result']}")
    print(f"logger_entries={discussion_result['logger_entries']}")
    print(f"log_file={discussion_result['log_file']}")

    if not args.keep_temp:
        shutil.rmtree(patch_result["temp_dir"], ignore_errors=True)

    if not patch_result["ok"] or not discussion_result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
