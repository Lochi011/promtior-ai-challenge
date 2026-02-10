"""Smoke tests for the Promtior Bionic Agent."""

from app.agent import agent_executor


def test_promtior_identity() -> None:
    """Verify the agent returns correct Promtior founding facts."""
    result = agent_executor.invoke({
        "question": "When was Promtior founded and who are the founders?"
    })

    answer: str = result["answer"]
    print(f"\nANSWER:\n{answer}\n")

    assert "2023" in answer, "Expected founding year 2023"
    assert "Emiliano" in answer, "Expected founder Emiliano Chinelli"
    assert "Ignacio" in answer, "Expected founder Ignacio Acuña"
    print("All identity assertions passed.")


def test_promtior_services() -> None:
    """Verify the agent can describe Promtior's services."""
    result = agent_executor.invoke({
        "question": "What services does Promtior offer?"
    })

    answer: str = result["answer"]
    print(f"\nANSWER:\n{answer}\n")

    assert len(answer) > 20, "Expected a substantive answer about services"
    assert "don't have enough" not in answer.lower(), "Agent should NOT refuse"
    print("Services assertion passed.")


def test_promtior_case_studies() -> None:
    """Verify the agent can describe Promtior's clients and case studies."""
    result = agent_executor.invoke({
        "question": "Tell me about Promtior's case studies and clients."
    })

    answer: str = result["answer"]
    print(f"\nANSWER:\n{answer}\n")

    assert len(answer) > 20, "Expected a substantive answer about clients"
    assert "don't have enough" not in answer.lower(), "Agent should NOT refuse"
    print("Case studies assertion passed.")


if __name__ == "__main__":
    test_promtior_identity()
    test_promtior_services()
    test_promtior_case_studies()