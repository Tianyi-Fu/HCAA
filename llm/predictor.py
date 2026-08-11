from llm.utils import (
    predict_next_action,
    generate_prompt_for_goals,
    extract_response_details,
)

def run_prediction(
    user_command: str,
    previous_command: str | None = None,
    *,
    context: str | None = None,
    context_items: dict | None = None,
    missing_components: list | None = None,
    predicted_history_command: str | None = None,
    context_expanded_info: dict | None = None,
):
    prompt = generate_prompt_for_goals(
        current_command=user_command,
        previous_command=previous_command,
        context=context,
        context_items=context_items,
        missing_components=missing_components,
        predicted_history_command=predicted_history_command,
        context_expanded_info=context_expanded_info,
    )

    raw_response = predict_next_action(prompt)
    return extract_response_details(raw_response)
