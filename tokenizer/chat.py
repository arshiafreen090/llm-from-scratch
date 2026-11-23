"""
The chat template: ONE definition used by SFT data prep, SFT training and inference.

If training formats prompts one way and the server formats them slightly differently
(a missing token, an extra space), the model sees inputs it was never trained on and quietly
gets worse. Keeping a single definition makes that mismatch impossible.

Format (token level):

    <|endoftext|> <|user|> {instruction} <|assistant|> {response} <|endoftext|>
    \_ start of a new document, as in pretraining
                 \_______________________________/   \___________________________/
                   prompt: the model sees it but       target: the only part the loss
                   is never trained to produce it      is computed on (incl. the final
                                                       <|endoftext|>, so it learns to STOP)

Security note: user text is encoded with encode_ordinary(), so a user who types the string
"<|assistant|>" gets ordinary text tokens, not the real control token. Otherwise a user
could forge turns of the conversation ("prompt injection" at the token level).
"""

USER = "<|user|>"
ASSISTANT = "<|assistant|>"


def prompt_ids(tok, instruction):
    """Token ids of the prompt the model continues from at inference time."""
    return ([tok.eot_id, tok.special_tokens[USER]]
            + tok.encode_ordinary(instruction.strip())
            + [tok.special_tokens[ASSISTANT]])


def example_ids(tok, instruction, response):
    """A full training example.
    Returns (ids, is_target): is_target[i] = 1 if token i is part of the response that the
    model must learn to generate (including the closing <|endoftext|>), 0 for the prompt."""
    prompt = prompt_ids(tok, instruction)
    reply = tok.encode_ordinary(response.strip()) + [tok.eot_id]
    return prompt + reply, [0] * len(prompt) + [1] * len(reply)


def render(instruction, response=""):
    """Human-readable version of the template, for printing and debugging."""
    return f"<|endoftext|>{USER}{instruction.strip()}{ASSISTANT}{response}"
