

"""
ONNX Decoder-Only Transformer
Tavily Context Retrieval + Question Answering

Flow:
Question -> Tavily -> Retrieved Context
         -> Context + Question -> ONNX -> Answer
"""

import os
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tavily import TavilyClient
from transformers import BertTokenizerFast

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from huggingface_hub import hf_hub_download

from dotenv import load_dotenv
import os

load_dotenv()
# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_DIR = Path("/tmp/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).parent
DEFAULT_TOKENIZER_DIR = BASE_DIR / "tokenizer"

DEFAULT_TOKENIZER_NAME = "bert-base-uncased"

MODEL_REPO_ID = "rav39439/summerization-model"

MODEL_FILENAME = "context_question_answer.onnx"

SEQ_LEN = 512

MAX_GEN = 150

MAX_CONTEXT_RESULTS = 4

MAX_CONTEXT_CHARACTERS = 5000

# Replace with the exact END_IDS used during training.
# SEP is used here as in your supplied inference code.


# =========================================================
# 2. FASTAPI INITIALIZATION
# =========================================================

app = FastAPI(
    title="ONNX Transformer Context QA API",
    description="Tavily-powered context-based question answering",
    version="1.0"
)

tokenizer = None

model = None

tavily_client = None


# =========================================================
# 3. REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):

    question: str

    max_gen: int = Field(
        default=MAX_GEN,
        ge=1,
        le=512
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0
    )


# =========================================================
# 4. LOAD TOKENIZER
# =========================================================


def load_tokenizer(tokenizer_path=None):

    configured_path = (
        tokenizer_path
        or os.environ.get("NEW_MODEL_TOKENIZER_PATH")
    )

    if configured_path:

        tokenizer_source = Path(
            configured_path
        ).expanduser()

        if not tokenizer_source.exists():

            raise FileNotFoundError(
                f"Tokenizer path does not exist: {tokenizer_source}"
            )

    elif DEFAULT_TOKENIZER_DIR.is_dir():

        tokenizer_source = DEFAULT_TOKENIZER_DIR

    else:

        tokenizer_source = os.environ.get(
            "NEW_MODEL_TOKENIZER_NAME",
            DEFAULT_TOKENIZER_NAME
        )

    loaded_tokenizer = BertTokenizerFast.from_pretrained(
        str(tokenizer_source)
    )

    # Do not add tokens if they are already present.
    special_tokens = [
        "<question>",
        "</question>",
        "<answer>",
        "</answer>",
        "<END>"
    ]

    missing_tokens = [
        token
        for token in special_tokens
        if token not in loaded_tokenizer.get_vocab()
    ]

    if missing_tokens:

        loaded_tokenizer.add_special_tokens({
            "additional_special_tokens": missing_tokens
        })

    # print("\nTokenizer loaded successfully.")

    # print("Tokenizer length:", len(loaded_tokenizer))

    # print("Vocabulary size:", loaded_tokenizer.vocab_size)

    # print("Special tokens:")

    for token in special_tokens:

        print(
            token,
            "->",
            loaded_tokenizer.convert_tokens_to_ids(token)
        )

    print("PAD token ID:", loaded_tokenizer.pad_token_id)

    print("SEP token ID:", loaded_tokenizer.sep_token_id)

    if loaded_tokenizer.pad_token_id is None:

        raise ValueError("Tokenizer PAD token is missing.")

    return loaded_tokenizer


# =========================================================
# 5. LOAD ONNX MODEL
# =========================================================

def load_onnx_model():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
        local_dir=MODEL_DIR
    )

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    # -----------------------------------------------------
    # Validate ONNX inputs
    # -----------------------------------------------------

    if len(session.get_inputs()) != 1:

        raise ValueError(
            "Expected exactly one ONNX input."
        )

    input_info = session.get_inputs()[0]

    if input_info.type != "tensor(int32)":

        raise TypeError(
            f"Expected tensor(int32), got {input_info.type}"
        )

    print("\nONNX model loaded successfully.")

    print("Input name:", input_info.name)

    print("Input shape:", input_info.shape)

    print("Output name:", session.get_outputs()[0].name)

    print("Output shape:", session.get_outputs()[0].shape)

    return session


# =========================================================
# 6. INITIALIZE TAVILY
# =========================================================

# def load_tavily_client():

#     api_key = os.getenv("TAVILY_API_KEY")

#     if not api_key:

#         raise RuntimeError(
#             "TAVILY_API_KEY environment variable is missing."
#         )

#     return TavilyClient(
#         api_key=api_key
#     )
from tavily import TavilyClient
import os

def load_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Check your .env file."
        )

    print("Tavily API key loaded successfully.")

    return TavilyClient(api_key=api_key)

# =========================================================
# 7. TAVILY WEB SEARCH
# =========================================================

# def search_web_context(question):

#     """
#     Retrieve web information using the user's question.

#     Returns:
#         context: Retrieved content.
#         sources: Source titles and URLs.
#     """

#     if tavily_client is None:

#         raise RuntimeError(
#             "Tavily client is not initialized."
#         )

#     print("\nSearching Tavily...")

#     print("Question:", question)

#     response = tavily_client.search(
#         query=question[:400],
#         search_depth="basic",
#         topic="general",
#         max_results=MAX_CONTEXT_RESULTS,
#         include_answer=False,
#         include_raw_content=False
#     )

#     results = response.get(
#         "results",
#         []
#     )

#     context_parts = []

#     sources = []

#     for index, result in enumerate(
#         results,
#         start=1
#     ):

#         title = result.get(
#             "title",
#             "Untitled"
#         )

#         content = result.get(
#             "content",
#             ""
#         )

#         url = result.get(
#             "url",
#             ""
#         )

#         if not content:

#             continue

#         context_parts.append(
#             f"""
# Source {index}:
# Title: {title}
# Content: {content}
# URL: {url}
# """.strip()
#         )

#         sources.append({

#             "title": title,

#             "url": url

#         })

#     context = "\n\n".join(
#         context_parts
#     )

#     context = context[:MAX_CONTEXT_CHARACTERS]

#     # print(
#     #     "Retrieved sources:",
#     #     len(sources)
#     # )

#     return context, sources


import re


def search_web_context(question):

    """
    Retrieve clean textual context using Tavily.

    Returns:
        context: Plain-text paragraphs limited to 600 tokens.
        sources: Source titles and URLs (separate from context).
    """

    if tavily_client is None:
        raise RuntimeError(
            "Tavily client is not initialized."
        )

    MAX_CONTEXT_TOKENS = 600

    print("\nSearching Tavily...")
    print("Question:", question)

    response = tavily_client.search(
        query=question[:400],
        search_depth="basic",
        topic="general",
        max_results=MAX_CONTEXT_RESULTS,
        include_answer=False,
        include_raw_content=False
    )

    results = response.get("results", [])

    context_parts = []
    sources = []

    for result in results:

        content = result.get("content", "")

        if not content:
            continue

        # Remove HTML tags.
        content = re.sub(
            r"<[^>]+>",
            " ",
            content
        )

        # Remove URLs.
        content = re.sub(
            r"https?://\S+",
            "",
            content
        )

        # Remove Markdown images and links.
        content = re.sub(
            r"!\[.*?\]\(.*?\)",
            "",
            content
        )

        content = re.sub(
            r"\[(.*?)\]\(.*?\)",
            r"\1",
            content
        )

        # Normalize whitespace.
        content = re.sub(
            r"\s+",
            " ",
            content
        ).strip()

        if not content:
            continue

        context_parts.append(content)

        # Sources remain separate from context.
        sources.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", "")
        })

    # Combine only textual paragraphs.
    context = "\n\n".join(context_parts)

    # Enforce an actual 600-token limit.
    tokens = tokenizer.encode(
        context,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS
    )

    context = tokenizer.decode(
        tokens,
        skip_special_tokens=True
    )

    print("Retrieved sources:", len(sources))
    print("Context tokens:", len(tokens))

    return context, sources


# =========================================================
# 8. VALIDATE TOKENIZER AND MODEL VOCABULARY
# =========================================================

def validate_vocabulary():

    model_vocab_size = model.get_outputs()[0].shape[-1]

    print("\nModel output vocabulary:", model_vocab_size)

    print("Tokenizer length:", len(tokenizer))

    if len(tokenizer) != model_vocab_size:

        raise ValueError(
            "Tokenizer/model vocabulary mismatch. "
            f"Tokenizer: {len(tokenizer)}, "
            f"Model: {model_vocab_size}"
        )

# =========================================================
# 9. END TOKEN CONFIGURATION
# =========================================================

def get_end_ids():

    # Use the exact END_IDS from training.
    # SEP is used as the default.

    end_ids = [
        tokenizer.sep_token_id
    ]

    if any(
        token_id is None
        for token_id in end_ids
    ):

        raise ValueError(
            "Invalid END_IDS configuration."
        )

    print("End token IDs:", end_ids)

    return end_ids


# =========================================================
# 10. PREPARE CONTEXT + QUESTION PROMPT
# =========================================================

def prepare_input_tokens(
    context,
    question
):

    """
    Prompt format:

    Context: Retrieved information

    Question: User question

    Answer:
    """

    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    if not token_ids:

        raise ValueError(
            "Context and question produced no tokens."
        )

    # Preserve the end of the prompt.
    token_ids = token_ids[
        -(SEQ_LEN - 1):
    ]

    return token_ids


# =========================================================
# 11. SINGLE ONNX GENERATION STEP
# =========================================================

def generate_next_token(
    token_ids,
    temperature=0.0
):

    """
    Execute one ONNX inference step.

    Input:
        Fixed-length int32 array of 512 tokens.

    Output:
        Next predicted token ID.
    """

    actual_length = len(token_ids)

    # Fixed-length right padding.
    padded_ids = (
        token_ids
        + [tokenizer.pad_token_id] *
        (SEQ_LEN - actual_length)
    )

    input_array = np.asarray(
        [padded_ids],
        dtype=np.int32
    )

    logits = model.run(
        None,
        {
            model.get_inputs()[0].name: input_array
        }
    )[0]

    # Select logits at the last actual token position.
    next_token_logits = logits[
        0,
        actual_length - 1,
        :
    ]

    if temperature == 0:
        next_id = int(np.argmax(next_token_logits))
    else:
        # Temperature-scaled sampling: lower values favor high-scoring tokens.
        scaled_logits = next_token_logits.astype(np.float64) / temperature
        scaled_logits -= np.max(scaled_logits)
        probabilities = np.exp(scaled_logits)
        probabilities /= probabilities.sum()
        next_id = int(np.random.choice(len(probabilities), p=probabilities))

    # Validate generated token ID.
    if next_id >= len(tokenizer):

        raise ValueError(
            f"Generated token ID {next_id} is outside "
            f"the tokenizer vocabulary ({len(tokenizer)}). "
            "Check the tokenizer used during training."
        )

    return next_id


# =========================================================
# 12. COMPLETE ANSWER GENERATION
# =========================================================

def generate_answer(
    context,
    question,
    max_gen=MAX_GEN,
    temperature=0.0
):

    if max_gen < 1:

        return ""

    # Prepare prompt.
    token_ids = prepare_input_tokens(
        context=context,
        question=question
    )

    generated_ids = []

    end_ids = get_end_ids()

    # Autoregressive generation.
    for step in range(max_gen):

        # Combine prompt and previously generated tokens.
        context_ids = (
            token_ids + generated_ids
        )[-SEQ_LEN:]

        next_id = generate_next_token(
            context_ids,
            temperature=temperature
        )

        generated_ids.append(
            next_id
        )

        # Stop-token checking.
        if (
            len(generated_ids) >= len(end_ids)
            and generated_ids[-len(end_ids):] == end_ids
        ):

            generated_ids = generated_ids[
                :-len(end_ids)
            ]

            break

    # Decode answer.
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return answer


# =========================================================
# 13. STREAMING ANSWER GENERATION
# =========================================================

def generate_answer_stream(
    context,
    question,
    max_gen=MAX_GEN,
    temperature=0.0
):

    """
    Generate and yield the complete decoded prefix.

    The API endpoint sends only newly generated text.
    """

    if max_gen < 1:

        return

    token_ids = prepare_input_tokens(
        context=context,
        question=question
    )

    generated_ids = []

    end_ids = get_end_ids()

    for step in range(max_gen):

        context_ids = (
            token_ids + generated_ids
        )[-SEQ_LEN:]

        next_id = generate_next_token(
            context_ids,
            temperature=temperature
        )

        generated_ids.append(
            next_id
        )

        # Stop-token checking.
        if (
            len(generated_ids) >= len(end_ids)
            and generated_ids[-len(end_ids):] == end_ids
        ):

            generated_ids = generated_ids[
                :-len(end_ids)
            ]

            break

        decoded_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        if decoded_text:

            yield decoded_text


# =========================================================
# 14. STARTUP EVENT
# =========================================================

@app.on_event("startup")
def startup_event():

    global tokenizer
    global model
    global tavily_client

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()

    print("\nLoading ONNX model...")

    model = load_onnx_model()

    validate_vocabulary()

    print("\nInitializing Tavily...")

    tavily_client = load_tavily_client()

    print("\nAll components loaded successfully.")


# =========================================================
# 15. NON-STREAMING API
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the lightweight streaming inference interface."""
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Streaming Question Answering</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #182230; background: #f4f7fb; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 28px 16px; }
    main { width: min(760px, 100%); background: white; border: 1px solid #e4eaf2; border-radius: 18px; padding: 28px; box-shadow: 0 16px 45px #19365a12; }
    h1 { margin: 0 0 8px; font-size: 1.65rem; }
    .sub { margin: 0 0 24px; color: #65758b; }
    label { display: block; font-weight: 650; margin: 16px 0 8px; }
    textarea { width: 100%; min-height: 116px; resize: vertical; padding: 13px; border: 1px solid #cbd5e1; border-radius: 10px; font: inherit; }
    .controls { display: grid; grid-template-columns: 1fr 120px; gap: 18px; align-items: end; }
    input[type=range] { width: 100%; accent-color: #3b6eea; }
    output { display: inline-block; min-width: 38px; font-variant-numeric: tabular-nums; color: #3b6eea; }
    input[type=number] { width: 100%; padding: 10px; border: 1px solid #cbd5e1; border-radius: 9px; font: inherit; }
    button { margin-top: 20px; padding: 11px 18px; border: 0; border-radius: 9px; background: #315fdb; color: white; font: inherit; font-weight: 650; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    .status { min-height: 24px; margin: 16px 0 8px; color: #65758b; font-size: .92rem; }
    section { margin-top: 18px; padding: 16px; border-radius: 11px; background: #f7f9fc; }
    section h2 { margin: 0 0 10px; font-size: .95rem; color: #53647a; }
    #answer, #context { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; }
    #context { max-height: 220px; overflow: auto; color: #53647a; font-size: .9rem; }
    @media (max-width: 520px) { main { padding: 20px; } .controls { grid-template-columns: 1fr 88px; gap: 10px; } }
  </style>
</head>
<body>
<main>
  <h1>Streaming Question Answering</h1>
  <p class="sub">Ask a question and watch the answer arrive as it is generated.</p>
  <form id="form">
    <label for="question">Your question</label>
    <textarea id="question" required placeholder="What would you like to know?"></textarea>
    <div class="controls">
      <div>
        <label for="temperature">Temperature <output id="temperatureValue">0.7</output></label>
        <input id="temperature" type="range" min="0" max="1" step="0.05" value="0.7">
      </div>
      <div>
        <label for="maxGen">Max tokens</label>
        <input id="maxGen" type="number" min="1" max="512" value="150">
      </div>
    </div>
    <button id="submit" type="submit">Generate answer</button>
  </form>
  <div class="status" id="status" role="status" aria-live="polite"></div>
  <section><h2>Streaming answer</h2><div id="answer">Your answer will appear here.</div></section>
  <section><h2>Retrieved context</h2><div id="context">Context will appear when generation starts.</div></section>
</main>
<script>
const form = document.querySelector('#form');
const temp = document.querySelector('#temperature');
const tempValue = document.querySelector('#temperatureValue');
temp.addEventListener('input', () => tempValue.value = Number(temp.value).toFixed(2).replace(/0+$/, '').replace(/\\.$/, ''));
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = document.querySelector('#submit');
  const status = document.querySelector('#status');
  const answer = document.querySelector('#answer');
  const context = document.querySelector('#context');
  button.disabled = true;
  answer.textContent = '';
  context.textContent = 'Loading context…';
  status.textContent = 'Searching and starting inference…';
  try {
    const response = await fetch('/generate/stream', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: document.querySelector('#question').value,
        max_gen: Number(document.querySelector('#maxGen').value), temperature: Number(temp.value)})
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail || `Request failed (${response.status})`);
    }
    status.textContent = 'Generating…';
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;
    while (!finished) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop();
      for (const eventText of events) {
        const line = eventText.split(/\r?\n/).find(line => line.startsWith('data: '));
        if (!line) continue;
        const data = line.slice(6);
        if (data === '[DONE]') { status.textContent = 'Done'; finished = true; break; }
        if (data === '<terminated>') { status.textContent = 'Generation stopped by the repetition/context guard.'; finished = true; break; }
        try {
          const item = JSON.parse(data);
          if (item.context !== undefined) context.textContent = item.context || '(No context returned)';
          if (item.token !== undefined) answer.textContent += item.token;
          if (item.error) throw new Error(item.error);
        } catch (error) {
          if (error instanceof SyntaxError) throw error;
          throw error;
        }
      }
      if (done) finished = true;
    }
    if (!answer.textContent) answer.textContent = '(No answer generated)';
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
    if (context.textContent === 'Loading context…') context.textContent = 'Context unavailable.';
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>""")

@app.post("/generate")
def generate(
    request: QuestionRequest
):

    try:

        # STEP 1: Retrieve context from Tavily.

        context, sources = search_web_context(
            request.question
        )

        # STEP 2: ONNX inference.

        answer = generate_answer(
            context=context,
            question=request.question,
            max_gen=request.max_gen,
            temperature=request.temperature
        )

        # STEP 3: Return response.

        return {

            "question": request.question,

            "context": context,

            "sources": "",

            "answer": answer

        }

    except Exception as e:

        print(
            "Generation error:",
            str(e)
        )

        raise HTTPException(
            status_code=502,
            detail=str(e)
        )


# =========================================================
# 16. STREAMING API
# =========================================================

MIN_CONTEXT_TERM_OVERLAP = 0.2

HALLUCINATION_STOPWORDS = frozenset(
    "a an and are as at be been being but by for from had has have "
    "he her hers him his i if in into is it its me more most my nor "
    "of on or our ours she so than that the their theirs them then "
    "there these they this those to too under up us was we were what "
    "when where which who why will with you your".split()
)

SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+")


def is_hallucinated(sentence, context):
    """Treat very low lexical overlap with context as a rough hallucination signal."""

    answer_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", sentence.lower())
        if len(term) > 2 and term not in HALLUCINATION_STOPWORDS
    }

    if not answer_terms:
        return False

    context_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", context.lower())
    }

    overlap = len(answer_terms & context_terms) / len(answer_terms)

    return overlap < MIN_CONTEXT_TERM_OVERLAP


def has_repeated_token(sentence):
    """Return whether a word occurs more than twice in the current sentence."""

    token_counts = {}

    for token in re.findall(r"[a-z0-9]+", sentence.lower()):
        token_counts[token] = token_counts.get(token, 0) + 1

        if token_counts[token] > 2:
            return True

    return False


def monitor_answer_stream(context, generated_text_stream):
    """Yield sentence events, terminating on low overlap or repeated words."""

    previous_text = ""
    pending_text = ""

    for current_text in generated_text_stream:
        new_text = current_text[len(previous_text):]
        previous_text = current_text
        pending_text += new_text

        while True:
            boundary = SENTENCE_BOUNDARY.search(pending_text)

            if boundary is None:
                break

            sentence = pending_text[:boundary.end()]
            pending_text = pending_text[boundary.end():]

            if (
                has_repeated_token(sentence)
                or is_hallucinated(sentence, context)
            ):
                yield {"signal": "<terminated>"}
                return

            yield {"token": sentence}

        if has_repeated_token(pending_text):
            yield {"signal": "<terminated>"}
            return

    if pending_text:
        if (
            has_repeated_token(pending_text)
            or is_hallucinated(pending_text, context)
        ):
            yield {"signal": "<terminated>"}
            return

        yield {"token": pending_text}


@app.post("/generate/stream")
def generate_stream(
    request: QuestionRequest
):

    # Retrieve context before starting stream.
    try:

        context, sources = search_web_context(
            request.question
        )

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=f"Tavily search failed: {str(e)}"
        )

    def stream():

        try:

            # First send context and sources.

            yield (
                "data: "
                + json.dumps({

                    "context": context,

                    "sources": ""

                })
                + "\n\n"
            )

            generated_text_stream = generate_answer_stream(
                context=context,
                question=request.question,
                max_gen=request.max_gen,
                temperature=request.temperature
            )

            for event in monitor_answer_stream(
                context,
                generated_text_stream
            ):

                if event.get("signal") == "<terminated>":
                    yield "data: <terminated>\n\n"
                    return

                yield (
                    "data: "
                    + json.dumps(event)
                    + "\n\n"
                )

            yield "data: [DONE]\n\n"

        except Exception as e:

            yield (
                "data: "
                + json.dumps({

                    "error": str(e)

                })
                + "\n\n"
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={

            "Cache-Control": "no-cache",

            "Connection": "keep-alive",

            "X-Accel-Buffering": "no"

        }
    )


# =========================================================
# 17. MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
