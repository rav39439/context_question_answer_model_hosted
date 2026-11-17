# import requests
# import json

# response = requests.post(
#     "http://127.0.0.1:8000/generate/stream",
#     json={
#         "question": "What is the capital of France?",
#         "max_gen": 40
#     },
#     stream=True
# )

# response.raise_for_status()

# print("Streaming response:\n")

# for line in response.iter_lines(decode_unicode=True):

#     if line:

#         if line.startswith("data: "):

#             data = line[6:]

#             if data == "[DONE]":
#                 print("\n\nGeneration completed.")
#                 break

#             result = json.loads(data)

#             if "token" in result:
#                 print(result["token"], end="", flush=True)

#             elif "error" in result:
#                 print("\nError:", result["error"])
#                 break

# response.close()



import requests
import json


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/generate/stream"

QUESTION = "How can I improve my language skills in a foreign language?"

MAX_GEN = 60


# =========================================================
# SEND STREAMING REQUEST
# =========================================================

response = requests.post(
    API_URL,
    json={
        "question": QUESTION,
        "max_gen": MAX_GEN
    },
    stream=True,
    timeout=(15, 300)
)

response.raise_for_status()


# =========================================================
# DISPLAY QUESTION
# =========================================================

print("\n" + "=" * 70)
print("TAVILY + ONNX TRANSFORMER")
print("=" * 70)

print("\nQUESTION:")
print(QUESTION)


# =========================================================
# PROCESS STREAMING RESPONSE
# =========================================================

print("\nRETRIEVED CONTEXT AND SOURCES:\n")

answer_started = False

try:

    for line in response.iter_lines(
        decode_unicode=True
    ):

        if not line:
            continue

        if not line.startswith("data: "):
            continue

        data = line[6:]

        if data == "<terminated>":

            print("\n\nGeneration terminated by the response monitor.")

            break

        if data == "[DONE]":

            print("\n\nGeneration completed.")

            break

        result = json.loads(data)

        # -------------------------------------------------
        # Retrieved context and sources
        # -------------------------------------------------

        if "sources" in result:

            # context = result.get(
            #     "context",
            #     ""
            # )

            # sources = result.get(
            #     "sources",
            #     []
            # )

            # print("-" * 70)

            # print("\nTAVILY CONTEXT:\n")

            # print(context if context else "No context retrieved.")

            # print("\nTAVILY SOURCES:\n")

            # for index, source in enumerate(
            #     sources,
            #     start=1
            # ):

            #     print(
            #         f"{index}. {source.get('title', 'Untitled')}"
            #     )

            #     print(
            #         f"   URL: {source.get('url', '')}"
            #     )

            # print("\n" + "=" * 70)

            print("GENERATED ANSWER:\n")

        # -------------------------------------------------
        # Generated answer tokens
        # -------------------------------------------------

        elif "token" in result:

            if not answer_started:

                answer_started = True

            print(
                result["token"],
                end="",
                flush=True
            )

        # -------------------------------------------------
        # Error handling
        # -------------------------------------------------

        elif "error" in result:

            print(
                "\n\nERROR:",
                result["error"]
            )

            break

finally:

    response.close()


# =========================================================
# COMPLETED
# =========================================================

print("\n" + "=" * 70)
