import re
import gradio as gr
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ── System prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are CyberDecoder, a friendly cybersecurity assistant. You translate cybersecurity jargon, CVE reports, security alerts, compliance terms, and technical vulnerabilities into plain English that anyone can understand, whether they are a business owner, a developer, or a non-technical user.

How to answer:
- Answer only what the user asked, in 2 to 4 short sentences of plain English.
- Do not add headings, severity ratings, lists of steps, or tags unless the user asks for them.
- After answering, end with one short question asking if they want to know more, tailored to the topic. For example: "Would you like to know how serious this is, or how to protect against it?"
- Never use unexplained technical acronyms. If you must use one, define it immediately.
- If something is outside your knowledge base, say: "This is not in my current knowledge base. For the most accurate answer, consult a certified security professional."
- Be calm, professional and friendly, like a knowledgeable friend, never alarmist or dismissive.
- If the user pastes a phishing email or suspicious link, point out the red flags without clicking or reproducing harmful content.
- If the user asks something unrelated to cybersecurity, politely redirect: "I'm specialized in cybersecurity topics. Try asking me about a security term, CVE, or threat you've encountered."
- Do not introduce yourself and do not mention who built you.
"""

# ── Prompt template (glues system prompt + retrieved context + question) ──
PROMPT_TEMPLATE = SYSTEM_PROMPT + """

Knowledge base context ({context_note}):
{context}

Conversation so far:
{history}

Question: {question}

Instructions for this answer:
- Answer only the question asked, in 2 to 4 short plain-English sentences, about the exact term the user named. Never swap in a different term.
- Do not use headings, severity ratings, step lists, or tags unless the user asks for them.
- If the context is unrelated or only loosely related, ignore it.
- If you do not confidently recognise a term, acronym, or organisation, do not invent a definition. Use the not-in-knowledge-base sentence and ask what the acronym stands for.
- End with one short question offering more detail, such as whether they want to know how serious it is, what could happen, or how to protect against it.

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "context_note", "history", "question"]
)

# ── Load vector DB + model ─────────────────────────────────────
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

llm = Ollama(model="llama3.2:3b", keep_alive="30m", num_ctx=2048, num_predict=250, temperature=0.2)

TYPING = '<span class="typing"><i></i><i></i><i></i></span>'

# ── Chat logic ─────────────────────────────────────────────────
TOPICS = [
    "What is SQL injection?",
    "What does CVSS 10 mean?",
    "Explain what ransomware is",
    "What is a zero-day vulnerability?",
    "What is phishing and how do I spot it?",
]


def _text(content):
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def respond(message, history):
    message = (message or "").strip()
    if not message:
        yield gr.update(), gr.update(), ""
        return
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": TYPING},
    ]
    yield gr.update(visible=False), gr.update(value=history, visible=True), ""

    prior_users = [_text(m["content"]) for m in history[:-2] if m["role"] == "user"]
    is_followup = bool(prior_users) and bool(
        re.match(r"^(yes|yeah|yep|sure|ok|okay|please|go on|tell me more|more)\b", message, re.I)
        or (len(message.split()) <= 8 and re.search(r"\b(it|this|that|they|them)\b", message, re.I))
    )
    search_text = prior_users[-1] + " " + message if is_followup else message
    hits = db.similarity_search_with_score(search_text, k=2)
    best = hits[0][1] if hits else 9.0
    if best <= 0.6:
        note, context = "strong match", "\n\n".join(d.page_content for d, _ in hits)
    elif best <= 1.1:
        note, context = "weak match, may not be about the same term", hits[0][0].page_content
    else:
        note, context = "no relevant match found", "None."
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", message)
    haystack = " ".join(d.page_content for d, _ in hits).upper()
    missing = [a for a in acronyms if a not in haystack]
    if best > 0.45 and missing:
        history[-1]["content"] = (
            "This is not in my current knowledge base. For the most accurate answer, "
            "consult a certified security professional.\n\n"
            f"I do not recognise {', '.join(missing)}. What does it stand for? "
            "If you tell me, I can explain it in plain English."
        )
        yield gr.update(visible=False), gr.update(value=history), ""
        return
    recent = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {_text(m['content'])}"
        for m in (history[:-2][-4:] if is_followup else [])
    ) or "None."
    full_prompt = prompt.format(
        context=context, context_note=note, history=recent, question=message
    )
    answer = ""
    for token in llm.stream(full_prompt):
        answer += token
        history[-1]["content"] = answer
        yield gr.update(visible=False), gr.update(value=history), ""


def new_chat():
    return gr.update(visible=True), gr.update(value=[], visible=False), ""


def topic_handler(topic):
    def handler(history):
        yield from respond(topic, history)
    return handler


# ── UI pieces ──────────────────────────────────────────────────
MASCOT = """
<svg viewBox="0 0 200 220" width="170" height="187" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <ellipse cx="100" cy="205" rx="38" ry="6" fill="#5b5fc7" opacity="0.18"/>
  <path d="M100 8 C122 40 160 62 154 108 C150 140 124 158 100 158 C76 158 50 140 46 108 C40 62 78 40 100 8 Z" fill="#5b55e6"/>
  <path d="M100 30 C110 52 128 66 126 92" stroke="#a9a2ff" stroke-width="5" fill="none" stroke-linecap="round" opacity="0.6"/>
  <circle cx="100" cy="62" r="5" fill="#5fe0d0"/>
  <ellipse class="eye" cx="78" cy="102" rx="14" ry="15" fill="#1a1a3a"/>
  <ellipse class="eye" cx="122" cy="102" rx="14" ry="15" fill="#1a1a3a"/>
  <circle cx="82" cy="97" r="4.5" fill="#fff"/>
  <circle cx="126" cy="97" r="4.5" fill="#fff"/>
  <path d="M60 142 Q100 168 140 142 L136 158 Q100 180 64 158 Z" fill="#5fe0d0"/>
  <path d="M46 128 C24 128 14 146 22 160 C30 170 44 160 50 148" fill="#4b47d6"/>
  <path d="M154 128 C176 128 186 146 178 160 C170 170 156 160 150 148" fill="#4b47d6"/>
  <path d="M84 156 C84 184 96 196 100 200 C104 196 116 184 116 156 Z" fill="#4b47d6"/>
</svg>
"""

BRAND = """
<div class="brand">
  <svg viewBox="0 0 40 44" width="30" height="33" aria-hidden="true">
    <path d="M20 2 C25 10 34 15 33 25 C32 34 26 39 20 39 C14 39 8 34 7 25 C6 15 15 10 20 2 Z" fill="#4b47d6"/>
    <circle cx="20" cy="17" r="2" fill="#5fe0d0"/>
  </svg>
  <span>CYBER-DECODER</span>
</div>
"""

HERO = f"""
<div class="hero">
  <span class="spark s1">&#10022;</span><span class="spark s2">&#10022;</span><span class="spark s3">&#10022;</span><span class="spark s4">&#10022;</span>
  {MASCOT}
  <h1>How can I decode it for you today?</h1>
  <p>Paste any CVE, security alert, phishing email, or confusing compliance term.
  I will turn it into plain English and tell you exactly what to do.</p>
</div>
"""

CSS = """
:root, .dark, [data-theme="dark"] {
    --body-text-color: #1d1d3a;
    --body-text-color-subdued: #55557a;
    --block-label-text-color: #1d1d3a;
    --input-background-fill: transparent;
    color-scheme: light;
}

html, body { background: #b8c2f4 !important; min-height: 100vh; }
gradio-app, .gradio-container, .app { background: transparent !important; position: relative; z-index: 1; }
.gradio-container { max-width: 1250px !important; margin: 0 auto !important; padding: 24px 16px !important; }
footer { display: none !important; }

/* ---------- frosted glass shell (flat tint, blur only) ---------- */
#shell {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 32px;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    box-shadow: inset 0 2px 0 rgba(255,255,255,0.9), 0 24px 60px rgba(60, 60, 150, 0.25);
    padding: 0 !important;
    overflow: hidden;
    gap: 0 !important;
    min-height: 82vh;
}
#sidebar {
    background: rgba(255, 255, 255, 0.35);
    padding: 22px 18px !important;
    gap: 10px !important;
    border-right: 1px solid rgba(255, 255, 255, 0.8);
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 800; letter-spacing: 0.08em;
         white-space: nowrap; font-size: 15px; color: #22224a; margin-bottom: 12px; }
.brand svg { animation: bob 3.2s ease-in-out infinite; }
.side-label { font-size: 13px; font-weight: 700; color: #22224a; margin: 14px 4px 2px; }

/* ---------- buttons ---------- */
.side-btn button, button.side-btn {
    background: #ffffff !important; border: 1px solid #ffffff !important; border-radius: 999px !important;
    color: #22224a !important; text-align: left !important; justify-content: flex-start !important;
    padding: 12px 18px !important; font-size: 14px !important; font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(60,60,160,0.18) !important;
    transition: transform .25s cubic-bezier(.34,1.56,.64,1), box-shadow .25s !important;
}
.side-btn button:hover, button.side-btn:hover { transform: translateY(-2px) scale(1.03); box-shadow: 0 12px 26px rgba(60,60,160,0.28) !important; }

.topic-btn button, button.topic-btn {
    background: transparent !important; border: 1px solid transparent !important; box-shadow: none !important;
    color: #22224a !important; text-align: left !important; justify-content: flex-start !important;
    padding: 7px 12px !important; font-size: 13px !important; border-radius: 14px !important;
    transition: all .2s ease !important;
}
.topic-btn button:hover, button.topic-btn:hover {
    background: rgba(255,255,255,0.75) !important; border-color: #ffffff !important;
    color: #3d3ad0 !important; transform: translateX(5px);
}
.tag-foot { margin-top: auto; font-size: 12px; color: #3d3ad0; font-weight: 700; padding: 8px 4px; }

#main { padding: 28px 36px !important; gap: 14px !important; background: transparent !important; }
#main > * { flex-grow: 0 !important; }

/* ---------- hero ---------- */
.hero { text-align: center; padding: 26px 10px 10px; position: relative; }
.hero svg { display: block; margin: 0 auto; animation: float 4s ease-in-out infinite; filter: drop-shadow(0 16px 14px rgba(61, 58, 208, 0.3)); }
.hero .eye { transform-box: fill-box; transform-origin: center; animation: blink 5s infinite; }
.hero h1 { font-size: 32px; font-weight: 800; margin: 12px 0 6px; color: #2a28b8; }
.hero p { color: #3f3f66; max-width: 560px; margin: 0 auto; font-size: 14px; line-height: 1.6; }
.spark { position: absolute; color: #ffffff; font-size: 22px; animation: twinkle 2.6s ease-in-out infinite; }
.spark.s1 { left: 30%; top: 40px; } .spark.s2 { right: 29%; top: 70px; animation-delay: .8s; font-size: 16px; }
.spark.s3 { left: 36%; top: 150px; animation-delay: 1.5s; font-size: 14px; } .spark.s4 { right: 34%; top: 26px; animation-delay: 2s; font-size: 12px; }
@keyframes float { 0%,100% { transform: translateY(0) rotate(-2deg); } 50% { transform: translateY(-14px) rotate(2deg); } }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
@keyframes blink { 0%,44%,50%,100% { transform: scaleY(1); } 47% { transform: scaleY(0.08); } }
@keyframes twinkle { 0%,100% { opacity: .2; transform: scale(.6) rotate(0); } 50% { opacity: 1; transform: scale(1.15) rotate(25deg); } }

/* ---------- chat ---------- */
#chat { background: transparent !important; border: none !important; box-shadow: none !important; }
#chat .bubble-wrap, #chat .message-wrap, #chat .placeholder-content { background: transparent !important; }
#chat .bot, #chat [data-testid="bot"] {
    background: #ffffff !important;
    border: 1px solid #ffffff !important; border-radius: 20px 20px 20px 6px !important;
    box-shadow: 0 8px 22px rgba(60,60,160,0.14) !important;
    backdrop-filter: blur(14px); animation: pop .35s cubic-bezier(.34,1.56,.64,1);
}
#chat .user, #chat [data-testid="user"] {
    background: #5b5fe0 !important;
    border: 1px solid #5b5fe0 !important; border-radius: 20px 20px 6px 20px !important;
    box-shadow: 0 8px 20px rgba(70,70,210,0.35) !important;
    animation: pop .35s cubic-bezier(.34,1.56,.64,1);
}
#chat .bot .message-content, #chat .bot .prose, #chat .bot > div, #chat .user .message-content, #chat .user > div { background: transparent !important; border: none !important; box-shadow: none !important; }
#chat .icon-button-wrapper, #chat .message-buttons, #chat .message-buttons-left, #chat .message-buttons-right { display: none !important; }
#chat .bot, #chat .bot *, #chat [data-testid="bot"], #chat [data-testid="bot"] * { color: #1d1d3a !important; opacity: 1 !important; }
#chat .user, #chat .user *, #chat [data-testid="user"], #chat [data-testid="user"] * { color: #ffffff !important; opacity: 1 !important; }
@keyframes pop { from { opacity: 0; transform: translateY(12px) scale(.95); } to { opacity: 1; transform: none; } }

.typing { display: inline-flex; gap: 6px; align-items: center; padding: 6px 4px; }
.typing i { width: 10px; height: 10px; border-radius: 50%; display: block; background: #5b5fe0; animation: dot 1.1s ease-in-out infinite; }
.typing i:nth-child(2) { animation-delay: .18s; } .typing i:nth-child(3) { animation-delay: .36s; }
@keyframes dot { 0%,70%,100% { transform: translateY(0) scale(.8); opacity: .45; } 35% { transform: translateY(-8px) scale(1.15); opacity: 1; } }

/* ---------- input + chips ---------- */
#input-row {
    background: rgba(255,255,255,0.92);
    border: 1px solid #ffffff; border-radius: 999px; padding: 6px 8px 6px 18px !important;
    align-items: center; gap: 8px !important; flex-grow: 0 !important;
    box-shadow: 0 10px 28px rgba(60,60,160,0.2);
    transition: box-shadow .25s, transform .25s;
}
#input-row:focus-within { box-shadow: 0 0 0 4px rgba(91,95,224,0.3), 0 14px 34px rgba(60,60,160,0.28); transform: translateY(-1px); }
#input-row textarea, #input-row input {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: #1d1d3a !important; -webkit-text-fill-color: #1d1d3a !important; caret-color: #3d3ad0 !important;
}
#input-row textarea::placeholder { color: #8a8aab !important; -webkit-text-fill-color: #8a8aab !important; }
#input-row .block, #input-row > div { background: transparent !important; border: none !important; box-shadow: none !important; }
#send-btn button, button#send-btn {
    background: #16162e !important; color: #ffffff !important; border-radius: 999px !important;
    min-width: 96px !important; border: none !important; font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(22,22,46,0.35) !important;
    transition: transform .25s cubic-bezier(.34,1.56,.64,1) !important;
}
#send-btn button:hover, button#send-btn:hover { transform: scale(1.07) rotate(-1.5deg); }

#chips { gap: 8px !important; flex-wrap: wrap !important; }
.chip button, button.chip {
    background: rgba(255,255,255,0.8) !important; border: 1px solid #ffffff !important; border-radius: 999px !important;
    color: #22224a !important; font-size: 12.5px !important; font-weight: 600 !important;
    padding: 8px 14px !important; box-shadow: 0 4px 12px rgba(60,60,160,0.14) !important;
    transition: transform .25s cubic-bezier(.34,1.56,.64,1) !important;
}
.chip button:hover, button.chip:hover { transform: translateY(-4px) rotate(-1.5deg) scale(1.05); background: #ffffff !important; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
"""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")

with gr.Blocks(title="CyberDecoder | #codingwithqueen") as demo:
    with gr.Row(elem_id="shell", equal_height=True):
        with gr.Column(scale=1, min_width=250, elem_id="sidebar"):
            gr.HTML(BRAND)
            new_btn = gr.Button("New Chat", elem_classes="side-btn")
            gr.HTML('<div class="side-label">Quick topics</div>')
            topic_btns = [gr.Button(t, elem_classes="topic-btn") for t in TOPICS]
            gr.HTML('<div class="tag-foot">#codingwithqueen</div>')

        with gr.Column(scale=4, elem_id="main"):
            hero = gr.HTML(HERO)
            chatbot = gr.Chatbot(elem_id="chat", visible=False, height=480,
                                 show_label=False, buttons=[])
            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(placeholder="Paste a CVE, alert, or security term here...",
                                 show_label=False, container=False, scale=6, lines=1,
                                 max_lines=4)
                send = gr.Button("Decode", elem_id="send-btn", scale=1)
            with gr.Row(elem_id="chips"):
                chip_btns = [gr.Button(t, elem_classes="chip", size="sm") for t in TOPICS[:4]]

    outputs = [hero, chatbot, msg]
    msg.submit(respond, [msg, chatbot], outputs)
    send.click(respond, [msg, chatbot], outputs)
    new_btn.click(new_chat, None, outputs)
    for btn, topic in zip(topic_btns, TOPICS):
        btn.click(topic_handler(topic), [chatbot], outputs)
    for btn, topic in zip(chip_btns, TOPICS[:4]):
        btn.click(topic_handler(topic), [chatbot], outputs)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
