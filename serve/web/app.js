// TinyLLaMA chat UI: plain JavaScript, no build step.
// Talks to the server's OpenAI-compatible endpoint (/v1/chat/completions) with streaming,
// so the page uses exactly the API that external clients use.
// Chats and settings live in this browser's localStorage only.
"use strict";

const $ = (id) => document.getElementById(id);
const DEFAULTS = { temperature: 0.7, topP: 0.9, maxTokens: 300, seed: "", apiKey: "" };
const EXAMPLES = [
  { title: "Rain, ball, happy", text: "Write a story. Use the words: rain, ball, happy." },
  { title: "A lost kitten", text: "Write a story about a girl who finds a lost kitten. Use the words: box, warm, smile." },
  { title: "With dialogue", text: "Write a short story. The story must use the words call, leak, foolish. Also, include some dialogue." },
  { title: "Happy ending", text: "Please write a simple story with a happy ending. Include these words: dog, moon, brave." },
];

// ---------- storage (wrapped: private windows can refuse localStorage) ----------
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* not saved */ } },
};

const state = {
  chats: store.get("tl.chats", []),
  currentId: null,
  settings: { ...DEFAULTS, ...store.get("tl.settings", {}) },
  info: null,
  controller: null,   // AbortController of the reply being streamed
};
const saveChats = () => store.set("tl.chats", state.chats.slice(0, 200));
const saveSettings = () => store.set("tl.settings", state.settings);
const currentChat = () => state.chats.find((c) => c.id === state.currentId) || null;

// ---------- small helpers ----------
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;   // always textContent: model output is never parsed as HTML
  return e;
}
function svg(path) {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", path);
  s.appendChild(p);
  return s;
}
let toastTimer;
function toast(message, ms = 3200) {
  const t = $("toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { toast("Couldn't copy: select the text and copy it by hand"); }
}

// ---------- required words: the same rule as experiments/eval_instruct.py ----------
function requiredWords(instruction) {
  const m = instruction.match(/words:?\s+([\w\s,'-]+?)(?:\.|$)/i);
  if (!m) return [];
  return m[1].split(/,|\band\b/).map((w) => w.trim().toLowerCase()).filter((w) => w && !w.includes(" "));
}
function wordPresent(word, text) {
  const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escaped}(s|es|d|ed|ing)?\\b`, "i").test(text);
}

// ---------- theme ----------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("themeLabel").textContent = theme === "dark" ? "Light mode" : "Dark mode";
}
applyTheme(store.get("tl.theme", matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
$("toggleTheme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  store.set("tl.theme", next);
  applyTheme(next);
};

// ---------- sidebar: chat list with search ----------
function dayGroup(ts) {
  const days = (Date.now() - ts) / 86400000;
  if (days < 1 && new Date(ts).getDate() === new Date().getDate()) return "Today";
  if (days < 7) return "Previous 7 days";
  return "Older";
}
function renderList() {
  const list = $("chatList");
  list.replaceChildren();
  const q = $("search").value.trim().toLowerCase();
  const chats = state.chats
    .filter((c) => !q || c.title.toLowerCase().includes(q) || c.messages.some((m) => m.content.toLowerCase().includes(q)))
    .sort((a, b) => b.updated - a.updated);
  if (!chats.length) {
    list.appendChild(el("div", "list-empty", q ? "No chats match your search." : "Your chats will appear here."));
    return;
  }
  let group = null;
  for (const chat of chats) {
    const g = dayGroup(chat.updated);
    if (g !== group) { list.appendChild(el("div", "list-label", g)); group = g; }
    const item = el("div", "chat-item" + (chat.id === state.currentId ? " active" : ""));
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.appendChild(el("span", "chat-title", chat.title));
    const del = el("button", "chat-del");
    del.title = "Delete chat";
    del.setAttribute("aria-label", "Delete chat");
    del.appendChild(svg("M6 6l12 12M18 6 6 18"));
    del.onclick = (e) => { e.stopPropagation(); deleteChat(chat.id); };
    item.appendChild(del);
    item.onclick = () => openChat(chat.id);
    item.onkeydown = (e) => { if (e.key === "Enter") openChat(chat.id); };
    list.appendChild(item);
  }
}
$("search").addEventListener("input", renderList);

function openChat(id) {
  if (state.controller) state.controller.abort();
  state.currentId = id;
  renderMessages();
  renderList();
  closeSidebar();
}
function newChat() {
  if (state.controller) state.controller.abort();
  state.currentId = null;
  renderMessages();
  renderList();
  closeSidebar();
  $("prompt").focus();
}
function deleteChat(id) {
  state.chats = state.chats.filter((c) => c.id !== id);
  saveChats();
  if (state.currentId === id) newChat(); else renderList();
}
$("newChat").onclick = newChat;
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
});

// mobile: the sidebar slides in over the page
const openSidebar = () => { $("app").classList.add("sidebar-open"); $("scrim").hidden = false; };
function closeSidebar() { $("app").classList.remove("sidebar-open"); $("scrim").hidden = true; }
$("openSidebar").onclick = openSidebar;
$("scrim").onclick = closeSidebar;

// ---------- conversation ----------
function renderExamples() {
  const box = $("examples");
  box.replaceChildren();
  for (const ex of EXAMPLES) {
    const card = el("button", "example");
    card.type = "button";
    card.appendChild(el("strong", "", ex.title));
    card.appendChild(el("span", "", ex.text));
    card.onclick = () => send(ex.text);
    box.appendChild(card);
  }
}

function renderMessages() {
  const chat = currentChat();
  const box = $("messages");
  box.replaceChildren();
  $("empty").hidden = !!(chat && chat.messages.length);
  if (!chat) return;
  chat.messages.forEach((m, i) => box.appendChild(messageEl(chat, m, i)));
  scrollToBottom(true);
}

function messageEl(chat, msg, index) {
  if (msg.role === "user") {
    const row = el("div", "msg user");
    row.appendChild(el("div", "bubble", msg.content));
    return row;
  }
  const row = el("div", "msg assistant");
  row.appendChild(el("div", "avatar", "TL"));
  const body = el("div", "body");
  const text = el("div", "text" + (msg.error ? " error" : ""), msg.error || msg.content);
  body.appendChild(text);
  body.appendChild(metaEl(chat, msg, index));
  row.appendChild(body);
  return row;
}

function metaEl(chat, msg, index) {
  const meta = el("div", "meta");
  const instruction = chat.messages[index - 1]?.content || "";
  const words = requiredWords(instruction);
  if (msg.content && words.length) {
    const used = words.filter((w) => wordPresent(w, msg.content)).length;
    meta.appendChild(el("span", "chip words", `${used}/${words.length} words`));
    for (const w of words) {
      const ok = wordPresent(w, msg.content);
      meta.appendChild(el("span", "chip " + (ok ? "ok" : "bad"), (ok ? "✓ " : "✗ ") + w));
    }
  }
  if (msg.stats) {
    const s = msg.stats;
    const parts = [`${s.tokens} tokens`];
    if (s.tps) parts.push(`${Math.round(s.tps)} tok/s`);
    if (s.seed !== undefined) parts.push(`seed ${s.seed}`);
    if (s.finish === "length") parts.push("hit the length limit");
    if (s.finish === "stopped") parts.push("stopped");
    meta.appendChild(el("span", "stat", parts.join(" · ")));
  }
  if (msg.content || msg.error) {
    const actions = el("span", "actions");
    const copy = el("button");
    copy.title = "Copy";
    copy.appendChild(svg("M9 9h10v10H9zM5 15V5h10"));
    copy.onclick = () => copyText(msg.content);
    const again = el("button");
    again.title = "Regenerate";
    again.appendChild(svg("M4 4v6h6M20 20v-6h-6M5.6 15a7 7 0 0 0 12.5 2M18.4 9A7 7 0 0 0 5.9 7"));
    again.onclick = () => regenerate(chat.id, index);
    actions.append(copy, again);
    meta.appendChild(actions);
  }
  return meta;
}

function nearBottom() {
  const c = $("conversation");
  return c.scrollHeight - c.scrollTop - c.clientHeight < 120;
}
function scrollToBottom(force) {
  const c = $("conversation");
  if (force || nearBottom()) c.scrollTop = c.scrollHeight;
}

// ---------- sending and streaming ----------
function setBusy(busy) {
  $("composer").classList.toggle("busy", busy);
  $("send").setAttribute("aria-label", busy ? "Stop" : "Send");
  updateSendButton();
}
function updateSendButton() {
  $("send").disabled = !state.controller && !$("prompt").value.trim();
}

function send(text) {
  text = text.trim();
  if (!text || state.controller) return;
  let chat = currentChat();
  if (!chat) {
    chat = { id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()), title: text.slice(0, 60),
             updated: Date.now(), messages: [] };
    state.chats.push(chat);
    state.currentId = chat.id;
  }
  chat.messages.push({ role: "user", content: text });
  chat.messages.push({ role: "assistant", content: "" });
  chat.updated = Date.now();
  saveChats();
  renderList();
  renderMessages();
  $("prompt").value = "";
  autoGrow();
  stream(chat, chat.messages.length - 1);
}

function regenerate(chatId, index) {
  const chat = state.chats.find((c) => c.id === chatId);
  if (!chat || state.controller) return;
  chat.messages[index] = { role: "assistant", content: "" };
  saveChats();
  renderMessages();
  stream(chat, index);
}

async function stream(chat, index) {
  const msg = chat.messages[index];
  const instruction = chat.messages[index - 1].content;
  const row = $("messages").children[index];
  const textEl = row.querySelector(".text");
  row.classList.add("streaming");

  const s = state.settings;
  const body = {
    model: state.info?.model || "tinyllama",
    messages: [{ role: "user", content: instruction }],   // the model answers single instructions
    stream: true,
    temperature: Number(s.temperature),
    top_p: Number(s.topP),
    max_tokens: Number(s.maxTokens),
  };
  if (s.seed !== "" && s.seed !== null) body.seed = Number(s.seed);
  const headers = { "Content-Type": "application/json" };
  if (s.apiKey) headers.Authorization = `Bearer ${s.apiKey}`;

  const controller = new AbortController();
  state.controller = controller;
  setBusy(true);
  let finished = false;
  try {
    const res = await fetch("/v1/chat/completions", { method: "POST", headers, body: JSON.stringify(body),
                                                      signal: controller.signal });
    if (!res.ok) {
      let message = `HTTP ${res.status}`;
      try { message = (await res.json()).error?.message || message; } catch { /* not JSON */ }
      if (res.status === 401) { openSettings(); $("apiKey").focus(); }
      throw new Error(message);
    }
    // Server-Sent Events: "data: {...}\n\n" blocks; a network read can end mid-block
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") { finished = true; break; }
          const event = JSON.parse(data);
          if (event.error) throw new Error(event.error.message);
          const choice = event.choices?.[0];
          if (choice?.delta?.content) {
            msg.content += choice.delta.content;
            textEl.textContent = msg.content;
            scrollToBottom(false);
          }
          if (choice?.finish_reason) {
            const x = event.x_tinyllama || {};
            msg.stats = { tokens: event.usage?.completion_tokens ?? 0, tps: x.decode_tokens_per_s,
                          seed: x.seed, finish: choice.finish_reason };
          }
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      msg.stats = { ...(msg.stats || {}), tokens: msg.stats?.tokens ?? approxTokens(msg.content), finish: "stopped" };
    } else if (msg.content) {
      toast(err.message);                 // keep the partial reply, report the error briefly
    } else {
      msg.error = `⚠ ${err.message}`;     // nothing arrived: the error takes the reply's place
    }
  } finally {
    if (state.controller === controller) state.controller = null;
    setBusy(false);
    chat.updated = Date.now();
    saveChats();
    // re-render this message with its final meta row (words, speed, seed, buttons)
    if (state.currentId === chat.id) {
      const fresh = messageEl(chat, msg, index);
      $("messages").children[index]?.replaceWith(fresh);
      scrollToBottom(false);
    }
  }
}
function approxTokens(text) { return Math.round(text.split(/\s+/).filter(Boolean).length * 1.3); }

// ---------- composer ----------
function autoGrow() {
  const t = $("prompt");
  t.style.height = "";   // back to one row (rows="1"), then grow to fit the text, up to 200 px
  if (t.value) t.style.height = Math.min(t.scrollHeight, 200) + "px";
  updateSendButton();
}
$("prompt").addEventListener("input", autoGrow);
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.controller) state.controller.abort();   // the button is "Stop" while a reply streams
  else send($("prompt").value);
});

// ---------- settings ----------
const fields = [["temperature", "temperature", "tempOut"], ["topP", "topP", "topPOut"], ["maxTokens", "maxTokens", "maxOut"]];
function syncSettings() {
  for (const [key, id, out] of fields) { $(id).value = state.settings[key]; $(out).textContent = state.settings[key]; }
  $("seed").value = state.settings.seed;
  $("apiKey").value = state.settings.apiKey;
}
for (const [key, id, out] of fields) {
  $(id).addEventListener("input", () => { state.settings[key] = Number($(id).value); $(out).textContent = $(id).value; saveSettings(); });
}
$("seed").addEventListener("input", () => { state.settings.seed = $("seed").value; saveSettings(); });
$("apiKey").addEventListener("input", () => { state.settings.apiKey = $("apiKey").value.trim(); saveSettings(); });
$("resetSettings").onclick = () => { state.settings = { ...DEFAULTS, apiKey: state.settings.apiKey }; saveSettings(); syncSettings(); };
function openSettings() { syncSettings(); $("settingsModal").showModal(); }
$("openSettings").onclick = () => { closeSidebar(); openSettings(); };

// ---------- API dialog ----------
function openApi() {
  closeSidebar();
  const base = `${location.origin}/v1`;
  const model = state.info?.model || "tinyllama";
  const keyRequired = state.info?.api_key_required;
  const key = keyRequired ? (state.settings.apiKey || "YOUR_API_KEY") : "any-text";
  $("baseUrl").textContent = base;
  $("apiModel").textContent = model;
  $("apiKeyInfo").textContent = keyRequired
    ? "required: ask the server's owner for one, then send it as a Bearer token"
    : "not required on this server (any text works)";
  $("curlSnippet").textContent =
`curl -N ${base}/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${key}" \\
  -d '{"model": "${model}", "stream": true,
       "messages": [{"role": "user", "content": "Write a story. Use the words: rain, ball, happy."}]}'`;
  $("pySnippet").textContent =
`from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="${key}")
stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Write a story. Use the words: rain, ball, happy."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)`;
  $("apiModal").showModal();
}
$("openApi").onclick = openApi;
$("openApiTop").onclick = openApi;
document.querySelectorAll(".copy").forEach((b) => { b.onclick = () => copyText($(b.dataset.copy).textContent); });

// ---------- server info and health ----------
async function loadInfo() {
  try {
    const info = await (await fetch("/info")).json();
    state.info = info;
    $("modelName").textContent = info.model;
    const quant = info.quant ? ` · ${info.quant}` : "";
    $("modelMeta").textContent = `${(info.params / 1e6).toFixed(0)}M params${quant} · ${info.device.toUpperCase()}`;
    $("statusDot").className = "status-dot ok";
    $("keyHint").textContent = info.api_key_required
      ? "This server requires a key. Stored only in this browser."
      : "This server doesn't require a key. Stored only in this browser.";
  } catch {
    $("modelName").textContent = "server unreachable";
    $("statusDot").className = "status-dot down";
  }
}
async function checkHealth() {
  try {
    const ok = (await fetch("/health")).ok;
    $("statusDot").className = "status-dot " + (ok ? "ok" : "down");
  } catch { $("statusDot").className = "status-dot down"; }
}

// ---------- start ----------
renderExamples();
renderList();
renderMessages();
autoGrow();
loadInfo();
setInterval(checkHealth, 30000);
