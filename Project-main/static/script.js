/* ==========================================================
   ZENITH OX â€” Chat UI (ChatGPT-style)
   ========================================================== */
(() => {
  const chatBox   = document.getElementById("chat-box");
  const form      = document.getElementById("chatForm");
  const input     = document.getElementById("user-input");
  const sendBtn   = document.getElementById("sendBtn");
  const clearBtn  = document.getElementById("clearBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const fileInput = document.getElementById("fileInput");
  const filePreview = document.getElementById("file-preview");

  /* ---------- auto-resize textarea ---------- */
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 150) + "px";
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });

  /* ---------- file upload preview ---------- */
  let pendingFile = null;
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    pendingFile = file;
    filePreview.classList.remove("hidden");
    filePreview.innerHTML = '<span>\u{1F4CE} ' + file.name + '</span>' +
      '<button type="button" id="removeFile">\u2715</button>';
    document.getElementById("removeFile").addEventListener("click", () => {
      pendingFile = null;
      fileInput.value = "";
      filePreview.classList.add("hidden");
    });
  });

  /* ---------- helpers ---------- */
  function addMessage(text, cls) {
    const div = document.createElement("div");
    div.className = "message " + (cls || "bot");
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
  }

  function addTyping() {
    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
  }

  /* ---------- code block detection ---------- */
  const CODE_RE = /```(\w*)\n([\s\S]*?)```/g;

  function hasCodeBlocks(text) {
    CODE_RE.lastIndex = 0;
    return CODE_RE.test(text);
  }

  function renderFormatted(container, text) {
    container.innerHTML = "";
    CODE_RE.lastIndex = 0;
    let lastIdx = 0;
    let match;

    while ((match = CODE_RE.exec(text)) !== null) {
      if (match.index > lastIdx) {
        const span = document.createElement("span");
        span.textContent = text.slice(lastIdx, match.index);
        container.appendChild(span);
      }
      const lang = match[1] || "plaintext";
      const code = match[2];
      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";
      const header = document.createElement("div");
      header.className = "code-header";
      header.innerHTML = '<span class="code-lang">' + lang + '</span>' +
        '<button class="copy-btn" title="Copy code">Copy</button>';
      wrapper.appendChild(header);
      const pre = document.createElement("pre");
      const codeEl = document.createElement("code");
      codeEl.className = "language-" + lang;
      codeEl.textContent = code;
      pre.appendChild(codeEl);
      wrapper.appendChild(pre);
      container.appendChild(wrapper);
      if (window.hljs) { hljs.highlightElement(codeEl); }
      header.querySelector(".copy-btn").addEventListener("click", function() {
        navigator.clipboard.writeText(code).then(() => {
          this.textContent = "Copied!";
          setTimeout(() => { this.textContent = "Copy"; }, 2000);
        });
      });
      lastIdx = match.index + match[0].length;
    }
    if (lastIdx < text.length) {
      const span = document.createElement("span");
      span.textContent = text.slice(lastIdx);
      container.appendChild(span);
    }
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function addDownloadButton(parentEl, url, filename) {
    const btn = document.createElement("a");
    btn.href = url;
    btn.download = filename || "download";
    btn.className = "download-btn";
    btn.textContent = "\uD83D\uDCE5 Download " + (filename || "file");
    parentEl.appendChild(btn);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function renderBotMessage(data) {
    const botEl = document.createElement("div");
    botEl.className = "message bot";
    chatBox.appendChild(botEl);
    if (hasCodeBlocks(data.response)) {
      renderFormatted(botEl, data.response);
    } else {
      botEl.textContent = data.response;
    }
    if (data.download_url) {
      addDownloadButton(botEl, data.download_url, data.download_name);
    }
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  /* ---------- load chat history on page load ---------- */
  async function loadHistory() {
    try {
      const r = await fetch("/history");
      const data = await r.json();
      if (data.ok && data.messages && data.messages.length > 0) {
        // Remove the default welcome message
        const welcome = chatBox.querySelector(".welcome");
        if (welcome) welcome.remove();
        for (const msg of data.messages) {
          if (msg.role === "user") {
            addMessage(msg.content, "user");
          } else {
            const botEl = document.createElement("div");
            botEl.className = "message bot";
            chatBox.appendChild(botEl);
            if (hasCodeBlocks(msg.content)) {
              renderFormatted(botEl, msg.content);
            } else {
              botEl.textContent = msg.content;
            }
          }
        }
      }
    } catch (e) { /* silently fail - just show empty chat */ }
  }
  loadHistory();

  /* ---------- send message ---------- */
  async function sendMessage(message) {
    addMessage(message, "user");
    const typingEl = addTyping();
    sendBtn.disabled = true;

    try {
      let r, data;
      if (pendingFile) {
        // File upload mode
        const fd = new FormData();
        fd.append("file", pendingFile);
        fd.append("message", message);
        r = await fetch("/upload-code", { method: "POST", body: fd });
        pendingFile = null;
        fileInput.value = "";
        filePreview.classList.add("hidden");
      } else {
        r = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message })
        });
      }
      data = await r.json();
      typingEl.remove();

      if (!data.ok) {
        addMessage("\u26A0 " + (data.error || "Unknown error"), "bot error");
        return;
      }
      renderBotMessage(data);
    } catch (err) {
      typingEl.remove();
      addMessage("\u26A0 Connection error: " + err.message, "bot error");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  /* ---------- events ---------- */
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg && !pendingFile) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(msg || "Analyze this code");
  });

  clearBtn.addEventListener("click", async () => {
    if (!confirm("Clear all chat memory for this mode?")) return;
    try {
      const r = await fetch("/clear", { method: "POST" });
      const data = await r.json();
      if (data.ok) {
        chatBox.innerHTML = "";
        addMessage("Memory cleared. Starting fresh.", "bot welcome");
      }
    } catch (err) {
      addMessage("\u26A0 Could not clear: " + err.message, "bot error");
    }
  });

  logoutBtn.addEventListener("click", () => { window.location.href = "/logout"; });
  input.focus();
})();