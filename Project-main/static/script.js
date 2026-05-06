/* ==========================================================
   ZENITH OX Ã¢â‚¬â€ Chat UI (Markdown + Math + Copy button)
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

  /* ---------- Configure marked.js ---------- */
  const renderer = new marked.Renderer();

  // Custom code block renderer: extract filename + highlight
  const FILE_RE = /^(?:#|\/\/|<!--|\/\*)\s*File:\s*(.+?)(?:\s*-->|\s*\*\/)?\s*$/i;

  renderer.code = function({ text, lang }) {
    const code = text || "";
    const language = lang || "plaintext";
    let filename = null;
    let cleanCode = code;

    // Extract filename from first line
    const lines = code.split("\n");
    if (lines.length > 0) {
      const m = lines[0].trim().match(FILE_RE);
      if (m) {
        filename = m[1].trim();
        cleanCode = lines.slice(1).join("\n").trim();
      }
    }

    // Highlight code
    let highlighted;
    try {
      if (hljs.getLanguage(language)) {
        highlighted = hljs.highlight(cleanCode, { language }).value;
      } else {
        highlighted = hljs.highlightAuto(cleanCode).value;
      }
    } catch (e) {
      highlighted = cleanCode.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    const label = filename ? '<strong>' + filename + '</strong>' : language;
    const escapedCode = cleanCode.replace(/`/g, "\\`").replace(/\$/g, "\\$");

    return '<div class="code-block-wrapper">' +
      '<div class="code-header">' +
        '<span class="code-lang">' + label + '</span>' +
        '<button class="copy-btn" onclick="copyCode(this)" data-code="' +
          btoa(unescape(encodeURIComponent(cleanCode))) + '">Copy</button>' +
      '</div>' +
      '<pre><code class="hljs language-' + language + '">' + highlighted + '</code></pre>' +
    '</div>';
  };

  marked.setOptions({
    renderer: renderer,
    breaks: true,
    gfm: true,
  });

  // Global copy function for code blocks
  window.copyCode = function(btn) {
    const code = decodeURIComponent(escape(atob(btn.dataset.code)));
    navigator.clipboard.writeText(code).then(() => {
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy"; }, 2000);
    });
  };

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

  /* ---------- file upload ---------- */
  let pendingFile = null;
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    pendingFile = file;
    filePreview.classList.remove("hidden");
    filePreview.innerHTML = '<span>\u{1F4CE} ' + file.name +
      ' (' + (file.size / 1024).toFixed(1) + ' KB)</span>' +
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

  function addFileIndicator(filename) {
    const div = document.createElement("div");
    div.className = "message user file-indicator";
    div.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-3px"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg> ' + filename;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function addTyping() {
    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
  }

  /* ---------- render bot message with markdown + math + copy ---------- */
  function renderBotMessage(responseText, downloadUrl, downloadName) {
    const wrapper = document.createElement("div");
    wrapper.className = "message bot";

    // Render markdown
    const content = document.createElement("div");
    content.className = "md-content";
    content.innerHTML = marked.parse(responseText);
    wrapper.appendChild(content);

    // Render math (KaTeX)
    if (window.renderMathInElement) {
      renderMathInElement(content, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    }

    // Download button
    if (downloadUrl) {
      const dl = document.createElement("a");
      dl.href = downloadUrl;
      dl.download = downloadName || "download";
      dl.className = "download-btn";
      dl.textContent = "\uD83D\uDCE5 Download " + (downloadName || "file");
      wrapper.appendChild(dl);
    }

    // Copy message button
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-copy-btn";
    copyBtn.title = "Copy response";
    copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(responseText).then(() => {
        copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>';
        setTimeout(() => {
          copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
        }, 2000);
      });
    });
    wrapper.appendChild(copyBtn);

    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  /* ---------- load chat history ---------- */
  async function loadHistory() {
    try {
      const r = await fetch("/history");
      const data = await r.json();
      if (data.ok && data.messages && data.messages.length > 0) {
        const welcome = chatBox.querySelector(".welcome");
        if (welcome) welcome.remove();
        for (const msg of data.messages) {
          if (msg.role === "user") {
            if (msg.content.startsWith("[Uploaded:")) {
              const fm = msg.content.match(/\[Uploaded: (.+?)\]\s*(.*)/);
              if (fm) {
                addFileIndicator(fm[1]);
                if (fm[2]) addMessage(fm[2], "user");
              } else {
                addMessage(msg.content, "user");
              }
            } else {
              addMessage(msg.content, "user");
            }
          } else {
            renderBotMessage(msg.content);
          }
        }
      }
    } catch (e) { /* silently fail */ }
  }
  loadHistory();

  /* ---------- send message ---------- */
  async function sendMessage(message) {
    if (pendingFile) addFileIndicator(pendingFile.name);
    if (message) addMessage(message, "user");
    const typingEl = addTyping();
    sendBtn.disabled = true;

    try {
      let r;
      if (pendingFile) {
        const fd = new FormData();
        fd.append("file", pendingFile);
        fd.append("message", message || "Please analyze this file");
        r = await fetch("/chat", { method: "POST", body: fd });
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
      const data = await r.json();
      typingEl.remove();

      if (!data.ok) {
        addMessage("\u26A0 " + (data.error || "Unknown error"), "bot error");
        return;
      }
      renderBotMessage(data.response, data.download_url, data.download_name);
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
    sendMessage(msg);
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