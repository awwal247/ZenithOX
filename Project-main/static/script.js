/* ==========================================================
   ZENITH OX â€” Chat UI (syntax highlighting + code downloads)
   ========================================================== */
(() => {
  const chatBox   = document.getElementById("chat-box");
  const form      = document.getElementById("chatForm");
  const input     = document.getElementById("user-input");
  const sendBtn   = document.getElementById("sendBtn");
  const clearBtn  = document.getElementById("clearBtn");
  const logoutBtn = document.getElementById("logoutBtn");

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

  function typeInto(el, text, speed) {
    el.textContent = "";
    return new Promise(resolve => {
      let i = 0;
      const iv = setInterval(() => {
        el.textContent += text.charAt(i++);
        chatBox.scrollTop = chatBox.scrollHeight;
        if (i >= text.length) { clearInterval(iv); resolve(); }
      }, speed || 14);
    });
  }

  /* ---------- code block detection ---------- */
  const CODE_RE = /```(\w*)\n([\s\S]*?)```/g;

  function hasCodeBlocks(text) {
    CODE_RE.lastIndex = 0;
    return CODE_RE.test(text);
  }

  /**
   * Render a response with syntax-highlighted code blocks.
   * Text outside code fences is rendered as plain text spans.
   * Code blocks get <pre><code> with highlight.js coloring.
   */
  function renderFormatted(container, text) {
    container.innerHTML = "";
    CODE_RE.lastIndex = 0;
    let lastIdx = 0;
    let match;

    while ((match = CODE_RE.exec(text)) !== null) {
      // Text before this code block
      if (match.index > lastIdx) {
        const textPart = text.slice(lastIdx, match.index);
        const span = document.createElement("span");
        span.textContent = textPart;
        container.appendChild(span);
      }

      // Code block
      const lang = match[1] || "plaintext";
      const code = match[2];

      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";

      // Header bar with language label + copy button
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

      // Apply highlight.js
      if (window.hljs) { hljs.highlightElement(codeEl); }

      // Copy button handler
      header.querySelector(".copy-btn").addEventListener("click", function() {
        navigator.clipboard.writeText(code).then(() => {
          this.textContent = "Copied!";
          setTimeout(() => { this.textContent = "Copy"; }, 2000);
        });
      });

      lastIdx = match.index + match[0].length;
    }

    // Remaining text after last code block
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

  /* ---------- send ---------- */
  async function sendMessage(message) {
    addMessage(message, "user");
    const typingEl = addTyping();
    sendBtn.disabled = true;

    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });
      const data = await r.json();
      typingEl.remove();

      if (!data.ok) {
        addMessage("\u26A0 " + (data.error || "Unknown error"), "bot error");
        return;
      }

      const botEl = document.createElement("div");
      botEl.className = "message bot";
      chatBox.appendChild(botEl);

      // If response contains code blocks, render with syntax highlighting
      if (hasCodeBlocks(data.response)) {
        renderFormatted(botEl, data.response);
      } else {
        // Plain text â€” use typing animation
        await typeInto(botEl, data.response);
      }

      // Download button (zip for developer, pptx for slides)
      if (data.download_url) {
        addDownloadButton(botEl, data.download_url, data.download_name);
      }

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
    if (!msg) return;
    input.value = "";
    sendMessage(msg);
  });

  clearBtn.addEventListener("click", async () => {
    if (!confirm("Clear all chat memory for this mode?")) return;
    try {
      const r = await fetch("/clear", { method: "POST" });
      const data = await r.json();
      if (data.ok) {
        chatBox.innerHTML = "";
        addMessage("\uD83E\uDDF9 Memory cleared. Starting fresh.", "bot welcome");
      }
    } catch (err) {
      addMessage("\u26A0 Could not clear memory: " + err.message, "bot error");
    }
  });

  logoutBtn.addEventListener("click", () => { window.location.href = "/logout"; });
  input.focus();
})();