/* ==========================================================
   ZENITH OX — Chat UI Script (Stage 2: mode-aware)
   ========================================================== */
(() => {
  const chatBox   = document.getElementById("chat-box");
  const form      = document.getElementById("chatForm");
  const input     = document.getElementById("user-input");
  const sendBtn   = document.getElementById("sendBtn");
  const clearBtn  = document.getElementById("clearBtn");
  const logoutBtn = document.getElementById("logoutBtn");

  /* ---------- helpers ---------- */
  function addMessage(text, cls = "bot") {
    const div = document.createElement("div");
    div.className = "message " + cls;
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
  }

  function addTyping() {
    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = `<span class="typing">
        <span></span><span></span><span></span>
      </span>`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
  }

  function typeInto(el, text, speed = 14) {
    el.textContent = "";
    return new Promise(resolve => {
      let i = 0;
      const iv = setInterval(() => {
        el.textContent += text.charAt(i++);
        chatBox.scrollTop = chatBox.scrollHeight;
        if (i >= text.length) { clearInterval(iv); resolve(); }
      }, speed);
    });
  }

  function addDownloadButton(parentEl, url, filename) {
    const btn = document.createElement("a");
    btn.href = url;
    btn.download = filename || "presentation.pptx";
    btn.className = "download-btn";
    btn.textContent = "\uD83D\uDCE5 Download " + (filename || "presentation.pptx");
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

      const botEl = addMessage("", "bot");
      await typeInto(botEl, data.response);

      // If there's a download link (PPTX mode), add a download button
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

  logoutBtn.addEventListener("click", () => {
    window.location.href = "/logout";
  });

  input.focus();
})();
