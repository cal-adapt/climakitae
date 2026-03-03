import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const EXAMPLES = [
  "Get monthly max temperature data for Los Angeles from 2030-2060",
  "How do I compute heating and cooling degree days?",
  "What WRF variables are available at 3km resolution?",
  "Generate code for precipitation data under SSP 3-7.0",
];

function CodeBlock({ children, language }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div style={{ position: "relative" }}>
      <button className="copy-btn" onClick={copy}>
        {copied ? "Copied!" : "Copy"}
      </button>
      <SyntaxHighlighter
        language={language || "python"}
        style={oneDark}
        customStyle={{
          margin: 0,
          borderRadius: "8px",
          fontSize: "0.82rem",
          background: "#161822",
        }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "user") {
    return <div className="message user">{msg.content}</div>;
  }

  return (
    <div className="message assistant">
      {msg.toolCalls?.length > 0 && (
        <div className="tool-calls">
          {msg.toolCalls.map((tc, i) => (
            <span key={i} className="tool-pill">
              {tc.name}
            </span>
          ))}
        </div>
      )}
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const text = String(children).replace(/\n$/, "");
            if (!inline && (match || text.includes("\n"))) {
              return <CodeBlock language={match?.[1]}>{text}</CodeBlock>;
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {msg.content}
      </ReactMarkdown>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [online, setOnline] = useState(null);
  const messagesEnd = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Health check
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setOnline(d.ollama))
      .catch(() => setOnline(false));
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  const send = useCallback(
    async (text) => {
      const content = text ?? input;
      if (!content.trim() || loading) return;

      const userMsg = { role: "user", content: content.trim() };
      const newMessages = [...messages, userMsg];
      setMessages(newMessages);
      setInput("");
      setLoading(true);
      setError(null);

      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: newMessages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }),
        });

        if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

        const data = await resp.json();
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.reply,
            toolCalls: data.tool_calls,
          },
        ]);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    },
    [input, messages, loading]
  );

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      <div className="header">
        <span className={`status-dot ${online === false ? "offline" : ""}`} />
        <div>
          <h1>Cal-Adapt Climate Data Assistant</h1>
          <span className="subtitle">
            climakitae code generation &middot; powered by Ollama
          </span>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {messages.length === 0 ? (
        <div className="welcome">
          <h2>What climate data do you need?</h2>
          <p>
            Ask about available variables, regions, or analysis methods. I'll
            generate ready-to-run Python code using climakitae.
          </p>
          <div className="examples">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                className="example-btn"
                onClick={() => send(ex)}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="messages">
          {messages.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}
          {loading && (
            <div className="typing">
              <span />
              <span />
              <span />
            </div>
          )}
          <div ref={messagesEnd} />
        </div>
      )}

      <div className="input-area">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about climate data, variables, or analysis..."
            rows={1}
          />
          <button
            className="send-btn"
            onClick={() => send()}
            disabled={!input.trim() || loading}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
