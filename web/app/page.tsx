"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Account } from "@/components/account";
import { useSession } from "@/lib/auth-client";

const EMO: Record<string, { e: string; c: string }> = {
  anger: { e: "😠", c: "#d9614c" },
  disgust: { e: "🤢", c: "#7ea653" },
  fear: { e: "😨", c: "#8b7bd8" },
  happy: { e: "😊", c: "#f0b53f" },
  sad: { e: "😢", c: "#5b8bd0" },
  surprise: { e: "😮", c: "#40b6b0" },
  neutral: { e: "😐", c: "#b5a893" },
};
const pct = (x: number) => Math.round(x * 100) + "%";

type View = { label: string; confidence: number; available: boolean };
type ChatResponse = {
  reply: string;
  conflicted: boolean;
  text_emotion: View;
  face_emotion: View;
  fused_emotion: View;
};
type Msg = { role: "user" | "bot"; content: string; data?: ChatResponse };

function Dots() {
  return (
    <span className="inline-flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-pulse"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </span>
  );
}

function Chips({ d }: { d: ChatResponse }) {
  const chip = (label: string, v: View) =>
    `${EMO[v.label]?.e ?? ""} ${label} ${v.label} ${pct(v.confidence)}`;
  return (
    <div className="flex flex-wrap gap-1.5 pt-0.5">
      <Badge variant="outline" className="font-normal text-muted-foreground">
        {chip("words:", d.text_emotion)}
      </Badge>
      <Badge variant="outline" className="font-normal text-muted-foreground">
        {d.face_emotion.available ? chip("face:", d.face_emotion) : "👤 no face"}
      </Badge>
      <Badge
        variant="outline"
        className="font-normal"
        style={{ borderColor: "color-mix(in srgb, var(--emo) 50%, transparent)" }}
      >
        {chip("fused:", d.fused_emotion)}
      </Badge>
      {d.conflicted && (
        <Badge variant="outline" className="font-normal text-amber-400 border-amber-400/40">
          ⚠ mixed signals
        </Badge>
      )}
    </div>
  );
}

export default function Page() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [mood, setMood] = useState<{ label: string; conf: number } | null>(null);
  const [camOk, setCamOk] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const history = useRef<{ role: string; content: string }[]>([]);

  const { data: session } = useSession();
  const authed = !!session?.user;
  const [conversations, setConversations] = useState<{ id: string; title: string }[]>([]);
  const [convId, setConvId] = useState<string | null>(null);

  const refreshHistory = async () => {
    const r = await fetch("/api/history");
    if (r.ok) setConversations((await r.json()).conversations ?? []);
  };

  // load / clear the chat list as auth state changes
  useEffect(() => {
    if (authed) refreshHistory();
    else {
      setConversations([]);
      setConvId(null);
    }
  }, [authed]);

  function newChat() {
    setMessages([]);
    setConvId(null);
    history.current = [];
  }

  async function loadConversation(id: string) {
    const r = await fetch(`/api/history?id=${id}`);
    if (!r.ok) return;
    const data = (await r.json()) as { messages: { role: "user" | "bot"; content: string }[] };
    setMessages(data.messages.map((m) => ({ role: m.role, content: m.content })));
    history.current = data.messages.map((m) => ({
      role: m.role === "bot" ? "assistant" : "user",
      content: m.content,
    }));
    setConvId(id);
  }

  useEffect(() => {
    let stream: MediaStream | undefined;
    navigator.mediaDevices
      ?.getUserMedia({ video: true })
      .then((s) => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
        setCamOk(true);
      })
      .catch(() => setCamOk(false));
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function frame(): string | null {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || !v.videoWidth) return null;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    return c.toDataURL("image/jpeg", 0.8);
  }

  function applyMood(label: string, conf: number) {
    document.body.style.setProperty("--emo", (EMO[label] ?? EMO.neutral).c);
    setMood({ label, conf });
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setBusy(true);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }, { role: "bot", content: "…" }]);
    const f = frame();
    try {
      const res = await fetch("/backend/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, frames: f ? [f] : [], history: history.current }),
      });
      const data: ChatResponse = await res.json();
      setMessages((m) => {
        const c = [...m];
        c[c.length - 1] = { role: "bot", content: data.reply };
        c[c.length - 2] = { ...c[c.length - 2], data };
        return c;
      });
      history.current.push({ role: "user", content: text }, { role: "assistant", content: data.reply });
      applyMood(data.fused_emotion.label, data.fused_emotion.confidence);
      if (authed) {
        const saved = await fetch("/api/history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversationId: convId,
            turn: { userText: text, botText: data.reply, emotion: data.fused_emotion.label },
          }),
        });
        if (saved.ok && !convId) {
          setConvId((await saved.json()).conversationId);
          refreshHistory();
        }
      }
    } catch {
      setMessages((m) => {
        const c = [...m];
        c[c.length - 1] = { role: "bot", content: "⚠ couldn't reach the server — is it running?" };
        return c;
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-dvh">
      {authed && (
        <aside className="hidden w-60 shrink-0 flex-col border-r bg-card/40 sm:flex">
          <div className="p-3">
            <Button onClick={newChat} variant="secondary" size="sm" className="w-full justify-start">
              + New chat
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto px-2 pb-3">
            {conversations.length === 0 && (
              <p className="px-3 py-2 text-xs text-muted-foreground">No saved chats yet.</p>
            )}
            {conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => loadConversation(c.id)}
                className={`w-full truncate rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-secondary ${
                  c.id === convId ? "bg-secondary text-foreground" : "text-muted-foreground"
                }`}
              >
                {c.title}
              </button>
            ))}
          </div>
        </aside>
      )}
      <div className="relative z-10 flex flex-1 flex-col overflow-hidden">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b px-6 py-4 backdrop-blur-sm">
        <div>
          <div className="font-heading text-2xl font-medium leading-none">
            Empath<span style={{ color: "var(--emo)" }}>.</span>
          </div>
          <div className="mt-1 text-[11px] uppercase tracking-wider text-muted-foreground">
            emotion-aware companion
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Account />
          <div className="relative h-11 w-11 shrink-0">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="h-11 w-11 rounded-full object-cover transition-opacity duration-500"
              style={{
                border: "2px solid color-mix(in srgb, var(--emo) 60%, transparent)",
                opacity: camOk ? 1 : 0,
              }}
            />
            {!camOk && (
              <div className="absolute inset-0 grid place-items-center rounded-full bg-secondary text-lg">
                📷
              </div>
            )}
          </div>
          <div
            className="h-8 w-8 rounded-full transition-all duration-700"
            style={{
              background:
                "radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--emo) 90%, white), var(--emo))",
              boxShadow: "0 0 18px color-mix(in srgb, var(--emo) 55%, transparent)",
            }}
          />
          <div className="leading-tight">
            <div className="font-heading text-sm capitalize">
              {mood ? `${EMO[mood.label]?.e ?? ""} ${mood.label}` : "…"}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {mood ? `${pct(mood.conf)} sure` : "reading your mood"}
            </div>
          </div>
        </div>
      </header>

      <main ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto py-8">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6">
          {messages.length === 0 && (
            <div className="mx-auto my-[12vh] max-w-md text-center animate-in fade-in slide-in-from-bottom-3 duration-700">
              <h1 className="font-heading text-3xl font-normal leading-tight">How are you feeling?</h1>
              <p className="mt-3 text-muted-foreground">
                Say anything. I read the emotion in your words and your face, and reply with that in
                mind — the room warms to your mood.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300 ${
                msg.role === "user" ? "flex-row-reverse" : ""
              }`}
            >
              <Avatar className="mt-0.5 h-8 w-8 rounded-xl">
                <AvatarFallback
                  className="rounded-xl text-sm"
                  style={
                    msg.role === "bot"
                      ? {
                          background:
                            "radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--emo) 80%, white), var(--emo))",
                        }
                      : {}
                  }
                >
                  {msg.role === "bot" ? "" : "🙂"}
                </AvatarFallback>
              </Avatar>

              <div
                className={`flex flex-col gap-1.5 ${
                  msg.role === "user" ? "max-w-[85%] items-end" : "min-w-0 flex-1"
                }`}
              >
                <div
                  className={
                    msg.role === "user"
                      ? "rounded-2xl rounded-br-sm border border-primary/25 bg-primary/20 px-4 py-2.5 text-[15px] leading-relaxed"
                      : "prose-chat rounded-2xl rounded-bl-sm border bg-card px-4 py-2.5 text-[15px] leading-relaxed"
                  }
                >
                  {msg.role === "bot" && msg.content === "…" ? (
                    <Dots />
                  ) : msg.role === "bot" ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>
                {msg.data && <Chips d={msg.data} />}
              </div>
            </div>
          ))}
        </div>
      </main>

      <div className="shrink-0 px-6 pb-6 pt-2">
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-3xl border bg-card p-2 pl-4 shadow-2xl">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Tell me what's on your mind…"
            rows={1}
            className="max-h-40 min-h-0 resize-none border-0 bg-transparent p-2 text-[15px] shadow-none focus-visible:ring-0"
          />
          <Button
            onClick={send}
            disabled={busy || !input.trim()}
            size="icon"
            className="h-10 w-10 shrink-0 rounded-full text-lg"
          >
            ↑
          </Button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground">
          Enter to send · Shift+Enter for a new line · camera optional
        </p>
      </div>

      <canvas ref={canvasRef} hidden />
      </div>
    </div>
  );
}
